"""API-key authentication with owner identity and brute-force protection (NFR-07).

Only SHA-256 digests of the configured keys are held in memory.  A presented key is hashed and
compared against **every** configured digest with ``hmac.compare_digest`` (no early exit), so
neither the comparison nor the number of keys leaks through timing.  Failed attempts are
throttled per client by a stricter token bucket and delayed slightly; once a client's failure
bucket is empty its requests are rejected *before* the key is evaluated, so guessing cannot
continue at full speed.  Successful requests are limited per key id (not per IP).
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from shortener.domain.errors import AuthenticationError
from shortener.observability.metrics import Metrics
from shortener.services.rate_limiter import RateLimitDecision, TokenBucketLimiter

MAX_API_KEY_LENGTH = 256


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller; ``key_id`` is the owner id stored in ``links.created_by``."""

    key_id: str


ANONYMOUS = Principal("anonymous")


def digest_api_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()


class ApiKeyAuthenticator:
    def __init__(self, keys: Sequence[tuple[str, bytes]]) -> None:
        self._keys = tuple(keys)

    def authenticate(self, presented: str | None) -> Principal | None:
        if not presented or len(presented) > MAX_API_KEY_LENGTH or not presented.isascii():
            return None
        candidate = digest_api_key(presented)
        matched: str | None = None
        for key_id, digest in self._keys:
            if hmac.compare_digest(candidate, digest):
                matched = key_id
        return Principal(matched) if matched is not None else None


@dataclass(frozen=True, slots=True)
class Authenticated:
    principal: Principal
    rate_limit: RateLimitDecision


class AuthService:
    def __init__(
        self,
        *,
        authenticator: ApiKeyAuthenticator,
        disabled: bool,
        api_limiter: TokenBucketLimiter,
        failure_limiter: TokenBucketLimiter,
        metrics: Metrics,
        failure_delay_seconds: float,
        sleep: Callable[[float], None],
    ) -> None:
        self._authenticator = authenticator
        self._disabled = disabled
        self._api_limiter = api_limiter
        self._failure_limiter = failure_limiter
        self._metrics = metrics
        self._failure_delay = failure_delay_seconds
        self._sleep = sleep

    @property
    def enabled(self) -> bool:
        return not self._disabled

    def authenticate(self, presented_key: str | None, client_id: str) -> Authenticated:
        """Authenticate and rate-limit one management request.

        ``client_id`` is the throttling identity for failures (the resolved client address).
        Raises ``AuthenticationError`` (401) or ``RateLimitedError`` (429).
        """
        principal = ANONYMOUS if self._disabled else self._verify(presented_key, client_id)
        decision = self._api_limiter.acquire(principal.key_id)
        if not decision.allowed:
            self._metrics.inc("shortener_rate_limit_rejections_total", scope="api")
            raise decision.to_error()
        return Authenticated(principal=principal, rate_limit=decision)

    def _verify(self, presented_key: str | None, client_id: str) -> Principal:
        lockout = self._failure_limiter.peek(client_id)
        if not lockout.allowed:
            self._metrics.inc("shortener_rate_limit_rejections_total", scope="auth_failure")
            raise lockout.to_error()
        principal = self._authenticator.authenticate(presented_key)
        if principal is None:
            self._failure_limiter.acquire(client_id)
            self._metrics.inc("shortener_auth_failures_total")
            self._sleep(self._failure_delay)
            raise AuthenticationError
        return principal
