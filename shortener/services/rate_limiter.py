"""In-memory token-bucket rate limiter with bounded state (NFR-04).

Trade-off (documented in ``docs/product-notes.md``): state is per process, so with N instances
the effective limit is up to N times the configured one.

The bucket map is bounded: attacker-chosen keys (spoofable client addresses, random API keys)
cannot grow memory without limit.  When full, the least recently used bucket is evicted.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

from shortener.clock import Clock
from shortener.domain.errors import RateLimitedError

DEFAULT_MAX_ENTRIES = 100_000


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds until one token is available (0 when allowed)
    reset_after: int  # seconds until the bucket is full again

    def to_error(self) -> RateLimitedError:
        return RateLimitedError(
            limit=self.limit,
            remaining=self.remaining,
            retry_after=self.retry_after,
            reset_after=self.reset_after,
        )

    def headers(self) -> dict[str, str]:
        """``X-RateLimit-*`` headers (``Reset`` is a delta in seconds), plus ``Retry-After``."""
        values = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_after),
        }
        if not self.allowed:
            values["Retry-After"] = str(max(self.retry_after, 1))
        return values


class _Bucket:
    __slots__ = ("tokens", "updated")

    def __init__(self, tokens: float, updated: float) -> None:
        self.tokens = tokens
        self.updated = updated


class TokenBucketLimiter:
    def __init__(
        self,
        *,
        capacity: int,
        refill_per_second: float,
        clock: Clock,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        if capacity < 1 or refill_per_second <= 0 or max_entries < 1:
            raise ValueError("capacity, refill rate and max_entries must be positive")
        self._capacity = capacity
        self._rate = refill_per_second
        self._clock = clock
        self._max_entries = max_entries
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def acquire(self, key: str) -> RateLimitDecision:
        """Consume one token for ``key`` if available."""
        with self._lock:
            bucket = self._refilled(key, create=True)
            allowed = bucket.tokens >= 1.0
            if allowed:
                bucket.tokens -= 1.0
            return self._decision(bucket, allowed)

    def peek(self, key: str) -> RateLimitDecision:
        """Report whether ``key`` could acquire now, without consuming a token or storing state."""
        with self._lock:
            bucket = self._refilled(key, create=False)
            return self._decision(bucket, bucket.tokens >= 1.0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)

    def _refilled(self, key: str, *, create: bool) -> _Bucket:
        now = self._clock.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(float(self._capacity), now)
            if create:
                self._make_room()
                self._buckets[key] = bucket
        else:
            elapsed = max(0.0, now - bucket.updated)
            bucket.tokens = min(float(self._capacity), bucket.tokens + elapsed * self._rate)
            bucket.updated = now
            self._buckets.move_to_end(key)
        return bucket

    def _make_room(self) -> None:
        """Free one slot by evicting the least recently used bucket.

        The LRU head is also the bucket closest to being fully replenished, so evicting it
        loses the least rate-limit memory (a replenished bucket equals a fresh one).
        """
        if len(self._buckets) >= self._max_entries:
            self._buckets.popitem(last=False)

    def _decision(self, bucket: _Bucket, allowed: bool) -> RateLimitDecision:
        missing_for_one = max(0.0, 1.0 - bucket.tokens)
        missing_for_full = max(0.0, self._capacity - bucket.tokens)
        return RateLimitDecision(
            allowed=allowed,
            limit=self._capacity,
            remaining=int(bucket.tokens),
            retry_after=0 if allowed else math.ceil(missing_for_one / self._rate),
            reset_after=math.ceil(missing_for_full / self._rate),
        )
