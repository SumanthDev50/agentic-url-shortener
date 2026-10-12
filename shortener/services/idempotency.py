"""Idempotent creation (FR-10).

The stored response and the side effect it describes are committed in **one transaction**:

1. take the write lock (``BEGIN IMMEDIATE``);
2. look the ``(key, owner)`` up under that lock - a hit with the same request hash replays the
   stored response, a hit with a different hash is rejected;
3. otherwise run the operation and store its response;
4. commit both or neither.

Because the lookup happens under the write lock, two concurrent requests with the same key
cannot both execute the operation: the loser blocks, then replays the winner's response.
Failed operations store nothing, so they can be retried with the same key.  Keys expire after
24 hours (an expired row is ignored and replaced).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from shortener.clock import Clock
from shortener.domain.errors import IdempotencyKeyReuseError
from shortener.observability.metrics import Metrics
from shortener.storage.db import Transactions
from shortener.storage.idempotency_repository import IdempotencyRecord, IdempotencyRepository

IDEMPOTENCY_TTL = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class IdempotentResult:
    body: str  # the serialised JSON response
    replayed: bool


def request_fingerprint(payload: dict[str, object]) -> str:
    """Stable SHA-256 of a JSON-serialisable request payload (key order independent)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


class IdempotencyService:
    def __init__(
        self,
        *,
        repository: IdempotencyRepository,
        transactions: Transactions,
        clock: Clock,
        metrics: Metrics,
    ) -> None:
        self._repository = repository
        self._transactions = transactions
        self._clock = clock
        self._metrics = metrics

    def execute(
        self,
        *,
        key: str | None,
        owner: str,
        request_hash: str,
        operation: Callable[[], str],
    ) -> IdempotentResult:
        """Run ``operation`` (returns the response JSON) at most once per ``(key, owner)``."""
        if key is None:
            return IdempotentResult(body=operation(), replayed=False)
        with self._transactions.transaction():
            stored = self._stored_response(key, owner, request_hash)
            if stored is not None:
                self._metrics.inc("shortener_idempotent_replays_total")
                return IdempotentResult(body=stored, replayed=True)
            body = operation()
            self._repository.insert(
                IdempotencyRecord(
                    key=key,
                    owner=owner,
                    request_hash=request_hash,
                    response_json=body,
                    created_at=self._clock.now(),
                )
            )
        return IdempotentResult(body=body, replayed=False)

    def purge_expired(self) -> int:
        return self._repository.delete_created_before(self._clock.now() - IDEMPOTENCY_TTL)

    def _stored_response(self, key: str, owner: str, request_hash: str) -> str | None:
        record = self._repository.get(key, owner)
        if record is None:
            return None
        if self._clock.now() - record.created_at >= IDEMPOTENCY_TTL:
            self._repository.delete(key, owner)
            return None
        if record.request_hash != request_hash:
            raise IdempotencyKeyReuseError
        return record.response_json
