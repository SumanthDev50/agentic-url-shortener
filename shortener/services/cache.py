"""Thread-safe LRU cache with per-entry TTL (NFR-02).

Bounded by ``max_entries`` (least recently used entries are evicted) and by TTL (expired entries
are dropped on access).  Time comes from an injected clock so tests need no sleeping.  Negative
caching is a caller concern: store a sentinel with a short ``ttl``.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

from shortener.clock import Clock

K = TypeVar("K")
V = TypeVar("V")


class LruTtlCache(Generic[K, V]):
    def __init__(self, *, max_entries: int, default_ttl: float, clock: Clock) -> None:
        if max_entries < 1 or default_ttl <= 0:
            raise ValueError("max_entries and default_ttl must be positive")
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._clock = clock
        self._entries: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        """The live value for ``key`` (refreshing its recency) or ``None``."""
        now = self._clock.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if now >= expires_at:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def put(self, key: K, value: V, ttl: float | None = None) -> None:
        expires_at = self._clock.monotonic() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            self._entries[key] = (value, expires_at)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def invalidate(self, key: K) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
