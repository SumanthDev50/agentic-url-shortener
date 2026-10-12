"""Injectable time sources (``types.Clock`` implementations).

``SystemClock`` is the production clock. ``FakeClock`` is a deterministic, thread-safe virtual
timeline for tests and demos: ``sleep`` advances virtual time *instantly*, so retry back-offs cost
nothing and runs are reproducible.

``FakeClock`` keeps ONE global timeline (engine-semantics §15): when several threads sleep, their
sleeps add up instead of overlapping.  Real concurrency must therefore be observed with real time.
Elapsed time is stored as an integer number of microseconds so that ``now()`` and ``monotonic()``
can never drift apart through float accumulation.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import UTC, datetime, timedelta

from orchestrator.core.errors import ConfigError

_DEFAULT_START = datetime(2026, 1, 1, tzinfo=UTC)
# A non-zero origin keeps ``if clock.monotonic():`` style bugs visible instead of hiding at t=0.
_MONOTONIC_ORIGIN = 1000.0
_MICROS = 1_000_000


class SystemClock:
    """Wall-clock UTC time, the process monotonic clock and real ``time.sleep``."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds``; negative values are treated as zero (like a due back-off)."""
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """Deterministic virtual clock. All methods are thread-safe."""

    def __init__(self, start: datetime | None = None) -> None:
        start = _DEFAULT_START if start is None else start
        if start.tzinfo is None or start.utcoffset() is None:
            raise ConfigError("FakeClock start must be timezone-aware")
        self._start = start.astimezone(UTC)
        self._elapsed_us = 0
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._start + timedelta(microseconds=self._elapsed_us)

    def monotonic(self) -> float:
        with self._lock:
            return _MONOTONIC_ORIGIN + self._elapsed_us / _MICROS

    def sleep(self, seconds: float) -> None:
        """Advance virtual time by ``seconds`` without waiting (negative values are a no-op)."""
        if seconds > 0:
            self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Move the virtual clock forward. Time never runs backwards: negative or non-finite
        values raise ``ValueError``."""
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
        ):
            raise ValueError("advance() needs a finite number of seconds")
        if seconds < 0:
            raise ValueError("a clock cannot run backwards")
        delta = round(seconds * _MICROS)
        with self._lock:
            self._elapsed_us += delta
