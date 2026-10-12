"""Clock abstraction.

Everything that depends on time (expiry, cache TTLs, rate limiting, retention) receives a
``Clock`` so tests can drive time deterministically instead of sleeping.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """Source of wall-clock and monotonic time."""

    def now(self) -> datetime:
        """Current time as a timezone-aware UTC datetime."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds, for measuring intervals (immune to wall-clock jumps)."""
        ...


class SystemClock:
    """The real clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()
