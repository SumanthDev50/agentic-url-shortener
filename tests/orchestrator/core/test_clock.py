"""Clock implementations: SystemClock basics, FakeClock determinism and thread-safety."""

from __future__ import annotations

import math
import threading
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from orchestrator.core.clock import FakeClock, SystemClock
from orchestrator.core.errors import ConfigError
from orchestrator.core.types import Clock


class TestSystemClock:
    def test_now_is_timezone_aware_utc(self) -> None:
        now = SystemClock().now()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)

    def test_monotonic_never_decreases(self) -> None:
        clock = SystemClock()
        samples = [clock.monotonic() for _ in range(50)]
        assert samples == sorted(samples)

    @pytest.mark.parametrize("seconds", [0, -1, -1e9])
    def test_sleep_ignores_non_positive_durations(self, seconds: float) -> None:
        before = time.monotonic()
        SystemClock().sleep(seconds)
        assert time.monotonic() - before < 0.05

    def test_sleep_really_sleeps(self) -> None:
        before = time.monotonic()
        SystemClock().sleep(0.02)
        assert time.monotonic() - before >= 0.015

    def test_satisfies_the_clock_protocol(self) -> None:
        assert isinstance(SystemClock(), Clock)


class TestFakeClock:
    def test_default_start_is_a_fixed_utc_instant(self) -> None:
        a, b = FakeClock(), FakeClock()
        assert a.now() == b.now() == datetime(2026, 1, 1, tzinfo=UTC)

    def test_custom_start_is_normalised_to_utc(self) -> None:
        start = datetime(2030, 5, 6, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        clock = FakeClock(start)
        assert clock.now() == start
        assert clock.now().utcoffset() == timedelta(0)

    def test_naive_start_is_rejected(self) -> None:
        with pytest.raises(ConfigError):
            FakeClock(datetime(2030, 1, 1))

    def test_time_stands_still_until_moved(self) -> None:
        clock = FakeClock()
        assert clock.now() == clock.now()
        assert clock.monotonic() == clock.monotonic()

    def test_sleep_advances_virtual_time_instantly(self) -> None:
        clock = FakeClock()
        start, mono = clock.now(), clock.monotonic()
        wall = time.monotonic()
        clock.sleep(3600)
        assert time.monotonic() - wall < 0.05  # an hour of virtual time cost no real time
        assert clock.now() - start == timedelta(hours=1)
        assert clock.monotonic() - mono == 3600

    def test_now_and_monotonic_stay_consistent(self) -> None:
        clock = FakeClock()
        start, mono = clock.now(), clock.monotonic()
        for step in (0.1, 2.5, 0.000001, 40):
            clock.advance(step)
            assert (clock.now() - start).total_seconds() == pytest.approx(clock.monotonic() - mono)

    def test_monotonic_origin_is_not_zero(self) -> None:
        assert FakeClock().monotonic() > 0

    def test_advance_accumulates_exactly_in_microseconds(self) -> None:
        clock = FakeClock()
        start = clock.now()
        for _ in range(10):
            clock.advance(0.1)  # 0.1 * 10 != 1.0 in floating point; microseconds make it exact
        assert clock.now() - start == timedelta(seconds=1)

    @pytest.mark.parametrize("seconds", [-0.001, math.nan, math.inf, -math.inf, True, "1", None])
    def test_advance_rejects_invalid_values(self, seconds: object) -> None:
        clock = FakeClock()
        before = clock.now()
        with pytest.raises(ValueError):
            clock.advance(seconds)  # type: ignore[arg-type]
        assert clock.now() == before

    @pytest.mark.parametrize("seconds", [0, -5])
    def test_sleep_of_non_positive_duration_is_a_no_op(self, seconds: float) -> None:
        clock = FakeClock()
        before = clock.now()
        clock.sleep(seconds)
        assert clock.now() == before

    def test_concurrent_sleeps_add_up_on_one_timeline(self) -> None:
        clock = FakeClock()
        start = clock.now()
        threads, per_thread = 8, 500

        def worker() -> None:
            for _ in range(per_thread):
                clock.sleep(0.001)

        pool = [threading.Thread(target=worker) for _ in range(threads)]
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        assert clock.now() - start == timedelta(seconds=threads * per_thread * 0.001)

    def test_readers_never_see_time_go_backwards(self) -> None:
        clock = FakeClock()
        stop = threading.Event()
        seen: list[datetime] = []

        def reader() -> None:
            while not stop.is_set():
                seen.append(clock.now())

        thread = threading.Thread(target=reader)
        thread.start()
        for _ in range(200):
            clock.advance(0.01)
        stop.set()
        thread.join()
        assert seen == sorted(seen)

    def test_satisfies_the_clock_protocol(self) -> None:
        assert isinstance(FakeClock(), Clock)
