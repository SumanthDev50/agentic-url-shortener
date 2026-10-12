"""Click recorder: bounded queue, drop counting, batching, flush on stop, failure isolation."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from shortener.domain.models import ClickEvent, NewLink
from shortener.observability.metrics import Metrics
from shortener.services.click_recorder import ClickRecorder, ClickWriter
from shortener.storage.click_repository import SqliteClickRepository
from shortener.storage.db import Database
from shortener.storage.link_repository import SqliteLinkRepository

T0 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def event(link_id: int = 1, visitor: str = "v") -> ClickEvent:
    return ClickEvent(link_id, T0, None, "Chrome", "desktop", visitor)


class CollectingSink:
    def __init__(self) -> None:
        self.batches: list[list[ClickEvent]] = []
        self.lock = threading.Lock()

    def __call__(self, events: Sequence[ClickEvent]) -> None:
        with self.lock:
            self.batches.append(list(events))

    @property
    def total(self) -> int:
        return sum(len(batch) for batch in self.batches)


def test_submit_never_blocks_and_drops_when_full() -> None:
    metrics = Metrics()
    sink = CollectingSink()
    recorder = ClickRecorder(sink, metrics=metrics, capacity=3, batch_size=10)  # not started
    results = [recorder.submit(event(visitor=str(i))) for i in range(5)]
    assert results == [True, True, True, False, False]
    assert recorder.depth == 3
    assert metrics.value("shortener_clicks_enqueued_total") == 3
    assert metrics.value("shortener_clicks_dropped_total", reason="queue_full") == 2
    assert sink.total == 0  # nothing is written synchronously on the caller's thread


def test_flush_without_worker_drains_inline_in_batches() -> None:
    sink = CollectingSink()
    recorder = ClickRecorder(sink, metrics=Metrics(), capacity=10, batch_size=2)
    for index in range(5):
        recorder.submit(event(visitor=str(index)))
    assert recorder.flush() is True
    assert [len(batch) for batch in sink.batches] == [2, 2, 1]
    assert recorder.depth == 0


def test_background_worker_writes_events_and_flush_waits_for_it() -> None:
    metrics = Metrics()
    sink = CollectingSink()
    recorder = ClickRecorder(sink, metrics=metrics, capacity=100, batch_size=10)
    recorder.start()
    recorder.start()  # idempotent
    try:
        for index in range(25):
            recorder.submit(event(visitor=str(index)))
        assert recorder.flush(timeout=5) is True
        assert sink.total == 25
        assert metrics.value("shortener_clicks_written_total") == 25
        assert all(len(batch) <= 10 for batch in sink.batches)
    finally:
        recorder.stop()


def test_stop_flushes_everything_queued_and_rejects_later_events() -> None:
    metrics = Metrics()
    gate = threading.Event()
    sink = CollectingSink()

    def slow_sink(events: Sequence[ClickEvent]) -> None:
        gate.wait(timeout=5)
        sink(events)

    recorder = ClickRecorder(slow_sink, metrics=metrics, capacity=100, batch_size=4)
    recorder.start()
    for index in range(10):
        recorder.submit(event(visitor=str(index)))
    gate.set()
    recorder.stop()
    assert sink.total == 10  # nothing accepted was lost on shutdown
    assert recorder.submit(event()) is False
    assert metrics.value("shortener_clicks_dropped_total", reason="shutdown") == 1


def test_failing_sink_is_counted_and_does_not_kill_the_worker() -> None:
    metrics = Metrics()
    attempts: list[int] = []
    delivered: list[ClickEvent] = []

    def flaky(events: Sequence[ClickEvent]) -> None:
        attempts.append(len(events))
        if len(attempts) == 1:
            raise RuntimeError("database is locked")
        delivered.extend(events)

    recorder = ClickRecorder(flaky, metrics=metrics, capacity=10, batch_size=1)
    recorder.start()
    try:
        recorder.submit(event(visitor="lost"))
        assert recorder.flush(timeout=5)
        recorder.submit(event(visitor="kept"))
        assert recorder.flush(timeout=5)
    finally:
        recorder.stop()
    assert [e.visitor_hash for e in delivered] == ["kept"]
    assert metrics.value("shortener_clicks_dropped_total", reason="write_failed") == 1
    assert metrics.value("shortener_analytics_errors_total") == 1
    assert metrics.value("shortener_clicks_written_total") == 1


def test_failing_sink_during_inline_drain_is_contained() -> None:
    def broken(_events: Sequence[ClickEvent]) -> None:
        raise OSError("disk full")

    recorder = ClickRecorder(broken, metrics=Metrics(), capacity=5, batch_size=5)
    recorder.submit(event())
    assert recorder.flush() is True  # no exception escapes


def test_recorder_can_be_restarted_after_stop() -> None:
    sink = CollectingSink()
    recorder = ClickRecorder(sink, metrics=Metrics(), capacity=5, batch_size=5)
    recorder.start()
    recorder.stop()
    recorder.start()
    try:
        assert recorder.submit(event()) is True
        assert recorder.flush(timeout=5)
        assert sink.total == 1
    finally:
        recorder.stop()


def test_concurrent_producers_never_exceed_capacity_or_lose_accepted_events() -> None:
    sink = CollectingSink()
    metrics = Metrics()
    recorder = ClickRecorder(sink, metrics=metrics, capacity=50, batch_size=7)
    recorder.start()
    accepted: list[int] = []

    def produce() -> None:
        count = sum(1 for _ in range(200) if recorder.submit(event()))
        accepted.append(count)

    threads = [threading.Thread(target=produce) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    recorder.stop()
    assert sink.total == sum(accepted)
    dropped = metrics.value("shortener_clicks_dropped_total", reason="queue_full")
    assert sum(accepted) + dropped == 1200


@pytest.mark.parametrize("kwargs", [{"capacity": 0}, {"batch_size": 0}])
def test_invalid_configuration(kwargs: dict[str, int]) -> None:
    arguments = {"capacity": 1, "batch_size": 1, **kwargs}
    with pytest.raises(ValueError, match="must be positive"):
        ClickRecorder(CollectingSink(), metrics=Metrics(), **arguments)


def test_queue_depth_gauge_is_rendered() -> None:
    metrics = Metrics()
    recorder = ClickRecorder(CollectingSink(), metrics=metrics, capacity=5, batch_size=5)
    metrics.register_gauge("shortener_click_queue_depth", lambda: float(recorder.depth))
    recorder.submit(event())
    recorder.submit(event())
    assert "shortener_click_queue_depth 2" in metrics.render()


# ----------------------------------------------------------------------------- writer


def test_click_writer_persists_events_and_counters_atomically(
    migrated_database: Database,
) -> None:
    database = migrated_database
    links = SqliteLinkRepository(database)
    clicks = SqliteClickRepository(database)
    link = links.insert(NewLink("abc", "https://example.com/", T0, None, "alice"))
    writer = ClickWriter(clicks, links, database)

    writer.write([event(link.id, "a"), event(link.id, "b")])
    assert links.get_by_code("abc").click_count == 2  # type: ignore[union-attr]
    assert clicks.summary(link.id, T0).total_clicks == 2

    # A failing batch (unknown link -> foreign key violation) must not bump any counter.
    with pytest.raises(Exception, match="FOREIGN KEY"):
        writer.write([event(link.id, "c"), event(9999, "d")])
    assert links.get_by_code("abc").click_count == 2  # type: ignore[union-attr]
    assert clicks.summary(link.id, T0).total_clicks == 2
