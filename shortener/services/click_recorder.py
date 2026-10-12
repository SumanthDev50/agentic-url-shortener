"""Non-blocking click ingestion (NFR-01).

``ClickRecorder.submit`` appends to a bounded in-memory queue and returns immediately: when
the queue is full the event is dropped and counted, it never blocks the redirect.  A single
background thread drains the queue in batches through a *sink*; a failing sink is logged and
counted, the thread survives, and nothing propagates to request handling.  ``stop`` flushes
everything still queued so a graceful shutdown loses no accepted click.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, deque
from collections.abc import Callable, Sequence

from shortener.domain.models import ClickEvent
from shortener.observability.metrics import Metrics
from shortener.storage.click_repository import ClickRepository
from shortener.storage.db import Transactions
from shortener.storage.link_repository import LinkRepository

logger = logging.getLogger(__name__)

DEFAULT_CAPACITY = 10_000
DEFAULT_BATCH_SIZE = 200
DEFAULT_JOIN_TIMEOUT = 10.0

ClickSink = Callable[[Sequence[ClickEvent]], None]


class ClickWriter:
    """Sink that persists a batch of clicks and bumps the link counters atomically.

    ``click_count`` is therefore eventually consistent: it lags a redirect by the queue delay
    (see ``docs/product-notes.md`` for the trade-off and the limited-click implications).
    """

    def __init__(
        self, clicks: ClickRepository, links: LinkRepository, transactions: Transactions
    ) -> None:
        self._clicks = clicks
        self._links = links
        self._transactions = transactions

    def write(self, events: Sequence[ClickEvent]) -> None:
        per_link = Counter(event.link_id for event in events)
        with self._transactions.transaction():
            self._clicks.insert_many(events)
            self._links.add_clicks(per_link)


class ClickRecorder:
    def __init__(
        self,
        sink: ClickSink,
        *,
        metrics: Metrics,
        capacity: int = DEFAULT_CAPACITY,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if capacity < 1 or batch_size < 1:
            raise ValueError("capacity and batch_size must be positive")
        self._sink = sink
        self._metrics = metrics
        self._capacity = capacity
        self._batch_size = batch_size
        self._queue: deque[ClickEvent] = deque()
        self._condition = threading.Condition()
        self._in_flight = 0
        self._closed = False
        self._thread: threading.Thread | None = None

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._queue)

    def submit(self, event: ClickEvent) -> bool:
        """Queue ``event``; ``False`` (and a counter) if it was dropped.  Never blocks."""
        with self._condition:
            if self._closed:
                reason = "shutdown"
            elif len(self._queue) >= self._capacity:
                reason = "queue_full"
            else:
                self._queue.append(event)
                self._condition.notify()
                reason = ""
        if reason:
            self._metrics.inc("shortener_clicks_dropped_total", reason=reason)
            return False
        self._metrics.inc("shortener_clicks_enqueued_total")
        return True

    def start(self) -> None:
        """Start the background writer (idempotent); re-opens a previously stopped recorder."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._closed = False
            self._thread = threading.Thread(target=self._run, name="click-writer", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = DEFAULT_JOIN_TIMEOUT) -> None:
        """Stop accepting clicks, write everything queued, and join the thread."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
        self._drain_inline()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until everything queued so far is written; ``False`` on timeout."""
        with self._condition:
            running = self._thread is not None and self._thread.is_alive()
            if running:
                return self._condition.wait_for(
                    lambda: not self._queue and self._in_flight == 0, timeout
                )
        self._drain_inline()
        return True

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: bool(self._queue) or self._closed)
                batch = self._take_batch()
                self._in_flight = len(batch)
            if not batch:
                return
            self._write(batch)
            with self._condition:
                self._in_flight = 0
                self._condition.notify_all()

    def _drain_inline(self) -> None:
        """Write whatever is left on the calling thread (after the worker has exited)."""
        while True:
            with self._condition:
                if self._thread is not None and self._thread.is_alive():
                    return  # the worker is still draining; it will finish the queue itself
                batch = self._take_batch()
            if not batch:
                return
            self._write(batch)

    def _take_batch(self) -> list[ClickEvent]:
        size = min(self._batch_size, len(self._queue))
        return [self._queue.popleft() for _ in range(size)]

    def _write(self, batch: Sequence[ClickEvent]) -> None:
        try:
            self._sink(batch)
        except Exception:
            self._metrics.inc(
                "shortener_clicks_dropped_total", float(len(batch)), reason="write_failed"
            )
            self._metrics.inc("shortener_analytics_errors_total")
            logger.exception("click_batch_dropped", extra={"fields": {"size": len(batch)}})
        else:
            self._metrics.inc("shortener_clicks_written_total", float(len(batch)))
