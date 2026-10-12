"""Periodic housekeeping: retention purge of clicks and expired idempotency keys.

Runs once synchronously at startup and then on a fixed interval from a daemon thread.  A job
that raises is logged and retried at the next interval; it never takes the service down.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DAILY_SECONDS = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class MaintenanceJob:
    name: str
    run: Callable[[], int]  # returns the number of rows removed


class MaintenanceRunner:
    def __init__(
        self, jobs: Sequence[MaintenanceJob], *, interval_seconds: float = DAILY_SECONDS
    ) -> None:
        self._jobs = tuple(jobs)
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> dict[str, int]:
        """Run every job now; failed jobs report ``-1``."""
        results: dict[str, int] = {}
        for job in self._jobs:
            try:
                results[job.name] = job.run()
            except Exception:
                logger.exception("maintenance_job_failed", extra={"fields": {"job": job.name}})
                results[job.name] = -1
        return results

    def start(self) -> None:
        """Run the jobs now, then keep running them every interval until :meth:`stop`."""
        if self._thread is not None and self._thread.is_alive():
            return
        self.run_once()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="maintenance", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.run_once()
