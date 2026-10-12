"""SQLite connection management (NFR-09).

* One connection per thread (``sqlite3`` connections must not be shared across concurrent
  users), created lazily with the pragmas the spec requires: ``foreign_keys=ON``,
  ``trusted_schema=OFF``, ``busy_timeout`` and WAL journaling.
* Connections run in autocommit mode (``isolation_level=None``); atomic units of work use
  :meth:`Database.transaction`, which issues an explicit ``BEGIN IMMEDIATE`` so writers queue on
  ``busy_timeout`` instead of failing with ``SQLITE_BUSY`` halfway through, and which nests
  (an inner ``transaction()`` joins the outer one).
* Only file databases are supported: every thread needs to see the same data.
"""

from __future__ import annotations

import sqlite3
import threading
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from pathlib import Path
from typing import Protocol

DEFAULT_BUSY_TIMEOUT_MS = 5000


class Transactions(Protocol):
    """What services need from storage to make several repository calls atomic."""

    def transaction(self) -> AbstractContextManager[object]: ...


class _ConnectionHolder:
    """Owns one connection; closes it when garbage collected (e.g. its thread ended).

    ``sqlite3.Connection`` itself is not weak-referenceable, hence this small wrapper.
    """

    __slots__ = ("__weakref__", "connection")

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection: sqlite3.Connection | None = connection

    def close(self) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            with suppress(sqlite3.Error):
                connection.close()

    def __del__(self) -> None:
        self.close()


class Database:
    """Per-thread SQLite connections for one database file."""

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS) -> None:
        text = str(path)
        if not text or text.startswith(":memory:") or "\x00" in text:
            raise ValueError("a file path is required (per-thread connections need shared state)")
        self._path = Path(text)
        self._busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()
        self._holders: weakref.WeakSet[_ConnectionHolder] = weakref.WeakSet()
        self._holders_lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def open(self) -> None:
        """Create the parent directory and verify the file can be opened."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.connection().execute("SELECT 1").fetchone()

    def connection(self) -> sqlite3.Connection:
        """The calling thread's connection (opened on first use)."""
        existing: _ConnectionHolder | None = getattr(self._local, "holder", None)
        if existing is not None and existing.connection is not None:
            return existing.connection
        connection = self._connect()
        holder = _ConnectionHolder(connection)
        self._local.holder = holder
        with self._holders_lock:
            self._holders.add(holder)
        return connection

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1000,  # sets PRAGMA busy_timeout (no bind params)
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run the block atomically; nested use joins the outermost transaction."""
        connection = self.connection()
        depth: int = getattr(self._local, "depth", 0)
        if depth > 0:
            self._local.depth = depth + 1
            try:
                yield connection
            finally:
                self._local.depth -= 1
            return
        connection.execute("BEGIN IMMEDIATE")
        self._local.depth = 1
        try:
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            with suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            raise
        finally:
            self._local.depth = 0

    def ping(self) -> bool:
        """Cheap readiness probe; never raises."""
        try:
            self.connection().execute("SELECT 1").fetchone()
        except sqlite3.Error:
            return False
        return True

    def close(self) -> None:
        """Close every live connection (call when no request is in flight)."""
        with self._holders_lock:
            holders = list(self._holders)
            self._holders.clear()
        for holder in holders:
            holder.close()
        self._local = threading.local()
