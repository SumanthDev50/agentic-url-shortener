"""Link persistence behind the ``LinkRepository`` interface.

Parameterised SQL only.  Each statement is a complete literal (never assembled from pieces) so
it is trivially auditable.  ``code`` uniqueness is enforced by the database: the unique
constraint is the source of truth, and a violation surfaces as :class:`CodeTakenError`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from shortener.domain.models import Link, NewLink
from shortener.domain.timeutil import format_utc, parse_utc
from shortener.storage.db import Database

MAX_PAGE_SIZE = 100
MAX_OFFSET = 1_000_000


class CodeTakenError(Exception):
    """The ``links.code`` unique constraint rejected an insert."""


class LinkRepository(Protocol):
    """Storage interface used by ``LinkService`` (swap SQLite for another backend here)."""

    def insert(self, new: NewLink) -> Link:
        """Insert a link; raise :class:`CodeTakenError` if the code already exists."""
        ...

    def get_by_code(self, code: str) -> Link | None: ...

    def list_by_owner(self, owner: str, limit: int, offset: int) -> Sequence[Link]:
        """Newest first (``id DESC`` is stable and index-backed)."""
        ...

    def count_by_owner(self, owner: str) -> int: ...

    def disable(self, code: str, at: datetime) -> bool:
        """Soft-delete; ``True`` if this call changed the row (idempotent otherwise)."""
        ...

    def add_clicks(self, counts: Mapping[int, int]) -> None:
        """Add per-link click increments (called by the background writer)."""
        ...


class SqliteLinkRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def insert(self, new: NewLink) -> Link:
        try:
            with self._database.transaction() as connection:
                cursor = connection.execute(
                    "INSERT INTO links (code, original_url, created_at, expires_at, created_by) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        new.code,
                        new.original_url,
                        format_utc(new.created_at),
                        _format_optional(new.expires_at),
                        new.created_by,
                    ),
                )
        except sqlite3.IntegrityError as error:
            if _is_unique_violation(error):
                raise CodeTakenError(new.code) from error
            raise
        link_id = cursor.lastrowid
        if link_id is None:  # pragma: no cover - sqlite always sets it after a successful INSERT
            raise RuntimeError("insert did not return a row id")
        return Link(
            id=link_id,
            code=new.code,
            original_url=new.original_url,
            created_at=new.created_at,
            expires_at=new.expires_at,
            disabled_at=None,
            click_count=0,
            created_by=new.created_by,
        )

    def get_by_code(self, code: str) -> Link | None:
        row = (
            self._database.connection()
            .execute(
                "SELECT id, code, original_url, created_at, expires_at, disabled_at, "
                "click_count, created_by FROM links WHERE code = ?",
                (code,),
            )
            .fetchone()
        )
        return _to_link(row) if row is not None else None

    def list_by_owner(self, owner: str, limit: int, offset: int) -> Sequence[Link]:
        rows = (
            self._database.connection()
            .execute(
                "SELECT id, code, original_url, created_at, expires_at, disabled_at, "
                "click_count, created_by FROM links WHERE created_by = ? "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (owner, _bounded(limit, 1, MAX_PAGE_SIZE), _bounded(offset, 0, MAX_OFFSET)),
            )
            .fetchall()
        )
        return [_to_link(row) for row in rows]

    def count_by_owner(self, owner: str) -> int:
        row = (
            self._database.connection()
            .execute("SELECT COUNT(*) AS total FROM links WHERE created_by = ?", (owner,))
            .fetchone()
        )
        return int(row["total"])

    def disable(self, code: str, at: datetime) -> bool:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE links SET disabled_at = ? WHERE code = ? AND disabled_at IS NULL",
                (format_utc(at), code),
            )
        return cursor.rowcount > 0

    def add_clicks(self, counts: Mapping[int, int]) -> None:
        if not counts:
            return
        with self._database.transaction() as connection:
            connection.executemany(
                "UPDATE links SET click_count = click_count + ? WHERE id = ?",
                [(count, link_id) for link_id, count in counts.items()],
            )


def _to_link(row: sqlite3.Row) -> Link:
    return Link(
        id=int(row["id"]),
        code=str(row["code"]),
        original_url=str(row["original_url"]),
        created_at=parse_utc(row["created_at"]),
        expires_at=_parse_optional(row["expires_at"]),
        disabled_at=_parse_optional(row["disabled_at"]),
        click_count=int(row["click_count"]),
        created_by=row["created_by"],
    )


def _format_optional(moment: datetime | None) -> str | None:
    return format_utc(moment) if moment is not None else None


def _parse_optional(text: str | None) -> datetime | None:
    return parse_utc(text) if text is not None else None


def _is_unique_violation(error: sqlite3.IntegrityError) -> bool:
    return getattr(
        error, "sqlite_errorname", ""
    ) == "SQLITE_CONSTRAINT_UNIQUE" and "links.code" in str(error)


def _bounded(value: int, low: int, high: int) -> int:
    """Clamp LIMIT/OFFSET to a safe range (they are bound as parameters, never formatted)."""
    return max(low, min(int(value), high))
