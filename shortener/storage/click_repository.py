"""Click persistence and aggregation behind the ``ClickRepository`` interface."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal, Protocol

from shortener.domain.models import ClickEvent, ClickSummary, NamedCount
from shortener.domain.timeutil import format_utc, parse_utc
from shortener.storage.db import Database

Dimension = Literal["browser", "device"]

#: Allow-list mapping for the breakdown dimension: user input never selects a column, and the
#: statements stay complete literals.
_BREAKDOWN_SQL: dict[str, str] = {
    "browser": (
        "SELECT browser AS name, COUNT(*) AS clicks FROM clicks "
        "WHERE link_id = ? AND clicked_at >= ? "
        "GROUP BY browser ORDER BY clicks DESC, name ASC LIMIT ?"
    ),
    "device": (
        "SELECT device AS name, COUNT(*) AS clicks FROM clicks "
        "WHERE link_id = ? AND clicked_at >= ? "
        "GROUP BY device ORDER BY clicks DESC, name ASC LIMIT ?"
    ),
}


class ClickRepository(Protocol):
    def insert_many(self, events: Sequence[ClickEvent]) -> None: ...

    def summary(self, link_id: int, since: datetime) -> ClickSummary: ...

    def daily_counts(self, link_id: int, since: datetime) -> dict[date, int]:
        """UTC calendar day -> clicks (days without clicks are absent)."""
        ...

    def top_referrers(self, link_id: int, since: datetime, limit: int) -> Sequence[NamedCount]: ...

    def breakdown(
        self, dimension: Dimension, link_id: int, since: datetime, limit: int
    ) -> Sequence[NamedCount]: ...

    def delete_before(self, cutoff: datetime) -> int:
        """Retention purge; returns the number of deleted events."""
        ...


class SqliteClickRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def insert_many(self, events: Sequence[ClickEvent]) -> None:
        if not events:
            return
        with self._database.transaction() as connection:
            connection.executemany(
                "INSERT INTO clicks (link_id, clicked_at, referrer_host, browser, device, "
                "visitor_hash) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        event.link_id,
                        format_utc(event.clicked_at),
                        event.referrer_host,
                        event.browser,
                        event.device,
                        event.visitor_hash,
                    )
                    for event in events
                ],
            )

    def summary(self, link_id: int, since: datetime) -> ClickSummary:
        row = (
            self._database.connection()
            .execute(
                "SELECT COUNT(*) AS total, COUNT(DISTINCT visitor_hash) AS visitors, "
                "MIN(clicked_at) AS first_click, MAX(clicked_at) AS last_click "
                "FROM clicks WHERE link_id = ? AND clicked_at >= ?",
                (link_id, format_utc(since)),
            )
            .fetchone()
        )
        return ClickSummary(
            total_clicks=int(row["total"]),
            unique_visitors=int(row["visitors"]),
            first_click_at=parse_utc(row["first_click"]) if row["first_click"] else None,
            last_click_at=parse_utc(row["last_click"]) if row["last_click"] else None,
        )

    def daily_counts(self, link_id: int, since: datetime) -> dict[date, int]:
        rows = (
            self._database.connection()
            .execute(
                "SELECT substr(clicked_at, 1, 10) AS day, COUNT(*) AS clicks FROM clicks "
                "WHERE link_id = ? AND clicked_at >= ? GROUP BY day",
                (link_id, format_utc(since)),
            )
            .fetchall()
        )
        return {date.fromisoformat(row["day"]): int(row["clicks"]) for row in rows}

    def top_referrers(self, link_id: int, since: datetime, limit: int) -> Sequence[NamedCount]:
        rows = (
            self._database.connection()
            .execute(
                "SELECT referrer_host AS name, COUNT(*) AS clicks FROM clicks "
                "WHERE link_id = ? AND clicked_at >= ? AND referrer_host IS NOT NULL "
                "GROUP BY referrer_host ORDER BY clicks DESC, name ASC LIMIT ?",
                (link_id, format_utc(since), max(1, int(limit))),
            )
            .fetchall()
        )
        return [NamedCount(name=str(row["name"]), clicks=int(row["clicks"])) for row in rows]

    def breakdown(
        self, dimension: Dimension, link_id: int, since: datetime, limit: int
    ) -> Sequence[NamedCount]:
        statement = _BREAKDOWN_SQL[dimension]
        rows = (
            self._database.connection()
            .execute(statement, (link_id, format_utc(since), max(1, int(limit))))
            .fetchall()
        )
        return [NamedCount(name=str(row["name"]), clicks=int(row["clicks"])) for row in rows]

    def delete_before(self, cutoff: datetime) -> int:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM clicks WHERE clicked_at < ?", (format_utc(cutoff),)
            )
        return max(cursor.rowcount, 0)
