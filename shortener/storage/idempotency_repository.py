"""Idempotency-key persistence (FR-10) behind the ``IdempotencyRepository`` interface."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from shortener.domain.timeutil import format_utc, parse_utc
from shortener.storage.db import Database


class IdempotencyKeyExistsError(Exception):
    """The ``(key, owner)`` primary key already has a stored response."""


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    key: str
    owner: str
    request_hash: str
    response_json: str
    created_at: datetime


class IdempotencyRepository(Protocol):
    def get(self, key: str, owner: str) -> IdempotencyRecord | None: ...

    def insert(self, record: IdempotencyRecord) -> None:
        """Store a response; raise :class:`IdempotencyKeyExistsError` on a primary-key clash."""
        ...

    def delete(self, key: str, owner: str) -> None: ...

    def delete_created_before(self, cutoff: datetime) -> int:
        """Purge expired keys; returns the number of deleted rows."""
        ...


class SqliteIdempotencyRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def get(self, key: str, owner: str) -> IdempotencyRecord | None:
        row = (
            self._database.connection()
            .execute(
                "SELECT key, owner, request_hash, response_json, created_at "
                "FROM idempotency_keys WHERE key = ? AND owner = ?",
                (key, owner),
            )
            .fetchone()
        )
        if row is None:
            return None
        return IdempotencyRecord(
            key=str(row["key"]),
            owner=str(row["owner"]),
            request_hash=str(row["request_hash"]),
            response_json=str(row["response_json"]),
            created_at=parse_utc(row["created_at"]),
        )

    def insert(self, record: IdempotencyRecord) -> None:
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    "INSERT INTO idempotency_keys "
                    "(key, owner, request_hash, response_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        record.key,
                        record.owner,
                        record.request_hash,
                        record.response_json,
                        format_utc(record.created_at),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise IdempotencyKeyExistsError(record.key) from error

    def delete(self, key: str, owner: str) -> None:
        with self._database.transaction() as connection:
            connection.execute(
                "DELETE FROM idempotency_keys WHERE key = ? AND owner = ?", (key, owner)
            )

    def delete_created_before(self, cutoff: datetime) -> int:
        with self._database.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM idempotency_keys WHERE created_at < ?", (format_utc(cutoff),)
            )
        return max(cursor.rowcount, 0)
