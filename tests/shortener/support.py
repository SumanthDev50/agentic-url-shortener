"""Shared test helpers for the shortener test-suite.

Secrets used by the tests are generated at runtime (``secrets.token_urlsafe``); no secret-shaped
literal exists in the repository.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from shortener.config import ApiKeyDigest, Settings
from shortener.domain.codegen import CodeGenerator
from shortener.main import create_app
from shortener.observability.logging import JsonFormatter
from shortener.services.auth import digest_api_key
from shortener.storage.db import Database
from shortener.storage.migrations import migrate

START = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
BASE_URL = "https://sho.example.com"
CLIENT_ADDRESS = "203.0.113.77"


class FakeClock:
    """Deterministic clock: ``advance`` moves wall-clock and monotonic time together."""

    def __init__(self, start: datetime = START) -> None:
        self._now = start
        self._monotonic = 1000.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds


def new_secret() -> str:
    """A fresh, valid API-key secret (never a literal)."""
    return secrets.token_urlsafe(32)


def build_migrated_template(directory: Path) -> Path:
    """A fully migrated database file; copying it is ~20x faster than migrating per test."""
    path = directory / "template.db"
    database = Database(path)
    database.open()
    migrate(database)
    database.close()
    return path


class SharedDatabase:
    """A migrated database file reused by the tests of a module.

    Creating a database file (plus its WAL/shm files) per test is the dominant I/O cost of the
    suite on Windows.  Instead one file is copied from the migrated template once; ``reset``
    empties the tables between tests.  An idle keep-alive connection stops SQLite from deleting
    and re-creating the WAL files every time the application closes its own connections.
    """

    def __init__(self, directory: Path, template: Path) -> None:
        self.path = directory / "shared.db"
        shutil.copyfile(template, self.path)
        self._keepalive = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._keepalive.execute("PRAGMA journal_mode = WAL")

    def reset(self) -> Path:
        self._keepalive.execute("DELETE FROM clicks")
        self._keepalive.execute("DELETE FROM links")
        self._keepalive.execute("DELETE FROM idempotency_keys")
        return self.path

    def close(self) -> None:
        self._keepalive.close()


def make_settings(
    database_path: Path,
    *,
    keys: dict[str, str] | None = None,
    **overrides: Any,
) -> Settings:
    """Settings for tests: no failure delay and an explicit analytics secret.

    ``database_path`` should point at a migrated file (see ``SharedDatabase``) unless the test
    exercises the application's own start-up migration.
    """
    key_map = keys if keys is not None else {"alice": new_secret()}
    values: dict[str, Any] = {
        "base_url": BASE_URL,
        "database_path": str(database_path),
        "api_keys": tuple(ApiKeyDigest(key_id, digest_api_key(s)) for key_id, s in key_map.items()),
        "analytics_secret": secrets.token_bytes(32),
        "own_hosts": ("sho.example.com",),
        "auth_failure_delay_ms": 0,
    }
    values.update(overrides)
    return Settings(**values)


def build_app(
    settings: Settings,
    *,
    clock: FakeClock | None = None,
    code_generator: CodeGenerator | None = None,
) -> FastAPI:
    return create_app(
        settings,
        clock=clock or FakeClock(),
        code_generator=code_generator,
        sleep=lambda _seconds: None,
    )


def api_client(app: FastAPI) -> TestClient:
    """A client that does not follow redirects and has a fixed public client address."""
    return TestClient(app, client=(CLIENT_ADDRESS, 50000), follow_redirects=False)


def dump_database(path: Path) -> str:
    """Every table row of the database as one string (to assert nothing sensitive is stored)."""
    connection = sqlite3.connect(path)
    try:
        return "\n".join(connection.iterdump())
    finally:
        connection.close()


class CollectingHandler(logging.Handler):
    """Collects formatted log lines (JSON) from the ``shortener`` loggers."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(JsonFormatter())
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)
