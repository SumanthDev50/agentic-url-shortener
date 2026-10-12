"""Fixtures for the shortener tests."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shortener.config import RateLimit, Settings
from shortener.container import Services
from shortener.storage.db import Database
from tests.shortener.support import (
    CollectingHandler,
    FakeClock,
    SharedDatabase,
    api_client,
    build_app,
    build_migrated_template,
    make_settings,
    new_secret,
)


@pytest.fixture(scope="session")
def migrated_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_migrated_template(tmp_path_factory.mktemp("template"))


@pytest.fixture(scope="module")
def shared_database(
    tmp_path_factory: pytest.TempPathFactory, migrated_template: Path
) -> Iterator[SharedDatabase]:
    database = SharedDatabase(tmp_path_factory.mktemp("db"), migrated_template)
    yield database
    database.close()


@pytest.fixture
def database_path(shared_database: SharedDatabase) -> Path:
    """The module's migrated database file, emptied for this test."""
    return shared_database.reset()


@pytest.fixture
def migrated_database(database_path: Path) -> Iterator[Database]:
    database = Database(database_path)
    database.open()
    yield database
    database.close()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def secrets_by_owner() -> dict[str, str]:
    """Two API keys (``alice`` and ``bob``) generated for this test only."""
    return {"alice": new_secret(), "bob": new_secret()}


@pytest.fixture
def settings(database_path: Path, secrets_by_owner: dict[str, str]) -> Settings:
    return make_settings(database_path, keys=secrets_by_owner)


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> FastAPI:
    return build_app(settings, clock=clock)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """A client running the application lifespan (migrations, background writer)."""
    with api_client(app) as test_client:
        yield test_client


@pytest.fixture
def services(app: FastAPI) -> Services:
    return cast(Services, app.state.services)


@pytest.fixture
def alice(secrets_by_owner: dict[str, str]) -> dict[str, str]:
    return {"X-API-Key": secrets_by_owner["alice"]}


@pytest.fixture
def bob(secrets_by_owner: dict[str, str]) -> dict[str, str]:
    return {"X-API-Key": secrets_by_owner["bob"]}


@pytest.fixture
def log_lines() -> Iterator[CollectingHandler]:
    """Captures JSON log output of every ``shortener`` logger during the test."""
    handler = CollectingHandler()
    logger = logging.getLogger("shortener")
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


# --------------------------------------------------------------------------------------------
# Shared application: one long-lived app per test module for tests that neither depend on nor
# change global state (validation errors, headers, 404s, ...).  It avoids paying the ~40 ms
# application start-up per test; rate limits are effectively unlimited so tests cannot starve
# each other.  Anything asserting totals, counters, limits or timing uses the per-test fixtures.
# --------------------------------------------------------------------------------------------

UNLIMITED = RateLimit(burst=1_000_000, per_minute=6_000_000)


@dataclass(frozen=True)
class SharedApp:
    client: TestClient
    alice: dict[str, str]
    bob: dict[str, str]
    services: Services


@pytest.fixture(scope="module")
def shared(
    tmp_path_factory: pytest.TempPathFactory, migrated_template: Path
) -> Iterator[SharedApp]:
    keys = {"alice": new_secret(), "bob": new_secret()}
    database = SharedDatabase(tmp_path_factory.mktemp("shared"), migrated_template)
    settings = make_settings(
        database.path,
        keys=keys,
        api_rate=UNLIMITED,
        redirect_rate=UNLIMITED,
        auth_failure_rate=UNLIMITED,
    )
    app = build_app(settings, clock=FakeClock())
    with api_client(app) as test_client:
        yield SharedApp(
            client=test_client,
            alice={"X-API-Key": keys["alice"]},
            bob={"X-API-Key": keys["bob"]},
            services=cast(Services, app.state.services),
        )
    database.close()
