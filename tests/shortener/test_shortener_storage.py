"""SQLite storage: connections/pragmas, migrations (incl. checksum tamper), repositories."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from shortener.domain.models import ClickEvent, NewLink
from shortener.storage.click_repository import SqliteClickRepository
from shortener.storage.db import Database
from shortener.storage.idempotency_repository import (
    IdempotencyKeyExistsError,
    IdempotencyRecord,
    SqliteIdempotencyRepository,
)
from shortener.storage.link_repository import CodeTakenError, SqliteLinkRepository
from shortener.storage.migrations import (
    Migration,
    MigrationError,
    checksum_of,
    load_migrations,
    migrate,
    split_statements,
)

T0 = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """An empty (unmigrated) database."""
    db = Database(tmp_path / "test.db")
    db.open()
    yield db
    db.close()


@pytest.fixture
def migrated(migrated_database: Database) -> Database:
    """A database that already has the packaged migrations applied (shared, emptied file)."""
    return migrated_database


def write_migrations(directory: Path, scripts: dict[str, str]) -> list[Migration]:
    directory.mkdir(parents=True, exist_ok=True)
    for name, sql in scripts.items():
        (directory / name).write_text(sql, encoding="utf-8", newline="\n")
    return load_migrations(directory)


# ----------------------------------------------------------------------------- connections


def test_pragmas_are_applied(database: Database) -> None:
    connection = database.connection()
    statements = {
        "foreign_keys": "PRAGMA foreign_keys",
        "trusted_schema": "PRAGMA trusted_schema",
        "journal_mode": "PRAGMA journal_mode",
        "busy_timeout": "PRAGMA busy_timeout",
    }

    def pragma(name: str) -> object:
        return connection.execute(statements[name]).fetchone()[0]

    assert pragma("foreign_keys") == 1
    assert pragma("trusted_schema") == 0
    assert pragma("journal_mode") == "wal"
    assert pragma("busy_timeout") == 5000


def test_each_thread_gets_its_own_connection(database: Database) -> None:
    seen: list[sqlite3.Connection] = []
    main_connection = database.connection()
    assert database.connection() is main_connection

    thread = threading.Thread(target=lambda: seen.append(database.connection()))
    thread.start()
    thread.join()
    assert seen[0] is not main_connection


def test_memory_and_empty_paths_are_rejected(tmp_path: Path) -> None:
    for bad in (":memory:", "", "x\x00y"):
        with pytest.raises(ValueError, match="file path"):
            Database(bad)


def test_open_creates_missing_parent_directories(tmp_path: Path) -> None:
    db = Database(tmp_path / "a" / "b" / "c.db")
    db.open()
    assert (tmp_path / "a" / "b" / "c.db").exists()
    db.close()


def test_close_closes_every_thread_connection_and_allows_reopening(database: Database) -> None:
    opened: list[sqlite3.Connection] = []
    thread = threading.Thread(target=lambda: opened.append(database.connection()))
    thread.start()
    thread.join()
    main_connection = database.connection()
    database.close()
    with pytest.raises(sqlite3.ProgrammingError):
        main_connection.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")
    assert database.ping()  # lazily reopened


def test_ping_reports_failure_instead_of_raising(tmp_path: Path) -> None:
    db = Database(tmp_path / "gone.db")
    db.open()
    connection = db.connection()
    connection.close()  # simulate a broken connection
    assert db.ping() is False
    db.close()


def test_transaction_commits_and_rolls_back(migrated: Database) -> None:
    with migrated.transaction() as connection:
        connection.execute("INSERT INTO idempotency_keys VALUES ('k1','o','h','{}','t')")
    with pytest.raises(RuntimeError, match="boom"), migrated.transaction() as connection:
        connection.execute("INSERT INTO idempotency_keys VALUES ('k2','o','h','{}','t')")
        raise RuntimeError("boom")
    rows = migrated.connection().execute("SELECT key FROM idempotency_keys").fetchall()
    assert [row["key"] for row in rows] == ["k1"]


def test_nested_transactions_join_the_outer_one(migrated: Database) -> None:
    with pytest.raises(RuntimeError), migrated.transaction() as outer:
        outer.execute("INSERT INTO idempotency_keys VALUES ('k1','o','h','{}','t')")
        with migrated.transaction() as inner:
            inner.execute("INSERT INTO idempotency_keys VALUES ('k2','o','h','{}','t')")
        raise RuntimeError
    count = migrated.connection().execute("SELECT COUNT(*) FROM idempotency_keys").fetchone()[0]
    assert count == 0  # the inner block did not commit independently


def test_transaction_depth_is_reset_after_failure(migrated: Database) -> None:
    with pytest.raises(RuntimeError), migrated.transaction():
        raise RuntimeError
    with migrated.transaction() as connection:  # a fresh outermost transaction works
        connection.execute("INSERT INTO idempotency_keys VALUES ('k','o','h','{}','t')")
    assert migrated.connection().in_transaction is False


def test_foreign_keys_are_enforced(migrated: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError), migrated.transaction() as connection:
        connection.execute(
            "INSERT INTO clicks (link_id, clicked_at, browser, device, visitor_hash) "
            "VALUES (999, 't', 'Chrome', 'desktop', 'h')"
        )


# ----------------------------------------------------------------------------- migrations


def test_packaged_migrations_are_ordered_and_checksummed() -> None:
    migrations = load_migrations()
    assert [m.version for m in migrations] == [1, 2]
    assert [m.name for m in migrations] == ["init", "idempotency"]
    assert all(len(m.checksum) == 64 for m in migrations)


def test_migrate_applies_everything_once(database: Database) -> None:
    assert migrate(database) == [1, 2]
    assert migrate(database) == []
    rows = (
        database.connection().execute("SELECT version, checksum FROM schema_migrations").fetchall()
    )
    assert [(row["version"], row["checksum"]) for row in rows] == [
        (m.version, m.checksum) for m in load_migrations()
    ]
    tables = {
        row[0]
        for row in database.connection().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"links", "clicks", "idempotency_keys", "schema_migrations"} <= tables


def test_checksum_ignores_newline_style() -> None:
    assert checksum_of("a\r\nb\r\n") == checksum_of("a\nb\n")
    assert checksum_of("a\n") != checksum_of("b\n")


def test_tampered_applied_migration_is_detected(database: Database, tmp_path: Path) -> None:
    original = write_migrations(tmp_path / "m", {"001_one.sql": "CREATE TABLE t (id INTEGER);\n"})
    migrate(database, original)
    tampered = write_migrations(
        tmp_path / "m", {"001_one.sql": "CREATE TABLE t (id INTEGER, extra TEXT);\n"}
    )
    with pytest.raises(MigrationError, match="modified after it was applied"):
        migrate(database, tampered)


def test_tampered_checksum_row_blocks_the_real_migrations(migrated: Database) -> None:
    with migrated.transaction() as connection:
        connection.execute("UPDATE schema_migrations SET checksum = 'tampered' WHERE version = 1")
    with pytest.raises(MigrationError, match="checksum mismatch"):
        migrate(migrated)


def test_database_newer_than_code_is_refused(database: Database, tmp_path: Path) -> None:
    both = write_migrations(
        tmp_path / "m",
        {
            "001_one.sql": "CREATE TABLE a (id INTEGER);",
            "002_two.sql": "CREATE TABLE b (id INTEGER);",
        },
    )
    migrate(database, both)
    with pytest.raises(MigrationError, match="does not know"):
        migrate(database, both[:1])


def test_non_contiguous_applied_versions_are_refused(database: Database, tmp_path: Path) -> None:
    both = write_migrations(
        tmp_path / "m",
        {
            "001_one.sql": "CREATE TABLE a (id INTEGER);",
            "002_two.sql": "CREATE TABLE b (id INTEGER);",
        },
    )
    migrate(database, both)
    with database.transaction() as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version = 1")
    with pytest.raises(MigrationError, match="not contiguous"):
        migrate(database, both)


def test_failed_migration_leaves_no_partial_schema(database: Database, tmp_path: Path) -> None:
    broken = write_migrations(
        tmp_path / "m",
        {
            "001_broken.sql": "CREATE TABLE ok_table (id INTEGER);\nCREATE TABLE ok_table (id INTEGER);\n"
        },
    )
    with pytest.raises(sqlite3.OperationalError):
        migrate(database, broken)
    tables = {
        row[0]
        for row in database.connection().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "ok_table" not in tables
    assert (
        database.connection().execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 0
    )


def test_load_migrations_validates_names_and_numbering(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="unexpected migration file name"):
        write_migrations(tmp_path / "bad-name", {"init.sql": "SELECT 1;"})
    with pytest.raises(MigrationError, match="without gaps"):
        write_migrations(tmp_path / "gap", {"001_a.sql": "SELECT 1;", "003_c.sql": "SELECT 1;"})
    with pytest.raises(MigrationError, match="without gaps"):
        write_migrations(tmp_path / "start", {"002_b.sql": "SELECT 1;"})


def test_non_sql_files_and_directories_are_ignored(tmp_path: Path) -> None:
    directory = tmp_path / "m"
    migrations = write_migrations(directory, {"001_a.sql": "SELECT 1;", "README.txt": "notes"})
    (directory / "subdir").mkdir()
    assert [m.version for m in load_migrations(directory)] == [1]
    assert len(migrations) == 1


def test_split_statements_handles_comments_triggers_and_literals() -> None:
    script = (
        "-- leading comment\n"
        "CREATE TABLE a (id INTEGER, note TEXT DEFAULT 'x;y');\n"
        "CREATE TRIGGER t AFTER INSERT ON a BEGIN\n"
        "  UPDATE a SET note = 'z;' WHERE id = new.id;\n"
        "END;\n"
        "-- trailing comment\n"
    )
    statements = split_statements(script)
    assert len(statements) == 2
    assert statements[0].endswith("'x;y');")
    assert statements[1].startswith("CREATE TRIGGER") and statements[1].endswith("END;")


def test_split_statements_rejects_unterminated_statement() -> None:
    with pytest.raises(MigrationError, match="unterminated"):
        split_statements("CREATE TABLE a (id INTEGER)")


def test_concurrent_migrators_apply_each_migration_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "race.db"
    Database(path).open()
    errors: list[BaseException] = []
    applied: list[list[int]] = []
    barrier = threading.Barrier(4)

    def run() -> None:
        database = Database(path)
        try:
            barrier.wait()
            applied.append(migrate(database))
        except BaseException as error:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(error)
        finally:
            database.close()

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(version for batch in applied for version in batch) == [1, 2]


# ----------------------------------------------------------------------------- links


def new_link(code: str, owner: str = "alice", **extra: object) -> NewLink:
    values: dict[str, object] = {
        "code": code,
        "original_url": f"https://example.com/{code}",
        "created_at": T0,
        "expires_at": None,
        "created_by": owner,
    }
    values.update(extra)
    return NewLink(**values)  # type: ignore[arg-type]


def test_link_insert_and_lookup_roundtrip(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    expiry = T0 + timedelta(days=1)
    created = repository.insert(new_link("abc1234", expires_at=expiry))
    fetched = repository.get_by_code("abc1234")
    assert fetched == created
    assert fetched is not None
    assert (fetched.expires_at, fetched.disabled_at, fetched.click_count) == (expiry, None, 0)
    assert repository.get_by_code("missing") is None


def test_duplicate_code_raises_code_taken_and_keeps_the_original(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    repository.insert(new_link("dup", original_url="https://example.com/first"))
    with pytest.raises(CodeTakenError):
        repository.insert(new_link("dup", original_url="https://example.com/second"))
    stored = repository.get_by_code("dup")
    assert stored is not None
    assert stored.original_url == "https://example.com/first"


def test_codes_are_case_sensitive(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    repository.insert(new_link("Promo"))
    repository.insert(new_link("promo"))
    assert repository.get_by_code("Promo") != repository.get_by_code("promo")


def test_other_integrity_errors_are_not_masked(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    with pytest.raises(sqlite3.IntegrityError):
        repository.insert(new_link("abc", original_url=None))  # NOT NULL violation


def test_list_is_newest_first_paginated_and_owner_scoped(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    for index in range(5):
        repository.insert(new_link(f"alice-{index}"))
    repository.insert(new_link("bob-0", owner="bob"))
    first = repository.list_by_owner("alice", 2, 0)
    second = repository.list_by_owner("alice", 2, 2)
    assert [link.code for link in first] == ["alice-4", "alice-3"]
    assert [link.code for link in second] == ["alice-2", "alice-1"]
    assert repository.count_by_owner("alice") == 5
    assert repository.count_by_owner("bob") == 1
    assert repository.count_by_owner("nobody") == 0
    assert repository.list_by_owner("alice", 100, 5) == []


def test_list_clamps_hostile_limit_and_offset(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    for index in range(3):
        repository.insert(new_link(f"code-{index}"))
    assert len(repository.list_by_owner("alice", 10**12, -50)) == 3
    assert len(repository.list_by_owner("alice", -5, 0)) == 1  # clamped to at least one row


def test_owner_values_are_parameterised(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    repository.insert(new_link("safe"))
    assert repository.list_by_owner("alice' OR '1'='1", 10, 0) == []
    assert repository.get_by_code("x' OR '1'='1") is None
    assert repository.count_by_owner("alice' OR '1'='1") == 0


def test_disable_is_idempotent(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    repository.insert(new_link("abc"))
    assert repository.disable("abc", T0) is True
    assert repository.disable("abc", T0 + timedelta(hours=1)) is False
    link = repository.get_by_code("abc")
    assert link is not None
    assert link.disabled_at == T0  # the first disable timestamp is kept
    assert repository.disable("missing", T0) is False


def test_add_clicks_accumulates_per_link(migrated: Database) -> None:
    repository = SqliteLinkRepository(migrated)
    first = repository.insert(new_link("first"))
    second = repository.insert(new_link("second"))
    repository.add_clicks({first.id: 3, second.id: 1})
    repository.add_clicks({first.id: 2})
    repository.add_clicks({})
    assert repository.get_by_code("first").click_count == 5  # type: ignore[union-attr]
    assert repository.get_by_code("second").click_count == 1  # type: ignore[union-attr]


# ----------------------------------------------------------------------------- clicks


def click(
    link_id: int,
    when: datetime,
    *,
    visitor: str = "v1",
    referrer: str | None = None,
    browser: str = "Chrome",
    device: str = "desktop",
) -> ClickEvent:
    return ClickEvent(link_id, when, referrer, browser, device, visitor)


def test_click_aggregations(migrated: Database) -> None:
    links = SqliteLinkRepository(migrated)
    clicks = SqliteClickRepository(migrated)
    link = links.insert(new_link("abc"))
    other = links.insert(new_link("other"))
    clicks.insert_many(
        [
            click(link.id, T0, visitor="v1", referrer="https://a.example"),
            click(link.id, T0 + timedelta(hours=1), visitor="v1", referrer="https://a.example"),
            click(
                link.id,
                T0 + timedelta(days=1),
                visitor="v2",
                referrer="https://b.example",
                browser="Firefox",
                device="mobile",
            ),
            click(link.id, T0 + timedelta(days=1), visitor="v3", browser="Firefox"),
            click(other.id, T0, visitor="v9", referrer="https://z.example"),
        ]
    )
    since = T0 - timedelta(days=1)
    summary = clicks.summary(link.id, since)
    assert (summary.total_clicks, summary.unique_visitors) == (4, 3)
    assert summary.first_click_at == T0
    assert summary.last_click_at == T0 + timedelta(days=1)
    assert clicks.daily_counts(link.id, since) == {date(2026, 1, 15): 2, date(2026, 1, 16): 2}
    referrers = clicks.top_referrers(link.id, since, 10)
    assert [(r.name, r.clicks) for r in referrers] == [
        ("https://a.example", 2),
        ("https://b.example", 1),
    ]
    browsers = clicks.breakdown("browser", link.id, since, 10)
    assert [(b.name, b.clicks) for b in browsers] == [("Chrome", 2), ("Firefox", 2)]
    devices = clicks.breakdown("device", link.id, since, 10)
    assert [(d.name, d.clicks) for d in devices] == [("desktop", 3), ("mobile", 1)]
    assert len(clicks.top_referrers(link.id, since, 1)) == 1


def test_click_window_excludes_older_events(migrated: Database) -> None:
    links = SqliteLinkRepository(migrated)
    clicks = SqliteClickRepository(migrated)
    link = links.insert(new_link("abc"))
    clicks.insert_many([click(link.id, T0 - timedelta(days=10)), click(link.id, T0)])
    summary = clicks.summary(link.id, T0 - timedelta(days=1))
    assert summary.total_clicks == 1


def test_empty_click_summary(migrated: Database) -> None:
    link = SqliteLinkRepository(migrated).insert(new_link("abc"))
    summary = SqliteClickRepository(migrated).summary(link.id, T0)
    assert (summary.total_clicks, summary.unique_visitors) == (0, 0)
    assert summary.first_click_at is None and summary.last_click_at is None


def test_unknown_breakdown_dimension_is_impossible(migrated: Database) -> None:
    with pytest.raises(KeyError):
        SqliteClickRepository(migrated).breakdown("visitor_hash", 1, T0, 5)  # type: ignore[arg-type]


def test_retention_purge_deletes_only_old_clicks(migrated: Database) -> None:
    links = SqliteLinkRepository(migrated)
    clicks = SqliteClickRepository(migrated)
    link = links.insert(new_link("abc"))
    clicks.insert_many([click(link.id, T0 - timedelta(days=100)), click(link.id, T0)])
    assert clicks.delete_before(T0 - timedelta(days=90)) == 1
    assert clicks.summary(link.id, T0 - timedelta(days=365)).total_clicks == 1
    clicks.insert_many([])  # no-op


# ----------------------------------------------------------------------------- idempotency


def record(key: str = "k", owner: str = "alice", **extra: object) -> IdempotencyRecord:
    values: dict[str, object] = {
        "key": key,
        "owner": owner,
        "request_hash": "h1",
        "response_json": '{"a":1}',
        "created_at": T0,
    }
    values.update(extra)
    return IdempotencyRecord(**values)  # type: ignore[arg-type]


def test_idempotency_repository_roundtrip_and_scoping(migrated: Database) -> None:
    repository = SqliteIdempotencyRepository(migrated)
    repository.insert(record())
    assert repository.get("k", "alice") == record()
    assert repository.get("k", "bob") is None  # keys are scoped per owner
    repository.insert(record(owner="bob"))  # same key, different owner: fine
    with pytest.raises(IdempotencyKeyExistsError):
        repository.insert(record())


def test_idempotency_delete_and_purge(migrated: Database) -> None:
    repository = SqliteIdempotencyRepository(migrated)
    repository.insert(record("old", created_at=T0 - timedelta(days=2)))
    repository.insert(record("new"))
    assert repository.delete_created_before(T0 - timedelta(days=1)) == 1
    assert repository.get("old", "alice") is None
    repository.delete("new", "alice")
    assert repository.get("new", "alice") is None
