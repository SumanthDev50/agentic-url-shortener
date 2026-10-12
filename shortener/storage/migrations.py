"""Versioned, forward-only SQL migrations with checksum verification (NFR-09).

* Files live in ``storage/migrations/NNN_name.sql`` and must be numbered ``001``, ``002``, ...
  without gaps.
* Each migration is applied in its own transaction together with its ``schema_migrations`` row,
  so a failure leaves no partial schema behind.
* The SHA-256 of every applied file is recorded.  If an applied file's content later differs,
  or the database contains a migration this build does not know, :func:`migrate` raises
  :class:`MigrationError` and the service refuses to start (immutability enforced at runtime,
  complementing the orchestrator's CHG-001 change-control rule).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable

from shortener.storage.db import Database

_FILE_PATTERN = re.compile(r"(?P<version>\d{3})_(?P<name>[a-z0-9_]+)\.sql")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    checksum   TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


class MigrationError(RuntimeError):
    """Migrations are inconsistent with the database; startup must be aborted."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str


def checksum_of(sql: str) -> str:
    """SHA-256 of the script with newlines normalised (stable across Windows/POSIX checkouts)."""
    return hashlib.sha256(sql.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def load_migrations(directory: Traversable | None = None) -> list[Migration]:
    """Read and validate the packaged (or given) migration directory."""
    source = directory if directory is not None else files("shortener.storage") / "migrations"
    migrations: list[Migration] = []
    for entry in source.iterdir():
        if not entry.is_file() or not entry.name.endswith(".sql"):
            continue
        match = _FILE_PATTERN.fullmatch(entry.name)
        if match is None:
            raise MigrationError(f"unexpected migration file name: {entry.name}")
        sql = entry.read_text(encoding="utf-8")
        migrations.append(
            Migration(int(match["version"]), match["name"], sql, checksum_of(sql)),
        )
    migrations.sort(key=lambda migration: migration.version)
    if [m.version for m in migrations] != list(range(1, len(migrations) + 1)):
        raise MigrationError("migration files must be numbered 001, 002, ... without gaps")
    return migrations


def split_statements(script: str) -> list[str]:
    """Split a script into single statements (aware of triggers and string literals)."""
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""
    leftover = "\n".join(
        line for line in buffer.splitlines() if line.strip() and not line.strip().startswith("--")
    )
    if leftover:
        raise MigrationError("migration ends with an unterminated statement")
    return statements


def migrate(database: Database, migrations: Sequence[Migration] | None = None) -> list[int]:
    """Apply pending migrations; return the versions applied by this call."""
    available = tuple(migrations) if migrations is not None else tuple(load_migrations())
    with database.transaction() as connection:
        connection.execute(_CREATE_TABLE)
        _verify_applied(_read_applied(connection), available)
    applied_now: list[int] = []
    for migration in available:
        if _apply_if_pending(database, migration):
            applied_now.append(migration.version)
    return applied_now


def _read_applied(connection: sqlite3.Connection) -> dict[int, str]:
    rows = connection.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return {int(row["version"]): str(row["checksum"]) for row in rows}


def _verify_applied(applied: Mapping[int, str], available: Sequence[Migration]) -> None:
    _verify_checksums(applied, available)
    if sorted(applied) != list(range(1, len(applied) + 1)):
        raise MigrationError("applied migrations are not contiguous; refusing to continue")


def _verify_checksums(applied: Mapping[int, str], available: Sequence[Migration]) -> None:
    known = {migration.version: migration for migration in available}
    for version, checksum in sorted(applied.items()):
        migration = known.get(version)
        if migration is None:
            raise MigrationError(
                f"database contains migration {version:03d} that this build does not know; "
                "downgrades are not supported"
            )
        if migration.checksum != checksum:
            raise MigrationError(
                f"migration {version:03d}_{migration.name} was modified after it was applied "
                "(checksum mismatch); applied migrations are immutable"
            )


def _apply_if_pending(database: Database, migration: Migration) -> bool:
    with database.transaction() as connection:
        stored = connection.execute(
            "SELECT checksum FROM schema_migrations WHERE version = ?", (migration.version,)
        ).fetchone()
        if stored is not None:
            # Another instance applied it between our check and now; it must be identical.
            _verify_checksums({migration.version: str(stored["checksum"])}, (migration,))
            return False
        for statement in split_statements(migration.sql):
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_migrations (version, checksum, applied_at) "
            "VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (migration.version, migration.checksum),
        )
    return True
