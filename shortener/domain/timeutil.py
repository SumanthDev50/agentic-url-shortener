"""UTC timestamp (de)serialisation shared by storage and API layers.

The storage format is fixed-width (``2026-01-31T12:00:00.000000Z``) so lexicographic comparison
in SQL equals chronological comparison.
"""

from __future__ import annotations

from datetime import UTC, datetime

STORAGE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def format_utc(moment: datetime) -> str:
    """Serialise an aware datetime to the fixed-width UTC storage format."""
    if moment.tzinfo is None:
        raise ValueError("naive datetimes are not allowed")
    return moment.astimezone(UTC).strftime(STORAGE_FORMAT)


def parse_utc(text: str) -> datetime:
    """Parse a value produced by :func:`format_utc` back into an aware UTC datetime."""
    return datetime.strptime(text, STORAGE_FORMAT).replace(tzinfo=UTC)
