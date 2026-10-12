"""Deterministic JSON helpers shared by every persistent structure.

* ``to_jsonable`` turns dataclasses / enums / datetimes / paths into plain JSON
  types (recursively). **Enums are checked before primitives** so ``IntEnum``
  members (``Severity``, ``RiskLevel``) serialise to their *name*, identically
  on every code path (``hash_jsonable(x) == hash_jsonable(x.to_dict())``).
* ``canonical_json`` produces a byte-stable, **pure-ASCII** encoding (sorted
  keys, no whitespace, ``ensure_ascii=True`` so U+2028/U+2029/U+0085 and lone
  surrogates can never split or corrupt a line-oriented log, ``allow_nan=False``).
* ``sha256_hex`` / ``hash_jsonable`` are the only hashing entry points for
  structured data. (File *content* hashing lives in ``fsutil.sha256_text``.)
* ``loads_strict`` is the only way untrusted JSON (state.json, index.json, audit
  lines, LLM output) is parsed: size/depth caps, no NaN/Infinity, no duplicate
  keys.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import PurePath
from typing import Any

MAX_JSON_BYTES_DEFAULT = 16 * 1024 * 1024
MAX_JSON_DEPTH_DEFAULT = 64


def _enum_value(obj: enum.Enum) -> Any:
    return obj.name if isinstance(obj, enum.IntEnum) else obj.value


def to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` into JSON-compatible primitives."""
    if isinstance(obj, enum.Enum):  # must precede the primitive check (IntEnum is an int)
        return _enum_value(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, bytes):
        return {"__type__": "bytes", "sha256": hashlib.sha256(obj).hexdigest(), "len": len(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            return to_jsonable(to_dict())
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_jsonable(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    raise TypeError(f"Cannot serialise object of type {type(obj).__name__}")


def _key(k: Any) -> str:
    if isinstance(k, enum.Enum):
        return str(_enum_value(k))
    return str(k)


def canonical_json(obj: Any) -> bytes:
    """Byte-stable, ASCII-only JSON encoding used for hashing and audit lines."""
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        try:
            data = data.encode("utf-8")
        except UnicodeEncodeError as exc:  # lone surrogate etc.
            raise ValueError("string is not valid UTF-8 (lone surrogate?)") from exc
    return hashlib.sha256(data).hexdigest()


def hash_jsonable(obj: Any) -> str:
    """SHA-256 of the canonical JSON encoding of ``obj``."""
    return sha256_hex(canonical_json(obj))


def parse_datetime(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp; **naive datetimes are rejected** (all times are UTC-aware)."""
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime not allowed: {value!r}")
    return dt


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant not allowed: {name}")


def _finite_float(text: str) -> float:
    """``1e999`` parses to ``inf`` without going through ``parse_constant``; reject it here."""
    value = float(text)
    if value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite JSON number not allowed: {text}")
    return value


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _depth(obj: Any, limit: int, level: int = 0) -> None:
    if level > limit:
        raise ValueError("JSON nesting too deep")
    if isinstance(obj, dict):
        for v in obj.values():
            _depth(v, limit, level + 1)
    elif isinstance(obj, list):
        for v in obj:
            _depth(v, limit, level + 1)


def loads_strict(
    data: bytes | str,
    *,
    max_bytes: int = MAX_JSON_BYTES_DEFAULT,
    max_depth: int = MAX_JSON_DEPTH_DEFAULT,
) -> Any:
    """Parse untrusted JSON: size-capped, depth-capped, no NaN/Infinity, no duplicate keys.

    Raises ``ValueError`` (never a crash type that could escape the scheduler).
    """
    raw = data.encode("utf-8", "strict") if isinstance(data, str) else data
    if len(raw) > max_bytes:
        raise ValueError(f"JSON document too large ({len(raw)} > {max_bytes} bytes)")
    try:
        text = raw.decode("utf-8")
        obj = json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            object_pairs_hook=_no_duplicates,
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting too deep") from exc
    _depth(obj, max_depth)
    return obj
