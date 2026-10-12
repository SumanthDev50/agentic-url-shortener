"""Tamper-evident audit log: secret redaction, hash chain, verification, trace export.

Why a hash chain
----------------
Every event is one line of canonical (sorted-key, pure-ASCII) JSON whose ``hash`` is the SHA-256 of
the canonical JSON of the *raw parsed dict minus ``hash``* and whose ``prev_hash`` is the previous
event's ``hash``.  Editing, deleting, inserting or reordering any line breaks the chain; so does an
unknown extra key, because the hash covers whatever keys are on the line.  Pure ASCII output means
U+2028/U+2029/U+0085 and lone surrogates (which ``str.splitlines`` would treat as line breaks) can
never split a record, and readers split on ``\\n`` only.

What the chain cannot do: detect a *clean* truncation at a line boundary or a full rewrite by a
privileged same-user attacker.  That is what anchors are for (``RunState.audit_head``, the runs-dir
anchor file, ``verify_file(path, anchor=...)``); a remote WORM / transparency log would be the real
fix (documented limitation, engine-semantics §17.2).

Redaction happens *before* hashing, so a secret never reaches the disk, the hash input or any
listener.  Every redaction pattern is linear-time: no nested quantifiers, bounded repetition, and
token patterns start only at token boundaries, so hostile input cannot trigger catastrophic
backtracking (fuzz-tested under a time bound).
"""

from __future__ import annotations

import enum
import hashlib
import logging
import math
import os
import queue
import re
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from orchestrator.core.errors import ConfigError, IntegrityError, StateError
from orchestrator.core.fsutil import append_line, is_link_or_reparse, read_bytes_retry
from orchestrator.core.paths import NODE_ID_RE, RUN_ID_RE
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.serde import canonical_json, hash_jsonable, loads_strict, to_jsonable
from orchestrator.core.types import Actor, AuditEvent, Clock, EventType

__all__ = [
    "AUDIT_FILENAME",
    "GENESIS_HASH",
    "GOVERNANCE_EVENTS",
    "Anchor",
    "AuditIntegrityError",
    "AuditLog",
    "Redactor",
    "VerifyResult",
    "export_trace",
    "read_run_events",
    "run_status_of",
    "segment_paths",
    "verify_run",
]

GENESIS_HASH = "0" * 64
AUDIT_FILENAME = "audit.jsonl"
REDACTED = "[REDACTED]"

MAX_STRING_BYTES = 8 * 1024  # per-string cap after redaction
MAX_SCAN_CHARS = 256 * 1024  # a longer string is clipped *before* scanning (bounded work)
MAX_EVENT_BYTES = 256 * 1024  # canonical size of one event's ``data``
_MAX_DEPTH = 32
_MAX_ITEMS = 1000
_MAX_NODES = 20_000
_MAX_PATTERN_CHARS = 512
_ACTOR_ID_CHARS = 128

_LOG = logging.getLogger(__name__)

GOVERNANCE_EVENTS: frozenset[EventType] = frozenset(
    {
        member
        for member in EventType
        if member.name.startswith(("APPROVAL_", "SAFE_STOP_", "KILL_", "ROLLBACK_", "EFFECT_"))
    }
    | {
        EventType.RUN_CREATED,
        EventType.RUN_SUCCEEDED,
        EventType.RUN_FAILED,
        EventType.RUN_SAFE_STOPPED,
        EventType.RUN_SEALED,
        EventType.NODE_PUBLISHING,
        EventType.AUDIT_TAIL_REPAIRED,
    }
)
"""Events that are fsynced before ``emit`` returns: they authorise or record a mutation that must
survive a crash (approvals, safe-stop, kill switch, publish journal, effects, terminal states)."""

# =============================================================================
# Redaction
# =============================================================================

_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "apikey",
    "authorization",
    "auth",
    "credential",
    "cookie",
    "session",
    "privatekey",
    "signature",
    "connectionstring",
    "dsn",
)

# Guarded patterns may only *start* where the previous character is not part of a token.  Two
# reasons: (1) an unbounded greedy tail (JWT) would otherwise rescan the rest of the string from
# every start position on input such as ``"eyJ" * 20000``; (2) short or common prefixes (``sk-``,
# ``sk_live_``) would otherwise fire inside ordinary words ("ri" + "sk-assessment-for-...").
# Distinctive prefixes (``AKIA``, ``ghp_``...) are *not* guarded so that glueing a letter in front
# of a secret cannot bypass them; their repetition is bounded, so a scan stays linear anyway.
_TB = r"(?<![A-Za-z0-9_])"
_KEY_WORDS = (
    r"(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|authorization|credentials?|"
    r"private[_-]?key|connection[_-]?string|access[_-]?key|signature|cookie|dsn)"
)

_Replacement = str | Callable[[re.Match[str]], str]


def _assignment(match: re.Match[str]) -> str:
    quote = match.group("dq") or match.group("sq") or ""
    return f"{match.group('head')}{quote}{REDACTED}{quote}"


def _azure(match: re.Match[str]) -> str:
    return f"{match.group('head') or match.group('head2')}{REDACTED}"


@dataclass(frozen=True)
class _Rule:
    """One value pattern plus the literals that must occur (lower-cased) for it to be worth running.

    The ``in`` checks run at memchr speed, so ordinary strings (ids, hashes, short messages) skip
    almost every regex; hostile strings that do contain a trigger still meet a linear regex.
    """

    pattern: re.Pattern[str]
    replacement: _Replacement
    triggers: tuple[str, ...]

    def apply(self, text: str, lowered: str) -> str:
        if any(trigger in lowered for trigger in self.triggers):
            return self.pattern.sub(self.replacement, text)
        return text


def _token(pattern: str, *triggers: str, guarded: bool = False) -> _Rule:
    return _Rule(re.compile((_TB if guarded else "") + pattern), REDACTED, triggers)


_RULES: tuple[_Rule, ...] = (
    _Rule(re.compile(r"(?<=://)[^\s/@?#]{1,256}(?=@)"), REDACTED, ("://",)),  # URL userinfo
    _token(
        r"(?:A3T[A-Z0-9]|AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|AIPA)[0-9A-Z]{16}",  # AWS key ids
        "a3t",
        "akia",
        "asia",
        "agpa",
        "aida",
        "aroa",
        "anpa",
        "anva",
        "aipa",
    ),
    _token(r"gh[pousr]_[A-Za-z0-9]{30,255}", "ghp_", "gho_", "ghu_", "ghs_", "ghr_"),  # GitHub
    _token(r"github_pat_[A-Za-z0-9_]{22,255}", "github_pat_"),
    _token(r"xox[abposr]-[A-Za-z0-9-]{10,255}", "xox"),  # Slack
    _token(r"xapp-[A-Za-z0-9-]{10,255}", "xapp-"),
    _Rule(
        re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/]{20,255}"),
        REDACTED,
        ("hooks.slack.com",),
    ),
    _token(  # JWT (the signature segment may be empty for unsecured tokens)
        r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]*)?", "eyj", guarded=True
    ),
    _token(r"AIza[0-9A-Za-z_-]{35}", "aiza"),  # Google API keys
    _token(
        r"(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,255}", "_live_", "_test_", guarded=True
    ),  # Stripe
    _token(r"whsec_[A-Za-z0-9]{16,255}", "whsec_"),
    _token(r"sk-(?:ant-)?[A-Za-z0-9_-]{20,255}", "sk-", guarded=True),  # OpenAI / Anthropic
    _Rule(
        re.compile(r"(?i)(?<![A-Za-z0-9_])bearer\s{1,16}[A-Za-z0-9._~+/=-]{8,}"),
        f"Bearer {REDACTED}",
        ("bearer",),
    ),
    _Rule(
        re.compile(
            rf"(?i)(?P<head>{_KEY_WORDS}[\"']?\s{{0,4}}[=:]\s{{0,4}})"
            r"(?:(?P<dq>\")[^\"\n]{1,1024}\"?|(?P<sq>')[^'\n]{1,1024}'?|[^\s\"',;&]{1,1024})"
        ),
        _assignment,
        (
            *("password", "passwd", "pwd", "secret", "token", "key", "authorization"),
            *("credential", "signature", "cookie", "dsn", "connection"),
        ),
    ),
    _Rule(  # Azure storage connection strings and SAS signatures
        re.compile(
            r"(?i)(?P<head>(?:AccountKey|SharedAccessKey|SharedAccessSignature)=)[^;\s\"']{8,}"
            r"|(?P<head2>(?<![A-Za-z0-9])sig=)[A-Za-z0-9%+/=_-]{16,}"
        ),
        _azure,
        ("accountkey=", "sharedaccess", "sig="),
    ),
)
_PEM_HEADER = re.compile(r"-----BEGIN [A-Z0-9 ]{1,40}-----")
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)[+*{]")


def _redact_pem(text: str) -> str:
    """Replace PEM blocks; an unterminated block is redacted to the end of the text.

    Done imperatively with ``str.find`` (every search resumes after the previous block) so
    unbalanced or repeated ``-----BEGIN`` markers cost O(n) in total.
    """
    out: list[str] = []
    pos = 0
    while True:
        start = text.find("-----BEGIN ", pos)
        if start < 0:
            break
        header = _PEM_HEADER.match(text, start)
        if header is None:
            out.append(text[pos : start + 11])
            pos = start + 11
            continue
        end_marker = text.find("-----END ", header.end())
        close = text.find("-----", end_marker + 9) if end_marker >= 0 else -1
        out.append(text[pos:start])
        out.append(f"{REDACTED} (PEM block)")
        if close < 0:
            pos = len(text)
            break
        pos = close + 5
    out.append(text[pos:])
    return "".join(out)


def _compile_extra(pattern: str | re.Pattern[str]) -> re.Pattern[str]:
    source = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    if not isinstance(source, str):
        raise ConfigError("extra redaction patterns must be text patterns")
    if len(source) > _MAX_PATTERN_CHARS:
        raise ConfigError(f"redaction pattern longer than {_MAX_PATTERN_CHARS} characters")
    if _NESTED_QUANTIFIER.search(source):
        raise ConfigError("redaction pattern with a nested quantifier is not allowed (ReDoS risk)")
    try:
        compiled = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern)
    except re.error as exc:
        raise ConfigError(f"invalid redaction pattern: {exc}") from exc
    if compiled.search("") is not None:
        raise ConfigError("redaction pattern must not match the empty string")
    return compiled


def _is_sensitive_key(key: object) -> bool:
    normalised = re.sub(r"[-_ ]", "", str(key)[:256].lower())
    return any(part in normalised for part in _SENSITIVE_KEY_PARTS)


class Redactor:
    """Removes secrets from text and nested structures before they are logged.

    ``redact_text`` applies the value patterns and caps the result at 8 KiB (marker
    ``...[truncated n bytes sha256=...]``; the digest is of the *redacted* text).  ``redact`` walks
    mappings/sequences: a value under a sensitive key (normalised: lower-cased, ``-_ `` removed,
    substring match) is replaced wholesale.  Strings and containers are replaced; ``None``,
    booleans and numbers are kept because they carry no credential material in this system (counts,
    ids, flags) and masking them would destroy metrics such as ``tokens``.  Depth, item and node
    counts are bounded so a hostile payload cannot blow up memory or time.
    """

    def __init__(self, extra_patterns: Iterable[str | re.Pattern[str]] = ()) -> None:
        self._extra = tuple(_compile_extra(p) for p in extra_patterns)

    # ------------------------------------------------------------------ text

    def redact_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        clipped = 0
        if len(text) > MAX_SCAN_CHARS:
            head = text[:MAX_SCAN_CHARS]
            # drop the (possibly partial) last token so a cut cannot leave a secret prefix behind
            cut = max(head.rfind(" "), head.rfind("\n"), head.rfind("\t"))
            if cut > 0:
                head = head[:cut]
            clipped = len(text) - len(head)
            text = head
        text = self._apply_patterns(text)
        return self._cap(text, clipped)

    def _apply_patterns(self, text: str) -> str:
        text = _redact_pem(text)
        lowered = text.lower()
        for rule in _RULES:  # triggers are judged on the pre-redaction text: only a superset
            text = rule.apply(text, lowered)
        for pattern in self._extra:
            text = pattern.sub(REDACTED, text)
        return text

    @staticmethod
    def _cap(text: str, clipped_chars: int) -> str:
        # UTF-8 needs at most 4 bytes per character, so short strings need no encoding at all.
        if clipped_chars == 0 and len(text) * 4 <= MAX_STRING_BYTES:
            return text
        raw = text.encode("utf-8", "surrogatepass")
        if clipped_chars == 0 and len(raw) <= MAX_STRING_BYTES:
            return text
        digest = hashlib.sha256(raw).hexdigest()[:16]
        dropped = max(0, len(raw) - MAX_STRING_BYTES) + clipped_chars
        kept = raw[:MAX_STRING_BYTES].decode("utf-8", "ignore")
        return f"{kept}...[truncated {dropped} bytes sha256={digest}]"

    # ------------------------------------------------------------------ structures

    def redact(self, obj: Any) -> Any:
        """Deep-redact ``obj`` (any JSON-like structure) and return plain JSON types."""
        return _Walk(self).value(obj, 0, False)


class _Walk:
    """One bounded traversal (the node budget is per call, shared by the whole structure)."""

    def __init__(self, redactor: Redactor) -> None:
        self._redactor = redactor
        self._budget = _MAX_NODES

    def value(self, v: Any, depth: int, sensitive: bool) -> Any:
        self._budget -= 1
        if self._budget < 0:
            return "[truncated: size limit]"
        if isinstance(v, enum.Enum):
            return self._other(v, depth) if not sensitive else REDACTED
        if v is None or isinstance(v, bool):
            return v
        if isinstance(v, float):
            return v if math.isfinite(v) else str(v)
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            if sensitive and v:
                return REDACTED
            return self._redactor.redact_text(v)
        if sensitive:
            return REDACTED
        if depth >= _MAX_DEPTH:
            return "[truncated: depth limit]"
        if isinstance(v, Mapping):
            return self._mapping(v, depth)
        if isinstance(v, (list, tuple)):
            return self._sequence(v, depth)
        if isinstance(v, (set, frozenset)):
            return self._sequence(sorted(v, key=repr), depth)
        return self._other(v, depth)

    def _mapping(self, mapping: Mapping[Any, Any], depth: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(mapping.items()):
            if index >= _MAX_ITEMS:
                out["_truncated"] = f"{len(mapping) - _MAX_ITEMS} more items"
                break
            name = self._redactor.redact_text(str(key))
            out[name] = self.value(item, depth + 1, _is_sensitive_key(key))
        return out

    def _sequence(self, items: Sequence[Any], depth: int) -> list[Any]:
        out = [self.value(item, depth + 1, False) for item in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            out.append(f"[truncated: {len(items) - _MAX_ITEMS} more items]")
        return out

    def _other(self, v: Any, depth: int) -> Any:
        """Enums, datetimes, bytes, dataclasses: convert with ``to_jsonable``, walk the result."""
        try:
            converted = to_jsonable(v)
        except (TypeError, ValueError, RecursionError):
            return f"<unserialisable {type(v).__name__}>"
        return self.value(converted, depth + 1, False)


# =============================================================================
# Chain verification
# =============================================================================

Anchor = str | tuple[int, str] | Mapping[str, Any]
"""A trusted record of the log head: a head hash, ``(seq, hash)`` or ``{"seq":..,"hash":..}``."""

_EVENT_KEYS = frozenset(
    {
        "seq",
        "ts",
        "run_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "type",
        "node_id",
        "attempt",
        "actor",
        "data",
        "prev_hash",
        "hash",
    }
)
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_TRACE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SEGMENT_RE = re.compile(r"audit\.(\d+)\.jsonl")
_SEGMENT_REASON = "segment_started"


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a chain verification.

    ``broken_at`` is the sequence number that failed (``None`` when the chain itself is intact but
    an anchor is not satisfied).  The trailing fields describe the last *good* event and whether the
    only defect is a repairable torn tail.
    """

    ok: bool
    count: int
    broken_at: int | None = None
    reason: str = ""
    last_good_seq: int = 0
    last_good_hash: str = GENESIS_HASH
    torn_tail: bool = False


class AuditIntegrityError(IntegrityError):
    """Audit log failed verification; carries what ``AuditLog.open_new_segment`` needs."""

    def __init__(self, message: str, result: VerifyResult) -> None:
        super().__init__(message)
        self.result = result
        self.broken_at = result.broken_at
        self.last_good_seq = result.last_good_seq
        self.last_good_hash = result.last_good_hash


class _Unparseable(Exception):
    """The line is not a JSON object (the only defect a crash can leave at the tail)."""


class _Invalid(Exception):
    """The line parses but violates the chain (tamper, never a torn write)."""


@dataclass(frozen=True)
class _Failure:
    broken_at: int
    reason: str
    tail: Literal["torn", "unterminated"] | None  # None: damage before the tail


@dataclass(frozen=True)
class _Scan:
    events: list[AuditEvent]
    valid_bytes: int  # prefix length covering exactly the verified, newline-terminated lines
    failure: _Failure | None


def _parse_line(raw: bytes) -> dict[str, Any]:
    try:
        parsed = loads_strict(raw)
    except ValueError as exc:
        raise _Unparseable(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise _Unparseable("line is not a JSON object")
    return parsed


def _check_event(
    raw: bytes,
    fields: dict[str, Any],
    *,
    expected_seq: int | None,
    prev_hash: str | None,
    run_id: str | None,
) -> AuditEvent:
    """Verify one parsed line against the chain state; ``expected_seq is None`` = first line."""
    if fields.keys() != _EVENT_KEYS:
        raise _Invalid("event has missing or unexpected keys")
    stored = fields["hash"]
    if not isinstance(stored, str) or not _HASH_RE.fullmatch(stored):
        raise _Invalid("malformed hash")
    body = {k: v for k, v in fields.items() if k != "hash"}
    try:
        computed = hash_jsonable(body)
        canonical = canonical_json(fields)
    except (TypeError, ValueError, RecursionError) as exc:
        raise _Invalid("event is not canonicalisable") from exc
    if computed != stored:
        raise _Invalid("hash mismatch (event content was altered)")
    if canonical != raw:
        raise _Invalid("line is not canonical JSON")
    seq = fields["seq"]
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        raise _Invalid("malformed sequence number")
    if expected_seq is not None and seq != expected_seq:
        raise _Invalid(f"sequence break: expected {expected_seq}, found {seq}")
    if prev_hash is not None and fields["prev_hash"] != prev_hash:
        raise _Invalid("prev_hash does not link to the previous event (reorder / insert / delete)")
    if run_id is not None and fields["run_id"] != run_id:
        raise _Invalid("event belongs to a different run")
    try:
        event = AuditEvent.from_dict(fields)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise _Invalid(f"malformed event: {sanitize_for_terminal(exc, 120)}") from exc
    if expected_seq is None:
        _check_first(event)
    return event


def _check_first(event: AuditEvent) -> None:
    """A log starts at the genesis, or with a segment-start event that restates its own anchor."""
    if event.seq == 1 and event.prev_hash == GENESIS_HASH:
        return
    data = event.data
    if (
        _is_segment_start(event)
        and data.get("broken_at") == event.seq
        and data.get("last_good_hash") == event.prev_hash
        and _HASH_RE.fullmatch(event.prev_hash) is not None
    ):
        return
    raise _Invalid("first event is neither the genesis nor a segment start")


def _is_segment_start(event: AuditEvent) -> bool:
    return (
        event.type is EventType.AUDIT_TAIL_REPAIRED and event.data.get("reason") == _SEGMENT_REASON
    )


def _scan(data: bytes, *, run_id: str | None) -> _Scan:
    """Verify every line of ``data``; stop at the first defect and classify it."""
    lines = data.split(b"\n")
    terminated = not data or data.endswith(b"\n")
    if terminated:
        lines.pop()  # the empty piece after the final newline
    events: list[AuditEvent] = []
    valid_bytes = 0
    expected_seq: int | None = None
    prev_hash: str | None = None
    pinned_run = run_id
    for index, raw in enumerate(lines):
        is_last = index == len(lines) - 1
        position = expected_seq if expected_seq is not None else (events[0].seq if events else 1)
        try:
            event = _check_event(
                raw,
                _parse_line(raw),
                expected_seq=expected_seq,
                prev_hash=prev_hash,
                run_id=pinned_run,
            )
        except _Unparseable as exc:
            reason = f"line {index + 1} is not valid JSON ({sanitize_for_terminal(exc, 120)})"
            return _Scan(
                events, valid_bytes, _Failure(position, reason, "torn" if is_last else None)
            )
        except _Invalid as exc:
            return _Scan(events, valid_bytes, _Failure(position, f"event {position}: {exc}", None))
        events.append(event)
        expected_seq, prev_hash, pinned_run = event.seq + 1, event.hash, event.run_id
        if is_last and not terminated:
            reason = "final line is not newline-terminated"
            return _Scan(
                events, valid_bytes + len(raw), _Failure(event.seq + 1, reason, "unterminated")
            )
        valid_bytes += len(raw) + 1
    return _Scan(events, valid_bytes, None)


def _anchor_pair(anchor: Anchor) -> tuple[int | None, str] | None:
    """Normalise an anchor to ``(seq or None, hash)``; ``None`` means "nothing to check"."""
    seq: int | None = None
    if isinstance(anchor, str):
        digest = anchor.strip().lower()
    elif isinstance(anchor, tuple) and len(anchor) == 2:
        seq, digest = anchor
    elif isinstance(anchor, Mapping):
        seq, digest = anchor.get("seq"), anchor.get("hash", "")
    else:
        raise ValueError("unsupported anchor type")
    if seq == 0 and digest in ("", GENESIS_HASH):
        return None  # the anchor of an empty log
    if not isinstance(digest, str) or not _HASH_RE.fullmatch(digest):
        raise ValueError("anchor hash must be 64 lowercase hex characters")
    if seq is not None and (isinstance(seq, bool) or not isinstance(seq, int) or seq < 1):
        raise ValueError("anchor seq must be a positive integer")
    return seq, digest


def _result_for(scan: _Scan, anchor: Anchor | None) -> VerifyResult:
    last = scan.events[-1] if scan.events else None
    count = len(scan.events)
    good_seq = last.seq if last else 0
    good_hash = last.hash if last else GENESIS_HASH

    def result(
        ok: bool, reason: str = "", broken_at: int | None = None, torn: bool = False
    ) -> VerifyResult:
        return VerifyResult(ok, count, broken_at, reason, good_seq, good_hash, torn)

    failure = scan.failure
    if failure is not None:
        return result(False, failure.reason, failure.broken_at, failure.tail is not None)
    if anchor is None:
        return result(True)
    try:
        pair = _anchor_pair(anchor)
    except ValueError as exc:
        return result(False, f"unusable anchor: {exc}")
    if pair is not None:
        seq, digest = pair
        if not any(e.hash == digest and (seq is None or e.seq == seq) for e in scan.events):
            return result(False, "anchor not found in the chain (truncated, replaced or other run)")
    return result(True)


def _spans(node_id: str | None, attempt: int | None) -> tuple[str, str | None]:
    """``(span_id, parent_span_id)`` from ``(node_id, attempt)`` (engine-semantics §17.6)."""
    if node_id is None:
        return "run", None
    if attempt is None:
        return f"n:{node_id}", "run"
    return f"n:{node_id}#{attempt}", f"n:{node_id}"


def _check_target(node_id: str | None, attempt: int | None) -> None:
    if node_id is not None and (not isinstance(node_id, str) or not NODE_ID_RE.fullmatch(node_id)):
        raise ValueError("invalid audit node_id")
    if attempt is not None:
        if node_id is None:
            raise ValueError("an audit attempt requires a node_id")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise ValueError("invalid audit attempt number")


# =============================================================================
# Segments
# =============================================================================


def segment_paths(run_dir: Path | str) -> list[Path]:
    """Existing audit files of a run in chain order: ``audit.jsonl``, ``audit.1.jsonl``, ..."""
    directory = Path(run_dir)
    numbered: list[tuple[int, Path]] = []
    if directory.is_dir():
        for entry in directory.iterdir():
            match = _SEGMENT_RE.fullmatch(entry.name)
            if match and _is_plain_file(entry):
                numbered.append((int(match.group(1)), entry))
    first = directory / AUDIT_FILENAME
    return ([first] if _is_plain_file(first) else []) + [path for _, path in sorted(numbered)]


def _is_plain_file(path: Path) -> bool:
    return path.is_file() and not is_link_or_reparse(path)


def _next_segment_path(directory: Path) -> Path:
    used = [int(m.group(1)) for p in directory.iterdir() if (m := _SEGMENT_RE.fullmatch(p.name))]
    return directory / f"audit.{max(used, default=0) + 1}.jsonl"


def read_run_events(run_dir: Path | str) -> list[AuditEvent]:
    """All verified events of a run, splicing segments (read-only; never repairs anything).

    A damaged segment is acceptable only if the next segment's first event restates the last good
    ``(seq, hash)`` it continues from; the damaged remainder is ignored.  A torn tail of the final
    segment is tolerated (that event was never fully committed); any other damage raises
    ``IntegrityError``.
    """
    return _splice(segment_paths(run_dir))[0]


def verify_run(run_dir: Path | str, anchor: Anchor | None = None) -> VerifyResult:
    """Verify the spliced chain of a run (all segments) and, optionally, an anchor.

    Unlike :func:`read_run_events` this also reports a torn tail (``ok=False``,
    ``torn_tail=True``), exactly like ``AuditLog.verify_file`` does for a single file.
    """
    paths = segment_paths(run_dir)
    if not paths:
        return VerifyResult(False, 0, reason="audit log not found")
    try:
        events, tail = _splice(paths)
    except AuditIntegrityError as exc:
        return exc.result
    except IntegrityError as exc:
        return VerifyResult(False, 0, reason=str(exc))
    return _result_for(_Scan(events, 0, tail), anchor)


def _splice(paths: Sequence[Path]) -> tuple[list[AuditEvent], _Failure | None]:
    """Events of all segments plus the tolerated torn-tail failure of the last one (if any)."""
    events: list[AuditEvent] = []
    tail: _Failure | None = None
    for index, path in enumerate(paths):
        scan = _scan(read_bytes_retry(path), run_id=events[0].run_id if events else None)
        last_segment = index == len(paths) - 1
        failure = scan.failure
        if failure is not None and last_segment and failure.tail is None:
            raise AuditIntegrityError(f"{path.name}: {failure.reason}", _result_for(scan, None))
        if last_segment:
            tail = failure
        if not scan.events:
            continue
        if index == 0:
            events = list(scan.events)
            continue
        first = scan.events[0]
        keep = [e for e in events if e.seq < first.seq]
        previous_hash = keep[-1].hash if keep else GENESIS_HASH
        if (
            not _is_segment_start(first)
            or len(keep) != first.seq - 1
            or first.prev_hash != previous_hash
        ):
            raise IntegrityError(f"{path.name} does not continue the previous segment")
        events = [*keep, *scan.events]
    return events, tail


# =============================================================================
# The log
# =============================================================================

_SYSTEM_ACTOR = Actor.system()
_Listener = Callable[[AuditEvent], None]


class _Dispatcher:
    """Delivers events to listeners from one daemon thread, in the order they were submitted.

    It holds no reference to the log, so ``weakref.finalize`` can stop the thread when an
    abandoned log is garbage collected instead of leaking an idle thread per run.
    """

    def __init__(self, name: str) -> None:
        self._inbox: queue.Queue[AuditEvent | threading.Event | None] = queue.Queue()
        self._listeners: tuple[_Listener, ...] = ()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def add(self, listener: _Listener) -> None:
        self._listeners = (*self._listeners, listener)

    def submit(self, event: AuditEvent) -> None:
        self._inbox.put(event)  # unbounded queue: never blocks, so it is safe under the log lock

    def flush(self, timeout: float) -> bool:
        if threading.current_thread() is self._thread:
            raise StateError("flush() from a listener would deadlock")
        if not self._thread.is_alive():
            return True
        marker = threading.Event()
        self._inbox.put(marker)
        return marker.wait(timeout)

    def stop(self, timeout: float = 0.0) -> None:
        """Process everything already queued, then end the thread (``timeout`` > 0 waits for it)."""
        self._inbox.put(None)
        if timeout > 0 and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            item = self._inbox.get()
            if item is None:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            for listener in self._listeners:
                try:
                    listener(item)
                except Exception as exc:  # observers must never break the audit trail
                    _LOG.warning(
                        "audit listener failed on event %d (%s): %s",
                        item.seq,
                        item.type.value,
                        type(exc).__name__,
                    )


class AuditLog:
    """Thread-safe, append-only, hash-chained JSONL audit log (satisfies ``types.AuditSink``).

    Opening an existing file verifies the whole chain and continues it.  ``mode="recover"``
    additionally repairs a *torn final line* (a crash artefact) and records it as
    ``AUDIT_TAIL_REPAIRED``; damage anywhere else raises :class:`AuditIntegrityError` in both
    modes.  ``seq`` (from 1) and ``ts`` are assigned inside the lock so the chain order is the
    append order.  Listeners are dispatched from a dedicated thread after the append and never under
    the lock, so a listener may itself append without deadlocking.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str,
        trace_id: str | None = None,
        clock: Clock,
        redactor: Redactor | None = None,
        mode: Literal["strict", "recover"] = "strict",
    ) -> None:
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise ConfigError("invalid run_id for the audit log")
        if trace_id is not None and not (
            isinstance(trace_id, str) and _TRACE_ID_RE.fullmatch(trace_id)
        ):
            raise ConfigError("invalid trace_id for the audit log")
        if mode not in ("strict", "recover"):
            raise ConfigError("audit mode must be 'strict' or 'recover'")
        self._path = Path(path)
        if is_link_or_reparse(self._path):
            raise ConfigError("the audit log path must not be a symlink or reparse point")
        self._run_id = run_id
        self._clock = clock
        self._redactor = redactor if redactor is not None else Redactor()
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []
        self._seq = 0
        self._hash = GENESIS_HASH
        self._failed = False
        self._closed = False
        self._dispatcher: _Dispatcher | None = None
        self._trace_id = trace_id or hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32]
        self._open_existing(mode, adopt_trace=trace_id is None)

    # ------------------------------------------------------------------ opening

    @property
    def path(self) -> Path:
        return self._path

    def _open_existing(self, mode: str, *, adopt_trace: bool) -> None:
        try:
            data = read_bytes_retry(self._path)
        except FileNotFoundError:
            return
        scan = _scan(data, run_id=self._run_id)
        failure = scan.failure
        if failure is not None and (failure.tail is None or mode != "recover"):
            raise AuditIntegrityError(
                f"audit log {self._path.name}: {failure.reason}", _result_for(scan, None)
            )
        if scan.events:
            last = scan.events[-1]
            self._events = list(scan.events)
            self._seq, self._hash = last.seq, last.hash
            if adopt_trace:
                self._trace_id = scan.events[0].trace_id
        if failure is not None:
            self._repair(failure, data, scan)

    def _repair(self, failure: _Failure, data: bytes, scan: _Scan) -> None:
        """Make the file append-safe again and say so in the chain (never silently)."""
        dropped = data[scan.valid_bytes :]
        if failure.tail == "torn":
            with open(self._path, "r+b") as handle:
                handle.truncate(scan.valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            detail: dict[str, Any] = {
                "repair": "truncated_torn_tail",
                "dropped_bytes": len(dropped),
            }
        else:
            append_line(self._path, b"", fsync=True)  # just the missing newline
            detail = {"repair": "added_missing_newline", "dropped_bytes": 0}
        detail["dropped_sha256"] = hashlib.sha256(dropped).hexdigest()
        detail["last_good_seq"] = self._seq
        payload = self._prepare_data(detail)
        with self._lock:
            self._append_locked(EventType.AUDIT_TAIL_REPAIRED, None, None, _SYSTEM_ACTOR, payload)

    @classmethod
    def open_new_segment(
        cls,
        path: Path | str,
        *,
        broken_at: int,
        last_good_hash: str,
        run_id: str,
        clock: Clock,
        trace_id: str | None = None,
        redactor: Redactor | None = None,
    ) -> AuditLog:
        """Fork a new chain segment next to a damaged ``path`` (``audit.<n>.jsonl``).

        The first event restates ``broken_at`` and ``last_good_hash`` and continues from them, so
        the verified prefix of the damaged file plus the new segment form one valid chain; the
        damaged file itself is never modified.  Callers get both values from the
        ``AuditIntegrityError`` raised on open (or ``verify_file``).
        """
        if isinstance(broken_at, bool) or not isinstance(broken_at, int) or broken_at < 1:
            raise ConfigError("broken_at must be a positive integer")
        if not isinstance(last_good_hash, str) or not _HASH_RE.fullmatch(last_good_hash):
            raise ConfigError("last_good_hash must be 64 lowercase hex characters")
        if (broken_at == 1) != (last_good_hash == GENESIS_HASH):
            raise ConfigError("last_good_hash must be the genesis hash exactly when broken_at is 1")
        directory = Path(path).parent
        segment = _next_segment_path(directory)
        log = cls(segment, run_id=run_id, trace_id=trace_id, clock=clock, redactor=redactor)
        # The fresh log is empty; presetting its chain state makes the first event the continuation.
        log._seq, log._hash = broken_at - 1, last_good_hash
        log.emit(
            EventType.AUDIT_TAIL_REPAIRED,
            data={
                "reason": _SEGMENT_REASON,
                "segment": segment.name,
                "broken_file": Path(path).name,
                "broken_at": broken_at,
                "last_good_hash": last_good_hash,
            },
        )
        return log

    # ------------------------------------------------------------------ appending

    def emit(
        self,
        type: EventType,
        *,
        node_id: str | None = None,
        attempt: int | None = None,
        actor: Actor = _SYSTEM_ACTOR,
        data: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        """Append one event and return it (as persisted: redacted, hashed, sequenced)."""
        event_type = EventType(type)
        _check_target(node_id, attempt)
        if not isinstance(actor, Actor):
            raise TypeError("actor must be an Actor")
        payload = self._prepare_data(data)
        with self._lock:
            event = self._append_locked(event_type, node_id, attempt, actor, payload)
            if (
                self._dispatcher is not None
            ):  # enqueue under the lock: dispatch order == chain order
                self._dispatcher.submit(event)
        return event

    def __call__(
        self,
        type: EventType,
        *,
        node_id: str | None = None,
        attempt: int | None = None,
        actor: Actor = _SYSTEM_ACTOR,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        """``types.AuditSink`` entry point (same keywords as :meth:`emit`)."""
        self.emit(type, node_id=node_id, attempt=attempt, actor=actor, data=data)

    def _prepare_data(self, data: Mapping[str, Any] | None) -> dict[str, Any]:
        """Serialise, redact and size-bound ``data`` (outside the lock: the expensive part)."""
        if data is None:
            return {}
        failure = {"_error": "audit data was not serialisable"}
        if not isinstance(data, Mapping):
            return failure
        try:
            clean: dict[str, Any] = self._redactor.redact(to_jsonable(dict(data)))
            blob = canonical_json(clean)
        except (TypeError, ValueError, RecursionError):
            return failure
        if len(blob) > MAX_EVENT_BYTES:
            return {
                "_truncated": True,
                "original_bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        return clean

    def _append_locked(
        self,
        event_type: EventType,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: Mapping[str, Any],
    ) -> AuditEvent:
        if self._failed:
            raise StateError(
                "audit log is in a failed state after a write error; refusing to append"
            )
        if self._closed:
            raise StateError("audit log is closed")
        ts = self._clock.now()
        if not isinstance(ts, datetime) or ts.utcoffset() is None:
            raise StateError("the clock must return timezone-aware datetimes")
        span_id, parent_span_id = _spans(node_id, attempt)
        body: dict[str, Any] = {
            "seq": self._seq + 1,
            "ts": ts.isoformat(),
            "run_id": self._run_id,
            "trace_id": self._trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "type": event_type.value,
            "node_id": node_id,
            "attempt": attempt,
            "actor": {
                "type": actor.type.value,
                "id": self._redactor.redact_text(str(actor.id))[:_ACTOR_ID_CHARS],
            },
            "data": data,
            "prev_hash": self._hash,
        }
        digest = hash_jsonable(body)
        full = {**body, "hash": digest}
        try:
            append_line(self._path, canonical_json(full), fsync=event_type in GOVERNANCE_EVENTS)
        except OSError as exc:
            self._failed = True  # a partial write may sit at the tail; never append behind it
            raise StateError(
                f"audit append failed ({type(exc).__name__}); log is now read-only"
            ) from exc
        event = AuditEvent.from_dict(full)
        self._seq, self._hash = event.seq, digest
        self._events.append(event)
        return event

    # ------------------------------------------------------------------ queries

    def events(self) -> list[AuditEvent]:
        """Events of this file in chain order (a copy)."""
        with self._lock:
            return list(self._events)

    def head(self) -> tuple[int, str]:
        """``(seq, hash)`` of the newest event; ``(0, GENESIS_HASH)`` for an empty log."""
        with self._lock:
            return self._seq, self._hash

    def verify(self, anchor: Anchor | None = None) -> VerifyResult:
        """Re-read the file from disk and verify the chain (and ``anchor``, if given)."""
        with self._lock:
            try:
                data = read_bytes_retry(self._path)
            except FileNotFoundError:
                data = b""
        return _result_for(_scan(data, run_id=self._run_id), anchor)

    @staticmethod
    def verify_file(path: Path | str, anchor: Anchor | None = None) -> VerifyResult:
        """Verify a log file without opening it for writing; never raises for damaged content."""
        try:
            data = read_bytes_retry(path)
        except FileNotFoundError:
            return VerifyResult(False, 0, reason="audit log not found")
        except (OSError, ValueError) as exc:  # ValueError: a path with an embedded NUL
            return VerifyResult(False, 0, reason=f"audit log unreadable ({type(exc).__name__})")
        return _result_for(_scan(data, run_id=None), anchor)

    # ------------------------------------------------------------------ listeners

    def add_listener(self, callback: _Listener) -> None:
        """Call ``callback(event)`` for every event appended from now on, from a dispatcher thread.

        Exceptions raised by a listener are logged (class name only) and swallowed: observers can
        never break or stall the audit trail.
        """
        if not callable(callback):
            raise TypeError("listener must be callable")
        with self._lock:
            if self._dispatcher is None:
                self._dispatcher = _Dispatcher(f"audit-dispatch-{self._run_id}")
                weakref.finalize(self, self._dispatcher.stop)
            self._dispatcher.add(callback)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every event appended so far has been dispatched to the listeners."""
        dispatcher = self._dispatcher
        return True if dispatcher is None else dispatcher.flush(timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Drain pending listener callbacks and stop the dispatcher. Further appends are refused."""
        with self._lock:
            self._closed = True
            dispatcher = self._dispatcher
        if dispatcher is not None:
            dispatcher.stop(timeout)

    # ------------------------------------------------------------------ pure helpers

    @staticmethod
    def normalize_events(events: Iterable[AuditEvent]) -> list[dict[str, Any]]:
        """Run-independent form of an event stream for golden tests.

        Drops everything derived from time or the chain (``seq``, ``ts``, ``trace_id``, span ids,
        ``prev_hash``, ``hash``) and makes the tool-call order of an attempt deterministic: its
        ``tool.call`` / ``tool.denied`` events move right behind that attempt's ``node.started`` and
        are sorted by content, because a worker thread's calls interleave with scheduler events
        non-deterministically.
        """
        stream = list(events)
        started = {(e.node_id, e.attempt) for e in stream if e.type is EventType.NODE_STARTED}
        tool_types = (EventType.TOOL_CALL, EventType.TOOL_DENIED)
        tools: dict[tuple[str | None, int | None], list[AuditEvent]] = {}
        for event in stream:
            key = (event.node_id, event.attempt)
            if event.type in tool_types and key in started:
                tools.setdefault(key, []).append(event)
        ordered: list[AuditEvent] = []
        for event in stream:
            key = (event.node_id, event.attempt)
            if event.type in tool_types and key in started:
                continue
            ordered.append(event)
            if event.type is EventType.NODE_STARTED:
                ordered.extend(sorted(tools.get(key, ()), key=_content_key))
        dropped = {"seq", "ts", "trace_id", "span_id", "parent_span_id", "prev_hash", "hash"}
        return [{k: v for k, v in e.to_dict().items() if k not in dropped} for e in ordered]


def _content_key(event: AuditEvent) -> tuple[str, bytes]:
    """Sort key for an attempt's tool events: kind first (calls before denials), then content."""
    return event.type.value, canonical_json({"actor": event.actor, "data": event.data})


# =============================================================================
# Trace export
# =============================================================================

_RUN_STATUS = {
    EventType.RUN_CREATED: "running",
    EventType.RUN_STARTED: "running",
    EventType.RUN_RESUMED: "running",
    EventType.RUN_WAITING: "waiting_human",
    EventType.RUN_SUCCEEDED: "succeeded",
    EventType.RUN_FAILED: "failed",
    EventType.RUN_SAFE_STOPPED: "safe_stopped",
}
_NODE_STATUS = {
    EventType.NODE_READY: "ready",
    EventType.NODE_SKIPPED: "skipped",
    EventType.NODE_STARTED: "running",
    EventType.NODE_RETRY_SCHEDULED: "ready",
    EventType.NODE_AWAITING_APPROVAL: "awaiting_approval",
    EventType.NODE_AWAITING_INPUT: "awaiting_input",
    EventType.NODE_SUCCEEDED: "succeeded",
    EventType.NODE_FAILED: "failed",
    EventType.NODE_BLOCKED: "blocked",
    EventType.NODE_STALE: "stale",
    EventType.NODE_RESET: "pending",
    EventType.NODE_REMOVED: "removed",
    EventType.NODE_CANCELLED: "cancelled",
    EventType.NODE_ROLLED_BACK: "rolled_back",
}
_ATTEMPT_STATUS = {
    EventType.NODE_STARTED: "running",
    EventType.NODE_ATTEMPT_FAILED: "failed",
    EventType.NODE_ATTEMPT_DISCARDED: "discarded",
    EventType.NODE_AWAITING_APPROVAL: "awaiting_approval",
    EventType.NODE_AWAITING_INPUT: "awaiting_input",
    EventType.NODE_SUCCEEDED: "succeeded",
    EventType.NODE_FAILED: "failed",
    EventType.NODE_CANCELLED: "cancelled",
}


def run_status_of(events: Sequence[AuditEvent]) -> str:
    """Latest run-level status in the stream: running | waiting_human | succeeded | failed |
    safe_stopped (``"unknown"`` if the stream holds no run-level event)."""
    status = "unknown"
    for event in events:
        status = _RUN_STATUS.get(event.type, status)
    return status


def _span(
    span_id: str, name: str, events: Sequence[AuditEvent], statuses: Mapping[EventType, str]
) -> dict[str, Any]:
    status = "unknown"
    for event in events:
        status = statuses.get(event.type, status)
    start, end = events[0].ts, events[-1].ts
    return {
        "span_id": span_id,
        "name": name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "duration_s": max(0.0, (end - start).total_seconds()),
        "status": status,
        "events": len(events),
    }


def export_trace(events: Sequence[AuditEvent]) -> dict[str, Any]:
    """Span tree ``run -> node -> attempt`` for a run (``trace.json``).

    A span covers its *whole lifetime* (first to last event, so an attempt includes time spent
    waiting for approval); ``status`` is the one implied by its last status-bearing event.  Spans
    carry only engine-derived ids and the failure ``kind``; no event payload is copied.
    """
    stream = list(events)
    if not stream:
        return {"trace_id": "", "run_id": "", **_empty_span(), "children": []}
    by_node: dict[str, list[AuditEvent]] = {}
    for event in stream:
        if event.node_id is not None:
            by_node.setdefault(event.node_id, []).append(event)
    children = [_node_span(node_id, node_events) for node_id, node_events in by_node.items()]
    root = _span("run", stream[0].run_id, stream, _RUN_STATUS)
    return {
        "trace_id": stream[0].trace_id,
        "run_id": stream[0].run_id,
        **root,
        "children": children,
    }


def _empty_span() -> dict[str, Any]:
    return {
        "span_id": "run",
        "name": "",
        "start": None,
        "end": None,
        "duration_s": 0.0,
        "status": "empty",
        "events": 0,
    }


def _node_span(node_id: str, events: Sequence[AuditEvent]) -> dict[str, Any]:
    span = _span(f"n:{node_id}", node_id, events, _NODE_STATUS)
    by_attempt: dict[int, list[AuditEvent]] = {}
    for event in events:
        if event.attempt is not None:
            by_attempt.setdefault(event.attempt, []).append(event)
    span["children"] = [_attempt_span(node_id, n, by_attempt[n]) for n in sorted(by_attempt)]
    return span


def _attempt_span(node_id: str, attempt: int, events: Sequence[AuditEvent]) -> dict[str, Any]:
    span = _span(f"n:{node_id}#{attempt}", f"{node_id}#{attempt}", events, _ATTEMPT_STATUS)
    for event in events:
        if event.type is EventType.NODE_ATTEMPT_FAILED and isinstance(event.data.get("kind"), str):
            span["failure_kind"] = sanitize_for_terminal(event.data["kind"], 64)
    return span
