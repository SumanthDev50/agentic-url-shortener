"""Versioned, content-addressed artifact store (the run's single source of produced data).

Layout (``root`` is ``<run_dir>/artifacts``)::

    root/index.json                                         atomic, the only authority
    root/<kind>/<slug(key)>-<sha256(key)[:8]>/v<N>/content.<ext>

* **Raw keys and agent file names never become path components.** The directory is a bounded
  slug plus a hash of the key, ``ext`` comes from a fixed allow-list, and every join goes
  through :func:`fsutil.safe_join` (validation, containment, no links).
* **The index is hostile input.** On load every entry is re-derived: id, storage path (it
  must equal the path computed from kind/key/version), hashes, producer and status
  invariants. A tampered index (``..``, absolute or drive paths, two artifacts sharing one
  file, two ACTIVE versions of one ``(kind, key)``) raises :class:`IntegrityError`.
* **Crash safety.** Content is written (atomically) before the index entry that names it, and
  the index is replaced atomically with the in-memory state swapped only after the write
  succeeded, so a failed write leaves memory and disk unchanged. A crash can leave an orphan
  content file, never an index entry without content.
* **Audit order.** Events are emitted *after* the index is durable, so nothing in the audit
  chain describes state that was not persisted; a failure between the two shows up in
  :meth:`ArtifactStore.verify_against_events` instead of being silently papered over.
* **Visibility.** Only ACTIVE artifacts are visible downstream (``latest``/``active``).
  STAGED artifacts belong to one attempt until the engine activates them (all-or-nothing,
  idempotent). Superseded and quarantined artifacts stay in the index as history.

Thread-safe (one re-entrant lock); not safe across processes: the run's ``RunLock`` guarantees
a single mutating process.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from orchestrator.core.errors import (
    ConfigError,
    IntegrityError,
    PathViolation,
    StateError,
    UndeclaredInputError,
)
from orchestrator.core.fsutil import (
    atomic_write_bytes,
    atomic_write_json,
    read_bytes_retry,
    safe_join,
    sha256_bytes,
)
from orchestrator.core.paths import (
    FILENAME_RE,
    KEY_RE,
    KIND_RE,
    NODE_ID_RE,
    make_artifact_id,
    validate_rel_path,
)
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.serde import canonical_json, loads_strict, sha256_hex, to_jsonable
from orchestrator.core.types import (
    EXTERNAL_PRODUCER,
    Actor,
    Artifact,
    ArtifactDraft,
    ArtifactReader,
    ArtifactRef,
    ArtifactStatus,
    AuditEvent,
    AuditSink,
    Clock,
    EventType,
)

INDEX_SCHEMA = 1
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024
MAX_INPUT_REFS = 500
MAX_REASON_CHARS = 300

# Bytes that replace the content of a tombstoned artifact (hash, size and metadata are kept).
TOMBSTONE = b"[content removed: artifact quarantined with tombstone]\n"

_EXTENSIONS = frozenset({"txt", "md", "json", "py", "sql", "patch", "diff"})
_MEDIA_EXTENSION = {
    "text/plain": "txt",
    "text/markdown": "md",
    "application/json": "json",
    "text/x-python": "py",
    "application/sql": "sql",
    "text/x-sql": "sql",
    "text/x-patch": "patch",
    "text/x-diff": "diff",
}
_MEDIA_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/[a-z0-9][a-z0-9!#$&^_.+-]{0,63}\Z")
_AGENT_ID_RE = re.compile(r"^(|[A-Za-z0-9][A-Za-z0-9._:-]{0,63})\Z")
_HASH_RE = re.compile(r"^[0-9a-f]{64}\Z")

# Metadata keys written by the store itself. They start with ``_`` precisely so that the
# reserved-key stripping applied to agent-supplied metadata makes them unforgeable.
_M_SUPERSEDED_BY = "_superseded_by"
_M_SUPERSEDED_REASON = "_superseded_reason"
_M_DEDUPED_TO = "_deduped_to"
_M_QUARANTINE_REASON = "_quarantine_reason"
_M_TOMBSTONE = "_tombstone"

_Key = tuple[str, str]  # (kind, key)


# =============================================================================
# helpers
# =============================================================================


def freeze_json(value: Any) -> Any:
    """Deep-immutable copy (mappings become read-only proxies, lists become tuples).

    Records handed to callers (artifacts, decisions) must not let them edit the store's own
    copy through nested metadata. ``to_jsonable`` serialises the frozen form identically.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze_json(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(v) for v in value)
    return value


def utc_now(clock: Clock) -> datetime:
    """``clock.now()`` normalised to UTC; naive datetimes are a configuration error."""
    now = clock.now()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ConfigError("clock.now() must return a timezone-aware datetime")
    return now.astimezone(UTC)


def _sort_key(art: Artifact) -> tuple[str, str, int]:
    return (art.kind, art.key, art.version)


def _storage_path(kind: str, key: str, version: int, ext: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")[:24].strip("-") or "_"
    return f"{kind}/{slug}-{sha256_hex(key)[:8]}/v{version}/content.{ext}"


def _note(reason: str) -> str:
    return sanitize_for_terminal(reason, MAX_REASON_CHARS)


def _encode_content(content: Any) -> tuple[bytes, bool]:
    """Stored bytes of a draft's content and whether it is structured (JSON)."""
    if isinstance(content, (bytes, bytearray, memoryview)):
        return bytes(content), False
    if isinstance(content, str):
        try:
            return content.encode("utf-8"), False
        except UnicodeEncodeError as exc:
            raise ConfigError("artifact text is not valid UTF-8 (lone surrogate?)") from exc
    if isinstance(content, (Mapping, Sequence)):
        try:
            text = json.dumps(
                to_jsonable(content), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise ConfigError(f"artifact content is not JSON-serialisable: {exc}") from exc
        return (text + "\n").encode("ascii"), True
    raise ConfigError(f"unsupported artifact content type: {type(content).__name__}")


def _media_type(draft: ArtifactDraft, structured: bool) -> str:
    if structured:
        return "application/json"
    declared = draft.media_type
    if declared is None:
        return "text/plain" if isinstance(draft.content, str) else "application/octet-stream"
    declared = declared.lower()
    if not _MEDIA_TYPE_RE.match(declared):
        raise ConfigError(f"invalid media type: {sanitize_for_terminal(declared, 80)!r}")
    return declared


def _choose_extension(filename: str | None, declared_media_type: str, structured: bool) -> str:
    if structured:
        return "json"
    ext = _MEDIA_EXTENSION.get(declared_media_type)  # a declared type wins over the file name
    if ext:
        return ext
    if filename and "." in filename:
        # only the allow-listed suffix is used; the file name itself is never a path component
        suffix = filename.rsplit(".", 1)[1].lower()
        if suffix in _EXTENSIONS:
            return suffix
    return "txt"


def _clean_metadata(raw: Mapping[str, Any], filename: str | None) -> dict[str, Any]:
    """Canonical (JSON round-tripped) metadata without reserved ``_*`` keys, size-bounded."""
    try:
        canonical = json.loads(canonical_json(raw))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ConfigError(f"artifact metadata is not JSON-serialisable: {exc}") from exc
    if not isinstance(canonical, dict):
        raise ConfigError("artifact metadata must be a mapping")
    meta = {k: v for k, v in canonical.items() if not k.startswith("_")}
    if filename is not None:
        meta["filename"] = filename
    if len(canonical_json(meta)) > MAX_METADATA_BYTES:
        raise ConfigError(f"artifact metadata exceeds {MAX_METADATA_BYTES} bytes")
    return meta


def _validate_draft(draft: ArtifactDraft) -> None:
    # the draft is a mutable dataclass: re-check what ``__post_init__`` checked at construction
    if not KIND_RE.match(draft.kind):
        raise ConfigError(f"invalid artifact kind: {sanitize_for_terminal(draft.kind, 60)!r}")
    if not KEY_RE.match(draft.key):
        raise ConfigError(f"invalid artifact key: {sanitize_for_terminal(draft.key, 60)!r}")
    if draft.filename is not None and not FILENAME_RE.match(draft.filename):
        raise ConfigError("invalid artifact filename")


def _validate_provenance(producer: str, attempt: int, agent_id: str, inputs: Sequence[Any]) -> None:
    if producer != EXTERNAL_PRODUCER and not NODE_ID_RE.match(producer):
        raise ConfigError(f"invalid producer: {sanitize_for_terminal(producer, 60)!r}")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ConfigError("attempt must be a non-negative integer")
    if not _AGENT_ID_RE.match(agent_id):
        raise ConfigError(f"invalid agent id: {sanitize_for_terminal(agent_id, 60)!r}")
    if len(inputs) > MAX_INPUT_REFS:
        raise ConfigError(f"too many input refs ({len(inputs)} > {MAX_INPUT_REFS})")
    if not all(isinstance(r, ArtifactRef) for r in inputs):
        raise ConfigError("inputs must be ArtifactRef objects")


@dataclass(frozen=True)
class _Event:
    type: EventType
    artifact: Artifact
    data: dict[str, Any]
    attempt: bool = False  # created-events belong to the attempt span, the rest to the node span


@dataclass(frozen=True)
class _Publication:
    updates: dict[str, Artifact]  # index changes (new/changed records)
    visible: dict[str, Artifact]  # candidate id -> the ACTIVE artifact that now represents it
    events: list[_Event]


# =============================================================================
# the store
# =============================================================================


class ArtifactStore:
    """See the module docstring. ``audit`` is any ``types.AuditSink`` (e.g. an ``AuditLog``)."""

    def __init__(
        self,
        root: Path | str,
        *,
        clock: Clock,
        audit: AuditSink | None = None,
        max_artifact_bytes: int = MAX_ARTIFACT_BYTES,
    ) -> None:
        if max_artifact_bytes <= 0:
            raise ConfigError("max_artifact_bytes must be positive")
        self._clock = clock
        self._audit = audit
        self._max_bytes = max_artifact_bytes
        self._lock = threading.RLock()
        Path(root).mkdir(parents=True, exist_ok=True)
        self.root = Path(root).resolve()
        self._index_path = self.root / "index.json"
        self._by_id: dict[str, Artifact] = {}
        self._versions: dict[_Key, list[str]] = {}
        self._paths: dict[str, str] = {}
        self._load_index()

    # ------------------------------------------------------------------ index I/O

    def _load_index(self) -> None:
        try:
            raw = read_bytes_retry(self._index_path)
        except FileNotFoundError:
            return
        try:
            doc = loads_strict(raw)
        except ValueError as exc:
            raise IntegrityError(f"corrupt artifact index: {exc}") from exc
        if (
            not isinstance(doc, dict)
            or doc.get("schema") != INDEX_SCHEMA
            or not isinstance(doc.get("artifacts"), list)
        ):
            raise IntegrityError("corrupt artifact index: unexpected document shape")
        for entry in doc["artifacts"]:
            art = self._parse_entry(entry)
            if art.id in self._by_id:
                raise IntegrityError(f"corrupt artifact index: duplicate id {art.id}")
            if art.path in self._paths:
                raise IntegrityError(f"corrupt artifact index: {art.path} used by two artifacts")
            self._by_id[art.id] = art
            self._register(art)
        for kk, ids in self._versions.items():
            actives = [i for i in ids if self._by_id[i].status == ArtifactStatus.ACTIVE]
            if len(actives) > 1:
                raise IntegrityError(f"corrupt artifact index: several ACTIVE versions of {kk[0]}")

    def _parse_entry(self, entry: Any) -> Artifact:
        try:
            art = Artifact.from_dict(entry)
            self._validate_loaded(art)
        except (ValueError, LookupError, TypeError, AttributeError, OverflowError) as exc:
            raise IntegrityError(f"corrupt artifact index entry: {exc}") from exc
        return dataclasses.replace(art, metadata=freeze_json(art.metadata))

    @staticmethod
    def _validate_loaded(art: Artifact) -> None:
        strings = (art.id, art.kind, art.key, art.path, art.content_hash, art.media_type)
        if not all(isinstance(s, str) for s in (*strings, art.producer, art.agent_id)):
            raise ValueError("non-string identifier")
        if not KIND_RE.match(art.kind) or not KEY_RE.match(art.key) or art.version < 1:
            raise ValueError("invalid kind/key/version")
        if art.id != make_artifact_id(art.kind, art.key, art.version):
            raise ValueError(f"id {art.id!r} does not match kind/key/version")
        if not _HASH_RE.match(art.content_hash) or art.size < 0 or art.attempt < 0:
            raise ValueError("invalid hash/size/attempt")
        if art.producer != EXTERNAL_PRODUCER and not NODE_ID_RE.match(art.producer):
            raise ValueError("invalid producer")
        if not _MEDIA_TYPE_RE.match(art.media_type) or not _AGENT_ID_RE.match(art.agent_id):
            raise ValueError("invalid media type / agent id")
        try:
            validate_rel_path(art.path)
        except PathViolation as exc:
            raise ValueError(f"unsafe storage path: {exc}") from exc
        ext = art.path.rsplit(".", 1)[-1]
        if ext not in _EXTENSIONS or art.path != _storage_path(art.kind, art.key, art.version, ext):
            raise ValueError(f"storage path {art.path!r} is not the canonical path of {art.id}")
        for ref in art.inputs:
            if not isinstance(ref.artifact_id, str) or not isinstance(ref.content_hash, str):
                raise ValueError("invalid input ref")

    def _register(self, art: Artifact) -> None:
        ids = self._versions.setdefault((art.kind, art.key), [])
        if art.id not in ids:
            ids.append(art.id)
            ids.sort(key=lambda i: self._by_id[i].version)
        self._paths[art.path] = art.id

    def _commit(self, updates: Mapping[str, Artifact]) -> None:
        """Persist ``updates`` (index first), then adopt them in memory: all or nothing."""
        merged = {**self._by_id, **updates}
        doc = {
            "schema": INDEX_SCHEMA,
            "artifacts": [a.to_dict() for a in sorted(merged.values(), key=_sort_key)],
        }
        atomic_write_json(self._index_path, doc)
        self._by_id = merged
        for art in updates.values():
            self._register(art)

    # ------------------------------------------------------------------ internals

    def _abs(self, rel: str) -> Path:
        try:
            return safe_join(self.root, rel)
        except PathViolation as exc:
            raise IntegrityError(f"artifact path rejected: {exc}") from exc

    def _record(self, artifact_id: str) -> Artifact:
        art = self._by_id.get(artifact_id) if isinstance(artifact_id, str) else None
        if art is None:
            raise StateError(f"unknown artifact: {sanitize_for_terminal(artifact_id, 80)}")
        return art

    def _matching_record(self, artifact: Artifact) -> Artifact:
        """The store's own record for ``artifact``; a caller-built object is never trusted."""
        rec = self._record(artifact.id)
        fields = ("kind", "key", "version", "path", "content_hash", "size")
        if any(getattr(rec, f) != getattr(artifact, f) for f in fields):
            raise IntegrityError(f"artifact {rec.id} does not match the store record")
        return rec

    def _active_of(self, kk: _Key) -> Artifact | None:
        for artifact_id in reversed(self._versions.get(kk, ())):
            art = self._by_id[artifact_id]
            if art.status == ArtifactStatus.ACTIVE:
                return art
        return None

    @staticmethod
    def _unique_ids(ids: Iterable[str]) -> list[str]:
        if isinstance(ids, (str, bytes)):
            raise TypeError("ids must be an iterable of artifact ids, not a single string")
        unique = list(dict.fromkeys(ids))
        if not all(isinstance(i, str) for i in unique):
            raise TypeError("artifact ids must be strings")
        return unique

    @staticmethod
    def _with_status(art: Artifact, status: ArtifactStatus, **meta: Any) -> Artifact:
        return dataclasses.replace(
            art, status=status, metadata=freeze_json({**art.metadata, **meta})
        )

    def _verified_bytes(self, rec: Artifact) -> bytes:
        if rec.metadata.get(_M_TOMBSTONE):
            raise StateError(f"content of {rec.id} was removed (tombstone)")
        path = self._abs(rec.path)
        try:
            if path.stat().st_size != rec.size:  # bound memory before reading a swapped file
                raise IntegrityError(f"size of {rec.id} differs from the index")
            raw = read_bytes_retry(path)
        except FileNotFoundError as exc:
            raise IntegrityError(f"content file of {rec.id} is missing") from exc
        if len(raw) != rec.size or sha256_bytes(raw) != rec.content_hash:
            raise IntegrityError(f"content hash mismatch for {rec.id}")
        return raw

    def _emit(self, events: Iterable[_Event]) -> None:
        if self._audit is None:
            return
        for ev in events:
            art = ev.artifact
            external = art.producer == EXTERNAL_PRODUCER
            actor = Actor.system()
            if ev.type == EventType.ARTIFACT_CREATED:
                actor = (
                    Actor.agent(art.agent_id) if art.agent_id else Actor.system("artifact-store")
                )
            self._audit(
                ev.type,
                node_id=None if external else art.producer,
                attempt=art.attempt if ev.attempt and not external else None,
                actor=actor,
                data=ev.data,
            )

    # ------------------------------------------------------------------ writing

    def put(
        self,
        draft: ArtifactDraft,
        *,
        producer: str,
        attempt: int = 0,
        agent_id: str = "",
        inputs: Sequence[ArtifactRef] = (),
        staged: bool = False,
    ) -> Artifact:
        """Store ``draft`` as the next version of its ``(kind, key)``.

        ``staged=True`` leaves it STAGED (invisible downstream). ``staged=False`` publishes it
        in the same index write, superseding the previous ACTIVE version of that ``(kind, key)``
        (external inputs such as an amended request); producer ownership is enforced and no
        content de-duplication happens: a deliberate new version stays a new version.
        """
        _validate_draft(draft)
        _validate_provenance(producer, attempt, agent_id, inputs)
        data, structured = _encode_content(draft.content)
        if len(data) > self._max_bytes:
            raise ConfigError(f"artifact exceeds {self._max_bytes} bytes ({len(data)})")
        if structured:
            try:
                loads_strict(data, max_bytes=self._max_bytes)  # must stay readable via read_json
            except ValueError as exc:
                raise ConfigError(f"artifact JSON is not acceptable: {exc}") from exc
        media_type = _media_type(draft, structured)
        ext = _choose_extension(draft.filename, (draft.media_type or "").lower(), structured)
        metadata = _clean_metadata(draft.metadata, draft.filename)
        kk = (draft.kind, draft.key)
        with self._lock:
            version = (
                max((self._by_id[i].version for i in self._versions.get(kk, ())), default=0) + 1
            )
            rel = _storage_path(draft.kind, draft.key, version, ext)
            try:
                validate_rel_path(rel)
            except PathViolation as exc:
                raise ConfigError(f"artifact kind {draft.kind!r} cannot be stored: {exc}") from exc
            if rel in self._paths:
                raise IntegrityError(f"storage path collision for {rel}")
            art = Artifact(
                id=make_artifact_id(draft.kind, draft.key, version),
                kind=draft.kind,
                key=draft.key,
                version=version,
                content_hash=sha256_bytes(data),
                media_type=media_type,
                size=len(data),
                path=rel,
                created_at=utc_now(self._clock),
                producer=producer,
                attempt=attempt,
                agent_id=agent_id,
                inputs=tuple(inputs),
                status=ArtifactStatus.STAGED,
                metadata=freeze_json(metadata),
            )
            plan = (
                None if staged else self._plan_publish([art], dedupe=False)
            )  # may refuse: no I/O yet
            atomic_write_bytes(self._abs(rel), data)
            created = _Event(
                EventType.ARTIFACT_CREATED,
                art,
                {"artifact_id": art.id, "content_hash": art.content_hash, "size": art.size},
                attempt=True,
            )
            if plan is None:
                self._commit({art.id: art})
                self._emit([created])
                return art
            self._commit(plan.updates)
            self._emit([created, *plan.events])
            return plan.visible[art.id]

    def _plan_publish(self, candidates: Sequence[Artifact], *, dedupe: bool) -> _Publication:
        """Decide (without side effects) what activating ``candidates`` changes.

        Refuses a ``(kind, key)`` owned by another producer and a version older than the ACTIVE
        one (no silent rollback). With ``dedupe`` a candidate whose content equals the ACTIVE
        one is itself marked SUPERSEDED and the existing ACTIVE artifact is kept: republishing
        identical content must not create staleness downstream.
        """
        updates: dict[str, Artifact] = {}
        visible: dict[str, Artifact] = {}
        events: list[_Event] = []
        for cand in candidates:
            current = self._active_of((cand.kind, cand.key))
            if current is not None:
                if current.producer != cand.producer:
                    raise StateError(
                        f"{cand.kind}/{cand.key!r} is owned by {current.producer!r}; "
                        f"{cand.producer!r} may not publish it"
                    )
                if cand.version < current.version:
                    raise StateError(f"{cand.id} is older than the ACTIVE {current.id}")
                if dedupe and current.content_hash == cand.content_hash:
                    dropped = self._with_status(
                        cand, ArtifactStatus.SUPERSEDED, **{_M_DEDUPED_TO: current.id}
                    )
                    updates[cand.id] = dropped
                    visible[cand.id] = current
                    events.append(
                        _Event(
                            EventType.ARTIFACT_UNCHANGED,
                            dropped,
                            {
                                "artifact_id": cand.id,
                                "kept": current.id,
                                "content_hash": cand.content_hash,
                            },
                        )
                    )
                    continue
            active = self._with_status(cand, ArtifactStatus.ACTIVE)
            updates[cand.id] = active
            visible[cand.id] = active
            events.append(
                _Event(
                    EventType.ARTIFACT_PUBLISHED,
                    active,
                    {"artifact_id": active.id, "content_hash": active.content_hash},
                )
            )
            if current is not None:
                replaced = self._with_status(
                    current, ArtifactStatus.SUPERSEDED, **{_M_SUPERSEDED_BY: active.id}
                )
                updates[current.id] = replaced
                events.append(
                    _Event(
                        EventType.ARTIFACT_SUPERSEDED,
                        replaced,
                        {
                            "artifact_id": current.id,
                            "reason": "replaced by a newer version",
                            "superseded_by": active.id,
                        },
                    )
                )
        return _Publication(updates, visible, events)

    def activate(self, ids: Iterable[str]) -> list[Artifact]:
        """Publish STAGED artifacts: all-or-nothing under one index write, idempotent.

        Returns, in the order of ``ids``, the ACTIVE artifact that represents each id: the
        artifact itself or, when its content equals the already-ACTIVE one
        (``ARTIFACT_UNCHANGED``), the existing ACTIVE artifact that was kept. Re-activating an id
        that is already ACTIVE (or was de-duplicated onto a still-ACTIVE artifact) is a no-op, so
        a crash-replayed publish cannot create versions or events twice.
        """
        wanted = self._unique_ids(ids)
        with self._lock:
            results: dict[str, Artifact] = {}
            candidates: list[Artifact] = []
            seen: set[_Key] = set()
            for artifact_id in wanted:
                art = self._record(artifact_id)
                kk = (art.kind, art.key)
                if kk in seen:
                    raise StateError(f"one activation names two versions of {art.kind}/{art.key!r}")
                seen.add(kk)
                if art.status == ArtifactStatus.ACTIVE:
                    results[artifact_id] = art
                elif art.status == ArtifactStatus.STAGED:
                    self._verified_bytes(art)  # tampered/missing content must never become visible
                    candidates.append(art)
                elif (kept := self._kept_by_dedupe(art)) is not None:
                    results[artifact_id] = kept
                else:
                    raise StateError(f"cannot activate {artifact_id}: status is {art.status.value}")
            plan = self._plan_publish(candidates, dedupe=True)
            if plan.updates:
                self._commit(plan.updates)
                self._emit(plan.events)
            results.update(plan.visible)
            return [results[i] for i in wanted]

    def _kept_by_dedupe(self, art: Artifact) -> Artifact | None:
        if art.status != ArtifactStatus.SUPERSEDED:
            return None
        target = self._by_id.get(str(art.metadata.get(_M_DEDUPED_TO)))
        if (
            target is not None
            and target.status == ArtifactStatus.ACTIVE
            and target.content_hash == art.content_hash
        ):
            return target
        return None

    def quarantine(
        self, ids: Iterable[str], reason: str, *, tombstone: bool = False
    ) -> list[Artifact]:
        """Mark artifacts QUARANTINED (all-or-nothing index write; idempotent).

        ``tombstone=True`` first replaces the stored bytes with :data:`TOMBSTONE`, keeping hash,
        size and metadata, so a secret that slipped into an artifact does not stay on disk. The
        bytes are replaced *before* the index changes: a crash in between leaves a tombstoned
        file that :meth:`verify_all` reports, never a secret behind a clean index.
        """
        wanted = self._unique_ids(ids)
        note = _note(reason)
        with self._lock:
            records = [self._record(i) for i in wanted]
            updates: dict[str, Artifact] = {}
            events: list[_Event] = []
            for rec in records:
                already = rec.status == ArtifactStatus.QUARANTINED
                if already and (not tombstone or rec.metadata.get(_M_TOMBSTONE)):
                    continue
                meta: dict[str, Any] = {} if already else {_M_QUARANTINE_REASON: note}
                if tombstone:
                    atomic_write_bytes(self._abs(rec.path), TOMBSTONE)
                    meta[_M_TOMBSTONE] = True
                new = self._with_status(rec, ArtifactStatus.QUARANTINED, **meta)
                updates[rec.id] = new
                events.append(
                    _Event(
                        EventType.ARTIFACT_QUARANTINED,
                        new,
                        {"artifact_id": rec.id, "reason": note, "tombstone": tombstone},
                    )
                )
            if updates:
                self._commit(updates)
                self._emit(events)
            return [updates.get(r.id, r) for r in records]

    def supersede(self, artifact_id: str, reason: str) -> Artifact:
        """ACTIVE -> SUPERSEDED (e.g. the outputs of a node removed by a re-plan); idempotent."""
        note = _note(reason)
        with self._lock:
            rec = self._record(artifact_id)
            if rec.status == ArtifactStatus.SUPERSEDED:
                return rec
            if rec.status != ArtifactStatus.ACTIVE:
                raise StateError(f"cannot supersede {artifact_id}: status is {rec.status.value}")
            new = self._with_status(rec, ArtifactStatus.SUPERSEDED, **{_M_SUPERSEDED_REASON: note})
            self._commit({new.id: new})
            self._emit(
                [
                    _Event(
                        EventType.ARTIFACT_SUPERSEDED,
                        new,
                        {"artifact_id": new.id, "reason": note, "superseded_by": None},
                    )
                ]
            )
            return new

    # ------------------------------------------------------------------ querying

    def get(self, artifact_id: str) -> Artifact:
        """The record of ``artifact_id`` (any status). Unknown ids raise :class:`StateError`."""
        with self._lock:
            return self._record(artifact_id)

    def resolve(self, artifact: Artifact) -> Artifact:
        """The store's current record for a caller-held ``artifact`` (fresh status). Raises
        :class:`IntegrityError` if the object's hash/path disagree with the store."""
        with self._lock:
            return self._matching_record(artifact)

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        """The ACTIVE artifact of ``(kind, key)``, if any."""
        with self._lock:
            return self._active_of((kind, key))

    def active(self, kind: str) -> list[Artifact]:
        """The ACTIVE artifact of every key of ``kind``, sorted by key."""
        with self._lock:
            found = [
                art
                for (k, _), ids in self._versions.items()
                if k == kind
                for art in (self._by_id[i] for i in ids)
                if art.status == ArtifactStatus.ACTIVE
            ]
        return sorted(found, key=lambda a: a.key)

    def versions(self, kind: str, key: str = "") -> list[Artifact]:
        """Every version of ``(kind, key)`` in any status, oldest first."""
        with self._lock:
            return [self._by_id[i] for i in self._versions.get((kind, key), ())]

    def all(self, include_inactive: bool = False) -> list[Artifact]:
        """ACTIVE artifacts (default) or every record, sorted by ``(kind, key, version)``."""
        with self._lock:
            records = list(self._by_id.values())
        if not include_inactive:
            records = [a for a in records if a.status == ArtifactStatus.ACTIVE]
        return sorted(records, key=_sort_key)

    def staged(self) -> list[Artifact]:
        with self._lock:
            records = [a for a in self._by_id.values() if a.status == ArtifactStatus.STAGED]
        return sorted(records, key=_sort_key)

    # ------------------------------------------------------------------ reading content

    def read_bytes(self, artifact: Artifact) -> bytes:
        """The stored bytes, verified against the recorded size and ``content_hash``."""
        with self._lock:
            rec = self._matching_record(artifact)
        return self._verified_bytes(rec)

    def read_text(self, artifact: Artifact) -> str:
        return self.read_bytes(artifact).decode("utf-8")

    def read_json(self, artifact: Artifact) -> Any:
        return loads_strict(self.read_bytes(artifact), max_bytes=self._max_bytes)

    # ------------------------------------------------------------------ verification

    def verify_all(self) -> list[str]:
        """Re-hash every stored file; returns human-readable problems (empty = intact)."""
        with self._lock:
            records = sorted(self._by_id.values(), key=_sort_key)
        problems: list[str] = []
        for rec in records:
            try:
                raw = read_bytes_retry(self._abs(rec.path))
            except FileNotFoundError:
                problems.append(f"{rec.id}: content file is missing")
                continue
            except (OSError, IntegrityError) as exc:
                problems.append(f"{rec.id}: unreadable ({exc})")
                continue
            if rec.metadata.get(_M_TOMBSTONE):
                if raw != TOMBSTONE:
                    problems.append(f"{rec.id}: tombstoned artifact still holds content")
            elif len(raw) != rec.size or sha256_bytes(raw) != rec.content_hash:
                problems.append(f"{rec.id}: content does not match the recorded hash")
        return problems

    def verify_against_events(self, events: Iterable[AuditEvent]) -> list[str]:
        """Cross-check the index with the ``ARTIFACT_CREATED`` events of the audit chain."""
        created: dict[str, list[Mapping[str, Any]]] = {}
        problems: list[str] = []
        for ev in events:
            if ev.type != EventType.ARTIFACT_CREATED:
                continue
            artifact_id = ev.data.get("artifact_id")
            if not isinstance(artifact_id, str):
                problems.append(f"event seq {ev.seq}: ARTIFACT_CREATED without an artifact_id")
                continue
            created.setdefault(artifact_id, []).append(ev.data)
        with self._lock:
            records = dict(self._by_id)
        for artifact_id, datas in sorted(created.items()):
            shown = sanitize_for_terminal(artifact_id, 80)
            rec = records.get(artifact_id)
            if rec is None:
                problems.append(
                    f"{shown}: audit chain records its creation but the index has no entry"
                )
            elif len(datas) > 1:
                problems.append(f"{shown}: created more than once in the audit chain")
            elif (datas[0].get("content_hash"), datas[0].get("size")) != (
                rec.content_hash,
                rec.size,
            ):
                problems.append(f"{shown}: index hash/size differ from the audit chain")
        problems.extend(
            f"{rec.id}: no ARTIFACT_CREATED event in the audit chain"
            for rec in sorted(records.values(), key=_sort_key)
            if rec.id not in created
        )
        return problems

    # ------------------------------------------------------------------ readers

    def snapshot(
        self, kinds: Iterable[str], accept_producer: Callable[[str], bool]
    ) -> PinnedReader:
        """Capture, atomically, the ACTIVE artifacts of ``kinds`` whose producer is accepted.

        The result is immutable: later publications, supersessions or quarantines cannot change
        what it serves (no torn reads), and kinds with no accepted artifact are recorded as absent.
        """
        wanted = frozenset(kinds)
        for kind in wanted:
            if not KIND_RE.match(kind):
                raise ConfigError(f"invalid artifact kind: {sanitize_for_terminal(kind, 60)!r}")
        with self._lock:
            candidates = [a for k in sorted(wanted) for a in self.active(k)]
        # ``accept_producer`` runs outside the lock (it may be arbitrary engine code)
        return PinnedReader(self, wanted, [a for a in candidates if accept_producer(a.producer)])

    def reader(self) -> ArtifactReader:
        """Unscoped, live, read-only view (no write methods are reachable from it)."""
        return _LiveReader(self)


class _LiveReader:
    """Read-only facade over a store: gates and agents never receive ``put``/``activate``."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        return self._store.latest(kind, key)

    def active(self, kind: str) -> list[Artifact]:
        return self._store.active(kind)

    def read_text(self, artifact: Artifact) -> str:
        return self._store.read_text(artifact)

    def read_bytes(self, artifact: Artifact) -> bytes:
        return self._store.read_bytes(artifact)

    def read_json(self, artifact: Artifact) -> Any:
        return self._store.read_json(artifact)


class PinnedReader:
    """Immutable, node-scoped :class:`ArtifactReader` (engine-semantics §4.1).

    Serves only the artifacts captured at construction and only for the declared ``kinds``;
    anything else raises :class:`UndeclaredInputError`. Content is still verified against the
    store's hashes on every read.
    """

    def __init__(
        self,
        store: ArtifactStore,
        kinds: Iterable[str],
        artifacts: Iterable[Artifact],
        *,
        extra: Iterable[Artifact] = (),
    ) -> None:
        self._store = store
        self._declared = frozenset(kinds)
        self._pinned = tuple(sorted(artifacts, key=_sort_key))
        self._extra = tuple(sorted(extra, key=_sort_key))
        if not {a.kind for a in self._pinned} <= self._declared:
            raise ConfigError("pinned artifacts must belong to declared kinds")
        self._kinds = self._declared | {a.kind for a in self._extra}
        items: dict[_Key, Artifact] = {}
        for art in (*self._pinned, *self._extra):  # extras (staged) override pinned ones
            items[(art.kind, art.key)] = art
        self._items = items
        self._ids = frozenset(a.id for a in items.values())

    @property
    def kinds(self) -> frozenset[str]:
        """Every kind this reader serves (declared inputs plus the kinds of any extras)."""
        return self._kinds

    @property
    def absent_kinds(self) -> tuple[str, ...]:
        """Declared kinds for which the snapshot found no accepted ACTIVE artifact."""
        present = {a.kind for a in self._pinned}
        return tuple(sorted(self._declared - present))

    def refs(self) -> tuple[ArtifactRef, ...]:
        """Refs of the dispatch-time snapshot (extras are never inputs of their own producer)."""
        return tuple(a.ref for a in self._pinned)

    def with_extra(self, artifacts: Iterable[Artifact]) -> PinnedReader:
        """A new reader that additionally serves the attempt's STAGED (or restored ACTIVE)
        artifacts, e.g. for exit gates. Each must match the store's own record."""
        resolved: list[Artifact] = []
        for art in artifacts:
            rec = self._store.resolve(art)
            if rec.status not in (ArtifactStatus.STAGED, ArtifactStatus.ACTIVE):
                raise StateError(f"{rec.id} is {rec.status.value} and cannot be served to a gate")
            resolved.append(rec)
        return PinnedReader(
            self._store, self._declared, self._pinned, extra=(*self._extra, *resolved)
        )

    def _require(self, kind: str) -> None:
        if kind not in self._kinds:
            raise UndeclaredInputError(
                f"artifact kind {sanitize_for_terminal(kind, 60)!r} is not a declared input"
            )

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        self._require(kind)
        return self._items.get((kind, key))

    def active(self, kind: str) -> list[Artifact]:
        self._require(kind)
        return sorted((a for (k, _), a in self._items.items() if k == kind), key=lambda a: a.key)

    def _served(self, artifact: Artifact) -> Artifact:
        self._require(artifact.kind)
        if artifact.id not in self._ids:
            raise UndeclaredInputError(
                f"artifact {sanitize_for_terminal(artifact.id, 80)} is not part of this snapshot"
            )
        return self._items[(artifact.kind, artifact.key)]

    def read_bytes(self, artifact: Artifact) -> bytes:
        return self._store.read_bytes(self._served(artifact))

    def read_text(self, artifact: Artifact) -> str:
        return self._store.read_text(self._served(artifact))

    def read_json(self, artifact: Artifact) -> Any:
        return self._store.read_json(self._served(artifact))
