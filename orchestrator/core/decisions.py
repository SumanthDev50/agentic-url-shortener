"""Append-only decision log: the *why* of a run (design, assumptions, approvals, retries...).

``decisions.jsonl`` holds one canonical-JSON ``Decision`` per line, ids ``D-0001``, ``D-0002``...
Together with ``Artifact.inputs`` it forms the lineage graph (``core/lineage.py``).

* **Append-only and fsynced.** A decision is a governance fact; it is durable before the
  ``DECISION_RECORDED`` audit event (which carries its hash) is emitted. The audit chain, not
  this file, is the integrity authority, so loading verifies *structure* only: ids are
  sequential, parents point at earlier decisions (the graph is acyclic by construction),
  every field is within its bounds.
* **Text is untrusted.** Summaries and rationales may come from agents or requirement text.
  They are stripped of terminal/ANSI/bidi control characters and length-capped *when
  recorded*, so no consumer of the file (``cat``, a report, an LLM prompt) ever sees an escape
  sequence; renderers still escape for their own medium.
* **Torn tail.** A crash can leave a final line without its newline. Because the audit event
  is emitted only after the append returned, such a line was never announced; it is dropped
  (and ``repaired_tail`` is set so the engine can record it). Any *terminated* line that does
  not parse is corruption and raises :class:`IntegrityError`.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from orchestrator.core.artifacts import freeze_json, utc_now
from orchestrator.core.errors import ConfigError, IntegrityError, StateError
from orchestrator.core.fsutil import append_line, atomic_write_bytes, read_bytes_retry
from orchestrator.core.paths import NODE_ID_RE, RUN_ID_RE
from orchestrator.core.sanitize import sanitize_block, sanitize_for_terminal, strip_controls
from orchestrator.core.serde import canonical_json, hash_jsonable, loads_strict
from orchestrator.core.types import (
    Actor,
    ActorType,
    ArtifactRef,
    AuditSink,
    Clock,
    Decision,
    DecisionKind,
    EventType,
)

MAX_DECISIONS = 20_000
MAX_SUMMARY_CHARS = 500
MAX_RATIONALE_CHARS = 4000
MAX_ALTERNATIVES = 20
MAX_REFS = 50
MAX_PARENTS = 50
MAX_ARTIFACT_REFS = 200
MAX_DATA_BYTES = 16 * 1024
MAX_RECORD_BYTES = 256 * 1024

DECISION_ID_RE = re.compile(r"^D-\d{4,}\Z")
_ACTOR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,99}\Z")
_REF_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z"
)  # requirement/task/risk/ADR/question ids


def _format_id(number: int) -> str:
    return f"D-{number:04d}"


def _items(name: str, value: Any) -> tuple[Any, ...]:
    """An order-preserving, de-duplicated tuple from a real sequence (never a bare string)."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a sequence")
    return tuple(dict.fromkeys(value))


def _canonical_data(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """JSON round-tripped (keys are strings, tuples are lists) and immutable."""
    canonical = json.loads(canonical_json(data))
    if not isinstance(canonical, dict):
        raise ValueError("data must be a mapping")
    return freeze_json(canonical)


def _check_refs(name: str, refs: Sequence[Any], limit: int) -> None:
    if len(refs) > limit:
        raise ValueError(f"too many {name} ({len(refs)} > {limit})")
    for ref in refs:
        if not (
            isinstance(ref, ArtifactRef)
            and all(
                isinstance(x, str) for x in (ref.artifact_id, ref.kind, ref.key, ref.content_hash)
            )
        ):
            raise ValueError(f"{name} must be ArtifactRef objects")


def _check_text(name: str, text: Any, limit: int, *, multiline: bool = False) -> None:
    if not isinstance(text, str) or len(text) > limit:
        raise ValueError(f"{name} must be a string of at most {limit} characters")
    if strip_controls(text, keep_newlines=multiline) != text:
        raise ValueError(f"{name} contains control characters")


def _check_decision(d: Decision) -> None:
    """Structural validation of a *final* decision (used on record and on load)."""
    if not (isinstance(d.actor, Actor) and isinstance(d.actor.type, ActorType)):
        raise ValueError("invalid actor")
    if not isinstance(d.actor.id, str) or not _ACTOR_ID_RE.match(d.actor.id):
        raise ValueError("invalid actor id")
    if d.node_id is not None and (
        not isinstance(d.node_id, str) or not NODE_ID_RE.match(d.node_id)
    ):
        raise ValueError("invalid node id")
    _check_text("summary", d.summary, MAX_SUMMARY_CHARS)
    if not d.summary:
        raise ValueError("summary must not be empty")
    _check_text("rationale", d.rationale, MAX_RATIONALE_CHARS, multiline=True)
    _check_text("outcome", d.outcome, MAX_SUMMARY_CHARS)
    if len(d.alternatives) > MAX_ALTERNATIVES:
        raise ValueError(f"too many alternatives ({len(d.alternatives)} > {MAX_ALTERNATIVES})")
    for alt in d.alternatives:
        _check_text("alternative", alt, MAX_SUMMARY_CHARS)
    _check_refs("inputs", d.inputs, MAX_ARTIFACT_REFS)
    _check_refs("outputs", d.outputs, MAX_ARTIFACT_REFS)
    if len(d.refs) > MAX_REFS or not all(isinstance(r, str) and _REF_RE.match(r) for r in d.refs):
        raise ValueError("invalid refs")
    if len(d.parents) > MAX_PARENTS or not all(
        isinstance(p, str) and DECISION_ID_RE.match(p) for p in d.parents
    ):
        raise ValueError("invalid parents")
    if len(canonical_json(d.data)) > MAX_DATA_BYTES:
        raise ValueError(f"data exceeds {MAX_DATA_BYTES} bytes")


class DecisionLog:
    """Thread-safe, append-only decision log for one run."""

    def __init__(
        self,
        path: Path | str,
        *,
        run_id: str,
        clock: Clock,
        audit: AuditSink | None = None,
    ) -> None:
        if not RUN_ID_RE.match(run_id):
            raise ConfigError(f"invalid run id: {sanitize_for_terminal(run_id, 60)!r}")
        self.path = Path(path)
        self.run_id = run_id
        self.repaired_tail = False  # True if a torn final line was dropped while opening
        self._clock = clock
        self._audit = audit
        self._lock = threading.RLock()
        self._decisions: list[Decision] = []
        self._by_id: dict[str, Decision] = {}
        self._by_output: dict[ArtifactRef, list[Decision]] = {}
        self._load()

    # ------------------------------------------------------------------ loading

    def _load(self) -> None:
        try:
            raw = read_bytes_retry(self.path)
        except FileNotFoundError:
            return
        *lines, tail = raw.split(b"\n")  # only "\n" separates records (never splitlines)
        for number, line in enumerate(lines, 1):
            self._adopt(self._parse_line(line, number))
        if tail:
            # a final line without its terminator: the append never completed, nothing announced it
            atomic_write_bytes(self.path, raw[: len(raw) - len(tail)])
            self.repaired_tail = True

    def _parse_line(self, line: bytes, number: int) -> Decision:
        try:
            decision = Decision.from_dict(loads_strict(line, max_bytes=MAX_RECORD_BYTES))
            decision = dataclasses.replace(decision, data=freeze_json(decision.data))
            _check_decision(decision)
        except (ValueError, LookupError, TypeError, AttributeError, OverflowError) as exc:
            raise IntegrityError(f"decision log line {number} is corrupt: {exc}") from exc
        if decision.id != _format_id(number):
            raise IntegrityError(
                f"decision log line {number} has id {decision.id!r}, expected sequence"
            )
        if decision.run_id != self.run_id:
            raise IntegrityError(f"decision log line {number} belongs to another run")
        unknown = [p for p in decision.parents if p not in self._by_id]
        if unknown:
            raise IntegrityError(f"decision {decision.id} lists unknown parent(s) {unknown}")
        return decision

    def _adopt(self, decision: Decision) -> None:
        self._decisions.append(decision)
        self._by_id[decision.id] = decision
        for ref in decision.outputs:
            self._by_output.setdefault(ref, []).append(decision)

    # ------------------------------------------------------------------ writing

    def record(
        self,
        kind: DecisionKind,
        summary: str,
        *,
        actor: Actor,
        node_id: str | None = None,
        rationale: str = "",
        inputs: Sequence[ArtifactRef] = (),
        outputs: Sequence[ArtifactRef] = (),
        alternatives: Sequence[str] = (),
        outcome: str = "",
        parents: Sequence[str] = (),
        refs: Sequence[str] = (),
        data: Mapping[str, Any] | None = None,
    ) -> Decision:
        """Append a decision and announce it (``DECISION_RECORDED``, carrying its hash).

        Text fields are sanitised and capped, everything else is validated; a violation raises
        :class:`ConfigError` before anything is written. ``parents`` must already exist
        (:class:`StateError` otherwise): a dangling parent would corrupt the lineage graph.
        """
        try:
            fields: dict[str, Any] = {
                "kind": DecisionKind(kind),
                "summary": sanitize_for_terminal(summary, MAX_SUMMARY_CHARS),
                "rationale": sanitize_block(rationale, MAX_RATIONALE_CHARS),
                "outcome": sanitize_for_terminal(outcome, MAX_SUMMARY_CHARS),
                "alternatives": tuple(
                    sanitize_for_terminal(a, MAX_SUMMARY_CHARS)
                    for a in _items("alternatives", alternatives)
                ),
                "inputs": _items("inputs", inputs),
                "outputs": _items("outputs", outputs),
                "parents": _items("parents", parents),
                "refs": _items("refs", refs),
                "data": _canonical_data(data or {}),
            }
        except (ValueError, TypeError, RecursionError) as exc:
            raise ConfigError(f"invalid decision: {exc}") from exc
        with self._lock:
            if len(self._decisions) >= MAX_DECISIONS:
                raise StateError(f"decision log is full ({MAX_DECISIONS} decisions)")
            decision = Decision(
                id=_format_id(len(self._decisions) + 1),
                run_id=self.run_id,
                ts=utc_now(self._clock),
                actor=actor,
                node_id=node_id,
                **fields,
            )
            try:
                _check_decision(decision)
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"invalid decision: {exc}") from exc
            missing = [p for p in decision.parents if p not in self._by_id]
            if missing:
                raise StateError(f"unknown parent decision(s): {missing}")
            append_line(self.path, canonical_json(decision), fsync=True)
            self._adopt(decision)
            if self._audit is not None:
                self._audit(
                    EventType.DECISION_RECORDED,
                    node_id=decision.node_id,
                    attempt=None,
                    actor=decision.actor,
                    data={
                        "decision_id": decision.id,
                        "kind": decision.kind.value,
                        "decision_hash": hash_jsonable(decision),
                    },
                )
            return decision

    # ------------------------------------------------------------------ querying

    def __len__(self) -> int:
        with self._lock:
            return len(self._decisions)

    def __contains__(self, decision_id: object) -> bool:
        with self._lock:
            return isinstance(decision_id, str) and decision_id in self._by_id

    def all(self) -> tuple[Decision, ...]:
        """Every decision in recording order."""
        with self._lock:
            return tuple(self._decisions)

    def get(self, decision_id: str) -> Decision:
        """The decision with this id; unknown ids raise :class:`StateError`."""
        with self._lock:
            found = self._by_id.get(decision_id) if isinstance(decision_id, str) else None
        if found is None:
            raise StateError(f"unknown decision: {sanitize_for_terminal(decision_id, 80)}")
        return found

    def for_node(self, node_id: str) -> tuple[Decision, ...]:
        with self._lock:
            return tuple(d for d in self._decisions if d.node_id == node_id)

    def of_kind(self, kind: DecisionKind) -> tuple[Decision, ...]:
        with self._lock:
            return tuple(d for d in self._decisions if d.kind == kind)

    def decisions_for(self, ref: ArtifactRef) -> tuple[Decision, ...]:
        """Decisions whose ``outputs`` contain ``ref`` (they explain that artifact)."""
        with self._lock:
            return tuple(self._by_output.get(ref, ()))
