"""Shared vocabulary of the orchestration layer (the *contract* between modules).

Every module (engine, agents, policies, tools, CLI) speaks in these types.  They
are deliberately small, immutable where possible and JSON round-trippable, so
that a run can be persisted, audited, resumed and replayed.

Design rules
------------
* Pure data + protocols.  No I/O, no threads (except ``CancelToken``), no imports
  from other ``orchestrator`` sub-packages except ``core.errors`` / ``core.serde``
  / ``core.paths``.
* Everything crossing a persistence boundary offers ``to_dict`` / ``from_dict``;
  ``hash_jsonable(x) == hash_jsonable(x.to_dict())`` for every serialisable type.
* Hostile input is rejected at construction: identifiers, paths and bounds are
  validated in ``__post_init__`` (``ConfigError`` / ``PathViolation``).
* Enums are ``StrEnum`` (stable string values in JSON) except ``Severity`` and
  ``RiskLevel`` (``IntEnum``: comparable; serialised by *name*).
* Objects held in sets/dicts are keyed by id or ref, never by object identity.
"""

from __future__ import annotations

import enum
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Protocol, runtime_checkable

from orchestrator.core.errors import AgentCancelled, ConfigError, StateError
from orchestrator.core.paths import (
    FILENAME_RE,
    KEY_RE,
    KIND_RE,
    NODE_ID_RE,
    casefold_key,
    validate_globs,
    validate_rel_path,
)
from orchestrator.core.serde import canonical_json, hash_jsonable, parse_datetime, to_jsonable

# =============================================================================
# Constants
# =============================================================================

EXTERNAL_PRODUCER = "external"  # Artifact.producer for human/CLI-supplied inputs
MAX_FILE_BYTES = 1 * 1024 * 1024
MAX_CHANGESET_OPS = 500

K_REQUEST = "request"
K_CLARIFICATION_ANSWERS = "clarification_answers"
CHANGESET_KINDS = frozenset({"changeset", "integrated_changeset", "docs_changeset"})
KEYED_KINDS = frozenset({"changeset"})  # produced by many nodes => key is ALWAYS the node id
EXTERNAL_KINDS = frozenset({K_REQUEST, K_CLARIFICATION_ANSWERS})

# =============================================================================
# Enumerations
# =============================================================================


class Severity(enum.IntEnum):
    """Severity of a policy finding. Higher is worse."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RiskLevel(enum.IntEnum):
    """Impact classification of an action / change. Drives approval requirements."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class PolicyCategory(enum.StrEnum):
    SECURITY = "security"
    COMPLIANCE = "compliance"
    CHANGE_CONTROL = "change_control"
    QUALITY = "quality"


class Enforcement(enum.StrEnum):
    """What the engine does with the findings of a policy evaluation."""

    ALLOW = "allow"
    WARN = "warn"  # recorded, run continues
    REQUIRE_APPROVAL = "require_approval"  # human waiver needed (MEDIUM by default)
    DENY = "deny"  # output rejected -> retry with feedback / fallback (HIGH)
    SAFE_STOP = "safe_stop"  # halt the whole run (CRITICAL)


DEFAULT_ENFORCEMENT: Mapping[Severity, Enforcement] = {
    Severity.INFO: Enforcement.WARN,
    Severity.LOW: Enforcement.WARN,
    Severity.MEDIUM: Enforcement.REQUIRE_APPROVAL,
    Severity.HIGH: Enforcement.DENY,
    Severity.CRITICAL: Enforcement.SAFE_STOP,
}


class NodeStatus(enum.StrEnum):
    PENDING = "pending"  # waiting for dependencies
    READY = "ready"  # dependencies satisfied; waiting to be dispatched (or in retry backoff)
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_INPUT = "awaiting_input"  # clarification requested from a human
    SUCCEEDED = "succeeded"
    FAILED = "failed"  # retries + fallback exhausted
    ROLLED_BACK = "rolled_back"  # side effects compensated (after FAILED or SUCCEEDED)
    SKIPPED = "skipped"  # conditional branch not taken
    STALE = "stale"  # was SUCCEEDED, an upstream artifact changed (pending revalidation)
    BLOCKED = "blocked"  # an upstream node failed / was rolled back / cancelled
    CANCELLED = "cancelled"  # run was safe-stopped / killed
    REMOVED = "removed"  # removed from the plan by a re-plan (tombstone)

    @property
    def is_settled(self) -> bool:
        """No further work will happen *unless* a re-plan / override re-opens the node.

        NOTE: settled != final. ``SUCCEEDED`` can become ``STALE``; ``FAILED`` /
        ``BLOCKED`` / ``CANCELLED`` can return to ``PENDING``.
        """
        return self in _SETTLED_NODE_STATUSES

    @property
    def satisfies_dependency(self) -> bool:
        """Downstream nodes may start when every dependency is in one of these."""
        return self in (NodeStatus.SUCCEEDED, NodeStatus.SKIPPED)

    @property
    def is_awaiting_human(self) -> bool:
        return self in (NodeStatus.AWAITING_APPROVAL, NodeStatus.AWAITING_INPUT)


_SETTLED_NODE_STATUSES = frozenset(
    {
        NodeStatus.SUCCEEDED,
        NodeStatus.FAILED,
        NodeStatus.ROLLED_BACK,
        NodeStatus.SKIPPED,
        NodeStatus.BLOCKED,
        NodeStatus.CANCELLED,
        NodeStatus.REMOVED,
    }
)

_NS = NodeStatus
ALLOWED_TRANSITIONS: Mapping[NodeStatus, frozenset[NodeStatus]] = {
    _NS.PENDING: frozenset({_NS.READY, _NS.SKIPPED, _NS.BLOCKED, _NS.REMOVED, _NS.CANCELLED}),
    _NS.READY: frozenset(
        {_NS.RUNNING, _NS.AWAITING_APPROVAL, _NS.SKIPPED, _NS.PENDING, _NS.FAILED, _NS.REMOVED, _NS.CANCELLED}
    ),
    _NS.RUNNING: frozenset(
        {
            _NS.AWAITING_APPROVAL,
            _NS.AWAITING_INPUT,
            _NS.SUCCEEDED,
            _NS.READY,  # retry / fallback (not_before backoff)
            _NS.FAILED,
            _NS.PENDING,  # invalidated mid-flight
            _NS.CANCELLED,
        }
    ),
    _NS.AWAITING_APPROVAL: frozenset(
        {_NS.SUCCEEDED, _NS.RUNNING, _NS.READY, _NS.FAILED, _NS.PENDING, _NS.CANCELLED}
    ),
    _NS.AWAITING_INPUT: frozenset({_NS.READY, _NS.PENDING, _NS.CANCELLED}),
    _NS.SUCCEEDED: frozenset({_NS.STALE, _NS.ROLLED_BACK, _NS.REMOVED}),
    _NS.STALE: frozenset({_NS.SUCCEEDED, _NS.READY, _NS.PENDING, _NS.REMOVED, _NS.CANCELLED}),
    _NS.FAILED: frozenset({_NS.ROLLED_BACK, _NS.PENDING, _NS.REMOVED}),
    _NS.ROLLED_BACK: frozenset({_NS.PENDING, _NS.REMOVED}),
    _NS.SKIPPED: frozenset({_NS.PENDING, _NS.REMOVED}),
    _NS.BLOCKED: frozenset({_NS.PENDING, _NS.REMOVED, _NS.CANCELLED}),
    _NS.CANCELLED: frozenset({_NS.PENDING}),
    _NS.REMOVED: frozenset({_NS.PENDING}),
}


def assert_transition(old: NodeStatus, new: NodeStatus) -> None:
    """Raise ``StateError`` for an illegal node status transition."""
    if new not in ALLOWED_TRANSITIONS[old]:
        raise StateError(f"illegal node transition {old.value} -> {new.value}")


class RunStatus(enum.StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"  # approvals and/or clarifications pending
    SAFE_STOPPING = "safe_stopping"  # journaled safe-stop in progress (crash-resumable)
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SAFE_STOPPED = "safe_stopped"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.SAFE_STOPPED)


class FailureKind(enum.StrEnum):
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    GATE_REJECTED = "gate_rejected"  # an exit gate failed
    POLICY_VIOLATION = "policy_violation"  # HIGH finding / denied tool call
    APPROVAL_REJECTED = "approval_rejected"  # a human rejected the output
    AUTONOMY_VIOLATION = "autonomy_violation"  # CRITICAL, never retried
    CANCELLED = "cancelled"
    FATAL = "fatal"  # unexpected exception; not retried on the same agent

    @classmethod
    def parse(cls, value: str) -> FailureKind:
        """Unknown strings map to FATAL (never raises inside the scheduler)."""
        try:
            return cls(value)
        except ValueError:
            return cls.FATAL


class Epoch(enum.StrEnum):
    PRIMARY = "primary"
    FALLBACK = "fallback"


class Stage(enum.StrEnum):
    """SDLC lifecycle stage a node belongs to (used for reporting and metrics)."""

    INTAKE = "intake"
    REQUIREMENTS = "requirements"
    ANALYSIS = "analysis"
    DESIGN = "design"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    RELEASE = "release"
    SUMMARY = "summary"


class GatePhase(enum.StrEnum):
    ENTRY = "entry"
    EXIT = "exit"


class GateDecision(enum.StrEnum):
    """Composition (engine): ALL gates are evaluated (no short-circuit except SAFE_STOP).
    Precedence: SAFE_STOP > FAIL > AWAIT_INPUT > REQUIRE_APPROVAL > PASS. Findings are unioned.
    An entry-gate FAIL is **not retried** (identical inputs): the node fails fast."""

    PASS = "pass"
    FAIL = "fail"
    REQUIRE_APPROVAL = "require_approval"
    AWAIT_INPUT = "await_input"
    SAFE_STOP = "safe_stop"


class ApprovalPosition(enum.StrEnum):
    BEFORE = "before"  # human approves *starting* the node (entry checkpoint)
    AFTER = "after"  # human approves the node's *output* (exit checkpoint)


class ApprovalPurpose(enum.StrEnum):
    NODE_START = "node_start"
    NODE_OUTPUT = "node_output"
    WAIVER = "waiver"  # human waives MEDIUM policy findings
    RELEASE = "release"


class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"  # >= required distinct approvers; any single rejection vetoes instead
    REJECTED = "rejected"
    INVALIDATED = "invalidated"  # node fingerprint changed after the request was raised
    CANCELLED = "cancelled"  # node/run cancelled (safe-stop, removal)


class ActorType(enum.StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"  # the engine itself
    POLICY = "policy"  # automated policy decision


class DecisionKind(enum.StrEnum):
    DESIGN = "design"
    ASSUMPTION = "assumption"
    SCOPE = "scope"
    CLARIFICATION = "clarification"
    APPROVAL = "approval"
    REJECTION = "rejection"
    WAIVER = "waiver"
    POLICY = "policy"
    RETRY = "retry"
    FALLBACK = "fallback"
    REMEDIATION = "remediation"
    ROLLBACK = "rollback"
    REPLAN = "replan"
    SAFE_STOP = "safe_stop"
    RELEASE = "release"


class ArtifactStatus(enum.StrEnum):
    STAGED = "staged"  # produced by an attempt; awaiting exit gates / approval; invisible downstream
    ACTIVE = "active"  # published
    SUPERSEDED = "superseded"  # replaced by a newer version
    QUARANTINED = "quarantined"  # produced by a rejected / discarded / rolled-back attempt


class FileOpKind(enum.StrEnum):
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"


class SubjectKind(enum.StrEnum):
    CHANGESET = "changeset"
    ARTIFACT = "artifact"
    ACTION = "action"
    PLAN = "plan"
    RUN = "run"


class ToolAction(enum.StrEnum):
    READ = "read_file"
    WRITE = "write_file"
    DELETE = "delete_file"
    RUN_TESTS = "run_tests"
    RUN_PYTHON = "run_python"


class EventType(enum.StrEnum):
    """Every state transition and governance action emits exactly one of these."""

    RUN_CREATED = "run.created"
    RUN_STARTED = "run.started"
    RUN_RESUMED = "run.resumed"
    RUN_WAITING = "run.waiting"
    RUN_SUCCEEDED = "run.succeeded"
    RUN_FAILED = "run.failed"
    RUN_SAFE_STOPPED = "run.safe_stopped"
    RUN_SEALED = "run.sealed"

    NODE_READY = "node.ready"
    NODE_SKIPPED = "node.skipped"
    NODE_STARTED = "node.started"  # exactly once per ATTEMPT (attempt = global per-node number)
    NODE_ATTEMPT_FAILED = "node.attempt_failed"
    NODE_ATTEMPT_DISCARDED = "node.attempt_discarded"  # late result of a cancelled/stale attempt
    NODE_RETRY_SCHEDULED = "node.retry_scheduled"
    NODE_FALLBACK_ACTIVATED = "node.fallback_activated"
    NODE_CACHE_HIT = "node.cache_hit"
    NODE_PUBLISHING = "node.publishing"  # journal entry written BEFORE publish mutations
    NODE_SUCCEEDED = "node.succeeded"
    NODE_FAILED = "node.failed"
    NODE_BLOCKED = "node.blocked"
    NODE_STALE = "node.stale"
    NODE_RESET = "node.reset"
    NODE_REMOVED = "node.removed"
    NODE_CANCELLED = "node.cancelled"
    NODE_ROLLED_BACK = "node.rolled_back"
    NODE_AWAITING_APPROVAL = "node.awaiting_approval"
    NODE_AWAITING_INPUT = "node.awaiting_input"
    NODE_ZOMBIE = "node.zombie"

    GATE_EVALUATED = "gate.evaluated"
    POLICY_EVALUATED = "policy.evaluated"  # positive evidence that rules ran (rule ids + subject hash)
    POLICY_VIOLATION = "policy.violation"
    POLICY_WAIVED = "policy.waived"
    INPUT_SCREENED = "input.screened"

    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_VOTE = "approval.vote"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_AUTO = "approval.auto"
    APPROVAL_INVALIDATED = "approval.invalidated"
    APPROVAL_CANCELLED = "approval.cancelled"
    APPROVAL_REFUSED = "approval.refused"  # self-approval, duplicate, agent identity, unknown, stale

    CLARIFICATION_REQUESTED = "clarification.requested"
    CLARIFICATION_ANSWERED = "clarification.answered"

    INPUT_CHANGED = "input.changed"
    PLAN_EXPANDED = "plan.expanded"
    PLAN_REPLANNED = "plan.replanned"
    REMEDIATION_ROUTED = "remediation.routed"

    ARTIFACT_CREATED = "artifact.created"
    ARTIFACT_PUBLISHED = "artifact.published"
    ARTIFACT_UNCHANGED = "artifact.unchanged"
    ARTIFACT_SUPERSEDED = "artifact.superseded"
    ARTIFACT_QUARANTINED = "artifact.quarantined"
    DECISION_RECORDED = "decision.recorded"

    TOOL_CALL = "tool.call"
    TOOL_DENIED = "tool.denied"
    AGENT_NOTE = "agent.note"
    EFFECT_STARTED = "effect.started"
    EFFECT_COMPLETED = "effect.completed"

    COMMAND_QUEUED = "command.queued"
    COMMAND_APPLIED = "command.applied"

    SAFE_STOP_TRIGGERED = "safe_stop.triggered"
    SAFE_STOP_STEP = "safe_stop.step"
    SAFE_STOP_OVERRIDDEN = "safe_stop.overridden"
    KILL_SWITCH_DETECTED = "kill_switch.detected"
    ROLLBACK_STARTED = "rollback.started"
    ROLLBACK_COMPLETED = "rollback.completed"
    ROLLBACK_FAILED = "rollback.failed"
    AUDIT_TAIL_REPAIRED = "audit.tail_repaired"


# Required keys of ``AuditEvent.data`` per event type (validated by
# tests/test_event_contract.py against real runs). Events not listed have free-form data.
# Span convention (derived inside AuditLog.append from node_id/attempt, never passed by callers):
#   run span  "run";  node span  "n:<node_id>";  attempt span  "n:<node_id>#<attempt>".
EVENT_DATA: Mapping[EventType, tuple[str, ...]] = {
    EventType.RUN_CREATED: ("scenario", "created_by", "graph_hash", "baseline_manifest_hash", "assurance"),
    EventType.RUN_WAITING: ("reason", "approvals", "questions"),
    EventType.NODE_STARTED: ("agent_id", "epoch", "stage", "generation"),
    EventType.NODE_ATTEMPT_FAILED: ("kind", "message", "epoch", "retryable", "rule_ids"),
    EventType.NODE_ATTEMPT_DISCARDED: ("generation", "reason"),
    EventType.NODE_RETRY_SCHEDULED: ("delay_s", "not_before"),
    EventType.NODE_FALLBACK_ACTIVATED: ("from_agent", "to_agent"),
    EventType.NODE_CACHE_HIT: ("fingerprint",),
    EventType.NODE_PUBLISHING: ("staged", "expansion_hash"),
    EventType.NODE_SUCCEEDED: ("stage", "outputs", "cache_hit"),
    EventType.NODE_FAILED: ("kind", "message"),
    EventType.GATE_EVALUATED: ("gate", "phase", "decision", "rule_ids"),
    EventType.POLICY_EVALUATED: ("subject", "subject_hash", "rules_evaluated", "findings"),
    EventType.POLICY_VIOLATION: ("rule_id", "severity", "enforcement", "location"),
    EventType.APPROVAL_REQUESTED: ("approval_id", "purpose", "risk", "required", "node_fingerprint"),
    EventType.APPROVAL_VOTE: ("approval_id", "approver", "approved", "channel", "evidence_hash"),
    EventType.APPROVAL_REFUSED: ("approval_id", "reason"),
    EventType.INPUT_CHANGED: ("kind", "key", "old_ref", "new_ref", "by"),
    EventType.PLAN_REPLANNED: ("decision_id", "added", "removed", "retained", "invalidated"),
    EventType.ARTIFACT_CREATED: ("artifact_id", "content_hash", "size"),
    EventType.DECISION_RECORDED: ("decision_id", "kind", "decision_hash"),
    EventType.ROLLBACK_STARTED: ("level",),
    EventType.ROLLBACK_COMPLETED: ("level",),
    EventType.SAFE_STOP_TRIGGERED: ("trigger", "reason"),
    EventType.SAFE_STOP_STEP: ("step",),
}


# =============================================================================
# Identity, findings, risk
# =============================================================================


@dataclass(frozen=True)
class Actor:
    """Who did something. ``label`` (``type:id``) is what appears in logs."""

    type: ActorType
    id: str

    @property
    def label(self) -> str:
        return f"{self.type.value}:{self.id}"

    @classmethod
    def human(cls, id: str) -> Actor:
        return cls(ActorType.HUMAN, id)

    @classmethod
    def agent(cls, id: str) -> Actor:
        return cls(ActorType.AGENT, id)

    @classmethod
    def system(cls, id: str = "engine") -> Actor:
        return cls(ActorType.SYSTEM, id)

    @classmethod
    def policy(cls, id: str) -> Actor:
        return cls(ActorType.POLICY, id)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "id": self.id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Actor:
        return cls(ActorType(d["type"]), d["id"])


def _sev(value: Any) -> Severity:
    return Severity[value] if isinstance(value, str) else Severity(value)


@dataclass(frozen=True)
class Finding:
    """A single policy / gate / validation observation.

    ``evidence`` must already be short and redacted by whoever creates the
    finding; the PolicyEngine additionally post-processes every finding.
    """

    rule_id: str
    category: PolicyCategory
    severity: Severity
    message: str
    location: str | None = None  # "path:line" or artifact id
    evidence: str | None = None
    remediation: str | None = None  # engine/rule-authored fix hint (safe to feed back to agents)

    def key(self) -> str:
        """Stable identity used to bind waivers and de-duplicate feedback."""
        return hash_jsonable([self.rule_id, self.location, self.evidence])[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "severity": self.severity.name,
            "message": self.message,
            "location": self.location,
            "evidence": self.evidence,
            "remediation": self.remediation,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Finding:
        return cls(
            rule_id=d["rule_id"],
            category=PolicyCategory(d["category"]),
            severity=_sev(d["severity"]),
            message=d["message"],
            location=d.get("location"),
            evidence=d.get("evidence"),
            remediation=d.get("remediation"),
        )


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    reasons: tuple[str, ...] = ()
    signals: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level.name, "reasons": list(self.reasons), "signals": to_jsonable(self.signals)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RiskAssessment:
        return cls(RiskLevel[d["level"]], tuple(d.get("reasons", ())), dict(d.get("signals", {})))


# =============================================================================
# Artifacts and change sets
# =============================================================================


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable pointer to one version of one artifact (used for lineage)."""

    artifact_id: str
    kind: str
    key: str
    version: int
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "key": self.key,
            "version": self.version,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ArtifactRef:
        return cls(d["artifact_id"], d["kind"], d.get("key", ""), int(d["version"]), d["content_hash"])


@dataclass
class ArtifactDraft:
    """What an agent hands to the engine; the engine turns it into a STAGED ``Artifact``.

    * ``content``: ``str``/``bytes`` stored verbatim; ``Mapping``/``Sequence`` stored as
      canonical pretty JSON. **Content must be deterministic** — no wall-clock times,
      durations, pids, random ids or absolute paths (put those in ``metadata``, which is
      excluded from the content hash and from fingerprints).
    * ``kind`` must be one of the node's ``produces``. ``key`` is forced by the engine:
      node id for ``KEYED_KINDS`` (e.g. ``changeset``), else ``""`` (agents may leave it blank).
    * ``filename`` is metadata only (never used as a path component).
    * metadata keys starting with ``_`` are reserved and stripped.
    """

    kind: str
    content: str | bytes | Mapping[str, Any] | Sequence[Any]
    key: str = ""
    filename: str | None = None
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not KIND_RE.match(self.kind):
            raise ConfigError(f"invalid artifact kind: {self.kind!r}")
        if not KEY_RE.match(self.key):
            raise ConfigError(f"invalid artifact key: {self.key!r}")
        if self.filename is not None and not FILENAME_RE.match(self.filename):
            raise ConfigError(f"invalid artifact filename: {self.filename!r}")


@dataclass(frozen=True)
class Artifact:
    """A stored, versioned, content-addressed output (or external input)."""

    id: str  # "<kind>[.<key>]@v<version>" (see paths.make_artifact_id)
    kind: str
    key: str
    version: int  # monotonically increasing per (kind, key)
    content_hash: str  # sha256 of stored bytes
    media_type: str
    size: int
    path: str  # relative to the artifact store root, POSIX, built from slugs/hashes only
    created_at: datetime
    producer: str  # node id, or EXTERNAL_PRODUCER for human/CLI inputs
    attempt: int  # global per-node attempt number (0 for external)
    agent_id: str
    inputs: tuple[ArtifactRef, ...] = ()  # the dispatch-time input snapshot, verbatim
    status: ArtifactStatus = ArtifactStatus.ACTIVE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:  # hashable by id (dict fields would break the default)
        return hash(self.id)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(self.id, self.kind, self.key, self.version, self.content_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "key": self.key,
            "version": self.version,
            "content_hash": self.content_hash,
            "media_type": self.media_type,
            "size": self.size,
            "path": self.path,
            "created_at": self.created_at.isoformat(),
            "producer": self.producer,
            "attempt": self.attempt,
            "agent_id": self.agent_id,
            "inputs": [r.to_dict() for r in self.inputs],
            "status": self.status.value,
            "metadata": to_jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Artifact:
        return cls(
            id=d["id"],
            kind=d["kind"],
            key=d.get("key", ""),
            version=int(d["version"]),
            content_hash=d["content_hash"],
            media_type=d["media_type"],
            size=int(d["size"]),
            path=d["path"],
            created_at=parse_datetime(d["created_at"]),
            producer=d["producer"],
            attempt=int(d.get("attempt", 0)),
            agent_id=d.get("agent_id", ""),
            inputs=tuple(ArtifactRef.from_dict(r) for r in d.get("inputs", ())),
            status=ArtifactStatus(d.get("status", "active")),
            metadata=dict(d.get("metadata", {})),
        )


@runtime_checkable
class ArtifactReader(Protocol):
    """Read access to artifacts.

    The engine hands agents/gates a **pinned, node-scoped** reader: it serves only
    kinds the node declared (``consumes`` / ``optional_consumes`` / ``overlay_from`` /
    ``skip_inputs``), only artifacts whose producer is a strict ancestor (or external),
    and only the versions captured at dispatch time (no torn reads). Reading anything
    else raises ``UndeclaredInputError``.
    """

    def latest(self, kind: str, key: str = "") -> Artifact | None: ...

    def active(self, kind: str) -> list[Artifact]:
        """All keys of ``kind``, sorted by key."""
        ...

    def read_text(self, artifact: Artifact) -> str: ...

    def read_bytes(self, artifact: Artifact) -> bytes: ...

    def read_json(self, artifact: Artifact) -> Any: ...


@dataclass(frozen=True)
class FileOp:
    """One text-file edit. Validated at construction (hostile JSON cannot build an escaping op).

    ``base_hash`` = ``fsutil.sha256_text`` of the path's content in the *producing attempt's
    view at the first touch of that path* (``None`` for CREATE). Text only, UTF-8, LF.
    """

    path: str
    kind: FileOpKind
    content: str | None = None  # CREATE / MODIFY: full new text
    base_hash: str | None = None  # MODIFY / DELETE

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", validate_rel_path(self.path))
        if self.kind == FileOpKind.DELETE:
            if self.content is not None:
                raise ConfigError("DELETE must not carry content")
            if not self.base_hash:
                raise ConfigError("DELETE requires base_hash")
        else:
            if not isinstance(self.content, str):
                raise ConfigError(f"{self.kind.value} requires text content")
            if "\x00" in self.content or "\r" in self.content:
                raise ConfigError("content must not contain NUL or CR (normalise to LF)")
            try:
                encoded = self.content.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ConfigError("content is not valid UTF-8") from exc
            if len(encoded) > MAX_FILE_BYTES:
                raise ConfigError(f"content exceeds {MAX_FILE_BYTES} bytes")
            if self.kind == FileOpKind.CREATE and self.base_hash is not None:
                raise ConfigError("CREATE must not carry base_hash")
            if self.kind == FileOpKind.MODIFY and not self.base_hash:
                raise ConfigError("MODIFY requires base_hash")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind.value, "content": self.content, "base_hash": self.base_hash}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FileOp:
        return cls(d["path"], FileOpKind(d["kind"]), d.get("content"), d.get("base_hash"))


@dataclass(frozen=True)
class ChangeSet:
    """An *immutable proposal* of file edits (never applied by the agent that made it).

    Agents never mutate the workspace; they propose ``ChangeSet`` artifacts.
    Rolling back an agent therefore means discarding an artifact, and ``integrate``
    composes change sets deterministically. Only ``promote`` has real side effects.
    Invariant: at most one op per path (case-insensitively unique).
    """

    ops: tuple[FileOp, ...] = ()
    description: str = ""
    author: str = ""  # overwritten by the engine with the real agent id

    def __post_init__(self) -> None:
        if len(self.ops) > MAX_CHANGESET_OPS:
            raise ConfigError(f"change set has too many ops ({len(self.ops)} > {MAX_CHANGESET_OPS})")
        seen: set[str] = set()
        for op in self.ops:
            k = casefold_key(op.path)
            if k in seen:
                raise ConfigError(f"duplicate (case-insensitive) path in change set: {op.path}")
            seen.add(k)

    def paths(self) -> tuple[str, ...]:
        return tuple(op.path for op in self.ops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "author": self.author,
            "ops": [op.to_dict() for op in self.ops],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ChangeSet:
        return cls(tuple(FileOp.from_dict(o) for o in d.get("ops", ())), d.get("description", ""), d.get("author", ""))


@runtime_checkable
class FileView(Protocol):
    """Read-only view of a text file tree (baseline, or baseline + change sets)."""

    def exists(self, path: str) -> bool: ...

    def read_text(self, path: str) -> str:
        """Raises ``FileNotFoundError`` if absent."""
        ...

    def list_files(self, glob: str = "**") -> list[str]:
        """Sorted POSIX relative paths matching ``glob`` (``paths.match_glob`` semantics)."""
        ...


# =============================================================================
# Time and cancellation
# =============================================================================


@runtime_checkable
class Clock(Protocol):
    """Injectable time source. The scheduler never calls ``sleep`` except when it is idle and
    only waiting for a retry back-off (instant on ``FakeClock``)."""

    def now(self) -> datetime:
        """Timezone-aware UTC wall-clock time."""
        ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class CancelToken:
    """Cooperative cancellation shared between the engine and a running attempt."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001 - a cancel callback must never break cancellation
                pass

    def add_callback(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on cancel (e.g. kill a sandbox process tree). Runs immediately if already cancelled."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(fn)
                return
        fn()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise AgentCancelled(self._reason or "attempt cancelled")

    def wait(self, timeout: float) -> bool:
        """Cancel-aware sleep for agents: returns True if cancelled. Agents must use this, not ``time.sleep``."""
        return self._event.wait(timeout)


# =============================================================================
# Workflow definition (nodes, gates, approvals, retries, autonomy)
# =============================================================================


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retries with exponential backoff (no unbounded loops anywhere)."""

    max_attempts: int = 3  # total attempts including the first
    base_delay_s: float = 0.2
    factor: float = 2.0
    max_delay_s: float = 5.0
    retry_on: frozenset[FailureKind] = frozenset(
        {
            FailureKind.TRANSIENT,
            FailureKind.TIMEOUT,
            FailureKind.GATE_REJECTED,
            FailureKind.POLICY_VIOLATION,
            FailureKind.APPROVAL_REJECTED,
        }
    )

    HARD_MAX_ATTEMPTS: ClassVar[int] = 5

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= self.HARD_MAX_ATTEMPTS:
            raise ConfigError(f"retry.max_attempts must be in [1, {self.HARD_MAX_ATTEMPTS}]")
        for name in ("base_delay_s", "factor", "max_delay_s"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                raise ConfigError(f"retry.{name} must be a finite number")
        if self.base_delay_s < 0 or self.max_delay_s < 0 or self.max_delay_s > 60:
            raise ConfigError("retry delays must be in [0, 60] seconds")
        if not 1.0 <= self.factor <= 4.0:
            raise ConfigError("retry.factor must be in [1, 4]")

    def delay_for(self, attempt: int) -> float:
        """Backoff before attempt ``attempt + 1`` (``attempt`` is 1-based)."""
        return min(self.max_delay_s, self.base_delay_s * (self.factor ** max(0, attempt - 1)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "base_delay_s": self.base_delay_s,
            "factor": self.factor,
            "max_delay_s": self.max_delay_s,
            "retry_on": sorted(k.value for k in self.retry_on),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RetryPolicy:
        default = cls()
        return cls(
            max_attempts=int(d.get("max_attempts", default.max_attempts)),
            base_delay_s=float(d.get("base_delay_s", default.base_delay_s)),
            factor=float(d.get("factor", default.factor)),
            max_delay_s=float(d.get("max_delay_s", default.max_delay_s)),
            retry_on=frozenset(FailureKind(k) for k in d["retry_on"]) if "retry_on" in d else default.retry_on,
        )


def _glob_covers(wide: str, narrow: str) -> bool:
    """Conservative subset test: True only if every path matching ``narrow`` matches ``wide``.

    Holds for equal patterns, for the universal ``**`` and for ``dir/**`` against any pattern
    that textually starts with ``dir/`` (it can only match below that directory).
    """
    return wide in ("**", narrow) or (wide.endswith("/**") and narrow.startswith(wide[:-2]))


def _narrow_globs(mine: tuple[str, ...], theirs: tuple[str, ...]) -> tuple[str, ...]:
    """The patterns of ``theirs`` that ``mine`` covers (so narrowing can never grant access)."""
    return tuple(p for p in theirs if any(_glob_covers(m, p) for m in mine))


@dataclass(frozen=True)
class AutonomyProfile:
    """Hard boundary of what an agent may do. Enforced by the ``Toolbox`` AND re-checked by
    SEC-005 on every produced change set. A node may only *narrow* its agent's profile."""

    read_globs: tuple[str, ...] = ("**",)
    write_globs: tuple[str, ...] = ()  # empty => the agent cannot propose file edits
    can_execute: bool = False  # may run tests / sandboxed python
    max_tool_calls: int = 200
    max_runtime_s: float = 120.0
    max_changed_files: int = 40
    max_file_bytes: int = MAX_FILE_BYTES
    max_total_bytes: int = 4 * MAX_FILE_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(self, "read_globs", validate_globs(tuple(self.read_globs)))
        object.__setattr__(self, "write_globs", validate_globs(tuple(self.write_globs)))
        if (
            self.max_tool_calls < 0
            or not 0 < self.max_runtime_s < float("inf")  # also rejects NaN
            or self.max_changed_files < 0
        ):
            raise ConfigError("invalid autonomy bounds")
        if self.max_file_bytes <= 0 or self.max_total_bytes <= 0:
            raise ConfigError("invalid autonomy byte bounds")

    def narrow(self, other: AutonomyProfile | None) -> AutonomyProfile:
        """Intersection with ``other`` (a node-level profile). Never widens.

        Globs cannot be intersected exactly, so ``other``'s patterns are kept only when this
        profile provably covers them (see :func:`_glob_covers`); the default read glob and an
        empty ``write_globs`` mean "no opinion" and keep this profile's patterns.
        """
        if other is None:
            return self
        return AutonomyProfile(
            read_globs=(
                self.read_globs
                if other.read_globs == ("**",)
                else _narrow_globs(self.read_globs, other.read_globs)
            ),
            write_globs=(
                _narrow_globs(self.write_globs, other.write_globs)
                if other.write_globs
                else self.write_globs
            ),
            can_execute=self.can_execute and other.can_execute,
            max_tool_calls=min(self.max_tool_calls, other.max_tool_calls),
            max_runtime_s=min(self.max_runtime_s, other.max_runtime_s),
            max_changed_files=min(self.max_changed_files, other.max_changed_files),
            max_file_bytes=min(self.max_file_bytes, other.max_file_bytes),
            max_total_bytes=min(self.max_total_bytes, other.max_total_bytes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_globs": list(self.read_globs),
            "write_globs": list(self.write_globs),
            "can_execute": self.can_execute,
            "max_tool_calls": self.max_tool_calls,
            "max_runtime_s": self.max_runtime_s,
            "max_changed_files": self.max_changed_files,
            "max_file_bytes": self.max_file_bytes,
            "max_total_bytes": self.max_total_bytes,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AutonomyProfile:
        default = cls()
        return cls(
            read_globs=tuple(d.get("read_globs", default.read_globs)),
            write_globs=tuple(d.get("write_globs", default.write_globs)),
            can_execute=bool(d.get("can_execute", False)),
            max_tool_calls=int(d.get("max_tool_calls", default.max_tool_calls)),
            max_runtime_s=float(d.get("max_runtime_s", default.max_runtime_s)),
            max_changed_files=int(d.get("max_changed_files", default.max_changed_files)),
            max_file_bytes=int(d.get("max_file_bytes", default.max_file_bytes)),
            max_total_bytes=int(d.get("max_total_bytes", default.max_total_bytes)),
        )


def _canonical_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """In-memory form == JSON round-tripped form (tuples become lists, keys become str)."""
    return json.loads(canonical_json(params))


@dataclass(frozen=True)
class GateSpec:
    """Serialisable reference to a gate implementation in the ``GateRegistry``."""

    type: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _canonical_params(self.params))

    def __hash__(self) -> int:
        return hash((self.type, hash_jsonable(self.params)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": to_jsonable(self.params)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GateSpec:
        return cls(d["type"], dict(d.get("params", {})))


@dataclass(frozen=True)
class ApprovalSpec:
    """Human checkpoint attached to a node.

    ``always=True``  -> a human must approve regardless of assessed risk.
    ``always=False`` -> approval only when assessed risk >= ``threshold`` (scaled by the run's
                        autonomy mode); below it the engine records an *automatic policy
                        approval* (still audited, never silent).
    """

    position: ApprovalPosition = ApprovalPosition.AFTER
    always: bool = False
    threshold: RiskLevel = RiskLevel.MEDIUM
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.value,
            "always": self.always,
            "threshold": self.threshold.name,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ApprovalSpec:
        return cls(
            ApprovalPosition(d.get("position", "after")),
            bool(d.get("always", False)),
            RiskLevel[d.get("threshold", "MEDIUM")],
            d.get("reason", ""),
        )


@dataclass(frozen=True)
class RemediationSpec:
    """Feedback edge: when this node's exit gate rejects its output with a quality/policy
    failure, route the findings *back* to the nodes in ``route_to`` (they are re-run with the
    findings as ``ctx.feedback``) and then re-run this node. Bounded by ``max_loops`` and
    ``EngineConfig.max_remediation_loops``. This is the non-linear "test failure -> fix
    implementation" loop of the SDLC. ``route_to`` entries are exact node ids or ``prefix.*``
    patterns (e.g. ``implement.*``) resolved against the live graph (dynamic expansion nodes)."""

    route_to: tuple[str, ...]
    max_loops: int = 2

    def __post_init__(self) -> None:
        if not self.route_to or not 1 <= self.max_loops <= 5:
            raise ConfigError("remediation needs route_to and 1 <= max_loops <= 5")

    def to_dict(self) -> dict[str, Any]:
        return {"route_to": list(self.route_to), "max_loops": self.max_loops}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RemediationSpec:
        return cls(tuple(d["route_to"]), int(d.get("max_loops", 2)))


@dataclass(frozen=True)
class NodeSpec:
    """One vertex of the workflow DAG. Fully serialisable (names, not callables).

    ``agent`` / ``fallback_agent`` / ``skip_if`` / ``compensator`` are keys into the runtime
    registries, so a persisted run can be resumed by a new process.

    View semantics (see docs/engine-semantics.md §6): the Toolbox file view of a node is
    ``baseline + ACTIVE artifacts of kinds in overlay_from produced by strict ancestors``.
    ``overlay_from`` kinds are implicitly part of the node's input snapshot/fingerprint.
    ``skip_inputs`` are the only kinds the ``skip_if`` predicate may read; they are treated as
    consumed for staleness and fingerprinting, and the predicate is re-evaluated when they change.
    """

    id: str
    title: str
    stage: Stage
    agent: str
    depends_on: tuple[str, ...] = ()
    consumes: tuple[str, ...] = ()  # artifact kinds that MUST exist (entry gate)
    optional_consumes: tuple[str, ...] = ()
    overlay_from: tuple[str, ...] = ()  # subset of CHANGESET_KINDS
    skip_inputs: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()  # artifact kinds the node MUST produce (exit gate)
    params: Mapping[str, Any] = field(default_factory=dict)
    fallback_agent: str | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    fallback_retry: RetryPolicy = field(default_factory=lambda: RetryPolicy(max_attempts=2))
    entry_gates: tuple[GateSpec, ...] = ()
    exit_gates: tuple[GateSpec, ...] = ()
    approval: ApprovalSpec | None = None
    remediation: RemediationSpec | None = None
    risk: RiskLevel = RiskLevel.LOW  # static baseline; the engine may raise it dynamically
    mandatory: bool = False  # governance node: a re-plan may never remove / alter it
    skip_if: str | None = None  # name of a registered skip predicate (forbidden on mandatory nodes)
    cacheable: bool = True  # False => never restored from fingerprint; always re-run
    side_effecting: bool = False  # True => compensator required; never cache-restored
    compensator: str | None = None
    idempotent: bool = False  # a side-effecting node may be retried only if idempotent
    compensates: tuple[str, ...] = ()  # side-effecting ancestors to compensate if THIS node fails terminally
    autonomy: AutonomyProfile | None = None  # narrows (never widens) the agent's profile
    timeout_s: float = 300.0  # effective deadline = min(timeout_s, profile.max_runtime_s)

    def __post_init__(self) -> None:
        if not NODE_ID_RE.match(self.id):
            raise ConfigError(f"invalid node id: {self.id!r}")
        if not self.agent:
            raise ConfigError(f"node {self.id}: agent is required")
        for kind in (*self.consumes, *self.optional_consumes, *self.produces, *self.overlay_from, *self.skip_inputs):
            if not KIND_RE.match(kind):
                raise ConfigError(f"node {self.id}: invalid artifact kind {kind!r}")
        if not set(self.overlay_from) <= CHANGESET_KINDS:
            raise ConfigError(f"node {self.id}: overlay_from must be a subset of {sorted(CHANGESET_KINDS)}")
        if bool(self.side_effecting) != bool(self.compensator):
            raise ConfigError(f"node {self.id}: side_effecting requires a compensator and vice versa")
        if self.mandatory and self.skip_if:
            raise ConfigError(f"node {self.id}: a mandatory node cannot have skip_if")
        if not isinstance(self.timeout_s, (int, float)) or not 0 < self.timeout_s < 24 * 3600:
            raise ConfigError(f"node {self.id}: timeout_s must be a finite positive number")
        object.__setattr__(self, "params", _canonical_params(self.params))
        if self.remediation and self.id in self.remediation.route_to:
            raise ConfigError(f"node {self.id}: remediation cannot route to itself")

    def __hash__(self) -> int:
        return hash(self.id)

    def spec_hash(self) -> str:
        return hash_jsonable(self.to_dict())

    @property
    def input_kinds(self) -> tuple[str, ...]:
        """Every kind that is part of the node's input snapshot and fingerprint."""
        seen: dict[str, None] = {}
        for k in (*self.consumes, *self.optional_consumes, *self.overlay_from, *self.skip_inputs):
            seen.setdefault(k, None)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "stage": self.stage.value,
            "agent": self.agent,
            "depends_on": list(self.depends_on),
            "consumes": list(self.consumes),
            "optional_consumes": list(self.optional_consumes),
            "overlay_from": list(self.overlay_from),
            "skip_inputs": list(self.skip_inputs),
            "produces": list(self.produces),
            "params": to_jsonable(self.params),
            "fallback_agent": self.fallback_agent,
            "retry": self.retry.to_dict(),
            "fallback_retry": self.fallback_retry.to_dict(),
            "entry_gates": [g.to_dict() for g in self.entry_gates],
            "exit_gates": [g.to_dict() for g in self.exit_gates],
            "approval": self.approval.to_dict() if self.approval else None,
            "remediation": self.remediation.to_dict() if self.remediation else None,
            "risk": self.risk.name,
            "mandatory": self.mandatory,
            "skip_if": self.skip_if,
            "cacheable": self.cacheable,
            "side_effecting": self.side_effecting,
            "compensator": self.compensator,
            "idempotent": self.idempotent,
            "compensates": list(self.compensates),
            "autonomy": self.autonomy.to_dict() if self.autonomy else None,
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> NodeSpec:
        return cls(
            id=d["id"],
            title=d.get("title", d["id"]),
            stage=Stage(d["stage"]),
            agent=d["agent"],
            depends_on=tuple(d.get("depends_on", ())),
            consumes=tuple(d.get("consumes", ())),
            optional_consumes=tuple(d.get("optional_consumes", ())),
            overlay_from=tuple(d.get("overlay_from", ())),
            skip_inputs=tuple(d.get("skip_inputs", ())),
            produces=tuple(d.get("produces", ())),
            params=dict(d.get("params", {})),
            fallback_agent=d.get("fallback_agent"),
            retry=RetryPolicy.from_dict(d.get("retry", {})),
            fallback_retry=(
                RetryPolicy.from_dict(d["fallback_retry"]) if "fallback_retry" in d else RetryPolicy(max_attempts=2)
            ),
            entry_gates=tuple(GateSpec.from_dict(g) for g in d.get("entry_gates", ())),
            exit_gates=tuple(GateSpec.from_dict(g) for g in d.get("exit_gates", ())),
            approval=ApprovalSpec.from_dict(d["approval"]) if d.get("approval") else None,
            remediation=RemediationSpec.from_dict(d["remediation"]) if d.get("remediation") else None,
            risk=RiskLevel[d.get("risk", "LOW")],
            mandatory=bool(d.get("mandatory", False)),
            skip_if=d.get("skip_if"),
            cacheable=bool(d.get("cacheable", True)),
            side_effecting=bool(d.get("side_effecting", False)),
            compensator=d.get("compensator"),
            idempotent=bool(d.get("idempotent", False)),
            compensates=tuple(d.get("compensates", ())),
            autonomy=AutonomyProfile.from_dict(d["autonomy"]) if d.get("autonomy") else None,
            timeout_s=float(d.get("timeout_s", 300.0)),
        )


@dataclass(frozen=True)
class Rewire:
    """Edges grafted onto an *existing* downstream node by a plan expansion (additive only)."""

    add_depends_on: tuple[str, ...] = ()
    add_overlay_from: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"add_depends_on": list(self.add_depends_on), "add_overlay_from": list(self.add_overlay_from)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Rewire:
        return cls(tuple(d.get("add_depends_on", ())), tuple(d.get("add_overlay_from", ())))


@dataclass(frozen=True)
class PlanExpansion:
    """Declarative graph growth owned by one node (dynamic planning / re-planning).

    **Untrusted input.** The engine (a) requires ``owner == ctx.node.id``, (b) forces the owner
    into every new node's ``depends_on``, (c) overwrites every governance attribute of each new
    node (approval, risk, retry, timeout, gates, mandatory, side-effect flags, skip_if, fallback)
    from the *template* declared on the owner (``params['expansion_template']``), clamping to
    hard bounds, (d) rejects agents/stages/produces outside ``params['expansion_policy']``
    (CHG-007/008), and (e) persists the expansion as a ``plan_expansion`` artifact published
    atomically with the owner's outputs. Rewire may only add edges.
    """

    owner: str
    nodes: tuple[NodeSpec, ...] = ()
    rewire: Mapping[str, Rewire] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "nodes": [n.to_dict() for n in self.nodes],
            "rewire": {k: v.to_dict() for k, v in self.rewire.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PlanExpansion:
        return cls(
            d["owner"],
            tuple(NodeSpec.from_dict(n) for n in d.get("nodes", ())),
            {k: Rewire.from_dict(v) for k, v in d.get("rewire", {}).items()},
        )


# =============================================================================
# Human interaction
# =============================================================================


@dataclass(frozen=True)
class ClarificationQuestion:
    """``id`` is content-derived (``Q-<category>-<sha1(normalised text)[:6]>``), never a sequence
    number, so answers survive amendments and re-runs."""

    id: str
    text: str
    rationale: str = ""
    options: tuple[str, ...] = ()  # when non-empty, answers must be one of these
    default_assumption: str | None = None  # what the agent will assume if accepted/unanswered
    blocking: bool = True
    related_requirements: tuple[str, ...] = ()
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "rationale": self.rationale,
            "options": list(self.options),
            "default_assumption": self.default_assumption,
            "blocking": self.blocking,
            "related_requirements": list(self.related_requirements),
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ClarificationQuestion:
        return cls(
            d["id"],
            d["text"],
            d.get("rationale", ""),
            tuple(d.get("options", ())),
            d.get("default_assumption"),
            bool(d.get("blocking", True)),
            tuple(d.get("related_requirements", ())),
            d.get("category", ""),
        )


@dataclass(frozen=True)
class ApprovalVote:
    """Identity is *asserted*, not authenticated (documented limitation): ``approver`` is a
    normalised ASCII name; ``os_user`` is recorded; strict mode also requires distinct os users."""

    approver: str  # normalised (NFKC, casefold, ASCII, reserved names denied)
    approved: bool
    comment: str
    ts: datetime
    channel: str = "cli"  # "cli" | "interactive" | "scripted"
    os_user: str = ""
    evidence_hash: str = ""  # the evidence this vote covers

    def to_dict(self) -> dict[str, Any]:
        return {
            "approver": self.approver,
            "approved": self.approved,
            "comment": self.comment,
            "ts": self.ts.isoformat(),
            "channel": self.channel,
            "os_user": self.os_user,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ApprovalVote:
        return cls(
            d["approver"],
            bool(d["approved"]),
            d.get("comment", ""),
            parse_datetime(d["ts"]),
            d.get("channel", "cli"),
            d.get("os_user", ""),
            d.get("evidence_hash", ""),
        )


@dataclass
class ApprovalRequest:
    """A pending/decided human checkpoint.

    Two hashes: ``node_fingerprint`` (the node's *inputs*; lazily compared when the node is
    re-evaluated — equal => approval still valid, different => INVALIDATED) and ``evidence_hash``
    (staged output content hashes + policy report + risk; re-verified at publish time).
    State machine: PENDING -> APPROVED (>= required distinct approvers) | REJECTED (any single
    rejection vetoes) | INVALIDATED | CANCELLED. An approver cannot change a vote; voting on a
    non-PENDING request is refused (and audited as APPROVAL_REFUSED).
    """

    id: str  # "A-0001"
    run_id: str
    node_id: str
    position: ApprovalPosition
    purpose: ApprovalPurpose
    risk: RiskLevel
    reasons: tuple[str, ...]
    summary: str  # ENGINE-composed from structured facts only (never agent prose)
    evidence: tuple[ArtifactRef, ...]
    node_fingerprint: str
    evidence_hash: str
    required_approvals: int
    attempt: int
    generation: int
    created_at: datetime
    requested_by: str = ""
    waives: tuple[Finding, ...] = ()
    staged: tuple[ArtifactRef, ...] = ()
    status: ApprovalStatus = ApprovalStatus.PENDING
    votes: list[ApprovalVote] = field(default_factory=list)
    decided_at: datetime | None = None
    agent_note: str = ""  # labelled, length-capped, sanitised agent prose (untrusted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "position": self.position.value,
            "purpose": self.purpose.value,
            "risk": self.risk.name,
            "reasons": list(self.reasons),
            "summary": self.summary,
            "evidence": [e.to_dict() for e in self.evidence],
            "node_fingerprint": self.node_fingerprint,
            "evidence_hash": self.evidence_hash,
            "required_approvals": self.required_approvals,
            "attempt": self.attempt,
            "generation": self.generation,
            "created_at": self.created_at.isoformat(),
            "requested_by": self.requested_by,
            "waives": [f.to_dict() for f in self.waives],
            "staged": [s.to_dict() for s in self.staged],
            "status": self.status.value,
            "votes": [v.to_dict() for v in self.votes],
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "agent_note": self.agent_note,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ApprovalRequest:
        return cls(
            id=d["id"],
            run_id=d["run_id"],
            node_id=d["node_id"],
            position=ApprovalPosition(d["position"]),
            purpose=ApprovalPurpose(d["purpose"]),
            risk=RiskLevel[d["risk"]],
            reasons=tuple(d.get("reasons", ())),
            summary=d.get("summary", ""),
            evidence=tuple(ArtifactRef.from_dict(e) for e in d.get("evidence", ())),
            node_fingerprint=d["node_fingerprint"],
            evidence_hash=d.get("evidence_hash", ""),
            required_approvals=int(d["required_approvals"]),
            attempt=int(d.get("attempt", 0)),
            generation=int(d.get("generation", 0)),
            created_at=parse_datetime(d["created_at"]),
            requested_by=d.get("requested_by", ""),
            waives=tuple(Finding.from_dict(f) for f in d.get("waives", ())),
            staged=tuple(ArtifactRef.from_dict(s) for s in d.get("staged", ())),
            status=ApprovalStatus(d.get("status", "pending")),
            votes=[ApprovalVote.from_dict(v) for v in d.get("votes", ())],
            decided_at=parse_datetime(d["decided_at"]) if d.get("decided_at") else None,
            agent_note=d.get("agent_note", ""),
        )


@dataclass(frozen=True)
class Waiver:
    """A human waiver of MEDIUM-or-lower policy findings. Bound to the finding AND to the evidence
    it was granted for: valid only while ``evidence_hash`` equals the hash of the current staged
    content, never across epochs/re-runs, and never for HIGH/CRITICAL findings."""

    id: str
    rule_id: str
    finding_key: str
    node_id: str
    evidence_hash: str
    justification: str
    approved_by: tuple[str, ...]
    approval_id: str
    decision_id: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "finding_key": self.finding_key,
            "node_id": self.node_id,
            "evidence_hash": self.evidence_hash,
            "justification": self.justification,
            "approved_by": list(self.approved_by),
            "approval_id": self.approval_id,
            "decision_id": self.decision_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Waiver:
        return cls(
            d["id"],
            d["rule_id"],
            d["finding_key"],
            d["node_id"],
            d["evidence_hash"],
            d.get("justification", ""),
            tuple(d.get("approved_by", ())),
            d["approval_id"],
            d["decision_id"],
            parse_datetime(d["created_at"]),
        )


@dataclass(frozen=True)
class SafeStopOverride:
    by: str
    reason: str
    extend: Mapping[str, int] = field(default_factory=dict)  # raise exhausted budgets (bounded)
    reset_breaker: bool = True
    co_approver: str | None = None  # required for integrity / autonomy_violation / compensation_failed


@dataclass(frozen=True)
class KillSignal:
    by: str
    reason: str
    ts: datetime
    scope: str  # "global" | "run"


# =============================================================================
# Decisions (decision lineage)
# =============================================================================


@dataclass(frozen=True)
class DecisionDraft:
    """A decision proposed by an agent. Recorded by the engine **only at publish** (never for
    rejected / quarantined / discarded attempts), with ``inputs`` = the attempt's input snapshot,
    ``outputs`` = published refs and ``parents`` = ``parents`` ∪ decisions whose outputs intersect
    the inputs. ``refs`` cite requirement/task/risk/ADR ids."""

    kind: DecisionKind
    summary: str
    rationale: str = ""
    alternatives: tuple[str, ...] = ()
    outcome: str = ""
    parents: tuple[str, ...] = ()  # existing decision ids; an unknown id rejects the attempt
    refs: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """Append-only record of *why* something happened. Forms the lineage graph together with
    artifact ``inputs`` (see ``core/lineage.py``)."""

    id: str  # "D-0001"
    run_id: str
    ts: datetime
    kind: DecisionKind
    actor: Actor
    node_id: str | None
    summary: str
    rationale: str = ""
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    alternatives: tuple[str, ...] = ()
    outcome: str = ""
    parents: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "ts": self.ts.isoformat(),
            "kind": self.kind.value,
            "actor": self.actor.to_dict(),
            "node_id": self.node_id,
            "summary": self.summary,
            "rationale": self.rationale,
            "inputs": [r.to_dict() for r in self.inputs],
            "outputs": [r.to_dict() for r in self.outputs],
            "alternatives": list(self.alternatives),
            "outcome": self.outcome,
            "parents": list(self.parents),
            "refs": list(self.refs),
            "data": to_jsonable(self.data),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Decision:
        return cls(
            id=d["id"],
            run_id=d["run_id"],
            ts=parse_datetime(d["ts"]),
            kind=DecisionKind(d["kind"]),
            actor=Actor.from_dict(d["actor"]),
            node_id=d.get("node_id"),
            summary=d["summary"],
            rationale=d.get("rationale", ""),
            inputs=tuple(ArtifactRef.from_dict(r) for r in d.get("inputs", ())),
            outputs=tuple(ArtifactRef.from_dict(r) for r in d.get("outputs", ())),
            alternatives=tuple(d.get("alternatives", ())),
            outcome=d.get("outcome", ""),
            parents=tuple(d.get("parents", ())),
            refs=tuple(d.get("refs", ())),
            data=dict(d.get("data", {})),
        )


# =============================================================================
# Audit
# =============================================================================


@dataclass(frozen=True)
class AuditEvent:
    """One line of the tamper-evident audit log (hash-chained, see core/audit.py).

    ``hash`` covers the canonical JSON of every other key (incl. unknown extra keys on read).
    ``seq`` starts at 1. ``attempt`` is the global per-node attempt number.
    """

    seq: int
    ts: datetime
    run_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    type: EventType
    node_id: str | None
    attempt: int | None
    actor: Actor
    data: Mapping[str, Any]
    prev_hash: str
    hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts.isoformat(),
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "type": self.type.value,
            "node_id": self.node_id,
            "attempt": self.attempt,
            "actor": self.actor.to_dict(),
            "data": to_jsonable(self.data),
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AuditEvent:
        return cls(
            seq=int(d["seq"]),
            ts=parse_datetime(d["ts"]),
            run_id=d["run_id"],
            trace_id=d["trace_id"],
            span_id=d["span_id"],
            parent_span_id=d.get("parent_span_id"),
            type=EventType(d["type"]),
            node_id=d.get("node_id"),
            attempt=d.get("attempt"),
            actor=Actor.from_dict(d["actor"]),
            data=dict(d.get("data", {})),
            prev_hash=d["prev_hash"],
            hash=d["hash"],
        )


class AuditSink(Protocol):
    """How tools/agents/stores emit audit events without importing ``core.audit``."""

    def __call__(
        self,
        type: EventType,
        *,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: Mapping[str, Any],
    ) -> None: ...


# =============================================================================
# Policy API
# =============================================================================


@dataclass(frozen=True)
class ChangeSetSubject:
    """Evaluated for every artifact whose kind is in ``CHANGESET_KINDS`` (never also as an
    ``ArtifactSubject``). SEC-005 checks every op path against ``profile.write_globs`` and the
    op count against ``max_changed_files`` — the engine fills ``profile`` from the registry
    entry of the executing agent narrowed by the node, never from agent output; a change set
    produced by an agent with ``profile=None`` fails closed (CRITICAL)."""

    kind: ClassVar[SubjectKind] = SubjectKind.CHANGESET
    changeset: ChangeSet
    baseline: FileView  # the producing attempt's view (see engine-semantics §6)
    node_id: str = ""
    author: str = ""
    impact: Mapping[str, Any] | None = None  # impact_report payload, if available
    profile: AutonomyProfile | None = None
    sources: tuple[ChangeSet, ...] = ()  # input change sets (integrate / docs / promote)


@dataclass(frozen=True)
class ArtifactSubject:
    kind: ClassVar[SubjectKind] = SubjectKind.ARTIFACT
    artifact: Artifact
    text: str | None = None  # decoded content for text artifacts
    untrusted_input: bool = False  # True for human-supplied inputs (SEC-006 applies)


@dataclass(frozen=True)
class ActionSubject:
    """A single tool action an agent is about to perform (autonomy boundary)."""

    kind: ClassVar[SubjectKind] = SubjectKind.ACTION
    agent_id: str
    node_id: str
    action: ToolAction
    target: str
    profile: AutonomyProfile


@dataclass(frozen=True)
class PlanSubject:
    """A proposed plan expansion (CHG-006 mandatory nodes, CHG-007 template conformance)."""

    kind: ClassVar[SubjectKind] = SubjectKind.PLAN
    run_id: str
    expansion: PlanExpansion
    graph_before: Mapping[str, NodeSpec]
    graph_after: Mapping[str, NodeSpec]
    mandatory: tuple[str, ...]
    allowed_agents: frozenset[str] = frozenset()
    allowed_produces: frozenset[str] = frozenset()
    allowed_stages: frozenset[str] = frozenset()


@runtime_checkable
class GovernanceView(Protocol):
    """Read-only governance data handed to AGENTS (no artifact access: lineage stays exact).
    Implementations return deep copies / immutable snapshots."""

    def node_statuses(self) -> Mapping[str, NodeStatus]: ...

    def decisions(self) -> tuple[Decision, ...]: ...

    def decisions_for(self, ref: ArtifactRef) -> tuple[Decision, ...]: ...

    def approvals(self) -> tuple[ApprovalRequest, ...]: ...

    def waivers(self) -> tuple[Waiver, ...]: ...

    def events(self) -> tuple[AuditEvent, ...]: ...


@runtime_checkable
class RunView(GovernanceView, Protocol):
    """Full immutable snapshot for gates / policy (implemented by the engine's ``RunSnapshot``)."""

    def params(self) -> Mapping[str, Any]: ...

    def node_specs(self) -> Mapping[str, NodeSpec]: ...

    def ancestors(self, node_id: str) -> frozenset[str]: ...

    def mandatory_nodes(self) -> tuple[str, ...]: ...

    def latest(self, kind: str, key: str = "") -> Artifact | None: ...

    def artifacts(self, include_inactive: bool = False) -> tuple[Artifact, ...]: ...

    def read_text(self, artifact: Artifact) -> str: ...

    def read_json(self, artifact: Artifact) -> Any: ...


@dataclass(frozen=True)
class RunSubject:
    """Release-time compliance subject. ``node_id`` is the gate position: CMP-001/003 and CHG-005
    consider only ancestors of ``node_id`` plus the artifacts staged by the gated attempt."""

    kind: ClassVar[SubjectKind] = SubjectKind.RUN
    run_id: str
    view: RunView
    node_id: str = ""


PolicySubject = ChangeSetSubject | ArtifactSubject | ActionSubject | PlanSubject | RunSubject


@runtime_checkable
class PolicyRule(Protocol):
    """One executable guardrail. Pure function of the subject (no I/O beyond the subject's own
    ``FileView``). Must fail closed on missing evidence (e.g. CHG-003 with no impact report)."""

    rule_id: str  # e.g. "SEC-001"
    category: PolicyCategory
    title: str
    severity: Severity  # default severity of the findings it emits
    subjects: frozenset[SubjectKind]

    def check(self, subject: PolicySubject) -> list[Finding]: ...


@dataclass(frozen=True)
class PolicyReport:
    """The engine branches ONLY on ``enforcement`` and the pre-computed finding groups, which the
    PolicyEngine fills from its (floor-protected) severity -> enforcement mapping."""

    findings: tuple[Finding, ...] = ()
    waived: tuple[Finding, ...] = ()
    enforcement: Enforcement = Enforcement.ALLOW
    blocking: tuple[Finding, ...] = ()  # DENY / SAFE_STOP
    needs_approval: tuple[Finding, ...] = ()  # REQUIRE_APPROVAL (waivable)
    warnings: tuple[Finding, ...] = ()
    rules_evaluated: tuple[str, ...] = ()

    @property
    def highest_severity(self) -> Severity | None:
        return max((f.severity for f in self.findings), default=None)

    @classmethod
    def from_findings(
        cls,
        findings: Sequence[Finding],
        *,
        waived: Sequence[Finding] = (),
        mapping: Mapping[Severity, Enforcement] = DEFAULT_ENFORCEMENT,
        rules_evaluated: Sequence[str] = (),
    ) -> PolicyReport:
        blocking: list[Finding] = []
        needs: list[Finding] = []
        warn: list[Finding] = []
        overall = Enforcement.ALLOW
        order = [Enforcement.ALLOW, Enforcement.WARN, Enforcement.REQUIRE_APPROVAL, Enforcement.DENY, Enforcement.SAFE_STOP]
        for f in findings:
            enf = mapping[f.severity]
            if order.index(enf) > order.index(overall):
                overall = enf
            if enf in (Enforcement.DENY, Enforcement.SAFE_STOP):
                blocking.append(f)
            elif enf == Enforcement.REQUIRE_APPROVAL:
                needs.append(f)
            elif enf == Enforcement.WARN:
                warn.append(f)
        return cls(
            tuple(findings), tuple(waived), overall, tuple(blocking), tuple(needs), tuple(warn), tuple(rules_evaluated)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enforcement": self.enforcement.value,
            "findings": [f.to_dict() for f in self.findings],
            "waived": [f.to_dict() for f in self.waived],
            "blocking": [f.to_dict() for f in self.blocking],
            "needs_approval": [f.to_dict() for f in self.needs_approval],
            "warnings": [f.to_dict() for f in self.warnings],
            "rules_evaluated": list(self.rules_evaluated),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PolicyReport:
        def fl(key: str) -> tuple[Finding, ...]:
            return tuple(Finding.from_dict(f) for f in d.get(key, ()))

        return cls(
            fl("findings"),
            fl("waived"),
            Enforcement(d.get("enforcement", "allow")),
            fl("blocking"),
            fl("needs_approval"),
            fl("warnings"),
            tuple(d.get("rules_evaluated", ())),
        )


@runtime_checkable
class PolicyEvaluator(Protocol):
    """What the engine / gates / toolbox see of the policy engine."""

    def evaluate(self, subject: PolicySubject) -> PolicyReport: ...

    def with_waivers(self, waivers: Sequence[Waiver]) -> PolicyEvaluator:
        """New evaluator honouring ``waivers`` (only for MEDIUM-or-lower findings)."""
        ...

    def assess_risk(self, subject: PolicySubject) -> RiskAssessment:
        """Risk of a change set (migrations, auth/security modules, deletions, size, API surface)."""
        ...


# =============================================================================
# Gate API
# =============================================================================


@dataclass(frozen=True)
class GateResult:
    gate: str
    decision: GateDecision
    message: str = ""
    findings: tuple[Finding, ...] = ()
    risk: RiskAssessment | None = None
    questions: tuple[ClarificationQuestion, ...] = ()  # for AWAIT_INPUT (merged by question id)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "decision": self.decision.value,
            "message": self.message,
            "findings": [f.to_dict() for f in self.findings],
            "risk": self.risk.to_dict() if self.risk else None,
            "questions": [q.to_dict() for q in self.questions],
            "evidence": to_jsonable(self.evidence),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GateResult:
        return cls(
            d["gate"],
            GateDecision(d["decision"]),
            d.get("message", ""),
            tuple(Finding.from_dict(f) for f in d.get("findings", ())),
            RiskAssessment.from_dict(d["risk"]) if d.get("risk") else None,
            tuple(ClarificationQuestion.from_dict(q) for q in d.get("questions", ())),
            dict(d.get("evidence", {})),
        )


@dataclass
class GateContext:
    run_id: str
    node: NodeSpec
    phase: GatePhase
    attempt: int
    params: Mapping[str, Any]  # GateSpec.params
    artifacts: ArtifactReader  # pinned, node-scoped (declared inputs)
    produced: tuple[Artifact, ...]  # exit phase: the attempt's STAGED artifacts
    reader: ArtifactReader  # also serves the staged artifacts (exit gates)
    policy: PolicyEvaluator
    baseline: FileView  # workspace baseline
    workspace_view: Callable[[], FileView]  # view_for(node) (+ staged changesets in exit phase)
    run_view: RunView
    clock: Clock


@runtime_checkable
class Gate(Protocol):
    name: str

    def evaluate(self, ctx: GateContext) -> GateResult: ...


# =============================================================================
# Agent API
# =============================================================================


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str  # already bounded and redacted
    stderr: str
    duration_s: float
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": self.duration_s,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CommandResult:
        return cls(
            tuple(d["command"]),
            int(d["returncode"]),
            d.get("stdout", ""),
            d.get("stderr", ""),
            float(d.get("duration_s", 0.0)),
            bool(d.get("timed_out", False)),
            bool(d.get("truncated", False)),
        )


@dataclass(frozen=True)
class SearchHit:
    path: str
    line: int
    text: str


@runtime_checkable
class Toolbox(Protocol):
    """The ONLY way an agent touches the world. Every call is autonomy-checked against the agent's
    (node-narrowed) ``AutonomyProfile`` and written to the audit log. Per-attempt instance; the
    ``CancelToken`` is honoured by every method.

    Reads see ``view_for(node)`` (baseline + overlay_from change sets of strict ancestors) plus this
    attempt's own staged writes. Writes are *staged* into the attempt's ``ChangeSet`` — nothing on
    disk changes. Path traversal / absolute / reserved names raise ``PathViolation`` (CRITICAL);
    a valid path outside ``write_globs`` raises ``ToolDenied`` (retryable; >= 3 denials in one
    attempt escalate to ``AutonomyViolation``).
    """

    def read_text(self, path: str) -> str: ...

    def exists(self, path: str) -> bool: ...

    def list_files(self, glob: str = "**") -> list[str]: ...

    def search(self, pattern: str, glob: str = "**/*.py", max_results: int = 100, *, regex: bool = False) -> list[SearchHit]:
        """Literal substring search by default; regex only through a restricted, length-capped grammar."""
        ...

    def write_text(self, path: str, content: str) -> None: ...

    def delete(self, path: str) -> None: ...

    def staged_changeset(self, description: str = "") -> ChangeSet: ...

    def run_tests(
        self,
        targets: Sequence[str] = (),
        *,
        timeout_s: float | None = None,
        mode: str = "view",
        overlay_paths: Sequence[str] = (),
    ) -> CommandResult:
        """``mode='view'``: materialise view + staged writes. ``mode='baseline_plus'``: materialise the
        BASELINE tree plus ONLY the files in ``overlay_paths`` taken from the view (fail-to-pass checks)."""
        ...

    def run_python(
        self,
        args: Sequence[str],
        *,
        timeout_s: float | None = None,
        mode: str = "view",
        overlay_paths: Sequence[str] = (),
    ) -> CommandResult:
        """Only ``['-m', <allow-listed module>, ...]`` or a script path inside the scratch dir; never ``-c``.
        ``mode`` / ``overlay_paths`` as for ``run_tests``."""
        ...

    def note(self, message: str, **data: Any) -> None: ...

    def usage(self) -> Mapping[str, float]:
        """``tool_calls``, ``bytes_written``, ``runtime_s`` (feeds ``AgentResult.usage`` and budgets)."""
        ...


class ToolboxFactory(Protocol):
    def __call__(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        agent_id: str,
        profile: AutonomyProfile,
        view: FileView,
        cancel: CancelToken,
        clock: Clock,
        policy: PolicyEvaluator,
        audit: AuditSink,
        baseline: FileView | None = None,
    ) -> Toolbox: ...


@dataclass(frozen=True)
class ApplyReceipt:
    """Proof of a real side effect (persisted in the ``release_bundle`` payload), enough to undo it."""

    target_id: str
    applied: tuple[str, ...]
    pre_hashes: Mapping[str, str | None]
    post_hashes: Mapping[str, str | None]
    reverse: ChangeSet

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "applied": list(self.applied),
            "pre_hashes": to_jsonable(self.pre_hashes),
            "post_hashes": to_jsonable(self.post_hashes),
            "reverse": self.reverse.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ApplyReceipt:
        return cls(
            d["target_id"],
            tuple(d.get("applied", ())),
            dict(d.get("pre_hashes", {})),
            dict(d.get("post_hashes", {})),
            ChangeSet.from_dict(d["reverse"]),
        )


class EffectPort(Protocol):
    """The ONLY channel for real side effects; injected only into ``side_effecting`` nodes.
    Audited, kill-switch aware, validates every path (``validate_rel_path`` + protected paths)."""

    def apply(self, cs: ChangeSet, *, dry_run: bool = False) -> ApplyReceipt:
        """Verifies ``base_hash`` against the target tree first; all-or-nothing (journal + temp+replace)."""
        ...

    def revert(self, receipt: ApplyReceipt) -> None:
        """Reverses only files whose current hash equals ``post_hashes``; otherwise raises
        (-> ROLLBACK_FAILED -> safe-stop). Idempotent."""
        ...


@dataclass
class AgentContext:
    run_id: str
    node: NodeSpec
    attempt: int  # epoch-local, 1-based (display only)
    attempt_no: int  # GLOBAL per-node number: the value in AuditEvent.attempt / Artifact.attempt / spans
    generation: int  # per-node dispatch generation (late results with an older one are discarded)
    agent_id: str
    is_fallback: bool
    params: Mapping[str, Any]
    artifacts: ArtifactReader  # pinned, node-scoped
    toolbox: Toolbox
    clock: Clock
    cancel: CancelToken
    run_view: GovernanceView
    feedback: tuple[Finding, ...] = ()  # engine-authored: rule_id/severity/location/remediation only
    effects: EffectPort | None = None

    # -- convenience accessors -------------------------------------------------
    def has(self, kind: str, key: str = "") -> bool:
        return self.artifacts.latest(kind, key) is not None

    def text(self, kind: str, key: str = "") -> str:
        art = self.artifacts.latest(kind, key)
        if art is None:
            raise KeyError(f"artifact kind {kind!r} (key={key!r}) is not available")
        return self.artifacts.read_text(art)

    def json(self, kind: str, key: str = "") -> Any:
        art = self.artifacts.latest(kind, key)
        if art is None:
            raise KeyError(f"artifact kind {kind!r} (key={key!r}) is not available")
        return self.artifacts.read_json(art)

    def all(self, kind: str) -> list[Artifact]:
        return self.artifacts.active(kind)


@dataclass
class AgentResult:
    artifacts: list[ArtifactDraft] = field(default_factory=list)
    decisions: list[DecisionDraft] = field(default_factory=list)
    plan_expansion: PlanExpansion | None = None
    clarifications: list[ClarificationQuestion] = field(default_factory=list)
    rationale: str = ""  # untrusted narrative; shown only in a labelled, sanitised block
    usage: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """Agents are stateless: the registry creates a fresh instance per attempt."""

    agent_id: str
    role: str
    version: str  # part of the node fingerprint
    autonomy: AutonomyProfile

    def run(self, ctx: AgentContext) -> AgentResult: ...


@runtime_checkable
class Compensator(Protocol):
    """Undoes the real-world side effects of a ``side_effecting`` node. Runs with a FRESH cancel
    token and its own timeout, from the persisted receipt — no live attempt context needed."""

    def compensate(self, receipt: ApplyReceipt, *, effects: EffectPort, run_view: GovernanceView) -> str:
        """Returns a human-readable description of what was undone."""
        ...


@dataclass(frozen=True)
class FailureInfo:
    kind: FailureKind
    message: str
    findings: tuple[Finding, ...] = ()
    retryable: bool = True


@dataclass
class AttemptOutcome:
    """The ONLY thing a worker thread returns to the scheduler loop. The worker builds the context
    from an immutable snapshot, runs the agent, validates and STAGES drafts, runs exit gates and
    policy against the staged artifacts, and returns this; the scheduler only applies transitions
    (publish / approvals / retry / fallback / replan)."""

    node_id: str
    generation: int
    attempt_no: int
    epoch: Epoch
    agent_id: str
    started_at: datetime
    ended_at: datetime
    result: AgentResult | None = None
    failure: FailureInfo | None = None
    staged: tuple[ArtifactRef, ...] = ()
    gate_results: tuple[GateResult, ...] = ()
    policy: PolicyReport | None = None
    risk: RiskAssessment | None = None
    inputs: tuple[ArtifactRef, ...] = ()
    usage: Mapping[str, float] = field(default_factory=dict)


SkipPredicate = Callable[[RunView, NodeSpec], str | None]
"""Returns a reason string if the node should be SKIPPED, else ``None``. May read only the kinds in
``node.skip_inputs`` (re-evaluated whenever they change: SKIPPED -> PENDING)."""
