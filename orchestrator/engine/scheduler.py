"""The orchestration engine: single-writer scheduler, attempt workers, publish protocol.

``Orchestrator`` is the public facade (``start`` / ``run`` / ``resume`` / ``approve`` / ``reject`` /
``answer`` / ``amend`` / ``kill`` / ``status`` / ``lineage`` / ``verify_audit`` / ``events`` /
``metrics``).  Behaviour follows ``docs/engine-semantics.md``; the pieces that matter most:

* **One writer.**  A run is driven by exactly one scheduler thread (the caller of ``run`` /
  ``resume``).  Only that thread mutates ``RunState``, node statuses, the graph and the publish step.
  Each attempt runs on a daemon worker thread that builds its context from an immutable snapshot,
  runs the agent, validates and *stages* the drafts, runs exit gates and policy, and returns one
  ``AttemptOutcome``.  The scheduler only applies transitions.
* **Nothing the agent says is trusted.**  Drafts are re-keyed, schema-validated, size-bounded and
  evaluated by the policy; change sets are admitted against the attempt's own view; plan expansions
  are normalised from the owner's template; feedback handed back to agents carries only
  engine-authored fields.
* **Publish protocol P1-P6** is journaled (``NODE_PUBLISHING`` before any mutation) so ``resume``
  can roll a crashed publish forward.
* **Verdicts are never cached.**  A fingerprint cache hit restores outputs but entry gates, exit
  gates, policy and approval validity are re-run over the restored artifacts.

Commands (``approve`` ...) issued while the run is live *in this process* are queued and applied by
the scheduler thread; otherwise the caller takes the run lock and applies them through the same code.

Known limitations (descoped, see also ``docs/risks-and-tradeoffs.md``):

* **No cross-process command inbox.**  If another live process holds ``run.lock``, ``approve`` /
  ``reject`` / ``answer`` / ``amend`` raise ``LockError``; ``kill`` still works across processes
  through the ``KILL`` file, which the live engine honours on its next tick.
* **No audit anchors file and no new audit segments.**  Tail truncation is detected through
  ``RunState.audit_head`` and ``seal.json`` only; a broken chain makes ``resume`` raise
  ``IntegrityError`` instead of forking a new segment.
* **``identity_mode='strict'`` is not implemented** (identities are attested only; the report says so).
* **No per-attempt span trace** beyond what the audit chain already carries.
* **Serialising re-plan on integration conflicts is not automatic**: a conflict fails the
  ``integration_clean`` gate; a human or a planner must change the plan.
* Waiver creation re-evaluates the (linear-time, size-capped) policy on the scheduler thread.
* A MEDIUM screening hit (SEC-006) in a human input is recorded (``INPUT_SCREENED``) and attributed
  to the submitting human; the node approvals downstream are the confirmation.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import difflib
import getpass
import hashlib
import inspect
import queue
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from orchestrator.core.artifacts import ArtifactStore, PinnedReader
from orchestrator.core.audit import (
    AUDIT_FILENAME,
    AuditLog,
    Redactor,
    VerifyResult,
    read_run_events,
    verify_run,
)
from orchestrator.core.clock import SystemClock
from orchestrator.core.decisions import DecisionLog
from orchestrator.core.errors import (
    AgentCancelled,
    AgentError,
    AgentTimeout,
    ApprovalError,
    AutonomyViolation,
    ConfigError,
    GraphError,
    IntegrityError,
    LockError,
    OrchestratorError,
    PlanPatchError,
    PolicyDenied,
    StateError,
    ToolDenied,
    TransientAgentError,
    UndeclaredInputError,
)
from orchestrator.core.fsutil import RunLock, atomic_write_json, read_bytes_retry
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.lineage import LineageGraph
from orchestrator.core.metrics import RunMetrics, compute_run_metrics
from orchestrator.core.paths import RUN_ID_RE
from orchestrator.core.sanitize import sanitize_for_terminal, strip_controls
from orchestrator.core.schemas import SCHEMA_VERSION, Kind, validate_payload
from orchestrator.core.serde import hash_jsonable, loads_strict, to_jsonable
from orchestrator.core.state import (
    AttemptOutcomeKind,
    AttemptRecord,
    ClarificationRound,
    Interval,
    NodeState,
    RunState,
    SafeStopRecord,
    StateStore,
)
from orchestrator.core.types import (
    CHANGESET_KINDS,
    DEFAULT_ENFORCEMENT,
    EXTERNAL_PRODUCER,
    KEYED_KINDS,
    K_CLARIFICATION_ANSWERS,
    K_REQUEST,
    Actor,
    AgentContext,
    AgentResult,
    ApplyReceipt,
    ApprovalPosition,
    ApprovalPurpose,
    ApprovalRequest,
    ApprovalSpec,
    ApprovalStatus,
    Artifact,
    ArtifactDraft,
    ArtifactRef,
    ArtifactStatus,
    ArtifactSubject,
    AttemptOutcome,
    AuditEvent,
    AutonomyProfile,
    CancelToken,
    ChangeSet,
    ClarificationQuestion,
    Clock,
    Decision,
    DecisionDraft,
    DecisionKind,
    Enforcement,
    EffectPort,
    Epoch,
    EventType,
    FailureInfo,
    FailureKind,
    FileView,
    Finding,
    GateContext,
    GateDecision,
    GatePhase,
    GateResult,
    GateSpec,
    NodeSpec,
    NodeStatus,
    PlanExpansion,
    PolicyCategory,
    PolicyEvaluator,
    RiskAssessment,
    RiskLevel,
    RunStatus,
    SafeStopOverride,
    Severity,
    ToolboxFactory,
    Waiver,
    assert_transition,
)
from orchestrator.engine.approvals import (
    ApprovalManager,
    ApprovalProvider,
    compose_summary,
    normalize_approver,
)
from orchestrator.engine.config import AutonomyMode, EngineConfig
from orchestrator.engine.expansion import normalize_expansion
from orchestrator.engine.gates import compose, run_gates
from orchestrator.engine.registry import Registries
from orchestrator.engine.replan import (
    Transition,
    compute_fingerprint,
    invalidate,
    resolve_remediation_targets,
)
from orchestrator.engine.report import write_run_outputs
from orchestrator.engine.safestop import KillSwitch, SafeStopController
from orchestrator.tools.changeset import ChangeSetConflict, diff_stats
from orchestrator.tools.changeset import digest as changeset_digest
from orchestrator.tools.effects import DirectoryEffectPort
from orchestrator.tools.toolbox import make_toolbox_factory
from orchestrator.tools.workspace import BaselineSnapshot, OverlayView

ENGINE_SCHEMA_VERSION = 1
ENGINE_ACTOR_ID = "engine"
RESERVED_APPROVER_NAMES = frozenset({"system", "engine", "policy", "scripted", "agent", "unknown"})
MAX_QUESTIONS = 20
MAX_ANSWER_BYTES = 2048
MAX_FEEDBACK_REMEDIATION = 300
MAX_DRAFT_DECISIONS = 50
MAX_PARENT_DECISIONS = 50
MAX_INPUT_REFS = 200
MAX_AGENT_NOTE = 500
HEARTBEAT_PERSIST_S = 5.0
COMMAND_WAIT_S = 30.0
LOCK_RETRY_S = 2.0
PENDING_DIR = "pending"

_SYSTEM = Actor.system(ENGINE_ACTOR_ID)
_TERMINAL_EXIT_CODES = {
    RunStatus.SUCCEEDED: 0,
    RunStatus.WAITING_HUMAN: 10,
    RunStatus.FAILED: 20,
    RunStatus.SAFE_STOPPED: 30,
}
_REF_OK = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-")
_QUESTION_ID_PREFIX = "Q-"

Finalizer = Callable[["FinalizerContext"], None]
EffectFactory = Callable[[Path, Any, "str | None", "int | None", "CancelToken | None"], EffectPort]
"""``(target_root, audit, node_id, attempt, cancel) -> EffectPort`` (built per attempt / compensation)."""


# =============================================================================
# public result / context types
# =============================================================================


@dataclass(frozen=True)
class RunResult:
    """What ``run`` / ``resume`` return (CLI exit codes: 0 succeeded, 10 waiting, 20 failed, 30 safe-stopped)."""

    run_id: str
    status: RunStatus
    reason: str = ""
    pending_approvals: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    audit_head: tuple[int, str] = (0, "")

    @property
    def exit_code(self) -> int:
        return _TERMINAL_EXIT_CODES.get(self.status, 20)


@dataclass(frozen=True)
class FinalizerContext:
    """Handed to every injected finalizer once, when the run reaches a terminal state.

    ``publish_artifact`` stores a draft as an ACTIVE artifact produced by the engine (audited);
    ``record_decision`` appends a decision (actor ``system``).
    """

    run_id: str
    status: RunStatus
    reason: str
    view: RunView
    run_dir: Path
    audit_head: tuple[int, str]
    publish_artifact: Callable[[ArtifactDraft], Artifact]
    record_decision: Callable[..., Decision]


# =============================================================================
# small helpers
# =============================================================================


def _clip(text: object, limit: int = 300) -> str:
    return sanitize_for_terminal(text, limit)


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _normalise_ws(text: str) -> str:
    return " ".join(text.lower().split())


def question_id_for(category: str, text: str) -> str:
    """Content-derived clarification question id (same grammar as ``agents.base.question_id``)."""
    digest = hashlib.sha1(  # noqa: S324 - identifier derivation, not a security control
        "\x1f".join(_normalise_ws(p) for p in (category, text)).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return f"Q-{category}-{digest[:6]}"


def _safe_ref(value: str) -> bool:
    return (
        bool(value) and len(value) <= 64 and value[0].isalnum() and all(c in _REF_OK for c in value)
    )


def _finding_feedback(finding: Finding) -> Finding:
    """Engine-authored feedback: rule id, severity, location and the rule's own remediation hint only."""
    return Finding(
        finding.rule_id,
        finding.category,
        finding.severity,
        f"{finding.rule_id} finding",
        location=_clip(finding.location, 160) if finding.location else None,
        evidence=None,
        remediation=_clip(finding.remediation, MAX_FEEDBACK_REMEDIATION)
        if finding.remediation
        else None,
    )


def _engine_finding(
    rule_id: str, message: str, *, location: str | None = None, severity: Severity = Severity.HIGH
) -> Finding:
    return Finding(
        rule_id,
        PolicyCategory.CHANGE_CONTROL,
        severity,
        _clip(message, 300),
        location=location,
        remediation="Fix the output so that it satisfies the engine contract.",
    )


def _risk_max(*levels: RiskLevel) -> RiskLevel:
    return max(levels)


def _utc_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _is_text_media(artifact: Artifact) -> bool:
    return artifact.media_type.startswith("text/") or artifact.media_type == "application/json"


def evidence_hash_of(
    staged: Sequence[ArtifactRef], findings: Sequence[Finding], risk: RiskLevel
) -> str:
    """Hash an approval covers: staged content hashes + the (pre-waiver) findings + the risk level."""
    return hash_jsonable(
        {
            "staged": sorted((r.artifact_id, r.content_hash) for r in staged),
            "findings": sorted(f.key() for f in findings),
            "risk": risk.name,
        }
    )


def _start_evidence_hash(inputs: Sequence[ArtifactRef], risk: RiskLevel) -> str:
    return hash_jsonable(
        {"inputs": sorted((r.artifact_id, r.content_hash) for r in inputs), "risk": risk.name}
    )


# =============================================================================
# immutable run snapshots (RunView / GovernanceView)
# =============================================================================


class GovernanceSnapshot:
    """The restricted ``GovernanceView`` agents get: no artifact access, deep-copied answers."""

    def __init__(self, base: RunSnapshot) -> None:
        self._base = base

    def node_statuses(self) -> Mapping[str, NodeStatus]:
        return self._base.node_statuses()

    def decisions(self) -> tuple[Decision, ...]:
        return self._base.decisions()

    def decisions_for(self, ref: ArtifactRef) -> tuple[Decision, ...]:
        return self._base.decisions_for(ref)

    def approvals(self) -> tuple[ApprovalRequest, ...]:
        return self._base.approvals()

    def waivers(self) -> tuple[Waiver, ...]:
        return self._base.waivers()

    def events(self) -> tuple[AuditEvent, ...]:
        return self._base.events()


class RunSnapshot:
    """Immutable ``RunView`` + ``GovernanceView`` (accessors return copies, never live objects).

    Append-only sources (audit events, decisions) are captured as a cut-off so the snapshot stays
    cheap; mutable sources (statuses, approvals, specs, params) are copied at construction.
    ``allowed_kinds`` / ``allowed_producers`` restrict artifact access (skip predicates may read only
    ``skip_inputs`` of ancestors).
    """

    def __init__(
        self,
        *,
        run_id: str,
        status: RunStatus,
        reason: str,
        params: Mapping[str, Any],
        specs: Mapping[str, NodeSpec],
        ancestors: Mapping[str, frozenset[str]],
        mandatory: tuple[str, ...],
        statuses: Mapping[str, NodeStatus],
        approvals: Sequence[ApprovalRequest],
        waivers: Sequence[Waiver],
        artifacts: Sequence[Artifact],
        decision_log: DecisionLog,
        decision_count: int,
        events_fn: Callable[[], list[AuditEvent]],
        event_count: int,
        store: ArtifactStore,
        extra_artifacts: Sequence[Artifact] = (),
        allowed_kinds: frozenset[str] | None = None,
        allowed_producers: frozenset[str] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.status = status
        self.reason = reason
        self._params = copy.deepcopy(dict(params))
        self._specs = dict(specs)
        self._ancestors = {k: frozenset(v) for k, v in ancestors.items()}
        self._mandatory = tuple(mandatory)
        self._statuses = dict(statuses)
        self._approvals = tuple(ApprovalRequest.from_dict(a.to_dict()) for a in approvals)
        self._waivers = tuple(waivers)
        self._artifacts = tuple(artifacts)
        self._extra = tuple(extra_artifacts)
        self._decision_log = decision_log
        self._decision_count = decision_count
        self._events_fn = events_fn
        self._event_count = event_count
        self._store = store
        self._allowed_kinds = allowed_kinds
        self._allowed_producers = allowed_producers
        self.meta: dict[str, Any] = dict(meta or {})

    # -- GovernanceView ----------------------------------------------------------------
    def node_statuses(self) -> Mapping[str, NodeStatus]:
        return dict(self._statuses)

    def decisions(self) -> tuple[Decision, ...]:
        return self._decision_log.all()[: self._decision_count]

    def decisions_for(self, ref: ArtifactRef) -> tuple[Decision, ...]:
        known = {d.id for d in self.decisions()}
        return tuple(d for d in self._decision_log.decisions_for(ref) if d.id in known)

    def approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(ApprovalRequest.from_dict(a.to_dict()) for a in self._approvals)

    def waivers(self) -> tuple[Waiver, ...]:
        return self._waivers

    def events(self) -> tuple[AuditEvent, ...]:
        out = []
        for ev in self._events_fn()[: self._event_count]:
            out.append(dataclasses.replace(ev, data=copy.deepcopy(dict(ev.data))))
        return tuple(out)

    # -- RunView ------------------------------------------------------------------------
    def params(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._params)

    def node_specs(self) -> Mapping[str, NodeSpec]:
        return {k: copy.deepcopy(v) for k, v in self._specs.items()}

    def ancestors(self, node_id: str) -> frozenset[str]:
        try:
            return self._ancestors[node_id]
        except KeyError:
            raise KeyError(f"unknown node {_clip(node_id, 60)!r}") from None

    def mandatory_nodes(self) -> tuple[str, ...]:
        return self._mandatory

    def _visible(self, art: Artifact) -> bool:
        if self._allowed_kinds is not None and art.kind not in self._allowed_kinds:
            return False
        return self._allowed_producers is None or art.producer in self._allowed_producers

    def artifacts(self, include_inactive: bool = False) -> tuple[Artifact, ...]:
        pool = [
            *self._artifacts,
            *(a for a in self._extra if a.id not in {x.id for x in self._artifacts}),
        ]
        if not include_inactive:
            pool = [a for a in pool if a.status == ArtifactStatus.ACTIVE]
        return tuple(a for a in pool if self._visible(a))

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        if self._allowed_kinds is not None and kind not in self._allowed_kinds:
            raise UndeclaredInputError(f"artifact kind {_clip(kind, 60)!r} is not readable here")
        for art in self.artifacts():
            if art.kind == kind and art.key == key:
                return art
        return None

    def _check_readable(self, artifact: Artifact) -> None:
        if not self._visible(artifact) or all(
            artifact.id != a.id for a in (*self._artifacts, *self._extra)
        ):
            raise UndeclaredInputError(
                f"artifact {_clip(artifact.id, 80)} is not part of this view"
            )

    def read_text(self, artifact: Artifact) -> str:
        self._check_readable(artifact)
        return self._store.read_text(artifact)

    def read_json(self, artifact: Artifact) -> Any:
        self._check_readable(artifact)
        return self._store.read_json(artifact)

    # -- conveniences for CLI / reports ---------------------------------------------------
    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(a for a in self.approvals() if a.status == ApprovalStatus.PENDING)

    def open_questions(self) -> tuple[ClarificationQuestion, ...]:
        return tuple(self.meta.get("open_questions", ()))

    def with_extra(self, artifacts: Sequence[Artifact]) -> RunSnapshot:
        clone = copy.copy(self)
        clone._extra = (*self._extra, *artifacts)
        return clone

    def governance(self) -> GovernanceSnapshot:
        return GovernanceSnapshot(self)


RunView = RunSnapshot  # public alias used in ``FinalizerContext``


# =============================================================================
# invariants (engine-semantics section 19)
# =============================================================================

_REPLAY_STATUS: Mapping[EventType, NodeStatus] = {
    EventType.NODE_READY: NodeStatus.READY,
    EventType.NODE_SKIPPED: NodeStatus.SKIPPED,
    EventType.NODE_STARTED: NodeStatus.RUNNING,
    EventType.NODE_RETRY_SCHEDULED: NodeStatus.READY,
    EventType.NODE_AWAITING_APPROVAL: NodeStatus.AWAITING_APPROVAL,
    EventType.NODE_AWAITING_INPUT: NodeStatus.AWAITING_INPUT,
    EventType.NODE_SUCCEEDED: NodeStatus.SUCCEEDED,
    EventType.NODE_FAILED: NodeStatus.FAILED,
    EventType.NODE_BLOCKED: NodeStatus.BLOCKED,
    EventType.NODE_STALE: NodeStatus.STALE,
    EventType.NODE_RESET: NodeStatus.PENDING,
    EventType.NODE_REMOVED: NodeStatus.REMOVED,
    EventType.NODE_CANCELLED: NodeStatus.CANCELLED,
    EventType.NODE_ROLLED_BACK: NodeStatus.ROLLED_BACK,
}


def replay_node_statuses(events: Sequence[AuditEvent]) -> dict[str, NodeStatus]:
    """Rebuild node statuses from the audit chain (property: equals ``state.json``)."""
    out: dict[str, NodeStatus] = {}
    for event in events:
        if event.node_id is None:
            continue
        status = _REPLAY_STATUS.get(event.type)
        if status is not None:
            out[event.node_id] = status
    return out


def check_invariants(
    state: RunState,
    *,
    graph: WorkflowGraph | None = None,
    store: ArtifactStore | None = None,
    events: Sequence[AuditEvent] | None = None,
) -> list[str]:
    """Violations of engine-semantics section 19 (empty list = healthy)."""
    problems: list[str] = []
    specs: Mapping[str, NodeSpec] = graph.nodes if graph is not None else {}
    for node_id, ns in state.nodes.items():
        if ns.status == NodeStatus.SUCCEEDED:
            for req in (r for r in state.approvals.values() if r.node_id == node_id):
                if req.status == ApprovalStatus.APPROVED:
                    distinct = {v.approver for v in req.votes if v.approved}
                    if len(distinct) < req.required_approvals:
                        problems.append(
                            f"{req.id}: approved with {len(distinct)} < {req.required_approvals} approvers"
                        )
            spec = specs.get(node_id)
            if spec is not None and spec.approval is not None and spec.approval.always:
                ok = any(
                    r.node_id == node_id
                    and r.status == ApprovalStatus.APPROVED
                    and r.node_fingerprint == ns.last_success_fingerprint
                    for r in state.approvals.values()
                )
                if not ok:
                    problems.append(
                        f"{node_id}: succeeded without a valid approval for its final fingerprint"
                    )
        if ns.status == NodeStatus.REMOVED and store is not None:
            for art in store.all():
                if art.producer == node_id:
                    problems.append(f"{art.id}: ACTIVE artifact of removed node {node_id}")
    for node_id, digest in state.mandatory_snapshot.items():
        if graph is not None:
            if node_id not in graph:
                problems.append(f"mandatory node {node_id} missing from the graph")
            elif graph.node(node_id).spec_hash() != digest:
                problems.append(f"mandatory node {node_id} was altered")
    if store is not None:
        seen: set[tuple[str, str]] = set()
        for art in store.all():
            if (art.kind, art.key) in seen:
                problems.append(f"two ACTIVE versions of {art.kind}/{art.key}")
            seen.add((art.kind, art.key))
        referenced = {r.artifact_id for ns in state.nodes.values() for r in ns.staged}
        for art in store.staged():
            if art.id not in referenced and state.status.is_terminal:
                problems.append(f"{art.id}: STAGED artifact left behind")
    if events is not None:
        replay = replay_node_statuses(events)
        for node_id, ns in state.nodes.items():
            expected = replay.get(node_id, NodeStatus.PENDING)
            if expected != ns.status:
                problems.append(
                    f"{node_id}: state says {ns.status.value}, audit replay says {expected.value}"
                )
        retries = sum(1 for e in events if e.type == EventType.NODE_RETRY_SCHEDULED)
        if state.counters.retries < retries:
            problems.append("counters.retries is below the audit-derived count")
        fallbacks = sum(1 for e in events if e.type == EventType.NODE_FALLBACK_ACTIVATED)
        if state.counters.fallbacks < fallbacks:
            problems.append("counters.fallbacks is below the audit-derived count")
        seq = int(state.audit_head.get("seq", 0))
        if seq and not any(e.seq == seq and e.hash == state.audit_head.get("hash") for e in events):
            problems.append("audit_head is not part of the audit chain")
    return problems


# =============================================================================
# internal dataclasses
# =============================================================================


@dataclass
class _Pending:
    """Everything a held attempt (awaiting approval / answers) needs to be published later.

    Persisted to ``<run_dir>/pending/<node>.json`` while the node waits so a fresh process can
    resume it.  All fields are plain JSON; the staged artifacts themselves live in the store.
    """

    node_id: str
    attempt_no: int
    generation: int
    epoch: str
    agent_id: str
    fingerprint: str
    decisions: list[dict[str, Any]] = field(default_factory=list)
    questions: list[dict[str, Any]] = field(default_factory=list)
    expansion: dict[str, Any] | None = None
    rationale: str = ""
    diff_stats: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    waivable: list[dict[str, Any]] = field(default_factory=list)
    risk: dict[str, Any] = field(default_factory=lambda: RiskAssessment(RiskLevel.LOW).to_dict())
    policy: dict[str, Any] | None = None
    inputs: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, float] = field(default_factory=dict)
    receipt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> _Pending:
        names = {f.name for f in dataclasses.fields(cls)}
        if not isinstance(raw, Mapping) or not set(raw) <= names:
            raise ValueError("malformed pending-attempt record")
        return cls(**dict(raw))

    # typed accessors -------------------------------------------------------------
    def finding_list(self) -> list[Finding]:
        return [Finding.from_dict(f) for f in self.findings]

    def waivable_list(self) -> list[Finding]:
        return [Finding.from_dict(f) for f in self.waivable]

    def risk_assessment(self) -> RiskAssessment:
        return RiskAssessment.from_dict(self.risk)

    def question_list(self) -> list[ClarificationQuestion]:
        return [ClarificationQuestion.from_dict(q) for q in self.questions]


@dataclass
class _Job:
    """One in-flight worker execution (an attempt or a cache revalidation)."""

    kind: str  # "attempt" | "revalidate"
    spec: NodeSpec
    generation: int
    attempt_no: int
    epoch: Epoch
    agent_id: str
    cancel: CancelToken
    pinned: PinnedReader
    fingerprint: str
    feedback: tuple[Finding, ...]
    snapshot: RunSnapshot
    policy: PolicyEvaluator
    graph: WorkflowGraph
    topo_index: Mapping[str, int]
    mandatory: tuple[str, ...]
    deadline_s: float
    restored: tuple[Artifact, ...] = ()
    thread: threading.Thread | None = None
    started_mono: float | None = None
    receipts: list[dict[str, Any]] = field(default_factory=list)
    zombie: bool = False


@dataclass
class _Completion:
    job: _Job
    outcome: AttemptOutcome
    pending: _Pending
    questions: tuple[ClarificationQuestion, ...] = ()
    restored_ok: bool = True


@dataclass
class _Command:
    kind: str
    params: dict[str, Any]
    future: Future[Any]


@dataclass
class _Wave:
    """An open re-plan wave: settles when every invalidated node has been revalidated."""

    trigger: str
    invalidated: list[str]
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)
    cancelled_inflight: list[str] = field(default_factory=list)
    cache_hit: list[str] = field(default_factory=list)
    re_run: list[str] = field(default_factory=list)
    pending: set[str] = field(default_factory=set)


def _kill_signal_id(signal: Any) -> str:
    return f"{signal.scope}:{_clip(signal.by, 64)}:{signal.ts.isoformat()}"


# =============================================================================
# the per-run engine
# =============================================================================


class _Engine:
    """Everything one run needs; the thread that calls :meth:`loop` is the single writer."""

    def __init__(self, orch: Orchestrator, state: RunState, run_dir: Path) -> None:
        self.orch = orch
        self.cfg: EngineConfig = orch.config
        self.clock: Clock = orch.clock
        self.registries: Registries = orch.registries
        self.state = state
        self.run_dir = run_dir
        self.run_id = state.run_id
        self.state_store = StateStore(orch.runs_dir)
        self.redactor = Redactor()
        self._completions: queue.Queue[_Completion] = queue.Queue()
        self._backlog: list[_Completion] = []
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._jobs: dict[str, _Job] = {}
        self._zombies: list[_Job] = []
        self._waves: list[_Wave] = []
        self._stopping = False
        self._done: RunResult | None = None
        self._dirty = False
        self._last_mono: float | None = None
        self._last_heartbeat_persist: datetime | None = None
        self._policy_cache: tuple[tuple[str, ...], PolicyEvaluator] | None = None
        self._pending_cache: dict[str, _Pending] = {}
        self._live = False
        self._lock: RunLock | None = None
        self.audit: AuditLog
        self.store: ArtifactStore
        self.decisions: DecisionLog
        self.baseline: BaselineSnapshot
        self.graph: WorkflowGraph
        self.approvals: ApprovalManager
        self.kill_switch: KillSwitch
        self.safe: SafeStopController

    # ------------------------------------------------------------------ construction
    def attach(self, *, audit_mode: str = "strict") -> None:
        """Open every persistent component of the run (raises ``IntegrityError`` on a broken chain)."""
        self.audit = AuditLog(
            self.run_dir / AUDIT_FILENAME,
            run_id=self.run_id,
            trace_id=self.state.trace_id,
            clock=self.clock,
            mode="recover" if audit_mode == "recover" else "strict",
        )
        self.store = ArtifactStore(
            self.run_dir / "artifacts",
            clock=self.clock,
            audit=self.audit,
            max_artifact_bytes=self.cfg.max_artifact_bytes,
        )
        self.decisions = DecisionLog(
            self.run_dir / "decisions.jsonl", run_id=self.run_id, clock=self.clock, audit=self.audit
        )
        self.baseline = BaselineSnapshot.load(self.run_dir / "baseline")
        self.baseline_view: FileView = self.baseline.view
        self.graph = WorkflowGraph.from_dict(self.state.graph)
        reserved = frozenset(
            {n.casefold() for n in (*self.registries.agents.ids(), *RESERVED_APPROVER_NAMES)}
        )
        self.reserved = reserved
        self.approvals = ApprovalManager(
            self.state,
            self.cfg,
            self.clock,
            self.audit,
            reserved_names=reserved,
            created_by=self.state.created_by,
        )
        self.kill_switch = KillSwitch(self.orch.runs_dir / "KILL", self.run_dir / "KILL")
        self.safe = SafeStopController(self.cfg)
        for listener in self.orch.listeners:
            self.audit.add_listener(listener)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.audit.flush(2.0)
        with contextlib.suppress(Exception):
            self.audit.close(2.0)

    # ------------------------------------------------------------------ small helpers
    def emit(
        self,
        etype: EventType,
        *,
        node_id: str | None = None,
        attempt: int | None = None,
        actor: Actor = _SYSTEM,
        **data: Any,
    ) -> AuditEvent:
        return self.audit.emit(etype, node_id=node_id, attempt=attempt, actor=actor, data=data)

    def save(self) -> None:
        seq, digest = self.audit.head()
        self.state.audit_head = {"seq": seq, "hash": digest}
        self.state_store.save(self.state, now=self.clock.now())
        self._dirty = False

    def now(self) -> datetime:
        return self.clock.now()

    def ns(self, node_id: str) -> NodeState:
        return self.state.nodes[node_id]

    def live_nodes(self) -> list[str]:
        """Node ids of the graph in topological order."""
        return self.graph.topological_order()

    def pending_path(self, node_id: str) -> Path:
        return self.run_dir / PENDING_DIR / f"{node_id}.json"

    def _policy_now(self) -> PolicyEvaluator:
        """The policy honouring the current waivers (rebuilt only when the waiver set changes)."""
        key = tuple(w.id for w in self.state.waivers)
        if not key:
            return self.orch.policy
        if self._policy_cache is None or self._policy_cache[0] != key:
            self._policy_cache = (key, self.orch.policy.with_waivers(tuple(self.state.waivers)))
        return self._policy_cache[1]

    def policy_hash(self) -> str:
        policy = self.orch.policy
        return hash_jsonable(
            [str(getattr(policy, "manifest_hash", "")), str(getattr(policy, "config_hash", ""))]
        )

    # ------------------------------------------------------------------ status transitions
    _STATUS_EVENT: Mapping[NodeStatus, EventType] = {
        NodeStatus.READY: EventType.NODE_READY,
        NodeStatus.SKIPPED: EventType.NODE_SKIPPED,
        NodeStatus.RUNNING: EventType.NODE_STARTED,
        NodeStatus.AWAITING_APPROVAL: EventType.NODE_AWAITING_APPROVAL,
        NodeStatus.AWAITING_INPUT: EventType.NODE_AWAITING_INPUT,
        NodeStatus.SUCCEEDED: EventType.NODE_SUCCEEDED,
        NodeStatus.FAILED: EventType.NODE_FAILED,
        NodeStatus.BLOCKED: EventType.NODE_BLOCKED,
        NodeStatus.STALE: EventType.NODE_STALE,
        NodeStatus.PENDING: EventType.NODE_RESET,
        NodeStatus.REMOVED: EventType.NODE_REMOVED,
        NodeStatus.CANCELLED: EventType.NODE_CANCELLED,
        NodeStatus.ROLLED_BACK: EventType.NODE_ROLLED_BACK,
    }

    def transition(
        self,
        node_id: str,
        new: NodeStatus,
        reason: str = "",
        *,
        etype: EventType | None = None,
        attempt: int | None = None,
        **data: Any,
    ) -> None:
        """The ONLY place a node status changes (always checks the transition table)."""
        ns = self.ns(node_id)
        assert_transition(ns.status, new)
        ns.status = new
        ns.reason = _clip(reason, 300) if reason else ""
        stamp = self.now()
        if new == NodeStatus.RUNNING and ns.started_at is None:
            ns.started_at = stamp
        if new in (NodeStatus.SUCCEEDED, NodeStatus.FAILED):
            ns.finished_at = stamp
        payload = dict(data)
        if reason and "reason" not in payload and new not in (NodeStatus.FAILED,):
            payload["reason"] = ns.reason
        self.emit(etype or self._STATUS_EVENT[new], node_id=node_id, attempt=attempt, **payload)
        self._dirty = True
        self._wave_progress(node_id, new)

    # ------------------------------------------------------------------ snapshots / views
    def snapshot(
        self,
        *,
        allowed_kinds: frozenset[str] | None = None,
        allowed_producers: frozenset[str] | None = None,
        extra: Sequence[Artifact] = (),
    ) -> RunSnapshot:
        ancestors = {n: frozenset(self.graph.ancestors(n)) for n in self.graph.nodes}
        return RunSnapshot(
            run_id=self.run_id,
            status=self.state.status,
            reason=self.state.status_reason,
            params=self.state.params,
            specs=self.graph.nodes,
            ancestors=ancestors,
            mandatory=tuple(self.state.mandatory_snapshot),
            statuses={k: v.status for k, v in self.state.nodes.items()},
            approvals=list(self.state.approvals.values()),
            waivers=list(self.state.waivers),
            artifacts=self.store.all(include_inactive=True),
            decision_log=self.decisions,
            decision_count=len(self.decisions),
            events_fn=self.audit.events,
            event_count=len(self.audit.events()),
            store=self.store,
            extra_artifacts=extra,
            allowed_kinds=allowed_kinds,
            allowed_producers=allowed_producers,
            meta={"open_questions": self.open_questions()},
        )

    def input_kinds(self, spec: NodeSpec) -> tuple[str, ...]:
        return _dedupe((*spec.input_kinds, *self.graph.overlay_of(spec.id)))

    def pin(self, spec: NodeSpec) -> PinnedReader:
        """Immutable input snapshot: ACTIVE artifacts of declared kinds from strict ancestors / external."""
        ancestors = frozenset(self.graph.ancestors(spec.id))
        return self.store.snapshot(
            self.input_kinds(spec),
            lambda producer: producer == EXTERNAL_PRODUCER or producer in ancestors,
        )

    def agent_for(self, spec: NodeSpec, epoch: Epoch) -> str:
        if epoch == Epoch.FALLBACK and spec.fallback_agent:
            return spec.fallback_agent
        return spec.agent

    def fingerprint(self, spec: NodeSpec, ns: NodeState, pinned: PinnedReader) -> str:
        agent_id = self.agent_for(spec, ns.epoch)
        version = str(self.registries.agents.create(agent_id).version)
        return str(
            compute_fingerprint(
                spec,
                self.graph,
                pinned,
                agent_version=version,
                epoch=ns.epoch,
                baseline_hash=self.state.baseline_manifest_hash,
                policy_hash=self.policy_hash(),
                feedback_keys=[f.key() for f in ns.feedback],
                engine_version=ENGINE_SCHEMA_VERSION,
            )
        )

    def view_for(self, job: _Job) -> FileView:
        """``baseline`` + ACTIVE overlay change sets of strict ancestors, ordered by (topo index, key)."""
        overlay_kinds = [k for k in self.graph_overlay(job) if k in CHANGESET_KINDS]
        arts: list[Artifact] = []
        for kind in overlay_kinds:
            arts.extend(a for a in job.pinned.active(kind) if a.producer != EXTERNAL_PRODUCER)
        arts.sort(key=lambda a: (job.topo_index.get(a.producer, 0), a.key))
        changesets = [ChangeSet.from_dict(job.pinned.read_json(a)) for a in arts]
        if not changesets:
            return self.baseline_view
        return OverlayView(self.baseline_view, changesets, strict=True)

    @staticmethod
    def graph_overlay(job: _Job) -> tuple[str, ...]:
        return _dedupe((*job.spec.overlay_from, *job.graph.overlay_of(job.spec.id)))

    def open_questions(self) -> tuple[ClarificationQuestion, ...]:
        """Blocking questions currently open (from the latest round of every waiting node)."""
        out: dict[str, ClarificationQuestion] = {}
        for node_id, ns in self.state.nodes.items():
            if ns.status != NodeStatus.AWAITING_INPUT:
                continue
            for rnd in reversed([r for r in self.state.clarifications if r.node_id == node_id]):
                for q in rnd.questions:
                    if q.id in ns.open_question_ids:
                        out.setdefault(q.id, q)
                break
        return tuple(out.values())

    # ------------------------------------------------------------------ class-level defaults
    _cache_hit_node: str | None = None
    _pending_stop: tuple[str, str] | None = None
    _default_factory_cache: ToolboxFactory | None = None
    _SETTLED_FOR_WAVE = frozenset(
        {
            NodeStatus.SUCCEEDED,
            NodeStatus.FAILED,
            NodeStatus.SKIPPED,
            NodeStatus.BLOCKED,
            NodeStatus.CANCELLED,
            NodeStatus.REMOVED,
            NodeStatus.ROLLED_BACK,
        }
    )

    # ------------------------------------------------------------------ re-plan waves
    def _open_wave(
        self,
        trigger: str,
        invalidated: Iterable[str],
        *,
        added: Iterable[str] = (),
        removed: Iterable[str] = (),
        retained: Iterable[str] = (),
        cancelled: Iterable[str] = (),
    ) -> None:
        names = sorted(set(invalidated))
        pending = {
            n for n in names if n in self.state.nodes and self.ns(n).status != NodeStatus.REMOVED
        }
        wave = _Wave(
            trigger,
            names,
            sorted(set(added)),
            sorted(set(removed)),
            sorted(set(retained)),
            sorted(set(cancelled)),
            pending=pending,
        )
        self._waves.append(wave)
        if not wave.pending:
            self._close_wave(wave)

    def _wave_progress(self, node_id: str, new: NodeStatus) -> None:
        if not self._waves or new not in self._SETTLED_FOR_WAVE:
            return
        for wave in list(self._waves):
            if node_id not in wave.pending:
                continue
            wave.pending.discard(node_id)
            if new == NodeStatus.SUCCEEDED:
                (wave.cache_hit if self._cache_hit_node == node_id else wave.re_run).append(node_id)
            if not wave.pending:
                self._close_wave(wave)

    def _close_wave(self, wave: _Wave, *, unsettled: Sequence[str] = ()) -> None:
        if wave in self._waves:
            self._waves.remove(wave)
        data: dict[str, Any] = {
            "trigger": wave.trigger,
            "invalidated": wave.invalidated,
            "cache_hit": sorted(wave.cache_hit),
            "re_run": sorted(wave.re_run),
            "added": wave.added,
            "removed": wave.removed,
            "retained": wave.retained,
            "cancelled_inflight": wave.cancelled_inflight,
            "unsettled": sorted(unsettled),
        }
        decision = self.decisions.record(
            DecisionKind.REPLAN,
            f"re-plan after {wave.trigger}: {len(wave.cache_hit)} reused, {len(wave.re_run)} re-run",
            actor=_SYSTEM,
            rationale=(
                "Minimal re-execution through input fingerprints; verdicts (gates, policy, approvals) "
                "were re-evaluated for every reused node."
            ),
            outcome=f"{len(wave.invalidated)} invalidated",
            data=data,
        )
        self.emit(
            EventType.PLAN_REPLANNED,
            decision_id=decision.id,
            added=wave.added,
            removed=wave.removed,
            retained=wave.retained,
            invalidated=wave.invalidated,
        )
        self._dirty = True

    def _flush_waves(self) -> None:
        for wave in list(self._waves):
            self._close_wave(wave, unsettled=sorted(wave.pending))

    # ------------------------------------------------------------------ housekeeping and safety
    def _housekeeping(self) -> None:
        mono = self.clock.monotonic()
        if self._last_mono is not None and mono > self._last_mono:
            self.state.active_runtime_s += mono - self._last_mono
        self._last_mono = mono
        if self._lock is not None:
            self._lock.heartbeat()
        now = self.now()
        last = self._last_heartbeat_persist
        if last is None or (now - last).total_seconds() >= HEARTBEAT_PERSIST_S:
            self.state.last_heartbeat = now
            self._last_heartbeat_persist = now
            self._dirty = True
        alive = [j for j in self._zombies if j.thread is not None and j.thread.is_alive()]
        self._zombies = alive
        self.state.counters.zombies = len(alive)

    def _check_safety(self) -> tuple[str, str] | None:
        if self._pending_stop is not None:
            return self._pending_stop
        acked = self.state.safe_stop.kill_acked if self.state.safe_stop else []
        for signal in self.kill_switch.signals():
            if _kill_signal_id(signal) in acked:
                continue
            self.emit(
                EventType.KILL_SWITCH_DETECTED,
                scope=signal.scope,
                by=_clip(signal.by, 64),
                reason=_clip(signal.reason, 200),
            )
            return (
                "kill_switch",
                f"kill switch tripped by {_clip(signal.by, 64)}: {_clip(signal.reason, 200)}",
            )
        trigger = self.safe.check(self.state, self.state.active_runtime_s)
        if trigger is not None:
            return trigger.trigger, trigger.reason
        return None

    def request_safe_stop(self, trigger: str, reason: str) -> None:
        if self._pending_stop is None:
            self._pending_stop = (trigger, _clip(reason, 300))

    # ------------------------------------------------------------------ commands (live, same process)
    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                result = self.apply_command(cmd.kind, **cmd.params)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
                cmd.future.set_exception(exc)
            else:
                cmd.future.set_result(result)

    def apply_command(self, kind: str, **params: Any) -> Any:
        handlers: dict[str, Callable[..., Any]] = {
            "approve": self.cmd_vote,
            "reject": self.cmd_vote,
            "answer": self.cmd_answer,
            "amend": self.cmd_amend,
        }
        handler = handlers.get(kind)
        if handler is None:
            raise StateError(f"unknown command {_clip(kind, 40)!r}")
        if self.state.status.is_terminal:
            raise StateError(
                f"run {self.run_id} is {self.state.status.value}; terminal runs are immutable"
            )
        result = handler(**params)
        self._dirty = True
        return result

    def cmd_vote(
        self,
        *,
        request_id: str,
        approver: str,
        approved: bool,
        comment: str = "",
        confirm: str = "",
        channel: str = "cli",
        os_user: str = "",
    ) -> dict[str, Any]:
        req = self.state.approvals.get(request_id)
        if req is None:
            raise ApprovalError(f"unknown approval request {_clip(request_id, 40)!r}")
        ns = self.state.nodes.get(req.node_id)
        updated = self.approvals.vote(
            request_id,
            approver,
            approved,
            _clip(comment, 400),
            channel=channel,
            os_user=os_user or _os_user(),
            confirm=confirm,
            current_fingerprint=ns.fingerprint if ns is not None else None,
        )
        return {
            "request_id": updated.id,
            "status": updated.status.value,
            "votes": len(updated.votes),
        }

    def _poll_providers(self) -> None:
        provider = self.orch.approver
        if provider is None:
            return
        pending = [r for r in self.state.approvals.values() if r.status == ApprovalStatus.PENDING]
        if not pending:
            return
        for vote in provider.poll(pending):
            req = self.state.approvals.get(vote.request_id)
            if req is None:
                continue
            ns = self.state.nodes.get(req.node_id)
            try:
                self.approvals.vote(
                    vote.request_id,
                    vote.approver,
                    vote.approved,
                    _clip(vote.comment, 400),
                    channel=vote.channel,
                    os_user=_os_user(),
                    current_fingerprint=ns.fingerprint if ns is not None else None,
                )
            except ApprovalError:
                continue  # already audited as APPROVAL_REFUSED
            self._dirty = True

    # ------------------------------------------------------------------ promotion (PENDING / STALE)
    def _skip_reason(self, spec: NodeSpec) -> str | None:
        if not spec.skip_if:
            return None
        predicate = self.registries.skips.get(spec.skip_if)
        view = self.snapshot(
            allowed_kinds=frozenset(spec.skip_inputs),
            allowed_producers=frozenset({*self.graph.ancestors(spec.id), EXTERNAL_PRODUCER}),
        )
        try:
            reason = predicate(view, spec)
        except Exception as exc:  # noqa: BLE001 - fail closed: a broken predicate never skips work
            self.emit(
                EventType.AGENT_NOTE,
                node_id=spec.id,
                message=f"skip predicate {spec.skip_if!r} failed ({type(exc).__name__}); node will run",
            )
            return None
        return _clip(reason, 200) if reason else None

    def _dep_statuses(self, node_id: str) -> tuple[list[str], list[NodeStatus]]:
        deps = list(self.graph.deps(node_id))
        return deps, [self.ns(d).status for d in deps]

    def _promote(self) -> None:
        for node_id in self.graph.topological_order():
            ns = self.ns(node_id)
            spec = self.graph.node(node_id)
            if ns.status == NodeStatus.PENDING:
                self._promote_pending(spec, ns)
            elif ns.status == NodeStatus.STALE and node_id not in self._jobs:
                self._promote_stale(spec, ns)

    def _promote_pending(self, spec: NodeSpec, ns: NodeState) -> None:
        deps, statuses = self._dep_statuses(spec.id)
        bad = [
            d
            for d, st in zip(deps, statuses, strict=True)
            if st
            in (NodeStatus.FAILED, NodeStatus.BLOCKED, NodeStatus.ROLLED_BACK, NodeStatus.CANCELLED)
        ]
        if bad:
            self.transition(
                spec.id, NodeStatus.BLOCKED, f"upstream {bad[0]} is {self.ns(bad[0]).status.value}"
            )
            return
        if not all(st.satisfies_dependency for st in statuses):
            return
        reason = self._skip_reason(spec)
        if reason:
            self.transition(spec.id, NodeStatus.SKIPPED, reason)
            return
        ns.not_before = None
        self.transition(spec.id, NodeStatus.READY, "dependencies satisfied")

    def _reset_epoch(self, ns: NodeState) -> None:
        ns.epoch_no += 1
        ns.epoch = Epoch.PRIMARY
        ns.primary_attempts = 0
        ns.fallback_attempts = 0
        ns.not_before = None

    def _promote_stale(self, spec: NodeSpec, ns: NodeState) -> None:
        _, statuses = self._dep_statuses(spec.id)
        if any(
            st
            in (NodeStatus.FAILED, NodeStatus.BLOCKED, NodeStatus.ROLLED_BACK, NodeStatus.CANCELLED)
            for st in statuses
        ):
            self._reset_epoch(ns)
            self.transition(spec.id, NodeStatus.PENDING, "upstream failed during revalidation")
            return
        if not all(st.satisfies_dependency for st in statuses):
            return
        pinned = self.pin(spec)
        fingerprint = self.fingerprint(spec, ns, pinned)
        if self._cacheable_hit(spec, ns, fingerprint):
            self._start_revalidation(spec, ns, pinned, fingerprint)
            return
        self._reset_epoch(ns)
        self.transition(spec.id, NodeStatus.READY, "inputs changed: re-run")

    def _cacheable_hit(self, spec: NodeSpec, ns: NodeState, fingerprint: str) -> bool:
        if not spec.cacheable or spec.side_effecting or not ns.outputs:
            return False
        if fingerprint != ns.last_success_fingerprint:
            return False
        for ref in ns.outputs:
            try:
                art = self.store.get(ref.artifact_id)
            except (KeyError, OrchestratorError):
                return False
            if art.status != ArtifactStatus.ACTIVE or art.content_hash != ref.content_hash:
                return False
        return True

    def _new_job(
        self,
        kind: str,
        spec: NodeSpec,
        ns: NodeState,
        pinned: PinnedReader,
        fingerprint: str,
        *,
        restored: tuple[Artifact, ...] = (),
    ) -> _Job:
        agent_id = self.agent_for(spec, ns.epoch)
        agent = self.registries.agents.create(agent_id)
        profile = agent.autonomy.narrow(spec.autonomy)
        deadline = min(spec.timeout_s, profile.max_runtime_s, self.cfg.max_attempt_s)
        return _Job(
            kind=kind,
            spec=spec,
            generation=ns.generation,
            attempt_no=ns.attempt_counter,
            epoch=ns.epoch,
            agent_id=agent_id,
            cancel=CancelToken(),
            pinned=pinned,
            fingerprint=fingerprint,
            feedback=tuple(ns.feedback),
            snapshot=self.snapshot(),
            policy=self._policy_now(),
            graph=self.graph.copy(),
            topo_index={n: self.graph.topo_index(n) for n in self.graph.nodes},
            mandatory=tuple(self.state.mandatory_snapshot),
            deadline_s=float(deadline),
            restored=restored,
        )

    def _start_job(self, job: _Job) -> None:
        job.started_mono = self.clock.monotonic()
        job.thread = threading.Thread(
            target=self._worker,
            args=(job,),
            name=f"agentic-{job.spec.id}-{job.attempt_no}",
            daemon=True,
        )
        self._jobs[job.spec.id] = job
        job.thread.start()

    def _start_revalidation(
        self, spec: NodeSpec, ns: NodeState, pinned: PinnedReader, fingerprint: str
    ) -> None:
        restored = tuple(self.store.get(r.artifact_id) for r in ns.outputs)
        ns.generation += 1
        job = self._new_job("revalidate", spec, ns, pinned, fingerprint, restored=restored)
        self._start_job(job)

    # ------------------------------------------------------------------ dispatch (READY)
    def _gate_context(
        self,
        job: _Job,
        phase: GatePhase,
        *,
        produced: Sequence[Artifact] = (),
        view: FileView | None = None,
    ) -> GateContext:
        reader = job.pinned.with_extra(produced) if produced else job.pinned
        run_view = job.snapshot.with_extra(produced) if produced else job.snapshot
        base_view = view if view is not None else self.baseline_view

        def workspace_view() -> FileView:
            changesets = [
                ChangeSet.from_dict(reader.read_json(a))
                for a in produced
                if a.kind in CHANGESET_KINDS
            ]
            return OverlayView(base_view, changesets, strict=True) if changesets else base_view

        return GateContext(
            run_id=self.run_id,
            node=job.spec,
            phase=phase,
            attempt=job.attempt_no,
            params={},
            artifacts=job.pinned,
            produced=tuple(produced),
            reader=reader,
            policy=job.policy,
            baseline=self.baseline_view,
            workspace_view=workspace_view,
            run_view=run_view,
            clock=self.clock,
        )

    def _dispatch(self) -> None:
        for node_id in self.graph.topological_order():
            if self._pending_stop is not None:
                return
            if len(self._jobs) >= self.cfg.max_parallel:
                return
            if len(self._zombies) >= self.cfg.max_zombie_threads:
                return
            ns = self.ns(node_id)
            if ns.status != NodeStatus.READY or node_id in self._jobs:
                continue
            if ns.not_before is not None and self.now() < ns.not_before:
                continue
            self._dispatch_one(self.graph.node(node_id), ns)

    def _fail_fast(self, spec: NodeSpec, ns: NodeState, kind: FailureKind, message: str) -> None:
        ns.last_failure_kind = kind
        self.transition(
            spec.id,
            NodeStatus.FAILED,
            message,
            kind=kind.value,
            message=_clip(message, 300),
            phase="entry",
        )
        self._compensate_for_failure(spec.id)

    def _compensate_for_failure(self, node_id: str) -> None:
        """Saga step: side-effecting ancestors named in ``NodeSpec.compensates`` are rolled back."""
        spec = self.graph.node(node_id)
        for target in spec.compensates:
            if target in self.state.nodes and self.ns(target).status == NodeStatus.SUCCEEDED:
                self.compensate(target, reason=f"{node_id} failed")

    def _dispatch_one(self, spec: NodeSpec, ns: NodeState) -> None:
        pinned = self.pin(spec)
        missing = [k for k in spec.consumes if not pinned.active(k)]
        if missing:
            self._fail_fast(
                spec,
                ns,
                FailureKind.GATE_REJECTED,
                f"missing required input(s): {', '.join(sorted(missing))}",
            )
            return
        fingerprint = self.fingerprint(spec, ns, pinned)
        ns.fingerprint = fingerprint
        if spec.entry_gates:
            probe = self._new_job("attempt", spec, ns, pinned, fingerprint)
            gate_results = run_gates(
                spec.entry_gates, self.registries.gates, self._gate_context(probe, GatePhase.ENTRY)
            )
            for gr in gate_results:
                self.emit(
                    EventType.GATE_EVALUATED,
                    node_id=spec.id,
                    gate=gr.gate,
                    phase="entry",
                    decision=gr.decision.value,
                    rule_ids=sorted({f.rule_id for f in gr.findings}),
                )
            comp = compose(gate_results)
            if comp.decision == GateDecision.SAFE_STOP:
                self.request_safe_stop(
                    "critical_finding", f"entry gate of {spec.id} raised a critical finding"
                )
                return
            if comp.decision == GateDecision.FAIL:
                rules = ", ".join(sorted({f.rule_id for f in comp.findings})) or "entry gate"
                self._fail_fast(
                    spec, ns, FailureKind.GATE_REJECTED, f"entry gate rejected the node ({rules})"
                )
                return
            if comp.decision == GateDecision.AWAIT_INPUT:
                self._fail_fast(
                    spec,
                    ns,
                    FailureKind.GATE_REJECTED,
                    "entry gate requires human input before this node can start",
                )
                return
        if self._before_approval_gate(spec, ns, pinned, fingerprint):
            return
        if ns.lifetime_attempts >= self.cfg.max_attempts_per_node:
            self._fail_fast(
                spec,
                ns,
                FailureKind.FATAL,
                f"lifetime attempt cap ({self.cfg.max_attempts_per_node}) reached",
            )
            return
        ns.generation += 1
        ns.attempt_counter += 1
        ns.lifetime_attempts += 1
        if ns.epoch == Epoch.PRIMARY:
            ns.primary_attempts += 1
        else:
            ns.fallback_attempts += 1
        job = self._new_job("attempt", spec, ns, pinned, fingerprint)
        ns.attempts.append(
            AttemptRecord(
                number=ns.attempt_counter,
                generation=ns.generation,
                epoch=ns.epoch,
                agent_id=job.agent_id,
                started_at=self.now(),
                input_fingerprint=fingerprint,
                inputs=pinned.refs(),
            )
        )
        self.transition(
            spec.id,
            NodeStatus.RUNNING,
            "",
            attempt=ns.attempt_counter,
            agent_id=job.agent_id,
            epoch=ns.epoch.value,
            stage=spec.stage.value,
            generation=ns.generation,
        )
        self._start_job(job)

    def _before_approval_gate(
        self, spec: NodeSpec, ns: NodeState, pinned: PinnedReader, fingerprint: str
    ) -> bool:
        """Handle a BEFORE approval; returns True when the node must wait for humans."""
        approval = spec.approval
        if approval is None or approval.position != ApprovalPosition.BEFORE:
            return False
        risk = spec.risk
        if not self._approval_required(approval, risk, False):
            return False
        if self.approvals.valid_for(spec.id, fingerprint) is not None:
            return False
        inputs = list(pinned.refs())
        request = self.approvals.request(
            node_id=spec.id,
            position=ApprovalPosition.BEFORE,
            purpose=ApprovalPurpose.NODE_START,
            risk=risk,
            reasons=tuple(
                r for r in (approval.reason, "real side effect" if spec.side_effecting else "") if r
            ),
            summary=self._summary(spec, risk, (), None, inputs),
            evidence=tuple(inputs),
            node_fingerprint=fingerprint,
            evidence_hash=_start_evidence_hash(inputs, risk),
            attempt=ns.attempt_counter,
            generation=ns.generation,
        )
        ns.pending_approval_id = request.id
        self.transition(
            spec.id,
            NodeStatus.AWAITING_APPROVAL,
            "approval required before start",
            approval_id=request.id,
        )
        return True

    def _summary(
        self,
        spec: NodeSpec,
        risk: RiskLevel,
        reasons: Sequence[str],
        diff: Mapping[str, Any] | None,
        evidence: Sequence[ArtifactRef],
        findings: Sequence[Finding] = (),
    ) -> str:
        return compose_summary(
            node=spec,
            risk=risk,
            reasons=tuple(reasons),
            findings=tuple(findings),
            diff_stats=diff,
            evidence=tuple(evidence),
            assurance=self.state.assurance,
            requested_by=self.state.created_by,
        )

    def _approval_required(
        self, approval: ApprovalSpec, risk: RiskLevel, has_waivable: bool
    ) -> bool:
        mode = self.cfg.autonomy_mode
        threshold = approval.threshold
        if mode == AutonomyMode.AUTONOMOUS and not approval.always:
            threshold = max(threshold, RiskLevel.HIGH)
        if approval.always or mode == AutonomyMode.SUPERVISED:
            return True
        return risk >= threshold or has_waivable

    # ------------------------------------------------------------------ worker thread
    def _worker(self, job: _Job) -> None:
        started = self.now()
        try:
            completion = (
                self._revalidate(job, started)
                if job.kind == "revalidate"
                else self._execute(job, started)
            )
        except BaseException as exc:  # noqa: BLE001 - nothing may escape a worker thread
            completion = self._failed_completion(job, started, exc)
        self._completions.put(completion)

    def _classify(self, exc: BaseException) -> FailureInfo:
        text = self.redactor.redact_text(_clip(f"{type(exc).__name__}: {exc}", 300))
        if isinstance(exc, AutonomyViolation):
            return FailureInfo(
                FailureKind.AUTONOMY_VIOLATION,
                text,
                (
                    Finding(
                        "SEC-005",
                        PolicyCategory.SECURITY,
                        Severity.CRITICAL,
                        _clip(str(exc), 200),
                        remediation="Stay inside the agent's autonomy profile.",
                    ),
                ),
                retryable=False,
            )
        if isinstance(exc, AgentCancelled):
            return FailureInfo(FailureKind.CANCELLED, text, retryable=False)
        if isinstance(exc, AgentTimeout):
            return FailureInfo(FailureKind.TIMEOUT, text)
        if isinstance(exc, ToolDenied):
            return FailureInfo(
                FailureKind.POLICY_VIOLATION,
                text,
                (
                    Finding(
                        "SEC-005",
                        PolicyCategory.SECURITY,
                        Severity.HIGH,
                        _clip(str(exc), 200),
                        remediation="Only write inside the agent's write_globs.",
                    ),
                ),
            )
        if isinstance(exc, PolicyDenied):
            return FailureInfo(
                FailureKind.POLICY_VIOLATION,
                text,
                tuple(f for f in exc.findings if isinstance(f, Finding)),
            )
        if isinstance(exc, ChangeSetConflict):
            return FailureInfo(
                FailureKind.GATE_REJECTED,
                text,
                (
                    _engine_finding(
                        "CHG-009",
                        f"change set conflicts with its base: {exc.reason}",
                        location=exc.path,
                    ),
                ),
            )
        if isinstance(exc, TransientAgentError):
            return FailureInfo(FailureKind.TRANSIENT, text)
        if isinstance(exc, AgentError):
            return FailureInfo(FailureKind.parse(exc.kind), text, retryable=exc.retryable)
        return FailureInfo(FailureKind.FATAL, text, retryable=False)

    def _failed_completion(self, job: _Job, started: datetime, exc: BaseException) -> _Completion:
        outcome = AttemptOutcome(
            node_id=job.spec.id,
            generation=job.generation,
            attempt_no=job.attempt_no,
            epoch=job.epoch,
            agent_id=job.agent_id,
            started_at=started,
            ended_at=self.now(),
            failure=self._classify(exc),
            inputs=job.pinned.refs(),
        )
        return _Completion(job, outcome, self._blank_pending(job))

    def _blank_pending(self, job: _Job) -> _Pending:
        return _Pending(
            node_id=job.spec.id,
            attempt_no=job.attempt_no,
            generation=job.generation,
            epoch=job.epoch.value,
            agent_id=job.agent_id,
            fingerprint=job.fingerprint,
            inputs=[r.to_dict() for r in job.pinned.refs()],
        )

    @property
    def toolbox_factory(self) -> ToolboxFactory:
        if self.orch.toolbox_factory is not None:
            return self.orch.toolbox_factory
        if self._default_factory_cache is None:
            self._default_factory_cache = make_toolbox_factory(
                materialise_dir=self.run_dir / "sandbox"
            )
        return self._default_factory_cache

    def _make_toolbox(self, job: _Job, profile: AutonomyProfile, view: FileView) -> Any:
        factory = self.toolbox_factory
        kwargs: dict[str, Any] = {
            "run_id": self.run_id,
            "node_id": job.spec.id,
            "attempt": job.attempt_no,
            "agent_id": job.agent_id,
            "profile": profile,
            "view": view,
            "cancel": job.cancel,
            "clock": self.clock,
            "policy": job.policy,
            "audit": self.audit.emit,
        }
        try:
            takes_baseline = "baseline" in inspect.signature(factory).parameters
        except (TypeError, ValueError):
            takes_baseline = False
        if takes_baseline:
            kwargs["baseline"] = self.baseline_view
        return factory(**kwargs)

    def _effects_for(self, job: _Job) -> EffectPort:
        root = self.state.apply_root
        if not root:
            raise AgentError(
                "this node performs a real side effect but the run has no apply_root",
                kind="fatal",
                retryable=False,
            )
        factory = self.orch.effect_factory or _default_effect_factory(self.orch.policy)
        inner = factory(Path(root), self.audit.emit, job.spec.id, job.attempt_no, job.cancel)
        return _RecordingEffects(inner, job.receipts)

    def _execute(self, job: _Job, started: datetime) -> _Completion:
        spec = job.spec
        agent = self.registries.agents.create(job.agent_id)
        profile = agent.autonomy.narrow(spec.autonomy)
        view = self.view_for(job)
        toolbox = self._make_toolbox(job, profile, view)
        ns = self.ns(spec.id)
        ctx = AgentContext(
            run_id=self.run_id,
            node=spec,
            attempt=ns.primary_attempts if job.epoch == Epoch.PRIMARY else ns.fallback_attempts,
            attempt_no=job.attempt_no,
            generation=job.generation,
            agent_id=job.agent_id,
            is_fallback=job.epoch == Epoch.FALLBACK,
            params=spec.params,
            artifacts=job.pinned,
            toolbox=toolbox,
            clock=self.clock,
            cancel=job.cancel,
            run_view=job.snapshot.governance(),
            feedback=job.feedback,
            effects=self._effects_for(job) if spec.side_effecting else None,
        )
        result = agent.run(ctx)
        if not isinstance(result, AgentResult):
            raise TypeError("agent returned a malformed result")
        job.cancel.raise_if_cancelled()
        staged, contract, expansion = self._stage(job, toolbox, result)
        gctx = self._gate_context(job, GatePhase.EXIT, produced=staged, view=view)
        implicit = run_gates(
            (GateSpec("artifacts_present"), GateSpec("policy", {"profile": profile.to_dict()})),
            self.registries.gates,
            gctx,
        )
        explicit = run_gates(spec.exit_gates, self.registries.gates, gctx)
        results = [*implicit, *explicit]
        if contract:
            results.append(
                GateResult(
                    "contract", GateDecision.FAIL, "output contract violations", tuple(contract)
                )
            )
        comp = compose(results)
        levels = [spec.risk, *(r.risk.level for r in results if r.risk is not None)]
        if any(f.severity == Severity.MEDIUM for f in comp.findings):
            levels.append(RiskLevel.MEDIUM)
        reasons = tuple(
            dict.fromkeys(x for r in results if r.risk is not None for x in r.risk.reasons)
        )
        risk = RiskAssessment(max(levels), reasons)
        outcome = AttemptOutcome(
            node_id=spec.id,
            generation=job.generation,
            attempt_no=job.attempt_no,
            epoch=job.epoch,
            agent_id=job.agent_id,
            started_at=started,
            ended_at=self.now(),
            result=result,
            staged=tuple(a.ref for a in staged),
            gate_results=tuple(results),
            risk=risk,
            inputs=job.pinned.refs(),
            usage={**{str(k): float(v) for k, v in result.usage.items()}, **dict(toolbox.usage())},
        )
        pending = self._blank_pending(job)
        pending.decisions = [_draft_to_dict(d) for d in result.decisions[:MAX_DRAFT_DECISIONS]]
        pending.questions = [q.to_dict() for q in result.clarifications[:MAX_QUESTIONS]]
        pending.expansion = expansion
        pending.rationale = _clip(result.rationale, MAX_AGENT_NOTE)
        pending.findings = [f.to_dict() for f in comp.findings]
        pending.waivable = [f.to_dict() for f in comp.findings if f.severity == Severity.MEDIUM]
        pending.risk = risk.to_dict()
        pending.usage = dict(outcome.usage)
        pending.diff_stats = self._diff_stats(view, staged, job)
        pending.receipt = dict(job.receipts[-1]) if job.receipts else None
        return _Completion(job, outcome, pending, tuple(result.clarifications[:MAX_QUESTIONS]))

    def _diff_stats(
        self, view: FileView, staged: Sequence[Artifact], job: _Job
    ) -> dict[str, Any] | None:
        stats: dict[str, int] | None = None
        for art in staged:
            if art.kind not in CHANGESET_KINDS:
                continue
            try:
                cs = ChangeSet.from_dict(job.pinned.with_extra([art]).read_json(art))
                one = diff_stats(view, cs).to_dict()
            except (ValueError, OrchestratorError):
                continue
            stats = one if stats is None else {k: stats[k] + one[k] for k in stats}
        return stats

    def _stage(
        self, job: _Job, toolbox: Any, result: AgentResult
    ) -> tuple[list[Artifact], list[Finding], dict[str, Any] | None]:
        """Validate and STAGE the agent's drafts; engine-detected contract violations become findings."""
        spec = job.spec
        findings: list[Finding] = []
        staged: list[Artifact] = []
        produced_kinds = set(spec.produces)
        seen: set[tuple[str, str]] = set()
        staged_ops = tuple(toolbox.staged_changeset().ops)
        for draft in result.artifacts:
            if draft.kind == Kind.PLAN_EXPANSION:
                findings.append(
                    _engine_finding(
                        "GATE-001",
                        "agents must not publish plan_expansion directly; return plan_expansion instead",
                    )
                )
                continue
            if draft.kind not in produced_kinds:
                findings.append(
                    _engine_finding(
                        "GATE-001",
                        f"artifact kind {_clip(draft.kind, 40)!r} is not declared in produces",
                        location=spec.id,
                    )
                )
                continue
            key = spec.id if draft.kind in KEYED_KINDS else ""
            if (draft.kind, key) in seen:
                findings.append(
                    _engine_finding(
                        "GATE-001", f"duplicate artifact {draft.kind}/{key}", location=spec.id
                    )
                )
                continue
            content: Any = draft.content
            if draft.kind in CHANGESET_KINDS:
                try:
                    raw = content if isinstance(content, Mapping) else loads_strict(content)
                    cs = ChangeSet.from_dict(raw)
                except (ValueError, KeyError, TypeError, OrchestratorError) as exc:
                    findings.append(
                        _engine_finding(
                            "GATE-002",
                            f"invalid change set: {type(exc).__name__}: {_clip(exc, 160)}",
                            location=spec.id,
                        )
                    )
                    continue
                cs = ChangeSet(cs.ops, cs.description, job.agent_id)
                if staged_ops and changeset_digest(cs) != changeset_digest(ChangeSet(staged_ops)):
                    findings.append(
                        _engine_finding(
                            "SEC-005",
                            "returned change set differs from the writes staged through the toolbox",
                            location=spec.id,
                        )
                    )
                    continue
                content = cs.to_dict()
            elif isinstance(content, (Mapping, list, tuple)):
                problems = validate_payload(draft.kind, to_jsonable(content))
                if problems:
                    findings.append(
                        _engine_finding(
                            "GATE-002",
                            f"invalid {draft.kind} payload: {_clip('; '.join(problems[:3]), 240)}",
                            location=spec.id,
                        )
                    )
                    continue
            metadata = {k: v for k, v in dict(draft.metadata).items() if not str(k).startswith("_")}
            try:
                art = self.store.put(
                    ArtifactDraft(
                        kind=draft.kind,
                        content=content,
                        key=key,
                        media_type=draft.media_type,
                        metadata=metadata,
                    ),
                    producer=spec.id,
                    attempt=job.attempt_no,
                    agent_id=job.agent_id,
                    inputs=job.pinned.refs(),
                    staged=True,
                )
            except (ConfigError, OrchestratorError, ValueError) as exc:
                findings.append(
                    _engine_finding(
                        "GATE-002",
                        f"artifact rejected by the store: {_clip(exc, 200)}",
                        location=spec.id,
                    )
                )
                continue
            staged.append(art)
            seen.add((draft.kind, key))
        known = {d.id for d in self.decisions.all()}
        for dd in result.decisions[:MAX_DRAFT_DECISIONS]:
            if dd.kind not in _AGENT_DECISION_KINDS:
                findings.append(
                    _engine_finding(
                        "GATE-003",
                        f"agents may not record {dd.kind.value!r} decisions",
                        location=spec.id,
                    )
                )
            unknown = [p for p in dd.parents if p not in known]
            if unknown:
                findings.append(
                    _engine_finding(
                        "GATE-003",
                        f"decision cites unknown parent(s): {_clip(unknown[:3], 80)}",
                        location=spec.id,
                    )
                )
            if not all(_safe_ref(r) for r in dd.refs):
                findings.append(
                    _engine_finding(
                        "GATE-003", "decision refs contain unsafe identifiers", location=spec.id
                    )
                )
        expansion_dict: dict[str, Any] | None = None
        if result.plan_expansion is not None or Kind.PLAN_EXPANSION in produced_kinds:
            raw_expansion = result.plan_expansion or PlanExpansion(owner=spec.id)
            normalised, exp_findings = normalize_expansion(spec, raw_expansion, job.graph, self.cfg)
            findings.extend(exp_findings)
            if not exp_findings:
                try:
                    job.graph.copy().apply_expansion(normalised)
                except (PlanPatchError, GraphError) as exc:
                    findings.append(
                        _engine_finding(
                            "CHG-007",
                            f"plan expansion rejected: {_clip(exc, 200)}",
                            location=spec.id,
                        )
                    )
                else:
                    expansion_dict = normalised.to_dict()
                    staged.append(
                        self.store.put(
                            ArtifactDraft(kind=Kind.PLAN_EXPANSION, content=expansion_dict),
                            producer=spec.id,
                            attempt=job.attempt_no,
                            agent_id=job.agent_id,
                            inputs=job.pinned.refs(),
                            staged=True,
                        )
                    )
        return staged, findings, expansion_dict

    def _revalidate(self, job: _Job, started: datetime) -> _Completion:
        """Cache-hit revalidation: verdicts are never restored, only outputs (engine-semantics 4.4)."""
        spec = job.spec
        agent = self.registries.agents.create(job.agent_id)
        profile = agent.autonomy.narrow(spec.autonomy)
        view = self.view_for(job)
        gctx = self._gate_context(job, GatePhase.EXIT, produced=job.restored, view=view)
        implicit = run_gates(
            (GateSpec("artifacts_present"), GateSpec("policy", {"profile": profile.to_dict()})),
            self.registries.gates,
            gctx,
        )
        explicit = run_gates(spec.exit_gates, self.registries.gates, gctx)
        outcome = AttemptOutcome(
            node_id=spec.id,
            generation=job.generation,
            attempt_no=job.attempt_no,
            epoch=job.epoch,
            agent_id=job.agent_id,
            started_at=started,
            ended_at=self.now(),
            staged=tuple(a.ref for a in job.restored),
            gate_results=(*implicit, *explicit),
            inputs=job.pinned.refs(),
        )
        return _Completion(job, outcome, self._blank_pending(job))

    # ------------------------------------------------------------------ applying outcomes
    def _drain_completions(self) -> None:
        backlog, self._backlog = self._backlog, []
        for completion in backlog:
            self._apply_completion(completion)
        while True:
            try:
                completion = self._completions.get_nowait()
            except queue.Empty:
                return
            self._apply_completion(completion)

    def _attempt_record(self, ns: NodeState, number: int) -> AttemptRecord | None:
        for rec in reversed(ns.attempts):
            if rec.number == number:
                return rec
        return None

    def _apply_completion(self, c: _Completion) -> None:
        job, out = c.job, c.outcome
        self._zombies = [z for z in self._zombies if z is not job]
        if self._jobs.get(job.spec.id) is job:
            del self._jobs[job.spec.id]
        ns = self.state.nodes.get(job.spec.id)
        if job.kind == "revalidate":
            self._apply_revalidation(c, ns)
            return
        if (
            ns is None
            or out.generation != ns.generation
            or ns.status != NodeStatus.RUNNING
            or self._pending_stop is not None
        ):
            self._discard(
                c, ns, "stale generation" if self._pending_stop is None else "run is stopping"
            )
            return
        self._apply_attempt(c, ns)

    def _discard(self, c: _Completion, ns: NodeState | None, reason: str) -> None:
        out = c.outcome
        self.state.counters.discarded_attempts += 1
        self.emit(
            EventType.NODE_ATTEMPT_DISCARDED,
            node_id=out.node_id,
            attempt=out.attempt_no,
            generation=out.generation,
            reason=reason,
        )
        self._quarantine_attempt_artifacts(out.node_id, out.attempt_no, f"discarded: {reason}")
        if ns is not None:
            rec = self._attempt_record(ns, out.attempt_no)
            if rec is not None:
                rec.outcome = AttemptOutcomeKind.DISCARDED
                rec.ended_at = rec.ended_at or out.ended_at
                rec.counts_vs_budget = False
        self._dirty = True

    def _quarantine_attempt_artifacts(
        self, node_id: str, attempt_no: int, reason: str, *, tombstone: bool = False
    ) -> None:
        ids = [
            a.id for a in self.store.staged() if a.producer == node_id and a.attempt == attempt_no
        ]
        if ids:
            self.store.quarantine(ids, reason, tombstone=tombstone)

    def _record_gate_events(self, spec: NodeSpec, out: AttemptOutcome) -> None:
        seen: dict[str, Finding] = {}
        for gr in out.gate_results:
            self.emit(
                EventType.GATE_EVALUATED,
                node_id=spec.id,
                attempt=out.attempt_no,
                gate=gr.gate,
                phase="exit",
                decision=gr.decision.value,
                rule_ids=sorted({f.rule_id for f in gr.findings}),
            )
            evaluations = (
                gr.evidence.get("policy_evaluations") if isinstance(gr.evidence, Mapping) else None
            )
            for ev in evaluations or ():
                if isinstance(ev, Mapping):
                    self.emit(
                        EventType.POLICY_EVALUATED,
                        node_id=spec.id,
                        attempt=out.attempt_no,
                        subject=_clip(ev.get("subject", ""), 120),
                        subject_hash=_clip(ev.get("subject_hash", ""), 80),
                        rules_evaluated=list(ev.get("rules_evaluated", ()))[:60],
                        findings=list(ev.get("findings", ()))[:50],
                    )
            for f in gr.findings:
                seen.setdefault(f.key(), f)
        for f in seen.values():
            if f.severity >= Severity.LOW:
                self.emit(
                    EventType.POLICY_VIOLATION,
                    node_id=spec.id,
                    attempt=out.attempt_no,
                    rule_id=f.rule_id,
                    severity=f.severity.name,
                    enforcement=DEFAULT_ENFORCEMENT_NAMES[f.severity],
                    location=_clip(f.location, 160) if f.location else "",
                )

    def _apply_attempt(self, c: _Completion, ns: NodeState) -> None:
        job, out, pending = c.job, c.outcome, c.pending
        spec = job.spec
        rec = self._attempt_record(ns, out.attempt_no)
        if rec is not None:
            rec.ended_at = out.ended_at
        if out.failure is not None:
            self._apply_failure(c, ns, rec)
            return
        if rec is not None:
            rec.gate_results = tuple(out.gate_results)
            rec.risk = out.risk
            rec.staged = tuple(out.staged)
        self._record_gate_events(spec, out)
        comp = compose(out.gate_results)
        if comp.decision == GateDecision.SAFE_STOP:
            self._quarantine_attempt_artifacts(
                spec.id, out.attempt_no, "critical finding", tombstone=True
            )
            if rec is not None:
                rec.outcome = AttemptOutcomeKind.REJECTED
                rec.findings = comp.findings
            self.request_safe_stop("critical_finding", f"{spec.id}: {_summarise(comp.findings)}")
            return
        if comp.decision == GateDecision.FAIL:
            tombstone = any(f.rule_id == "SEC-001" for f in comp.findings)
            self._quarantine_attempt_artifacts(
                spec.id, out.attempt_no, "rejected by gates", tombstone=tombstone
            )
            kind = (
                FailureKind.POLICY_VIOLATION
                if any(f.rule_id.split("-")[0] in _POLICY_PREFIXES for f in comp.findings)
                else FailureKind.GATE_REJECTED
            )
            message = f"output rejected: {_summarise(comp.findings)}"
            if rec is not None:
                rec.outcome = AttemptOutcomeKind.REJECTED
                rec.failure_kind = kind
                rec.message = _clip(message, 300)
                rec.findings = comp.findings
            self._emit_attempt_failed(
                spec, out.attempt_no, kind, message, job.epoch, True, comp.findings
            )
            self.decisions.record(
                DecisionKind.REJECTION,
                f"attempt {out.attempt_no} of {spec.id} rejected",
                actor=_SYSTEM,
                node_id=spec.id,
                rationale=_clip(message, 300),
                data={
                    "attempt": out.attempt_no,
                    "finding_keys": [f.key() for f in comp.findings][:30],
                    "kind": kind.value,
                },
            )
            self.safe.record_attempt(self.state, job.agent_id, kind)
            self._after_failure(spec, ns, kind, message, comp.findings, True, rejected=True)
            return
        questions = [q for q in (*c.questions, *comp.questions) if q.blocking]
        if comp.decision == GateDecision.AWAIT_INPUT or questions:
            merged = list({q.id: q for q in (*c.questions, *comp.questions)}.values())
            self._hold_for_input(
                spec, ns, merged, pending=pending, staged=out.staged, attempt_no=out.attempt_no
            )
            return
        self._approve_or_publish(c, ns, rec, comp.findings)

    def _emit_attempt_failed(
        self,
        spec: NodeSpec,
        attempt_no: int,
        kind: FailureKind,
        message: str,
        epoch: Epoch,
        retryable: bool,
        findings: Sequence[Finding],
    ) -> None:
        self.emit(
            EventType.NODE_ATTEMPT_FAILED,
            node_id=spec.id,
            attempt=attempt_no,
            kind=kind.value,
            message=_clip(message, 300),
            epoch=epoch.value,
            retryable=retryable,
            rule_ids=sorted({f.rule_id for f in findings})[:30],
        )

    def _apply_failure(self, c: _Completion, ns: NodeState, rec: AttemptRecord | None) -> None:
        job, out = c.job, c.outcome
        spec = job.spec
        failure = out.failure
        assert failure is not None
        if rec is not None:
            rec.outcome = (
                AttemptOutcomeKind.CANCELLED
                if failure.kind == FailureKind.CANCELLED
                else AttemptOutcomeKind.FAILED
            )
            rec.failure_kind = failure.kind
            rec.message = _clip(failure.message, 300)
            rec.findings = failure.findings
        self._quarantine_attempt_artifacts(
            spec.id, out.attempt_no, f"attempt failed ({failure.kind.value})"
        )
        self._emit_attempt_failed(
            spec,
            out.attempt_no,
            failure.kind,
            failure.message,
            job.epoch,
            failure.retryable,
            failure.findings,
        )
        self.safe.record_attempt(self.state, job.agent_id, failure.kind)
        if failure.kind == FailureKind.AUTONOMY_VIOLATION:
            self.request_safe_stop("autonomy_violation", f"{spec.id}: {failure.message}")
            return
        kind = FailureKind.TRANSIENT if failure.kind == FailureKind.CANCELLED else failure.kind
        self._after_failure(
            spec, ns, kind, failure.message, failure.findings, failure.retryable, rejected=False
        )

    def _after_failure(
        self,
        spec: NodeSpec,
        ns: NodeState,
        kind: FailureKind,
        message: str,
        findings: Sequence[Finding],
        retryable: bool,
        *,
        rejected: bool,
    ) -> None:
        """Retry (bounded, back-off) -> fallback agent -> remediation -> FAILED (engine-semantics 9)."""
        feedback = tuple(_finding_feedback(f) for f in findings[:20])
        if (
            rejected
            and spec.remediation is not None
            and kind in (FailureKind.GATE_REJECTED, FailureKind.POLICY_VIOLATION)
            and self._try_remediation(spec, ns, findings, feedback)
        ):
            return
        policy = spec.retry if ns.epoch == Epoch.PRIMARY else spec.fallback_retry
        used = ns.primary_attempts if ns.epoch == Epoch.PRIMARY else ns.fallback_attempts
        can_retry = (
            retryable
            and kind in policy.retry_on
            and kind != FailureKind.FATAL
            and used < policy.max_attempts
            and ns.lifetime_attempts < self.cfg.max_attempts_per_node
        )
        if can_retry:
            delay = policy.delay_for(used)
            ns.not_before = self.now() + timedelta(seconds=delay)
            ns.feedback = feedback
            ns.last_failure_kind = kind
            self.state.counters.retries += 1
            self.state.counters.budgeted_attempts += 1
            self.transition(
                spec.id,
                NodeStatus.READY,
                "retry scheduled",
                etype=EventType.NODE_RETRY_SCHEDULED,
                attempt=ns.attempt_counter,
                delay_s=round(delay, 3),
                not_before=_utc_iso(ns.not_before),
            )
            return
        if (
            ns.epoch == Epoch.PRIMARY
            and spec.fallback_agent
            and kind != FailureKind.AUTONOMY_VIOLATION
        ):
            ns.epoch = Epoch.FALLBACK
            ns.fallback_attempts = 0
            ns.not_before = None
            ns.feedback = feedback
            ns.last_failure_kind = kind
            self.state.counters.fallbacks += 1
            self.state.counters.budgeted_attempts += 1
            self.decisions.record(
                DecisionKind.FALLBACK,
                f"fallback agent {spec.fallback_agent} takes over {spec.id}",
                actor=_SYSTEM,
                node_id=spec.id,
                rationale=f"primary agent {spec.agent} exhausted its bounded attempts ({kind.value}): {_clip(message, 200)}",
                data={
                    "from_agent": spec.agent,
                    "to_agent": spec.fallback_agent,
                    "kind": kind.value,
                },
            )
            self.transition(
                spec.id,
                NodeStatus.READY,
                "fallback activated",
                etype=EventType.NODE_FALLBACK_ACTIVATED,
                attempt=ns.attempt_counter,
                from_agent=spec.agent,
                to_agent=spec.fallback_agent,
            )
            return
        ns.last_failure_kind = kind
        ns.feedback = ()
        self.transition(
            spec.id, NodeStatus.FAILED, message, kind=kind.value, message=_clip(message, 300)
        )
        self._compensate_for_failure(spec.id)

    def _try_remediation(
        self,
        spec: NodeSpec,
        ns: NodeState,
        findings: Sequence[Finding],
        feedback: tuple[Finding, ...],
    ) -> bool:
        rem = spec.remediation
        assert rem is not None
        if (
            ns.remediation_loops >= rem.max_loops
            or self.state.counters.remediation_loops >= self.cfg.max_remediation_loops
        ):
            return False
        targets = [
            t
            for t in resolve_remediation_targets(self.graph, rem)
            if t in self.state.nodes and self.ns(t).status == NodeStatus.SUCCEEDED
        ]
        if not targets:
            return False
        ns.remediation_loops += 1
        self.state.counters.remediation_loops += 1
        decision = self.decisions.record(
            DecisionKind.REMEDIATION,
            f"route the findings of {spec.id} back to {', '.join(targets)}",
            actor=_SYSTEM,
            node_id=spec.id,
            rationale="The defect is upstream of this node: re-running it on identical inputs cannot change the outcome.",
            data={
                "targets": targets,
                "finding_keys": [f.key() for f in findings][:30],
                "loop": ns.remediation_loops,
            },
        )
        self.emit(
            EventType.REMEDIATION_ROUTED,
            node_id=spec.id,
            targets=targets,
            decision_id=decision.id,
            loop=ns.remediation_loops,
        )
        self._reset_epoch(ns)
        ns.feedback = ()
        self.transition(spec.id, NodeStatus.PENDING, "remediation: waiting for upstream fixes")
        for target in targets:
            tns = self.ns(target)
            tns.feedback = feedback
            self.transition(target, NodeStatus.STALE, "remediation feedback")
            self._reset_epoch(tns)
            self.transition(target, NodeStatus.PENDING, "remediation feedback")
        downstream = invalidate(self.graph, self.state.nodes, set(), tuple(targets))
        self._apply_transitions(downstream, f"remediation of {spec.id}")
        self._open_wave(f"remediation of {spec.id}", [*targets, spec.id])
        return True

    # ------------------------------------------------------------------ approvals and publishing
    def _approve_or_publish(
        self, c: _Completion, ns: NodeState, rec: AttemptRecord | None, findings: Sequence[Finding]
    ) -> None:
        spec, out, pending = c.job.spec, c.outcome, c.pending
        risk = out.risk or RiskAssessment(spec.risk)
        waivable = pending.waivable_list()
        approval = (
            spec.approval
            if (spec.approval and spec.approval.position == ApprovalPosition.AFTER)
            else None
        )
        needed = (
            self._approval_required(approval, risk.level, bool(waivable))
            if approval is not None
            else bool(waivable)
        )
        if needed:
            self._write_pending(pending)
            ns.staged = tuple(out.staged)
            ns.pending_attempt = out.attempt_no
            reasons = [
                *([approval.reason] if approval is not None and approval.reason else []),
                *risk.reasons,
                *(f"waiver requested for {f.rule_id}" for f in waivable),
            ]
            request = self.approvals.request(
                node_id=spec.id,
                position=ApprovalPosition.AFTER,
                purpose=ApprovalPurpose.NODE_OUTPUT,
                risk=risk.level,
                reasons=tuple(reasons),
                summary=self._summary(
                    spec, risk.level, reasons, pending.diff_stats, out.staged, findings
                ),
                evidence=tuple(out.staged),
                node_fingerprint=c.job.fingerprint,
                evidence_hash=evidence_hash_of(out.staged, findings, risk.level),
                attempt=out.attempt_no,
                generation=out.generation,
                waives=tuple(waivable),
                staged=tuple(out.staged),
                agent_note=pending.rationale,
            )
            ns.pending_approval_id = request.id
            if rec is not None:
                rec.outcome = AttemptOutcomeKind.AWAITING_APPROVAL
            self.transition(
                spec.id,
                NodeStatus.AWAITING_APPROVAL,
                "approval required",
                attempt=out.attempt_no,
                approval_id=request.id,
            )
            return
        if approval is not None:
            self.emit(
                EventType.APPROVAL_AUTO,
                node_id=spec.id,
                risk=risk.level.name,
                reason="assessed risk is below the approval threshold",
            )
            self.decisions.record(
                DecisionKind.APPROVAL,
                f"automatic policy approval of {spec.id} (risk {risk.level.name})",
                actor=Actor.policy("approval-policy"),
                node_id=spec.id,
                rationale="Assessed risk is below the node's approval threshold; recorded, never silent.",
                inputs=out.inputs[:MAX_INPUT_REFS],
                data={"risk": risk.level.name, "automatic": True},
            )
        self._publish(spec, ns, pending, out.staged, rec)

    def _write_pending(self, pending: _Pending) -> None:
        path = self.pending_path(pending.node_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, pending.to_dict())
        self._pending_cache[pending.node_id] = pending

    def _load_pending(self, node_id: str) -> _Pending | None:
        cached = self._pending_cache.get(node_id)
        if cached is not None:
            return cached
        path = self.pending_path(node_id)
        try:
            raw = loads_strict(read_bytes_retry(path))
            pending = _Pending.from_dict(raw)
        except (OSError, ValueError, TypeError):
            return None
        self._pending_cache[node_id] = pending
        return pending

    def _clear_pending(self, node_id: str) -> None:
        self._pending_cache.pop(node_id, None)
        with contextlib.suppress(OSError):
            self.pending_path(node_id).unlink()

    def _quarantine_staged_of(self, ns: NodeState, reason: str = "superseded") -> None:
        if ns.staged:
            self.store.quarantine([r.artifact_id for r in ns.staged], reason)
        ns.staged = ()
        ns.pending_attempt = None
        ns.open_question_ids = ()
        self._clear_pending(ns.node_id)

    def _process_approvals(self) -> None:
        for req in list(self.state.approvals.values()):
            if req.status not in (ApprovalStatus.APPROVED, ApprovalStatus.REJECTED):
                continue
            ns = self.state.nodes.get(req.node_id)
            if (
                ns is None
                or ns.status != NodeStatus.AWAITING_APPROVAL
                or ns.pending_approval_id != req.id
            ):
                continue
            if req.status == ApprovalStatus.APPROVED:
                self._on_approved(req, ns)
            else:
                self._on_rejected(req, ns)

    def _approvers(self, req: ApprovalRequest, *, approved: bool = True) -> list[str]:
        return [v.approver for v in req.votes if v.approved == approved]

    def _on_approved(self, req: ApprovalRequest, ns: NodeState) -> None:
        spec = self.graph.node(req.node_id)
        approvers = self._approvers(req)
        scripted = any(v.channel == "scripted" for v in req.votes)
        last = next((v.comment for v in reversed(req.votes) if v.approved and v.comment), "")
        decision = self.decisions.record(
            DecisionKind.APPROVAL,
            f"{', '.join(approvers)} approved {spec.id} ({req.purpose.value}, risk {req.risk.name})",
            actor=Actor.human(approvers[0] if approvers else "unknown"),
            node_id=spec.id,
            rationale=_clip(last, 300),
            inputs=req.evidence[:MAX_INPUT_REFS],
            data={
                "approval_id": req.id,
                "risk": req.risk.name,
                "approvers": approvers,
                "channels": sorted({v.channel for v in req.votes}),
                "scripted": scripted,
            },
        )
        ns.pending_approval_id = None
        if req.purpose == ApprovalPurpose.NODE_START:
            self.transition(spec.id, NodeStatus.READY, "approved to start")
            return
        pending = self._load_pending(spec.id)
        if (
            pending is None
            or evidence_hash_of(req.staged, pending.finding_list(), pending.risk_assessment().level)
            != req.evidence_hash
        ):
            self._quarantine_staged_of(ns, "approved evidence is no longer reproducible")
            self.transition(spec.id, NodeStatus.READY, "evidence changed since approval: re-run")
            return
        self._create_waivers(req, spec, decision.id, approvers, last)
        self._publish(spec, ns, pending, req.staged, self._attempt_record(ns, req.attempt))

    def _create_waivers(
        self,
        req: ApprovalRequest,
        spec: NodeSpec,
        approval_decision: str,
        approvers: Sequence[str],
        comment: str,
    ) -> None:
        for finding in req.waives:
            if finding.severity > Severity.MEDIUM:
                continue  # HIGH / CRITICAL findings are never waivable
            wid = f"W-{len(self.state.waivers) + 1:04d}"
            decision = self.decisions.record(
                DecisionKind.WAIVER,
                f"waiver of {finding.rule_id} for {spec.id}",
                actor=Actor.human(approvers[0] if approvers else "unknown"),
                node_id=spec.id,
                rationale=_clip(comment or "accepted by the approving reviewers", 300),
                parents=(approval_decision,),
                data={
                    "rule_id": finding.rule_id,
                    "finding_key": finding.key(),
                    "approval_id": req.id,
                },
            )
            self.state.waivers.append(
                Waiver(
                    id=wid,
                    rule_id=finding.rule_id,
                    finding_key=finding.key(),
                    node_id=spec.id,
                    evidence_hash=req.evidence_hash,
                    justification=_clip(comment or "accepted by the approving reviewers", 300),
                    approved_by=tuple(approvers),
                    approval_id=req.id,
                    decision_id=decision.id,
                    created_at=self.now(),
                )
            )
            self.emit(
                EventType.POLICY_WAIVED,
                node_id=spec.id,
                rule_id=finding.rule_id,
                finding_key=finding.key(),
                approval_id=req.id,
            )

    def _on_rejected(self, req: ApprovalRequest, ns: NodeState) -> None:
        spec = self.graph.node(req.node_id)
        rejecters = self._approvers(req, approved=False)
        comment = next(
            (v.comment for v in reversed(req.votes) if not v.approved and v.comment), "rejected"
        )
        self.decisions.record(
            DecisionKind.REJECTION,
            f"{', '.join(rejecters) or 'a reviewer'} rejected {spec.id} ({req.purpose.value})",
            actor=Actor.human(rejecters[0] if rejecters else "unknown"),
            node_id=spec.id,
            rationale=_clip(comment, 300),
            inputs=req.evidence[:MAX_INPUT_REFS],
            data={"approval_id": req.id, "rejecters": rejecters},
        )
        ns.pending_approval_id = None
        rec = self._attempt_record(ns, req.attempt)
        if rec is not None:
            rec.outcome = AttemptOutcomeKind.REJECTED
            rec.failure_kind = FailureKind.APPROVAL_REJECTED
        message = f"rejected by reviewer: {_clip(comment, 200)}"
        if req.purpose == ApprovalPurpose.NODE_START:
            ns.last_failure_kind = FailureKind.APPROVAL_REJECTED
            self.transition(
                spec.id,
                NodeStatus.FAILED,
                message,
                kind=FailureKind.APPROVAL_REJECTED.value,
                message=message,
            )
            self._compensate_for_failure(spec.id)
            return
        self._quarantine_staged_of(ns, "rejected by a human reviewer")
        feedback = Finding(
            "HUM-REJECT",
            PolicyCategory.COMPLIANCE,
            Severity.MEDIUM,
            "a reviewer rejected the output",
            remediation=_clip(comment, MAX_FEEDBACK_REMEDIATION),
        )
        self._after_failure(
            spec, ns, FailureKind.APPROVAL_REJECTED, message, (feedback,), True, rejected=False
        )

    def _record_agent_decisions(
        self, spec: NodeSpec, pending: _Pending, refs: Sequence[ArtifactRef]
    ) -> None:
        inputs = [ArtifactRef.from_dict(r) for r in pending.inputs][:MAX_INPUT_REFS]
        derived: list[str] = []
        for ref in inputs:
            derived.extend(d.id for d in self.decisions.decisions_for(ref))
        for raw in pending.decisions:
            parents = tuple(dict.fromkeys([*raw.get("parents", ()), *derived]))[
                :MAX_PARENT_DECISIONS
            ]
            self.decisions.record(
                DecisionKind(raw["kind"]),
                raw["summary"],
                actor=Actor.agent(pending.agent_id),
                node_id=spec.id,
                rationale=raw.get("rationale", ""),
                inputs=inputs,
                outputs=tuple(refs),
                alternatives=tuple(raw.get("alternatives", ())),
                outcome=raw.get("outcome", ""),
                parents=parents,
                refs=tuple(raw.get("refs", ())),
                data=raw.get("data", {}),
            )

    def _publish(
        self,
        spec: NodeSpec,
        ns: NodeState,
        pending: _Pending,
        staged: Sequence[ArtifactRef],
        rec: AttemptRecord | None,
    ) -> None:
        """Publish protocol P1-P6 (engine-semantics 7); journaled so a crash can be rolled forward."""
        ids = [r.artifact_id for r in staged]
        self._write_pending(
            pending
        )  # journal: lets ``resume`` finish a publish that crashed half way
        self.emit(
            EventType.NODE_PUBLISHING,
            node_id=spec.id,
            attempt=pending.attempt_no,
            staged=ids,
            expansion_hash=hash_jsonable(pending.expansion) if pending.expansion else "",
        )
        published = self.store.activate(ids)
        new_refs = tuple(a.ref for a in published)
        previous = {(r.kind, r.key): r.content_hash for r in ns.outputs}
        current = {(r.kind, r.key): r.content_hash for r in new_refs}
        changed_kinds = {k for (k, key), h in current.items() if previous.get((k, key)) != h}
        changed_kinds |= {k for (k, key) in previous if (k, key) not in current}
        expansion_changed = False
        if pending.expansion is not None:
            expansion_changed = self._apply_expansion(spec, pending.expansion)
        self._record_agent_decisions(spec, pending, new_refs)
        ns.outputs = new_refs
        ns.staged = ()
        ns.pending_attempt = None
        ns.pending_approval_id = None
        ns.last_success_fingerprint = pending.fingerprint
        ns.fingerprint = pending.fingerprint
        ns.feedback = ()
        ns.last_failure_kind = None
        ns.open_question_ids = ()
        if pending.receipt:
            ns.receipt = pending.receipt
        if rec is not None:
            rec.outcome = AttemptOutcomeKind.PUBLISHED
        self.safe.record_attempt(self.state, pending.agent_id, None)
        self.transition(
            spec.id,
            NodeStatus.SUCCEEDED,
            "",
            attempt=pending.attempt_no,
            stage=spec.stage.value,
            outputs=[a.id for a in published],
            cache_hit=False,
        )
        self._clear_pending(spec.id)
        if changed_kinds or expansion_changed:
            transitions = invalidate(
                self.graph,
                self.state.nodes,
                changed_kinds,
                (spec.id,) if expansion_changed or changed_kinds else (),
            )
            self._apply_transitions(transitions, f"{spec.id} published new content")
        self.save()

    def _apply_expansion(self, spec: NodeSpec, raw: Mapping[str, Any]) -> bool:
        """P4: grow / shrink the graph from the owner's (already normalised and validated) expansion."""
        expansion = PlanExpansion.from_dict(raw)
        previous = self.graph.expansion_of(spec.id)
        diff = self.graph.apply_expansion(expansion)
        added, removed = list(diff.added), list(diff.removed)
        for node_id in added:
            if node_id in self.state.nodes:  # a tombstone being re-added
                tomb = self.ns(node_id)
                tomb.epoch_no += 1
                tomb.outputs = ()
                self.transition(node_id, NodeStatus.PENDING, "re-added by re-plan")
            else:
                self.state.nodes[node_id] = NodeState(node_id)
        for node_id in removed:
            self._remove_node(node_id)
        self.state.graph = self.graph.to_dict()
        self.state.graph_hash = self.graph.graph_hash()
        self.emit(
            EventType.PLAN_EXPANDED,
            node_id=spec.id,
            owner=spec.id,
            added=sorted(added),
            removed=sorted(removed),
            retained=sorted(diff.retained),
            changed=sorted(diff.changed),
            rewired=sorted(diff.rewired),
            expansion_hash=hash_jsonable(raw),
            graph_hash=self.state.graph_hash,
        )
        if previous is not None:
            self.state.counters.replans += 1
        if previous is not None and (added or removed or diff.changed):
            self._open_wave(
                f"re-expansion by {spec.id}",
                [*diff.changed, *added],
                added=added,
                removed=removed,
                retained=diff.retained,
            )
        return bool(added or removed or diff.changed or diff.rewired)

    def _remove_node(self, node_id: str) -> None:
        ns = self.ns(node_id)
        job = self._jobs.pop(node_id, None)
        if job is not None:
            job.cancel.cancel("node removed")
            self._zombies.append(job)
            ns.generation += 1
        self.approvals.cancel(node_id, "node removed")
        ns.pending_approval_id = None
        for art in self.store.all():
            if art.producer == node_id:
                self.store.supersede(art.id, "node removed")
        self._quarantine_staged_of(ns, "node removed")
        ns.outputs = ()
        if ns.status in (
            NodeStatus.RUNNING,
            NodeStatus.AWAITING_APPROVAL,
            NodeStatus.AWAITING_INPUT,
        ):
            self.transition(node_id, NodeStatus.PENDING, "node removed")
        elif ns.status == NodeStatus.CANCELLED:
            self.transition(node_id, NodeStatus.PENDING, "node removed")
        elif ns.status == NodeStatus.READY:
            self.transition(node_id, NodeStatus.PENDING, "node removed")
        if ns.status != NodeStatus.REMOVED:
            self.transition(node_id, NodeStatus.REMOVED, "removed by re-plan")

    def _apply_transitions(self, transitions: Sequence[Transition], trigger: str) -> None:
        if not transitions:
            return
        touched: list[str] = []
        cancelled: list[str] = []
        for t in transitions:
            if t.node_id not in self.state.nodes:
                continue
            ns = self.ns(t.node_id)
            if ns.status == t.to and not t.bump_epoch:
                continue
            if t.cancel_running and ns.status == NodeStatus.RUNNING:
                job = self._jobs.pop(t.node_id, None)
                if job is not None:
                    job.cancel.cancel("invalidated by an upstream change")
                    self._zombies.append(job)
                ns.generation += 1
                cancelled.append(t.node_id)
            if t.quarantine_staged:
                self._quarantine_staged_of(ns, "invalidated")
            if t.cancel_approval:
                self.approvals.cancel(t.node_id, "invalidated by an upstream change")
                ns.pending_approval_id = None
            if t.bump_epoch:
                self._reset_epoch(ns)
            if ns.status == t.to:
                continue
            self.transition(t.node_id, t.to, t.reason)
            if t.to in (NodeStatus.STALE, NodeStatus.PENDING):
                touched.append(t.node_id)
        if touched:
            self._open_wave(trigger, touched, cancelled=cancelled)

    def _apply_revalidation(self, c: _Completion, ns: NodeState | None) -> None:
        job, out = c.job, c.outcome
        spec = job.spec
        if (
            ns is None
            or out.generation != ns.generation
            or ns.status != NodeStatus.STALE
            or self._pending_stop is not None
        ):
            return
        comp = compose(out.gate_results)
        approvals_ok = True
        if self.approvals.for_node(spec.id):
            approvals_ok = self.approvals.valid_for(spec.id, job.fingerprint) is not None
        self._record_gate_events(spec, out)
        if comp.decision == GateDecision.PASS and approvals_ok and out.failure is None:
            ns.cache_hits += 1
            ns.fingerprint = job.fingerprint
            self.emit(EventType.NODE_CACHE_HIT, node_id=spec.id, fingerprint=job.fingerprint)
            self._cache_hit_node = spec.id
            try:
                self.transition(
                    spec.id,
                    NodeStatus.SUCCEEDED,
                    "unchanged inputs: outputs restored, verdicts re-evaluated",
                    stage=spec.stage.value,
                    outputs=[r.artifact_id for r in ns.outputs],
                    cache_hit=True,
                )
            finally:
                self._cache_hit_node = None
            return
        self._reset_epoch(ns)
        self.transition(spec.id, NodeStatus.READY, "revalidation failed: re-run")

    def _check_deadlines(self) -> None:
        mono = self.clock.monotonic()
        for node_id, job in list(self._jobs.items()):
            if job.started_mono is None or mono - job.started_mono <= job.deadline_s:
                continue
            ns = self.ns(node_id)
            spec = job.spec
            job.cancel.cancel("timeout")
            ns.generation += 1
            self._jobs.pop(node_id, None)
            self._zombies.append(job)
            message = f"attempt exceeded its {job.deadline_s:.0f}s deadline"
            if job.kind == "revalidate":
                self._reset_epoch(ns)
                if ns.status == NodeStatus.STALE:
                    self.transition(node_id, NodeStatus.READY, "revalidation timed out: re-run")
                continue
            rec = self._attempt_record(ns, job.attempt_no)
            if rec is not None:
                rec.outcome = AttemptOutcomeKind.FAILED
                rec.failure_kind = FailureKind.TIMEOUT
                rec.message = message
                rec.ended_at = self.now()
            self._quarantine_attempt_artifacts(node_id, job.attempt_no, "timeout")
            self._emit_attempt_failed(
                spec, job.attempt_no, FailureKind.TIMEOUT, message, job.epoch, True, ()
            )
            self.safe.record_attempt(self.state, job.agent_id, FailureKind.TIMEOUT)
            if ns.status == NodeStatus.RUNNING:
                self._after_failure(
                    spec, ns, FailureKind.TIMEOUT, message, (), True, rejected=False
                )

    # ------------------------------------------------------------------ clarification (AWAITING_INPUT)
    def _hold_for_input(
        self,
        spec: NodeSpec,
        ns: NodeState,
        questions: Sequence[ClarificationQuestion],
        *,
        pending: _Pending | None,
        staged: Sequence[ArtifactRef],
        attempt_no: int,
    ) -> None:
        blocking = [q for q in questions if q.blocking][:MAX_QUESTIONS] or list(questions)[
            :MAX_QUESTIONS
        ]
        ids = tuple(dict.fromkeys(q.id for q in blocking))
        if ns.clarification_round >= self.cfg.max_clarification_rounds:
            self._quarantine_attempt_artifacts(spec.id, attempt_no, "clarification exhausted")
            ns.last_failure_kind = FailureKind.GATE_REJECTED
            self.transition(
                spec.id,
                NodeStatus.FAILED,
                "clarification_exhausted",
                kind=FailureKind.GATE_REJECTED.value,
                message="clarification rounds exhausted: the requirement stays ambiguous (never guess)",
            )
            self._compensate_for_failure(spec.id)
            return
        rec = self._attempt_record(ns, attempt_no)
        if rec is not None:
            rec.outcome = AttemptOutcomeKind.AWAITING_INPUT
            rec.counts_vs_budget = False
        if pending is not None:
            # a clarification pause is not a failure: refund the attempt budget
            if ns.epoch == Epoch.PRIMARY:
                ns.primary_attempts = max(0, ns.primary_attempts - 1)
            else:
                ns.fallback_attempts = max(0, ns.fallback_attempts - 1)
            ns.lifetime_attempts = max(0, ns.lifetime_attempts - 1)
            self._write_pending(pending)
        ns.staged = tuple(staged)
        ns.pending_attempt = attempt_no if pending is not None else None
        ns.open_question_ids = ids
        ns.clarification_round += 1
        self.state.clarifications.append(
            ClarificationRound(ns.clarification_round, spec.id, tuple(blocking), self.now())
        )
        self.emit(
            EventType.CLARIFICATION_REQUESTED,
            node_id=spec.id,
            questions=list(ids),
            round=ns.clarification_round,
        )
        self.transition(
            spec.id, NodeStatus.AWAITING_INPUT, "clarification required", questions=list(ids)
        )

    def _clean_input(self, value: object, limit_bytes: int) -> str:
        if not isinstance(value, str):
            raise StateError("input must be text")
        if "\x00" in value:
            raise StateError("input must not contain NUL characters")
        cleaned = strip_controls(value, keep_newlines=True)
        if len(cleaned.encode("utf-8", "replace")) > limit_bytes:
            raise StateError(f"input exceeds {limit_bytes} bytes")
        return cleaned

    def _ingest(self, draft: ArtifactDraft, *, by: str) -> Artifact:
        """Store a human-supplied artifact: stage -> screen as untrusted input -> publish."""
        art = self.store.put(
            draft,
            producer=EXTERNAL_PRODUCER,
            attempt=0,
            agent_id=f"human:{by}",
            inputs=(),
            staged=True,
        )
        report = self.orch.policy.evaluate(
            ArtifactSubject(art, text=self.store.read_text(art), untrusted_input=True)
        )
        rule_ids = sorted({f.rule_id for f in report.findings})
        if report.enforcement in (Enforcement.DENY, Enforcement.SAFE_STOP):
            self.store.quarantine([art.id], "rejected by input screening", tombstone=True)
            self.emit(
                EventType.INPUT_SCREENED,
                kind=draft.kind,
                rule_ids=rule_ids,
                outcome="rejected",
                by=by,
            )
            raise StateError(f"input rejected by screening ({', '.join(rule_ids) or 'policy'})")
        if report.findings:
            self.emit(
                EventType.INPUT_SCREENED,
                kind=draft.kind,
                rule_ids=rule_ids,
                outcome="flagged",
                by=by,
            )
        return self.store.activate([art.id])[0]

    def cmd_answer(
        self, *, answers: Mapping[str, str], by: str, accept_default: Sequence[str] = ()
    ) -> dict[str, Any]:
        name = normalize_approver(by, reserved=self.reserved)
        if self.state.counters.human_inputs >= self.cfg.max_human_inputs:
            raise StateError(
                f"too many human inputs for one run (limit {self.cfg.max_human_inputs})"
            )
        open_q = {q.id: q for q in self.open_questions()}
        if not open_q:
            raise StateError("no clarification question is open")
        resolved: dict[str, str] = {}
        assumed: set[str] = set()
        for qid, text in answers.items():
            question = open_q.get(qid)
            if question is None:
                raise StateError(f"unknown or already answered question {_clip(qid, 40)!r}")
            value = self._clean_input(text, MAX_ANSWER_BYTES)
            if question.options and value not in question.options:
                raise StateError(f"answer to {qid} must be one of the offered options")
            resolved[qid] = value
        for qid in accept_default:
            question = open_q.get(qid)
            if question is None or not question.default_assumption:
                raise StateError(f"question {_clip(qid, 40)!r} has no default assumption to accept")
            if qid not in resolved:
                resolved[qid] = question.default_assumption
                assumed.add(qid)
        if not resolved:
            raise StateError("no answers were given")
        previous = self.store.latest(K_CLARIFICATION_ANSWERS)
        merged: dict[str, Any] = {}
        if previous is not None:
            prior = self.store.read_json(previous)
            merged = dict(prior.get("answers", {})) if isinstance(prior, Mapping) else {}
        round_no = max((r.round for r in self.state.clarifications), default=1)
        for qid, value in resolved.items():
            merged[qid] = {"answer": value, "by": name, "round": round_no}
        art = self._ingest(
            ArtifactDraft(
                kind=K_CLARIFICATION_ANSWERS, content={"schema": SCHEMA_VERSION, "answers": merged}
            ),
            by=name,
        )
        for qid, value in resolved.items():
            owner = next(
                (
                    r.node_id
                    for r in reversed(self.state.clarifications)
                    if qid in {q.id for q in r.questions}
                ),
                None,
            )
            self.decisions.record(
                DecisionKind.ASSUMPTION if qid in assumed else DecisionKind.CLARIFICATION,
                f"{name} {'accepted the default for' if qid in assumed else 'answered'} {qid}",
                actor=Actor.human(name),
                node_id=owner,
                rationale=_clip(open_q[qid].text, 200),
                outputs=(art.ref,),
                refs=(qid,) if _safe_ref(qid) else (),
                data={"question_id": qid, "answer": _clip(value, 300)},
            )
        self.emit(EventType.CLARIFICATION_ANSWERED, questions=sorted(resolved), by=name)
        self.emit(
            EventType.INPUT_CHANGED,
            kind=K_CLARIFICATION_ANSWERS,
            key="",
            old_ref=previous.id if previous is not None else "",
            new_ref=art.id,
            by=name,
        )
        self.state.counters.human_inputs += 1
        for rnd in self.state.clarifications:
            if rnd.answered_at is None and all(q.id in merged for q in rnd.questions if q.blocking):
                rnd.answered_at = self.now()
        transitions = invalidate(
            self.graph,
            self.state.nodes,
            {K_CLARIFICATION_ANSWERS},
            (),
            wake_kind=K_CLARIFICATION_ANSWERS,
        )
        kept: list[Transition] = []
        for t in transitions:
            ns = self.ns(t.node_id)
            waking = ns.status == NodeStatus.AWAITING_INPUT and t.to == NodeStatus.READY
            if waking and any(q not in merged for q in ns.open_question_ids):
                continue  # partial answers keep the node waiting
            if waking:
                self._quarantine_staged_of(ns, "superseded by the clarified re-run")
                ns.generation += 1
            kept.append(t)
        self._apply_transitions(kept, "clarification answers")
        return {"artifact": art.id, "answered": sorted(resolved)}

    def cmd_amend(self, *, new_text: str, by: str, reason: str) -> dict[str, Any]:
        name = normalize_approver(by, reserved=self.reserved)
        if self.state.counters.human_inputs >= self.cfg.max_human_inputs:
            raise StateError(
                f"too many human inputs for one run (limit {self.cfg.max_human_inputs})"
            )
        text = self._clean_input(new_text, self.cfg.max_input_bytes)
        why = self._clean_input(reason or "amendment", 600)
        previous = self.store.latest(K_REQUEST)
        old_text = self.store.read_text(previous) if previous is not None else ""
        if text.strip() == old_text.strip():
            raise StateError("the amendment does not change the request")
        art = self._ingest(
            ArtifactDraft(kind=K_REQUEST, content=text, media_type="text/markdown"), by=name
        )
        diff = "".join(
            difflib.unified_diff(
                old_text.splitlines(True), text.splitlines(True), "request@old", "request@new", n=2
            )
        )
        decision = self.decisions.record(
            DecisionKind.SCOPE,
            f"{name} amended the request: {_clip(why, 160)}",
            actor=Actor.human(name),
            rationale=_clip(why, 300),
            inputs=(previous.ref,) if previous is not None else (),
            outputs=(art.ref,),
            data={"diff": _clip(diff, 4000), "reason": _clip(why, 300)},
        )
        self.emit(
            EventType.INPUT_CHANGED,
            kind=K_REQUEST,
            key="",
            old_ref=previous.id if previous is not None else "",
            new_ref=art.id,
            by=name,
        )
        self.state.counters.human_inputs += 1
        self._apply_transitions(
            invalidate(self.graph, self.state.nodes, {K_REQUEST}, ()), "request amended"
        )
        return {"artifact": art.id, "decision": decision.id}

    # ------------------------------------------------------------------ rollback / compensation
    def compensate(self, node_id: str, *, reason: str) -> bool:
        """Undo the real side effect of a SUCCEEDED node from its persisted receipt (node-level rollback)."""
        spec = self.graph.node(node_id)
        ns = self.ns(node_id)
        if (
            ns.status != NodeStatus.SUCCEEDED
            or not spec.compensator
            or not ns.receipt
            or not self.state.apply_root
        ):
            return False
        self.emit(
            EventType.ROLLBACK_STARTED,
            node_id=node_id,
            level="node",
            compensator=spec.compensator,
            reason=_clip(reason, 200),
        )
        try:
            compensator = self.registries.compensators.get(spec.compensator)
            factory = self.orch.effect_factory or _default_effect_factory(self.orch.policy)
            effects = factory(
                Path(self.state.apply_root),
                self.audit.emit,
                node_id,
                ns.attempt_counter,
                CancelToken(),
            )
            note = compensator.compensate(
                ApplyReceipt.from_dict(ns.receipt),
                effects=effects,
                run_view=self.snapshot().governance(),
            )
        except Exception as exc:  # noqa: BLE001 - a failed compensation escalates to safe-stop
            self.emit(
                EventType.ROLLBACK_FAILED,
                node_id=node_id,
                level="node",
                message=self.redactor.redact_text(_clip(f"{type(exc).__name__}: {exc}", 300)),
            )
            if self.state.safe_stop is not None:
                self.state.safe_stop.compensation_incomplete.append(node_id)
            self.request_safe_stop("compensation_failed", f"could not compensate {node_id}")
            return False
        self.state.counters.compensations += 1
        self.decisions.record(
            DecisionKind.ROLLBACK,
            f"compensated {node_id}: {_clip(note, 160)}",
            actor=_SYSTEM,
            node_id=node_id,
            rationale=_clip(reason, 300),
            data={"compensator": spec.compensator},
        )
        ns.receipt = None
        self.transition(node_id, NodeStatus.ROLLED_BACK, reason)
        self.emit(EventType.ROLLBACK_COMPLETED, node_id=node_id, level="node")
        self._dirty = True
        return True

    # ------------------------------------------------------------------ safe-stop (journaled S0-S6)
    def _ss_done(self, rec: SafeStopRecord, step: int) -> None:
        rec.step = step + 1
        self.emit(EventType.SAFE_STOP_STEP, step=f"S{step}")
        self.save()

    def _safe_stop(self, trigger: str, reason: str) -> RunResult:
        previous = self.state.safe_stop
        if previous is None or previous.step >= 7:
            rec = SafeStopRecord(
                trigger,
                _clip(reason, 300),
                kill_acked=list(previous.kill_acked) if previous else [],
            )
            self.state.safe_stop = rec
        else:
            rec = previous  # resuming an interrupted procedure
            trigger, reason = rec.trigger, rec.reason
        self._stopping = True
        if rec.step <= 0:  # S0: record the decision FIRST
            self.state.status = RunStatus.SAFE_STOPPING
            self.state.status_reason = _clip(reason, 300)
            decision = self.decisions.record(
                DecisionKind.SAFE_STOP,
                f"safe-stop ({trigger})",
                actor=_SYSTEM,
                rationale=_clip(reason, 300),
                outcome="halting the run",
                data={"trigger": trigger},
            )
            rec.decision_id = decision.id
            self.emit(EventType.SAFE_STOP_TRIGGERED, trigger=trigger, reason=_clip(reason, 300))
            self._ss_done(rec, 0)
        if rec.step <= 1:  # S1: cancel everything in flight
            for node_id, job in list(self._jobs.items()):
                job.cancel.cancel("safe-stop")
                self._zombies.append(job)
                self.ns(node_id).generation += 1
            self._jobs.clear()
            for req in list(self.state.approvals.values()):
                if req.status == ApprovalStatus.PENDING:
                    self.approvals.cancel(req.node_id, "run safe-stopped")
            for ns in self.state.nodes.values():
                ns.open_question_ids = ()
                ns.pending_approval_id = None
            self._ss_done(rec, 1)
        if rec.step <= 2:  # S2: quiesce (threads cannot be killed; survivors are zombies)
            deadline = self.clock.monotonic() + self.cfg.safe_stop_quiesce_s
            for job in list(self._zombies):
                if job.thread is None:
                    continue
                remaining = max(0.0, deadline - self.clock.monotonic())
                job.thread.join(timeout=min(remaining, 0.5))
            self._reap_late_completions()
            self._ss_done(rec, 2)
        if rec.step <= 3:  # S3: nothing staged survives
            ids = [a.id for a in self.store.staged()]
            if ids:
                self.store.quarantine(ids, "run safe-stopped")
            for ns in self.state.nodes.values():
                ns.staged = ()
                ns.pending_attempt = None
                self._clear_pending(ns.node_id)
            self._ss_done(rec, 3)
        if rec.step <= 4:  # S4: compensate completed side effects (reverse topological order)
            if self.cfg.rollback_on_safe_stop:
                for node_id in reversed(self.graph.topological_order()):
                    ns = self.ns(node_id)
                    if ns.status == NodeStatus.SUCCEEDED and ns.receipt:
                        self._pending_stop = None
                        self.compensate(node_id, reason=f"run safe-stopped ({trigger})")
            self._ss_done(rec, 4)
        if rec.step <= 5:  # S5: whatever is unfinished is cancelled
            for node_id in self.graph.topological_order():
                ns = self.ns(node_id)
                if ns.status in (
                    NodeStatus.PENDING,
                    NodeStatus.READY,
                    NodeStatus.RUNNING,
                    NodeStatus.AWAITING_APPROVAL,
                    NodeStatus.AWAITING_INPUT,
                    NodeStatus.STALE,
                    NodeStatus.BLOCKED,
                ):
                    self.transition(node_id, NodeStatus.CANCELLED, f"run safe-stopped ({trigger})")
            self._ss_done(rec, 5)
        rec.step = 7
        return self._finish(RunStatus.SAFE_STOPPED, f"safe-stop ({trigger}): {reason}")

    def _reap_late_completions(self) -> None:
        while True:
            try:
                c = self._completions.get_nowait()
            except queue.Empty:
                return
            self.state.counters.discarded_attempts += 1
            self._quarantine_attempt_artifacts(
                c.job.spec.id, c.outcome.attempt_no, "run safe-stopped"
            )

    # ------------------------------------------------------------------ finishing
    def _publish_engine_artifact(self, draft: ArtifactDraft) -> Artifact:
        return self.store.put(
            draft, producer="engine", attempt=0, agent_id="engine", inputs=(), staged=False
        )

    def _record_system_decision(self, kind: DecisionKind, summary: str, **kwargs: Any) -> Decision:
        return self.decisions.record(kind, summary, actor=_SYSTEM, **kwargs)

    def _close_human_wait(self) -> None:
        if self.state.human_wait and self.state.human_wait[-1].end is None:
            self.state.human_wait[-1].end = self.now()

    def _result(self, status: RunStatus, reason: str) -> RunResult:
        pending = tuple(
            r.id for r in self.state.approvals.values() if r.status == ApprovalStatus.PENDING
        )
        questions = tuple(q.id for q in self.open_questions())
        return RunResult(
            self.run_id, status, _clip(reason, 300), pending, questions, self.audit.head()
        )

    def _finish(self, status: RunStatus, reason: str) -> RunResult:
        self._flush_waves()
        self._close_human_wait()
        self.state.status = status
        self.state.status_reason = _clip(reason, 300)
        self.state.finished_at = self.now()
        etype = {
            RunStatus.SUCCEEDED: EventType.RUN_SUCCEEDED,
            RunStatus.FAILED: EventType.RUN_FAILED,
            RunStatus.SAFE_STOPPED: EventType.RUN_SAFE_STOPPED,
        }[status]
        self.emit(etype, reason=_clip(reason, 300))
        self.save()
        ctx = FinalizerContext(
            run_id=self.run_id,
            status=status,
            reason=_clip(reason, 300),
            view=self.snapshot(),
            run_dir=self.run_dir,
            audit_head=self.audit.head(),
            publish_artifact=self._publish_engine_artifact,
            record_decision=self._record_system_decision,
        )
        for finalizer in self.orch.finalizers:
            try:
                finalizer(ctx)
            except Exception as exc:  # noqa: BLE001 - a finalizer must never lose the run's terminal state
                self.emit(
                    EventType.AGENT_NOTE,
                    message=f"finalizer failed: {type(exc).__name__}: {_clip(exc, 160)}",
                )
        problems = check_invariants(
            self.state, graph=self.graph, store=self.store, events=self.audit.events()
        )
        if problems:
            self.emit(
                EventType.AGENT_NOTE,
                message="invariant violations: " + _clip("; ".join(problems[:5]), 400),
            )
        events = self.audit.events()
        metrics = compute_run_metrics(events, graph=self.graph)
        files = write_run_outputs(
            self.run_dir,
            state=self.state,
            events=events,
            decisions=self.decisions.all(),
            metrics=metrics,
            artifacts=self.store.all(include_inactive=True),
            posture=self.cfg.posture(),
        )
        self.emit(
            EventType.RUN_SEALED, files=sorted(files), head_seq=events[-1].seq if events else 0
        )
        return self._result(status, reason)

    def _wait_for_humans(self) -> RunResult:
        self._flush_waves()
        self.state.status = RunStatus.WAITING_HUMAN
        pending = [
            r.id for r in self.state.approvals.values() if r.status == ApprovalStatus.PENDING
        ]
        questions = [q.id for q in self.open_questions()]
        reason = f"waiting for {len(pending)} approval(s) and {len(questions)} answer(s)"
        self.state.status_reason = reason
        if not self.state.human_wait or self.state.human_wait[-1].end is not None:
            self.state.human_wait.append(Interval(self.now()))
        self.emit(EventType.RUN_WAITING, reason=reason, approvals=pending, questions=questions)
        self.save()
        return self._result(RunStatus.WAITING_HUMAN, reason)

    def _compute_status(self) -> tuple[RunStatus, str]:
        statuses = {n: self.ns(n).status for n in self.graph.nodes}
        if self._jobs:
            return RunStatus.RUNNING, ""
        failing = (
            NodeStatus.FAILED,
            NodeStatus.BLOCKED,
            NodeStatus.ROLLED_BACK,
            NodeStatus.CANCELLED,
        )
        for node_id, status in statuses.items():
            if status in (NodeStatus.RUNNING, NodeStatus.READY):
                return RunStatus.RUNNING, ""
            if status in (NodeStatus.PENDING, NodeStatus.STALE):
                _, dep_status = self._dep_statuses(node_id)
                if all(d.satisfies_dependency for d in dep_status) or any(
                    d in failing for d in dep_status
                ):
                    return RunStatus.RUNNING, ""
        if any(s.is_awaiting_human for s in statuses.values()):
            return RunStatus.WAITING_HUMAN, ""
        bad = sorted(
            n for n, s in statuses.items() if s not in (NodeStatus.SUCCEEDED, NodeStatus.SKIPPED)
        )
        mandatory_bad = sorted(
            n
            for n in self.state.mandatory_snapshot
            if n in statuses and statuses[n] != NodeStatus.SUCCEEDED
        )
        if not bad and not mandatory_bad:
            return RunStatus.SUCCEEDED, "all nodes completed"
        parts = [f"{n} is {statuses[n].value}" for n in (mandatory_bad or bad)[:6]]
        return RunStatus.FAILED, "; ".join(parts)

    def _idle_wait(self) -> None:
        now = self.now()
        if not self._jobs and self._completions.empty():
            waits = [
                ns.not_before
                for ns in self.state.nodes.values()
                if ns.status == NodeStatus.READY
                and ns.not_before is not None
                and ns.not_before > now
            ]
            if waits:
                remaining = (min(waits) - now).total_seconds()
                if remaining > 0:
                    self.clock.sleep(remaining)
                    return
        try:
            self._backlog.append(self._completions.get(timeout=self.cfg.poll_interval_s))
        except queue.Empty:
            pass

    # ------------------------------------------------------------------ the loop
    def loop(self) -> RunResult:
        self._live = True
        self.orch._live_engines[self.run_id] = self
        self._last_mono = self.clock.monotonic()
        try:
            try:
                return self._loop_body()
            except (IntegrityError, StateError):
                raise
            except Exception as exc:  # noqa: BLE001 - an engine fault must still stop the run safely
                self.emit(
                    EventType.AGENT_NOTE,
                    message=f"engine fault: {type(exc).__name__}: {_clip(exc, 200)}",
                )
                return self._safe_stop("engine_fault", f"{type(exc).__name__}: {_clip(exc, 200)}")
        finally:
            self._live = False
            self.orch._live_engines.pop(self.run_id, None)
            for job in self._jobs.values():
                job.cancel.cancel("engine stopped")

    def _loop_body(self) -> RunResult:
        if self.state.safe_stop is not None and self.state.status == RunStatus.SAFE_STOPPING:
            return self._safe_stop(self.state.safe_stop.trigger, self.state.safe_stop.reason)
        while True:
            self._housekeeping()
            stop = self._check_safety()
            if stop is not None:
                return self._safe_stop(*stop)
            self._drain_commands()
            self._drain_completions()
            self._poll_providers()
            self._process_approvals()
            self._check_deadlines()
            if self._pending_stop is not None:
                return self._safe_stop(*self._pending_stop)
            self._promote()
            self._dispatch()
            status, reason = self._compute_status()
            if self._dirty:
                self.save()
            if status in (RunStatus.SUCCEEDED, RunStatus.FAILED):
                return self._finish(status, reason)
            if status == RunStatus.WAITING_HUMAN and not self._jobs:
                return self._wait_for_humans()
            self._idle_wait()

    # ------------------------------------------------------------------ resume support
    def reconcile(self) -> None:
        """Make persisted state consistent with the audit chain after a pause or a crash (section 7/17)."""
        events = self.audit.events()
        head = self.state.audit_head
        seq = int(head.get("seq", 0))
        if seq and not any(e.seq == seq and e.hash == head.get("hash") for e in events):
            raise IntegrityError(
                "the audit log does not contain the head recorded in the run state (truncated or replaced)"
            )
        problems = self.store.verify_against_events(events)
        if problems:
            raise IntegrityError(
                "artifact index disagrees with the audit chain: " + "; ".join(problems[:3])
            )
        last_publishing: dict[str, AuditEvent] = {}
        succeeded_seq: dict[str, int] = {}
        for event in events:
            if event.node_id is None:
                continue
            if event.type == EventType.NODE_PUBLISHING:
                last_publishing[event.node_id] = event
            elif event.type == EventType.NODE_SUCCEEDED:
                succeeded_seq[event.node_id] = event.seq
        for node_id, event in last_publishing.items():
            ns = self.state.nodes.get(node_id)
            if ns is None or succeeded_seq.get(node_id, 0) > event.seq:
                continue
            if ns.status not in (NodeStatus.RUNNING, NodeStatus.AWAITING_APPROVAL):
                continue
            ids = [str(i) for i in event.data.get("staged", ())]
            pending = self._load_pending(node_id)
            try:
                arts = [self.store.get(i) for i in ids]
            except (KeyError, OrchestratorError):
                continue
            if (
                pending is not None
                and arts
                and all(a.status == ArtifactStatus.ACTIVE for a in arts)
            ):
                self._publish(
                    self.graph.node(node_id),
                    ns,
                    pending,
                    [a.ref for a in arts],
                    self._attempt_record(ns, pending.attempt_no),
                )
        for node_id, ns in self.state.nodes.items():
            if ns.status == NodeStatus.RUNNING:
                self._quarantine_attempt_artifacts(node_id, ns.attempt_counter, "crash recovery")
                ns.generation += 1
                rec = self._attempt_record(ns, ns.attempt_counter)
                if rec is not None:
                    rec.outcome = AttemptOutcomeKind.FAILED
                    rec.message = "interrupted by a crash"
                self.transition(node_id, NodeStatus.READY, "crash recovered: attempt re-queued")
            elif ns.status == NodeStatus.AWAITING_APPROVAL and ns.pending_approval_id:
                req = self.state.approvals.get(ns.pending_approval_id)
                if (
                    req is not None
                    and req.purpose == ApprovalPurpose.NODE_OUTPUT
                    and self._load_pending(node_id) is None
                ):
                    self.approvals.cancel(node_id, "pending attempt record lost")
                    self._quarantine_staged_of(ns, "pending attempt record lost")
                    ns.pending_approval_id = None
                    self.transition(node_id, NodeStatus.READY, "pending record lost: re-run")
            elif (
                ns.status == NodeStatus.AWAITING_INPUT
                and ns.pending_attempt is not None
                and self._load_pending(node_id) is None
            ):
                self._quarantine_staged_of(ns, "pending attempt record lost")
                self.transition(node_id, NodeStatus.READY, "pending record lost: re-run")
        referenced = {r.artifact_id for ns in self.state.nodes.values() for r in ns.staged}
        orphans = [a.id for a in self.store.staged() if a.id not in referenced]
        if orphans:
            self.store.quarantine(orphans, "orphaned staged artifact after resume")
        self.save()

    # ------------------------------------------------------------------ resume / override
    def begin_resume(self, override: SafeStopOverride | None) -> None:
        st = self.state
        st.resume_count += 1
        self._close_human_wait()
        if st.status == RunStatus.SAFE_STOPPED:
            self._apply_override(override)
        self.reconcile()
        st.status = RunStatus.RUNNING
        st.status_reason = ""
        st.finished_at = None
        self.emit(EventType.RUN_RESUMED, resume_count=st.resume_count)
        self._dirty = True

    def _apply_override(self, override: SafeStopOverride | None) -> None:
        rec = self.state.safe_stop
        if override is None or rec is None:
            raise StateError(
                "a safe-stopped run can only be resumed with an attributed override (by, reason)"
            )
        if self.state.counters.safe_stop_overrides >= self.cfg.max_overrides:
            raise StateError(f"override limit reached ({self.cfg.max_overrides})")
        by = normalize_approver(override.by, reserved=self.reserved)
        co: str | None = None
        if rec.trigger in (
            "integrity",
            "autonomy_violation",
            "compensation_failed",
            "critical_finding",
        ):
            if not override.co_approver:
                raise StateError(
                    f"overriding a {rec.trigger} safe-stop needs a second approver (co_approver)"
                )
            co = normalize_approver(override.co_approver, reserved=self.reserved)
            if co == by:
                raise StateError("the co-approver must be a different person")
        reason = self._clean_input(override.reason, 600)
        self.decisions.record(
            DecisionKind.SAFE_STOP,
            f"{by} overrode the safe-stop ({rec.trigger})",
            actor=Actor.human(by),
            rationale=_clip(reason, 300),
            parents=(rec.decision_id,) if rec.decision_id else (),
            data={
                "trigger": rec.trigger,
                "co_approver": co or "",
                "extend": to_jsonable(dict(override.extend)),
            },
        )
        self.emit(EventType.SAFE_STOP_OVERRIDDEN, by=by, trigger=rec.trigger, co_approver=co or "")
        for signal in self.kill_switch.signals():
            rec.kill_acked.append(_kill_signal_id(signal))
        counters = self.state.counters
        if rec.trigger == "budget":
            counters.budgeted_attempts = 0
            counters.replans = 0
            counters.human_inputs = 0
            counters.remediation_loops = 0
            self.state.active_runtime_s = 0.0
        if override.reset_breaker:
            counters.breaker_failures = {}
            counters.breaker_open = []
        for node_id, ns in self.state.nodes.items():
            if ns.status in (NodeStatus.CANCELLED, NodeStatus.ROLLED_BACK, NodeStatus.STALE):
                self._reset_epoch(ns)
                ns.generation += 1
                self.transition(node_id, NodeStatus.PENDING, "re-opened by safe-stop override")
        counters.safe_stop_overrides += 1
        rec.step = 7
        self._stopping = False
        self._pending_stop = None


# =============================================================================
# module-level helpers
# =============================================================================

_POLICY_PREFIXES = frozenset({"SEC", "CMP", "CHG", "POL"})
_AGENT_DECISION_KINDS = frozenset(
    {DecisionKind.DESIGN, DecisionKind.ASSUMPTION, DecisionKind.SCOPE, DecisionKind.RELEASE}
)
DEFAULT_ENFORCEMENT_NAMES: Mapping[Severity, str] = {
    sev: enf.value for sev, enf in DEFAULT_ENFORCEMENT.items()
}
_FALLBACK_PROTECTED = (
    ".git/**",
    ".github/**",
    "Dockerfile",
    "pyproject.toml",
    ".env",
    ".env.*",
    "**/*.pem",
    "**/*.key",
)


def _os_user() -> str:
    try:
        return _clip(getpass.getuser(), 64)
    except Exception:  # noqa: BLE001 - os user is informational only
        return ""


def _summarise(findings: Sequence[Finding]) -> str:
    head = (
        "; ".join(f"{f.rule_id} {_clip(f.message, 80)}" for f in list(findings)[:3]) or "no details"
    )
    return head + (f" (+{len(findings) - 3} more)" if len(findings) > 3 else "")


def _draft_to_dict(draft: DecisionDraft) -> dict[str, Any]:
    return {
        "kind": draft.kind.value,
        "summary": _clip(draft.summary, 300),
        "rationale": _clip(draft.rationale, 600),
        "alternatives": [_clip(a, 200) for a in draft.alternatives[:10]],
        "outcome": _clip(draft.outcome, 300),
        "parents": list(draft.parents[:MAX_PARENT_DECISIONS]),
        "refs": list(draft.refs[:20]),
        "data": to_jsonable(draft.data),
    }


class _RecordingEffects:
    """Wraps an ``EffectPort`` so the engine keeps the receipt of every real apply (for compensation)."""

    def __init__(self, inner: EffectPort, sink: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._sink = sink

    def apply(self, cs: ChangeSet, *, dry_run: bool = False) -> ApplyReceipt:
        receipt = self._inner.apply(cs, dry_run=dry_run)
        if not dry_run:
            self._sink.append(receipt.to_dict())
        return receipt

    def revert(self, receipt: ApplyReceipt) -> None:
        self._inner.revert(receipt)


def _default_effect_factory(policy: PolicyEvaluator) -> EffectFactory:
    config = getattr(policy, "config", None)
    protected = tuple(getattr(config, "protected_paths", ())) or _FALLBACK_PROTECTED

    def factory(
        root: Path, audit: Any, node_id: str | None, attempt: int | None, cancel: CancelToken | None
    ) -> EffectPort:
        return DirectoryEffectPort(
            root,
            protected_globs=protected,
            audit=audit,
            node_id=node_id,
            attempt=attempt,
            cancel=cancel,
        )

    return factory


# =============================================================================
# public facade
# =============================================================================


@dataclass(frozen=True)
class RunSummary:
    """Read-only status of a run (what ``agentic status`` prints)."""

    run_id: str
    scenario: str
    status: RunStatus
    reason: str
    assurance: str
    posture: str
    nodes: Mapping[str, str]
    pending_approvals: tuple[Mapping[str, Any], ...]
    open_questions: tuple[Mapping[str, Any], ...]
    audit_head: tuple[int, str]
    counters: Mapping[str, Any]


class Orchestrator:
    """Public facade of the engine (see the module docstring and ``docs/engine-semantics.md``)."""

    def __init__(
        self,
        runs_dir: Path | str,
        registries: Registries,
        *,
        policy: PolicyEvaluator,
        config: EngineConfig | None = None,
        clock: Clock | None = None,
        approver: ApprovalProvider | None = None,
        toolbox_factory: ToolboxFactory | None = None,
        effect_factory: EffectFactory | None = None,
        finalizers: Sequence[Finalizer] = (),
        listeners: Sequence[Callable[[AuditEvent], None]] = (),
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.registries = registries
        self.policy = policy
        self.config = config or EngineConfig()
        self.clock: Clock = clock or SystemClock()
        self.approver = approver
        self.toolbox_factory = toolbox_factory
        self.effect_factory = effect_factory
        self.finalizers = tuple(finalizers)
        self.listeners = tuple(listeners)
        self._live_engines: dict[str, _Engine] = {}
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._store = StateStore(self.runs_dir)
        for gate_type in ("artifacts_present", "policy"):
            if not registries.gates.has(gate_type):
                raise ConfigError(f"the gate registry must provide the implicit gate {gate_type!r}")

    # ------------------------------------------------------------------ lifecycle
    def _reserved(self) -> frozenset[str]:
        return frozenset(
            {n.casefold() for n in (*self.registries.agents.ids(), *RESERVED_APPROVER_NAMES)}
        )

    def _new_run_id(self, scenario: str) -> str:
        safe = "".join(c for c in scenario if c.isalnum() or c in "-_")[:24] or "run"
        stamp = self.clock.now().strftime("%Y%m%dT%H%M%SZ")
        return f"{safe}-{stamp}-{uuid.uuid4().hex[:4]}"

    def start(
        self,
        graph: WorkflowGraph,
        *,
        scenario: str,
        request_text: str,
        baseline_source: Path | str | None = None,
        include: Sequence[str] = ("**",),
        exclude: Sequence[str] = (),
        params: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        created_by: str = "operator",
        apply_root: Path | str | None = None,
    ) -> str:
        graph.validate()
        rid = run_id or self._new_run_id(scenario)
        if not RUN_ID_RE.match(rid):
            raise ConfigError(f"invalid run id: {_clip(rid, 60)!r}")
        creator = normalize_approver(created_by, reserved=self._reserved())
        run_dir = self._store.create(rid)
        baseline = (
            BaselineSnapshot.create(
                baseline_source,
                run_dir / "baseline",
                include=tuple(include),
                exclude=tuple(exclude),
            )
            if baseline_source is not None
            else BaselineSnapshot.empty(run_dir / "baseline")
        )
        state = RunState(
            run_id=rid,
            scenario=scenario,
            created_by=creator,
            created_at=self.clock.now(),
            trace_id=uuid.uuid4().hex,
        )
        state.nodes = {node_id: NodeState(node_id) for node_id in graph.nodes}
        state.graph = graph.to_dict()
        state.graph_hash = graph.graph_hash()
        state.mandatory_snapshot = graph.mandatory_snapshot()
        state.baseline_manifest_hash = baseline.manifest_hash
        state.apply_root = str(apply_root) if apply_root else None
        state.config = self.config.to_dict()
        state.params = to_jsonable(dict(params or {}))
        state.autonomy_mode = self.config.autonomy_mode.value
        engine = _Engine(self, state, run_dir)
        engine.attach()
        try:
            engine.emit(
                EventType.RUN_CREATED,
                actor=Actor.human(creator),
                scenario=scenario,
                created_by=creator,
                graph_hash=state.graph_hash,
                baseline_manifest_hash=state.baseline_manifest_hash,
                assurance=state.assurance,
                policy_hash=engine.policy_hash(),
                config_hash=hash_jsonable(self.config.to_dict()),
                posture=self.config.posture(),
            )
            engine._ingest(
                ArtifactDraft(
                    kind=K_REQUEST,
                    content=engine._clean_input(request_text, self.config.max_input_bytes),
                    media_type="text/markdown",
                ),
                by=creator,
            )
            engine.save()
        finally:
            engine.close()
        return rid

    def run(self, run_id: str) -> RunResult:
        return self._drive(run_id, resume=False, override=None)

    def resume(self, run_id: str, *, override: SafeStopOverride | None = None) -> RunResult:
        return self._drive(run_id, resume=True, override=override)

    def _drive(self, run_id: str, *, resume: bool, override: SafeStopOverride | None) -> RunResult:
        run_dir = self._store.run_dir(run_id)
        lock = RunLock(run_dir)
        lock.acquire()
        engine: _Engine | None = None
        try:
            state = self._store.load(run_id)
            if state.status.is_terminal and not (
                state.status == RunStatus.SAFE_STOPPED and override is not None
            ):
                raise StateError(f"run {run_id} is already {state.status.value}")
            check = verify_run(run_dir)
            if not check.ok and not check.torn_tail:
                raise IntegrityError(f"audit chain of run {run_id} is broken: {check.reason}")
            engine = _Engine(self, state, run_dir)
            engine._lock = lock
            engine.attach(audit_mode="recover" if check.torn_tail else "strict")
            if resume or state.status != RunStatus.CREATED:
                engine.begin_resume(override)
            else:
                state.status = RunStatus.RUNNING
                state.started_at = engine.now()
                engine.emit(EventType.RUN_STARTED)
                engine.save()
            return engine.loop()
        finally:
            if engine is not None:
                engine.close()
            lock.release()

    # ------------------------------------------------------------------ commands
    def _command(self, run_id: str, kind: str, **params: Any) -> Any:
        live = self._live_engines.get(run_id)
        if live is not None and live._live:
            cmd = _Command(kind, params, Future())
            live._commands.put(cmd)
            return cmd.future.result(timeout=COMMAND_WAIT_S)
        run_dir = self._store.run_dir(run_id)
        lock = RunLock(run_dir)
        if not lock.try_acquire():
            raise LockError(
                f"run {run_id} is being driven by another live process; wait for it to pause "
                "(use `agentic kill` for an emergency stop)"
            )
        try:
            engine = _Engine(self, self._store.load(run_id), run_dir)
            engine.attach()
            try:
                result = engine.apply_command(kind, **params)
                engine._flush_waves()
                engine.save()
            finally:
                engine.close()
            return result
        finally:
            lock.release()

    def approve(
        self,
        run_id: str,
        request_id: str,
        approver: str,
        comment: str = "",
        *,
        confirm: str = "",
        channel: str = "cli",
    ) -> dict[str, Any]:
        return self._command(
            run_id,
            "approve",
            request_id=request_id,
            approver=approver,
            approved=True,
            comment=comment,
            confirm=confirm,
            channel=channel,
        )

    def reject(
        self,
        run_id: str,
        request_id: str,
        approver: str,
        comment: str = "",
        *,
        confirm: str = "",
        channel: str = "cli",
    ) -> dict[str, Any]:
        return self._command(
            run_id,
            "reject",
            request_id=request_id,
            approver=approver,
            approved=False,
            comment=comment,
            confirm=confirm,
            channel=channel,
        )

    def answer(
        self,
        run_id: str,
        answers: Mapping[str, str],
        by: str,
        *,
        accept_default: Sequence[str] = (),
    ) -> dict[str, Any]:
        return self._command(
            run_id, "answer", answers=dict(answers), by=by, accept_default=tuple(accept_default)
        )

    def amend(self, run_id: str, new_request_text: str, by: str, reason: str) -> dict[str, Any]:
        return self._command(run_id, "amend", new_text=new_request_text, by=by, reason=reason)

    def kill(self, run_id: str, by: str, reason: str) -> RunResult | None:
        """Operator safe-stop: trips the kill switch (honoured by a live engine on its next tick,
        across processes); with no live engine the safe-stop procedure is performed right here."""
        run_dir = self._store.run_dir(run_id)
        switch = KillSwitch(self.runs_dir / "KILL", run_dir / "KILL")
        switch.trip("run", _clip(by, 64), _clip(reason, 200))
        if run_id in self._live_engines:
            return None
        lock = RunLock(run_dir)
        if not lock.try_acquire():
            return None
        try:
            state = self._store.load(run_id)
            if state.status.is_terminal:
                return None
            engine = _Engine(self, state, run_dir)
            engine._lock = lock
            engine.attach()
            try:
                return engine._safe_stop(
                    "kill_switch", f"kill switch tripped by {_clip(by, 64)}: {_clip(reason, 200)}"
                )
            finally:
                engine.close()
        finally:
            lock.release()

    # ------------------------------------------------------------------ read-only accessors
    def status(self, run_id: str) -> RunSummary:
        state = self._store.load(run_id)
        pending: list[Mapping[str, Any]] = []
        for req in state.approvals.values():
            if req.status == ApprovalStatus.PENDING:
                pending.append(
                    {
                        "id": req.id,
                        "node": req.node_id,
                        "purpose": req.purpose.value,
                        "risk": req.risk.name,
                        "required": req.required_approvals,
                        "votes": [
                            f"{v.approver}:{'yes' if v.approved else 'no'}" for v in req.votes
                        ],
                        "confirm": req.evidence_hash[:8],
                        "summary": req.summary,
                        "agent_note": req.agent_note,
                    }
                )
        questions: list[Mapping[str, Any]] = []
        for node_id, ns in state.nodes.items():
            if ns.status != NodeStatus.AWAITING_INPUT:
                continue
            rounds = [r for r in state.clarifications if r.node_id == node_id]
            if not rounds:
                continue
            for q in rounds[-1].questions:
                if q.id in ns.open_question_ids:
                    questions.append(
                        {
                            "id": q.id,
                            "node": node_id,
                            "text": q.text,
                            "options": list(q.options),
                            "default": q.default_assumption,
                            "blocking": q.blocking,
                        }
                    )
        head = state.audit_head
        return RunSummary(
            run_id=state.run_id,
            scenario=state.scenario,
            status=state.status,
            reason=state.status_reason,
            assurance=state.assurance,
            posture="see report",
            nodes={k: v.status.value for k, v in state.nodes.items()},
            pending_approvals=tuple(pending),
            open_questions=tuple(questions),
            audit_head=(int(head.get("seq", 0)), str(head.get("hash", ""))),
            counters=state.counters.to_dict(),
        )

    def events(self, run_id: str) -> list[AuditEvent]:
        return read_run_events(self._store.run_dir(run_id))

    def verify_audit(self, run_id: str, anchor: Any = None) -> VerifyResult:
        return verify_run(self._store.run_dir(run_id), anchor)

    def metrics(self, run_id: str) -> RunMetrics:
        state = self._store.load(run_id)
        graph = WorkflowGraph.from_dict(state.graph)
        return compute_run_metrics(self.events(run_id), graph=graph)

    def lineage(self, run_id: str, artifact_id: str) -> str:
        run_dir = self._store.run_dir(run_id)
        store = ArtifactStore(run_dir / "artifacts", clock=self.clock)
        decisions = DecisionLog(run_dir / "decisions.jsonl", run_id=run_id, clock=self.clock)
        return LineageGraph(store, decisions).explain(artifact_id)
