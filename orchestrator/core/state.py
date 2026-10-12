"""Persistent run state (the shape the engine, replan, safe-stop, approvals, report and CLI all share).

``state.json`` is a *cache of the audit chain's governance facts*, not the authority: on resume the
engine reconciles it against the hash-chained audit log (graph hash, approvals with their votes,
SUCCEEDED nodes with ``node.succeeded`` events, artifact index vs ``artifact.created`` events).
Security-relevant config (approval floors, budgets, kill-switch path) always comes from the
*current process*, merged with hard limits; the snapshot stored here is compared and recorded only.

Field semantics are normative — see ``docs/engine-semantics.md``.
"""

from __future__ import annotations

import enum
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.core.errors import PathViolation, StateError
from orchestrator.core.fsutil import atomic_write_json, read_bytes_retry
from orchestrator.core.paths import RUN_ID_RE, validate_rel_path
from orchestrator.core.serde import loads_strict, parse_datetime, to_jsonable
from orchestrator.core.types import (
    ApprovalRequest,
    ArtifactRef,
    ClarificationQuestion,
    Epoch,
    FailureKind,
    Finding,
    GateResult,
    NodeStatus,
    RiskAssessment,
    RunStatus,
    Waiver,
)

STATE_SCHEMA = 1
MAX_STATE_BYTES = 16 * 1024 * 1024


class AttemptOutcomeKind(enum.StrEnum):
    RUNNING = "running"
    PUBLISHED = "published"
    REJECTED = "rejected"  # exit gate / policy / human rejection
    FAILED = "failed"  # agent error
    CANCELLED = "cancelled"
    DISCARDED = "discarded"  # late result of a stale generation
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_INPUT = "awaiting_input"


def _dt(value: str | None) -> datetime | None:
    return parse_datetime(value) if value else None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


@dataclass
class AttemptRecord:
    number: int  # global per-node attempt number (AuditEvent.attempt / Artifact.attempt / span ids)
    generation: int
    epoch: Epoch
    agent_id: str
    started_at: datetime
    ended_at: datetime | None = None
    outcome: AttemptOutcomeKind = AttemptOutcomeKind.RUNNING
    failure_kind: FailureKind | None = None
    message: str = ""
    findings: tuple[Finding, ...] = ()
    gate_results: tuple[GateResult, ...] = ()
    risk: RiskAssessment | None = None
    input_fingerprint: str = ""
    inputs: tuple[ArtifactRef, ...] = ()
    staged: tuple[ArtifactRef, ...] = ()
    counts_vs_budget: bool = True  # False for cancelled / discarded / needs-input attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "generation": self.generation,
            "epoch": self.epoch.value,
            "agent_id": self.agent_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": _iso(self.ended_at),
            "outcome": self.outcome.value,
            "failure_kind": self.failure_kind.value if self.failure_kind else None,
            "message": self.message,
            "findings": [f.to_dict() for f in self.findings],
            "gate_results": [g.to_dict() for g in self.gate_results],
            "risk": self.risk.to_dict() if self.risk else None,
            "input_fingerprint": self.input_fingerprint,
            "inputs": [r.to_dict() for r in self.inputs],
            "staged": [r.to_dict() for r in self.staged],
            "counts_vs_budget": self.counts_vs_budget,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AttemptRecord:
        return cls(
            number=int(d["number"]),
            generation=int(d["generation"]),
            epoch=Epoch(d["epoch"]),
            agent_id=d["agent_id"],
            started_at=parse_datetime(d["started_at"]),
            ended_at=_dt(d.get("ended_at")),
            outcome=AttemptOutcomeKind(d.get("outcome", "running")),
            failure_kind=FailureKind(d["failure_kind"]) if d.get("failure_kind") else None,
            message=d.get("message", ""),
            findings=tuple(Finding.from_dict(f) for f in d.get("findings", ())),
            gate_results=tuple(GateResult.from_dict(g) for g in d.get("gate_results", ())),
            risk=RiskAssessment.from_dict(d["risk"]) if d.get("risk") else None,
            input_fingerprint=d.get("input_fingerprint", ""),
            inputs=tuple(ArtifactRef.from_dict(r) for r in d.get("inputs", ())),
            staged=tuple(ArtifactRef.from_dict(r) for r in d.get("staged", ())),
            counts_vs_budget=bool(d.get("counts_vs_budget", True)),
        )


@dataclass
class NodeState:
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    epoch_no: int = 0  # +1 on every STALE/reset/re-add/override; per-epoch budgets reset (lifetime cap applies)
    generation: int = 0  # ++ on every dispatch and every cancel; older-generation results are discarded
    attempt_counter: int = 0  # monotonic per node, never reset (the "attempt number" everywhere)
    epoch: Epoch = Epoch.PRIMARY  # which agent family is currently executing
    primary_attempts: int = 0  # consumed in the current epoch_no
    fallback_attempts: int = 0
    lifetime_attempts: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)
    not_before: datetime | None = None  # retry back-off gate (absolute UTC; persisted)
    fingerprint: str | None = None  # input fingerprint at the most recent dispatch / evaluation
    last_success_fingerprint: str | None = None
    outputs: tuple[ArtifactRef, ...] = ()  # currently ACTIVE published refs
    staged: tuple[ArtifactRef, ...] = ()  # STAGED refs awaiting approval / answers
    pending_attempt: int | None = None  # attempt number that owns ``staged``
    pending_approval_id: str | None = None
    open_question_ids: tuple[str, ...] = ()
    clarification_round: int = 0
    feedback: tuple[Finding, ...] = ()  # engine-authored feedback for the next attempt
    remediation_loops: int = 0
    reason: str = ""  # skip / block / stale / failure reason
    last_failure_kind: FailureKind | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cache_hits: int = 0
    receipt: dict[str, Any] | None = None  # ApplyReceipt dict once a side effect was applied
    zombie_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "epoch_no": self.epoch_no,
            "generation": self.generation,
            "attempt_counter": self.attempt_counter,
            "epoch": self.epoch.value,
            "primary_attempts": self.primary_attempts,
            "fallback_attempts": self.fallback_attempts,
            "lifetime_attempts": self.lifetime_attempts,
            "attempts": [a.to_dict() for a in self.attempts],
            "not_before": _iso(self.not_before),
            "fingerprint": self.fingerprint,
            "last_success_fingerprint": self.last_success_fingerprint,
            "outputs": [r.to_dict() for r in self.outputs],
            "staged": [r.to_dict() for r in self.staged],
            "pending_attempt": self.pending_attempt,
            "pending_approval_id": self.pending_approval_id,
            "open_question_ids": list(self.open_question_ids),
            "clarification_round": self.clarification_round,
            "feedback": [f.to_dict() for f in self.feedback],
            "remediation_loops": self.remediation_loops,
            "reason": self.reason,
            "last_failure_kind": self.last_failure_kind.value if self.last_failure_kind else None,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "cache_hits": self.cache_hits,
            "receipt": to_jsonable(self.receipt) if self.receipt is not None else None,
            "zombie_attempts": self.zombie_attempts,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> NodeState:
        return cls(
            node_id=d["node_id"],
            status=NodeStatus(d["status"]),
            epoch_no=int(d.get("epoch_no", 0)),
            generation=int(d.get("generation", 0)),
            attempt_counter=int(d.get("attempt_counter", 0)),
            epoch=Epoch(d.get("epoch", "primary")),
            primary_attempts=int(d.get("primary_attempts", 0)),
            fallback_attempts=int(d.get("fallback_attempts", 0)),
            lifetime_attempts=int(d.get("lifetime_attempts", 0)),
            attempts=[AttemptRecord.from_dict(a) for a in d.get("attempts", ())],
            not_before=_dt(d.get("not_before")),
            fingerprint=d.get("fingerprint"),
            last_success_fingerprint=d.get("last_success_fingerprint"),
            outputs=tuple(ArtifactRef.from_dict(r) for r in d.get("outputs", ())),
            staged=tuple(ArtifactRef.from_dict(r) for r in d.get("staged", ())),
            pending_attempt=d.get("pending_attempt"),
            pending_approval_id=d.get("pending_approval_id"),
            open_question_ids=tuple(d.get("open_question_ids", ())),
            clarification_round=int(d.get("clarification_round", 0)),
            feedback=tuple(Finding.from_dict(f) for f in d.get("feedback", ())),
            remediation_loops=int(d.get("remediation_loops", 0)),
            reason=d.get("reason", ""),
            last_failure_kind=FailureKind(d["last_failure_kind"]) if d.get("last_failure_kind") else None,
            started_at=_dt(d.get("started_at")),
            finished_at=_dt(d.get("finished_at")),
            cache_hits=int(d.get("cache_hits", 0)),
            receipt=dict(d["receipt"]) if d.get("receipt") is not None else None,
            zombie_attempts=int(d.get("zombie_attempts", 0)),
        )


@dataclass
class Counters:
    """Run-wide bounded budgets. See engine-semantics §9 for what counts toward each."""

    budgeted_attempts: int = 0  # retries + fallbacks (first attempts of an epoch are not counted)
    retries: int = 0
    fallbacks: int = 0
    compensations: int = 0  # node-level + run-level rollbacks (attempt quarantines are NOT counted)
    replans: int = 0  # agent/failure-initiated re-plan waves
    human_inputs: int = 0  # amend / answer
    remediation_loops: int = 0
    discarded_attempts: int = 0
    zombies: int = 0
    safe_stop_overrides: int = 0
    breaker_failures: dict[str, int] = field(default_factory=dict)  # per agent_id consecutive failures
    breaker_open: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budgeted_attempts": self.budgeted_attempts,
            "retries": self.retries,
            "fallbacks": self.fallbacks,
            "compensations": self.compensations,
            "replans": self.replans,
            "human_inputs": self.human_inputs,
            "remediation_loops": self.remediation_loops,
            "discarded_attempts": self.discarded_attempts,
            "zombies": self.zombies,
            "safe_stop_overrides": self.safe_stop_overrides,
            "breaker_failures": dict(self.breaker_failures),
            "breaker_open": list(self.breaker_open),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Counters:
        c = cls()
        for name in (
            "budgeted_attempts",
            "retries",
            "fallbacks",
            "compensations",
            "replans",
            "human_inputs",
            "remediation_loops",
            "discarded_attempts",
            "zombies",
            "safe_stop_overrides",
        ):
            setattr(c, name, int(d.get(name, 0)))
        c.breaker_failures = {str(k): int(v) for k, v in d.get("breaker_failures", {}).items()}
        c.breaker_open = [str(x) for x in d.get("breaker_open", ())]
        return c


@dataclass
class ClarificationRound:
    round: int
    node_id: str
    questions: tuple[ClarificationQuestion, ...]
    asked_at: datetime
    answered_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "node_id": self.node_id,
            "questions": [q.to_dict() for q in self.questions],
            "asked_at": self.asked_at.isoformat(),
            "answered_at": _iso(self.answered_at),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ClarificationRound:
        return cls(
            int(d["round"]),
            d["node_id"],
            tuple(ClarificationQuestion.from_dict(q) for q in d.get("questions", ())),
            parse_datetime(d["asked_at"]),
            _dt(d.get("answered_at")),
        )


@dataclass
class Interval:
    start: datetime
    end: datetime | None = None  # open interval is closed at ``last_heartbeat`` on crash recovery

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "end": _iso(self.end)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Interval:
        return cls(parse_datetime(d["start"]), _dt(d.get("end")))


@dataclass
class SafeStopRecord:
    trigger: str  # kill_switch | budget | circuit_breaker | critical_finding | autonomy_violation | integrity | engine_fault | compensation_failed | worker_exhaustion
    reason: str
    step: int = 0  # last completed journal step S0..S6 (crash-resumable)
    compensation_incomplete: list[str] = field(default_factory=list)
    decision_id: str | None = None
    kill_acked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "reason": self.reason,
            "step": self.step,
            "compensation_incomplete": list(self.compensation_incomplete),
            "decision_id": self.decision_id,
            "kill_acked": list(self.kill_acked),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SafeStopRecord:
        return cls(
            d["trigger"],
            d.get("reason", ""),
            int(d.get("step", 0)),
            list(d.get("compensation_incomplete", ())),
            d.get("decision_id"),
            list(d.get("kill_acked", ())),
        )


@dataclass
class RunState:
    run_id: str
    scenario: str
    created_by: str
    created_at: datetime
    trace_id: str
    status: RunStatus = RunStatus.CREATED
    status_reason: str = ""
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    version: int = 0  # +1 on every save (optimistic concurrency for offline mutators)
    audit_head: dict[str, Any] = field(default_factory=lambda: {"seq": 0, "hash": ""})
    assurance: str = "HUMAN"  # "HUMAN" | "DEMO_SCRIPTED" (any scripted vote downgrades; shown in every report)
    autonomy_mode: str = "guided"  # supervised | guided | autonomous
    config: dict[str, Any] = field(default_factory=dict)  # engine config snapshot (recorded, never trusted)
    params: dict[str, Any] = field(default_factory=dict)  # scenario parameters
    baseline_manifest_hash: str = ""
    apply_root: str | None = None
    graph: dict[str, Any] = field(default_factory=dict)  # WorkflowGraph.to_dict()
    graph_hash: str = ""
    mandatory_snapshot: dict[str, str] = field(default_factory=dict)  # node id -> spec_hash pinned at start
    nodes: dict[str, NodeState] = field(default_factory=dict)
    counters: Counters = field(default_factory=Counters)
    approvals: dict[str, ApprovalRequest] = field(default_factory=dict)
    waivers: list[Waiver] = field(default_factory=list)
    clarifications: list[ClarificationRound] = field(default_factory=list)
    human_wait: list[Interval] = field(default_factory=list)
    active_runtime_s: float = 0.0  # excludes human-wait
    safe_stop: SafeStopRecord | None = None
    resume_count: int = 0
    last_heartbeat: datetime | None = None
    applied_commands: list[str] = field(default_factory=list)  # inbox command ids already applied
    decision_counter: int = 0
    approval_counter: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "scenario": self.scenario,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "trace_id": self.trace_id,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "updated_at": _iso(self.updated_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "version": self.version,
            "audit_head": dict(self.audit_head),
            "assurance": self.assurance,
            "autonomy_mode": self.autonomy_mode,
            "config": to_jsonable(self.config),
            "params": to_jsonable(self.params),
            "baseline_manifest_hash": self.baseline_manifest_hash,
            "apply_root": self.apply_root,
            "graph": to_jsonable(self.graph),
            "graph_hash": self.graph_hash,
            "mandatory_snapshot": dict(self.mandatory_snapshot),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "counters": self.counters.to_dict(),
            "approvals": {k: v.to_dict() for k, v in self.approvals.items()},
            "waivers": [w.to_dict() for w in self.waivers],
            "clarifications": [c.to_dict() for c in self.clarifications],
            "human_wait": [i.to_dict() for i in self.human_wait],
            "active_runtime_s": self.active_runtime_s,
            "safe_stop": self.safe_stop.to_dict() if self.safe_stop else None,
            "resume_count": self.resume_count,
            "last_heartbeat": _iso(self.last_heartbeat),
            "applied_commands": list(self.applied_commands),
            "decision_counter": self.decision_counter,
            "approval_counter": self.approval_counter,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunState:
        if d.get("schema") != STATE_SCHEMA:
            raise StateError(f"unsupported state schema: {d.get('schema')!r}")
        return cls(
            run_id=d["run_id"],
            scenario=d["scenario"],
            created_by=d["created_by"],
            created_at=parse_datetime(d["created_at"]),
            trace_id=d["trace_id"],
            status=RunStatus(d.get("status", "created")),
            status_reason=d.get("status_reason", ""),
            updated_at=_dt(d.get("updated_at")),
            started_at=_dt(d.get("started_at")),
            finished_at=_dt(d.get("finished_at")),
            version=int(d.get("version", 0)),
            audit_head=dict(d.get("audit_head", {"seq": 0, "hash": ""})),
            assurance=d.get("assurance", "HUMAN"),
            autonomy_mode=d.get("autonomy_mode", "guided"),
            config=dict(d.get("config", {})),
            params=dict(d.get("params", {})),
            baseline_manifest_hash=d.get("baseline_manifest_hash", ""),
            apply_root=d.get("apply_root"),
            graph=dict(d.get("graph", {})),
            graph_hash=d.get("graph_hash", ""),
            mandatory_snapshot=dict(d.get("mandatory_snapshot", {})),
            nodes={k: NodeState.from_dict(v) for k, v in d.get("nodes", {}).items()},
            counters=Counters.from_dict(d.get("counters", {})),
            approvals={k: ApprovalRequest.from_dict(v) for k, v in d.get("approvals", {}).items()},
            waivers=[Waiver.from_dict(w) for w in d.get("waivers", ())],
            clarifications=[ClarificationRound.from_dict(c) for c in d.get("clarifications", ())],
            human_wait=[Interval.from_dict(i) for i in d.get("human_wait", ())],
            active_runtime_s=float(d.get("active_runtime_s", 0.0)),
            safe_stop=SafeStopRecord.from_dict(d["safe_stop"]) if d.get("safe_stop") else None,
            resume_count=int(d.get("resume_count", 0)),
            last_heartbeat=_dt(d.get("last_heartbeat")),
            applied_commands=list(d.get("applied_commands", ())),
            decision_counter=int(d.get("decision_counter", 0)),
            approval_counter=int(d.get("approval_counter", 0)),
        )


class StateStore:
    """Atomic persistence of ``RunState`` (``<runs_dir>/<run_id>/state.json``).

    Not multi-process safe by itself: mutating processes must hold ``fsutil.RunLock`` (the engine
    holds it for the lifetime of ``run``/``resume``; offline CLI mutators take it briefly).
    """

    def __init__(self, runs_dir: Path | str) -> None:
        self.runs_dir = Path(runs_dir)
        self._lock = threading.Lock()

    def run_dir(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
            raise StateError(f"invalid run id: {run_id!r}")
        try:
            # Windows silently drops trailing dots ("a." is directory "a") and treats "con"/"nul"/...
            # as devices; validate_rel_path rejects exactly those aliases
            validate_rel_path(run_id)
        except PathViolation as exc:
            raise StateError(f"invalid run id: {run_id!r} ({exc})") from exc
        path = (self.runs_dir / run_id).resolve()
        if not path.is_relative_to(self.runs_dir.resolve()):
            raise StateError(f"run id escapes runs dir: {run_id!r}")
        return path

    def exists(self, run_id: str) -> bool:
        return (self.run_dir(run_id) / "state.json").is_file()

    def create(self, run_id: str) -> Path:
        path = self.run_dir(run_id)
        if path.exists():
            raise StateError(f"run already exists: {run_id}")
        path.mkdir(parents=True)
        return path

    def save(self, state: RunState, *, now: datetime | None = None) -> None:
        with self._lock:
            state.version += 1
            if now is not None:
                state.updated_at = now
            atomic_write_json(self.run_dir(state.run_id) / "state.json", state.to_dict())

    def load(self, run_id: str) -> RunState:
        path = self.run_dir(run_id) / "state.json"
        try:
            raw = read_bytes_retry(path)
        except FileNotFoundError as exc:
            raise StateError(f"no such run: {run_id}") from exc
        try:
            state = RunState.from_dict(loads_strict(raw, max_bytes=MAX_STATE_BYTES))
        except (ValueError, LookupError, TypeError, AttributeError, OverflowError) as exc:
            # AttributeError: a well-formed JSON document of the wrong shape (e.g. a list) hits ``.get``
            raise StateError(f"corrupt state for run {run_id}: {exc}") from exc
        if state.run_id != run_id:
            # ``save`` writes to ``run_dir(state.run_id)``: a copied/tampered file must not redirect it
            raise StateError(f"state file of run {run_id} claims to belong to run {state.run_id!r}")
        return state

    def list_runs(self) -> list[str]:
        if not self.runs_dir.is_dir():
            return []
        return sorted(p.name for p in self.runs_dir.iterdir() if (p / "state.json").is_file())
