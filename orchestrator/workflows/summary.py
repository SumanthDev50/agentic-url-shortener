"""Engineering-summary finalizer (L5; engine-semantics section 18).

``engineering_summary_finalizer`` is injected into ``Orchestrator(finalizers=[...])``.  It runs for
EVERY terminal state (success, failure, safe-stop) and publishes one ``engineering_summary``
artifact (Markdown) with the five required sections:

1. Plan and rationale   - requirements, decomposition tasks, design ADRs with their alternatives,
                          and every recorded decision with its id;
2. Artifacts produced   - kind, id and hash prefix of each active artifact, the patch files;
3. Risks, trade-offs and validation - threat-model risks, policy findings and waivers, test
                          results, fail-to-pass, API compatibility, security report, readiness
                          verdict, approvals with four-eyes and assurance;
4. Assumptions          - requirement assumptions, deferred scope with decision ids, answered
                          clarifications;
5. Limitations          - COMPUTED from the facts of the run, never boilerplate about things that
                          did not happen: unsupported or deferred requirements, scripted approvals
                          (DEMO_SCRIPTED), injected faults, skipped nodes and checks, failed /
                          blocked / cancelled nodes and why, the safe-stop reason, the manual steps
                          left (review and apply the patch), attested-only identity, deterministic
                          reference agents, final status and audit head.

Citations
---------
Every bullet cites its evidence in parentheses as code spans: an artifact id (``design_doc@v1``), a
decision id (``D-0007``) or, for facts that only the audit log records (a skipped node, a policy
violation, the final status), the audit event as ``audit:<seq>``.  Section 2 cites by listing the
ids themselves.

Tolerance
---------
A failed or safe-stopped run misses artifacts.  Every read goes through ``ctx.view`` defensively: an
absent artifact becomes ``not produced: node X failed - <reason>`` (the producing node comes from
the view's node specs, else from the standard SDLC contract), an unreadable one (failed integrity
check, malformed JSON) is reported as such, and a minimal view double that lacks the optional
accessors (``artifacts``, ``waivers``, ``node_specs``, ``params``) is tolerated.

Determinism and safety
----------------------
The content has no timestamps, durations, pids, random ids or absolute paths (artifact content
rule).  Everything from artifacts, agents or humans passes ``core.sanitize`` through the helpers of
``engine/report.py`` (single line, escaped Markdown, no raw HTML, no links; identifiers as
allow-listed code spans; the patch file list in a fence that cannot be closed from the inside).
Lists are capped and say how much was left out.

Injected faults are read from the run parameter ``faults`` (``[{node, fault, attempts}]``, the
scenario runner records it); when it is absent no fault claim is made.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from orchestrator.core.errors import OrchestratorError
from orchestrator.core.sanitize import code_fence
from orchestrator.core.schemas import ALL_KINDS, Kind
from orchestrator.core.types import (
    ActorType,
    ApprovalRequest,
    Artifact,
    ArtifactDraft,
    AuditEvent,
    Decision,
    DecisionKind,
    EventType,
    NodeSpec,
    NodeStatus,
    RunStatus,
    RunView,
    Waiver,
)
from orchestrator.engine.report import (
    ASSURANCE_SCRIPTED,
    SCRIPTED_CHANNEL,
    md_ident,
    md_text,
)

if TYPE_CHECKING:
    from orchestrator.engine.scheduler import FinalizerContext

__all__ = ["engineering_summary_finalizer", "render_engineering_summary"]

T = TypeVar("T")

_ET = EventType
_READ_ERRORS = (OrchestratorError, OSError, ValueError, LookupError, TypeError, AttributeError)
_ITEMS = 80  # entries shown per list; the rest is counted, not listed
_TEXT = 200
_SHORT = 120
_HASH_PREFIX = 12
_HEX = frozenset("0123456789abcdef")
_DIFF_HEADER = re.compile(r"^diff --git a/\S+ b/(\S+)$")
_NOTE_EVENTS = frozenset(
    {_ET.NODE_FAILED, _ET.NODE_BLOCKED, _ET.NODE_SKIPPED, _ET.NODE_CANCELLED, _ET.NODE_ROLLED_BACK}
)
_FAULT_DECISIONS = frozenset(
    {DecisionKind.RETRY, DecisionKind.REJECTION, DecisionKind.FALLBACK, DecisionKind.REMEDIATION}
)
_APPROVAL_DECISIONS = frozenset(
    {DecisionKind.APPROVAL, DecisionKind.REJECTION, DecisionKind.WAIVER}
)
# Producing node per artifact kind in the standard workflow (docs/sdlc-contract.md); used only
# when the view offers no node specs.
_PRODUCERS: Mapping[str, str] = {
    Kind.INTAKE_REPORT: "intake",
    Kind.REQUIREMENTS: "requirements",
    Kind.IMPACT_REPORT: "codebase-analysis",
    Kind.THREAT_MODEL: "threat-model",
    Kind.DESIGN_DOC: "design",
    Kind.API_CONTRACT: "design",
    Kind.TASK_PLAN: "decompose",
    Kind.INTEGRATED_CHANGESET: "integrate",
    Kind.INTEGRATION_REPORT: "integrate",
    Kind.TEST_REPORT: "verify",
    Kind.SECURITY_REPORT: "security-scan",
    Kind.DOCS_CHANGESET: "docs",
    Kind.READINESS_REPORT: "release-readiness",
    Kind.TRACEABILITY_MATRIX: "release-readiness",
    Kind.RELEASE_PATCH: "promote",
    Kind.RELEASE_BUNDLE: "promote",
    Kind.RELEASE_VERIFICATION: "release-verify",
}
_NO_STATUS = "has no recorded status"
_NODE_VERB: Mapping[NodeStatus, str] = {
    NodeStatus.FAILED: "failed",
    NodeStatus.BLOCKED: "was blocked",
    NodeStatus.CANCELLED: "was cancelled",
    NodeStatus.SKIPPED: "was skipped",
    NodeStatus.ROLLED_BACK: "was rolled back",
    NodeStatus.REMOVED: "was removed by a re-plan",
    NodeStatus.PENDING: "never started",
    NodeStatus.READY: "never started",
    NodeStatus.RUNNING: "was still running",
    NodeStatus.AWAITING_APPROVAL: "was waiting for approval",
    NodeStatus.AWAITING_INPUT: "was waiting for input",
    NodeStatus.STALE: "was stale (revalidation pending)",
    NodeStatus.SUCCEEDED: "succeeded but left no active artifact of this kind",
}


# =============================================================================
# small helpers
# =============================================================================


def _safe(read: Callable[[], T], default: T) -> T:
    try:
        return read()
    except _READ_ERRORS:
        return default


def _read(view: object, accessor: str, default: T) -> T:
    """``view.<accessor>()``; ``default`` if the view lacks the accessor or the read fails."""
    try:
        return getattr(view, accessor)()
    except _READ_ERRORS:
        return default


def _map(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: object) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _strs(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _reason_of(data: Mapping[str, Any]) -> str:
    """Why a node ended as it did: ``reason``, else ``kind: message`` of a failure event."""
    reason = data.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    message, kind = data.get("message"), data.get("kind")
    if isinstance(message, str):
        return f"{kind}: {message}" if isinstance(kind, str) else message
    return ""


def _hex(value: object, limit: int = 64) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.lower() if ch in _HEX)[:limit]


def _t(value: object, limit: int = _TEXT) -> str:
    return md_text("" if value is None else value, limit)


def _id(value: object) -> str:
    return "-" if value is None or value == "" else md_ident(value)


def _cite(*ids: str | None) -> str:
    unique = list(dict.fromkeys(i for i in ids if i))
    return " (" + ", ".join(md_ident(i) for i in unique) + ")" if unique else ""


def _yes(value: object) -> str:
    return "yes" if value is True else "no" if value is False else "not checked"


def _more(count: int, what: str, cite: str) -> list[str]:
    return [f"- ... {count} more {what} not listed{cite}"] if count > 0 else []


def _join_ids(values: Iterable[object], limit: int = 8) -> str:
    items = [md_ident(v) for v in values if v not in (None, "")]
    extra = f", +{len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(items[:limit]) + extra if items else "none"


# =============================================================================
# the run as the finalizer sees it
# =============================================================================


class _Source:
    """Defensive reader of the finalizer context (a failed run has missing artifacts)."""

    def __init__(self, ctx: FinalizerContext) -> None:
        view: RunView = ctx.view
        self.view = view
        self.run_id = ctx.run_id
        self.status: RunStatus = ctx.status
        self.reason = ctx.reason
        self.head_seq = int(ctx.audit_head[0])
        self.head_hash = _hex(ctx.audit_head[1])
        self.events: tuple[AuditEvent, ...] = tuple(_read(view, "events", ()))
        self.decisions: tuple[Decision, ...] = tuple(_read(view, "decisions", ()))
        self.approvals: tuple[ApprovalRequest, ...] = tuple(
            sorted(_read(view, "approvals", ()), key=lambda r: (r.created_at, r.id))
        )
        self.waivers: tuple[Waiver, ...] = tuple(
            sorted(_read(view, "waivers", ()), key=lambda w: w.id)
        )
        self.statuses: Mapping[str, NodeStatus] = _read(view, "node_statuses", {})
        self.specs: Mapping[str, NodeSpec] = _read(view, "node_specs", {})
        self.params: Mapping[str, Any] = _map(_read(view, "params", {}))
        self.active: tuple[Artifact, ...] = self._active_artifacts()
        self.unreadable: list[tuple[str, str]] = []
        self._docs: dict[str, tuple[Artifact, Mapping[str, Any]] | None] = {}
        self.head_cite = _cite(f"audit:{self.head_seq}")
        self.by_id: dict[str, Decision] = {d.id: d for d in self.decisions}
        self.scripted_requests: list[ApprovalRequest] = [
            a for a in self.approvals if any(v.channel == SCRIPTED_CHANNEL for v in a.votes)
        ]
        # one pass over the audit log: node reasons, approval events, agents, scripted votes
        self.notes: dict[str, tuple[int, str]] = {}
        self.approval_seqs: dict[str, list[int]] = {}
        self.vote_seqs: list[int] = []
        self.scripted_vote_seqs: list[int] = []
        self.failures: dict[str, list[int]] = {}
        self.first_started = self.head_seq
        seen_start = False
        agents: set[str] = set()
        for event in self.events:
            data = _map(event.data)
            if event.node_id is not None and event.type in _NOTE_EVENTS:
                self.notes[event.node_id] = (event.seq, _reason_of(data))
            if event.type.value.startswith("approval.") and isinstance(
                data.get("approval_id"), str
            ):
                self.approval_seqs.setdefault(data["approval_id"], []).append(event.seq)
            if event.type is _ET.NODE_ATTEMPT_FAILED and event.node_id is not None:
                self.failures.setdefault(event.node_id, []).append(event.seq)
            if event.type is _ET.APPROVAL_VOTE:
                self.vote_seqs.append(event.seq)
                if data.get("channel") == SCRIPTED_CHANNEL:
                    self.scripted_vote_seqs.append(event.seq)
            if event.type is _ET.NODE_STARTED:
                if not seen_start:
                    self.first_started, seen_start = event.seq, True
                if data.get("agent_id"):
                    agents.add(str(data["agent_id"]))
        self.agents: list[str] = sorted(agents)
        self.scripted = bool(self.scripted_requests or self.scripted_vote_seqs)

    # -- reading ---------------------------------------------------------------------------
    def _active_artifacts(self) -> tuple[Artifact, ...]:
        try:
            return tuple(self.view.artifacts())
        except AttributeError:  # a minimal view: the latest artifact of every known kind
            found = [_safe(partial(self.view.latest, kind), None) for kind in sorted(ALL_KINDS)]
            return tuple(a for a in found if a is not None)
        except _READ_ERRORS:
            return ()

    def artifact(self, kind: str) -> Artifact | None:
        return _safe(partial(self.view.latest, kind), None)

    def doc(self, kind: str) -> tuple[Artifact, Mapping[str, Any]] | None:
        """``(artifact, payload)`` of the latest ``kind``; ``None`` if absent or unreadable."""
        if kind in self._docs:
            return self._docs[kind]
        art = self.artifact(kind)
        result: tuple[Artifact, Mapping[str, Any]] | None = None
        if art is not None:
            try:
                raw = self.view.read_json(art)
            except _READ_ERRORS as exc:
                self.unreadable.append((art.id, type(exc).__name__))
            else:
                if isinstance(raw, Mapping):
                    result = (art, raw)
                else:
                    self.unreadable.append((art.id, "malformed"))
        self._docs[kind] = result
        return result

    def text(self, art: Artifact) -> str | None:
        try:
            return str(self.view.read_text(art))
        except _READ_ERRORS as exc:
            self.unreadable.append((art.id, type(exc).__name__))
            return None

    # -- explaining what is missing --------------------------------------------------------
    def producer(self, kind: str) -> str | None:
        """The node producing ``kind``: from the node specs, else (no specs) the standard SDLC."""
        if not self.specs:
            return _PRODUCERS.get(kind)
        return next((n for n in sorted(self.specs) if kind in self.specs[n].produces), None)

    def absent(self, kind: str) -> str:
        """One bullet saying why ``kind`` has no readable artifact."""
        art = self.artifact(kind)
        if art is not None:
            return (
                f"- {md_ident(kind)} could not be read: integrity or format check failed"
                f"{_cite(art.id)}"
            )
        node = self.producer(kind)
        if node is None:
            return f"- not produced: no node of this run produces {md_ident(kind)}{self.head_cite}"
        status = self.statuses.get(node)
        verb = _NODE_VERB.get(status, _NO_STATUS) if status is not None else _NO_STATUS
        seq, reason = self.notes.get(node, (self.head_seq, ""))
        why = f" - {_t(reason, _SHORT)}" if reason else ""
        return (
            f"- not produced: {md_ident(kind)}: node {md_ident(node)} {verb}{why}"
            f"{_cite(f'audit:{seq}')}"
        )

    # -- decisions and approvals ------------------------------------------------------------
    def decision_ids(self, test: Callable[[Decision], bool], limit: int = 3) -> list[str]:
        return [d.id for d in self.decisions if test(d)][:limit]

    def scope_decisions(self, requirement_id: str) -> list[str]:
        return self.decision_ids(
            lambda d: (
                d.kind is DecisionKind.SCOPE
                and (
                    requirement_id in d.refs or _map(d.data).get("requirement_id") == requirement_id
                )
            )
        )

    def assumption_decisions(self, assumption_id: str) -> list[str]:
        return self.decision_ids(
            lambda d: d.kind is DecisionKind.ASSUMPTION and assumption_id in d.refs
        )

    def clarification_decisions(self, question_id: str) -> list[str]:
        return self.decision_ids(
            lambda d: (
                d.kind is DecisionKind.CLARIFICATION
                and _map(d.data).get("question_id") == question_id
            )
        )

    def node_decisions(
        self, node_id: str, kinds: frozenset[DecisionKind] | None = None
    ) -> list[str]:
        return self.decision_ids(
            lambda d: d.node_id == node_id and (kinds is None or d.kind in kinds)
        )

    def node_cite(self, node_id: str) -> str:
        """Decisions of the node, else the audit event that last changed its status."""
        ids = self.node_decisions(node_id)
        if ids:
            return _cite(*ids)
        seq = self.notes.get(node_id, (self.head_seq, ""))[0]
        return _cite(f"audit:{seq}")

    def approval_cite(self, request: ApprovalRequest) -> str:
        ids = self.node_decisions(request.node_id, _APPROVAL_DECISIONS)
        if ids:
            return _cite(*ids)
        seqs = self.approval_seqs.get(request.id, [])[:3]
        return _cite(*(f"audit:{s}" for s in seqs)) if seqs else self.head_cite

    def clarification_parents(self, decision_ids: Iterable[str], limit: int = 2) -> list[str]:
        """Human clarification decisions the given decisions build on."""
        found = [
            parent
            for decision_id in decision_ids
            for parent in self.by_id[decision_id].parents
            if parent in self.by_id and self.by_id[parent].kind is DecisionKind.CLARIFICATION
        ]
        return found[:limit]


# =============================================================================
# header
# =============================================================================


def _header(src: _Source) -> list[str]:
    reason = f" - {_t(src.reason)}" if src.reason else ""
    assurance = (
        f"{md_ident(ASSURANCE_SCRIPTED)} (approvals were scripted, not a human review)"
        if src.scripted
        else f"{md_ident('HUMAN')} (no scripted approval was recorded)"
    )
    return [
        "# Engineering summary",
        "",
        f"- Run: {md_ident(src.run_id)}{src.head_cite}",
        f"- Final status: {md_ident(src.status.value)}{reason}{src.head_cite}",
        f"- Audit head at the terminal state: seq {src.head_seq}, hash `{src.head_hash or '-'}`"
        f"{src.head_cite}",
        f"- Assurance: {assurance}{src.head_cite}",
        "",
    ]


# =============================================================================
# 1. Plan and rationale
# =============================================================================


def _requirements(src: _Source) -> list[str]:
    out = ["### Requirements", ""]
    found = src.doc(Kind.REQUIREMENTS)
    if found is None:
        return [*out, src.absent(Kind.REQUIREMENTS), ""]
    art, data = found
    cite = _cite(art.id)
    goal = _map(data.get("problem")).get("goal")
    if goal:
        out.append(f"- Goal: {_t(goal)}{cite}")
    reqs = _rows(data.get("requirements"))
    for req in reqs[:_ITEMS]:
        criteria = len(_rows(req.get("acceptance_criteria")))
        out.append(
            f"- {_id(req.get('id'))} {_t(req.get('text'), 240)} - {_t(req.get('priority'), 16)}, "
            f"{_t(req.get('kind'), 24)}, {_t(req.get('status'), 16)}, "
            f"{criteria} acceptance criteria{cite}"
        )
    out += _more(len(reqs) - _ITEMS, "requirements", cite)
    if not reqs:
        out.append(f"- no requirement was recorded{cite}")
    return [*out, ""]


def _decomposition(src: _Source) -> list[str]:
    out = ["### Decomposition", ""]
    found = src.doc(Kind.TASK_PLAN)
    if found is None:
        return [*out, src.absent(Kind.TASK_PLAN), ""]
    art, data = found
    cite = _cite(art.id)
    tasks = _rows(data.get("tasks"))
    for task in tasks[:_ITEMS]:
        covers = _join_ids(_strs(task.get("requirement_ids")), 6)
        after = _join_ids(_strs(task.get("depends_on")), 6)
        out.append(
            f"- {_id(task.get('id'))} {_t(task.get('title'), 160)} - "
            f"group {_id(task.get('group'))}, {_t(task.get('kind'), 16)}; "
            f"covers {covers}; depends on {after}{cite}"
        )
    out += _more(len(tasks) - _ITEMS, "tasks", cite)
    if not tasks:
        out.append(f"- no task was planned{cite}")
    groups = _rows(data.get("groups"))
    if groups:
        listed = ", ".join(
            f"{_id(g.get('id'))} -> {_id(g.get('node_id'))}" for g in groups[:_ITEMS]
        )
        out.append(f"- Groups and their nodes: {listed}{cite}")
    path = _strs(data.get("critical_path"))
    if path:
        out.append(f"- Critical path: {' -> '.join(md_ident(p) for p in path[:_ITEMS])}{cite}")
    return [*out, ""]


def _design(src: _Source) -> list[str]:
    out = ["### Design decisions (ADRs)", ""]
    found = src.doc(Kind.DESIGN_DOC)
    if found is None:
        return [*out, src.absent(Kind.DESIGN_DOC), ""]
    art, data = found
    cite = _cite(art.id)
    out.append(
        f"- Design: {_t(data.get('summary'))} - {len(_rows(data.get('components')))} components, "
        f"change reference {_id(data.get('change_reference'))}{cite}"
    )
    adrs = _rows(data.get("adrs"))
    for adr in adrs[:_ITEMS]:
        out.append(f"- {_id(adr.get('id'))} {_t(adr.get('title'), 160)}{cite}")
        out.append(f"  - Decision: {_t(adr.get('decision'), 400)}")
        alternatives = [
            f"{_t(a.get('option'), 100)} (rejected because: {_t(a.get('rejected_because'), 200)})"
            for a in _rows(adr.get("alternatives"))[:6]
        ]
        out.append("  - Alternatives considered: " + ("; ".join(alternatives) or "none recorded"))
        out.append(f"  - Rationale: {_t(adr.get('rationale'), 400)}")
        out.append(f"  - Consequences: {_t(adr.get('consequences'), 300)}")
    out += _more(len(adrs) - _ITEMS, "ADRs", cite)
    if not adrs:
        out.append(f"- no ADR was recorded{cite}")
    return [*out, ""]


def _decision_log(src: _Source) -> list[str]:
    out = ["### Decisions", ""]
    if not src.decisions:
        return [*out, f"- no decision was recorded{src.head_cite}", ""]
    for d in src.decisions[:_ITEMS]:
        line = f"- {_id(d.id)} {_id(d.kind.value)} by {_id(d.actor.label)}"
        if d.node_id:
            line += f" on {_id(d.node_id)}"
        line += f": {_t(d.summary, 160)}"
        if d.rationale:
            line += f" - why: {_t(d.rationale, 160)}"
        if d.alternatives:
            line += "; alternatives: " + ", ".join(_t(a, 60) for a in d.alternatives[:3])
        if d.parents:
            line += f"; builds on {_join_ids(d.parents, 5)}"
        out.append(line)
    out += _more(len(src.decisions) - _ITEMS, "decisions", src.head_cite)
    return [*out, ""]


def _plan_section(src: _Source) -> list[str]:
    return [
        "## 1. Plan and rationale",
        "",
        *_requirements(src),
        *_decomposition(src),
        *_design(src),
        *_decision_log(src),
    ]


# =============================================================================
# 2. Artifacts produced
# =============================================================================


def _patch_paths(text: str) -> list[str]:
    paths = [m.group(1) for line in text.splitlines() if (m := _DIFF_HEADER.match(line))]
    return paths[:200]


def _patch_files(src: _Source) -> list[str]:
    out = ["### Patch files", ""]
    patch = src.artifact(Kind.RELEASE_PATCH)
    bundle = src.doc(Kind.RELEASE_BUNDLE)
    files: list[str] = []
    cites: list[str] = []
    mode = ""
    if bundle is not None:
        files = _strs(bundle[1].get("files"))[:200]
        mode = str(bundle[1].get("mode", ""))
        cites.append(bundle[0].id)
    if patch is not None:
        cites.append(patch.id)
        if not files:
            text = src.text(patch)
            files = _patch_paths(text) if text is not None else []
    if not files:
        if patch is None and bundle is None:
            return [*out, src.absent(Kind.RELEASE_PATCH), ""]
        return [*out, f"- the patch lists no file{_cite(*cites)}", ""]
    mode_note = f" (release mode {_t(mode, 16)})" if mode else ""
    out += [f"- {len(files)} file(s) in the patch{mode_note}{_cite(*cites)}:", ""]
    out += [code_fence("\n".join(files)), ""]
    return out


def _artifacts_section(src: _Source) -> list[str]:
    out = ["## 2. Artifacts produced", ""]
    active = sorted(src.active, key=lambda a: (a.kind, a.key, a.version, a.id))
    if not active:
        out += [f"- no artifact is active: the run produced nothing{src.head_cite}", ""]
    else:
        out += ["| Kind | Id | Producer | Hash |", "| --- | --- | --- | --- |"]
        out += [
            f"| {_id(a.kind)} | {_id(a.id)} | {_id(a.producer)} | "
            f"`{_hex(a.content_hash, _HASH_PREFIX) or '-'}` |"
            for a in active[: 2 * _ITEMS]
        ]
        out.append("")
        out += _more(len(active) - 2 * _ITEMS, "artifacts", src.head_cite)
        if len(active) > 2 * _ITEMS:
            out.append("")
    return [*out, *_patch_files(src)]


# =============================================================================
# 3. Risks, trade-offs and validation
# =============================================================================


def _threat_risks(src: _Source) -> list[str]:
    out = ["### Threat-model risks", ""]
    found = src.doc(Kind.THREAT_MODEL)
    if found is None:
        return [*out, src.absent(Kind.THREAT_MODEL), ""]
    art, data = found
    cite = _cite(art.id)
    risks = _rows(data.get("risks"))
    for risk in risks[:_ITEMS]:
        mitigations = _rows(risk.get("mitigations"))
        covered = [m.get("requirement_id") for m in mitigations if m.get("requirement_id")]
        out.append(
            f"- {_id(risk.get('id'))} {_t(risk.get('title'), 160)} - "
            f"{_t(risk.get('category'), 32)}, likelihood {_t(risk.get('likelihood'), 16)}, "
            f"impact {_t(risk.get('impact'), 16)}, status {_t(risk.get('status'), 16)}; "
            f"{len(mitigations)} mitigation(s) covering {_join_ids(covered, 6)}{cite}"
        )
    out += _more(len(risks) - _ITEMS, "risks", cite)
    if not risks:
        out.append(f"- no risk was recorded{cite}")
    return [*out, ""]


def _design_risk(src: _Source) -> list[str]:
    out = ["### Design risk, trade-offs and rollback", ""]
    found = src.doc(Kind.DESIGN_DOC)
    if found is None:
        return [*out, src.absent(Kind.DESIGN_DOC), ""]
    art, data = found
    cite = _cite(art.id)
    risk = _map(data.get("risk"))
    changes = _rows(data.get("api_changes"))
    reasons = "; ".join(_t(r, 100) for r in _strs(risk.get("reasons"))[:5]) or "none stated"
    out.append(f"- Design risk {_id(risk.get('level'))}: {reasons}{cite}")
    for adr in _rows(data.get("adrs"))[:10]:
        rejected = ", ".join(_t(a.get("option"), 60) for a in _rows(adr.get("alternatives"))[:4])
        out.append(
            f"- Trade-off {_id(adr.get('id'))} {_t(adr.get('title'), 100)}: chosen over "
            f"{rejected or 'no recorded alternative'}; accepted consequence: "
            f"{_t(adr.get('consequences'), 200)}{cite}"
        )
    out.append(
        f"- {len(_rows(data.get('data_model_changes')))} data model change(s), "
        f"{len(changes)} API change(s) of which {sum(1 for c in changes if c.get('breaking'))} "
        f"breaking{cite}"
    )
    out.append(f"- Rollback plan: {_t(data.get('rollback_plan'))}{cite}")
    return [*out, ""]


def _policy_and_waivers(src: _Source) -> list[str]:
    out = ["### Policy findings and waivers", ""]
    grouped: dict[str, list[AuditEvent]] = {}
    for event in src.events:
        if event.type is _ET.POLICY_VIOLATION:
            grouped.setdefault(str(_map(event.data).get("rule_id", "unknown")), []).append(event)
    for rule in sorted(grouped)[:_ITEMS]:
        events = grouped[rule]
        severities = sorted({str(_map(e.data).get("severity", "?")) for e in events})
        enforcement = sorted({str(_map(e.data).get("enforcement", "?")) for e in events})
        nodes = sorted({e.node_id for e in events if e.node_id})
        out.append(
            f"- {_id(rule)} x{len(events)}: severity {_t(', '.join(severities), 40)}, "
            f"enforcement {_t(', '.join(enforcement), 40)}; nodes {_join_ids(nodes, 5)}"
            f"{_cite(*(f'audit:{e.seq}' for e in events[:3]))}"
        )
    if not grouped:
        out.append(f"- no policy violation was recorded{src.head_cite}")
    for waiver in src.waivers[:_ITEMS]:
        who = ", ".join(_t(name, 40) for name in waiver.approved_by) or "unknown"
        out.append(
            f"- Waiver {_id(waiver.id)}: {_id(waiver.rule_id)} on {_id(waiver.node_id)} "
            f"accepted by {who}: {_t(waiver.justification, 160)}"
            f"{_cite(waiver.decision_id, waiver.approval_id)}"
        )
    if not src.waivers:
        out.append(f"- no waiver was granted{src.head_cite}")
    return [*out, ""]


def _test_results(src: _Source) -> list[str]:
    out = ["### Test results", ""]
    found = src.doc(Kind.TEST_REPORT)
    if found is None:
        return [*out, src.absent(Kind.TEST_REPORT), ""]
    art, data = found
    cite = _cite(art.id)
    tests = _rows(data.get("tests"))
    passed = sum(1 for t in tests if t.get("outcome") == "passed")
    verdict = "passed" if data.get("passed") is True else "FAILED"
    out.append(
        f"- Suite {verdict}: {_t(data.get('total'), 12)} test(s), {passed} passed, "
        f"{_t(data.get('failed'), 12)} failed, {_t(data.get('errors'), 12)} errors, "
        f"{_t(data.get('skipped'), 12)} skipped{cite}"
    )
    bad = [t for t in tests if t.get("outcome") in ("failed", "error")]
    for test in bad[:10]:
        out.append(f"- {_t(test.get('outcome'), 12)}: {_t(test.get('nodeid'), 160)}{cite}")
    out += _more(len(bad) - 10, "failing tests", cite)
    baseline = _map(data.get("baseline"))
    out.append(f"- Baseline suite ran: {_yes(baseline.get('ran'))}{cite}")
    return [*out, ""]


def _fail_to_pass(src: _Source) -> list[str]:
    out = ["### Fail-to-pass", ""]
    found = src.doc(Kind.TEST_REPORT)
    if found is None:
        return [*out, src.absent(Kind.TEST_REPORT), ""]
    art, data = found
    f2p = _map(data.get("fail_to_pass"))
    tests = _strs(f2p.get("tests"))
    out.append(
        f"- Checked: {_yes(f2p.get('checked'))}; every authored test fails on the baseline and "
        f"passes on the integrated view: {_yes(f2p.get('ok'))}; {len(tests)} test(s) examined"
        f"{_cite(art.id)}"
    )
    return [*out, ""]


def _api_compat(src: _Source) -> list[str]:
    out = ["### API compatibility", ""]
    found = src.doc(Kind.TEST_REPORT)
    contract = src.doc(Kind.API_CONTRACT)
    if found is None:
        out.append(src.absent(Kind.TEST_REPORT))
    else:
        art, data = found
        compat = _map(data.get("api_compat"))
        breaking, additive = _strs(compat.get("breaking")), _strs(compat.get("additive"))
        out.append(
            f"- Compared with the baseline OpenAPI: checked {_yes(compat.get('checked'))}, "
            f"{len(breaking)} breaking, {len(additive)} additive change(s)"
            f"{_cite(art.id)}"
        )
        out += [f"- breaking: {_t(item, 160)}{_cite(art.id)}" for item in breaking[:10]]
    if contract is not None:
        art, data = contract
        out.append(
            f"- Declared contract: {len(_rows(data.get('paths')))} path(s), "
            f"breaking {_yes(data.get('breaking'))}{_cite(art.id)}"
        )
    return [*out, ""]


def _security(src: _Source) -> list[str]:
    out = ["### Security report", ""]
    found = src.doc(Kind.SECURITY_REPORT)
    if found is None:
        return [*out, src.absent(Kind.SECURITY_REPORT), ""]
    art, data = found
    cite = _cite(art.id)
    findings = _rows(data.get("findings"))
    out.append(
        f"- Scan {'clean' if data.get('passed') is True else 'NOT clean'}: "
        f"{_t(data.get('files_scanned'), 12)} file(s) scanned, {len(findings)} finding(s), "
        f"{len(_strs(data.get('rules_evaluated')))} rule(s) evaluated{cite}"
    )
    for finding in findings[:10]:
        out.append(
            f"- {_id(finding.get('rule_id'))} {_t(finding.get('severity'), 12)} at "
            f"{_t(finding.get('location'), 80)}: {_t(finding.get('message'), 160)}{cite}"
        )
    out += _more(len(findings) - 10, "findings", cite)
    return [*out, ""]


def _readiness(src: _Source) -> list[str]:
    out = ["### Readiness verdict", ""]
    found = src.doc(Kind.READINESS_REPORT)
    if found is None:
        out.append(src.absent(Kind.READINESS_REPORT))
    else:
        art, data = found
        cite = _cite(art.id)
        risk = _map(data.get("risk"))
        out.append(
            f"- Verdict {_id(data.get('verdict'))}, release risk {_id(risk.get('level'))}{cite}"
        )
        checks = _rows(data.get("checks"))
        failed = [c for c in checks if c.get("passed") is not True]
        out.append(
            f"- {len(checks) - len(failed)} of {len(checks)} readiness check(s) passed{cite}"
        )
        for check in failed[:10]:
            out.append(
                f"- check {_id(check.get('id'))} failed: {_t(check.get('title'), 100)} - "
                f"{_t(check.get('detail'), 160)}{cite}"
            )
    trace = src.doc(Kind.TRACEABILITY_MATRIX)
    if trace is not None:
        art, data = trace
        gaps = _strs(data.get("gaps"))
        out.append(
            f"- Traceability complete: {_yes(data.get('complete'))}, {len(gaps)} gap(s)"
            f"{_cite(art.id)}"
        )
        out += [f"- gap: {_t(gap, 160)}{_cite(art.id)}" for gap in gaps[:10]]
    verification = src.doc(Kind.RELEASE_VERIFICATION)
    if verification is not None:
        art, data = verification
        out.append(
            f"- Release verification: applied cleanly {_yes(data.get('applied_cleanly'))}, tests "
            f"passed {_yes(data.get('tests_passed'))}, health check {_yes(data.get('healthz_ok'))}"
            f"{_cite(art.id)}"
        )
    return [*out, ""]


def _approvals(src: _Source) -> list[str]:
    out = ["### Approvals, four-eyes and assurance", ""]
    scripted = src.scripted
    out.append(
        f"- Assurance {_id(ASSURANCE_SCRIPTED if scripted else 'HUMAN')}: "
        + (
            "approvals were scripted, not a human review"
            if scripted
            else "no scripted approval was recorded"
        )
        + f"; four-eyes is ATTESTED-ONLY (approver names are asserted, not authenticated)"
        f"{src.head_cite}"
    )
    for request in src.approvals[:_ITEMS]:
        votes = "; ".join(
            f"{_t(v.approver, 40)} {'approved' if v.approved else 'rejected'} via "
            f"{_t(v.channel, 16)}" + (f" ({_t(v.comment, 80)})" if v.comment else "")
            for v in request.votes
        )
        distinct = len({v.approver for v in request.votes if v.approved})
        waives = f", waives {len(request.waives)} finding(s)" if request.waives else ""
        out.append(
            f"- {_id(request.id)} on {_id(request.node_id)} ({_t(request.purpose.value, 16)}, risk "
            f"{_id(request.risk.name)}): {_id(request.status.value)}, {distinct} distinct "
            f"approver(s) of {request.required_approvals} required{waives}; "
            f"{votes or 'no vote'}{src.approval_cite(request)}"
        )
    out += _more(len(src.approvals) - _ITEMS, "approval requests", src.head_cite)
    if not src.approvals:
        out.append(f"- no approval was requested{src.head_cite}")
    return [*out, ""]


def _risks_section(src: _Source) -> list[str]:
    return [
        "## 3. Risks, trade-offs and validation",
        "",
        *_threat_risks(src),
        *_design_risk(src),
        *_policy_and_waivers(src),
        *_test_results(src),
        *_fail_to_pass(src),
        *_api_compat(src),
        *_security(src),
        *_readiness(src),
        *_approvals(src),
    ]


# =============================================================================
# 4. Assumptions
# =============================================================================


def _requirement_assumptions(src: _Source) -> list[str]:
    out = ["### Requirement assumptions", ""]
    found = src.doc(Kind.REQUIREMENTS)
    if found is None:
        return [*out, src.absent(Kind.REQUIREMENTS), ""]
    art, data = found
    assumptions = _rows(data.get("assumptions"))
    for item in assumptions[:_ITEMS]:
        asm = str(item.get("id", ""))
        ids = src.assumption_decisions(asm)
        out.append(f"- {_id(asm)} {_t(item.get('text'))}{_cite(art.id, *ids)}")
    out += _more(len(assumptions) - _ITEMS, "assumptions", _cite(art.id))
    if not assumptions:
        out.append(f"- no assumption was recorded{_cite(art.id)}")
    return [*out, ""]


def _deferred_scope(src: _Source) -> list[str]:
    out = ["### Deferred scope", ""]
    found = src.doc(Kind.REQUIREMENTS)
    if found is None:
        return [*out, src.absent(Kind.REQUIREMENTS), ""]
    art, data = found
    held = [
        r for r in _rows(data.get("requirements")) if r.get("status") in ("deferred", "descoped")
    ]
    for req in held[:_ITEMS]:
        ids = src.scope_decisions(str(req.get("id", "")))
        out.append(
            f"- {_id(req.get('id'))} is {_t(req.get('status'), 16)}: {_t(req.get('text'), 160)}"
            f"{_cite(art.id, *ids, *src.clarification_parents(ids))}"
        )
    out += _more(len(held) - _ITEMS, "requirements", _cite(art.id))
    if not held:
        out.append(f"- no requirement was deferred or descoped{_cite(art.id)}")
    scope = [
        d for d in src.decisions if d.kind is DecisionKind.SCOPE and d.actor.type is ActorType.HUMAN
    ]
    for d in scope[:10]:
        out.append(f"- Human scope decision {_id(d.id)}: {_t(d.summary, 160)}")
    return [*out, ""]


def _clarifications(src: _Source) -> list[str]:
    out = ["### Clarifications answered", ""]
    art = src.artifact(Kind.CLARIFICATION_ANSWERS)
    if art is None:
        return [
            *out,
            f"- no clarification was answered: none was needed or asked{src.head_cite}",
            "",
        ]
    found = src.doc(Kind.CLARIFICATION_ANSWERS)
    if found is None:
        return [*out, src.absent(Kind.CLARIFICATION_ANSWERS), ""]
    answers = _map(found[1].get("answers"))
    for qid in sorted(answers)[:_ITEMS]:
        entry = _map(answers[qid])
        ids = src.clarification_decisions(qid)
        out.append(
            f"- {_id(qid)} answered by {_t(entry.get('by'), 40)} in round "
            f"{_t(entry.get('round'), 8)}: {_t(entry.get('answer'), 160)}{_cite(art.id, *ids)}"
        )
    out += _more(len(answers) - _ITEMS, "answers", _cite(art.id))
    if not answers:
        out.append(f"- the answers artifact is empty{_cite(art.id)}")
    return [*out, ""]


def _assumptions_section(src: _Source) -> list[str]:
    return [
        "## 4. Assumptions",
        "",
        *_requirement_assumptions(src),
        *_deferred_scope(src),
        *_clarifications(src),
    ]


# =============================================================================
# 5. Limitations (computed from the facts)
# =============================================================================


def _unsupported(src: _Source) -> list[str]:
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for kind in (Kind.REQUIREMENTS, Kind.TASK_PLAN):
        found = src.doc(kind)
        if found is None:
            continue
        for item in _rows(found[1].get("unsupported")):
            key = (str(item.get("requirement_id")), str(item.get("reason")))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                f"- Unsupported requirement {_id(item.get('requirement_id'))}: "
                f"{_t(item.get('reason'), 160)} - the reference agents cannot deliver it"
                f"{_cite(found[0].id)}"
            )
    return out[:_ITEMS]


def _deferred(src: _Source) -> list[str]:
    found = src.doc(Kind.REQUIREMENTS)
    if found is None:
        return []
    art, data = found
    out = []
    for req in _rows(data.get("requirements")):
        if req.get("status") in ("deferred", "descoped"):
            ids = src.scope_decisions(str(req.get("id", "")))
            out.append(
                f"- Not delivered by this run: {_id(req.get('id'))} was "
                f"{_t(req.get('status'), 16)} by a scope decision{_cite(art.id, *ids)}"
            )
    return out[:_ITEMS]


def _scripted(src: _Source) -> list[str]:
    if not src.scripted:
        return []
    requests = src.scripted_requests
    cites: list[str] = []
    for request in requests[:3]:
        cites += src.node_decisions(request.node_id, _APPROVAL_DECISIONS)[:1]
    if not cites:
        cites = [f"audit:{seq}" for seq in src.scripted_vote_seqs[:3]] or [f"audit:{src.head_seq}"]
    return [
        f"- Approvals were scripted (assurance {md_ident(ASSURANCE_SCRIPTED)}), "
        f"not a human review: scripted votes on {_join_ids([r.id for r in requests])}"
        f"{_cite(*cites)}"
    ]


def _faults(src: _Source) -> list[str]:
    raw = src.params.get("faults")
    out: list[str] = []
    for fault in _rows(raw)[:20]:
        pattern = str(fault.get("node", ""))[:80]
        nodes = {n for n in (*src.statuses, *(d.node_id or "" for d in src.decisions)) if n}
        hit = sorted(n for n in nodes if fnmatch.fnmatchcase(n, pattern))
        ids = [i for node in hit for i in src.node_decisions(node, _FAULT_DECISIONS)[:2]][:4]
        if not ids:
            ids = [f"audit:{s}" for node in hit for s in src.failures.get(node, [])[:1]][:3]
        out.append(
            f"- Injected fault {_id(fault.get('fault'))} on nodes matching {_id(pattern)} "
            f"(up to {_t(fault.get('attempts'), 8)} attempt(s)); affected: {_join_ids(hit, 5)}"
            f"{_cite(*ids) if ids else src.head_cite}"
        )
    return out


def _node_limits(src: _Source) -> list[str]:
    out: list[str] = []
    groups = (
        (NodeStatus.SKIPPED, "Skipped node"),
        (NodeStatus.FAILED, "Failed node"),
        (NodeStatus.BLOCKED, "Blocked node"),
        (NodeStatus.CANCELLED, "Cancelled node"),
        (NodeStatus.ROLLED_BACK, "Rolled-back node"),
    )
    for status, label in groups:
        nodes = sorted(n for n, s in src.statuses.items() if s is status)
        for node in nodes[:20]:
            reason = src.notes.get(node, (0, ""))[1]
            why = f": {_t(reason, _SHORT)}" if reason else ""
            out.append(f"- {label} {_id(node)}{why}{src.node_cite(node)}")
        out += _more(len(nodes) - 20, label.lower() + "s", src.head_cite)
    return out


def _safe_stop(src: _Source) -> list[str]:
    stops = [e for e in src.events if e.type is _ET.SAFE_STOP_TRIGGERED]
    ids = src.decision_ids(lambda d: d.kind is DecisionKind.SAFE_STOP)
    if not stops and src.status is not RunStatus.SAFE_STOPPED and not ids:
        return []
    out = []
    for event in stops[:5]:
        data = _map(event.data)
        out.append(
            f"- Safe-stop triggered by {_id(data.get('trigger'))}: {_t(data.get('reason'), 160)}"
            f"{_cite(*ids, f'audit:{event.seq}')}"
        )
    if not stops:
        out.append(
            f"- The run was safe-stopped: {_t(src.reason, 160)}"
            f"{_cite(*ids) if ids else src.head_cite}"
        )
    return out


def _skipped_checks(src: _Source) -> list[str]:
    found = src.doc(Kind.TEST_REPORT)
    if found is None:
        return []
    art, data = found
    missing = []
    if _map(data.get("baseline")).get("ran") is not True:
        missing.append("the baseline suite did not run")
    if _map(data.get("fail_to_pass")).get("checked") is not True:
        missing.append("fail-to-pass was not checked")
    if _map(data.get("api_compat")).get("checked") is not True:
        missing.append("API compatibility was not compared with the baseline")
    if src.params.get("perf_budget_ms") and _map(data.get("perf")).get("checked") is not True:
        missing.append("the performance budget was not checked")
    return [f"- Check not performed: {text}{_cite(art.id)}" for text in missing]


def _manual_steps(src: _Source) -> list[str]:
    patch = src.artifact(Kind.RELEASE_PATCH)
    bundle = src.doc(Kind.RELEASE_BUNDLE)
    out: list[str] = []
    if patch is None:
        out.append(
            "- Manual steps: no patch was produced, so there is nothing to review or apply"
            f"{src.head_cite}"
        )
    elif bundle is not None and bundle[1].get("mode") == "applied":
        out.append(
            "- Manual steps: the patch was applied to the target tree by the promote node; review "
            f"the working tree and commit it yourself{_cite(patch.id, bundle[0].id)}"
        )
    else:
        out.append(
            "- Manual steps left: review the patch and apply it to your tree yourself "
            "(this run did not modify any target tree)"
            f"{_cite(patch.id, bundle[0].id if bundle else None)}"
        )
    found = src.doc(Kind.READINESS_REPORT)
    if found is not None:
        for item in _strs(found[1].get("open_items"))[:10]:
            out.append(f"- Open item: {_t(item, 160)}{_cite(found[0].id)}")
    return out


def _unreadable(src: _Source) -> list[str]:
    return [
        f"- Artifact {_id(art_id)} could not be read ({_t(error, 40)}): the sections that "
        f"depend on it are incomplete{_cite(art_id)}"
        for art_id, error in dict.fromkeys(src.unreadable)
    ][:_ITEMS]


def _identity_and_agents(src: _Source) -> list[str]:
    agents = src.agents
    vote_cite = _cite(*(f"audit:{seq}" for seq in src.vote_seqs[:2])) or src.head_cite
    started = _cite(f"audit:{src.first_started}")
    out = [
        "- Approver identity is attested, not authenticated (four-eyes: ATTESTED-ONLY): "
        f"names are asserted by the caller{vote_cite}"
    ]
    if not agents:
        out.append(f"- Agents used: unknown, no node start was recorded{src.head_cite}")
    elif any("llm" in agent.lower() for agent in agents):
        out.append(
            f"- LLM-backed agent(s) took part ({_join_ids(agents, 6)}): their output was "
            f"treated as untrusted proposals{started}"
        )
    else:
        out.append(
            f"- Work was done by deterministic reference agents ({_join_ids(agents, 6)}): "
            "rule-based heuristics with no LLM, limited to the supported capabilities and "
            f"reporting the rest as unsupported{started}"
        )
    return out


def _final(src: _Source) -> list[str]:
    reason = f" - {_t(src.reason, 160)}" if src.reason else ""
    return [
        f"- Final status {_id(src.status.value)}{reason}; audit head seq {src.head_seq}, hash "
        f"`{src.head_hash or '-'}`{src.head_cite}"
    ]


def _limitations_section(src: _Source) -> list[str]:
    bullets = [
        *_unsupported(src),
        *_deferred(src),
        *_scripted(src),
        *_faults(src),
        *_node_limits(src),
        *_safe_stop(src),
        *_skipped_checks(src),
        *_manual_steps(src),
        *_unreadable(src),
        *_identity_and_agents(src),
        *_final(src),
    ]
    return ["## 5. Limitations", "", *bullets, ""]


# =============================================================================
# public API
# =============================================================================


def render_engineering_summary(ctx: FinalizerContext) -> str:
    """The Markdown of the engineering summary (a pure function of what ``ctx.view`` holds)."""
    src = _Source(ctx)
    sections: Sequence[list[str]] = (
        _header(src),
        _plan_section(src),
        _artifacts_section(src),
        _risks_section(src),
        _assumptions_section(src),
        _limitations_section(src),
    )
    return "\n".join(line for section in sections for line in section).rstrip("\n") + "\n"


def engineering_summary_finalizer(ctx: FinalizerContext) -> None:
    """Publish the ``engineering_summary`` artifact for the terminal state in ``ctx``."""
    ctx.publish_artifact(
        ArtifactDraft(
            kind=Kind.ENGINEERING_SUMMARY,
            content=render_engineering_summary(ctx),
            media_type="text/markdown",
        )
    )
