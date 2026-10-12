"""Approvals: the human checkpoints of a run (engine-semantics 10.1, 10.4 and 10.7).

``ApprovalManager`` owns the ``ApprovalRequest`` records in ``RunState.approvals`` and applies the
state machine ``PENDING -> APPROVED | REJECTED | INVALIDATED | CANCELLED``:

* ``APPROVED`` needs ``required_approvals`` **distinct** approvers (never below the hard floors
  LOW 1, MEDIUM 1, HIGH 2); **any single rejection vetoes**; a vote can never be changed; a
  request that is not ``PENDING`` accepts no votes; a vote is applied only while its node waits for
  exactly that request (when the node is known) and, if the caller supplies
  ``current_fingerprint``, only while the request still matches it.
* Identity is **asserted, not authenticated**. ``normalize_approver`` (NFKC + casefold + the ASCII
  pattern ``^[a-z0-9][a-z0-9._@-]{1,63}$``) makes width and case variants of one name collapse and
  homoglyphs fail; registered agent ids and ``system``/``engine``/``policy``/``scripted``/
  ``agent``/``unknown`` (and ``agent:``/``system:``/``policy:`` prefixes) are denied; the run
  creator cannot approve MEDIUM+ requests; ``EngineConfig.approver_allowlist`` is honoured; in
  ``identity_mode='strict'`` the approvers of a HIGH request must also have distinct
  operating-system users. A rejection is the safe direction: it is exempt from the creator rule
  and from the CLI confirmation code.
* A CLI approval must carry ``confirm`` = the first 8 hex characters of the request's evidence
  hash (shown by ``status``); ``interactive`` and ``scripted`` votes are exempt. Scripted votes
  need ``EngineConfig.allow_scripted`` and downgrade ``RunState.assurance`` to ``DEMO_SCRIPTED``.

Every refusal is audited first (``APPROVAL_REFUSED{approval_id, reason}``) and then raised as
``ApprovalRefused`` (an ``ApprovalError`` carrying ``reason``). Reasons: ``self_approval``,
``duplicate``, ``agent_identity``, ``unknown_approver``, ``stale_fingerprint``, ``terminal`` (the
request can no longer receive votes: decided, closed, or its node no longer waits for it),
``insufficient_role`` (channel not allowed, strict-mode identity missing), plus ``unknown_request``
and ``confirm_mismatch``. State changes are write-ahead: the audit event is appended before the
state is mutated, so a failing audit log can never leave an unaudited approval behind.

``compose_summary`` builds the human-facing summary from structured facts only (risk reasons, rule
ids/severities/locations, diff statistics, artifact refs with hash prefixes, fingerprint prefixes,
requester, assurance); agent prose is never copied into it.

Providers are non-blocking: ``poll`` hands the scheduler the votes collected so far and never waits
for a human. ``QueueApprover`` never votes (votes arrive through ``approve``/``reject``),
``ScriptedApprover`` replays pre-declared decisions (demos/CI) and ``InteractiveApprover`` asks a
human on its own daemon thread.

Limitations: identities are not authenticated (``os_user`` is only as honest as its caller); a
scripted decision answers the first matching request only; an interactive vote that the manager
refuses is not asked for again in the same session (use ``approve``), and a blocked ``input_fn``
cannot be interrupted (the thread is a daemon).
"""

from __future__ import annotations

import queue
import re
import threading
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from orchestrator.core.errors import ApprovalError
from orchestrator.core.sanitize import sanitize_block, sanitize_for_terminal
from orchestrator.core.state import RunState
from orchestrator.core.types import (
    Actor,
    ApprovalPosition,
    ApprovalPurpose,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalVote,
    ArtifactRef,
    AuditSink,
    Clock,
    EventType,
    Finding,
    NodeSpec,
    NodeStatus,
    RiskLevel,
    Severity,
)
from orchestrator.engine.config import MIN_APPROVALS, EngineConfig

ASSURANCE_SCRIPTED = "DEMO_SCRIPTED"
CHANNELS = frozenset({"cli", "interactive", "scripted"})
RESERVED_NAMES = frozenset({"system", "engine", "policy", "scripted", "agent", "unknown"})
RESERVED_PREFIXES = ("agent:", "system:", "policy:")
CONFIRM_CHARS = 8
MAX_NAME_INPUT = 256
MAX_COMMENT_CHARS = 500
MAX_NOTE_CHARS = 500
MAX_REASON_CHARS = 300
MAX_REASONS = 50
MAX_SUMMARY_CHARS = 6000
SHOWN_REASONS = 10
SHOWN_FINDINGS = 20
SHOWN_EVIDENCE = 15
SHOWN_STATS = 20
MAX_PROMPT_MISTAKES = 5

_APPROVER_RE = re.compile(r"[a-z0-9][a-z0-9._@-]{1,63}")
_STAT_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,30}")
_ENGINE = Actor.system("engine")


class ApprovalRefused(ApprovalError):
    """An approval operation was refused; ``reason`` is the ``APPROVAL_REFUSED`` reason code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _fold(name: str) -> str:
    """NFKC + casefold (+ NFKC again: case folding can yield compatibility characters)."""
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", name).casefold())


def _q(value: object) -> str:
    return repr(sanitize_for_terminal(value, 40))


def normalize_approver(name: str, *, reserved: frozenset[str]) -> str:
    """Canonical approver identity: NFKC, casefold, ASCII ``^[a-z0-9][a-z0-9._@-]{1,63}$``.

    Raises ``ApprovalRefused`` (an ``ApprovalError``) with reason ``agent_identity`` for reserved
    names (``reserved`` plus the built-in system names and ``agent:``/``system:``/``policy:``
    prefixes) and ``unknown_approver`` for anything that is not a well-formed name. Non-ASCII
    homoglyphs (Cyrillic ``a`` ...) never normalise to ASCII, so they are refused, while width and
    case variants of one name collapse onto it.
    """
    if not isinstance(name, str):
        raise ApprovalRefused("unknown_approver", "approver name must be a string")
    if len(name) > MAX_NAME_INPUT:
        raise ApprovalRefused("unknown_approver", "approver name is too long")
    folded = _fold(name)
    if (
        folded.startswith(RESERVED_PREFIXES)
        or folded in RESERVED_NAMES
        or folded in {_fold(r) for r in reserved}
    ):
        raise ApprovalRefused(
            "agent_identity", f"approver name {_q(folded)} is reserved (agent or system identity)"
        )
    if not _APPROVER_RE.fullmatch(folded):
        raise ApprovalRefused(
            "unknown_approver",
            "approver names must match ^[a-z0-9][a-z0-9._@-]{1,63}$ after NFKC and casefold",
        )
    return folded


# --------------------------------------------------------------------------- summary


def _text(value: object, limit: int) -> str:
    return sanitize_for_terminal(value, limit)


def _more(total: int, shown: int) -> list[str]:
    return [f"- ... and {total - shown} more"] if total > shown else []


def compose_summary(
    *,
    node: NodeSpec | str,
    risk: RiskLevel,
    reasons: Sequence[str],
    findings: Sequence[Finding],
    diff_stats: Mapping[str, Any] | None,
    evidence: Sequence[ArtifactRef],
    assurance: str,
    requested_by: str,
    node_fingerprint: str = "",
    evidence_hash: str = "",
) -> str:
    """Engine-composed approval summary: structured facts only, sanitised, deterministic.

    Finding messages, evidence snippets, node titles and any agent prose are deliberately absent:
    a reviewer sees rule ids, severities, locations, counts and hashes, which an attacker-influenced
    artifact cannot dress up as instructions. Only ``int`` entries of ``diff_stats`` are shown.
    The fixed-size lines come first and the variable lists are capped per section, so even a
    hostile input that hits the overall size cap only loses the tail of a list.
    """
    if isinstance(node, str):
        lines = [f"node: {_text(node, 60)}"]
    else:
        stage = node.stage.value
        lines = [f"node: {_text(node.id, 60)} (stage {stage}, agent {_text(node.agent, 60)})"]
    lines.append(f"risk: {RiskLevel(risk).name}")
    who = f"requested by: {_text(requested_by, 100)}; assurance: {_text(assurance, 20)}"
    if assurance == ASSURANCE_SCRIPTED:
        who += " (approvals were scripted, not a human review)"
    lines.append(who)
    if node_fingerprint:
        lines.append(f"inputs fingerprint: {_text(node_fingerprint, 64)[:12]}")
    if evidence_hash:
        lines.append(f"evidence hash: {_text(evidence_hash, 64)[:CONFIRM_CHARS]}")
    reason_list = [reasons] if isinstance(reasons, str) else list(reasons)
    if reason_list:
        lines.append("risk reasons:")
        lines.extend(f"- {_text(r, 160)}" for r in reason_list[:SHOWN_REASONS])
        lines.extend(_more(len(reason_list), SHOWN_REASONS))
    ordered = sorted(findings, key=lambda f: (-int(f.severity), f.rule_id, f.location or ""))
    if ordered:
        lines.append(f"policy findings ({len(ordered)}):")
        for finding in ordered[:SHOWN_FINDINGS]:
            where = f" at {_text(finding.location, 100)}" if finding.location else ""
            lines.append(f"- {_text(finding.rule_id, 24)} {finding.severity.name}{where}")
        lines.extend(_more(len(ordered), SHOWN_FINDINGS))
    counts = [
        (key, value)
        for key, value in (diff_stats or {}).items()
        if isinstance(key, str)
        and _STAT_KEY_RE.fullmatch(key)
        and isinstance(value, int)
        and not isinstance(value, bool)
    ]
    if counts:
        shown = sorted(counts, key=lambda kv: kv[0])[:SHOWN_STATS]
        lines.append("changes: " + " ".join(f"{key}={value}" for key, value in shown))
    if evidence:
        lines.append(f"evidence artifacts ({len(evidence)}):")
        for ref in evidence[:SHOWN_EVIDENCE]:
            digest = _text(ref.content_hash, 64)[:12]
            lines.append(f"- {_text(ref.artifact_id, 80)} sha256:{digest}")
        lines.extend(_more(len(evidence), SHOWN_EVIDENCE))
    return sanitize_block("\n".join(lines), MAX_SUMMARY_CHARS)


# --------------------------------------------------------------------------- manager


class ApprovalManager:
    """Raises, votes on, invalidates and cancels the approval requests of one run.

    The scheduler thread is the only intended writer; every mutation nevertheless takes a lock so
    that concurrent callers can never double-count a vote.
    """

    def __init__(
        self,
        state: RunState,
        config: EngineConfig,
        clock: Clock,
        audit: AuditSink,
        reserved_names: frozenset[str],
        created_by: str,
    ) -> None:
        self.state = state
        self.config = config
        self.clock = clock
        self._audit = audit
        self._reserved = frozenset(reserved_names)
        self._created_by = _fold(created_by) if isinstance(created_by, str) else ""
        self._requested_by = _text(created_by, 100)
        allowlist = config.approver_allowlist
        self._allowed = None if allowlist is None else frozenset(_fold(n) for n in allowlist)
        self._lock = threading.RLock()

    # -- queries -------------------------------------------------------------------
    def required_for(self, risk: RiskLevel) -> int:
        """Distinct approvers a request of ``risk`` needs (never below the hard floors)."""
        level = RiskLevel(risk)
        return max(MIN_APPROVALS[level], int(self.config.approvals_by_risk[level]))

    def is_satisfied(self, request_id: str) -> bool:
        """``APPROVED`` *and* backed by enough distinct approving votes (state is only a cache)."""
        with self._lock:
            req = self.state.approvals.get(request_id)
            return (
                req is not None
                and req.status == ApprovalStatus.APPROVED
                and len(self._approvers(req)) >= self._needed(req)
            )

    def is_rejected(self, request_id: str) -> bool:
        with self._lock:
            req = self.state.approvals.get(request_id)
            return req is not None and req.status == ApprovalStatus.REJECTED

    def pending(self) -> list[ApprovalRequest]:
        with self._lock:
            return self._ordered(lambda r: r.status == ApprovalStatus.PENDING)

    def for_node(self, node_id: str) -> list[ApprovalRequest]:
        with self._lock:
            return self._ordered(lambda r: r.node_id == node_id)

    def valid_for(self, node_id: str, fingerprint: str) -> ApprovalRequest | None:
        """The latest satisfied ``APPROVED`` request of the node raised for ``fingerprint``."""
        with self._lock:
            valid = [
                r
                for r in self._ordered(lambda r: r.node_id == node_id)
                if r.node_fingerprint == fingerprint and self.is_satisfied(r.id)
            ]
            return valid[-1] if valid else None

    @staticmethod
    def compose_summary(**facts: Any) -> str:
        """``compose_summary`` as a method (same keywords)."""
        return compose_summary(**facts)

    # -- raising -------------------------------------------------------------------
    def request(
        self,
        *,
        node_id: str,
        position: ApprovalPosition,
        purpose: ApprovalPurpose,
        risk: RiskLevel,
        reasons: Sequence[str],
        summary: str,
        evidence: Sequence[ArtifactRef],
        node_fingerprint: str,
        evidence_hash: str,
        attempt: int,
        generation: int,
        waives: Sequence[Finding] = (),
        staged: Sequence[ArtifactRef] = (),
        agent_note: str = "",
    ) -> ApprovalRequest:
        """Open a ``PENDING`` request (ids ``A-0001`` ...) and audit ``APPROVAL_REQUESTED``."""
        level = RiskLevel(risk)
        waived = tuple(waives)
        if any(f.severity > Severity.MEDIUM for f in waived):
            raise ApprovalError("HIGH and CRITICAL findings can never be waived")
        if not isinstance(node_fingerprint, str) or not node_fingerprint:
            raise ApprovalError("an approval request needs the node fingerprint it covers")
        if not isinstance(evidence_hash, str):
            raise ApprovalError("an approval request needs the evidence hash it covers")
        reason_list = [reasons] if isinstance(reasons, str) else list(reasons)
        with self._lock:
            rid = self._next_id()
            req = ApprovalRequest(
                id=rid,
                run_id=self.state.run_id,
                node_id=node_id,
                position=ApprovalPosition(position),
                purpose=ApprovalPurpose(purpose),
                risk=level,
                reasons=tuple(_text(r, MAX_REASON_CHARS) for r in reason_list[:MAX_REASONS]),
                summary=sanitize_block(summary, MAX_SUMMARY_CHARS),
                evidence=tuple(evidence),
                node_fingerprint=node_fingerprint,
                evidence_hash=evidence_hash,
                required_approvals=self.required_for(level),
                attempt=attempt,
                generation=generation,
                created_at=self.clock.now(),
                requested_by=self._requested_by,
                waives=waived,
                staged=tuple(staged),
                agent_note=_text(agent_note, MAX_NOTE_CHARS),
            )
            self._emit(
                EventType.APPROVAL_REQUESTED,
                req,
                None,
                approval_id=rid,
                purpose=req.purpose.value,
                risk=level.name,
                required=req.required_approvals,
                node_fingerprint=node_fingerprint,
                position=req.position.value,
                evidence_hash=evidence_hash,
            )
            self.state.approvals[rid] = req
            return req

    # -- voting --------------------------------------------------------------------
    def vote(
        self,
        request_id: str,
        approver: str,
        approved: bool,
        comment: str = "",
        *,
        channel: str = "cli",
        os_user: str = "",
        confirm: str = "",
        current_fingerprint: str | None = None,
    ) -> ApprovalRequest:
        """Record one vote and complete the request when the state machine says so.

        Every refusal is audited as ``APPROVAL_REFUSED`` first and then raised
        (``ApprovalRefused``, an ``ApprovalError``); a refused vote changes nothing.
        ``current_fingerprint`` is the node's freshly computed fingerprint: a request raised for a
        different one is refused as stale.
        """
        if not isinstance(approved, bool):
            raise TypeError("approved must be a bool")
        with self._lock:
            req = self.state.approvals.get(request_id) if isinstance(request_id, str) else None
            try:
                if req is None:
                    raise ApprovalRefused(
                        "unknown_request", f"no approval request {_q(request_id)}"
                    )
                name, user = self._admit(
                    req, approver, approved, channel, os_user, confirm, current_fingerprint
                )
            except ApprovalRefused as refusal:
                self._emit(
                    EventType.APPROVAL_REFUSED,
                    req,
                    None,
                    approval_id=req.id if req is not None else _text(request_id, 40),
                    reason=refusal.reason,
                    approver=_text(approver, 64) if isinstance(approver, str) else "<non-text>",
                    channel=_text(channel, 16) if isinstance(channel, str) else "<non-text>",
                )
                raise
            return self._apply(req, name, approved, comment, channel, user)

    def _admit(
        self,
        req: ApprovalRequest,
        approver: str,
        approved: bool,
        channel: str,
        os_user: str,
        confirm: str,
        current_fingerprint: str | None,
    ) -> tuple[str, str]:
        """Every vote check, most fundamental first. Returns ``(approver, os_user)`` or raises."""
        if req.status != ApprovalStatus.PENDING:
            raise ApprovalRefused(
                "terminal", f"approval {req.id} is {req.status.value} and accepts no more votes"
            )
        if any(not v.approved for v in req.votes):
            raise ApprovalRefused("terminal", f"approval {req.id} already carries a rejection")
        node = self.state.nodes.get(req.node_id)
        if node is not None and (
            node.status != NodeStatus.AWAITING_APPROVAL or node.pending_approval_id != req.id
        ):
            raise ApprovalRefused(
                "terminal", f"node {req.node_id} is not waiting for approval {req.id}"
            )
        if current_fingerprint is not None and current_fingerprint != req.node_fingerprint:
            raise ApprovalRefused(
                "stale_fingerprint", f"approval {req.id} was raised for different node inputs"
            )
        if not isinstance(channel, str) or channel not in CHANNELS:
            raise ApprovalRefused("insufficient_role", f"unknown approval channel {_q(channel)}")
        if channel == "scripted" and not self.config.allow_scripted:
            raise ApprovalRefused("insufficient_role", "scripted approvals are disabled")
        name = normalize_approver(approver, reserved=self._reserved)
        if self._allowed is not None and name not in self._allowed:
            raise ApprovalRefused("unknown_approver", f"{_q(name)} is not an allowed approver")
        if approved and req.risk >= RiskLevel.MEDIUM and name == self._created_by:
            raise ApprovalRefused(
                "self_approval",
                f"{_q(name)} created this run and cannot approve {req.risk.name} work",
            )
        if name in {_fold(v.approver) for v in req.votes}:
            raise ApprovalRefused(
                "duplicate", f"{_q(name)} already voted on {req.id}; a vote cannot be changed"
            )
        user = _text(os_user, 64) if isinstance(os_user, str) else ""
        if approved:
            self._check_oversight(req, channel, user, confirm)
        return name, user

    def _check_oversight(
        self, req: ApprovalRequest, channel: str, os_user: str, confirm: str
    ) -> None:
        """Checks that only apply to approvals (a veto is always allowed)."""
        if self.config.identity_mode == "strict" and req.risk >= RiskLevel.HIGH:
            if not os_user:
                raise ApprovalRefused(
                    "insufficient_role",
                    "identity_mode=strict needs the operating-system user of every HIGH approver",
                )
            if _fold(os_user) in {_fold(v.os_user) for v in req.votes if v.approved and v.os_user}:
                raise ApprovalRefused(
                    "duplicate", "identity_mode=strict: that operating-system user already approved"
                )
        if channel == "cli":
            expected = req.evidence_hash[:CONFIRM_CHARS].lower()
            given = confirm.strip().lower() if isinstance(confirm, str) else ""
            if len(expected) < CONFIRM_CHARS or given != expected:
                raise ApprovalRefused(
                    "confirm_mismatch",
                    "a CLI approval needs --confirm <first 8 characters of the evidence hash "
                    "shown by status>",
                )

    def _apply(
        self,
        req: ApprovalRequest,
        name: str,
        approved: bool,
        comment: str,
        channel: str,
        os_user: str,
    ) -> ApprovalRequest:
        now = self.clock.now()
        note = _text(comment, MAX_COMMENT_CHARS) if isinstance(comment, str) else ""
        ballot = ApprovalVote(name, approved, note, now, channel, os_user, req.evidence_hash)
        if channel == "scripted":
            self.state.assurance = ASSURANCE_SCRIPTED
        # Write-ahead and atomic: every audit event is appended first, and only then are the vote
        # and the resulting status applied together, so a failing audit log can never leave a
        # recorded rejection on a request that is still PENDING (the veto must always hold).
        self._emit(
            EventType.APPROVAL_VOTE,
            req,
            Actor.human(name),
            approval_id=req.id,
            approver=name,
            approved=approved,
            channel=channel,
            evidence_hash=req.evidence_hash,
            os_user=os_user,
        )
        decided: ApprovalStatus | None = None
        if not approved:
            self._emit(
                EventType.APPROVAL_REJECTED,
                req,
                None,
                approval_id=req.id,
                approver=name,
                evidence_hash=req.evidence_hash,
            )
            decided = ApprovalStatus.REJECTED
        else:
            approvers = self._approvers(req) | {name}
            needed = self._needed(req)
            if len(approvers) >= needed:
                self._emit(
                    EventType.APPROVAL_GRANTED,
                    req,
                    None,
                    approval_id=req.id,
                    approvers=sorted(approvers),
                    required=needed,
                    evidence_hash=req.evidence_hash,
                )
                decided = ApprovalStatus.APPROVED
        req.votes.append(ballot)
        if decided is not None:
            req.status = decided
            req.decided_at = now
        return req

    # -- closing -------------------------------------------------------------------
    def invalidate(self, node_id: str, reason: str) -> list[ApprovalRequest]:
        """The node's inputs changed: its ``PENDING`` and ``APPROVED`` requests become
        ``INVALIDATED`` (a new request is raised on re-run); rejected/closed ones stay as record."""
        return self._close(
            node_id,
            reason,
            frozenset({ApprovalStatus.PENDING, ApprovalStatus.APPROVED}),
            ApprovalStatus.INVALIDATED,
            EventType.APPROVAL_INVALIDATED,
        )

    def cancel(self, node_id: str, reason: str) -> list[ApprovalRequest]:
        """The node was reset, removed or cancelled (or the run safe-stopped): its ``PENDING``
        requests become ``CANCELLED`` (the caller quarantines their staged artifacts)."""
        return self._close(
            node_id,
            reason,
            frozenset({ApprovalStatus.PENDING}),
            ApprovalStatus.CANCELLED,
            EventType.APPROVAL_CANCELLED,
        )

    def _close(
        self,
        node_id: str,
        reason: str,
        from_states: frozenset[ApprovalStatus],
        to_state: ApprovalStatus,
        event: EventType,
    ) -> list[ApprovalRequest]:
        text = _text(reason, 200)
        with self._lock:
            closed = []
            for req in self._ordered(lambda r: r.node_id == node_id):
                if req.status not in from_states:
                    continue
                self._emit(
                    event, req, None, approval_id=req.id, reason=text, previous=req.status.value
                )
                req.status = to_state
                if req.decided_at is None:
                    req.decided_at = self.clock.now()
                closed.append(req)
            return closed

    # -- internals -----------------------------------------------------------------
    def _ordered(self, keep: Callable[[ApprovalRequest], bool]) -> list[ApprovalRequest]:
        chosen = [r for r in self.state.approvals.values() if keep(r)]
        return sorted(chosen, key=lambda r: (len(r.id), r.id))

    def _next_id(self) -> str:
        """``A-0001`` ...; never reuses an id that exists (a lowered counter cannot overwrite)."""
        while True:
            self.state.approval_counter += 1
            rid = f"A-{self.state.approval_counter:04d}"
            if rid not in self.state.approvals:
                return rid

    @staticmethod
    def _approvers(req: ApprovalRequest) -> set[str]:
        return {_fold(v.approver) for v in req.votes if v.approved}

    @staticmethod
    def _needed(req: ApprovalRequest) -> int:
        return max(req.required_approvals, MIN_APPROVALS[req.risk])

    def _emit(
        self, etype: EventType, req: ApprovalRequest | None, actor: Actor | None, **data: Any
    ) -> None:
        self._audit(
            etype,
            node_id=req.node_id if req is not None else None,
            attempt=req.attempt if req is not None else None,
            actor=actor or _ENGINE,
            data=data,
        )


# --------------------------------------------------------------------------- providers


@dataclass(frozen=True)
class ProviderVote:
    """A vote a provider hands to the scheduler, which feeds it through ``ApprovalManager.vote``."""

    request_id: str
    approver: str
    approved: bool
    comment: str = ""
    channel: str = "interactive"


@runtime_checkable
class ApprovalProvider(Protocol):
    """Non-blocking source of votes: ``poll`` returns what is available *now* and never waits."""

    def poll(self, requests: Sequence[ApprovalRequest]) -> list[ProviderVote]: ...


class QueueApprover:
    """Votes arrive via ``approve`` / ``reject`` commands: this provider itself never votes."""

    def poll(self, requests: Sequence[ApprovalRequest]) -> list[ProviderVote]:
        return []


@dataclass(frozen=True)
class ScriptedDecision:
    """One pre-declared decision of a demo/CI run: who answers which node's request and how.

    Validated at construction (scripts may come from files) so that a malformed entry fails when
    the scenario is loaded, never inside the scheduler's polling loop."""

    node_id: str
    position: ApprovalPosition | None = None
    approvers: tuple[str, ...] = ()
    approved: bool = True
    comment: str = ""

    def __post_init__(self) -> None:
        names = (self.approvers,) if isinstance(self.approvers, str) else tuple(self.approvers)
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("a scripted decision needs a node id")
        if not all(isinstance(name, str) for name in names):
            raise ValueError("scripted approvers must be strings")
        if not isinstance(self.approved, bool) or not isinstance(self.comment, str):
            raise ValueError("scripted 'approved' must be a bool and 'comment' a string")
        object.__setattr__(self, "approvers", names)
        if self.position is not None:
            object.__setattr__(self, "position", ApprovalPosition(self.position))


class ScriptedApprover:
    """Replays ``ScriptedDecision`` s on the ``scripted`` channel (needs ``allow_scripted``).

    A decision answers the *first* matching pending request only (a re-run of the node is never
    auto-approved by an old script). Each approver is handed out once, and no more approvers than
    the request still needs (a rejection: one), so refused votes are never replayed in a loop.
    """

    def __init__(self, decisions: Sequence[ScriptedDecision]) -> None:
        self._decisions = tuple(decisions)
        self._bound: dict[int, str] = {}
        self._handed: dict[int, set[str]] = {}

    def poll(self, requests: Sequence[ApprovalRequest]) -> list[ProviderVote]:
        votes: list[ProviderVote] = []
        for req in requests:
            if req.status != ApprovalStatus.PENDING:
                continue
            index = self._match(req)
            if index is None:
                continue
            self._bound[index] = req.id
            decision = self._decisions[index]
            handed = self._handed.setdefault(index, set())
            voted = {_fold(v.approver) for v in req.votes}
            if decision.approved:
                approving = {_fold(v.approver) for v in req.votes if v.approved}
                room = max(0, req.required_approvals - len(approving))
            else:
                room = 1
            for name in decision.approvers:
                key = _fold(name)
                if room <= 0:
                    break
                if key in handed or key in voted:
                    continue
                handed.add(key)
                room -= 1
                votes.append(
                    ProviderVote(req.id, name, decision.approved, decision.comment, "scripted")
                )
        return votes

    def _match(self, req: ApprovalRequest) -> int | None:
        for index, decision in enumerate(self._decisions):
            if self._bound.get(index, req.id) != req.id:
                continue
            if decision.node_id == req.node_id and decision.position in (None, req.position):
                return index
        return None


@dataclass(frozen=True)
class _Prompt:
    request_id: str
    header: str
    body: tuple[str, ...]
    need: int


class InteractiveApprover:
    """Asks a human on its own daemon thread; ``poll`` never blocks, it returns the votes so far.

    ``input_fn(prompt) -> str`` behaves like ``input`` and ``output_fn(line)`` like ``print``;
    both are only ever called from the worker thread. For each new pending request the human is
    shown the engine-composed summary (the agent note under an "untrusted" label) and is asked for
    approver names and y/n answers until the request has enough approvals, someone rejects (a veto
    ends the dialogue), a blank name skips it, or the input ends/fails (the provider then stops
    asking). The manager still applies every identity rule to what is entered here.
    """

    def __init__(self, input_fn: Callable[[str], str], output_fn: Callable[[str], None]) -> None:
        self._input = input_fn
        self._output = output_fn
        self._cond = threading.Condition()
        self._work: queue.SimpleQueue[_Prompt | None] = queue.SimpleQueue()
        self._ready: list[ProviderVote] = []
        self._seen: set[str] = set()
        self._outstanding = 0
        self._closed = False
        self._thread: threading.Thread | None = None

    def poll(self, requests: Sequence[ApprovalRequest]) -> list[ProviderVote]:
        with self._cond:
            if not self._closed:
                for req in requests:
                    if req.status == ApprovalStatus.PENDING and req.id not in self._seen:
                        self._seen.add(req.id)
                        self._outstanding += 1
                        self._work.put(self._prompt_for(req))
                if self._outstanding and self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run, name="interactive-approver", daemon=True
                    )
                    self._thread.start()
            votes, self._ready = self._ready, []
            return votes

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until every queued request was dealt with (tests, orderly shutdown)."""
        with self._cond:
            return self._cond.wait_for(lambda: self._outstanding == 0, timeout)

    def close(self, timeout: float = 1.0) -> None:
        """Stop asking; a worker blocked in ``input_fn`` cannot be interrupted (daemon thread)."""
        with self._cond:
            self._closed = True
            thread = self._thread
        if thread is not None:
            self._work.put(None)
            thread.join(timeout)

    # -- worker ----------------------------------------------------------------------
    @staticmethod
    def _prompt_for(req: ApprovalRequest) -> _Prompt:
        approving = {_fold(v.approver) for v in req.votes if v.approved}
        need = max(1, req.required_approvals - len(approving))
        header = (
            f"Approval {_text(req.id, 20)}: node {_text(req.node_id, 60)}, "
            f"risk {req.risk.name}, {need} more approver(s) needed"
        )
        body = [f"  {line}" for line in sanitize_block(req.summary, MAX_SUMMARY_CHARS).split("\n")]
        if req.agent_note:
            body.append(f"  agent-authored (untrusted): {_text(req.agent_note, MAX_NOTE_CHARS)}")
        return _Prompt(req.id, _text(header, 300), tuple(body), need)

    def _run(self) -> None:
        while True:
            prompt = self._work.get()
            if prompt is None:
                return
            with self._cond:
                stop = self._closed  # close() was called while an earlier dialogue was running
            failed = False
            if not stop:
                try:
                    self._dialogue(prompt)
                except Exception:
                    failed = True
            with self._cond:
                self._outstanding -= 1
                if stop or failed:
                    self._closed = True
                    self._drain()
                self._cond.notify_all()
            if stop or failed:
                return

    def _drain(self) -> None:
        while True:
            try:
                dropped = self._work.get_nowait()
            except queue.Empty:
                return
            if dropped is not None:
                self._outstanding -= 1

    def _dialogue(self, prompt: _Prompt) -> None:
        self._output(prompt.header)
        for line in prompt.body:
            self._output(line)
        answered: set[str] = set()
        approvals = 0
        mistakes = 0
        while approvals < prompt.need and mistakes < MAX_PROMPT_MISTAKES:
            label = (
                f"approver for {prompt.request_id} ({approvals + 1}/{prompt.need}, blank skips): "
            )
            name = self._input(label).strip()[: MAX_NAME_INPUT + 1]
            if not name:
                return
            if _fold(name) in answered:
                self._output("that approver already answered; enter another name")
                mistakes += 1
                continue
            reply = self._input("approve? [y/n]: ").strip().lower()
            if reply not in ("y", "yes", "n", "no"):
                self._output("please answer y or n")
                mistakes += 1
                continue
            approved = reply in ("y", "yes")
            comment = _text(self._input("comment (optional): "), MAX_COMMENT_CHARS)
            answered.add(_fold(name))
            with self._cond:
                self._ready.append(
                    ProviderVote(prompt.request_id, name, approved, comment, "interactive")
                )
            if not approved:
                return
            approvals += 1


__all__ = [
    "ASSURANCE_SCRIPTED",
    "ApprovalManager",
    "ApprovalProvider",
    "ApprovalRefused",
    "InteractiveApprover",
    "ProviderVote",
    "QueueApprover",
    "ScriptedApprover",
    "ScriptedDecision",
    "compose_summary",
    "normalize_approver",
]
