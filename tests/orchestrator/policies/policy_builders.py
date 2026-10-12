"""Builders shared by the policy tests: subjects, runtime-assembled fake secrets, a ``RunView``.

Secret-looking strings are assembled at runtime from fragments so that no secret-shaped literal
exists in the repository (organisation rule); every helper returning one says so in its name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from orchestrator.core.fsutil import sha256_text
from orchestrator.core.paths import make_artifact_id
from orchestrator.core.types import (
    EXTERNAL_PRODUCER,
    ActionSubject,
    Actor,
    ApprovalPosition,
    ApprovalPurpose,
    ApprovalRequest,
    ApprovalSpec,
    ApprovalStatus,
    ApprovalVote,
    Artifact,
    ArtifactStatus,
    ArtifactSubject,
    AuditEvent,
    AutonomyProfile,
    ChangeSet,
    ChangeSetSubject,
    Decision,
    DecisionKind,
    FileOp,
    Finding,
    NodeSpec,
    NodeStatus,
    PlanExpansion,
    PlanSubject,
    PolicyCategory,
    PolicySubject,
    RiskLevel,
    RunSubject,
    Severity,
    Stage,
    ToolAction,
    Waiver,
)
from orchestrator.policies.base import RuleBase
from orchestrator.policies.config import PolicyConfig
from orchestrator.tools.changeset import make_create, make_delete, make_modify
from orchestrator.tools.workspace import MemoryView

T0 = datetime(2026, 1, 1, tzinfo=UTC)


# ----------------------------------------------------------------------------- fake secrets
# (runtime assembled)


def fake_aws_key_id() -> str:
    return "AK" + "IA" + "X" * 16


def fake_github_token() -> str:
    return "gh" + "p_" + "aB3dE5" * 6


def fake_stripe_key() -> str:
    return "sk" + "_live_" + "a1B2c3D4" * 3


def fake_jwt() -> str:
    return "ey" + "Jhbgci0123.ey" + "Jzdwiabcd123.sig" + "natur3abc"


def fake_private_key_header() -> str:
    return "-----BEGIN " + "RSA PRIV" + "ATE KEY-----"


def fake_password_literal() -> str:
    return "Zx9kQ2" + "mLp4Rt7"


# ----------------------------------------------------------------------------- change sets


def view(files: Mapping[str, str] | None = None) -> MemoryView:
    return MemoryView(files or {})


def create(path: str, content: str) -> FileOp:
    return make_create(path, content)


def modify(base: MemoryView, path: str, content: str) -> FileOp:
    return make_modify(base, path, content)


def delete(base: MemoryView, path: str) -> FileOp:
    return make_delete(base, path)


def profile(**overrides: Any) -> AutonomyProfile:
    params: dict[str, Any] = {"write_globs": ("**",)}
    params.update(overrides)
    return AutonomyProfile(**params)


def cs_subject(
    ops: Iterable[FileOp],
    base: MemoryView | None = None,
    *,
    profile_: AutonomyProfile | None = None,
    impact: Mapping[str, Any] | None = None,
    sources: Sequence[ChangeSet] = (),
    node_id: str = "implement.core",
    author: str = "agent-1",
    description: str = "change",
    no_profile: bool = False,
) -> ChangeSetSubject:
    return ChangeSetSubject(
        ChangeSet(tuple(ops), description, author),
        base if base is not None else view(),
        node_id=node_id,
        author=author,
        impact=impact,
        profile=None if no_profile else (profile_ or profile()),
        sources=tuple(sources),
    )


def py_subject(path: str, source: str, **kwargs: Any) -> ChangeSetSubject:
    return cs_subject([create(path, source)], **kwargs)


def rule_ids(findings: Iterable[Finding]) -> list[str]:
    return [f.rule_id for f in findings]


# ----------------------------------------------------------------------------- artifacts / actions
# / plans


def make_artifact(
    kind: str = "request",
    *,
    key: str = "",
    version: int = 1,
    producer: str = EXTERNAL_PRODUCER,
    status: ArtifactStatus = ArtifactStatus.ACTIVE,
    content: str = "x",
    metadata: Mapping[str, Any] | None = None,
) -> Artifact:
    return Artifact(
        id=make_artifact_id(kind, key, version),
        kind=kind,
        key=key,
        version=version,
        content_hash=sha256_text(content),
        media_type="text/plain",
        size=len(content),
        path=f"{kind}/v{version}/content.txt",
        created_at=T0,
        producer=producer,
        attempt=0,
        agent_id="",
        status=status,
        metadata=dict(metadata or {}),
    )


def artifact_subject(
    text: str | None, *, untrusted: bool = False, **kwargs: Any
) -> ArtifactSubject:
    return ArtifactSubject(make_artifact(content=text or "", **kwargs), text, untrusted)


def action_subject(
    action: ToolAction, target: str, prof: AutonomyProfile | None = None
) -> ActionSubject:
    return ActionSubject("agent-1", "implement.core", action, target, prof or profile())


def node(
    node_id: str,
    *,
    depends_on: Sequence[str] = (),
    agent: str = "agent-a",
    stage: Stage = Stage.IMPLEMENTATION,
    produces: Sequence[str] = ("changeset",),
    mandatory: bool = False,
    approval: ApprovalSpec | None = None,
    **kwargs: Any,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        title=node_id,
        stage=stage,
        agent=agent,
        depends_on=tuple(depends_on),
        produces=tuple(produces),
        mandatory=mandatory,
        approval=approval,
        **kwargs,
    )


def plan_subject(
    expansion: PlanExpansion,
    before: Mapping[str, NodeSpec],
    after: Mapping[str, NodeSpec] | None = None,
    **kwargs: Any,
) -> PlanSubject:
    merged = dict(before)
    merged.update({n.id: n for n in expansion.nodes})
    defaults: dict[str, Any] = {
        "allowed_agents": frozenset({"agent-a"}),
        "allowed_produces": frozenset({"changeset"}),
        "allowed_stages": frozenset({"implementation"}),
        "mandatory": (),
    }
    defaults.update(kwargs)
    return PlanSubject("run-1", expansion, dict(before), dict(after or merged), **defaults)


# ----------------------------------------------------------------------------- run view


class FakeRunView:
    """Minimal in-memory ``RunView``; artifacts are registered with their decoded payload."""

    def __init__(
        self,
        specs: Iterable[NodeSpec] = (),
        *,
        statuses: Mapping[str, NodeStatus] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self._specs = {s.id: s for s in specs}
        self._statuses = dict(
            statuses or {s.id: NodeStatus.SUCCEEDED for s in self._specs.values()}
        )
        self._params = dict(params or {})
        self._artifacts: list[Artifact] = []
        self._payloads: dict[str, Any] = {}
        self._decisions: list[Decision] = []
        self._approvals: list[ApprovalRequest] = []
        self._waivers: list[Waiver] = []
        self._events: list[AuditEvent] = []

    # -- population
    def add_artifact(
        self,
        kind: str,
        payload: Any,
        *,
        producer: str,
        status: ArtifactStatus = ArtifactStatus.ACTIVE,
        key: str = "",
        version: int = 1,
    ) -> Artifact:
        art = make_artifact(kind, key=key, version=version, producer=producer, status=status)
        self._artifacts.append(art)
        self._payloads[art.id] = payload
        return art

    def add_decision(
        self,
        kind: DecisionKind,
        *,
        actor: Actor | None = None,
        node_id: str | None = None,
        refs: Sequence[str] = (),
        decision_id: str | None = None,
    ) -> Decision:
        decision = Decision(
            id=decision_id or f"D-{len(self._decisions) + 1:04d}",
            run_id="run-1",
            ts=T0,
            kind=kind,
            actor=actor or Actor.human("alice"),
            node_id=node_id,
            summary="s",
            refs=tuple(refs),
        )
        self._decisions.append(decision)
        return decision

    def add_approval(self, request: ApprovalRequest) -> None:
        self._approvals.append(request)

    def add_waiver(self, waiver: Waiver) -> None:
        self._waivers.append(waiver)

    # -- GovernanceView
    def node_statuses(self) -> Mapping[str, NodeStatus]:
        return dict(self._statuses)

    def decisions(self) -> tuple[Decision, ...]:
        return tuple(self._decisions)

    def decisions_for(self, ref: Any) -> tuple[Decision, ...]:
        return ()

    def approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(self._approvals)

    def waivers(self) -> tuple[Waiver, ...]:
        return tuple(self._waivers)

    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    # -- RunView
    def params(self) -> Mapping[str, Any]:
        return dict(self._params)

    def node_specs(self) -> Mapping[str, NodeSpec]:
        return dict(self._specs)

    def ancestors(self, node_id: str) -> frozenset[str]:
        if node_id not in self._specs:
            raise KeyError(node_id)
        seen: set[str] = set()
        stack = list(self._specs[node_id].depends_on)
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(self._specs[current].depends_on)
        return frozenset(seen)

    def mandatory_nodes(self) -> tuple[str, ...]:
        return tuple(s.id for s in self._specs.values() if s.mandatory)

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        found = [
            a
            for a in self._artifacts
            if a.kind == kind and a.key == key and a.status == ArtifactStatus.ACTIVE
        ]
        return max(found, key=lambda a: a.version, default=None)

    def artifacts(self, include_inactive: bool = False) -> tuple[Artifact, ...]:
        return tuple(
            a for a in self._artifacts if include_inactive or a.status == ArtifactStatus.ACTIVE
        )

    def read_text(self, artifact: Artifact) -> str:
        return str(self._payloads[artifact.id])

    def read_json(self, artifact: Artifact) -> Any:
        payload = self._payloads[artifact.id]
        if isinstance(payload, Exception):
            raise payload
        return payload


def release_chain() -> list[NodeSpec]:
    """requirements -> verify -> release-readiness (the gated node)."""
    return [
        node("requirements", produces=("requirements",), stage=Stage.REQUIREMENTS),
        node(
            "verify", depends_on=("requirements",), produces=("test_report",), stage=Stage.TESTING
        ),
        node("design", depends_on=("requirements",), produces=("design_doc",), stage=Stage.DESIGN),
        node(
            "release-readiness",
            depends_on=("verify", "design"),
            produces=("readiness_report",),
            stage=Stage.RELEASE,
        ),
    ]


def run_subject(view_: FakeRunView, node_id: str = "release-readiness") -> RunSubject:
    return RunSubject("run-1", view_, node_id)


def approval(
    node_id: str,
    *,
    request_id: str = "A-0001",
    status: ApprovalStatus = ApprovalStatus.APPROVED,
    voters: Sequence[str] = ("alice",),
    required: int = 1,
    risk: RiskLevel = RiskLevel.MEDIUM,
    fingerprint: str = "fp-1",
    evidence_hash: str = "ev-1",
    vote_evidence: str | None = None,
    created_offset: int = 0,
) -> ApprovalRequest:
    votes = [
        ApprovalVote(v, True, "", T0, "cli", "os", vote_evidence or evidence_hash) for v in voters
    ]
    return ApprovalRequest(
        id=request_id,
        run_id="run-1",
        node_id=node_id,
        position=ApprovalPosition.AFTER,
        purpose=ApprovalPurpose.NODE_OUTPUT,
        risk=risk,
        reasons=(),
        summary="",
        evidence=(),
        node_fingerprint=fingerprint,
        evidence_hash=evidence_hash,
        required_approvals=required,
        attempt=1,
        generation=1,
        created_at=T0.replace(second=created_offset),
        status=status,
        votes=votes,
    )


def waiver_for(
    finding: Finding, *, node_id: str, evidence: str, rule_id: str | None = None
) -> Waiver:
    return Waiver(
        id=f"W-{finding.key()[:6]}",
        rule_id=rule_id or finding.rule_id,
        finding_key=finding.key(),
        node_id=node_id,
        evidence_hash=evidence,
        justification="accepted risk",
        approved_by=("alice",),
        approval_id="A-0001",
        decision_id="D-0001",
        created_at=T0,
    )


def make_finding(rule_id: str = "SEC-006", severity: Severity = Severity.MEDIUM) -> Finding:
    return Finding(rule_id, PolicyCategory.SECURITY, severity, "m", location="x:1", evidence="e")


def check(
    rule_cls: type[RuleBase], subject: PolicySubject, config: PolicyConfig | None = None
) -> list[Finding]:
    """Instantiate one rule with ``config`` (default policy config) and run it on ``subject``."""
    return rule_cls(config).check(subject)
