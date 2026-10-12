"""The standard SDLC workflow graph (exact transcription of ``docs/sdlc-contract.md``).

``build_sdlc_graph(options)`` returns a validated ``WorkflowGraph``. The implementation nodes are
*not* in the static graph: the ``decompose`` node publishes a ``PlanExpansion`` and the engine
instantiates ``implement.<group>`` nodes from the owner's ``expansion_template`` (governance
attributes come from the human-authored template, never from the agent) within the owner's
``expansion_policy``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    ApprovalPosition,
    ApprovalSpec,
    AutonomyProfile,
    FailureKind,
    GateSpec,
    NodeSpec,
    RemediationSpec,
    RetryPolicy,
    RiskLevel,
    Stage,
)

IMPLEMENT_PREFIX = "implement."
STATIC_NODE_IDS = (
    "intake",
    "requirements",
    "codebase-analysis",
    "threat-model",
    "design",
    "decompose",
    "author-tests",
    "integrate",
    "verify",
    "security-scan",
    "docs",
    "release-readiness",
    "promote",
    "release-verify",
)

_TRANSIENT_ONLY = RetryPolicy(
    max_attempts=2, retry_on=frozenset({FailureKind.TRANSIENT, FailureKind.TIMEOUT})
)


@dataclass(frozen=True)
class SdlcOptions:
    promote_mode: str = "bundle"  # "bundle" | "apply"
    llm_roles: tuple[
        str, ...
    ] = ()  # roles served by '<role>.llm' primary with the deterministic agent as fallback
    params: Mapping[str, Any] = field(default_factory=dict)  # scenario params (e.g. perf_budget_ms)

    def __post_init__(self) -> None:
        if self.promote_mode not in ("bundle", "apply"):
            raise ValueError("promote_mode must be 'bundle' or 'apply'")


def _shape(kind: str) -> GateSpec:
    return GateSpec("json_shape", {"kind": kind})


def _agent(options: SdlcOptions, role: str) -> tuple[str, str | None]:
    if role in options.llm_roles:
        return f"{role}.llm", role
    return role, None


def expansion_template() -> NodeSpec:
    """Governance template for dynamically created ``implement.<group>`` nodes."""
    return NodeSpec(
        id="implement.template",
        title="implementation group",
        stage=Stage.IMPLEMENTATION,
        agent="implementer",
        fallback_agent="implementer.safe",
        depends_on=("decompose",),
        consumes=(Kind.TASK_PLAN, Kind.DESIGN_DOC),
        optional_consumes=(Kind.IMPACT_REPORT,),
        produces=(Kind.CHANGESET,),
        retry=RetryPolicy(max_attempts=3),
        fallback_retry=RetryPolicy(max_attempts=2),
        exit_gates=(GateSpec("syntax_valid"), GateSpec("scope_within_plan")),
        approval=ApprovalSpec(
            ApprovalPosition.AFTER,
            always=False,
            threshold=RiskLevel.MEDIUM,
            reason="implementation review",
        ),
        risk=RiskLevel.LOW,
        autonomy=AutonomyProfile(write_globs=("**",), max_changed_files=40),
        timeout_s=120.0,
    )


def expansion_policy() -> dict[str, Any]:
    return {
        "id_prefix": IMPLEMENT_PREFIX,
        "allowed_agents": ["implementer", "implementer.safe"],
        "allowed_stages": [Stage.IMPLEMENTATION.value],
        "allowed_produces": [Kind.CHANGESET],
        "rewire_targets": ["integrate"],
        "max_nodes": 8,
    }


def build_sdlc_graph(options: SdlcOptions | None = None) -> WorkflowGraph:
    opts = options or SdlcOptions()
    apply_mode = opts.promote_mode == "apply"
    perf_budget_ms = opts.params.get("perf_budget_ms", 250)

    req_agent, req_fallback = _agent(opts, "requirements")
    impl_template = expansion_template()

    nodes = [
        NodeSpec(
            id="intake",
            title="Intake and classification",
            stage=Stage.INTAKE,
            agent="intake",
            consumes=(Kind.REQUEST,),
            produces=(Kind.INTAKE_REPORT,),
            exit_gates=(_shape(Kind.INTAKE_REPORT),),
            timeout_s=60.0,
        ),
        NodeSpec(
            id="requirements",
            title="Requirement understanding",
            stage=Stage.REQUIREMENTS,
            agent=req_agent,
            fallback_agent=req_fallback,
            depends_on=("intake",),
            consumes=(Kind.REQUEST, Kind.INTAKE_REPORT),
            optional_consumes=(Kind.CLARIFICATION_ANSWERS,),
            produces=(Kind.REQUIREMENTS,),
            exit_gates=(
                _shape(Kind.REQUIREMENTS),
                GateSpec("acceptance_criteria_testable"),
                GateSpec("ambiguity_dispositions_complete"),
                GateSpec("no_blocking_questions"),
            ),
            approval=ApprovalSpec(
                ApprovalPosition.AFTER,
                always=False,
                threshold=RiskLevel.MEDIUM,
                reason="assumptions require sign-off",
            ),
            timeout_s=60.0,
        ),
        NodeSpec(
            id="codebase-analysis",
            title="Codebase impact analysis",
            stage=Stage.ANALYSIS,
            agent="analyst",
            depends_on=("requirements",),
            consumes=(Kind.REQUIREMENTS,),
            skip_inputs=(Kind.INTAKE_REPORT,),
            skip_if="greenfield",
            produces=(Kind.IMPACT_REPORT,),
            exit_gates=(_shape(Kind.IMPACT_REPORT),),
            timeout_s=120.0,
        ),
        NodeSpec(
            id="threat-model",
            title="Threat model (STRIDE-lite)",
            stage=Stage.SECURITY,
            agent="threat-modeler",
            depends_on=("requirements",),
            consumes=(Kind.REQUIREMENTS,),
            produces=(Kind.THREAT_MODEL,),
            exit_gates=(_shape(Kind.THREAT_MODEL),),
            timeout_s=60.0,
        ),
        NodeSpec(
            id="design",
            title="Architecture and design",
            stage=Stage.DESIGN,
            agent="architect",
            depends_on=("requirements", "threat-model", "codebase-analysis"),
            consumes=(Kind.REQUIREMENTS, Kind.THREAT_MODEL),
            optional_consumes=(Kind.IMPACT_REPORT,),
            produces=(Kind.DESIGN_DOC, Kind.API_CONTRACT),
            exit_gates=(
                _shape(Kind.DESIGN_DOC),
                _shape(Kind.API_CONTRACT),
                GateSpec("design_risk"),
                GateSpec("design_covers_requirements"),
            ),
            approval=ApprovalSpec(
                ApprovalPosition.AFTER,
                always=False,
                threshold=RiskLevel.MEDIUM,
                reason="design review",
            ),
            timeout_s=60.0,
        ),
        NodeSpec(
            id="decompose",
            title="Task decomposition and plan expansion",
            stage=Stage.PLANNING,
            agent="planner",
            depends_on=("design",),
            consumes=(Kind.REQUIREMENTS, Kind.DESIGN_DOC),
            optional_consumes=(Kind.IMPACT_REPORT,),
            produces=(Kind.TASK_PLAN, Kind.PLAN_EXPANSION),
            exit_gates=(_shape(Kind.TASK_PLAN), GateSpec("plan_covers_requirements")),
            params={
                "expansion_template": impl_template.to_dict(),
                "expansion_policy": expansion_policy(),
            },
            timeout_s=60.0,
        ),
        NodeSpec(
            id="author-tests",
            title="Author tests from acceptance criteria",
            stage=Stage.TESTING,
            agent="test-author",
            depends_on=("decompose",),
            consumes=(Kind.REQUIREMENTS, Kind.TASK_PLAN, Kind.DESIGN_DOC),
            produces=(Kind.CHANGESET,),
            exit_gates=(GateSpec("syntax_valid"), GateSpec("tests_cover_acceptance_criteria")),
            autonomy=AutonomyProfile(write_globs=("tests/**",), max_changed_files=40),
            timeout_s=120.0,
        ),
        NodeSpec(
            id="integrate",
            title="Integrate parallel change sets",
            stage=Stage.IMPLEMENTATION,
            agent="integrator",
            depends_on=("decompose", "author-tests"),
            consumes=(Kind.CHANGESET, Kind.TASK_PLAN),
            optional_consumes=(Kind.IMPACT_REPORT,),
            produces=(Kind.INTEGRATED_CHANGESET, Kind.INTEGRATION_REPORT),
            exit_gates=(_shape(Kind.INTEGRATION_REPORT), GateSpec("integration_clean")),
            timeout_s=120.0,
        ),
        NodeSpec(
            id="verify",
            title="Verify: tests, fail-to-pass, API compatibility",
            stage=Stage.TESTING,
            agent="verifier",
            depends_on=("integrate",),
            consumes=(Kind.REQUIREMENTS, Kind.INTEGRATED_CHANGESET, Kind.TASK_PLAN),
            optional_consumes=(Kind.IMPACT_REPORT,),
            overlay_from=(Kind.INTEGRATED_CHANGESET,),
            produces=(Kind.TEST_REPORT,),
            exit_gates=(
                _shape(Kind.TEST_REPORT),
                GateSpec("tests_passed"),
                GateSpec("fail_to_pass"),
                GateSpec("api_compat"),
            ),
            retry=_TRANSIENT_ONLY,
            mandatory=True,
            remediation=RemediationSpec(route_to=("implement.*", "author-tests"), max_loops=2),
            params={"perf_budget_ms": perf_budget_ms},
            autonomy=AutonomyProfile(can_execute=True, max_runtime_s=300.0),
            timeout_s=300.0,
        ),
        NodeSpec(
            id="security-scan",
            title="Security scan of the integrated change",
            stage=Stage.SECURITY,
            agent="security-scanner",
            depends_on=("integrate",),
            consumes=(Kind.INTEGRATED_CHANGESET,),
            optional_consumes=(Kind.IMPACT_REPORT,),
            overlay_from=(Kind.INTEGRATED_CHANGESET,),
            produces=(Kind.SECURITY_REPORT,),
            exit_gates=(_shape(Kind.SECURITY_REPORT), GateSpec("security_clean")),
            retry=_TRANSIENT_ONLY,
            mandatory=True,
            timeout_s=120.0,
        ),
        NodeSpec(
            id="docs",
            title="Generate documentation",
            stage=Stage.DOCUMENTATION,
            agent="documenter",
            depends_on=("integrate",),
            consumes=(
                Kind.REQUIREMENTS,
                Kind.DESIGN_DOC,
                Kind.INTEGRATED_CHANGESET,
                Kind.TASK_PLAN,
            ),
            optional_consumes=(Kind.API_CONTRACT,),
            overlay_from=(Kind.INTEGRATED_CHANGESET,),
            produces=(Kind.DOCS_CHANGESET,),
            exit_gates=(GateSpec("syntax_valid"),),
            autonomy=AutonomyProfile(
                write_globs=("docs/**", "CHANGELOG.md", "README.md"), max_changed_files=20
            ),
            timeout_s=120.0,
        ),
        NodeSpec(
            id="release-readiness",
            title="Release readiness and go/no-go",
            stage=Stage.RELEASE,
            agent="release-manager",
            depends_on=("verify", "security-scan", "docs"),
            consumes=(
                Kind.REQUIREMENTS,
                Kind.DESIGN_DOC,
                Kind.TASK_PLAN,
                Kind.INTEGRATED_CHANGESET,
                Kind.INTEGRATION_REPORT,
                Kind.TEST_REPORT,
                Kind.SECURITY_REPORT,
                Kind.DOCS_CHANGESET,
            ),
            optional_consumes=(Kind.IMPACT_REPORT, Kind.THREAT_MODEL, Kind.INTAKE_REPORT),
            produces=(Kind.READINESS_REPORT, Kind.TRACEABILITY_MATRIX),
            exit_gates=(
                _shape(Kind.READINESS_REPORT),
                _shape(Kind.TRACEABILITY_MATRIX),
                GateSpec("traceability"),
                GateSpec("release_ready"),
            ),
            approval=ApprovalSpec(ApprovalPosition.AFTER, always=True, reason="release go/no-go"),
            mandatory=True,
            cacheable=False,
            risk=RiskLevel.MEDIUM,
            timeout_s=120.0,
        ),
        NodeSpec(
            id="promote",
            title="Promote: build the reviewed patch bundle"
            + (" and apply it" if apply_mode else ""),
            stage=Stage.RELEASE,
            agent="promoter",
            depends_on=("release-readiness",),
            consumes=(Kind.INTEGRATED_CHANGESET, Kind.DOCS_CHANGESET, Kind.READINESS_REPORT),
            overlay_from=(Kind.INTEGRATED_CHANGESET, Kind.DOCS_CHANGESET),
            produces=(Kind.RELEASE_PATCH, Kind.RELEASE_BUNDLE),
            # the promoter applies for real only if its node params say so (``ctx.params["mode"]``)
            params={"mode": "apply"} if apply_mode else {},
            entry_gates=(GateSpec("baseline_unchanged"),) if apply_mode else (),
            exit_gates=(_shape(Kind.RELEASE_BUNDLE),),
            approval=ApprovalSpec(
                ApprovalPosition.BEFORE, always=True, reason="real apply to the target tree"
            )
            if apply_mode
            else None,
            side_effecting=apply_mode,
            compensator="apply_bundle" if apply_mode else None,
            risk=RiskLevel.HIGH if apply_mode else RiskLevel.LOW,
            mandatory=True,
            cacheable=not apply_mode,
            retry=RetryPolicy(max_attempts=1) if apply_mode else RetryPolicy(max_attempts=2),
            timeout_s=120.0,
        ),
        NodeSpec(
            id="release-verify",
            title="Verify the released bundle on a fresh baseline copy",
            stage=Stage.RELEASE,
            agent="release-verifier",
            depends_on=("promote",),
            consumes=(Kind.RELEASE_BUNDLE, Kind.RELEASE_PATCH),
            overlay_from=(Kind.INTEGRATED_CHANGESET, Kind.DOCS_CHANGESET),
            produces=(Kind.RELEASE_VERIFICATION,),
            exit_gates=(_shape(Kind.RELEASE_VERIFICATION), GateSpec("release_verified")),
            retry=_TRANSIENT_ONLY,
            mandatory=True,
            compensates=("promote",) if apply_mode else (),
            autonomy=AutonomyProfile(can_execute=True, max_runtime_s=300.0),
            timeout_s=300.0,
        ),
    ]
    graph = WorkflowGraph(nodes)
    graph.validate()
    return graph
