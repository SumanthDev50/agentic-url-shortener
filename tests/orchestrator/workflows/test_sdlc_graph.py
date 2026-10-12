"""The standard SDLC workflow graph (``orchestrator/workflows/sdlc.py``).

The node table of ``docs/sdlc-contract.md`` is transcribed below (``CONTRACT``) and compared with
the graph ``build_sdlc_graph`` builds: stage, agent, dependencies, artifact kinds, gate types,
approval and the mandatory/skip flags.  Further tests pin the parts that make the workflow what
it is: the expansion template/policy of ``decompose``, the remediation loop into dynamic
``implement.*`` nodes, the parallel structure and the difference between bundle and apply mode.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import pytest

from orchestrator.core.errors import GraphError
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.schemas import SHAPES, Kind
from orchestrator.core.types import (
    ApprovalPosition,
    FailureKind,
    GateSpec,
    NodeSpec,
    PlanExpansion,
    Rewire,
    RiskLevel,
    Stage,
)
from orchestrator.workflows.sdlc import (
    IMPLEMENT_PREFIX,
    STATIC_NODE_IDS,
    SdlcOptions,
    build_sdlc_graph,
    expansion_policy,
    expansion_template,
)

GRAPH = build_sdlc_graph()
APPLY = build_sdlc_graph(SdlcOptions(promote_mode="apply"))

CONTRACT_NODE_IDS = (
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
MANDATORY = {"verify", "security-scan", "release-readiness", "promote", "release-verify"}
# Nodes whose exit gates carry ``json_shape`` in the contract.  ``integrate`` also validates the
# shape of its report explicitly although the table lists only ``integration_clean``: harmless
# (the implicit ``artifacts_present`` gate checks the schema anyway), so allowed, not required.
SHAPE_NODES = {
    "intake",
    "requirements",
    "codebase-analysis",
    "threat-model",
    "design",
    "decompose",
    "verify",
    "security-scan",
    "release-readiness",
    "promote",
    "release-verify",
}
EXTRA_SHAPE_NODES = {"integrate"}

# node: (stage, agent, depends_on, consumes, optional_consumes, overlay_from, skip_inputs, produces,
#        non-json_shape exit gates, approval as (position, always, threshold) or None)
AFTER, BEFORE = ApprovalPosition.AFTER, ApprovalPosition.BEFORE
CONTRACT: dict[str, tuple[Any, ...]] = {
    "intake": (Stage.INTAKE, "intake", (), ("request",), (), (), (), ("intake_report",), (), None),
    "requirements": (
        Stage.REQUIREMENTS,
        "requirements",
        ("intake",),
        ("request", "intake_report"),
        ("clarification_answers",),
        (),
        (),
        ("requirements",),
        (
            "acceptance_criteria_testable",
            "ambiguity_dispositions_complete",
            "no_blocking_questions",
        ),
        (AFTER, False, RiskLevel.MEDIUM),
    ),
    "codebase-analysis": (
        Stage.ANALYSIS,
        "analyst",
        ("requirements",),
        ("requirements",),
        (),
        (),
        ("intake_report",),
        ("impact_report",),
        (),
        None,
    ),
    "threat-model": (
        Stage.SECURITY,
        "threat-modeler",
        ("requirements",),
        ("requirements",),
        (),
        (),
        (),
        ("threat_model",),
        (),
        None,
    ),
    "design": (
        Stage.DESIGN,
        "architect",
        ("requirements", "threat-model", "codebase-analysis"),
        ("requirements", "threat_model"),
        ("impact_report",),
        (),
        (),
        ("design_doc", "api_contract"),
        ("design_risk", "design_covers_requirements"),
        (AFTER, False, RiskLevel.MEDIUM),
    ),
    "decompose": (
        Stage.PLANNING,
        "planner",
        ("design",),
        ("requirements", "design_doc"),
        ("impact_report",),
        (),
        (),
        ("task_plan", "plan_expansion"),
        ("plan_covers_requirements",),
        None,
    ),
    "author-tests": (
        Stage.TESTING,
        "test-author",
        ("decompose",),
        ("requirements", "task_plan", "design_doc"),
        (),
        (),
        (),
        ("changeset",),
        ("syntax_valid", "tests_cover_acceptance_criteria"),
        None,
    ),
    "integrate": (
        Stage.IMPLEMENTATION,
        "integrator",
        ("decompose", "author-tests"),
        ("changeset", "task_plan"),
        ("impact_report",),
        (),
        (),
        ("integrated_changeset", "integration_report"),
        ("integration_clean",),
        None,
    ),
    "verify": (
        Stage.TESTING,
        "verifier",
        ("integrate",),
        ("requirements", "integrated_changeset", "task_plan"),
        ("impact_report",),
        ("integrated_changeset",),
        (),
        ("test_report",),
        ("tests_passed", "fail_to_pass", "api_compat"),
        None,
    ),
    "security-scan": (
        Stage.SECURITY,
        "security-scanner",
        ("integrate",),
        ("integrated_changeset",),
        ("impact_report",),
        ("integrated_changeset",),
        (),
        ("security_report",),
        ("security_clean",),
        None,
    ),
    "docs": (
        Stage.DOCUMENTATION,
        "documenter",
        ("integrate",),
        ("requirements", "design_doc", "integrated_changeset", "task_plan"),
        ("api_contract",),
        ("integrated_changeset",),
        (),
        ("docs_changeset",),
        ("syntax_valid",),
        None,
    ),
    "release-readiness": (
        Stage.RELEASE,
        "release-manager",
        ("verify", "security-scan", "docs"),
        (
            "requirements",
            "design_doc",
            "task_plan",
            "integrated_changeset",
            "integration_report",
            "test_report",
            "security_report",
            "docs_changeset",
        ),
        ("impact_report", "threat_model", "intake_report"),
        (),
        (),
        ("readiness_report", "traceability_matrix"),
        ("traceability", "release_ready"),
        (AFTER, True, RiskLevel.MEDIUM),
    ),
    "promote": (
        Stage.RELEASE,
        "promoter",
        ("release-readiness",),
        ("integrated_changeset", "docs_changeset", "readiness_report"),
        (),
        ("integrated_changeset", "docs_changeset"),
        (),
        ("release_patch", "release_bundle"),
        (),
        None,  # bundle mode; apply mode: see test_apply_mode_differs_from_bundle_mode
    ),
    "release-verify": (
        Stage.RELEASE,
        "release-verifier",
        ("promote",),
        ("release_bundle", "release_patch"),
        (),
        ("integrated_changeset", "docs_changeset"),
        (),
        ("release_verification",),
        ("release_verified",),
        None,
    ),
}


def gate_types(specs: Sequence[GateSpec]) -> list[str]:
    return list(dict.fromkeys(spec.type for spec in specs))


def approval_of(node: NodeSpec) -> tuple[Any, ...] | None:
    a = node.approval
    return None if a is None else (a.position, a.always, a.threshold)


def template() -> NodeSpec:
    return NodeSpec.from_dict(GRAPH.node("decompose").params["expansion_template"])


# ---------------------------------------------------------------------------- shape


def test_the_graph_validates_in_both_promote_modes() -> None:
    for graph in (build_sdlc_graph(), build_sdlc_graph(SdlcOptions(promote_mode="bundle")), APPLY):
        graph.validate()
        assert len(graph) == 14


def test_there_are_fourteen_static_nodes_and_no_summary_node() -> None:
    assert STATIC_NODE_IDS == CONTRACT_NODE_IDS
    assert tuple(GRAPH.nodes) == CONTRACT_NODE_IDS  # insertion order is the documented order
    assert set(GRAPH.nodes) == set(CONTRACT.keys())
    assert "engineering-summary" not in GRAPH  # the summary is an engine finalizer
    assert not any(n.startswith(IMPLEMENT_PREFIX) for n in GRAPH.nodes)  # implement.* is dynamic


@pytest.mark.parametrize("node_id", CONTRACT_NODE_IDS)
def test_node_matches_the_contract_table(node_id: str) -> None:
    stage, agent, deps, consumes, optional, overlay, skip_inputs, produces, gates, approval = (
        CONTRACT[node_id]
    )
    node = GRAPH.node(node_id)
    assert node.stage == stage
    assert node.agent == agent
    assert node.fallback_agent is None
    assert node.depends_on == deps
    assert node.consumes == consumes
    assert node.optional_consumes == optional
    assert node.overlay_from == overlay
    assert node.skip_inputs == skip_inputs
    assert node.produces == produces
    assert approval_of(node) == approval
    assert not node.entry_gates  # bundle mode: only ``promote`` (apply) has an entry gate
    exit_types = gate_types(node.exit_gates)
    assert [t for t in exit_types if t != "json_shape"] == list(gates)
    has_shape = "json_shape" in exit_types
    assert has_shape == (node_id in SHAPE_NODES) or node_id in EXTRA_SHAPE_NODES
    for spec in node.exit_gates:
        if spec.type == "json_shape":  # every shape gate validates a kind the node produces
            assert spec.params["kind"] in node.produces
            assert spec.params["kind"] in SHAPES


def test_mandatory_nodes_cannot_be_skipped_and_only_the_analysis_has_a_skip_predicate() -> None:
    assert {n for n, spec in GRAPH.nodes.items() if spec.mandatory} == MANDATORY
    assert {n for n, spec in GRAPH.nodes.items() if spec.skip_if} == {"codebase-analysis"}
    assert GRAPH.node("codebase-analysis").skip_if == "greenfield"
    assert all(not GRAPH.node(n).skip_if for n in MANDATORY)
    assert set(GRAPH.to_dict().keys()) >= {"nodes", "expansions"}


def test_node_specific_governance_flags() -> None:
    verify = GRAPH.node("verify")
    assert verify.remediation is not None
    assert verify.retry.retry_on == frozenset({FailureKind.TRANSIENT, FailureKind.TIMEOUT})
    assert verify.autonomy is not None and verify.autonomy.can_execute  # the only executing node
    assert GRAPH.node("release-verify").autonomy is not None
    assert GRAPH.node("release-verify").autonomy.can_execute  # type: ignore[union-attr]
    readiness = GRAPH.node("release-readiness")
    assert (readiness.cacheable, readiness.risk) == (False, RiskLevel.MEDIUM)  # reads the RunView
    assert readiness.approval is not None and readiness.approval.always  # human go/no-go, always
    executing = {n for n, s in GRAPH.nodes.items() if s.autonomy and s.autonomy.can_execute}
    assert executing == {"verify", "release-verify"}
    writers = {n for n, s in GRAPH.nodes.items() if s.autonomy and s.autonomy.write_globs}
    assert writers == {"author-tests", "docs"}  # implement.* writers come from the template
    assert GRAPH.node("author-tests").autonomy.write_globs == ("tests/**",)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------- expansion


def test_decompose_owns_the_expansion_template_and_policy() -> None:
    decompose = GRAPH.node("decompose")
    assert set(decompose.params) == {"expansion_template", "expansion_policy"}
    assert decompose.params["expansion_policy"] == expansion_policy()
    assert [n for n, s in GRAPH.nodes.items() if "expansion_template" in s.params] == ["decompose"]
    assert Kind.PLAN_EXPANSION in decompose.produces  # published atomically with the task plan
    tpl = template()
    assert tpl == expansion_template()  # the persisted dict round-trips to the template


def test_the_expansion_template_is_the_contract_row_of_the_implement_nodes() -> None:
    tpl = template()
    assert (tpl.agent, tpl.fallback_agent) == ("implementer", "implementer.safe")
    assert tpl.stage == Stage.IMPLEMENTATION
    assert tpl.depends_on == ("decompose",)
    assert tpl.consumes == ("task_plan", "design_doc")
    assert tpl.optional_consumes == ("impact_report",)
    assert tpl.produces == ("changeset",)
    assert gate_types(tpl.exit_gates) == ["syntax_valid", "scope_within_plan"]
    assert not tpl.entry_gates
    assert approval_of(tpl) == (AFTER, False, RiskLevel.MEDIUM)  # risk based, never "always"
    # governance the engine copies onto every dynamic node must not be an escape hatch
    assert (tpl.mandatory, tpl.side_effecting, tpl.skip_if, tpl.compensator) == (
        False,
        False,
        None,
        None,
    )
    assert tpl.autonomy is not None and tpl.autonomy.can_execute is False
    assert tpl.retry.max_attempts == 3 and tpl.fallback_retry.max_attempts == 2


def test_the_expansion_policy_allows_exactly_what_the_template_needs() -> None:
    policy = expansion_policy()
    tpl = template()
    assert policy["id_prefix"] == IMPLEMENT_PREFIX == "implement."
    assert policy["allowed_agents"] == ["implementer", "implementer.safe"]
    assert policy["allowed_stages"] == ["implementation"]
    assert policy["allowed_produces"] == ["changeset"]
    assert policy["rewire_targets"] == ["integrate"]
    assert policy["max_nodes"] == 8
    assert tpl.agent in policy["allowed_agents"] and tpl.fallback_agent in policy["allowed_agents"]
    assert tpl.stage.value in policy["allowed_stages"]
    assert set(tpl.produces) <= set(policy["allowed_produces"])


def test_remediation_routes_into_nodes_that_do_not_exist_yet() -> None:
    """Regression: ``implement.*`` matches nothing in the static graph and must still validate.

    The validator accepts a ``prefix.*`` pattern without a current match when some node owns an
    ``expansion_template``; without that owner the same graph is rejected.
    """
    remediation = GRAPH.node("verify").remediation
    assert remediation is not None
    assert (remediation.route_to, remediation.max_loops) == (("implement.*", "author-tests"), 2)
    assert GRAPH.resolve_pattern("implement.*") == []
    GRAPH.validate()
    ownerless = WorkflowGraph(
        dataclasses.replace(spec, params={}) if spec.id == "decompose" else spec
        for spec in GRAPH.nodes.values()
    )
    with pytest.raises(GraphError, match="matches no node"):
        ownerless.validate()


def _three_group_expansion() -> PlanExpansion:
    nodes = tuple(dataclasses.replace(expansion_template(), id=f"implement.{g}") for g in "abc")
    return PlanExpansion(
        "decompose", nodes, {"integrate": Rewire(add_depends_on=tuple(n.id for n in nodes))}
    )


def test_a_three_group_expansion_runs_in_parallel_with_author_tests_and_joins_at_integrate() -> (
    None
):
    graph = build_sdlc_graph()
    graph.apply_expansion(_three_group_expansion())  # validates the whole graph again
    layers = graph.layers()
    parallel = next(layer for layer in layers if "author-tests" in layer)
    assert set(parallel) == {"author-tests", "implement.a", "implement.b", "implement.c"}
    assert graph.resolve_pattern("implement.*") == ["implement.a", "implement.b", "implement.c"]
    # the rewire adds edges to the effective dependencies; the static spec stays untouched
    assert set(graph.deps("integrate")) == {
        "decompose",
        "author-tests",
        *graph.resolve_pattern("implement.*"),
    }
    assert graph.node("integrate").depends_on == ("decompose", "author-tests")
    assert [graph.owner_of(n) for n in graph.resolve_pattern("implement.*")] == ["decompose"] * 3
    assert {"implement.a", "implement.b", "implement.c"} <= graph.ancestors("verify")
    assert all(n.mandatory is False for n in graph.nodes.values() if n.id.startswith("implement."))
    assert graph.producers_of("changeset") == [
        "author-tests",
        "implement.a",
        "implement.b",
        "implement.c",
    ]


# ---------------------------------------------------------------------------- parallel structure


def test_the_static_graph_has_the_documented_parallel_structure() -> None:
    layers = GRAPH.layers()
    assert layers == [
        ["intake"],
        ["requirements"],
        ["codebase-analysis", "threat-model"],  # analysis runs beside the threat model
        ["design"],
        ["decompose"],
        ["author-tests"],  # implement.* join this layer once ``decompose`` expanded
        ["integrate"],  # the barrier
        ["verify", "security-scan", "docs"],  # three independent checks of the integrated change
        ["release-readiness"],
        ["promote"],
        ["release-verify"],
    ]
    assert GRAPH.node("integrate").depends_on == ("decompose", "author-tests")
    for parallel in ("verify", "security-scan", "docs"):
        assert GRAPH.node(parallel).depends_on == ("integrate",)
    assert GRAPH.ancestors("verify").isdisjoint({"security-scan", "docs"})  # siblings do not wait
    assert {"verify", "security-scan", "docs"} <= GRAPH.ancestors("release-readiness")
    assert GRAPH.ancestors("design") >= {"codebase-analysis", "threat-model", "requirements"}


# ---------------------------------------------------------------------------- promote modes


def test_apply_mode_differs_from_bundle_mode_only_at_promote_and_release_verify() -> None:
    bundle_promote, apply_promote = GRAPH.node("promote"), APPLY.node("promote")
    # the promoter reads ``ctx.params["mode"]``: without it an "apply" graph would silently bundle
    assert (bundle_promote.params, apply_promote.params) == ({}, {"mode": "apply"})
    assert (bundle_promote.side_effecting, bundle_promote.compensator) == (False, None)
    assert approval_of(bundle_promote) is None
    assert bundle_promote.entry_gates == ()
    assert (bundle_promote.risk, bundle_promote.cacheable) == (RiskLevel.LOW, True)
    assert bundle_promote.retry.max_attempts == 2
    assert GRAPH.node("release-verify").compensates == ()

    assert (apply_promote.side_effecting, apply_promote.compensator) == (True, "apply_bundle")
    assert approval_of(apply_promote) == (BEFORE, True, RiskLevel.MEDIUM)  # BEFORE, always
    assert gate_types(apply_promote.entry_gates) == ["baseline_unchanged"]
    assert (apply_promote.risk, apply_promote.cacheable) == (RiskLevel.HIGH, False)
    assert apply_promote.retry.max_attempts == 1  # a real apply is never retried blindly
    assert apply_promote.mandatory and bundle_promote.mandatory
    assert APPLY.node("release-verify").compensates == ("promote",)  # saga rollback on failure
    assert "apply" in apply_promote.title and "apply" not in bundle_promote.title

    others = [n for n in CONTRACT_NODE_IDS if n not in {"promote", "release-verify"}]
    for node_id in others:
        assert GRAPH.node(node_id).to_dict() == APPLY.node(node_id).to_dict(), node_id
    bundle_verify = GRAPH.node("release-verify").to_dict()
    apply_verify = APPLY.node("release-verify").to_dict()
    assert {k for k in bundle_verify if bundle_verify[k] != apply_verify[k]} == {"compensates"}
    assert GRAPH.to_dict() != APPLY.to_dict()


# ---------------------------------------------------------------------------- options


def test_options_are_validated_and_reach_the_nodes() -> None:
    with pytest.raises(ValueError, match="promote_mode"):
        SdlcOptions(promote_mode="deploy")
    llm = build_sdlc_graph(SdlcOptions(llm_roles=("requirements",)))
    requirements = llm.node("requirements")
    assert (requirements.agent, requirements.fallback_agent) == ("requirements.llm", "requirements")
    assert GRAPH.node("requirements").fallback_agent is None
    budget = build_sdlc_graph(SdlcOptions(params={"perf_budget_ms": 90}))
    assert budget.node("verify").params == {"perf_budget_ms": 90}
    assert GRAPH.node("verify").params == {"perf_budget_ms": 250}


def test_building_the_graph_is_deterministic_and_does_not_share_state() -> None:
    first, second = build_sdlc_graph(), build_sdlc_graph()
    assert first.to_dict() == second.to_dict() == GRAPH.to_dict()
    assert first.graph_hash() == second.graph_hash() != APPLY.graph_hash()
    assert WorkflowGraph.from_dict(GRAPH.to_dict()).graph_hash() == GRAPH.graph_hash()
    first.apply_expansion(_three_group_expansion())
    assert len(first) == 17 and len(second) == 14 and len(GRAPH) == 14  # no leakage between builds
    assert first.to_dict() != second.to_dict()
