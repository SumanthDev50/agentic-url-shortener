"""Tests for the planner agent: tasks, groups, critical path and the ``implement.*`` expansion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from orchestrator.agents.planner import INTEGRATE_NODE, PlannerAgent
from orchestrator.agents.recipes.base import Capability, GroupSpec, RecipeParamError
from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import AgentError
from orchestrator.core.paths import NODE_ID_RE
from orchestrator.core.schemas import GROUP_ID_RE, TASK_ID_RE, Kind, validate_payload
from orchestrator.core.serde import canonical_json
from orchestrator.core.types import (
    K_CLARIFICATION_ANSWERS,
    AgentResult,
    Decision,
    Enforcement,
    GateContext,
    GateDecision,
    GatePhase,
    PlanSubject,
    Rewire,
    Stage,
)
from orchestrator.engine.config import EngineConfig
from orchestrator.engine.expansion import normalize_expansion
from orchestrator.policies.engine import PolicyEngine
from orchestrator.tools.workspace import MemoryView
from orchestrator.workflows.gates_sdlc import (
    AcceptanceCriteriaTestable,
    AmbiguityDispositionsComplete,
    DesignCoversRequirements,
    DesignRisk,
    PlanCoversRequirements,
)
from orchestrator.workflows.sdlc import build_sdlc_graph, expansion_policy
from tests.orchestrator.agents.analysis_rig import (
    MODES,
    FakeRunView,
    MemoryReader,
    activate,
    ambiguous_inputs,
    capability_entry,
    fake_export,
    fake_linkcheck,
    fake_max_clicks,
    fingerprint,
    install_catalog,
    payload,
    req_entry,
    requirements_payload,
    run_agent,
    run_chain,
    scenario_request,
)


def design_payload(requirement_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": 1,
        "summary": "s",
        "components": [
            {
                "name": "c",
                "responsibility": "r",
                "touches": [],
                "requirement_ids": list(requirement_ids),
            }
        ],
        "adrs": [],
        "data_model_changes": [],
        "api_changes": [],
        "rollback_plan": "r",
        "change_reference": "CR-000000",
        "risk": {"level": "LOW", "reasons": []},
    }


def plan(
    entries: Sequence[dict[str, Any]],
    caps: Sequence[dict[str, Any]] = (),
    *,
    unsupported: Sequence[dict[str, Any]] = (),
    covered: Sequence[str] | None = None,
    node_params: Mapping[str, Any] | None = None,
) -> AgentResult:
    ids = [e["id"] for e in entries] if covered is None else list(covered)
    return run_agent(
        PlannerAgent(),
        artifacts={
            Kind.REQUIREMENTS: requirements_payload(
                entries, capabilities=caps, unsupported=unsupported
            ),
            Kind.DESIGN_DOC: design_payload(ids),
        },
        node_id="decompose",
        stage=Stage.PLANNING,
        node_params=node_params,
    )


def linkcheck_plan(cap: Capability | None = None) -> AgentResult:
    cap = cap or fake_linkcheck()
    entries = [
        req_entry(
            1,
            "The checker must classify each result.",
            aspects=["classification"],
            capabilities=["linkcheck"],
        ),
        req_entry(
            2,
            "A check must time out after 5 seconds.",
            aspects=["timeout"],
            capabilities=["linkcheck"],
        ),
    ]
    return plan(entries, [capability_entry(cap, ["R-001", "R-002"])])


@pytest.fixture
def linkcheck(monkeypatch: pytest.MonkeyPatch) -> Capability:
    cap = fake_linkcheck()
    install_catalog(monkeypatch, cap, fake_max_clicks(), fake_export())
    return cap


def test_tasks_groups_and_critical_path_of_one_capability(linkcheck: Capability) -> None:
    result = linkcheck_plan(linkcheck)
    p = payload(result, Kind.TASK_PLAN)
    assert [(t["id"], t["kind"], t["group"], t["depends_on"]) for t in p["tasks"]] == [
        ("T-001", "implement", "linkcheck-guard", []),
        ("T-002", "implement", "linkcheck-checker", ["T-001"]),
        ("T-003", "implement", "linkcheck-report", []),
        ("T-004", "test", "tests", []),
        ("T-005", "doc", "docs", ["T-001", "T-002", "T-003", "T-004"]),
    ]
    assert [(g["id"], g["node_id"], g["task_ids"]) for g in p["groups"]] == [
        ("linkcheck-guard", "implement.linkcheck-guard", ["T-001"]),
        ("linkcheck-checker", "implement.linkcheck-checker", ["T-002"]),
        ("linkcheck-report", "implement.linkcheck-report", ["T-003"]),
    ]
    assert p["critical_path"] == ["T-001", "T-002", "T-005"]
    assert p["unsupported"] == []
    assert validate_payload(Kind.TASK_PLAN, p) == []
    assert all(TASK_ID_RE.match(t["id"]) and GROUP_ID_RE.match(t["group"]) for t in p["tasks"])


def test_implement_tasks_carry_the_recipe_scope_the_requirements_and_the_validated_params(
    linkcheck: Capability,
) -> None:
    p = payload(linkcheck_plan(linkcheck), Kind.TASK_PLAN)
    guard = p["tasks"][0]
    assert guard["capability"] == "linkcheck" and guard["requirement_ids"] == ["R-001", "R-002"]
    assert guard["write_scope"] == ["linkcheck/guard.py"] and guard["expected_paths"] == [
        "linkcheck/guard.py"
    ]
    assert guard["params"] == {
        "recipe_group": "guard",
        "recipe_params": {"timeout_seconds": 5, "max_retries": 3},
    }
    assert p["groups"][0]["params"] == {"timeout_seconds": 5, "max_retries": 3}
    test_task, doc_task = p["tasks"][3], p["tasks"][4]
    assert (test_task["requirement_ids"], test_task["write_scope"]) == (
        ["R-001", "R-002"],
        ["tests/**"],
    )
    assert (doc_task["capability"], doc_task["write_scope"]) == ("", ["docs/**", "CHANGELOG.md"])


def test_the_expansion_has_one_narrowed_implement_node_per_group_and_rewires_integrate(
    linkcheck: Capability,
) -> None:
    result = linkcheck_plan(linkcheck)
    expansion = result.plan_expansion
    assert expansion is not None and expansion.owner == "decompose"
    assert [n.id for n in expansion.nodes] == [
        "implement.linkcheck-guard",
        "implement.linkcheck-checker",
        "implement.linkcheck-report",
    ]
    guard, checker, report = expansion.nodes
    assert (guard.agent, guard.fallback_agent, guard.stage) == (
        "implementer",
        "implementer.safe",
        Stage.IMPLEMENTATION,
    )
    assert guard.depends_on == ("decompose",)
    assert checker.depends_on == ("decompose", "implement.linkcheck-guard")
    assert guard.produces == (Kind.CHANGESET,) and guard.consumes == (
        Kind.TASK_PLAN,
        Kind.DESIGN_DOC,
    )
    assert guard.autonomy is not None and guard.autonomy.write_globs == ("linkcheck/guard.py",)
    assert guard.autonomy.can_execute is False
    assert report.params == {
        "group": "report",  # the recipe's own group id: the implementer passes it to build_group
        "capability": "linkcheck",
        "recipe_params": {"timeout_seconds": 5, "max_retries": 3},
        "task_ids": ["T-003"],
        "write_scope": ["linkcheck/report.py"],
        "expected_paths": ["linkcheck/report.py"],
    }
    assert expansion.rewire == {
        INTEGRATE_NODE: Rewire(add_depends_on=tuple(n.id for n in expansion.nodes))
    }
    assert all(NODE_ID_RE.match(n.id) for n in expansion.nodes)


def test_capabilities_are_planned_in_id_order_with_continuous_task_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    link, clicks, export = fake_linkcheck(), fake_max_clicks(), fake_export()
    install_catalog(monkeypatch, link, clicks, export)
    entries = [
        req_entry(
            1, "Classify each result.", aspects=["classification"], capabilities=["linkcheck"]
        ),
        req_entry(
            2, "A link must stop.", aspects=["gone-after-limit"], capabilities=["max_clicks"]
        ),
        req_entry(
            3,
            "Export the clicks as CSV.",
            aspects=["csv-export"],
            capabilities=["analytics_export"],
        ),
    ]
    caps = [
        capability_entry(link, ["R-001"]),
        capability_entry(clicks, ["R-002"]),
        capability_entry(export, ["R-003"]),
    ]
    p = payload(plan(entries, caps), Kind.TASK_PLAN)
    implement = [t for t in p["tasks"] if t["kind"] == "implement"]
    assert [t["capability"] for t in implement] == ["analytics_export"] * 3 + ["linkcheck"] * 3 + [
        "max_clicks"
    ] * 3
    assert [t["id"] for t in p["tasks"]] == [f"T-{n:03d}" for n in range(1, 12)]
    # dependencies stay inside their capability and point at earlier tasks
    for task in implement:
        assert all(d < task["id"] for d in task["depends_on"])
        assert all(
            next(t for t in p["tasks"] if t["id"] == d)["capability"] == task["capability"]
            for d in task["depends_on"]
        )


def test_groups_are_emitted_in_dependency_order_whatever_the_recipe_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = fake_linkcheck()
    cap.recipe.group_specs = (  # type: ignore[attr-defined]
        GroupSpec("report", "Report", ("a/report.py",), ("a/report.py",), depends_on=("core",)),
        GroupSpec("extra", "Extra", ("a/extra.py",), ("a/extra.py",)),
        GroupSpec("core", "Core", ("a/core.py",), ("a/core.py",)),
    )
    install_catalog(monkeypatch, cap)
    p = payload(linkcheck_plan(cap), Kind.TASK_PLAN)
    assert [t["group"] for t in p["tasks"] if t["kind"] == "implement"] == [
        "linkcheck-extra",
        "linkcheck-core",
        "linkcheck-report",
    ]
    assert p["tasks"][2]["depends_on"] == ["T-002"]


def test_long_group_ids_are_shortened_to_a_valid_unique_deterministic_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap = fake_export()
    cap.recipe.group_specs = (  # type: ignore[attr-defined]
        GroupSpec("security-csv-encoding", "A", ("a/a.py",), ("a/a.py",)),
        GroupSpec("security-csv-formula", "B", ("a/b.py",), ("a/b.py",)),
    )
    install_catalog(monkeypatch, cap)
    entries = [
        req_entry(1, "Export CSV.", aspects=["csv-export"], capabilities=["analytics_export"])
    ]
    caps = [capability_entry(cap, ["R-001"])]
    first = payload(plan(entries, caps), Kind.TASK_PLAN)
    again = payload(plan(entries, caps), Kind.TASK_PLAN)
    ids = [g["id"] for g in first["groups"]]
    assert len(set(ids)) == 2 and ids == [g["id"] for g in again["groups"]]
    assert all(len(i) <= 31 and GROUP_ID_RE.match(i) for i in ids)
    assert all(NODE_ID_RE.match(g["node_id"]) for g in first["groups"])
    assert all(i.startswith("analytics_export-secur-") for i in ids)


def test_unsupported_requirements_are_reported_not_planned(linkcheck: Capability) -> None:
    entries = [
        req_entry(
            1, "Classify each result.", aspects=["classification"], capabilities=["linkcheck"]
        ),
        req_entry(2, "The system must send a weekly digest."),
    ]
    unsupported = [
        {
            "requirement_id": "R-002",
            "reason": "no capability in the catalog covers this requirement",
        }
    ]
    p = payload(
        plan(entries, [capability_entry(linkcheck, ["R-001"])], unsupported=unsupported),
        Kind.TASK_PLAN,
    )
    assert p["unsupported"] == unsupported
    assert all("R-002" not in t["requirement_ids"] for t in p["tasks"])
    assert [t["requirement_ids"] for t in p["tasks"] if t["kind"] == "test"] == [["R-001"]]


def test_a_plan_without_anything_to_implement_has_no_tasks_and_no_expansion(
    linkcheck: Capability,
) -> None:
    unsupported = [
        {
            "requirement_id": "R-001",
            "reason": "no capability in the catalog covers this requirement",
        }
    ]
    result = plan([req_entry(1, "The system must send a digest.")], unsupported=unsupported)
    p = payload(result, Kind.TASK_PLAN)
    assert (p["tasks"], p["groups"], p["critical_path"]) == ([], [], [])
    assert p["unsupported"] == unsupported
    assert result.plan_expansion is not None
    assert result.plan_expansion.nodes == () and result.plan_expansion.rewire == {}


def test_a_deferred_requirement_needs_no_task(linkcheck: Capability) -> None:
    entries = [
        req_entry(
            1, "Classify each result.", aspects=["classification"], capabilities=["linkcheck"]
        ),
        req_entry(2, "Maybe later.", status="deferred"),
    ]
    p = payload(
        plan(entries, [capability_entry(linkcheck, ["R-001"])], covered=["R-001"]), Kind.TASK_PLAN
    )
    assert all("R-002" not in t["requirement_ids"] for t in p["tasks"])


def _fatal(
    entries: Sequence[dict[str, Any]], caps: Sequence[dict[str, Any]], **kwargs: Any
) -> AgentError:
    with pytest.raises(AgentError) as caught:
        plan(entries, caps, **kwargs)
    assert caught.value.kind == "fatal" and caught.value.retryable is False
    return caught.value


def _with_groups(
    monkeypatch: pytest.MonkeyPatch, *groups: GroupSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cap = fake_linkcheck()
    cap.recipe.group_specs = tuple(groups)  # type: ignore[attr-defined]
    install_catalog(monkeypatch, cap)
    return (
        [
            req_entry(
                1, "Classify each result.", aspects=["classification"], capabilities=["linkcheck"]
            )
        ],
        [capability_entry(cap, ["R-001"])],
    )


def test_cyclic_group_dependencies_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    entries, caps = _with_groups(
        monkeypatch,
        GroupSpec("a", "A", ("x/a.py",), ("x/a.py",), depends_on=("b",)),
        GroupSpec("b", "B", ("x/b.py",), ("x/b.py",), depends_on=("a",)),
    )
    assert "cyclic" in str(_fatal(entries, caps))


def test_a_group_that_depends_on_itself_is_a_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    entries, caps = _with_groups(
        monkeypatch, GroupSpec("a", "A", ("x/a.py",), ("x/a.py",), depends_on=("a",))
    )
    assert "cyclic" in str(_fatal(entries, caps))


def test_unknown_dependencies_duplicate_ids_and_empty_group_lists_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries, caps = _with_groups(
        monkeypatch, GroupSpec("a", "A", ("x/a.py",), ("x/a.py",), depends_on=("zzz",))
    )
    assert "unknown group" in str(_fatal(entries, caps))
    entries, caps = _with_groups(
        monkeypatch,
        GroupSpec("a", "A", ("x/a.py",), ("x/a.py",)),
        GroupSpec("a", "B", ("x/b.py",), ("x/b.py",)),
    )
    assert "duplicate group ids" in str(_fatal(entries, caps))
    entries, caps = _with_groups(monkeypatch)
    assert "no groups" in str(_fatal(entries, caps))


def test_an_unknown_capability_and_hostile_parameters_are_rejected(linkcheck: Capability) -> None:
    entries = [req_entry(1, "Do it.")]
    assert "unknown capability" in str(
        _fatal(entries, [{"id": "nope", "title": "x", "requirement_ids": ["R-001"], "params": {}}])
    )
    for params in ({"max_retries": 999}, {"timeout_seconds": "5; rm"}, {"extra": 1}):
        caps = [{"id": "linkcheck", "title": "x", "requirement_ids": ["R-001"], "params": params}]
        with pytest.raises(RecipeParamError):
            plan(entries, caps)


def test_the_plan_must_follow_the_design(linkcheck: Capability) -> None:
    entries = [
        req_entry(
            1, "Classify each result.", aspects=["classification"], capabilities=["linkcheck"]
        ),
        req_entry(
            2,
            "A check must time out after 5 seconds.",
            aspects=["timeout"],
            capabilities=["linkcheck"],
        ),
    ]
    error = _fatal(entries, [capability_entry(linkcheck, ["R-001", "R-002"])], covered=["R-001"])
    assert "R-002" in str(error)


def test_the_node_count_is_checked_against_the_owner_expansion_policy(
    linkcheck: Capability,
) -> None:
    entries = [
        req_entry(
            1, "Classify each result.", aspects=["classification"], capabilities=["linkcheck"]
        )
    ]
    caps = [capability_entry(linkcheck, ["R-001"])]
    error = _fatal(entries, caps, node_params={"expansion_policy": {"max_nodes": 2}})
    assert "3 implementation nodes" in str(error)
    assert (
        plan(entries, caps, node_params={"expansion_policy": {"max_nodes": 3}}).plan_expansion
        is not None
    )
    assert (
        plan(entries, caps, node_params={"expansion_policy": {"max_nodes": True}}).plan_expansion
        is not None
    )
    assert plan(entries, caps).plan_expansion is not None


def test_the_planner_is_read_only_and_deterministic(linkcheck: Capability) -> None:
    agent = PlannerAgent()
    assert agent.autonomy.write_globs == () and agent.autonomy.read_globs == ()
    assert (agent.agent_id, agent.role) == ("planner", "planner")
    first, second = linkcheck_plan(linkcheck), linkcheck_plan(linkcheck)
    assert fingerprint(first) == fingerprint(second)
    assert first.usage["tool_calls"] == 0


# ----------------------------------------------------------------- real graph, policy and scenarios


def _overlaps(a: str, b: str) -> bool:
    if a == b or "**" in (a, b):
        return True
    return any(x.endswith("/**") and y.startswith(x[:-2]) for x, y in ((a, b), (b, a)))


def _check_plan_against_the_gates(artifacts: Mapping[str, Any], result: AgentResult) -> None:
    """The semantics of ``plan_covers_requirements`` and the integration contract."""
    task_plan = payload(result, Kind.TASK_PLAN)
    expansion = result.plan_expansion
    assert expansion is not None
    requirements = artifacts[Kind.REQUIREMENTS]
    active = {r["id"] for r in requirements["requirements"] if r["status"] == "active"}
    unsupported = {u["requirement_id"] for u in task_plan["unsupported"]}
    covered = {rid for t in task_plan["tasks"] for rid in t["requirement_ids"]}
    assert active <= covered | unsupported
    node_of = {g["id"]: g["node_id"] for g in task_plan["groups"]}
    node_ids = {n.id for n in expansion.nodes}
    for task in task_plan["tasks"]:
        if task["kind"] == "implement":
            assert node_of[task["group"]] in node_ids
    assert set(node_of.values()) == node_ids
    groups = task_plan["groups"]
    for i, first in enumerate(groups):
        for second in groups[i + 1 :]:
            if first["capability"] == second["capability"]:
                assert not any(
                    _overlaps(a, b) for a in first["write_scope"] for b in second["write_scope"]
                ), (
                    first["id"],
                    second["id"],
                )


def _apply_to_the_real_graph(result: AgentResult) -> None:
    """The engine path: raw expansion -> ``normalize_expansion`` -> graph -> policy."""
    graph = build_sdlc_graph()
    raw = result.plan_expansion
    assert raw is not None
    expansion, findings = normalize_expansion(graph.node("decompose"), raw, graph, EngineConfig())
    assert findings == [], [f.message for f in findings]
    assert [n.id for n in expansion.nodes] == [n.id for n in raw.nodes]
    assert all(
        n.params == raw_node.params for n, raw_node in zip(expansion.nodes, raw.nodes, strict=True)
    )
    before = graph.nodes
    graph.apply_expansion(expansion)
    graph.validate()
    after = graph.nodes
    assert set(expansion.rewire[INTEGRATE_NODE].add_depends_on) <= set(graph.deps(INTEGRATE_NODE))
    policy = expansion_policy()
    subject = PlanSubject(
        run_id="run-1",
        expansion=expansion,
        graph_before=before,
        graph_after=after,
        mandatory=tuple(i for i, n in after.items() if n.mandatory),
        allowed_agents=frozenset(policy["allowed_agents"]),
        allowed_produces=frozenset(policy["allowed_produces"]),
        allowed_stages=frozenset(policy["allowed_stages"]),
    )
    report = PolicyEngine().evaluate(subject)
    assert report.enforcement == Enforcement.ALLOW and report.blocking == ()


def test_the_expansion_of_the_fake_plan_is_accepted_by_the_real_graph_and_policy(
    linkcheck: Capability,
) -> None:
    graph = build_sdlc_graph()
    template_params = graph.node("decompose").params
    result = plan(
        [
            req_entry(
                1, "Classify each result.", aspects=["classification"], capabilities=["linkcheck"]
            )
        ],
        [capability_entry(linkcheck, ["R-001"])],
        node_params=template_params,
    )
    _apply_to_the_real_graph(result)


def test_a_realistic_expansion_is_not_rejected_by_sec_001_as_unscannable(
    linkcheck: Capability,
) -> None:
    """Regression: SEC-001 scanned the expansion as one JSON line, denying any plan of 3+ nodes."""
    result = linkcheck_plan(linkcheck)
    assert result.plan_expansion is not None
    assert (
        len(canonical_json(result.plan_expansion.to_dict())) > 2000
    )  # the size the line cap used to refuse
    graph = build_sdlc_graph()
    before = graph.nodes
    graph.apply_expansion(result.plan_expansion)
    policy = expansion_policy()
    subject = PlanSubject(
        "run-1",
        result.plan_expansion,
        before,
        graph.nodes,
        ("verify",),
        frozenset(policy["allowed_agents"]),
        frozenset(policy["allowed_produces"]),
        frozenset(policy["allowed_stages"]),
    )
    assert [f for f in PolicyEngine().evaluate(subject).findings if f.rule_id == "SEC-001"] == []


@pytest.fixture(params=MODES)
def env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, dict[str, str]]:
    return str(request.param), activate(str(request.param), monkeypatch)


def _plan_scenario(
    request_text: str, files: Mapping[str, str] | None, **chain_kwargs: Any
) -> tuple[Mapping[str, Any], AgentResult]:
    chain = run_chain(request_text, files, through="architect", **chain_kwargs)
    inputs = chain.node_inputs(Kind.REQUIREMENTS, Kind.DESIGN_DOC, Kind.IMPACT_REPORT)
    node_params = build_sdlc_graph().node("decompose").params
    result = run_agent(
        PlannerAgent(),
        files=files,
        artifacts=inputs,
        node_id="decompose",
        stage=Stage.PLANNING,
        node_params=node_params,
    )
    return chain.artifacts, result


def test_the_greenfield_scenario_plans_at_least_three_parallel_groups(
    env: tuple[str, dict[str, str]],
) -> None:
    mode, _ = env
    artifacts, result = _plan_scenario(scenario_request("greenfield"), None)
    _check_plan_against_the_gates(artifacts, result)
    _apply_to_the_real_graph(result)
    assert result.plan_expansion is not None
    assert len(result.plan_expansion.nodes) >= 3
    if mode == "real":
        assert len([n for n in result.plan_expansion.nodes if n.depends_on == ("decompose",)]) >= 3
    assert payload(result, Kind.TASK_PLAN)["unsupported"] == []


def test_the_brownfield_scenario_plans_disjoint_groups_for_the_click_limit(
    env: tuple[str, dict[str, str]],
) -> None:
    _, files = env
    artifacts, result = _plan_scenario(scenario_request("brownfield"), files)
    _check_plan_against_the_gates(artifacts, result)
    _apply_to_the_real_graph(result)
    plan_ = payload(result, Kind.TASK_PLAN)
    assert {g["capability"] for g in plan_["groups"]} == {"max_clicks"} and len(
        plan_["groups"]
    ) >= 3


def test_the_ambiguous_scenario_plans_the_export_after_the_clarification(
    env: tuple[str, dict[str, str]],
) -> None:
    _, files = env
    answers, decisions = ambiguous_inputs(files)
    artifacts, result = _plan_scenario(
        scenario_request("ambiguous"), files, answers=answers, decisions=decisions
    )
    _check_plan_against_the_gates(artifacts, result)
    _apply_to_the_real_graph(result)
    plan_ = payload(result, Kind.TASK_PLAN)
    assert {g["capability"] for g in plan_["groups"]} == {"analytics_export"} and len(
        plan_["groups"]
    ) >= 3
    deferred = [
        r["id"] for r in artifacts[Kind.REQUIREMENTS]["requirements"] if r["status"] == "deferred"
    ]
    assert deferred and all(
        rid not in t["requirement_ids"] for t in plan_["tasks"] for rid in deferred
    )


# ----------------------------------------------------------------------------- the real SDLC gates

SDLC_GATES: tuple[tuple[Any, str], ...] = (
    (AcceptanceCriteriaTestable, "requirements"),
    (AmbiguityDispositionsComplete, "requirements"),
    (DesignRisk, "design"),
    (DesignCoversRequirements, "design"),
    (PlanCoversRequirements, "decompose"),
)


def _gate(
    gate: Any, node_id: str, artifacts: Mapping[str, Any], decisions: Sequence[Decision] = ()
) -> Any:
    clock = FakeClock()
    reader = MemoryReader(artifacts, clock)
    ctx = GateContext(
        "run-1",
        build_sdlc_graph().node(node_id),
        GatePhase.EXIT,
        1,
        {},
        reader,
        (),
        reader,
        PolicyEngine(),
        MemoryView({}),
        lambda: MemoryView({}),
        FakeRunView(reader, decisions),
        clock,
    )
    return gate().evaluate(ctx)


def _gate_inputs(
    request_text: str, files: Mapping[str, str] | None, **chain_kwargs: Any
) -> dict[str, Any]:
    artifacts, result = _plan_scenario(request_text, files, **chain_kwargs)
    inputs = dict(artifacts)
    inputs[Kind.TASK_PLAN] = payload(result, Kind.TASK_PLAN)
    assert result.plan_expansion is not None
    inputs[Kind.PLAN_EXPANSION] = result.plan_expansion.to_dict()
    return inputs


@pytest.mark.parametrize(
    ("scenario", "level"), [("greenfield", "LOW"), ("brownfield", "HIGH"), ("ambiguous", "MEDIUM")]
)
def test_the_real_sdlc_gates_accept_the_output_of_the_whole_analysis_chain(
    env: tuple[str, dict[str, str]], scenario: str, level: str
) -> None:
    _, files = env
    decisions: tuple[Decision, ...] = ()
    chain_kwargs: dict[str, Any] = {}
    if scenario == "ambiguous":
        answers, decisions = ambiguous_inputs(files)
        chain_kwargs = {"answers": answers, "decisions": decisions}
    inputs = _gate_inputs(
        scenario_request(scenario), None if scenario == "greenfield" else files, **chain_kwargs
    )
    if scenario == "ambiguous":
        inputs[K_CLARIFICATION_ANSWERS] = chain_kwargs["answers"]
    for gate, node_id in SDLC_GATES:
        result = _gate(gate, node_id, inputs, decisions)
        assert result.decision == GateDecision.PASS, (gate.__name__, result.message)
        if gate is DesignRisk:
            assert result.risk is not None and result.risk.level.name == level


def test_an_unsupported_requirement_turns_the_plan_gate_into_a_clarification(
    env: tuple[str, dict[str, str]],
) -> None:
    request = (
        scenario_request("greenfield") + "\n- The system must send a weekly digest by e-mail.\n"
    )
    inputs = _gate_inputs(request, None)
    assert len(inputs[Kind.REQUIREMENTS]["unsupported"]) == 1
    for gate, node_id in (
        (AcceptanceCriteriaTestable, "requirements"),
        (DesignCoversRequirements, "design"),
    ):
        assert _gate(gate, node_id, inputs).decision == GateDecision.PASS
    result = _gate(PlanCoversRequirements, "decompose", inputs)
    assert result.decision == GateDecision.AWAIT_INPUT
    assert [q.category for q in result.questions] == ["unsupported"]
    assert len(payload_groups(inputs)) >= 3  # the supported part is still fully planned


def payload_groups(inputs: Mapping[str, Any]) -> list[Any]:
    return list(inputs[Kind.TASK_PLAN]["groups"])
