"""Governed normalisation of plan expansions: template overwrite, allow-lists, hostile input."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from orchestrator.core.errors import PlanPatchError
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.types import (
    ApprovalPosition,
    ApprovalSpec,
    AutonomyProfile,
    Finding,
    GateSpec,
    NodeSpec,
    PlanExpansion,
    PolicyCategory,
    RemediationSpec,
    RetryPolicy,
    Rewire,
    RiskLevel,
    Severity,
    Stage,
)
from orchestrator.engine.config import EngineConfig
from orchestrator.engine.expansion import (
    MANDATORY_EXIT_GATES,
    MAX_FINDINGS,
    normalize_expansion,
)
from orchestrator.workflows.sdlc import build_sdlc_graph

CONFIG = EngineConfig()


def spec(node_id: str, deps: Sequence[str] = (), **kwargs: Any) -> NodeSpec:
    fields: dict[str, Any] = {"title": node_id, "stage": Stage.IMPLEMENTATION, "agent": "agent"}
    fields.update(kwargs)
    return NodeSpec(id=node_id, depends_on=tuple(deps), **fields)


def template(**overrides: Any) -> NodeSpec:
    fields: dict[str, Any] = {
        "title": "implementation template",
        "stage": Stage.IMPLEMENTATION,
        "agent": "implementer",
        "fallback_agent": "implementer.safe",
        "depends_on": ("decompose",),
        "consumes": ("task_plan", "design_doc"),
        "produces": ("changeset",),
        "retry": RetryPolicy(max_attempts=3),
        "fallback_retry": RetryPolicy(max_attempts=2),
        "entry_gates": (GateSpec("inputs_ok"),),
        "exit_gates": (GateSpec("syntax_valid"), GateSpec("scope_within_plan")),
        "approval": ApprovalSpec(
            ApprovalPosition.AFTER, always=False, threshold=RiskLevel.MEDIUM, reason="review"
        ),
        "risk": RiskLevel.LOW,
        "autonomy": AutonomyProfile(write_globs=("src/**",), max_changed_files=40),
        "timeout_s": 120.0,
    }
    fields.update(overrides)
    return NodeSpec(id="implement.template", **fields)


def policy(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id_prefix": "implement.",
        "allowed_agents": ["implementer", "implementer.safe"],
        "allowed_stages": ["implementation"],
        "allowed_produces": ["changeset"],
        "rewire_targets": ["integrate"],
        "max_nodes": 5,
    }
    result.update(overrides)
    return result


def owner_params(tpl: Any = None, pol: Any = None) -> dict[str, Any]:
    return {
        "expansion_template": template().to_dict() if tpl is None else tpl,
        "expansion_policy": policy() if pol is None else pol,
    }


def build_graph(params: Mapping[str, Any] | None = None, **graph_kwargs: Any) -> WorkflowGraph:
    graph = WorkflowGraph(
        [
            spec("design", stage=Stage.DESIGN, produces=("design_doc",)),
            spec(
                "decompose",
                ["design"],
                consumes=("design_doc",),
                produces=("task_plan",),
                params=owner_params() if params is None else dict(params),
            ),
            spec("author-tests", ["decompose"], consumes=("task_plan",), produces=("changeset",)),
            spec(
                "implement.legacy", ["decompose"], consumes=("task_plan",), produces=("changeset",)
            ),
            spec(
                "implement.guard",
                ["decompose"],
                consumes=("task_plan",),
                produces=("changeset",),
                mandatory=True,
            ),
            spec(
                "integrate",
                ["decompose", "author-tests", "implement.legacy", "implement.guard"],
                consumes=("changeset", "task_plan"),
                produces=("integrated_changeset",),
            ),
            spec(
                "docs",
                ["integrate"],
                consumes=("integrated_changeset",),
                produces=("docs_changeset",),
            ),
            spec(
                "verify",
                ["integrate"],
                consumes=("integrated_changeset",),
                produces=("test_report",),
                mandatory=True,
            ),
        ],
        **graph_kwargs,
    )
    graph.validate()
    return graph


def proposal(name: str, **kwargs: Any) -> NodeSpec:
    fields: dict[str, Any] = {
        "title": f"implement {name}",
        "stage": Stage.IMPLEMENTATION,
        "agent": "implementer",
        "depends_on": ("decompose",),
        "params": {"group": name},
    }
    fields.update(kwargs)
    return NodeSpec(id=f"implement.{name}", **fields)


def expansion(*nodes: NodeSpec, rewire: Mapping[str, Rewire] | None = None) -> PlanExpansion:
    wired = (
        {"integrate": Rewire(add_depends_on=tuple(n.id for n in nodes))}
        if rewire is None
        else rewire
    )
    return PlanExpansion("decompose", tuple(nodes), dict(wired))


def normalise(
    exp: Any, graph: WorkflowGraph | None = None, config: EngineConfig = CONFIG
) -> tuple[PlanExpansion, list[Finding]]:
    graph = graph or build_graph()
    return normalize_expansion(graph.node("decompose"), exp, graph, config)


def rules(findings: Sequence[Finding]) -> set[str]:
    return {f.rule_id for f in findings}


def texts(findings: Sequence[Finding]) -> str:
    return " | ".join(f.message for f in findings)


def rejected(exp: Any, graph: WorkflowGraph | None = None) -> tuple[PlanExpansion, list[Finding]]:
    result, findings = normalise(exp, graph)
    assert findings, "expected the expansion to be rejected"
    assert all(f.rule_id in {"CHG-006", "CHG-007"} for f in findings)
    return result, findings


# =============================================================================
# the conformant case
# =============================================================================


def test_conformant_expansion_is_built_from_the_template_and_applies_cleanly() -> None:
    graph = build_graph()
    before = copy.deepcopy(graph.to_dict())
    exp = expansion(proposal("a"), proposal("b", depends_on=("decompose", "implement.a")))
    result, findings = normalise(exp, graph)
    assert findings == []
    assert result.owner == "decompose"
    assert [n.id for n in result.nodes] == ["implement.a", "implement.b"]
    tpl = template()
    for node in result.nodes:
        assert node.agent == tpl.agent and node.fallback_agent == tpl.fallback_agent
        assert (node.stage, node.consumes, node.produces) == (tpl.stage, tpl.consumes, tpl.produces)
        assert node.approval == tpl.approval and node.entry_gates == tpl.entry_gates
        assert (node.retry, node.fallback_retry, node.timeout_s) == (
            tpl.retry,
            tpl.fallback_retry,
            120.0,
        )
        assert not (node.mandatory or node.side_effecting or node.skip_if or node.compensator)
    assert result.nodes[1].depends_on == ("decompose", "implement.a")
    assert result.rewire == {"integrate": Rewire(add_depends_on=("implement.a", "implement.b"))}
    assert graph.to_dict() == before  # the live graph is only read
    diff = graph.copy().apply_expansion(result)  # what the engine will do at publish
    assert diff.added == ("implement.a", "implement.b") and diff.rewired == ("integrate",)


def test_governance_attributes_are_overwritten_not_trusted() -> None:
    hostile = RetryPolicy()
    object.__setattr__(hostile, "max_attempts", 10**6)  # bypasses RetryPolicy validation
    node = proposal(
        "a",
        approval=None,
        risk=RiskLevel.LOW,
        retry=hostile,
        fallback_retry=hostile,
        timeout_s=80000.0,
        entry_gates=(),
        exit_gates=(GateSpec("always_pass"),),
        remediation=RemediationSpec(route_to=("author-tests",)),
        cacheable=False,
        consumes=("secrets_dump",),
        optional_consumes=("threat_model",),
        overlay_from=("changeset",),
        skip_inputs=("request",),
    )
    result, findings = normalise(expansion(node))
    assert findings == []
    (built,) = result.nodes
    tpl = template()
    assert built.approval == tpl.approval
    assert built.retry.max_attempts == 3 and built.fallback_retry.max_attempts == 2
    assert built.timeout_s == 120.0
    assert built.entry_gates == tpl.entry_gates
    assert [g.type for g in built.exit_gates] == [
        "syntax_valid",
        "scope_within_plan",
        "artifacts_present",
    ]
    assert built.remediation is None and built.cacheable is True
    assert built.consumes == tpl.consumes and built.optional_consumes == ()
    assert built.overlay_from == () and built.skip_inputs == ()


def test_risk_can_be_raised_by_the_proposal_but_never_lowered() -> None:
    result, findings = normalise(expansion(proposal("a", risk=RiskLevel.HIGH)))
    assert findings == [] and result.nodes[0].risk == RiskLevel.HIGH
    custom = owner_params(template(risk=RiskLevel.MEDIUM).to_dict())
    result, findings = normalise(expansion(proposal("a", risk=RiskLevel.LOW)), build_graph(custom))
    assert findings == [] and result.nodes[0].risk == RiskLevel.MEDIUM


def test_the_owner_is_forced_into_depends_on_and_template_dependencies_come_first() -> None:
    result, findings = normalise(expansion(proposal("a", depends_on=())))
    assert findings == [] and result.nodes[0].depends_on == ("decompose",)
    tpl = template(depends_on=("design",)).to_dict()
    result, findings = normalise(
        expansion(proposal("a", depends_on=("decompose", "author-tests"))),
        build_graph(owner_params(tpl)),
    )
    assert findings == []
    assert result.nodes[0].depends_on == ("design", "decompose", "author-tests")


def test_retry_and_timeout_of_the_template_are_clamped_to_the_hard_bounds() -> None:
    wild = template().to_dict()
    wild["retry"] = {"max_attempts": 99, "factor": 100, "base_delay_s": -4, "max_delay_s": 1e6}
    wild["fallback_retry"] = {"max_attempts": 0}
    wild["timeout_s"] = 99999
    graph = build_graph(owner_params(wild))
    result, findings = normalise(expansion(proposal("a")), graph, EngineConfig(max_attempt_s=600.0))
    assert findings == []
    (node,) = result.nodes
    assert node.retry.max_attempts == RetryPolicy.HARD_MAX_ATTEMPTS
    assert (node.retry.factor, node.retry.base_delay_s, node.retry.max_delay_s) == (4.0, 0.0, 60.0)
    assert node.fallback_retry.max_attempts == 1
    assert node.timeout_s == 600.0
    result, _ = normalise(expansion(proposal("a")), graph, EngineConfig(max_attempt_s=30.0))
    assert result.nodes[0].timeout_s == 30.0  # the process config can be stricter than the template


@pytest.mark.parametrize(
    "broken",
    [
        {"timeout_s": -5},
        {"timeout_s": "soon"},
        {"timeout_s": True},
        {"retry": "often"},
        {"retry": {"max_attempts": "many"}},
        {"retry": {"factor": None}},
        {"stage": "alchemy"},
        {"agent": ""},
        {"approval": {"position": "sideways"}},
    ],
    ids=lambda b: next(iter(b)) + "=" + str(next(iter(b.values())))[:12],
)
def test_an_unusable_template_makes_the_owner_unable_to_expand(broken: dict[str, Any]) -> None:
    wild = {**template().to_dict(), **broken}
    result, findings = rejected(expansion(proposal("a")), build_graph(owner_params(wild)))
    assert [f.rule_id for f in findings] == ["CHG-007"] and "may not expand" in findings[0].message
    assert result.nodes == () and result.rewire == {}


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"expansion_template": template().to_dict()},  # no policy: nothing is allowed
        {"expansion_policy": policy()},  # no template: nothing to build from
        {"expansion_template": "text", "expansion_policy": policy()},
        {"expansion_template": [1], "expansion_policy": policy()},
        {"expansion_template": {}, "expansion_policy": policy()},
        owner_params(pol=["implement."]),
        owner_params(pol=policy(allowed_agents="implementer")),
        owner_params(pol=policy(allowed_stages=[1])),
        owner_params(pol=policy(max_nodes=-1)),
        owner_params(pol=policy(max_nodes=True)),
        owner_params(pol=policy(max_nodes="8")),
        owner_params(pol=policy(id_prefix=7)),
        owner_params(pol=policy(rewire_targets=["*"])),
        owner_params(pol=policy(rewire_targets=["a*b", "x.*.y"])),
    ],
)
def test_an_owner_without_a_valid_template_and_policy_cannot_expand(params: dict[str, Any]) -> None:
    result, findings = rejected(expansion(proposal("a")), build_graph(params))
    assert "may not expand" in texts(findings) and result.nodes == ()


# =============================================================================
# E0 / E2: what the proposal itself may not do
# =============================================================================


def test_expansion_must_be_proposed_by_its_owner() -> None:
    exp = PlanExpansion("verify", (proposal("a"),), {})
    _, findings = rejected(exp)
    assert "is not the producing node" in texts(findings)


def test_owner_must_be_a_node_of_the_graph() -> None:
    graph = build_graph()
    stranger = spec("stranger", params=owner_params())
    result, findings = normalize_expansion(
        stranger, PlanExpansion("stranger", (proposal("a"),), {}), graph, CONFIG
    )
    assert "not a node of the plan" in texts(findings) and result.nodes == ()


@pytest.mark.parametrize("field", ["agent", "fallback_agent"])
def test_a_privileged_agent_is_rejected(field: str) -> None:
    bad = proposal("evil", **{field: "promoter"})
    good = proposal("fine")
    result, findings = rejected(expansion(good, bad))
    assert rules(findings) == {"CHG-007"}
    assert "agent 'promoter' is not allowed" in texts(findings)
    assert [n.id for n in result.nodes] == ["implement.fine"]  # only conformant nodes are returned
    assert {f.location for f in findings} == {"implement.evil"}


def test_foreign_stage_and_foreign_produces_are_rejected() -> None:
    _, findings = rejected(expansion(proposal("a", stage=Stage.RELEASE)))
    assert "stage 'release' is not allowed" in texts(findings)
    _, findings = rejected(expansion(proposal("a", produces=("changeset", "release_patch"))))
    assert "produces 'release_patch'" in texts(findings) and len(findings) == 1


@pytest.mark.parametrize(
    "privilege",
    [
        {"mandatory": True},
        {"side_effecting": True, "compensator": "undo"},
        {"skip_if": "greenfield"},
        {"compensates": ("author-tests",)},
    ],
    ids=lambda p: next(iter(p)),
)
def test_privileged_flags_are_rejected_and_do_not_survive(privilege: dict[str, Any]) -> None:
    result, findings = rejected(expansion(proposal("a", **privilege)))
    assert "sets governance attribute" in texts(findings)
    assert result.nodes == ()


def test_node_ids_must_follow_the_prefix_and_be_unique() -> None:
    outside = spec("docs2", ["decompose"], produces=("changeset",), agent="implementer")
    _, findings = rejected(expansion(outside, rewire={}))
    assert "outside the id prefix" in texts(findings)
    prefix_only = NodeSpec("implement.x", "x", Stage.IMPLEMENTATION, "implementer")
    assert normalise(expansion(prefix_only, rewire={}))[1] == []
    twice = expansion(proposal("a"), proposal("a"), rewire={})
    result, findings = rejected(twice)
    assert "used twice" in texts(findings) and [n.id for n in result.nodes] == ["implement.a"]


def test_a_node_may_not_claim_an_existing_node_nor_replace_a_mandatory_one() -> None:
    _, findings = rejected(expansion(proposal("legacy"), rewire={}))
    assert rules(findings) == {"CHG-007"} and "clashes with an existing node" in texts(findings)
    result, findings = rejected(expansion(proposal("guard"), proposal("fine"), rewire={}))
    assert rules(findings) == {"CHG-006"} and "replace a mandatory node" in texts(findings)
    assert [n.id for n in result.nodes] == ["implement.fine"]


def test_previous_nodes_of_the_same_owner_may_be_redeclared_but_not_those_of_others() -> None:
    graph = build_graph()
    first, findings = normalise(expansion(proposal("a"), proposal("b")), graph)
    assert findings == []
    graph.apply_expansion(first)
    second, findings = normalise(expansion(proposal("b"), proposal("c")), graph)
    assert findings == []
    diff = graph.copy().apply_expansion(second)
    assert (diff.added, diff.removed, diff.retained) == (
        ("implement.c",),
        ("implement.a",),
        ("implement.b",),
    )
    other = spec("planner2", ["design"], params=owner_params())
    graph.add_node(other)
    graph.apply_expansion(
        PlanExpansion(
            "planner2", (spec("implement.p2", ["planner2"], produces=("changeset",)),), {}
        )
    )
    _, findings = rejected(expansion(proposal("p2"), rewire={}), graph)
    assert "clashes with an existing node" in texts(findings)


def test_dependencies_must_exist() -> None:
    ok = proposal("b", depends_on=("decompose", "implement.a", "design"))
    result, findings = normalise(expansion(proposal("a"), ok))
    assert findings == [] and result.nodes[1].depends_on == ("decompose", "implement.a", "design")
    for bad in ("ghost", "implement.b"):  # unknown node / itself
        _, findings = rejected(expansion(proposal("b", depends_on=("decompose", bad)), rewire={}))
        assert "invalid dependency" in texts(findings)


# =============================================================================
# rewires
# =============================================================================


def test_rewire_targets_must_be_allowed_by_the_owner_policy() -> None:
    _, findings = rejected(
        expansion(proposal("a"), rewire={"docs": Rewire(add_depends_on=("implement.a",))})
    )
    assert rules(findings) == {"CHG-007"} and "outside the allowed targets" in texts(findings)


def test_rewiring_a_mandatory_node_violates_change_control_even_if_the_policy_lists_it() -> None:
    graph = build_graph(owner_params(pol=policy(rewire_targets=["integrate", "verify"])))
    result, findings = rejected(
        expansion(proposal("a"), rewire={"verify": Rewire(add_depends_on=("implement.a",))}), graph
    )
    assert rules(findings) == {"CHG-006"} and "rewires a mandatory node" in texts(findings)
    assert result.rewire == {}


def test_rewire_may_not_touch_the_owner_or_its_own_nodes_or_unknown_nodes() -> None:
    graph = build_graph(
        owner_params(pol=policy(rewire_targets=["decompose", "implement.*", "ghost"]))
    )
    for target, expected in (
        ("decompose", "own owner"),
        ("implement.a", "own nodes"),
        ("ghost", "does not exist"),
    ):
        _, findings = rejected(expansion(proposal("a"), rewire={target: Rewire()}), graph)
        assert expected in texts(findings)


def test_rewire_edges_are_validated() -> None:
    bad_dep = Rewire(add_depends_on=("ghost",))
    self_dep = Rewire(add_depends_on=("integrate",))
    bad_overlay = Rewire(add_overlay_from=("design_doc",))
    for rewire, expected in (
        (bad_dep, "invalid dependency"),
        (self_dep, "invalid dependency"),
        (bad_overlay, "non-change-set kind"),
    ):
        _, findings = rejected(expansion(proposal("a"), rewire={"integrate": rewire}))
        assert expected in texts(findings)
    ok = Rewire(add_depends_on=("author-tests",), add_overlay_from=("changeset",))
    result, findings = normalise(expansion(proposal("a"), rewire={"integrate": ok}))
    assert findings == [] and result.rewire == {"integrate": ok}


def test_conformant_rewires_are_deduplicated_and_prefix_patterns_work() -> None:
    graph = build_graph(owner_params(pol=policy(rewire_targets=["do.*", "docs", "integrate"])))
    wired = {
        "integrate": Rewire(
            add_depends_on=("implement.a", "implement.a", "author-tests"),
            add_overlay_from=("changeset", "changeset"),
        )
    }
    result, findings = normalise(expansion(proposal("a"), rewire=wired), graph)
    assert findings == []
    assert result.rewire["integrate"] == Rewire(("implement.a", "author-tests"), ("changeset",))


def test_malformed_rewire_entries_are_findings_not_crashes() -> None:
    for rewire in ({5: Rewire()}, {"integrate": "later"}, {"Bad Id": Rewire()}, [("integrate", 1)]):
        _, findings = rejected(PlanExpansion("decompose", (proposal("a"),), rewire))  # type: ignore[arg-type]
        assert findings
    huge = {f"node{i}": Rewire() for i in range(300)}
    _, findings = rejected(expansion(proposal("a"), rewire=huge))
    assert "rewires more than" in texts(findings)
    wide = Rewire(add_depends_on=tuple(f"n{i}" for i in range(2000)))
    _, findings = rejected(expansion(proposal("a"), rewire={"integrate": wide}))
    assert "short lists" in texts(findings)


# =============================================================================
# limits and shape
# =============================================================================


def test_expansion_size_is_limited_by_the_owner_policy() -> None:
    graph = build_graph(owner_params(pol=policy(max_nodes=2)))
    nodes = [proposal(c) for c in "abc"]
    result, findings = rejected(expansion(*nodes), graph)
    assert "more than 2 node(s)" in texts(findings) and result.nodes == ()
    assert normalise(expansion(*nodes[:2]), graph)[1] == []
    no_limit = build_graph(
        owner_params(pol={k: v for k, v in policy().items() if k != "max_nodes"})
    )
    assert (
        rejected(expansion(proposal("a")), no_limit)[0].nodes == ()
    )  # a missing limit allows none
    assert normalise(expansion(rewire={}), no_limit)[1] == []  # ... but an empty plan is fine


def test_an_endless_node_stream_is_cut_off() -> None:
    endless = PlanExpansion("decompose", itertools.repeat(proposal("a")), {})  # type: ignore[arg-type]
    result, findings = rejected(endless)
    assert "more than 5 node(s)" in texts(findings) and result.nodes == ()


def test_findings_are_capped() -> None:
    graph = build_graph(owner_params(pol=policy(max_nodes=200)))
    nodes = [
        proposal(f"n{i}", agent="promoter", stage=Stage.RELEASE, produces=("release_patch",))
        for i in range(60)
    ]
    _, findings = rejected(expansion(*nodes, rewire={}), graph)
    assert len(findings) == MAX_FINDINGS
    assert "further problem(s) omitted" in findings[-1].message


def test_findings_are_high_change_control_findings_with_clean_text() -> None:
    nasty = proposal("a", agent="pro\x1b[31mmoter\n", produces=("changeset", "x" * 40))
    _, findings = rejected(expansion(nasty, proposal("b", mandatory=True)))
    assert len(findings) == len(set(findings))
    for finding in findings:
        assert (
            finding.severity == Severity.HIGH and finding.category == PolicyCategory.CHANGE_CONTROL
        )
        assert finding.remediation
        text = finding.message + (finding.location or "") + (finding.evidence or "")
        assert not any(ord(ch) < 32 for ch in text)
    assert len({f.key() for f in findings}) == len(findings)  # distinct problems, distinct keys


# =============================================================================
# params, titles, autonomy, gates
# =============================================================================


def test_agent_params_cannot_carry_expansion_rights_but_the_template_may() -> None:
    sneaky = proposal(
        "a", params={"group": "a", "expansion_template": {"x": 1}, "expansion_policy": {}}
    )
    _, findings = rejected(expansion(sneaky))
    assert len(findings) == 2 and "only the owner's author may set" in findings[0].message
    carrying = template(params={"expansion_policy": {"max_nodes": 1}, "budget": 5}).to_dict()
    graph = build_graph(owner_params(carrying))
    result, findings = normalise(
        expansion(proposal("a", params={"group": "a", "budget": 99})), graph
    )
    assert findings == []
    assert result.nodes[0].params == {
        "group": "a",
        "budget": 5,
        "expansion_policy": {"max_nodes": 1},
    }
    again, findings = normalise(result, graph)  # the template's own rights are not "smuggled"
    assert findings == [] and again == result


def test_title_is_sanitised_and_falls_back_to_the_id() -> None:
    result, _ = normalise(expansion(proposal("a", title="\x1b[31mred\nteam\x07")))
    assert result.nodes[0].title == "redteam"
    result, _ = normalise(
        expansion(proposal("a", title=" \x1b[0m "), proposal("b", title="x" * 500))
    )
    assert result.nodes[0].title == "implement.a"
    assert len(result.nodes[1].title) <= 120


def test_autonomy_is_the_template_narrowed_by_the_node() -> None:
    wide = AutonomyProfile(
        write_globs=("src/a.py", "docs/**"), max_changed_files=5, can_execute=True
    )
    result, findings = normalise(expansion(proposal("a", autonomy=wide), proposal("b")))
    assert findings == []
    narrowed, inherited = result.nodes[0].autonomy, result.nodes[1].autonomy
    assert narrowed is not None and inherited is not None
    assert narrowed.write_globs == ("src/a.py",)  # "docs/**" is outside the template's scope
    assert narrowed.max_changed_files == 5 and narrowed.can_execute is False
    assert inherited == template().autonomy


def test_a_scope_request_outside_the_template_is_rejected() -> None:
    outside = AutonomyProfile(write_globs=("docs/**",))
    _, findings = rejected(expansion(proposal("a", autonomy=outside)))
    assert "write scope is outside" in texts(findings)
    read_outside = AutonomyProfile(read_globs=("secrets/**",))
    read_only_template = owner_params(
        template(autonomy=AutonomyProfile(read_globs=("src/**",))).to_dict()
    )
    _, findings = rejected(
        expansion(proposal("a", autonomy=read_outside)), build_graph(read_only_template)
    )
    assert "read scope is outside" in texts(findings)


def test_without_template_autonomy_the_nodes_own_profile_applies() -> None:
    graph = build_graph(owner_params(template(autonomy=None).to_dict()))
    own = AutonomyProfile(write_globs=("anything/**",))
    result, findings = normalise(expansion(proposal("a", autonomy=own), proposal("b")), graph)
    assert findings == [] and result.nodes[0].autonomy == own and result.nodes[1].autonomy is None


def test_mandatory_exit_gates_are_added_exactly_once() -> None:
    assert [g.type for g in MANDATORY_EXIT_GATES] == ["artifacts_present"]
    plain = owner_params(template(exit_gates=()).to_dict())
    result, _ = normalise(expansion(proposal("a")), build_graph(plain))
    assert [g.type for g in result.nodes[0].exit_gates] == ["artifacts_present"]
    already = owner_params(template(exit_gates=(GateSpec("artifacts_present"),)).to_dict())
    result, _ = normalise(expansion(proposal("a")), build_graph(already))
    assert [g.type for g in result.nodes[0].exit_gates] == ["artifacts_present"]


# =============================================================================
# trial application, hostile shapes, determinism
# =============================================================================


def test_a_plan_that_cannot_be_applied_is_rejected_before_publish() -> None:
    cyclic = proposal("a", depends_on=("decompose", "verify"))  # verify depends on integrate
    _, findings = rejected(expansion(cyclic))
    assert rules(findings) == {"CHG-007"} and "cannot be applied" in texts(findings)
    assert "cycle" in texts(findings)
    small = build_graph(max_nodes=8)  # exactly full
    _, findings = rejected(expansion(proposal("a")), small)
    assert "cannot be applied" in texts(findings) and "max_nodes" in texts(findings)


def test_a_trial_application_that_would_alter_a_mandatory_node_is_a_change_control_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(self: WorkflowGraph, expansion: PlanExpansion) -> Any:
        raise PlanPatchError("mandatory node 'verify' may not be removed or altered")

    monkeypatch.setattr(WorkflowGraph, "apply_expansion", refuse)
    _, findings = rejected(expansion(proposal("a")))
    assert rules(findings) == {"CHG-006"} and "cannot be applied" in texts(findings)


@pytest.mark.parametrize(
    "hostile",
    [
        {"owner": "decompose", "nodes": []},
        None,
        "implement.a",
        PlanExpansion("decompose", 123, {}),  # type: ignore[arg-type]
        PlanExpansion("decompose", (object(), None, "node"), {}),  # type: ignore[arg-type]
        PlanExpansion("decompose", (), None),  # type: ignore[arg-type]
        PlanExpansion(None, (), {}),  # type: ignore[arg-type]
    ],
    ids=["dict", "none", "str", "int-nodes", "junk-nodes", "none-rewire", "none-owner"],
)
def test_hostile_shapes_never_raise(hostile: Any) -> None:
    result, findings = normalise(hostile)
    assert isinstance(result, PlanExpansion) and result.owner == "decompose"
    assert findings and all(f.severity == Severity.HIGH for f in findings)


def test_a_poisoned_node_object_is_a_finding_not_an_exception() -> None:
    poisoned = proposal("a")
    object.__setattr__(poisoned, "depends_on", 5)
    result, findings = normalise(expansion(poisoned))
    assert "could not be processed" in texts(findings) and result.nodes == ()
    odd = proposal("b")
    object.__setattr__(odd, "stage", "implementation")  # a bare string where an enum belongs
    assert normalise(expansion(odd))[1] == []


def test_normalisation_is_deterministic_idempotent_and_does_not_touch_the_graph() -> None:
    graph = build_graph()
    before = copy.deepcopy(graph.to_dict())
    exp = expansion(proposal("b"), proposal("a", depends_on=("decompose", "implement.b")))
    first, findings = normalise(exp, graph)
    second, _ = normalise(exp, graph)
    assert findings == [] and first == second
    again, findings = normalise(first, graph)
    assert (
        findings == []
        and again == first
        and [n.spec_hash() for n in again.nodes] == [n.spec_hash() for n in first.nodes]
    )
    normalise(
        expansion(proposal("x", agent="promoter")), graph
    )  # a rejected one leaves it alone too
    assert graph.to_dict() == before


def test_the_real_sdlc_template_and_policy_accept_a_planner_style_expansion() -> None:
    graph = build_sdlc_graph()
    owner = graph.node("decompose")
    nodes = tuple(
        NodeSpec(
            id=f"implement.{name}",
            title=f"implement {name}",
            stage=Stage.IMPLEMENTATION,
            agent="implementer",
            depends_on=("decompose",),
            params={"group": name, "write_scope": [f"shortener/{name}.py"]},
            autonomy=AutonomyProfile(write_globs=(f"shortener/{name}.py",)),
        )
        for name in ("linkcheck-core", "linkcheck-api", "linkcheck-docs")
    )
    exp = PlanExpansion(
        "decompose", nodes, {"integrate": Rewire(add_depends_on=tuple(n.id for n in nodes))}
    )
    result, findings = normalize_expansion(owner, exp, graph, CONFIG)
    assert findings == []
    assert {n.agent for n in result.nodes} == {"implementer"}
    assert all(
        n.approval is not None and n.fallback_agent == "implementer.safe" for n in result.nodes
    )
    assert (
        PlanExpansion.from_dict(result.to_dict()) == result
    )  # survives the plan_expansion artifact
    live = graph.copy()
    live.apply_expansion(result)
    live.validate()
    assert normalize_expansion(owner, result, live, CONFIG)[1] == []  # re-normalising is a no-op
    evil = PlanExpansion(
        "decompose", (NodeSpec("implement.x", "x", Stage.RELEASE, "promoter"),), {}
    )
    assert normalize_expansion(owner, evil, graph, CONFIG)[1]
    assert normalize_expansion(
        owner, PlanExpansion("decompose", nodes, {"verify": Rewire()}), graph, CONFIG
    )[1]
