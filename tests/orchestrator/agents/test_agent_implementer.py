"""Tests for the implementer agents: recipe-driven change-set proposals through the toolbox."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.implementer import (
    GroupAssignment,
    ImplementerAgent,
    SafeImplementerAgent,
)
from orchestrator.agents.recipes.base import RecipeAnchorError, RecipeParamError
from orchestrator.core.errors import AgentCancelled, AgentError, PathViolation, ToolDenied
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.paths import NODE_ID_RE, match_any
from orchestrator.core.schemas import Kind
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import (
    Agent,
    AutonomyProfile,
    CancelToken,
    DecisionKind,
    EventType,
    FileOpKind,
)
from orchestrator.tools.changeset import apply_changeset
from orchestrator.tools.codebase import CodebaseModel
from tests.orchestrator.agents.support_impl_chain import (
    LINKCHECK_POLICY,
    FakeRecipe,
    Rig,
    changeset_of,
    content_of,
    default_params,
    design_doc_payload,
    evaluate,
    fake_capability,
    feedback_finding,
    group_node,
    lookup,
    plan_group,
    plan_task,
    product_files,
    real_capability,
    run_group,
    task_plan_payload,
)

NODE = "implement.fake-alpha"


def assignment(group: str = "alpha", **recipe_params: Any) -> dict[str, Any]:
    return {
        "group": group,
        "capability": "fake",
        "recipe_params": recipe_params,
        "write_scope": [f"pkg/{group}/**"],
    }


def setup(
    tmp_path: Path,
    recipe: FakeRecipe | None = None,
    *,
    files: dict[str, str] | None = None,
    agent_class: type[ImplementerAgent] = ImplementerAgent,
    requires_baseline: bool = False,
    **rig_kwargs: Any,
) -> tuple[Rig, ImplementerAgent, FakeRecipe]:
    recipe = recipe or FakeRecipe()
    agent = agent_class(lookup(fake_capability(recipe, requires_baseline=requires_baseline)))
    return Rig(tmp_path, files or {}, **rig_kwargs), agent, recipe


# ----------------------------------------------------------------------------- the happy path


def test_builds_the_requested_group_as_a_changeset(tmp_path: Path) -> None:
    rig, agent, recipe = setup(tmp_path)
    result = rig.run(agent, "implement", NODE, params=assignment(limit=4))
    cs = changeset_of(result)
    assert [d.kind for d in result.artifacts] == [Kind.CHANGESET]
    assert content_of(cs) == {
        "pkg/alpha/__init__.py": "",
        "pkg/alpha/core.py": "LIMIT = 4\n\n\ndef limit() -> int:\n    return LIMIT\n",
        "pkg/alpha/fast_path.py": "FAST = True\n",
    }
    assert cs.author == "implementer"
    assert cs.description == "implement fake/alpha: alpha group"
    assert [(g, p) for g, _, p in recipe.group_calls] == [("alpha", {"limit": 4})]


def test_recipe_defaults_apply_and_other_groups_stay_untouched(tmp_path: Path) -> None:
    rig, agent, recipe = setup(tmp_path)
    result = rig.run(agent, "implement", "implement.fake-beta", params=assignment("beta"))
    assert {p.split("/")[1] for p in changeset_of(result).paths()} == {"beta"}
    assert recipe.group_calls[0][2] == {"limit": 3}


def test_the_decision_cites_the_plan_and_records_the_files(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    plan = task_plan_payload(
        [plan_group("alpha", task_ids=("T-001", "T-002"))],
        [plan_task("T-001", "alpha", requirement_ids=("R-002",)), plan_task("T-002", "alpha")],
    )
    rig.publish(Kind.TASK_PLAN, plan)
    result = rig.run(agent, "implement", NODE, params=assignment())
    (decision,) = result.decisions
    assert decision.kind is DecisionKind.DESIGN
    assert decision.refs == ("T-001", "T-002", "R-001", "R-002")
    assert decision.data["paths"] == list(changeset_of(result).paths())
    assert decision.data["conservative"] is False
    assert decision.outcome == "3 file(s) proposed"


def test_usage_reports_the_toolbox_calls(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    result = rig.run(agent, "implement", NODE, params=assignment())
    assert result.usage["tool_calls"] >= 3
    assert result.usage["bytes_written"] > 0


def test_the_output_is_deterministic_across_runs_and_clocks(tmp_path: Path) -> None:
    first = setup(tmp_path / "a", clock_start=datetime(2026, 1, 1, tzinfo=UTC))
    second = setup(tmp_path / "b", clock_start=datetime(2031, 6, 7, 8, 9, tzinfo=UTC))
    one = first[0].run(first[1], "implement", NODE, params=assignment(limit=2))
    two = second[0].run(second[1], "implement", NODE, params=assignment(limit=2))
    assert hash_jsonable([d.content for d in one.artifacts]) == hash_jsonable(
        [d.content for d in two.artifacts]
    )
    assert one.decisions == two.decisions


def test_the_result_passes_the_real_policy_engine(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    ctx = rig.context(agent, "implement", NODE, params=assignment())
    cs = changeset_of(agent.run(ctx))
    profile = agent.autonomy.narrow(ctx.node.autonomy)
    report = evaluate(cs, rig.baseline, node_id=NODE, profile=profile)
    assert not report.blocking and not report.needs_approval, report.findings


# ----------------------------------------------------------------------------- the assignment


@pytest.mark.parametrize(
    "bad",
    [
        {"limit": 0},
        {"limit": 11},
        {"limit": "3"},
        {"limit": True},
        {"limit": 2.5},
        {"limit": None},
        {"nope": 1},
        {"limit": 3, "extra": "x"},
    ],
)
def test_hostile_recipe_params_are_rejected_before_the_recipe_runs(
    tmp_path: Path, bad: dict[str, Any]
) -> None:
    rig, agent, recipe = setup(tmp_path)
    with pytest.raises(RecipeParamError) as err:
        rig.run(agent, "implement", NODE, params=assignment(**bad))
    assert err.value.kind == "fatal" and not err.value.retryable
    assert recipe.group_calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        {"group": None},
        {"group": "Bad Group"},
        {"group": 5},
        {"group": "x" * 40},
        {"capability": None},
        {"capability": "Fake Cap"},
        {"capability": 7},
        {"recipe_params": []},
        {"recipe_params": None},
        {"write_scope": "pkg/**"},
        {"write_scope": [""]},
        {"write_scope": [1]},
        {"write_scope": ["pkg/**"] * 51},
    ],
)
def test_malformed_assignments_are_fatal(tmp_path: Path, mutation: dict[str, Any]) -> None:
    rig, agent, recipe = setup(tmp_path)
    params = {**assignment(), **mutation}
    params = {k: v for k, v in params.items() if v is not None}
    with pytest.raises(AgentError) as err:
        rig.run(agent, "implement", NODE, params=params)
    assert err.value.kind == "fatal" and not err.value.retryable
    assert recipe.group_calls == []


def test_an_unknown_group_is_fatal_and_never_reaches_the_recipe(tmp_path: Path) -> None:
    rig, agent, recipe = setup(tmp_path)
    with pytest.raises(AgentError, match="has no group 'delta'") as err:
        rig.run(agent, "implement", NODE, params=assignment("delta"))
    assert err.value.kind == "fatal" and not err.value.retryable
    assert recipe.group_calls == []


def test_an_unknown_capability_is_fatal(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    params = {**assignment(), "capability": "other"}
    with pytest.raises(AgentError, match="unknown capability 'other'") as err:
        rig.run(agent, "implement", NODE, params=params)
    assert err.value.kind == "fatal"


def test_the_default_lookup_is_the_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def catalog(capability_id: str) -> Any:
        asked.append(capability_id)
        raise KeyError(capability_id)

    monkeypatch.setattr("orchestrator.agents.implementer.get_capability", catalog)
    params = {**assignment(), "capability": "no_such_capability"}
    with pytest.raises(AgentError, match="unknown capability") as err:
        Rig(tmp_path).run(ImplementerAgent(), "implement", NODE, params=params)
    assert err.value.kind == "fatal" and asked == ["no_such_capability"]


def test_the_task_plan_fills_in_missing_node_params(tmp_path: Path) -> None:
    rig, agent, recipe = setup(tmp_path)
    plan = task_plan_payload(
        [plan_group("beta", params={"limit": 6}, task_ids=("T-002",))],
        [plan_task("T-002", "beta", requirement_ids=("R-001", "R-002"))],
    )
    rig.publish(Kind.TASK_PLAN, plan)
    result = rig.run(agent, "implement", "implement.fake-beta", params={})
    assert "LIMIT = 6" in str(content_of(changeset_of(result))["pkg/beta/core.py"])
    assert result.decisions[0].refs == ("T-002", "R-001", "R-002")
    assert recipe.group_calls[0][0] == "beta"


def test_node_params_win_over_the_task_plan(tmp_path: Path) -> None:
    rig, agent, recipe = setup(tmp_path)
    rig.publish(Kind.TASK_PLAN, task_plan_payload([plan_group("alpha", params={"limit": 9})]))
    rig.run(agent, "implement", NODE, params=assignment(limit=2))
    assert recipe.group_calls[0][2] == {"limit": 2}


def test_the_recipe_group_comes_from_the_plan_tasks_not_from_the_plan_group_id(
    tmp_path: Path,
) -> None:
    rig, agent, recipe = setup(tmp_path)
    plan = task_plan_payload(
        [plan_group("beta", params={"limit": 5}, task_ids=("T-002",))],
        [plan_task("T-002", "beta")],
    )
    assert plan["groups"][0]["id"] == "fake-beta"  # the plan's id names a node, not a recipe group
    rig.publish(Kind.TASK_PLAN, plan)
    rig.run(agent, "implement", "implement.fake-beta", params={})
    assert recipe.group_calls[0][0] == "beta"


def test_without_params_or_a_matching_plan_group_the_node_cannot_run(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    rig.publish(Kind.TASK_PLAN, task_plan_payload([plan_group("alpha", node_id="implement.x")]))
    with pytest.raises(AgentError, match="no valid 'group'"):
        rig.run(agent, "implement", "implement.y", params={})


def test_group_assignment_is_a_plain_value_object() -> None:
    work = GroupAssignment("alpha", "fake", {}, ())
    assert (work.task_ids, work.requirement_ids) == ((), ())


# ----------------------------------------------------------------------------- scope and drift


def test_a_write_outside_the_write_scope_is_fatal(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path, FakeRecipe(outside="pkg/other/stray.py"))
    with pytest.raises(AgentError, match=r"outside its write_scope: pkg/other/stray.py") as err:
        rig.run(agent, "implement", NODE, params=assignment())
    assert err.value.kind == "fatal" and not err.value.retryable


def test_the_toolbox_denies_writes_outside_the_node_autonomy(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path, FakeRecipe(outside="pkg/other/stray.py"))
    narrowed = AutonomyProfile(write_globs=("pkg/alpha/**",))
    with pytest.raises(ToolDenied):
        rig.run(agent, "implement", NODE, params=assignment(), node_autonomy=narrowed)


def test_a_hostile_write_scope_glob_is_a_path_violation(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    params = {**assignment(), "write_scope": ["../**"]}
    with pytest.raises(PathViolation):
        rig.run(agent, "implement", NODE, params=params)


def test_an_anchored_edit_of_an_existing_file_is_a_modify_op(tmp_path: Path) -> None:
    files = {"pkg/registry.py": "ITEMS = []  # MARK\n"}
    rig, agent, _ = setup(tmp_path, FakeRecipe(anchor=("pkg/registry.py", "# MARK")), files=files)
    params = {**assignment(), "write_scope": ["pkg/**"]}
    cs = changeset_of(rig.run(agent, "implement", NODE, params=params))
    op = next(o for o in cs.ops if o.path == "pkg/registry.py")
    assert op.kind is FileOpKind.MODIFY
    assert op.base_hash == sha256_text(files["pkg/registry.py"])
    assert apply_changeset(rig.baseline, cs)["pkg/registry.py"] == "ITEMS = []  # MARK  # patched\n"


def test_anchor_drift_is_fatal_and_produces_nothing(tmp_path: Path) -> None:
    files = {"pkg/registry.py": "ITEMS = []  # RENAMED\n"}
    rig, agent, _ = setup(tmp_path, FakeRecipe(anchor=("pkg/registry.py", "# MARK")), files=files)
    params = {**assignment(), "write_scope": ["pkg/**"]}
    with pytest.raises(RecipeAnchorError, match="anchor '# MARK' not found") as err:
        rig.run(agent, "implement", NODE, params=params)
    assert err.value.kind == "fatal" and not err.value.retryable


def test_a_group_that_changes_nothing_is_fatal(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path, FakeRecipe(writes_nothing=True))
    with pytest.raises(AgentError, match="produced no file changes") as err:
        rig.run(agent, "implement", NODE, params=assignment())
    assert err.value.kind == "fatal"


def test_a_cancelled_attempt_stops_at_the_first_toolbox_call(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    cancel = CancelToken()
    cancel.cancel("kill switch")
    with pytest.raises(AgentCancelled):
        rig.run(agent, "implement", NODE, params=assignment(), cancel=cancel)


# ----------------------------------------------------------------------------- recipe inputs


def test_the_codebase_model_is_built_only_for_baseline_capabilities(tmp_path: Path) -> None:
    files = {"app/main.py": "def run() -> int:\n    return 1\n"}
    rig, agent, recipe = setup(tmp_path, files=files, requires_baseline=True)
    rig.run(agent, "implement", NODE, params=assignment())
    assert isinstance(recipe.group_calls[0][1].model, CodebaseModel)
    rig2, agent2, recipe2 = setup(tmp_path / "greenfield", files=files)
    rig2.run(agent2, "implement", NODE, params=assignment())
    assert recipe2.group_calls[0][1].model is None


def test_the_codebase_model_is_skipped_when_the_tool_call_budget_is_too_small(
    tmp_path: Path,
) -> None:
    files = {f"app/m{i}.py": f"VALUE = {i}\n" for i in range(80)}
    rig, agent, recipe = setup(tmp_path, files=files, requires_baseline=True)
    tight = AutonomyProfile(write_globs=("**",), max_tool_calls=120)  # the sdlc default is 200
    result = rig.run(agent, "implement", NODE, params=assignment(), node_autonomy=tight)
    assert recipe.group_calls[0][1].model is None  # the recipe still ran, on file content alone
    assert changeset_of(result).ops
    skipped = [n for n in rig.audit.of(EventType.AGENT_NOTE) if "model skipped" in n["message"]]
    assert [n["data"] for n in skipped] == [{"files": 80}]
    roomy = AutonomyProfile(write_globs=("**",), max_tool_calls=400)
    rig.run(agent, "implement", NODE, params=assignment(), node_autonomy=roomy)
    assert isinstance(recipe.group_calls[1][1].model, CodebaseModel)


def test_the_design_document_reaches_the_recipe_when_declared(tmp_path: Path) -> None:
    rig, agent, recipe = setup(tmp_path)
    rig.run(agent, "implement", NODE, params=assignment())
    assert recipe.group_calls[0][1].design is None
    rig.publish(Kind.DESIGN_DOC, design_doc_payload())
    rig.run(agent, "implement", NODE, params=assignment())
    assert recipe.group_calls[1][1].design == design_doc_payload()
    rig.run(agent, "implement", NODE, params=assignment(), undeclared=(Kind.DESIGN_DOC,))
    assert recipe.group_calls[2][1].design is None  # an undeclared kind is simply absent


def test_engine_feedback_is_noted_but_does_not_change_the_output(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    plain = rig.run(agent, "implement", NODE, params=assignment())
    again = rig.run(
        agent,
        "implement",
        NODE,
        params=assignment(),
        feedback=(feedback_finding("SEC-001"), feedback_finding("QLT-001")),
    )
    assert plain.artifacts[0].content == again.artifacts[0].content
    notes = [n for n in rig.audit.of(EventType.AGENT_NOTE) if "feedback" in n["message"]]
    assert [n["data"]["rule_ids"] for n in notes] == [["QLT-001", "SEC-001"]]


# ----------------------------------------------------------------------------- conservative mode


@pytest.mark.parametrize(
    ("agent_class", "agent_id", "conservative"),
    [(ImplementerAgent, "implementer", False), (SafeImplementerAgent, "implementer.safe", True)],
)
def test_identity_and_protocol(
    agent_class: type[ImplementerAgent], agent_id: str, conservative: bool
) -> None:
    agent = agent_class()
    assert isinstance(agent, Agent) and NODE_ID_RE.match(agent.agent_id)
    assert (agent.agent_id, agent.role, agent.version) == (agent_id, "implementer", "1")
    assert agent.conservative is conservative
    assert not agent.autonomy.can_execute
    assert agent.autonomy.write_globs == ("**",)  # a node's write_scope can only narrow it


def test_a_node_write_scope_survives_narrowing() -> None:
    node = AutonomyProfile(write_globs=("pkg/alpha/**",))
    assert ImplementerAgent.autonomy.narrow(node).write_globs == ("pkg/alpha/**",)
    assert SafeImplementerAgent.autonomy.narrow(node).write_globs == ("pkg/alpha/**",)


def test_the_safe_agent_has_tighter_budgets() -> None:
    primary, safe = ImplementerAgent.autonomy, SafeImplementerAgent.autonomy
    assert safe.max_changed_files < primary.max_changed_files
    assert safe.max_total_bytes < primary.max_total_bytes


def test_the_conservative_flag_reaches_the_recipe(tmp_path: Path) -> None:
    for agent_class, expected in ((ImplementerAgent, False), (SafeImplementerAgent, True)):
        rig, agent, recipe = setup(tmp_path / agent_class.__name__, agent_class=agent_class)
        rig.run(agent, "implement", NODE, params=assignment())
        assert recipe.group_calls[0][1].anchors == {"conservative": expected}


def test_the_safe_agent_drops_optional_extras_when_the_recipe_honours_the_flag(
    tmp_path: Path,
) -> None:
    primary = setup(tmp_path / "p")
    safe = setup(tmp_path / "s", agent_class=SafeImplementerAgent)
    full = changeset_of(primary[0].run(primary[1], "implement", NODE, params=assignment()))
    minimal = changeset_of(safe[0].run(safe[1], "implement", NODE, params=assignment()))
    assert "pkg/alpha/fast_path.py" in content_of(full)
    assert set(content_of(minimal)) == set(content_of(full)) - {"pkg/alpha/fast_path.py"}
    assert minimal.author == "implementer.safe"


def test_the_safe_agent_matches_the_primary_when_the_recipe_ignores_the_flag(
    tmp_path: Path,
) -> None:
    recipe = FakeRecipe(honours_conservative=False)
    primary = setup(tmp_path / "p", recipe)
    safe = setup(tmp_path / "s", recipe, agent_class=SafeImplementerAgent)
    full = changeset_of(primary[0].run(primary[1], "implement", NODE, params=assignment()))
    same = changeset_of(safe[0].run(safe[1], "implement", NODE, params=assignment()))
    assert content_of(full) == content_of(same)
    assert full.ops == same.ops


def test_the_safe_agent_refuses_deletions_but_the_primary_may_delete(tmp_path: Path) -> None:
    files = {"pkg/old.py": "OLD = 1\n"}
    params = {**assignment(), "write_scope": ["pkg/**"]}
    recipe = FakeRecipe(delete="pkg/old.py")
    rig, agent, _ = setup(tmp_path / "p", recipe, files=files)
    cs = changeset_of(rig.run(agent, "implement", NODE, params=params))
    assert any(o.kind is FileOpKind.DELETE and o.path == "pkg/old.py" for o in cs.ops)
    rig2, safe, _ = setup(tmp_path / "s", recipe, files=files, agent_class=SafeImplementerAgent)
    with pytest.raises(AgentError, match=r"never deletes files: pkg/old.py") as err:
        rig2.run(safe, "implement", NODE, params=params)
    assert err.value.kind == "fatal" and not err.value.retryable


def test_the_safe_budget_denies_a_change_the_primary_may_make(tmp_path: Path) -> None:
    recipe = FakeRecipe(extra_files=25)
    rig, agent, _ = setup(tmp_path / "p", recipe)
    assert len(changeset_of(rig.run(agent, "implement", NODE, params=assignment())).ops) == 28
    rig2, safe, _ = setup(tmp_path / "s", recipe, agent_class=SafeImplementerAgent)
    with pytest.raises(ToolDenied, match="max_changed_files"):
        rig2.run(safe, "implement", NODE, params=assignment())


# ----------------------------------------------------------------------------- the real recipes


@pytest.mark.parametrize("agent_class", [ImplementerAgent, SafeImplementerAgent])
def test_every_group_of_the_real_linkcheck_recipe_builds_inside_its_scope(
    tmp_path: Path, agent_class: type[ImplementerAgent]
) -> None:
    capability = real_capability("linkcheck")
    rig = Rig(tmp_path)
    groups = capability.recipe.groups(default_params(capability))
    assert len(groups) >= 3
    for group in groups:
        agent = agent_class(lookup(capability))
        cs = changeset_of(run_group(rig, agent, capability, group))
        assert cs.ops and all(match_any(p, group.write_scope) for p in cs.paths()), group.id
        profile = agent.autonomy.narrow(AutonomyProfile(write_globs=group.write_scope))
        report = evaluate(
            cs,
            rig.baseline,
            node_id=group_node(capability, group),
            profile=profile,
            config=LINKCHECK_POLICY,
        )
        assert not report.blocking, (group.id, report.findings)


@pytest.mark.parametrize("agent_class", [ImplementerAgent, SafeImplementerAgent])
def test_the_real_max_clicks_recipe_edits_the_real_product_by_anchors(
    tmp_path: Path, agent_class: type[ImplementerAgent]
) -> None:
    capability = real_capability("max_clicks")
    rig = Rig(tmp_path, product_files())
    for group in capability.recipe.groups(default_params(capability)):
        cs = changeset_of(run_group(rig, agent_class(lookup(capability)), capability, group))
        assert cs.ops and all(match_any(p, group.write_scope) for p in cs.paths()), group.id
        applied = apply_changeset(rig.baseline, cs)  # strict: every base hash matches the product
        assert set(applied) == set(cs.paths())
        assert any(o.kind is FileOpKind.MODIFY for o in cs.ops)


def test_real_anchor_drift_in_the_product_is_a_fatal_error(tmp_path: Path) -> None:
    capability = real_capability("max_clicks")
    files = product_files()
    victim = "shortener/storage/link_repository.py"
    files[victim] = files[victim].replace("click_count", "hit_total")
    rig = Rig(tmp_path, files)
    storage = next(
        g for g in capability.recipe.groups(default_params(capability)) if g.id == "storage"
    )
    with pytest.raises(RecipeAnchorError) as err:
        run_group(rig, ImplementerAgent(lookup(capability)), capability, storage)
    assert err.value.kind == "fatal" and not err.value.retryable
