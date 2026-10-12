"""Tests for the test-author agent: one ``build_tests`` call per active capability."""

from __future__ import annotations

import ast
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.recipes.base import RecipeParamError
from orchestrator.agents.test_author import (
    TestAuthorAgent,
    active_capabilities,
    requirements_view,
)
from orchestrator.core.errors import AgentError, ToolDenied
from orchestrator.core.paths import NODE_ID_RE, match_any
from orchestrator.core.schemas import Kind
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import Agent, AutonomyProfile, EventType
from tests.orchestrator.agents.support_impl_chain import (
    FakeRecipe,
    Rig,
    changeset_of,
    content_of,
    default_params,
    design_doc_payload,
    evaluate,
    fake_capability,
    feedback_finding,
    lookup,
    plan_group,
    product_files,
    real_capability,
    requirement,
    requirements_payload,
    task_plan_payload,
)

NODE = "author-tests"
COVERS = re.compile(r"Covers:\s*(.+)")


def covered(source: str) -> set[str]:
    """Every id on a ``Covers:`` docstring line of the test functions in ``source``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef):
            match = COVERS.search(ast.get_docstring(node) or "")
            if match:
                found.update(part.strip() for part in match.group(1).split(","))
    return found


def setup(
    tmp_path: Path,
    *recipes: FakeRecipe,
    requires_baseline: bool = False,
    files: dict[str, str] | None = None,
    **rig_kwargs: Any,
) -> tuple[Rig, TestAuthorAgent, tuple[FakeRecipe, ...]]:
    chosen = recipes or (FakeRecipe(),)
    agent = TestAuthorAgent(
        lookup(*(fake_capability(r, requires_baseline=requires_baseline) for r in chosen))
    )
    return Rig(tmp_path, files or {}, **rig_kwargs), agent, chosen


def publish_requirements(rig: Rig, *reqs: dict[str, Any], **kwargs: Any) -> None:
    rig.publish(Kind.REQUIREMENTS, requirements_payload(reqs, **kwargs))


# ----------------------------------------------------------------------------- the happy path


def test_builds_the_tests_of_each_active_capability_exactly_once(tmp_path: Path) -> None:
    first, second = FakeRecipe("fake"), FakeRecipe("other")
    rig, agent, _ = setup(tmp_path, first, second)
    publish_requirements(
        rig,
        requirement(1, capabilities=("fake",)),
        requirement(2, capabilities=("other",)),
        requirement(3, capabilities=("fake",)),
    )
    result = rig.run(agent, "author-tests", NODE)
    cs = changeset_of(result)
    assert sorted(cs.paths()) == ["tests/fake/test_fake.py", "tests/other/test_other.py"]
    assert len(first.test_calls) == 1 and len(second.test_calls) == 1
    assert cs.author == "test-author" and result.usage != {}
    assert cs.description == "author tests for capabilities: fake, other"


def test_the_generated_tests_carry_covers_lines_for_every_criterion(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    publish_requirements(
        rig,
        requirement(1, criteria=(("alpha", "a"), ("beta", "b"))),
        requirement(2),
    )
    cs = changeset_of(rig.run(agent, "author-tests", NODE))
    text = str(content_of(cs)["tests/fake/test_fake.py"])
    assert covered(text) == {"R-001", "AC-001-1", "AC-001-2", "R-002", "AC-002-1"}


def test_the_decision_cites_the_active_requirements(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    publish_requirements(rig, requirement(1), requirement(2, status="deferred"), requirement(4))
    result = rig.run(agent, "author-tests", NODE)
    (decision,) = result.decisions
    assert decision.refs == ("R-001", "R-004")
    assert decision.data["capabilities"] == ["fake"]
    assert decision.outcome == "1 test file(s) proposed"


def test_the_output_is_deterministic_across_runs_and_clocks(tmp_path: Path) -> None:
    runs = []
    for index, start in enumerate(
        (datetime(2026, 1, 1, tzinfo=UTC), datetime(2030, 5, 5, tzinfo=UTC))
    ):
        rig, agent, _ = setup(tmp_path / str(index), clock_start=start)
        publish_requirements(rig, requirement(1), requirement(2))
        runs.append(rig.run(agent, "author-tests", NODE))
    assert hash_jsonable([d.content for d in runs[0].artifacts]) == hash_jsonable(
        [d.content for d in runs[1].artifacts]
    )
    assert runs[0].decisions == runs[1].decisions


def test_the_result_passes_the_real_policy_engine(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    publish_requirements(rig, requirement(1))
    ctx = rig.context(agent, "author-tests", NODE)
    cs = changeset_of(agent.run(ctx))
    profile = agent.autonomy.narrow(ctx.node.autonomy)
    report = evaluate(cs, rig.baseline, node_id=NODE, profile=profile)
    assert not report.blocking and not report.needs_approval, report.findings


# ----------------------------------------------------------------------------- active capabilities


def test_deferred_and_descoped_requirements_are_never_built(tmp_path: Path) -> None:
    kept, dropped = FakeRecipe("fake"), FakeRecipe("other")
    rig, agent, _ = setup(tmp_path, kept, dropped)
    publish_requirements(
        rig,
        requirement(1, capabilities=("fake",)),
        requirement(2, capabilities=("fake", "other"), status="deferred"),
        requirement(3, capabilities=("other",), status="descoped"),
    )
    cs = changeset_of(rig.run(agent, "author-tests", NODE))
    assert list(cs.paths()) == ["tests/fake/test_fake.py"]
    assert dropped.test_calls == []
    (ctx, _), *_ = kept.test_calls
    assert ctx.requirements is not None
    assert [r["id"] for r in ctx.requirements["requirements"]] == ["R-001"]


def test_each_recipe_only_sees_the_requirements_of_its_own_capability(tmp_path: Path) -> None:
    first, second = FakeRecipe("fake"), FakeRecipe("other")
    rig, agent, _ = setup(tmp_path, first, second)
    publish_requirements(
        rig,
        requirement(1, capabilities=("fake",)),
        requirement(2, capabilities=("other",)),
        requirement(3, capabilities=("fake", "other")),
    )
    rig.run(agent, "author-tests", NODE)
    seen = {
        recipe.id: [r["id"] for r in recipe.test_calls[0][0].requirements["requirements"]]  # type: ignore[index]
        for recipe in (first, second)
    }
    assert seen == {"fake": ["R-001", "R-003"], "other": ["R-002", "R-003"]}


def test_a_recipe_cannot_alter_the_requirements_other_recipes_see(tmp_path: Path) -> None:
    vandal, bystander = FakeRecipe("aaa", mutate_requirements=True), FakeRecipe("zzz")
    rig, agent, _ = setup(tmp_path, vandal, bystander)
    publish_requirements(
        rig, requirement(1, capabilities=("aaa", "zzz")), requirement(2, capabilities=("zzz",))
    )
    rig.run(agent, "author-tests", NODE)
    ctx, _ = bystander.test_calls[0]
    assert ctx.requirements is not None
    assert [r["id"] for r in ctx.requirements["requirements"]] == ["R-001", "R-002"]


def test_no_active_capability_yields_an_empty_change_set_and_no_decision(tmp_path: Path) -> None:
    recipe = FakeRecipe()
    rig, agent, _ = setup(tmp_path, recipe)
    publish_requirements(rig, requirement(1, status="deferred"))
    result = rig.run(agent, "author-tests", NODE)
    assert changeset_of(result).ops == ()
    assert result.decisions == [] and recipe.test_calls == []
    assert changeset_of(result).description == "author tests for capabilities: none"


def test_active_capabilities_helper_orders_and_filters() -> None:
    payload = requirements_payload(
        [
            requirement(2, capabilities=("zeta", "alpha")),
            requirement(1, capabilities=("alpha",)),
            requirement(3, capabilities=("gone",), status="descoped"),
        ]
    )
    payload["requirements"][0]["capabilities"].append(7)  # a non-string entry is ignored
    assert active_capabilities(payload) == {"alpha": ["R-001", "R-002"], "zeta": ["R-002"]}
    view = requirements_view(payload, "zeta")
    assert [r["id"] for r in view["requirements"]] == ["R-002"]
    assert len(payload["requirements"]) == 3  # the original is untouched


# ----------------------------------------------------------------------------- parameters


@pytest.mark.parametrize(
    ("plan_params", "requirement_params", "expected"),
    [
        ({"limit": 7}, {"limit": 5}, 7),  # the plan (what the implementers use) wins
        (None, {"limit": 5}, 5),  # else the capability entry of the requirements
        (None, None, 3),  # else the recipe default
    ],
)
def test_recipe_params_come_from_the_plan_then_the_requirements(
    tmp_path: Path,
    plan_params: dict[str, Any] | None,
    requirement_params: dict[str, Any] | None,
    expected: int,
) -> None:
    rig, agent, (recipe,) = setup(tmp_path)
    publish_requirements(
        rig,
        requirement(1),
        capability_params={"fake": requirement_params} if requirement_params else None,
    )
    if plan_params is not None:
        rig.publish(Kind.TASK_PLAN, task_plan_payload([plan_group("alpha", params=plan_params)]))
    rig.run(agent, "author-tests", NODE)
    assert recipe.test_calls[0][1] == {"limit": expected}


@pytest.mark.parametrize("bad", [{"limit": 99}, {"limit": "x"}, {"unknown": 1}])
def test_invalid_params_are_fatal_before_any_recipe_runs(
    tmp_path: Path, bad: dict[str, Any]
) -> None:
    rig, agent, (recipe,) = setup(tmp_path)
    publish_requirements(rig, requirement(1), capability_params={"fake": bad})
    with pytest.raises(RecipeParamError) as err:
        rig.run(agent, "author-tests", NODE)
    assert err.value.kind == "fatal" and recipe.test_calls == []


def test_the_design_document_reaches_the_recipe_when_present(tmp_path: Path) -> None:
    rig, agent, (recipe,) = setup(tmp_path)
    publish_requirements(rig, requirement(1))
    rig.run(agent, "author-tests", NODE)
    assert recipe.test_calls[0][0].design is None
    rig.publish(Kind.DESIGN_DOC, design_doc_payload())
    rig.run(agent, "author-tests", NODE)
    assert recipe.test_calls[1][0].design == design_doc_payload()


def test_no_codebase_model_is_built_because_tests_come_from_the_requirements(
    tmp_path: Path,
) -> None:
    files = {"app/main.py": "def run() -> int:\n    return 1\n"}
    rig, agent, (recipe,) = setup(tmp_path, requires_baseline=True, files=files)
    publish_requirements(rig, requirement(1))
    result = rig.run(agent, "author-tests", NODE)
    assert recipe.test_calls[0][0].model is None
    assert result.usage["tool_calls"] < 10  # no whole-workspace read


def test_engine_feedback_is_noted_but_does_not_change_the_output(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    publish_requirements(rig, requirement(1))
    plain = rig.run(agent, "author-tests", NODE)
    again = rig.run(agent, "author-tests", NODE, feedback=(feedback_finding("QLT-001"),))
    assert plain.artifacts[0].content == again.artifacts[0].content
    notes = [n for n in rig.audit.of(EventType.AGENT_NOTE) if "feedback" in n["message"]]
    assert [n["data"]["rule_ids"] for n in notes] == [["QLT-001"]]


# ----------------------------------------------------------------------------- failures


def test_an_unknown_capability_is_fatal(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    publish_requirements(rig, requirement(1, capabilities=("mystery",)))
    with pytest.raises(AgentError, match="unknown capability 'mystery'") as err:
        rig.run(agent, "author-tests", NODE)
    assert err.value.kind == "fatal" and not err.value.retryable


def test_a_malformed_capability_id_is_fatal_and_never_echoed_raw(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path)
    publish_requirements(rig, requirement(1, capabilities=("Bad\x1b[31m cap",)))
    with pytest.raises(AgentError, match="malformed capability id") as err:
        rig.run(agent, "author-tests", NODE)
    assert "\x1b" not in str(err.value)


def test_the_default_lookup_is_the_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def catalog(capability_id: str) -> Any:
        asked.append(capability_id)
        raise KeyError(capability_id)

    monkeypatch.setattr("orchestrator.agents.test_author.get_capability", catalog)
    rig = Rig(tmp_path)
    publish_requirements(rig, requirement(1, capabilities=("no_such_capability",)))
    with pytest.raises(AgentError, match="unknown capability"):
        rig.run(TestAuthorAgent(), "author-tests", NODE)
    assert asked == ["no_such_capability"]


def test_active_capabilities_whose_recipes_stage_nothing_are_fatal(tmp_path: Path) -> None:
    rig, agent, _ = setup(tmp_path, FakeRecipe(writes_nothing=True))
    publish_requirements(rig, requirement(1))
    with pytest.raises(AgentError, match="staged no test files") as err:
        rig.run(agent, "author-tests", NODE)
    assert err.value.kind == "fatal"


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"requirements": "nope"},
        {"requirements": [1, 2]},
        {"requirements": [{"id": "R-001", "status": "active", "capabilities": "fake"}]},
    ],
)
def test_malformed_requirements_are_fatal(tmp_path: Path, payload: Any) -> None:
    rig, agent, recipe = setup(tmp_path)
    rig.publish(Kind.REQUIREMENTS, payload, validate=False)
    with pytest.raises(AgentError) as err:
        rig.run(agent, "author-tests", NODE)
    assert err.value.kind == "fatal" and recipe[0].test_calls == []


def test_tests_may_only_be_written_under_tests(tmp_path: Path) -> None:
    class Sneaky(FakeRecipe):
        def build_tests(self, ctx: Any) -> None:
            ctx.toolbox.write_text("src/not_a_test.py", "X = 1\n")

    rig, agent, _ = setup(tmp_path, Sneaky())
    publish_requirements(rig, requirement(1))
    with pytest.raises(ToolDenied, match="write_globs"):
        rig.run(agent, "author-tests", NODE)


# ----------------------------------------------------------------------------- identity


def test_identity_protocol_and_autonomy() -> None:
    agent = TestAuthorAgent()
    assert isinstance(agent, Agent) and NODE_ID_RE.match(agent.agent_id)
    assert (agent.agent_id, agent.role, agent.version) == ("test-author", "test-author", "1")
    assert agent.autonomy.write_globs == ("tests/**",) and not agent.autonomy.can_execute
    node = AutonomyProfile(write_globs=("tests/**",))
    assert agent.autonomy.narrow(node).write_globs == ("tests/**",)
    assert TestAuthorAgent.__test__ is False  # pytest must not collect the class


# ----------------------------------------------------------------------------- the real recipes


def all_aspect_criteria(capability: Any) -> list[tuple[str, str]]:
    params = default_params(capability)
    return [(a.id, a.criterion.format_map(params)) for a in capability.aspects]


@pytest.mark.parametrize("name", ["linkcheck", "max_clicks"])
def test_the_real_recipes_cover_every_acceptance_criterion(tmp_path: Path, name: str) -> None:
    capability = real_capability(name)
    criteria = all_aspect_criteria(capability)
    rig = Rig(tmp_path, {} if name == "linkcheck" else product_files())
    publish_requirements(rig, requirement(1, capabilities=(name,), criteria=criteria))
    agent = TestAuthorAgent(lookup(capability))
    ctx = rig.context(agent, "author-tests", NODE)
    cs = changeset_of(agent.run(ctx))
    assert cs.ops and all(match_any(p, ("tests/**",)) for p in cs.paths())
    ids: set[str] = set()
    for text in content_of(cs).values():
        ids |= covered(str(text))
    assert {f"AC-001-{i}" for i in range(1, len(criteria) + 1)} <= ids
