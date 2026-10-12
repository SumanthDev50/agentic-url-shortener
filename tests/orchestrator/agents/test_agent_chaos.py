"""Tests for the chaos agent: every fault produces the violation it is meant to provoke."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from orchestrator.agents import chaos as chaos_module
from orchestrator.agents.chaos import (
    FAULTS,
    REGRESSION_STATEMENT,
    ChaosAgent,
    fake_access_key_id,
    leak_line,
    leak_module_text,
)
from orchestrator.agents.implementer import ImplementerAgent, SafeImplementerAgent
from orchestrator.agents.recipes.base import RecipeAnchorError
from orchestrator.core.errors import (
    AgentError,
    AutonomyViolation,
    ConfigError,
    PathViolation,
    TransientAgentError,
)
from orchestrator.core.schemas import Kind
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import (
    Agent,
    AgentContext,
    AgentResult,
    ArtifactDraft,
    AutonomyProfile,
    ChangeSet,
    Enforcement,
    EventType,
    FileOp,
    FileOpKind,
    Toolbox,
)
from orchestrator.engine.registry import AgentRegistry
from orchestrator.policies.scanners import scan_secrets
from orchestrator.tools.sandbox import SandboxRunner
from orchestrator.tools.workspace import MemoryView, OverlayView, materialize
from tests.orchestrator.agents.support_impl_chain import (
    LINKCHECK_POLICY,
    FakeRecipe,
    Rig,
    build_chain,
    changeset_of,
    content_of,
    default_params,
    evaluate,
    fake_capability,
    feedback_finding,
    group_assignment,
    group_node,
    lookup,
    product_files,
    real_capability,
    rule_ids,
    run_group,
)

NODE = "implement.fake-alpha"
ASSIGNMENT = {
    "group": "alpha",
    "capability": "fake",
    "recipe_params": {},
    "write_scope": ["pkg/alpha/**"],
}
MIGRATIONS = {
    "shortener/storage/migrations/001_init.sql": "CREATE TABLE links (id INTEGER);\n",
    "shortener/storage/migrations/002_more.sql": "ALTER TABLE links ADD COLUMN note TEXT;\n",
}
WIDE = AutonomyProfile(write_globs=("pkg/**", "shortener/storage/migrations/*.sql"))


class StubAgent:
    """Runs ``behave(ctx)`` (it may stage writes through the toolbox) and counts its calls."""

    agent_id = "stub"
    role = "stub-role"
    version = "9"
    autonomy = AutonomyProfile(read_globs=("**",), write_globs=("**",))

    def __init__(self, behave: Callable[[AgentContext], AgentResult] | None = None) -> None:
        self._behave = behave or (lambda ctx: AgentResult(rationale="inner work"))
        self.calls = 0

    def run(self, ctx: AgentContext) -> AgentResult:
        self.calls += 1
        return self._behave(ctx)


def staged_result(*writes: tuple[str, str], deletes: tuple[str, ...] = ()) -> StubAgent:
    """A stub that stages ``writes`` / ``deletes`` through the toolbox; returns the change set."""

    def behave(ctx: AgentContext) -> AgentResult:
        for path, content in writes:
            ctx.toolbox.write_text(path, content)
        for path in deletes:
            ctx.toolbox.delete(path)
        cs = ctx.toolbox.staged_changeset("stub change")
        return AgentResult([ArtifactDraft(Kind.CHANGESET, cs.to_dict())], rationale="inner work")

    return StubAgent(behave)


def implementer(tmp_path: Path, **recipe_options: Any) -> tuple[Rig, ImplementerAgent, FakeRecipe]:
    recipe = FakeRecipe(**recipe_options)
    return Rig(tmp_path), ImplementerAgent(lookup(fake_capability(recipe))), recipe


def notes(rig: Rig) -> list[dict[str, Any]]:
    return [n for n in rig.audit.of(EventType.AGENT_NOTE) if "chaos" in n["message"]]


# ----------------------------------------------------------------------------- configuration


def test_the_fault_catalogue() -> None:
    assert FAULTS == ("transient", "secret_leak", "migration_edit", "path_escape", "regression")


@pytest.mark.parametrize("fault", ["boom", "", "TRANSIENT", None, 7])
def test_unknown_faults_are_rejected(fault: Any) -> None:
    with pytest.raises(ConfigError, match="unknown fault"):
        ChaosAgent(StubAgent(), fault)


@pytest.mark.parametrize("attempts", [0, -1, 1001, True, 2.5, "3"])
def test_invalid_attempt_counts_are_rejected(attempts: Any) -> None:
    with pytest.raises(ConfigError, match="attempts"):
        ChaosAgent(StubAgent(), "transient", attempts)


def test_only_agents_can_be_wrapped() -> None:
    with pytest.raises(ConfigError, match="needs an Agent"):
        ChaosAgent(cast(Any, object()), "transient")


def test_the_wrapper_reports_the_identity_of_the_wrapped_agent() -> None:
    inner = StubAgent()
    wrapper = ChaosAgent(inner, "regression")
    assert isinstance(wrapper, Agent)
    assert (wrapper.agent_id, wrapper.role, wrapper.version) == ("stub", "stub-role", "9")
    assert wrapper.autonomy is inner.autonomy
    assert (wrapper.fault, wrapper.attempts, wrapper.on_fallback) == ("regression", 1, False)
    assert wrapper.note == "chaos[regression]"


def test_a_wrapped_agent_can_be_registered_like_any_other() -> None:
    registry = AgentRegistry()
    registry.register("implementer", lambda: ChaosAgent(ImplementerAgent(), "transient", 2))
    created = registry.create("implementer")
    assert isinstance(created, ChaosAgent) and created.agent_id == "implementer"


# ----------------------------------------------------------------------------- transient


def test_transient_fails_the_first_attempts_then_delegates(tmp_path: Path) -> None:
    rig, inner = Rig(tmp_path), StubAgent()
    chaos = ChaosAgent(inner, "transient", 2)
    for attempt_no in (1, 2):
        with pytest.raises(TransientAgentError) as err:
            rig.run(chaos, "implement", NODE, attempt_no=attempt_no)
        assert err.value.kind == "transient" and err.value.retryable
        assert str(err.value) == (f"chaos: injected transient failure (attempt {attempt_no} of 2)")
        assert err.value.details == {"chaos": "transient"}
    assert inner.calls == 0
    result = rig.run(chaos, "implement", NODE, attempt_no=3)
    assert result.rationale == "inner work" and inner.calls == 1


def test_the_default_is_a_one_shot_fault(tmp_path: Path) -> None:
    rig, inner = Rig(tmp_path), StubAgent()
    chaos = ChaosAgent(inner, "transient")
    with pytest.raises(TransientAgentError):
        rig.run(chaos, "implement", NODE, attempt_no=1)
    assert rig.run(chaos, "implement", NODE, attempt_no=2).rationale == "inner work"


# ----------------------------------------------------------------------------- when faults fire

# (fault, attempts, attempt_no, is_fallback, on_fallback, feedback rule ids, fires)
FIRING = [
    ("path_escape", None, 1, False, False, (), True),
    ("path_escape", None, 2, False, False, (), False),  # one-shot: attempts exhausted
    ("path_escape", 99, 2, False, False, ("SEC-005",), True),  # told about its own mistake: repeats
    ("path_escape", 99, 2, False, False, ("QLT-001",), False),  # other feedback = repair round
    ("path_escape", 99, 2, False, False, ("QLT-001", "SEC-005"), True),
    ("path_escape", 99, 3, True, False, ("SEC-005",), False),  # the fallback repairs
    ("path_escape", 99, 4, True, True, ("SEC-005",), True),  # exhausted-style double failure
    ("path_escape", 5, 6, True, True, (), False),  # attempts must cover both epochs
    ("transient", 99, 2, False, False, ("SEC-005",), False),  # no own rule: any feedback repairs
    ("transient", 5, 4, True, True, (), True),
]


@pytest.mark.parametrize(
    ("fault", "attempts", "attempt_no", "is_fallback", "on_fallback", "feedback", "fires"), FIRING
)
def test_firing_rules(
    tmp_path: Path,
    fault: str,
    attempts: int | None,
    attempt_no: int,
    is_fallback: bool,
    on_fallback: bool,
    feedback: tuple[str, ...],
    fires: bool,
) -> None:
    rig, inner = Rig(tmp_path), StubAgent()
    chaos = ChaosAgent(inner, fault, attempts, on_fallback=on_fallback)
    ctx = rig.context(
        chaos,
        "implement",
        NODE,
        attempt_no=attempt_no,
        is_fallback=is_fallback,
        feedback=[feedback_finding(rule) for rule in feedback],
    )
    if fires:
        with pytest.raises((TransientAgentError, PathViolation)):
            chaos.run(ctx)
        assert inner.calls == 0 and len(notes(rig)) == 1
    else:
        assert chaos.run(ctx).rationale == "inner work"
        assert inner.calls == 1 and notes(rig) == []  # silence when delegating


# ----------------------------------------------------------------------------- path_escape


def test_path_escape_is_rejected_by_the_toolbox_as_a_critical_violation(tmp_path: Path) -> None:
    rig, inner = Rig(tmp_path), StubAgent()
    chaos = ChaosAgent(inner, "path_escape")
    ctx = rig.context(chaos, "implement", NODE)
    with pytest.raises(PathViolation) as err:
        chaos.run(ctx)
    assert isinstance(err.value, AutonomyViolation) and err.value.kind == "autonomy_violation"
    assert not err.value.retryable and inner.calls == 0
    with pytest.raises(AutonomyViolation):  # the violation poisoned the toolbox
        ctx.toolbox.exists("pkg/anything.py")
    denied = rig.audit.of(EventType.TOOL_DENIED)
    assert denied[0]["critical"] is True and denied[0]["tool"] == "write_text"


def test_path_escape_fails_closed_even_if_a_toolbox_accepts_the_path(tmp_path: Path) -> None:
    class Permissive:
        def note(self, message: str, **data: Any) -> None:
            return None

        def write_text(self, path: str, content: str) -> None:
            return None

    rig = Rig(tmp_path)
    chaos = ChaosAgent(StubAgent(), "path_escape")
    ctx = rig.context(chaos, "implement", NODE)
    ctx.toolbox = cast(Toolbox, Permissive())
    with pytest.raises(PathViolation, match="accepted a path outside the workspace"):
        chaos.run(ctx)


# ----------------------------------------------------------------------------- secret_leak


def test_secret_leak_plants_a_credential_that_the_real_policy_denies(tmp_path: Path) -> None:
    rig, inner, _ = implementer(tmp_path)
    chaos = ChaosAgent(inner, "secret_leak")
    ctx = rig.context(chaos, "implement", NODE, params=ASSIGNMENT)
    result = chaos.run(ctx)
    cs = changeset_of(result)
    leak = "pkg/alpha/chaos_leaked_credentials.py"
    assert content_of(cs)[leak] == leak_module_text()
    assert fake_access_key_id() in str(content_of(cs)[leak])
    assert ctx.toolbox.staged_changeset().ops == cs.ops  # the engine's consistency rule
    profile = chaos.autonomy.narrow(ctx.node.autonomy)
    report = evaluate(cs, rig.baseline, node_id=NODE, profile=profile)
    assert "SEC-001" in rule_ids(report) and report.enforcement is Enforcement.DENY
    shown = " ".join(f"{f.message} {f.evidence}" for f in report.findings)
    assert fake_access_key_id() not in shown  # the engine masks what it reports
    clean = evaluate(
        changeset_of(inner.run(rig.context(inner, "implement", NODE, params=ASSIGNMENT))),
        rig.baseline,
        node_id=NODE,
        profile=profile,
    )
    assert "SEC-001" not in rule_ids(clean)  # control: the honest output is clean


def test_secret_leak_is_announced_and_accounted_for(tmp_path: Path) -> None:
    rig, inner, _ = implementer(tmp_path)
    chaos = ChaosAgent(inner, "secret_leak")
    honest = inner.run(rig.context(inner, "implement", NODE, params=ASSIGNMENT))
    result = chaos.run(rig.context(chaos, "implement", NODE, params=ASSIGNMENT))
    assert result.rationale.startswith(
        "chaos[secret_leak]: planted a credential-shaped literal in "
    )
    assert result.rationale.endswith(honest.rationale)
    assert result.usage["tool_calls"] > honest.usage["tool_calls"]
    assert result.decisions == honest.decisions  # decisions and authorship are the inner's
    assert changeset_of(result).author == "implementer"
    assert [n["data"] for n in notes(rig)] == [{"fault": "secret_leak", "attempt_no": 1}]


def test_secret_leak_stays_inside_the_narrowed_write_scope(tmp_path: Path) -> None:
    rig, inner, _ = implementer(tmp_path)
    chaos = ChaosAgent(inner, "secret_leak")
    narrowed = AutonomyProfile(write_globs=("pkg/alpha/**",))
    cs = changeset_of(rig.run(chaos, "implement", NODE, params=ASSIGNMENT, node_autonomy=narrowed))
    assert "pkg/alpha/chaos_leaked_credentials.py" in cs.paths()


def test_secret_leak_falls_back_to_the_write_scope_when_nothing_was_written(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    chaos = ChaosAgent(staged_result(), "secret_leak")
    scope = AutonomyProfile(write_globs=("docs/**",))
    cs = changeset_of(rig.run(chaos, "implement", NODE, node_autonomy=scope))
    assert cs.paths() == ("docs/chaos_leaked_credentials.py",)


def test_secret_leak_without_a_writable_target_is_not_faked(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    chaos = ChaosAgent(staged_result(), "secret_leak")
    only_one_file = AutonomyProfile(write_globs=("only.py",))
    result = rig.run(chaos, "implement", NODE, node_autonomy=only_one_file)
    assert changeset_of(result).ops == ()
    assert result.rationale == (
        "chaos[secret_leak]: not applied (no applicable target for this fault). inner work"
    )


def test_secret_leak_taints_an_edited_file_when_the_scope_has_no_room_for_a_new_one(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    chaos = ChaosAgent(
        staged_result(("only.py", "VALUE = 1\n"), ("README.md", "x\n")), "secret_leak"
    )
    result = rig.run(chaos, "implement", NODE, node_autonomy=AutonomyProfile(write_globs=("**",)))
    assert "chaos_leaked_credentials.py" in " ".join(changeset_of(result).paths())
    exact = AutonomyProfile(write_globs=("only.py",))
    only = ChaosAgent(staged_result(("only.py", "VALUE = 1\n")), "secret_leak")
    cs = changeset_of(Rig(tmp_path / "exact").run(only, "implement", NODE, node_autonomy=exact))
    assert cs.paths() == ("only.py",)
    assert content_of(cs)["only.py"] == f"VALUE = 1\n{leak_line()}\n"
    assert scan_secrets(str(content_of(cs)["only.py"]), "only.py")  # and the scanners see it


def test_secret_leak_never_overwrites_an_existing_file(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"docs/chaos_leaked_credentials.py": "KEEP = 1\n"})
    chaos = ChaosAgent(staged_result(), "secret_leak")
    result = rig.run(
        chaos, "implement", NODE, node_autonomy=AutonomyProfile(write_globs=("docs/**",))
    )
    assert changeset_of(result).ops == ()


def test_the_fake_secret_is_detectable_but_never_a_literal_in_the_source() -> None:
    assert scan_secrets(fake_access_key_id(), "fake")  # a real scanner recognises the shape
    source = Path(chaos_module.__file__).read_text(encoding="utf-8")
    assert scan_secrets(source, "chaos.py") == []  # ... yet the source holds no such literal
    assert fake_access_key_id() not in source


# ----------------------------------------------------------------------------- migration_edit


def test_migration_edit_modifies_an_applied_migration_and_chg001_denies_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path, MIGRATIONS)
    inner = staged_result(("pkg/alpha/new.py", "X = 1\n"))
    chaos = ChaosAgent(inner, "migration_edit")
    ctx = rig.context(chaos, "implement", NODE, node_autonomy=WIDE)
    result = chaos.run(ctx)
    cs = changeset_of(result)
    op = next(o for o in cs.ops if o.path.endswith("001_init.sql"))
    assert op.kind is FileOpKind.MODIFY and op.base_hash
    assert str(op.content).endswith("\n-- chaos: edited an applied migration\n")
    assert ctx.toolbox.staged_changeset().ops == cs.ops
    report = evaluate(cs, rig.baseline, node_id=NODE, profile=chaos.autonomy.narrow(WIDE))
    assert "CHG-001" in rule_ids(report) and report.enforcement is Enforcement.DENY
    finding = next(f for f in report.findings if f.rule_id == "CHG-001")
    assert finding.location == "shortener/storage/migrations/001_init.sql"


def test_migration_edit_never_picks_a_migration_the_wrapped_agent_just_created(
    tmp_path: Path,
) -> None:
    created = "shortener/storage/migrations/003_new.sql"
    inner = staged_result((created, "ALTER TABLE links ADD COLUMN x INTEGER;\n"))
    rig = Rig(tmp_path, MIGRATIONS)
    cs = changeset_of(
        rig.run(ChaosAgent(inner, "migration_edit"), "implement", NODE, node_autonomy=WIDE)
    )
    edited = {o.path for o in cs.ops if o.kind is FileOpKind.MODIFY}
    assert edited == {"shortener/storage/migrations/001_init.sql"}
    assert next(o for o in cs.ops if o.path == created).kind is FileOpKind.CREATE
    only_new = Rig(tmp_path / "bare")  # no applied migration at all: nothing to edit
    result = only_new.run(
        ChaosAgent(inner, "migration_edit"), "implement", NODE, node_autonomy=WIDE
    )
    assert {o.path for o in changeset_of(result).ops} == {created}
    assert "not applied (no applicable target" in result.rationale


def test_migration_edit_respects_the_write_scope(tmp_path: Path) -> None:
    rig = Rig(tmp_path, MIGRATIONS)
    inner = staged_result(("pkg/alpha/new.py", "X = 1\n"))
    result = rig.run(
        ChaosAgent(inner, "migration_edit"),
        "implement",
        NODE,
        node_autonomy=AutonomyProfile(write_globs=("pkg/**",)),
    )
    assert {o.path for o in changeset_of(result).ops} == {"pkg/alpha/new.py"}
    assert "not applied" in result.rationale


# ----------------------------------------------------------------------------- regression


def test_regression_breaks_a_source_file_without_breaking_its_syntax(tmp_path: Path) -> None:
    rig, inner, _ = implementer(tmp_path)
    chaos = ChaosAgent(inner, "regression")
    ctx = rig.context(chaos, "implement", NODE, params=ASSIGNMENT)
    cs = changeset_of(chaos.run(ctx))
    broken = str(content_of(cs)["pkg/alpha/core.py"])
    assert broken.endswith("\n" + REGRESSION_STATEMENT)
    code = compile(broken, "pkg/alpha/core.py", "exec")  # still valid Python
    with pytest.raises(RuntimeError, match="chaos: injected regression"):
        exec(code, {})  # noqa: S102 - the planted statement is the thing under test
    report = evaluate(
        cs, rig.baseline, node_id=NODE, profile=chaos.autonomy.narrow(ctx.node.autonomy)
    )
    assert "QLT-001" not in rule_ids(report)  # the syntax gate cannot see it: only the tests can
    assert ctx.toolbox.staged_changeset().ops == cs.ops


def test_regression_skips_tests_markers_and_deletions(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"pkg/gone.py": "GONE = 1\n"})
    only_exempt = staged_result(
        ("tests/helper.py", "H = 1\n"),
        ("pkg/test_thing.py", "T = 1\n"),
        ("conftest.py", "C = 1\n"),
        ("pkg/__init__.py", ""),
        deletes=("pkg/gone.py",),
    )
    result = rig.run(ChaosAgent(only_exempt, "regression"), "implement", NODE)
    assert "not applied (no applicable target" in result.rationale
    assert REGRESSION_STATEMENT not in " ".join(
        str(v) for v in content_of(changeset_of(result)).values()
    )
    with_source = staged_result(("pkg/test_thing.py", "T = 1\n"), ("pkg/real.py", "R = 1\n"))
    cs = changeset_of(
        Rig(tmp_path / "b").run(ChaosAgent(with_source, "regression"), "implement", NODE)
    )
    assert content_of(cs)["pkg/real.py"] == "R = 1\n" + REGRESSION_STATEMENT
    assert content_of(cs)["pkg/test_thing.py"] == "T = 1\n"


@pytest.mark.slow
def test_a_regression_makes_the_suite_fail_in_the_real_sandbox(tmp_path: Path) -> None:
    baseline = {
        "tests/test_alpha.py": (
            "from pkg.alpha.core import limit\n\n\ndef test_limit() -> None:\n"
            "    assert limit() == 3\n"
        )
    }
    runner = SandboxRunner(tmp_path / "sandbox")

    def suite(cs: ChangeSet) -> int:
        scratch = runner.new_scratch()
        try:
            materialize(OverlayView(MemoryView(baseline), [cs]), scratch)
            result = runner.run(runner.pytest_argv(scratch, ["tests"]), cwd=scratch, timeout_s=120)
            return result.returncode
        finally:
            runner.discard(scratch)

    inner = ImplementerAgent(lookup(fake_capability(FakeRecipe())))
    rig = Rig(tmp_path / "rig", baseline)
    honest = changeset_of(inner.run(rig.context(inner, "implement", NODE, params=ASSIGNMENT)))
    chaos = ChaosAgent(inner, "regression")
    broken = changeset_of(chaos.run(rig.context(chaos, "implement", NODE, params=ASSIGNMENT)))
    assert suite(honest) == 0
    assert suite(broken) != 0


# ----------------------------------------------------------------------------- persistence rules


def test_content_faults_repeat_for_their_own_findings_and_stand_down_otherwise(
    tmp_path: Path,
) -> None:
    rig, inner, _ = implementer(tmp_path)
    leak = ChaosAgent(inner, "secret_leak", 99)
    migrate = ChaosAgent(staged_result(("pkg/alpha/x.py", "X = 1\n")), "migration_edit", 99)
    migrations = Rig(tmp_path / "m", MIGRATIONS)

    def leaked(feedback: list[str]) -> bool:
        ctx = rig.context(
            leak,
            "implement",
            NODE,
            params=ASSIGNMENT,
            attempt_no=2,
            feedback=[feedback_finding(r) for r in feedback],
        )
        return any(
            p.endswith("chaos_leaked_credentials.py") for p in changeset_of(leak.run(ctx)).paths()
        )

    def migration_edited(feedback: list[str]) -> bool:
        ctx = migrations.context(
            migrate,
            "implement",
            NODE,
            attempt_no=2,
            node_autonomy=WIDE,
            feedback=[feedback_finding(r) for r in feedback],
        )
        return any(o.kind is FileOpKind.MODIFY for o in changeset_of(migrate.run(ctx)).ops)

    assert leaked(["SEC-001"]) and not leaked(["CHG-001"]) and not leaked(["GATE-tests"])
    assert migration_edited(["CHG-001"]) and not migration_edited(["SEC-001"])


def test_a_regression_stands_down_for_any_feedback(tmp_path: Path) -> None:
    rig, inner, _ = implementer(tmp_path)
    chaos = ChaosAgent(inner, "regression", 99)
    ctx = rig.context(
        chaos,
        "implement",
        NODE,
        params=ASSIGNMENT,
        attempt_no=2,
        feedback=[feedback_finding("TST-001")],
    )
    assert REGRESSION_STATEMENT not in str(
        content_of(changeset_of(chaos.run(ctx)))["pkg/alpha/core.py"]
    )


def test_the_fallback_repairs_unless_it_is_wrapped_too(tmp_path: Path) -> None:
    rig, inner, _ = implementer(tmp_path)
    plain = ChaosAgent(inner, "secret_leak", 99)
    stubborn = ChaosAgent(inner, "secret_leak", 99, on_fallback=True)

    def paths(agent: ChaosAgent) -> tuple[str, ...]:
        ctx = rig.context(
            agent, "implement", NODE, params=ASSIGNMENT, attempt_no=4, is_fallback=True
        )
        return changeset_of(agent.run(ctx)).paths()

    assert not any("chaos_leaked" in p for p in paths(plain))
    assert any("chaos_leaked" in p for p in paths(stubborn))


# ----------------------------------------------------------------------------- not faked


def test_a_missing_change_set_is_reported_not_faked(tmp_path: Path) -> None:
    result = Rig(tmp_path).run(ChaosAgent(StubAgent(), "secret_leak"), "implement", NODE)
    assert result.artifacts == []
    assert result.rationale == (
        "chaos[secret_leak]: not applied (the wrapped agent produced no change set). inner work"
    )


def test_edits_that_bypassed_the_toolbox_are_left_alone(tmp_path: Path) -> None:
    def behave(ctx: AgentContext) -> AgentResult:
        cs = ChangeSet((FileOp("pkg/a.py", FileOpKind.CREATE, "A = 1\n"),), "by hand")
        return AgentResult([ArtifactDraft(Kind.CHANGESET, cs.to_dict())], rationale="inner work")

    result = Rig(tmp_path).run(ChaosAgent(StubAgent(behave), "regression"), "implement", NODE)
    assert content_of(changeset_of(result)) == {"pkg/a.py": "A = 1\n"}
    assert "did not stage its edits through the toolbox" in result.rationale


@pytest.mark.parametrize("content", ["just text", {"ops": [{"path": "../x", "kind": "create"}]}])
def test_an_unreadable_change_set_is_left_alone(tmp_path: Path, content: Any) -> None:
    stub = StubAgent(lambda ctx: AgentResult([ArtifactDraft(Kind.CHANGESET, content)]))
    result = Rig(tmp_path).run(ChaosAgent(stub, "secret_leak"), "implement", NODE)
    assert result.artifacts[0].content == content
    assert "the change set is not readable" in result.rationale


def test_errors_of_the_wrapped_agent_propagate_untouched(tmp_path: Path) -> None:
    def behave(ctx: AgentContext) -> AgentResult:
        raise RecipeAnchorError("drift")

    with pytest.raises(AgentError, match="drift") as err:
        Rig(tmp_path).run(ChaosAgent(StubAgent(behave), "secret_leak"), "implement", NODE)
    assert isinstance(err.value, RecipeAnchorError)


def test_the_output_is_deterministic_across_runs_and_clocks(tmp_path: Path) -> None:
    hashes = []
    for index, start in enumerate(
        (datetime(2026, 1, 1, tzinfo=UTC), datetime(2033, 3, 3, tzinfo=UTC))
    ):
        rig = Rig(tmp_path / str(index), clock_start=start)
        inner = ImplementerAgent(lookup(fake_capability(FakeRecipe())))
        chaos = ChaosAgent(inner, "secret_leak")
        result = chaos.run(rig.context(chaos, "implement", NODE, params=ASSIGNMENT))
        hashes.append(hash_jsonable([d.content for d in result.artifacts]))
    assert hashes[0] == hashes[1]


# ----------------------------------------------------------------------------- the real recipes


def real_group(capability: Any, group_id: str) -> Any:
    return next(g for g in capability.recipe.groups(default_params(capability)) if g.id == group_id)


def verdict(rig: Rig, capability: Any, group: Any, agent: Agent, ctx: AgentContext) -> Any:
    cs = changeset_of(agent.run(ctx))
    profile = agent.autonomy.narrow(AutonomyProfile(write_globs=group.write_scope))
    return evaluate(
        cs,
        rig.baseline,
        node_id=group_node(capability, group),
        profile=profile,
        config=LINKCHECK_POLICY,
    )


def test_secret_leak_on_a_real_group_is_denied_by_sec001(tmp_path: Path) -> None:
    capability = real_capability("linkcheck")
    group = real_group(capability, "guard")
    rig = Rig(tmp_path)
    chaos = ChaosAgent(ImplementerAgent(lookup(capability)), "secret_leak")
    ctx = rig.context(
        chaos,
        "implement",
        group_node(capability, group),
        params=group_assignment(capability, group, default_params(capability)),
        node_autonomy=AutonomyProfile(write_globs=group.write_scope),
    )
    report = verdict(rig, capability, group, chaos, ctx)
    assert rule_ids(report) == {"SEC-001"} and report.enforcement is Enforcement.DENY


def test_migration_edit_only_hits_groups_whose_scope_contains_migrations(tmp_path: Path) -> None:
    capability = real_capability("max_clicks")
    rig = Rig(tmp_path, product_files())
    chaos = ChaosAgent(ImplementerAgent(lookup(capability)), "migration_edit", 99)
    storage = run_group(rig, chaos, capability, real_group(capability, "storage"))
    edited = [o.path for o in changeset_of(storage).ops if o.kind is FileOpKind.MODIFY]
    assert any("/migrations/" in p for p in edited)
    elsewhere = run_group(rig, chaos, capability, real_group(capability, "service"))
    assert "not applied (no applicable target" in elsewhere.rationale


def test_the_brownfield_flow_denied_denied_again_then_repaired_by_the_fallback(
    tmp_path: Path,
) -> None:
    """CHG-001 deny -> retry with findings (still faulty) -> exhausted -> the fallback repairs."""
    capability = real_capability("max_clicks")
    group = real_group(capability, "storage")
    rig = Rig(tmp_path, product_files())
    look = lookup(capability)
    primary = ChaosAgent(ImplementerAgent(look), "migration_edit", 99)
    fallback = SafeImplementerAgent(look)
    own = ("CHG-001",)
    flow: list[tuple[Agent, int, bool, tuple[str, ...]]] = [
        (primary, 1, False, ()),
        (primary, 2, False, own),
        (primary, 3, False, own),
        (fallback, 4, True, own),
    ]
    denied = []
    for agent, attempt_no, is_fallback, feedback in flow:
        ctx = rig.context(
            agent,
            "implement",
            group_node(capability, group),
            params=group_assignment(capability, group, default_params(capability)),
            node_autonomy=AutonomyProfile(write_globs=group.write_scope),
            attempt_no=attempt_no,
            is_fallback=is_fallback,
            feedback=[feedback_finding(r) for r in feedback],
        )
        denied.append("CHG-001" in rule_ids(verdict(rig, capability, group, agent, ctx)))
    assert denied == [True, True, True, False]


@pytest.mark.slow
def test_a_regression_in_one_real_group_fails_the_generated_suite(tmp_path: Path) -> None:
    def wrap(group: Any, agent: Agent) -> Agent:
        return ChaosAgent(agent, "regression") if group.id == "core" else agent

    chain = build_chain(tmp_path / "chain", real_capability("linkcheck"), {}, wrap=wrap)
    assert chain.report["conflicts"] == []  # the fault is invisible to merging and to syntax
    runner = SandboxRunner(tmp_path / "sandbox")
    scratch = runner.new_scratch()
    try:
        materialize(OverlayView(MemoryView({}), [chain.merged]), scratch)
        result = runner.run(runner.pytest_argv(scratch, ["tests"]), cwd=scratch, timeout_s=180)
    finally:
        runner.discard(scratch)
    assert result.returncode != 0
    assert "chaos: injected regression" in result.stdout
