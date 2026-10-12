"""Regression tests for the toolbox execution modes and the baseline hand-over.

* ``run_tests`` / ``run_python`` ``mode="view"`` (default) materialise view + staged writes;
  ``mode="baseline_plus"`` materialises the BASELINE tree plus ONLY the ``overlay_paths`` taken
  from the view (the fail-to-pass check: new tests against the old code).
* ``ScopedToolbox`` / the factory accept an optional ``baseline`` (``None`` => the start view).
* ``baseline_view()`` gives a guarded read-only view of the baseline; ``baseline_view_of`` finds it.
* the sandbox's pytest argv carries ``-rA`` (one PASSED/FAILED/ERROR line per test id).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import AgentCancelled, AutonomyViolation, ToolDenied
from orchestrator.core.types import (
    Actor,
    AutonomyProfile,
    CancelToken,
    CommandResult,
    EventType,
    FileView,
    PolicyReport,
    RiskAssessment,
    RiskLevel,
)
from orchestrator.tools.sandbox import DEFAULT_MAX_OUTPUT_BYTES, SandboxRunner
from orchestrator.tools.toolbox import (
    EXEC_MODES,
    MAX_OVERLAY_PATHS,
    InvalidToolArgument,
    ScopedToolbox,
    baseline_view_of,
    make_toolbox_factory,
)
from orchestrator.tools.workspace import MemoryView


@pytest.fixture(autouse=True)
def fast_fs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scratch trees are throw-away: skip the temp-file + replace dance of ``atomic_write_bytes``
    (hundreds of ms per file on some Windows set-ups) in the unit tests that only inspect *which*
    files get materialised.  The slow end-to-end test undoes the patch and uses the real writer."""

    def plain_write(path: Path | str, data: bytes, *, mode: int = 0o600) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    monkeypatch.setattr("orchestrator.core.fsutil.atomic_write_bytes", plain_write)


BASELINE = {
    "app/__init__.py": "",
    "app/core.py": "def answer():\n    return 41\n",
    "tests/test_old.py": "def test_old():\n    assert True\n",
    "secret/notes.txt": "private\n",
}
NEW_TEST = "tests/test_new.py"
PROFILE = AutonomyProfile(
    read_globs=("app/**", "tests/**", "tools/**"),
    write_globs=("app/**", "tests/**", "tools/**"),
    can_execute=True,
    max_tool_calls=100,
    max_file_bytes=2048,
    max_total_bytes=16384,
)


class AllowPolicy:
    def evaluate(self, subject: Any) -> PolicyReport:
        return PolicyReport()

    def with_waivers(self, waivers: Any) -> AllowPolicy:
        return self

    def assess_risk(self, subject: Any) -> RiskAssessment:
        return RiskAssessment(RiskLevel.LOW)


@dataclass
class Event:
    type: EventType
    data: dict[str, Any]


class Recorder:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(
        self,
        type: EventType,
        *,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: Mapping[str, Any],
    ) -> None:
        self.events.append(Event(type, dict(data)))

    def of(self, kind: EventType) -> list[Event]:
        return [e for e in self.events if e.type == kind]


class TreeRunner(SandboxRunner):
    """Records the tree it would have run in instead of spawning a process."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.trees: list[dict[str, str]] = []
        self.argvs: list[list[str]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str,
        timeout_s: float,
        cancel: CancelToken | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> CommandResult:
        root = Path(cwd)
        self.trees.append(
            {
                p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
                for p in sorted(root.rglob("*"))
                if p.is_file()
            }
        )
        self.argvs.append(list(argv))
        return CommandResult(tuple(argv), 0, "recorded", "", 0.0)

    @property
    def leftovers(self) -> list[str]:
        return sorted(p.name for p in self.scratch_root.iterdir())


@dataclass
class Rig:
    box: ScopedToolbox
    audit: Recorder
    runner: TreeRunner
    cancel: CancelToken


def build(
    tmp_path: Path,
    *,
    view_files: Mapping[str, str] | None = None,
    baseline: Mapping[str, str] | None = BASELINE,
    **profile_overrides: Any,
) -> Rig:
    """``baseline=None``: none handed in; ``{}``: an explicit EMPTY baseline (greenfield)."""
    profile = dataclasses.replace(PROFILE, **profile_overrides) if profile_overrides else PROFILE
    runner, audit, cancel = TreeRunner(tmp_path / "scratch"), Recorder(), CancelToken()
    view = MemoryView(view_files if view_files is not None else BASELINE)
    box = ScopedToolbox(
        run_id="run-1",
        node_id="verify",
        attempt=1,
        agent_id="verifier",
        profile=profile,
        view=view,
        cancel=cancel,
        clock=FakeClock(),
        policy=AllowPolicy(),
        audit=audit,
        runner=runner,
        baseline=MemoryView(baseline) if baseline is not None else None,
    )
    return Rig(box, audit, runner, cancel)


# the view the integrated tree has: baseline + a modified module + the new test + a new module
INTEGRATED = {
    **BASELINE,
    "app/core.py": "def answer():\n    return 42\n",
    "app/extra.py": "EXTRA = 1\n",
    NEW_TEST: "def test_new():\n    assert True\n",
}


# ------------------------------------------------------------------ execution modes


def test_view_mode_is_the_default_and_materialises_view_plus_staged_writes(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    rig.box.write_text("tests/test_staged.py", "def test_staged():\n    pass\n")
    rig.box.run_tests(["tests"])
    tree = rig.runner.trees[0]
    assert (
        tree["app/core.py"] == INTEGRATED["app/core.py"]
    )  # the view's content, not the baseline's
    assert NEW_TEST in tree and "tests/test_staged.py" in tree and "app/extra.py" in tree
    assert EXEC_MODES == ("view", "baseline_plus")


def test_baseline_plus_runs_the_baseline_plus_only_the_overlay_files(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    rig.box.write_text("tests/test_staged.py", "def test_staged():\n    pass\n")
    rig.box.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST])
    tree = rig.runner.trees[0]
    assert tree[NEW_TEST] == INTEGRATED[NEW_TEST]  # overlay file: taken from the view
    assert tree["app/core.py"] == BASELINE["app/core.py"]  # everything else: the BASELINE version
    assert "app/extra.py" not in tree  # new in the view, not listed => absent
    assert "tests/test_staged.py" not in tree  # staged but not listed => absent
    assert set(tree) == {*BASELINE, NEW_TEST}
    assert "-rfEsxXp" in rig.runner.argvs[0] and rig.runner.argvs[0][-1] == NEW_TEST


def test_an_overlay_file_replaces_the_baseline_version_of_that_path(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    rig.box.run_tests(["app"], mode="baseline_plus", overlay_paths=["app/core.py"])
    tree = rig.runner.trees[0]
    assert tree["app/core.py"] == INTEGRATED["app/core.py"]
    assert tree["tests/test_old.py"] == BASELINE["tests/test_old.py"]


def test_overlay_files_may_be_staged_writes_of_this_attempt(tmp_path: Path) -> None:
    rig = build(tmp_path)
    rig.box.write_text("tools/probe.py", "print('probe')\n")
    rig.box.run_python(["tools/probe.py"], mode="baseline_plus", overlay_paths=["tools/probe.py"])
    tree = rig.runner.trees[0]
    assert tree["tools/probe.py"] == "print('probe')\n" and set(tree) == {
        *BASELINE,
        "tools/probe.py",
    }


def test_an_explicit_empty_baseline_is_greenfield_and_holds_only_the_overlay(
    tmp_path: Path,
) -> None:
    rig = build(tmp_path, view_files=INTEGRATED, baseline={})
    rig.box.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST])
    assert set(rig.runner.trees[0]) == {NEW_TEST}


def test_without_a_baseline_the_start_view_stands_in_for_it(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED, baseline=None)
    rig.box.write_text("tests/test_staged.py", "def test_staged():\n    pass\n")
    rig.box.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST])
    # documented fallback: baseline := start view (staged writes are not part of it)
    assert set(rig.runner.trees[0]) == set(INTEGRATED)


def test_run_python_supports_both_modes_too(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files={**INTEGRATED, "tools/helper.py": "x = 1\n"})
    rig.box.run_python(["tools/helper.py"])
    assert "tools/helper.py" in rig.runner.trees[0] and "app/extra.py" in rig.runner.trees[0]
    rig.box.run_python(["tools/helper.py"], mode="baseline_plus", overlay_paths=["tools/helper.py"])
    assert set(rig.runner.trees[1]) == {*BASELINE, "tools/helper.py"}


def test_the_scratch_directory_is_always_removed(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    rig.box.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST])
    rig.box.run_tests(["tests"])
    assert rig.runner.leftovers == []


def test_the_audit_record_names_the_mode_and_the_overlay_size(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    rig.box.run_tests(["tests"])
    rig.box.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST, "app/core.py"])
    first, second = rig.audit.of(EventType.TOOL_CALL)
    assert (first.data["mode"], first.data["overlay"]) == ("view", 0)
    assert (second.data["mode"], second.data["overlay"]) == ("baseline_plus", 2)


def test_duplicate_overlay_paths_are_collapsed(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    rig.box.run_tests(
        [NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST, NEW_TEST, NEW_TEST]
    )
    assert rig.audit.of(EventType.TOOL_CALL)[0].data["overlay"] == 1


# ------------------------------------------------------------------ hostile arguments


def test_an_unknown_mode_is_refused_before_anything_is_materialised(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    for bad in ("baseline", "", "VIEW", None, 3):
        with pytest.raises(InvalidToolArgument):
            rig.box.run_tests(["tests"], mode=bad)  # type: ignore[arg-type]
    assert rig.runner.trees == [] and rig.runner.leftovers == []


def test_overlay_paths_belong_to_baseline_plus_only(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    with pytest.raises(InvalidToolArgument, match="baseline_plus"):
        rig.box.run_tests(["tests"], overlay_paths=[NEW_TEST])  # view mode + overlay: ambiguous
    for bad in (NEW_TEST, b"tests/test_new.py", 5, None):
        with pytest.raises(InvalidToolArgument):
            rig.box.run_tests(["tests"], mode="baseline_plus", overlay_paths=bad)  # type: ignore[arg-type]
    assert rig.runner.trees == []


def test_an_overlay_path_must_exist_in_the_view(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    with pytest.raises(InvalidToolArgument, match="does not exist"):
        rig.box.run_tests(["tests"], mode="baseline_plus", overlay_paths=["tests/test_ghost.py"])
    rig.box.delete("tests/test_old.py")  # staged delete: gone from the view as well
    with pytest.raises(InvalidToolArgument, match="does not exist"):
        rig.box.run_tests(["tests"], mode="baseline_plus", overlay_paths=["tests/test_old.py"])
    assert rig.runner.trees == []


@pytest.mark.parametrize(
    "hostile", ["../outside.py", "/etc/passwd", "tests\\test_new.py", "a/../b.py", "", "C:/x.py"]
)
def test_overlay_path_traversal_is_a_critical_violation_that_poisons_the_box(
    tmp_path: Path, hostile: str
) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    with pytest.raises(AutonomyViolation):
        rig.box.run_tests(["tests"], mode="baseline_plus", overlay_paths=[hostile])
    with pytest.raises(AutonomyViolation):  # poisoned: nothing works any more
        rig.box.list_files()
    assert rig.runner.trees == [] and rig.runner.leftovers == []


def test_overlay_paths_outside_the_read_scope_are_denied(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files={**INTEGRATED, "secret/key_notes.py": "x = 1\n"})
    with pytest.raises(ToolDenied, match="read_globs"):
        rig.box.run_tests(["tests"], mode="baseline_plus", overlay_paths=["secret/key_notes.py"])
    assert rig.runner.trees == []


def test_too_many_overlay_paths_are_refused(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    paths = [f"tests/t{i}.py" for i in range(MAX_OVERLAY_PATHS + 1)]
    with pytest.raises(InvalidToolArgument, match="overlay"):
        rig.box.run_tests(["tests"], mode="baseline_plus", overlay_paths=paths)


def test_an_overlay_file_that_would_replace_a_baseline_directory_is_refused_cleanly(
    tmp_path: Path,
) -> None:
    # the view has a FILE called "tests" (nonsense, but hostile trees exist); the baseline has the
    # directory tests/: materialising the overlay would have to replace a directory
    rig = build(tmp_path, view_files={**INTEGRATED, "tests": "i am a file\n"}, read_globs=("**",))
    with pytest.raises(InvalidToolArgument, match="conflicts"):
        rig.box.run_tests(["tests"], mode="baseline_plus", overlay_paths=["tests"])
    assert rig.runner.leftovers == []  # and the scratch directory is gone


def test_baseline_plus_still_needs_the_execution_permission(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED, can_execute=False)
    with pytest.raises(ToolDenied, match="execution"):
        rig.box.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST])
    assert rig.runner.trees == []


def test_a_cancelled_toolbox_runs_nothing_in_either_mode(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    rig.cancel.cancel("stop")
    with pytest.raises(AgentCancelled):
        rig.box.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST])
    assert rig.runner.trees == []


# ------------------------------------------------------------------ factory hand-over


def test_the_factory_hands_the_baseline_to_the_toolbox(tmp_path: Path) -> None:
    runner = TreeRunner(tmp_path / "scratch")
    factory = make_toolbox_factory(runner=runner)
    common: dict[str, Any] = {
        "run_id": "run-1",
        "node_id": "verify",
        "attempt": 1,
        "agent_id": "verifier",
        "profile": PROFILE,
        "view": MemoryView(INTEGRATED),
        "cancel": CancelToken(),
        "clock": FakeClock(),
        "policy": AllowPolicy(),
        "audit": Recorder(),
    }
    with_baseline = factory(**common, baseline=MemoryView(BASELINE))
    with_baseline.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST])
    assert runner.trees[0]["app/core.py"] == BASELINE["app/core.py"]
    assert "app/extra.py" not in runner.trees[0]

    without = factory(**common)  # the keyword is optional: existing callers keep working
    without.run_tests([NEW_TEST], mode="baseline_plus", overlay_paths=[NEW_TEST])
    assert runner.trees[1]["app/core.py"] == INTEGRATED["app/core.py"]  # start view as baseline


# ------------------------------------------------------------------ baseline_view


def test_baseline_view_reads_the_baseline_not_the_view_or_staged_writes(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED)
    rig.box.write_text("app/staged.py", "S = 1\n")
    base = rig.box.baseline_view()
    assert isinstance(base, FileView)
    assert base.read_text("app/core.py") == BASELINE["app/core.py"]
    assert rig.box.read_text("app/core.py") == INTEGRATED["app/core.py"]  # the view is unchanged
    assert base.exists("app/core.py") and not base.exists("app/extra.py")
    assert not base.exists("app/staged.py")
    assert base.list_files("app/**") == ["app/__init__.py", "app/core.py"]
    with pytest.raises(FileNotFoundError):
        base.read_text("app/extra.py")


def test_baseline_view_falls_back_to_the_start_view_without_a_baseline(tmp_path: Path) -> None:
    rig = build(tmp_path, view_files=INTEGRATED, baseline=None)
    rig.box.write_text("app/staged.py", "S = 1\n")
    base = rig.box.baseline_view()
    assert base.read_text("app/core.py") == INTEGRATED["app/core.py"]
    assert not base.exists("app/staged.py")  # staged writes never leak into the baseline


def test_baseline_view_obeys_scope_budget_and_audit(tmp_path: Path) -> None:
    rig = build(tmp_path, max_tool_calls=4)
    base = rig.box.baseline_view()
    assert base.read_text("app/core.py") == BASELINE["app/core.py"]
    assert base.list_files("**") == [
        "app/__init__.py",
        "app/core.py",
        "tests/test_old.py",
    ]  # secret/** is outside read_globs: never listed
    with pytest.raises(ToolDenied, match="read_globs"):
        base.read_text("secret/notes.txt")
    tools = [e.data["tool"] for e in rig.audit.of(EventType.TOOL_CALL)]
    assert tools == ["baseline_read_text", "baseline_list_files"]
    assert rig.box.usage()["tool_calls"] == 3  # every baseline read is a counted call
    base.exists("app/core.py")  # the 4th call: the last one the budget allows
    with pytest.raises(Exception, match="budget"):
        base.exists("app/core.py")


def test_baseline_view_refuses_unsafe_paths_and_big_files(tmp_path: Path) -> None:
    big = {**BASELINE, "app/big.py": "x" * 5000}
    rig = build(tmp_path, view_files=big, baseline=big)
    base = rig.box.baseline_view()
    with pytest.raises(ToolDenied, match="max_file_bytes"):
        base.read_text("app/big.py")
    with pytest.raises(AutonomyViolation):
        base.read_text("../escape.txt")
    with pytest.raises(AutonomyViolation):  # poisoned afterwards
        base.list_files("**")


def test_baseline_view_honours_the_cancel_token(tmp_path: Path) -> None:
    rig = build(tmp_path)
    base = rig.box.baseline_view()
    rig.cancel.cancel("stop")
    with pytest.raises(AgentCancelled):
        base.read_text("app/core.py")
    with pytest.raises(AgentCancelled):
        rig.box.baseline_view()


def test_baseline_view_of_finds_the_extension_and_tolerates_its_absence(tmp_path: Path) -> None:
    rig = build(tmp_path)
    found = baseline_view_of(rig.box)
    assert found is not None and found.read_text("app/core.py") == BASELINE["app/core.py"]
    assert baseline_view_of(object()) is None
    assert baseline_view_of(MemoryView({"a.txt": "x"})) is None

    class Broken:
        def baseline_view(self) -> str:
            return "not a view"

    assert baseline_view_of(Broken()) is None


# ------------------------------------------------------------------ real sandbox (slow)


@pytest.mark.slow
def test_fail_to_pass_end_to_end_with_the_real_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()  # the real atomic writer, the real sandbox
    baseline = {"calc/__init__.py": "", "tests/test_base.py": "def test_base():\n    assert True\n"}
    integrated = {
        **baseline,
        "calc/ops.py": "def add(a, b):\n    return a + b\n",
        "tests/test_ops.py": (
            "from calc.ops import add\n\n\n"
            "def test_add():\n    assert add(1, 2) == 3\n\n\n"
            "def test_add_negative():\n    assert add(-1, -2) == -3\n"
        ),
    }
    runner = SandboxRunner(tmp_path / "scratch")
    box = ScopedToolbox(
        run_id="run-1",
        node_id="verify",
        attempt=1,
        agent_id="verifier",
        profile=PROFILE,
        view=MemoryView(integrated),
        cancel=CancelToken(),
        clock=FakeClock(),
        policy=AllowPolicy(),
        audit=Recorder(),
        runner=runner,
        baseline=MemoryView(baseline),
    )
    on_view = box.run_tests(["tests"], timeout_s=120)
    assert on_view.returncode == 0, on_view.stdout + on_view.stderr
    assert "PASSED tests/test_ops.py::test_add" in on_view.stdout  # -rA: one line per test id
    assert "PASSED tests/test_base.py::test_base" in on_view.stdout

    on_baseline = box.run_tests(
        ["tests/test_ops.py"],
        timeout_s=120,
        mode="baseline_plus",
        overlay_paths=["tests/test_ops.py"],
    )
    assert on_baseline.returncode != 0  # the new tests FAIL against the old code
    assert "ERROR tests/test_ops.py" in on_baseline.stdout
    assert "PASSED" not in on_baseline.stdout
    assert sorted(p.name for p in runner.scratch_root.iterdir()) == []
