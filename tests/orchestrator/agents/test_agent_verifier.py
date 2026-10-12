"""Tests for the verifier agent.

Fast tests drive the agent through a ``ScriptedToolbox`` (real reads, writes, policy and audit;
``run_tests`` / ``run_python`` answer from a script); the ``slow`` tests run the real sandbox on a
tiny synthetic project.  The rig is ``delivery_support.Rig`` (engine-like context, declared kinds
only).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.verifier import (
    MAX_AUTHORED_FILES,
    OPENAPI_HELPER,
    VerifierAgent,
    covers_by_test,
    helper_source,
    parse_pytest_output,
)
from orchestrator.core.errors import (
    AgentCancelled,
    AgentError,
    AgentTimeout,
    ToolDenied,
    TransientAgentError,
)
from orchestrator.core.types import AgentContext, AgentResult, CancelToken, CommandResult
from tests.orchestrator.agents.delivery_support import (
    FakeRecipe,
    Rig,
    assert_valid,
    command_result,
    content_hashes,
    fake_capability,
    install_capabilities,
    payload,
    plain_writes,
    requirements_payload,
    task_plan_payload,
)

TEST_FILE = "tests/test_ops.py"
IDS = (f"{TEST_FILE}::test_add", f"{TEST_FILE}::test_add_zero")
IMPL = {"calc/__init__.py": "", "calc/ops.py": "def add(a, b):\n    return a + b\n"}
TESTS_SRC = (
    "from calc.ops import add\n\n\n"
    "def test_add():\n    '''Covers: R-001, AC-001-1'''\n    assert add(1, 2) == 3\n\n\n"
    "def test_add_zero():\n    '''Covers: R-001, AC-001-2'''\n    assert add(0, 0) == 0\n"
)
WORD = {"PASSED": "passed", "FAILED": "failed", "XFAIL": "xfailed", "XPASS": "xpassed"}
HELPER = OPENAPI_HELPER
APP_CHANGES = [{"method": "GET", "path": "/x", "change": "add", "breaking": False}]
OLD_SPEC = {
    "openapi": "3.1.0",
    "paths": {"/a": {"get": {"responses": {"200": {"description": "x"}}}}},
}
NEW_SPEC = {
    "openapi": "3.1.0",
    "paths": {
        "/a": {"get": {"responses": {"200": {"description": "x"}}}},
        "/b": {"get": {"responses": {"200": {"description": "y"}}}},
    },
}
GONE_SPEC = {"openapi": "3.1.0", "paths": {}}


# ------------------------------------------------------------------ helpers


def pytest_stdout(entries: Sequence[tuple[str, str]], *, skipped: int = 0, noise: str = "") -> str:
    """Realistic ``pytest -q -rA`` stdout; ``entries`` = (PASSED|FAILED|ERROR|XFAIL|XPASS, id)."""
    counts: Counter[str] = Counter()
    for outcome, _ in entries:
        counts["error" if outcome == "ERROR" else WORD[outcome]] += 1
    if skipped:
        counts["skipped"] += skipped
    if not counts:
        return f"{noise}\nno tests ran in 0.01s\n"
    parts = []
    for word in ("failed", "passed", "skipped", "xfailed", "xpassed", "error"):
        if counts[word]:
            label = "errors" if word == "error" and counts[word] > 1 else word
            parts.append(f"{counts[word]} {label}")
    header = "=" * 20 + " short test summary info " + "=" * 20
    lines = [noise, header, *(f"{o} {n}" for o, n in entries), ", ".join(parts) + " in 0.05s"]
    return "\n".join(lines) + "\n"


def run_result(
    entries: Sequence[tuple[str, str]], rc: int | None = None, **kwargs: Any
) -> CommandResult:
    if rc is None:
        broken = any(o in ("FAILED", "ERROR") for o, _ in entries)
        collection = any(o == "ERROR" and "::" not in n for o, n in entries)
        rc = 2 if collection else 1 if broken else 0
    return command_result(pytest_stdout(entries, **kwargs), rc)


GREEN = run_result([("PASSED", IDS[0]), ("PASSED", IDS[1])])
FILE_ERROR = run_result([("ERROR", TEST_FILE)])


def handler(
    view: CommandResult = GREEN, baseline: CommandResult = FILE_ERROR
) -> Callable[[Sequence[str], str, Sequence[str]], CommandResult]:
    def answer(targets: Sequence[str], mode: str, overlay: Sequence[str]) -> CommandResult:
        return view if mode == "view" else baseline

    return answer


def make_rig(
    tmp_path: Path,
    integrated: dict[str, str] | None = None,
    *,
    baseline: dict[str, str] | None = None,
    requirements: dict[str, Any] | None = None,
    clock_start: datetime | None = None,
) -> Rig:
    rig = Rig(tmp_path, baseline_files=baseline or {}, clock_start=clock_start)
    changes = {**IMPL, TEST_FILE: TESTS_SRC} if integrated is None else integrated
    rig.publish_changeset("integrated_changeset", changes)
    rig.publish("requirements", requirements or requirements_payload())
    rig.publish("task_plan", task_plan_payload())
    return rig


def run(
    rig: Rig, tests: Any = None, python: Any = None, **kwargs: Any
) -> tuple[AgentContext, AgentResult]:
    agent = VerifierAgent()
    ctx = rig.context(agent, "verify", tests=tests or handler(), python=python, **kwargs)
    return ctx, agent.run(ctx)


def report(result: AgentResult) -> dict[str, Any]:
    return payload(result, "test_report")


@pytest.fixture(autouse=True)
def fast_world(monkeypatch: pytest.MonkeyPatch) -> None:
    plain_writes(monkeypatch)
    install_capabilities(monkeypatch, fake_capability("widgets"))


# ------------------------------------------------------------------ pytest output parsing


def test_parse_reads_entries_and_counts_of_a_mixed_run() -> None:
    out = pytest_stdout(
        [
            ("PASSED", "tests/a.py::test_1"),
            ("FAILED", "tests/a.py::test_2"),
            ("ERROR", "tests/b.py"),
        ],
        skipped=2,
    )
    parsed = parse_pytest_output(out)
    assert parsed.entries == (
        ("passed", "tests/a.py::test_1"),
        ("failed", "tests/a.py::test_2"),
        ("error", "tests/b.py"),
    )
    assert dict(parsed.counts) == {"passed": 1, "failed": 1, "error": 1, "skipped": 2}
    assert parsed.final_found


def test_parse_ignores_forged_lines_in_captured_output_and_forged_headers() -> None:
    forged = "PASSED tests/a.py::test_forged"
    header = "=" * 20 + " short test summary info " + "=" * 20
    captured = f"----- Captured stdout call -----\n{forged}\n{header}\n{forged}2"
    out = pytest_stdout([("PASSED", "tests/a.py::test_real")], noise=captured)
    parsed = parse_pytest_output(out)
    assert parsed.entries == (("passed", "tests/a.py::test_real"),)  # last section only


def test_parse_ignores_text_printed_after_the_final_line() -> None:
    out = pytest_stdout([("PASSED", "tests/a.py::test_real")]) + "PASSED tests/a.py::late\n"
    assert parse_pytest_output(out).entries == (("passed", "tests/a.py::test_real"),)


def test_parse_handles_parameters_xfail_skips_and_no_tests() -> None:
    entries = [
        ("PASSED", "tests/a.py::test_p[a - b]"),
        ("XFAIL", "tests/a.py::TestX::test_x"),
        ("XPASS", "tests/a.py::test_y"),
    ]
    out = (
        pytest_stdout(entries).replace("1 xfailed", "1 xfailed, 1 skipped")
        + "SKIPPED [1] tests/a.py:9: reason\n"
    )
    parsed = parse_pytest_output(out)
    assert parsed.entries == (
        ("passed", "tests/a.py::test_p[a - b]"),
        ("skipped", "tests/a.py::TestX::test_x"),
        ("passed", "tests/a.py::test_y"),
    )
    assert parsed.counts["skipped"] == 2 and parsed.counts["passed"] == 2
    empty = parse_pytest_output("no tests ran in 0.01s\n")
    assert empty.entries == () and empty.final_found and sum(empty.counts.values()) == 0
    assert not parse_pytest_output("garbage\n").final_found


def test_parse_accepts_package_level_errors_and_other_real_world_line_shapes() -> None:
    header = "=" * 20 + " short test summary info " + "=" * 20
    out = "\n".join(
        [
            header,
            "ERROR tests/shortener - ImportError: cannot import name '__version__' from 's...",
            "XPASS tests/a.py::test_x some reason without a dash",
            "FAILED tests/ünï_code_t/t.py::TestG::test_é[p 1] - assert [1] == 2",
            "PASSED tests/a.py::test_ok",
            "2 passed, 1 failed, 1 error in 0.50s",
        ]
    )
    parsed = parse_pytest_output(out + "\n")
    assert parsed.entries == (
        ("error", "tests/shortener"),
        ("passed", "tests/a.py::test_x"),
        ("failed", "tests/ünï_code_t/t.py::TestG::test_é[p 1]"),
        ("passed", "tests/a.py::test_ok"),
    )
    assert dict(parsed.counts) == {"passed": 2, "failed": 1, "error": 1, "skipped": 0}


def test_covers_come_from_docstring_lines_functions_and_classes() -> None:
    source = (
        "def test_a():\n    '''Covers: R-001, AC-001-2\n\n    Covers: R-002'''\n\n"
        "def test_plain():\n    pass\n\n"
        "class TestG:\n    '''Covers: R-009'''\n"
        "    def test_inherits(self):\n        pass\n"
        "    def test_own(self):\n        '''Covers: AC-009-1, not-an-id, R-1'''\n"
    )
    found = covers_by_test(source)
    assert found["test_a"] == {"R-001", "AC-001-2", "R-002"}
    assert found["test_plain"] == frozenset()
    assert found["TestG::test_inherits"] == {"R-009"}
    assert found["TestG::test_own"] == {"AC-009-1"}  # its own line wins; junk ids are dropped


@pytest.mark.parametrize(
    "source",
    ["def test_(:\n", "x = " + "(" * 5000 + ")" * 5000, "x = 1" + "+1" * 60_000, "\x00", "é" * 10],
    ids=["syntax-error", "nested-parens", "deep-binop", "nul", "unicode"],
)
def test_covers_survive_hostile_source(source: str) -> None:
    assert isinstance(covers_by_test(source), dict)  # never raises


# ------------------------------------------------------------------ report content


def test_green_greenfield_report(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    ctx, result = run(rig)
    rep = report(result)
    assert_valid(result)
    assert rep["passed"] is True and rep["total"] == 2 and rep["failed"] == 0 and rep["errors"] == 0
    assert [t["nodeid"] for t in rep["tests"]] == list(IDS)
    assert rep["tests"][0] == {
        "nodeid": IDS[0],
        "outcome": "passed",
        "covers": ["AC-001-1", "R-001"],
    }
    assert rep["fail_to_pass"] == {"checked": True, "ok": True, "tests": list(IDS), "vacuous": []}
    assert rep["baseline"] == {"ran": True, "passed": False, "failed_tests": list(IDS)}
    assert rep["perf"] == {"checked": False, "budget_ms": None, "ok": None}
    assert rep["api_compat"] == {"checked": False, "breaking": [], "additive": []}
    assert ctx.toolbox.calls == [  # type: ignore[attr-defined]
        ("run_tests", ("tests",), "view", ()),
        ("run_tests", (TEST_FILE,), "baseline_plus", (TEST_FILE,)),
    ]


def test_the_report_has_no_times_or_durations_only_metadata_does(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    _, result = run(rig)
    text = json.dumps(report(result)).lower()
    assert not any(word in text for word in ("duration", "seconds", "timestamp", '_s"', "elapsed"))
    meta = result.artifacts[0].metadata
    assert meta["pytest_runs"] == 2 and "duration_s" in meta and meta["targets"] == ["tests"]


def test_a_failing_test_makes_the_report_not_passed_and_not_fail_to_pass(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    broken = run_result([("PASSED", IDS[0]), ("FAILED", IDS[1])])
    _, result = run(rig, handler(view=broken))
    rep = report(result)
    assert rep["passed"] is False and rep["failed"] == 1
    assert rep["fail_to_pass"]["ok"] is False and rep["fail_to_pass"]["tests"] == [IDS[0]]


def test_a_collection_error_in_the_suite_is_reported_with_the_files_covers(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    _, result = run(rig, handler(view=FILE_ERROR))
    rep = report(result)
    assert rep["passed"] is False and rep["errors"] == 1
    assert rep["tests"] == [
        {"nodeid": TEST_FILE, "outcome": "error", "covers": ["AC-001-1", "AC-001-2", "R-001"]}
    ]
    assert rep["fail_to_pass"]["ok"] is False  # no test id of the authored file passed


def test_a_teardown_error_beats_the_pass_of_the_same_test(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    mixed = run_result([("PASSED", IDS[0]), ("ERROR", IDS[0]), ("PASSED", IDS[1])])
    _, result = run(rig, handler(view=mixed))
    rep = report(result)
    outcomes = {t["nodeid"]: t["outcome"] for t in rep["tests"]}
    assert outcomes == {IDS[0]: "error", IDS[1]: "passed"} and rep["passed"] is False


def test_xfail_counts_as_skipped_and_xpass_as_passed(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    out = run_result([("PASSED", IDS[0]), ("XFAIL", IDS[1])], skipped=0)
    _, result = run(rig, handler(view=out))
    rep = report(result)
    assert rep["skipped"] == 1 and rep["passed"] is True
    assert {t["nodeid"]: t["outcome"] for t in rep["tests"]}[IDS[1]] == "skipped"


def test_no_tests_directory_means_no_run_and_a_failing_report(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, integrated=IMPL)
    ctx, result = run(rig)
    rep = report(result)
    assert ctx.toolbox.calls == []  # type: ignore[attr-defined]
    assert rep["passed"] is False and rep["total"] == 0 and rep["tests"] == []
    assert rep["fail_to_pass"]["checked"] is False and rep["fail_to_pass"]["ok"] is False


def test_covers_of_a_test_file_with_a_syntax_error_are_empty_not_fatal(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, integrated={**IMPL, TEST_FILE: "def test_add(:\n"})
    _, result = run(rig)
    assert all(t["covers"] == [] for t in report(result)["tests"])


# ------------------------------------------------------------------ fail-to-pass


def test_authored_tests_that_already_pass_on_the_baseline_are_vacuous(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    _, result = run(rig, handler(baseline=GREEN))
    f2p = report(result)["fail_to_pass"]
    assert f2p["ok"] is False and f2p["tests"] == [] and f2p["vacuous"] == list(IDS)
    assert report(result)["baseline"]["passed"] is True


def test_per_test_failures_on_the_baseline_count_as_fail_to_pass(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    failing = run_result([("FAILED", IDS[0]), ("FAILED", IDS[1])])
    _, result = run(rig, handler(baseline=failing))
    assert report(result)["fail_to_pass"] == {
        "checked": True,
        "ok": True,
        "tests": list(IDS),
        "vacuous": [],
    }


def test_a_mixed_baseline_is_not_ok_and_names_the_vacuous_test(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    mixed = run_result([("FAILED", IDS[0]), ("PASSED", IDS[1])])
    _, result = run(rig, handler(baseline=mixed))
    f2p = report(result)["fail_to_pass"]
    assert f2p["ok"] is False and f2p["tests"] == [IDS[0]] and f2p["vacuous"] == [IDS[1]]


def test_files_not_reached_because_of_a_collection_error_are_rerun(tmp_path: Path) -> None:
    second = "tests/test_more.py"
    more_src = TESTS_SRC.replace("test_add", "test_more").replace("AC-001", "AC-002")
    rig = make_rig(tmp_path, integrated={**IMPL, TEST_FILE: TESTS_SRC, second: more_src})
    more_ids = [f"{second}::test_more", f"{second}::test_more_zero"]
    both = run_result([("PASSED", i) for i in (*IDS, *more_ids)])

    def answer(targets: Sequence[str], mode: str, overlay: Sequence[str]) -> CommandResult:
        if mode == "view":
            return both
        if TEST_FILE in targets:  # round 1: this file breaks collection, the other is not reached
            return FILE_ERROR
        return run_result([("FAILED", i) for i in more_ids])

    ctx, result = run(rig, answer)
    assert report(result)["fail_to_pass"]["ok"] is True
    base_calls = [c for c in ctx.toolbox.calls if c[2] == "baseline_plus"]  # type: ignore[attr-defined]
    assert [c[3] for c in base_calls] == [(second, TEST_FILE), (second,)]


def test_a_package_level_collection_error_concludes_the_files_below_it(tmp_path: Path) -> None:
    inside = "tests/pkg/test_a.py"
    outside = "tests/test_b.py"
    ids_a = [f"{inside}::test_add", f"{inside}::test_add_zero"]
    ids_b = [f"{outside}::test_add", f"{outside}::test_add_zero"]
    rig = make_rig(
        tmp_path,
        integrated={**IMPL, inside: TESTS_SRC, outside: TESTS_SRC, "tests/pkg/__init__.py": ""},
    )
    package_error = command_result(
        pytest_stdout([]).replace(
            "no tests ran in 0.01s",
            "=" * 20 + " short test summary info " + "=" * 20 + "\n"
            "ERROR tests/pkg - ImportError: boom\n1 error in 0.01s",
        ),
        2,
    )

    def answer(targets: Sequence[str], mode: str, overlay: Sequence[str]) -> CommandResult:
        if mode == "view":
            return run_result([("PASSED", i) for i in (*ids_a, *ids_b)])
        if inside in targets:  # round 1: the package breaks collection, the other file is not run
            return package_error
        return run_result([("FAILED", i) for i in ids_b])

    ctx, result = run(rig, answer)
    f2p = report(result)["fail_to_pass"]
    assert f2p["ok"] is True and f2p["tests"] == sorted([*ids_a, *ids_b])
    base_calls = [c for c in ctx.toolbox.calls if c[2] == "baseline_plus"]  # type: ignore[attr-defined]
    assert [c[3] for c in base_calls] == [(inside, outside), (outside,)]


def test_a_baseline_that_never_concludes_is_not_ok_and_is_bounded(tmp_path: Path) -> None:
    files = {f"tests/test_f{i}.py": TESTS_SRC for i in range(6)}
    rig = make_rig(tmp_path, integrated={**IMPL, **files})
    ids = [f"{p}::{n}" for p in files for n in ("test_add", "test_add_zero")]

    def answer(targets: Sequence[str], mode: str, overlay: Sequence[str]) -> CommandResult:
        if mode == "view":
            return run_result([("PASSED", i) for i in ids])
        return run_result([("ERROR", targets[0])])  # concludes exactly one file per round

    ctx, result = run(rig, answer)
    assert report(result)["fail_to_pass"]["ok"] is False
    assert len([c for c in ctx.toolbox.calls if c[2] == "baseline_plus"]) == 4  # type: ignore[attr-defined]

    ctx2, result2 = run(
        rig, handler(baseline=run_result([], rc=5))
    )  # nothing learned: stop at once
    assert report(result2)["fail_to_pass"]["ok"] is False
    assert len([c for c in ctx2.toolbox.calls if c[2] == "baseline_plus"]) == 1  # type: ignore[attr-defined]


def test_without_authored_test_files_nothing_is_checked(tmp_path: Path) -> None:
    old = {"tests/test_old.py": "def test_old():\n    assert True\n"}
    rig = make_rig(tmp_path, integrated=IMPL, baseline=old)
    only_old = run_result([("PASSED", "tests/test_old.py::test_old")])
    ctx, result = run(rig, handler(view=only_old))
    rep = report(result)
    assert rep["fail_to_pass"] == {"checked": False, "ok": False, "tests": [], "vacuous": []}
    assert rep["baseline"] == {"ran": False, "passed": None, "failed_tests": []}
    assert [c[2] for c in ctx.toolbox.calls] == ["view"]  # type: ignore[attr-defined]


def test_modified_existing_test_files_are_not_authored(tmp_path: Path) -> None:
    old = {"tests/test_old.py": "def test_old():\n    assert True\n"}
    changed = {**IMPL, "tests/test_old.py": "def test_old():\n    assert 1 == 1\n"}
    rig = make_rig(tmp_path, integrated=changed, baseline=old)
    only_old = run_result([("PASSED", "tests/test_old.py::test_old")])
    ctx, result = run(rig, handler(view=only_old))
    assert report(result)["fail_to_pass"]["checked"] is False
    assert [c[2] for c in ctx.toolbox.calls] == ["view"]  # type: ignore[attr-defined]


def test_helpers_and_non_test_files_under_tests_are_not_authored_tests(tmp_path: Path) -> None:
    extra = {"tests/helpers.py": "X = 1\n", "tests/data.txt": "x\n", "other/test_x.py": "x = 1\n"}
    rig = make_rig(tmp_path, integrated={**IMPL, TEST_FILE: TESTS_SRC, **extra})
    ctx, _ = run(rig)
    overlays = [c[3] for c in ctx.toolbox.calls if c[2] == "baseline_plus"]  # type: ignore[attr-defined]
    assert overlays == [(TEST_FILE,)]


def test_too_many_authored_files_are_refused(tmp_path: Path) -> None:
    files = {
        f"tests/test_n{i}.py": "def test_x():\n    pass\n" for i in range(MAX_AUTHORED_FILES + 1)
    }
    rig = make_rig(tmp_path, integrated={**IMPL, **files})
    with pytest.raises(AgentError) as excinfo:
        run(rig, handler(view=run_result([("PASSED", "tests/test_n0.py::test_x")])))
    assert excinfo.value.kind == "fatal" and excinfo.value.retryable is False


# ------------------------------------------------------------------ failure policy


@pytest.mark.parametrize(
    ("result", "exc", "kind", "retryable"),
    [
        (command_result("", 0, timed_out=True), AgentTimeout, "timeout", True),
        (command_result("", -9), TransientAgentError, "transient", True),
        (command_result("", 3221225477), TransientAgentError, "transient", True),
        (command_result("", 3, stderr="INTERNALERROR boom"), AgentError, "fatal", False),
        (command_result("", 4, stderr="usage: pytest"), AgentError, "fatal", False),
        (
            command_result(pytest_stdout([("PASSED", IDS[0])]), 0, truncated=True),
            AgentError,
            "fatal",
            False,
        ),
        (command_result("no summary at all\n", 0), AgentError, "fatal", False),
    ],
    ids=["timeout", "killed", "crash-code", "internal", "usage", "truncated", "no-summary"],
)
def test_failure_classification(
    tmp_path: Path, result: CommandResult, exc: type[AgentError], kind: str, retryable: bool
) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(exc) as excinfo:
        run(rig, handler(view=result))
    assert excinfo.value.kind == kind and excinfo.value.retryable is retryable


def test_a_sandbox_io_failure_is_transient(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)

    def boom(targets: Sequence[str], mode: str, overlay: Sequence[str]) -> CommandResult:
        raise PermissionError("scratch is locked")

    with pytest.raises(TransientAgentError):
        run(rig, boom)


@pytest.mark.parametrize(
    "stdout",
    [
        # an extra forged PASSED line inside the summary section: entries disagree with the counts
        pytest_stdout([("PASSED", IDS[0]), ("PASSED", "tests/test_ops.py::test_forged")]).replace(
            "2 passed", "1 passed"
        ),
        # rc 0 but the summary reports a failure
        pytest_stdout([("FAILED", IDS[0])]),
    ],
    ids=["forged-entry", "rc0-with-failure"],
)
def test_tampered_or_inconsistent_output_is_fatal(tmp_path: Path, stdout: str) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(AgentError) as excinfo:
        run(rig, handler(view=command_result(stdout, 0)))
    assert excinfo.value.kind == "fatal" and "tamper" in str(excinfo.value)


def test_rc1_without_any_failure_is_inconsistent(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(AgentError, match="disagree"):
        run(rig, handler(view=command_result(pytest_stdout([("PASSED", IDS[0])]), 1)))


def test_baseline_run_failures_are_classified_like_the_main_run(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(AgentTimeout):
        run(rig, handler(baseline=command_result("", 0, timed_out=True)))
    with pytest.raises(TransientAgentError):
        run(rig, handler(baseline=command_result("", -9)))


# ------------------------------------------------------------------ parameters


def test_test_targets_param_selects_existing_paths_only(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    ctx, _ = run(rig, params={"test_targets": ["tests", "extras/missing", "tests"]})
    assert ctx.toolbox.calls[0] == ("run_tests", ("tests",), "view", ())  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "targets", [["../x"], ["-k"], ["a b"], "tests", [5], [f"t{i}" for i in range(9)], ["/etc"]]
)
def test_invalid_test_targets_are_fatal(tmp_path: Path, targets: Any) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(AgentError) as excinfo:
        run(rig, params={"test_targets": targets})
    assert excinfo.value.kind == "fatal"


@pytest.mark.parametrize("value", [0, -1, "x", True, 601, float("nan")])
def test_invalid_timeouts_are_fatal(tmp_path: Path, value: Any) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(AgentError):
        run(rig, params={"test_timeout_s": value})


# ------------------------------------------------------------------ perf


def perf_rig(tmp_path: Path, budget_in_requirements: float | None = None) -> Rig:
    reqs = requirements_payload()
    if budget_in_requirements is not None:
        reqs["capabilities"][0]["params"] = {"perf_budget_ms": budget_in_requirements}
    return make_rig(tmp_path, requirements=reqs)


PERF_ID = "tests/test_perf.py::test_perf_budget"


def test_perf_is_recorded_only_with_a_budget_and_a_perf_test(tmp_path: Path) -> None:
    rig = perf_rig(tmp_path)
    with_perf = run_result([("PASSED", IDS[0]), ("PASSED", IDS[1]), ("PASSED", PERF_ID)])
    _, with_budget = run(rig, handler(view=with_perf), params={"perf_budget_ms": 50})
    assert report(with_budget)["perf"] == {"checked": True, "budget_ms": 50.0, "ok": True}
    _, no_budget = run(rig, handler(view=with_perf))
    assert report(no_budget)["perf"] == {"checked": False, "budget_ms": None, "ok": None}
    _, no_test = run(rig, params={"perf_budget_ms": 50})
    assert report(no_test)["perf"] == {"checked": False, "budget_ms": 50.0, "ok": None}


def test_a_failing_perf_test_is_not_ok_and_budget_may_come_from_requirements(
    tmp_path: Path,
) -> None:
    rig = perf_rig(tmp_path, budget_in_requirements=75)
    out = run_result([("PASSED", IDS[0]), ("PASSED", IDS[1]), ("FAILED", PERF_ID)])
    _, result = run(rig, handler(view=out))
    assert report(result)["perf"] == {"checked": True, "budget_ms": 75.0, "ok": False}
    assert report(result)["passed"] is False


@pytest.mark.parametrize("value", [0, -5, True, "5", 4e9, float("nan")])
def test_invalid_perf_budgets_are_fatal(tmp_path: Path, value: Any) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(AgentError):
        run(rig, params={"perf_budget_ms": value})


# ------------------------------------------------------------------ api compatibility


def api_rig(tmp_path: Path, *, baseline: dict[str, str] | None = None) -> Rig:
    product = {"shortener/__init__.py": "", "shortener/main.py": "def create_app():\n    pass\n"}
    return make_rig(
        tmp_path, integrated={**IMPL, **product, TEST_FILE: TESTS_SRC}, baseline=baseline
    )


def spec_handler(
    new: Any, old: Any
) -> Callable[[Sequence[str], str, Sequence[str]], CommandResult]:
    def answer(args: Sequence[str], mode: str, overlay: Sequence[str]) -> CommandResult:
        spec = new if mode == "view" else old
        if isinstance(spec, CommandResult):
            return spec
        return command_result(json.dumps(spec, sort_keys=True) + "\n")

    return answer


@pytest.fixture
def api_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    install_capabilities(
        monkeypatch, fake_capability("widgets", FakeRecipe(api_changes=APP_CHANGES))
    )


@pytest.mark.usefixtures("api_capability")
def test_additive_api_change_is_diffed_through_the_staged_helper(tmp_path: Path) -> None:
    rig = api_rig(tmp_path)
    ctx, result = run(rig, python=spec_handler(NEW_SPEC, OLD_SPEC))
    api = report(result)["api_compat"]
    assert api["checked"] is True and api["breaking"] == []
    assert any("/b" in line for line in api["additive"])
    calls = [c for c in ctx.toolbox.calls if c[0] == "run_python"]  # type: ignore[attr-defined]
    assert calls == [
        ("run_python", (HELPER,), "view", ()),
        ("run_python", (HELPER,), "baseline_plus", (HELPER,)),
    ]
    assert ctx.toolbox.read_text(HELPER) == helper_source("shortener.main:create_app")
    assert result.artifacts[0].metadata["api_compat_reason"] == "exported"


def test_the_helper_gives_the_shortener_a_runtime_generated_key_and_keeps_auth_on() -> None:
    source = helper_source("shortener.main:create_app")
    compile(source, "helper.py", "exec")  # valid Python
    assert 'TARGET = "shortener.main:create_app"' in source
    assert "secrets.token_urlsafe(32)" in source and "SHORTENER_API_KEYS" in source
    assert "AUTH_DISABLED" not in source  # a security control is never switched off to export
    assert "SHORTENER_OPENAPI_DUMP" not in source  # the product's strict config rejects it
    assert not re.search(r"[A-Za-z0-9_-]{24,}", source)  # no literal key
    other = helper_source("myapp.web:make")
    assert 'TARGET = "myapp.web:make"' in other
    assert "SHORTENER" not in other and "orchestrator" not in other  # self-contained, generic


@pytest.mark.usefixtures("api_capability")
def test_a_removed_path_is_a_breaking_change(tmp_path: Path) -> None:
    rig = api_rig(tmp_path)
    _, result = run(rig, python=spec_handler(GONE_SPEC, OLD_SPEC))
    api = report(result)["api_compat"]
    assert api["checked"] is True and any("/a" in line for line in api["breaking"])


@pytest.mark.usefixtures("api_capability")
@pytest.mark.parametrize(
    ("new", "old", "needle"),
    [
        (command_result("", 3, stderr="openapi dump failed: ConfigError"), OLD_SPEC, "exit 3"),
        (command_result("", 0, timed_out=True), OLD_SPEC, "timed out"),
        (command_result("{}", 0, truncated=True), OLD_SPEC, "truncated"),
        (command_result("not json at all"), OLD_SPEC, "not JSON"),
        (command_result('{"hello": 1}'), OLD_SPEC, "not an OpenAPI"),
        (NEW_SPEC, command_result("", 3, stderr="ModuleNotFoundError: shortener"), "baseline"),
    ],
    ids=["exit", "timeout", "truncated", "bad-json", "not-openapi", "greenfield-baseline"],
)
def test_an_impossible_export_is_recorded_as_unchecked_with_a_reason(
    tmp_path: Path, new: Any, old: Any, needle: str
) -> None:
    rig = api_rig(tmp_path)
    _, result = run(rig, python=spec_handler(new, old))
    assert report(result)["api_compat"] == {"checked": False, "breaking": [], "additive": []}
    assert needle in result.artifacts[0].metadata["api_compat_reason"]
    assert_valid(result)


def test_no_declared_api_change_means_no_export_at_all(tmp_path: Path) -> None:
    rig = api_rig(tmp_path)  # the default fake recipe declares no api_changes
    ctx, result = run(rig, python=spec_handler(NEW_SPEC, OLD_SPEC))
    assert [c for c in ctx.toolbox.calls if c[0] == "run_python"] == []  # type: ignore[attr-defined]
    assert report(result)["api_compat"]["checked"] is False
    assert "no capability declares" in result.artifacts[0].metadata["api_compat_reason"]


@pytest.mark.usefixtures("api_capability")
def test_an_absent_product_means_unchecked(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)  # no shortener/ package in the tree
    ctx, result = run(rig, python=spec_handler(NEW_SPEC, OLD_SPEC))
    assert "not part of the tree" in result.artifacts[0].metadata["api_compat_reason"]
    assert [c for c in ctx.toolbox.calls if c[0] == "run_python"] == []  # type: ignore[attr-defined]


@pytest.mark.usefixtures("api_capability")
def test_a_custom_app_import_is_validated_and_used(tmp_path: Path) -> None:
    rig = make_rig(
        tmp_path,
        integrated={
            **IMPL,
            "myapp/__init__.py": "",
            "myapp/web.py": "x = 1\n",
            TEST_FILE: TESTS_SRC,
        },
    )
    ctx, _ = run(
        rig, python=spec_handler(NEW_SPEC, OLD_SPEC), params={"app_import": "myapp.web:make"}
    )
    assert 'TARGET = "myapp.web:make"' in ctx.toolbox.read_text(HELPER)
    for bad in ("x; import os", "a:b:c", "../x:y", "mod:fn\n", "", 7, "mod.:fn"):
        with pytest.raises(AgentError) as excinfo:
            run(rig, python=spec_handler(NEW_SPEC, OLD_SPEC), params={"app_import": bad})
        assert excinfo.value.kind == "fatal"


def test_unknown_or_broken_capabilities_leave_api_compat_unchecked(tmp_path: Path) -> None:
    rig = Rig(tmp_path, baseline_files={})
    rig.publish_changeset("integrated_changeset", {**IMPL, TEST_FILE: TESTS_SRC})
    rig.publish("requirements", requirements_payload())
    ghost = task_plan_payload(capability="ghost")
    rig.publish("task_plan", ghost)
    _, result = run(rig)
    assert report(result)["api_compat"]["checked"] is False
    assert "ghost" in result.artifacts[0].metadata["api_compat_reason"]


def test_the_helper_is_the_only_thing_the_verifier_may_write(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    ctx = rig.context(VerifierAgent(), "verify", tests=handler())
    ctx.toolbox.write_text(f"{HELPER}", "x = 1\n")  # allowed
    with pytest.raises(ToolDenied):
        ctx.toolbox.write_text("calc/ops.py", "x = 1\n")
    assert VerifierAgent.autonomy.write_globs == ("_agentic_verify/**",)
    assert VerifierAgent.autonomy.can_execute is True


# ------------------------------------------------------------------ cross-cutting


def test_the_report_is_deterministic_across_clocks_and_runs(tmp_path: Path) -> None:
    first = make_rig(tmp_path / "a", clock_start=datetime(2026, 1, 1, tzinfo=UTC))
    second = make_rig(tmp_path / "b", clock_start=datetime(2031, 6, 7, 8, tzinfo=UTC))
    _, one = run(first)
    _, two = run(second)
    assert content_hashes(one) == content_hashes(two)
    assert one.artifacts[0].content == two.artifacts[0].content


def test_a_cancelled_attempt_raises_agent_cancelled(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    cancel = CancelToken()
    cancel.cancel("stop")
    with pytest.raises(AgentCancelled):
        run(rig, cancel=cancel)


def test_the_agent_reads_only_its_declared_inputs(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)  # requirements, task_plan, integrated_changeset: nothing else exists
    _, result = run(rig)
    assert result.usage["tool_calls"] > 0 and result.rationale


# ------------------------------------------------------------------ real sandbox (slow)


@pytest.mark.slow
def test_real_sandbox_green_run_with_fail_to_pass(tmp_path: Path) -> None:
    rig = Rig(tmp_path, with_runner=True)
    rig.publish_changeset("integrated_changeset", {**IMPL, TEST_FILE: TESTS_SRC})
    rig.publish("requirements", requirements_payload())
    rig.publish("task_plan", task_plan_payload())
    agent = VerifierAgent()
    result = agent.run(rig.context(agent, "verify"))
    rep = report(result)
    assert_valid(result)
    assert rep["passed"] is True and rep["total"] == 2
    assert [t["nodeid"] for t in rep["tests"]] == list(IDS)
    assert rep["tests"][0]["covers"] == ["AC-001-1", "R-001"]
    assert rep["fail_to_pass"]["ok"] is True and rep["fail_to_pass"]["tests"] == list(IDS)
    assert rep["baseline"]["ran"] is True and rep["baseline"]["passed"] is False


@pytest.mark.slow
def test_real_sandbox_vacuous_and_failing_tests(tmp_path: Path) -> None:
    src = (
        "def test_always_passes():\n    '''Covers: R-001, AC-001-1'''\n    assert True\n\n\n"
        "def test_fails():\n    '''Covers: R-001, AC-001-2'''\n    assert 1 == 2\n"
    )
    rig = Rig(tmp_path, with_runner=True)
    rig.publish_changeset("integrated_changeset", {"tests/test_mix.py": src})
    rig.publish("requirements", requirements_payload())
    rig.publish("task_plan", task_plan_payload())
    agent = VerifierAgent()
    rep = report(agent.run(rig.context(agent, "verify")))
    outcomes = {t["nodeid"].split("::")[1]: t["outcome"] for t in rep["tests"]}
    assert outcomes == {"test_always_passes": "passed", "test_fails": "failed"}
    assert rep["passed"] is False and rep["failed"] == 1
    assert rep["fail_to_pass"]["ok"] is False
    assert rep["fail_to_pass"]["vacuous"] == ["tests/test_mix.py::test_always_passes"]
    assert rep["baseline"]["failed_tests"] == ["tests/test_mix.py::test_fails"]


@pytest.mark.slow
def test_real_sandbox_openapi_export_and_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_capabilities(
        monkeypatch, fake_capability("widgets", FakeRecipe(api_changes=APP_CHANGES))
    )
    app_src = (
        "class App:\n    def __init__(self, paths):\n        self._paths = paths\n\n"
        "    def openapi(self):\n"
        "        ok = {'get': {'responses': {'200': {'description': 'ok'}}}}\n"
        "        return {'openapi': '3.1.0', 'paths': {p: ok for p in self._paths}}\n\n\n"
        "def create_app():\n    return App(PATHS)\n"
    )
    baseline = {
        "shortener/__init__.py": "",
        "shortener/main.py": "PATHS = ['/a']\n" + app_src,
    }
    integrated = {"shortener/main.py": "PATHS = ['/a', '/b']\n" + app_src}
    rig = Rig(tmp_path, baseline_files=baseline, with_runner=True)
    rig.publish_changeset("integrated_changeset", integrated)
    rig.publish("requirements", requirements_payload())
    rig.publish("task_plan", task_plan_payload())
    agent = VerifierAgent()
    result = agent.run(rig.context(agent, "verify"))
    api = report(result)["api_compat"]
    assert api["checked"] is True, result.artifacts[0].metadata["api_compat_reason"]
    assert api["breaking"] == [] and any("/b" in line for line in api["additive"])
