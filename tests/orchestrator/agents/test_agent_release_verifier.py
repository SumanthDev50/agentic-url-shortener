"""Tests for the release verifier agent and its strict unified-diff parser.

The patch under test is generated with ``tools.changeset.to_unified_diff`` (the promoter's own
generator) from change sets built here, so the parser is checked against the real dialect.
Fast tests
use a ``ScriptedToolbox``; the ``slow`` test runs the real sandbox.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.release_verifier import (
    MAX_PATCH_FILES,
    SMOKE_SOURCE,
    SMOKE_TEST,
    PatchError,
    ReleaseVerifierAgent,
    apply_file_patch,
    parse_patch,
)
from orchestrator.core.errors import (
    AgentCancelled,
    AgentError,
    AgentTimeout,
    ToolDenied,
    TransientAgentError,
)
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import AgentContext, AgentResult, CancelToken, CommandResult
from orchestrator.tools.changeset import apply_changeset, compute_ops, to_unified_diff
from orchestrator.tools.workspace import MemoryView
from tests.orchestrator.agents.delivery_support import (
    Rig,
    assert_valid,
    command_result,
    content_hashes,
    payload,
    plain_writes,
)

BASE = {
    "calc/__init__.py": "",
    "calc/ops.py": "def add(a, b):\n    return a + b\n",
    "README.md": "# Calc\n\nintro\n",
    "old/gone.txt": "bye\n",
}
INTEGRATED = {
    "calc/ops.py": "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
    "tests/test_ops.py": (
        "from calc.ops import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    ),
    "old/gone.txt": None,
}
DOCS = {"README.md": "# Calc\n\nintro\n\nSee docs/ops.md.\n", "docs/ops.md": "# Ops\n"}
GREEN = command_result("3 passed in 0.05s\n", 0)
FAILING = command_result("1 failed, 2 passed in 0.05s\n", 1)
Handler = Callable[[Sequence[str], str, Sequence[str]], CommandResult]


# ------------------------------------------------------------------ helpers


def tree_after(base: Mapping[str, str], desired: Mapping[str, str | None]) -> dict[str, str]:
    tree = dict(base)
    for path, content in desired.items():
        if content is None:
            tree.pop(path, None)
        else:
            tree[path] = content
    return tree


def prepare(
    tmp_path: Path,
    *,
    baseline: Mapping[str, str] = BASE,
    integrated: Mapping[str, str | None] = INTEGRATED,
    docs: Mapping[str, str | None] = DOCS,
    clock_start: datetime | None = None,
    with_runner: bool = False,
) -> tuple[Rig, str, dict[str, Any]]:
    """A rig with the two reviewed change sets published; returns (rig, patch, bundle)."""
    rig = Rig(tmp_path, baseline_files=baseline, clock_start=clock_start, with_runner=with_runner)
    first = rig.publish_changeset("integrated_changeset", dict(integrated))
    after_first = tree_after(baseline, apply_changeset(MemoryView(baseline), first))
    rig.publish_changeset("docs_changeset", dict(docs), base=after_first)
    final = tree_after(after_first, docs)
    desired = {p: final.get(p) for p in {*integrated, *docs}}
    merged = compute_ops(MemoryView(baseline), desired)
    patch = to_unified_diff(MemoryView(baseline), merged)
    bundle = {
        "schema": 1,
        "mode": "bundle",
        "patch_sha256": sha256_text(patch),
        "files": sorted(merged.paths()),
        "base_manifest_hash": hash_jsonable({}),
        "receipt": None,
    }
    return rig, patch, bundle


def publish_release(rig: Rig, patch: str, bundle: Mapping[str, Any]) -> None:
    rig.publish("release_patch", patch)
    rig.publish("release_bundle", dict(bundle))


def verify(
    rig: Rig, tests: Handler | None = None, **kwargs: Any
) -> tuple[AgentContext, AgentResult]:
    agent = ReleaseVerifierAgent()
    ctx = rig.context(agent, "release-verify", tests=tests or (lambda t, m, o: GREEN), **kwargs)
    return ctx, agent.run(ctx)


def verification(result: AgentResult) -> dict[str, Any]:
    return payload(result, "release_verification")


def problems(result: AgentResult) -> list[str]:
    return list(result.artifacts[0].metadata["problems"])


@pytest.fixture(autouse=True)
def fast_files(monkeypatch: pytest.MonkeyPatch) -> None:
    plain_writes(monkeypatch)


@pytest.fixture
def released(tmp_path: Path) -> tuple[Rig, str, dict[str, Any]]:
    rig, patch, bundle = prepare(tmp_path)
    publish_release(rig, patch, bundle)
    return rig, patch, bundle


# ------------------------------------------------------------------ patch parser and applier


def patch_of(
    baseline: Mapping[str, str], desired: Mapping[str, str | None]
) -> tuple[str, dict[str, str | None]]:
    view = MemoryView(baseline)
    return to_unified_diff(view, compute_ops(view, dict(desired))), dict(desired)


def apply_all(baseline: Mapping[str, str], patch: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for item in parse_patch(patch):
        out[item.path] = apply_file_patch(item, baseline.get(item.path))
    return out


def test_the_real_dialect_round_trips_create_modify_delete() -> None:
    patch, _ = patch_of(BASE, {**INTEGRATED, **DOCS})
    assert apply_all(BASE, patch) == {**INTEGRATED, **DOCS}
    kinds = {p.path: p.kind for p in parse_patch(patch)}
    assert kinds["old/gone.txt"] == "delete" and kinds["tests/test_ops.py"] == "create"
    assert kinds["calc/ops.py"] == "modify"


def test_multiple_hunks_and_missing_trailing_newlines() -> None:
    long_old = "".join(f"line {i}\n" for i in range(40))
    long_new = long_old.replace("line 2\n", "line two\n").replace("line 37\n", "line 37!\n")
    base = {"f.txt": long_old, "n.txt": "a\nb", "e.txt": "x\n"}
    patch, desired = patch_of(
        base, {"f.txt": long_new, "n.txt": "a\nc", "e.txt": "", "z.txt": "last no newline"}
    )
    assert patch.count("@@ -") >= 3
    assert apply_all(base, patch) == desired


def test_empty_files_and_empty_deletes() -> None:
    base = {"empty.txt": "", "full.txt": "x\n"}
    patch, _ = patch_of(base, {"empty.txt": None, "new_empty.txt": ""})
    assert apply_all(base, patch) == {"empty.txt": None, "new_empty.txt": ""}
    with pytest.raises(PatchError, match="content behind"):  # a delete of a non-empty file
        apply_file_patch(parse_patch(patch)[0], "not empty\n")


def test_unusual_paths_and_line_separators_survive() -> None:
    base = {"dir with space/ünï.txt": "a\x0cb\nc\n"}
    patch, desired = patch_of(
        base, {"dir with space/ünï.txt": "a\x0cb\nd\n", "new dir/x y.txt": "p\u2028q\n"}
    )
    assert apply_all(base, patch) == desired


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        ("garbage\n", "diff --git"),
        ("diff --git a/x b/y\n", "diff --git"),
        ("diff --git a/../x b/../x\n", "unsafe path"),
        ("diff --git a/x b/x\r\n", "LF"),
        ("diff --git a/x b/x\n--- a/x\n", "missing"),
        ("diff --git a/x b/x\n--- a/other\n+++ b/x\n", "disagree"),
        ("diff --git a/x b/x\nnew file mode 100644\n--- a/x\n+++ b/x\n", "disagree"),
        ("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n", "shorter"),
        ("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ nonsense @@\n", "hunk header"),
        ("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\nx\n", "shorter"),
        ("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,2 +1 @@\n-a\n", "shorter"),
    ],
)
def test_the_parser_rejects_anything_outside_the_dialect(text: str, needle: str) -> None:
    with pytest.raises(PatchError, match=needle):
        parse_patch(text)


def test_the_parser_bounds_files_and_line_length() -> None:
    many = "".join(
        f"diff --git a/f{i} b/f{i}\nnew file mode 100644\n" for i in range(MAX_PATCH_FILES + 1)
    )
    with pytest.raises(PatchError, match="too many"):
        parse_patch(many)
    huge = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+" + "y" * 250_000 + "\n"
    with pytest.raises(PatchError, match="too long"):
        parse_patch(huge)


def test_the_applier_refuses_a_patch_that_does_not_fit_the_baseline() -> None:
    patch, _ = patch_of(BASE, {"calc/ops.py": "def add(a, b):\n    return a * b\n"})
    item = parse_patch(patch)[0]
    with pytest.raises(PatchError, match="do not match"):
        apply_file_patch(item, "def add(a, b):\n    return a - b\n")  # drifted baseline
    with pytest.raises(PatchError, match="missing"):
        apply_file_patch(item, None)
    create = parse_patch(patch_of({}, {"n.txt": "x\n"})[0])[0]
    with pytest.raises(PatchError, match="already exists"):
        apply_file_patch(create, "present\n")
    shifted = parse_patch("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -9,1 +9,1 @@\n-a\n+b\n")[0]
    with pytest.raises(PatchError, match="out of range"):
        apply_file_patch(shifted, "a\n")


# ------------------------------------------------------------------ agent: the patch checks


def test_a_clean_bundle_verifies(released: tuple[Rig, str, dict[str, Any]]) -> None:
    rig, _, bundle = released
    ctx, result = verify(rig)
    assert_valid(result)
    assert verification(result) == {
        "schema": 1,
        "target": "bundle-scratch",
        "applied_cleanly": True,
        "tests_passed": True,
        "healthz_ok": None,
        "checked_files": len(bundle["files"]),
    }
    assert problems(result) == []
    assert ctx.toolbox.calls == [("run_tests", ("tests",), "view", ())]  # type: ignore[attr-defined]


def test_a_patch_that_does_not_hash_to_the_bundle_fails(tmp_path: Path) -> None:
    rig, patch, bundle = prepare(tmp_path)
    publish_release(rig, patch.replace("sub", "mul", 1), bundle)  # content changed, hash not
    _, result = verify(rig)
    assert verification(result)["applied_cleanly"] is False
    assert any("patch_sha256" in p for p in problems(result))


def test_a_patch_that_produces_other_content_than_reviewed_fails(tmp_path: Path) -> None:
    rig, patch, bundle = prepare(tmp_path)
    forged = patch.replace("+    return a - b", "+    return a - b + 1")
    assert forged != patch
    publish_release(rig, forged, {**bundle, "patch_sha256": sha256_text(forged)})
    _, result = verify(rig)
    assert verification(result)["applied_cleanly"] is False
    assert any("differs from the reviewed tree" in p for p in problems(result))


def test_a_patch_touching_an_unreviewed_path_fails(tmp_path: Path) -> None:
    rig, patch, bundle = prepare(tmp_path)
    sneaky = patch + "diff --git a/evil.py b/evil.py\nnew file mode 100644\n"
    publish_release(rig, sneaky, {**bundle, "patch_sha256": sha256_text(sneaky)})
    _, result = verify(rig)
    assert verification(result)["applied_cleanly"] is False
    assert any("outside the reviewed" in p for p in problems(result))
    assert any("different files" in p for p in problems(result))


def test_bundle_files_must_match_the_patch(tmp_path: Path) -> None:
    rig, patch, bundle = prepare(tmp_path)
    publish_release(rig, patch, {**bundle, "files": bundle["files"][:-1]})
    _, result = verify(rig)
    assert verification(result)["applied_cleanly"] is False
    assert any("different files" in p for p in problems(result))


def test_a_patch_that_belongs_to_other_change_sets_fails(tmp_path: Path) -> None:
    _, patch, bundle = prepare(tmp_path)
    drifted = Rig(
        tmp_path / "other",
        baseline_files={**BASE, "calc/ops.py": "def add(a, b):\n    return a - b\n"},
    )
    drifted.publish_changeset("integrated_changeset", {"docs.txt": "x\n"})
    drifted.publish_changeset("docs_changeset", {})
    publish_release(drifted, patch, bundle)
    _, result = verify(drifted)
    assert verification(result)["applied_cleanly"] is False
    assert problems(result)


def test_an_unparseable_patch_fails_cleanly(tmp_path: Path) -> None:
    rig, _, bundle = prepare(tmp_path)
    junk = "this is not a patch\n"
    publish_release(rig, junk, {**bundle, "patch_sha256": sha256_text(junk)})
    _, result = verify(rig)
    assert verification(result)["applied_cleanly"] is False
    assert any("cannot be parsed" in p for p in problems(result))
    assert verification(result)["checked_files"] == 0


def test_a_toolbox_without_a_handed_in_baseline_fails_closed(
    released: tuple[Rig, str, dict[str, Any]],
) -> None:
    rig, _, _ = released
    _, result = verify(rig, baseline_handed_in=False)
    # the toolbox falls back to its own (post-change) view, which is NOT the baseline: the patch
    # (create/modify/delete against the baseline) cannot apply to it
    assert verification(result)["applied_cleanly"] is False
    assert problems(result)


def test_a_toolbox_with_no_baseline_view_at_all_fails_closed(
    released: tuple[Rig, str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig, _, _ = released
    monkeypatch.setattr("orchestrator.agents.release_verifier.baseline_view_of", lambda box: None)
    _, result = verify(rig)
    assert verification(result)["applied_cleanly"] is False
    assert any("no baseline view" in p for p in problems(result))


# ------------------------------------------------------------------ agent: applied mode


def applied_bundle(rig: Rig, bundle: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    view = rig.view_for(rig.node("release-verify"))
    post = {p: sha256_text(view.read_text(p)) if view.exists(p) else None for p in bundle["files"]}
    receipt = {"target_id": "dir:abc", "applied": bundle["files"], "post_hashes": post}
    receipt.update(overrides)
    return {**bundle, "mode": "applied", "receipt": receipt}


def test_an_applied_bundle_is_compared_with_its_receipt(tmp_path: Path) -> None:
    rig, patch, bundle = prepare(tmp_path)
    publish_release(rig, patch, applied_bundle(rig, bundle))
    _, result = verify(rig)
    assert verification(result)["target"] == "applied-target"
    assert verification(result)["applied_cleanly"] is True and problems(result) == []


def test_a_receipt_with_a_wrong_hash_fails(tmp_path: Path) -> None:
    rig, patch, bundle = prepare(tmp_path)
    good = applied_bundle(rig, bundle)
    bad_hashes = dict(good["receipt"]["post_hashes"])
    bad_hashes["calc/ops.py"] = sha256_text("something else\n")
    publish_release(rig, patch, applied_bundle(rig, bundle, post_hashes=bad_hashes))
    _, result = verify(rig)
    assert verification(result)["applied_cleanly"] is False
    assert any("applied file differs" in p for p in problems(result))


def test_a_receipt_that_misses_files_or_is_absent_fails(tmp_path: Path) -> None:
    rig, patch, bundle = prepare(tmp_path)
    short = dict(applied_bundle(rig, bundle)["receipt"]["post_hashes"])
    short.pop("README.md")
    publish_release(rig, patch, applied_bundle(rig, bundle, post_hashes=short))
    assert verification(verify(rig)[1])["applied_cleanly"] is False

    rig2, patch2, bundle2 = prepare(tmp_path / "second")
    publish_release(rig2, patch2, {**bundle2, "mode": "applied", "receipt": None})
    _, result = verify(rig2)
    assert verification(result)["applied_cleanly"] is False
    assert any("no receipt" in p for p in problems(result))


# ------------------------------------------------------------------ agent: tests and smoke


def test_failing_tests_are_a_recorded_fact_not_an_error(
    released: tuple[Rig, str, dict[str, Any]],
) -> None:
    rig, _, _ = released
    _, result = verify(rig, lambda t, m, o: FAILING)
    out = verification(result)
    assert out["tests_passed"] is False and out["applied_cleanly"] is True


@pytest.mark.parametrize(
    "stdout", ["", "1 skipped in 0.01s\n", "no tests ran in 0.01s\n", "1 passed, 1 error in 0.1s\n"]
)
def test_green_needs_a_pass_and_no_failure(
    released: tuple[Rig, str, dict[str, Any]], stdout: str
) -> None:
    rig, _, _ = released
    _, result = verify(rig, lambda t, m, o: command_result(stdout, 0))  # even rc 0 is not enough
    assert verification(result)["tests_passed"] is False


def test_a_nonzero_exit_is_never_green(released: tuple[Rig, str, dict[str, Any]]) -> None:
    rig, _, _ = released

    def exit_with(code: int) -> Handler:
        return lambda targets, mode, overlay: command_result("3 passed in 0.1s\n", code)

    for code in (1, 2, 3, 4, 5):
        _, result = verify(rig, exit_with(code))
        assert verification(result)["tests_passed"] is False


@pytest.mark.parametrize(
    ("result", "exc"),
    [
        (command_result("", 0, timed_out=True), AgentTimeout),
        (command_result("", -9), TransientAgentError),
        (command_result("", 3221225477), TransientAgentError),
    ],
    ids=["timeout", "killed", "crash"],
)
def test_timeouts_and_crashes_are_retryable(
    released: tuple[Rig, str, dict[str, Any]], result: CommandResult, exc: type[AgentError]
) -> None:
    rig, _, _ = released
    with pytest.raises(exc) as excinfo:
        verify(rig, lambda t, m, o: result)
    assert excinfo.value.retryable is True


def test_a_sandbox_io_failure_is_transient(released: tuple[Rig, str, dict[str, Any]]) -> None:
    rig, _, _ = released

    def boom(targets: Sequence[str], mode: str, overlay: Sequence[str]) -> CommandResult:
        raise PermissionError("locked")

    with pytest.raises(TransientAgentError):
        verify(rig, boom)


def test_a_tree_without_tests_is_not_green_and_runs_nothing(tmp_path: Path) -> None:
    rig, patch, bundle = prepare(
        tmp_path,
        integrated={"calc/ops.py": INTEGRATED["calc/ops.py"]},
        docs={"docs/ops.md": "# Ops\n"},
    )
    publish_release(rig, patch, bundle)
    ctx, result = verify(rig)
    assert verification(result)["tests_passed"] is False
    assert ctx.toolbox.calls == []  # type: ignore[attr-defined]
    assert any("no tests" in p for p in problems(result))


PRODUCT = {
    "shortener/__init__.py": "",
    "shortener/main.py": "def create_app(settings=None):\n    pass\n",
    "shortener/config.py": "class Settings:\n    pass\n",
}


def product_release(tmp_path: Path) -> Rig:
    rig, patch, bundle = prepare(tmp_path, integrated={**INTEGRATED, **PRODUCT})
    publish_release(rig, patch, bundle)
    return rig


def test_the_healthz_smoke_runs_only_when_the_product_is_present(tmp_path: Path) -> None:
    rig = product_release(tmp_path)
    ctx, result = verify(rig)
    assert verification(result)["healthz_ok"] is True
    assert ctx.toolbox.calls == [  # type: ignore[attr-defined]
        ("run_tests", ("tests",), "view", ()),
        ("run_tests", (SMOKE_TEST,), "view", ()),
    ]
    assert ctx.toolbox.read_text(SMOKE_TEST) == SMOKE_SOURCE  # static text, nothing interpolated
    assert "healthz" in SMOKE_SOURCE and "auth_disabled=True" in SMOKE_SOURCE


def test_a_failing_smoke_is_recorded(tmp_path: Path) -> None:
    rig = product_release(tmp_path)

    def answer(targets: Sequence[str], mode: str, overlay: Sequence[str]) -> CommandResult:
        return FAILING if SMOKE_TEST in targets else GREEN

    _, result = verify(rig, answer)
    out = verification(result)
    assert out["healthz_ok"] is False and out["tests_passed"] is True


def test_no_product_means_no_smoke_and_no_staged_file(
    released: tuple[Rig, str, dict[str, Any]],
) -> None:
    rig, _, _ = released
    ctx, result = verify(rig)
    assert verification(result)["healthz_ok"] is None
    assert not ctx.toolbox.exists(SMOKE_TEST)


def test_the_smoke_source_is_valid_python() -> None:
    compile(SMOKE_SOURCE, SMOKE_TEST, "exec")  # the static text must at least parse


# ------------------------------------------------------------------ cross-cutting


def test_the_verification_is_deterministic_across_clocks(tmp_path: Path) -> None:
    one, patch_a, bundle_a = prepare(tmp_path / "a", clock_start=datetime(2026, 1, 1, tzinfo=UTC))
    two, patch_b, bundle_b = prepare(
        tmp_path / "b", clock_start=datetime(2031, 5, 6, 7, tzinfo=UTC)
    )
    publish_release(one, patch_a, bundle_a)
    publish_release(two, patch_b, bundle_b)
    assert content_hashes(verify(one)[1]) == content_hashes(verify(two)[1])


def test_a_cancelled_attempt_raises(released: tuple[Rig, str, dict[str, Any]]) -> None:
    rig, _, _ = released
    cancel = CancelToken()
    cancel.cancel("stop")
    with pytest.raises(AgentCancelled):
        verify(rig, cancel=cancel)


def test_the_agent_may_only_write_its_smoke_directory(
    released: tuple[Rig, str, dict[str, Any]],
) -> None:
    rig, _, _ = released
    ctx = rig.context(ReleaseVerifierAgent(), "release-verify", tests=lambda t, m, o: GREEN)
    ctx.toolbox.write_text(SMOKE_TEST, "x = 1\n")
    with pytest.raises(ToolDenied):
        ctx.toolbox.write_text("calc/ops.py", "x = 1\n")
    assert ReleaseVerifierAgent.autonomy.write_globs == ("_agentic_release/**",)
    assert ReleaseVerifierAgent.autonomy.can_execute is True


def test_problems_are_bounded_and_sanitised(tmp_path: Path) -> None:
    many = {f"d/f{i}.txt": "x\n" for i in range(40)}
    rig, patch, bundle = prepare(tmp_path, baseline={}, integrated=many, docs={})
    other = patch.replace("+x\n", "+y\n")
    publish_release(rig, other, {**bundle, "patch_sha256": sha256_text(other)})
    _, result = verify(rig)
    listed = problems(result)
    assert 0 < len(listed) <= 20 and all("\x1b" not in p and len(p) <= 240 for p in listed)


# ------------------------------------------------------------------ real sandbox (slow)


@pytest.mark.slow
def test_real_sandbox_release_with_smoke(tmp_path: Path) -> None:
    product = {
        "shortener/__init__.py": "",
        "shortener/config.py": (
            "from dataclasses import dataclass\n\n\n@dataclass\nclass Settings:\n"
            "    auth_disabled: bool = False\n    database_path: str = 'x.db'\n"
        ),
        "shortener/main.py": (
            "from fastapi import FastAPI\n\n\n"
            "def create_app(settings=None):\n"
            "    app = FastAPI()\n\n"
            "    @app.get('/healthz')\n"
            "    def healthz():\n"
            "        return {'status': 'ok'}\n\n"
            "    return app\n"
        ),
    }
    rig, patch, bundle = prepare(tmp_path, integrated={**INTEGRATED, **product}, with_runner=True)
    publish_release(rig, patch, bundle)
    agent = ReleaseVerifierAgent()
    result = agent.run(rig.context(agent, "release-verify"))
    out = verification(result)
    assert out["applied_cleanly"] is True, problems(result)
    assert out["tests_passed"] is True, problems(result)
    assert out["healthz_ok"] is True
    assert out["target"] == "bundle-scratch" and out["checked_files"] == len(bundle["files"])
