"""Tests for the promoter agent (``promote`` node) and the ``apply_bundle`` compensator.

The agent runs in the shared ``delivery_support.Rig``: an engine-like context with the real artifact
store (only the declared inputs are served), a real ``ScopedToolbox`` over baseline + integrated +
docs with the baseline handed to the factory, and the real policy engine.  Patches are verified with
an independent mini applier defined below (a plain line walker that shares no code with production).
Apply-mode tests use a real ``DirectoryEffectPort`` over a ``tmp_path`` tree that holds the baseline
files (at most 4 files: the port's own writer is slow on Windows); contract tests use a stub port.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.promoter import (
    MAX_BASELINE_READS,
    MODES,
    ApplyBundleCompensator,
    PromoterAgent,
)
from orchestrator.core.errors import (
    AgentCancelled,
    AgentError,
    AutonomyViolation,
    ToolDenied,
    UndeclaredInputError,
)
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.schemas import Kind
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import (
    MAX_CHANGESET_OPS,
    AgentContext,
    AgentResult,
    ApplyReceipt,
    AutonomyProfile,
    CancelToken,
    ChangeSet,
    Compensator,
    DecisionKind,
    EffectPort,
    EventType,
    FileOpKind,
    FileView,
    NodeSpec,
)
from orchestrator.tools.changeset import ChangeSetConflict, compute_ops, to_unified_diff
from orchestrator.tools.effects import (
    DirectoryEffectPort,
    EffectError,
    ProtectedPathError,
    RevertRefused,
    RollbackIncompleteError,
    UnsafeTargetError,
    allow_any_worktree,
)
from orchestrator.tools.toolbox import BudgetExhausted
from orchestrator.tools.workspace import BaselineSnapshot, MemoryView, OverlayView
from tests.orchestrator.agents.delivery_support import (
    Rig,
    assert_valid,
    content_hashes,
    payload,
    plain_writes,
    requirements_payload,
    text_of,
)
from tests.orchestrator.policies.policy_builders import FakeRunView

PROTECTED = (".github/**", "Dockerfile", ".env*", "pyproject.toml")
NO_EOL = "\\ No newline at end of file"
HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
GIT = shutil.which("git")

BASELINE: dict[str, str] = {
    "app/main.py": "def main():\n    return 'v0'\n",
    "app/util.py": "def helper():\n    return 1\n",
    "app/legacy.py": "OLD = True\n",
    "README.md": "# Demo\n\nSome text.\n",
    "docs/guide.md": "guide v0\n",
    "tests/test_main.py": "def test_main():\n    assert True\n",
}
# what the integrator produced (relative to the baseline) ...
INTEGRATED_FILES: dict[str, str | None] = {
    "app/main.py": "def main():\n    return 'v1'\n",
    "app/util.py": "def helper():\n    return 2\n",
    "app/legacy.py": None,
    "app/feature.py": "FEATURE = 1\n",
}
# ... and what the documenter added on top of baseline + integrated
DOCS_FILES: dict[str, str | None] = {
    "app/main.py": "def main():\n    return 'v2'\n",
    "app/feature.py": "FEATURE = 1\n\n# documented\n",
    "README.md": "# Demo\n\nSome text.\n\nNew section.\n",
    "docs/guide.md": "guide v0\nguide docs\n",
}

# a smaller scenario for the tests that really write to a directory
SMALL: dict[str, str] = {
    "app/main.py": "def main():\n    return 'v0'\n",
    "app/legacy.py": "OLD = True\n",
    "README.md": "# Demo\n",
}
SMALL_INTEGRATED: dict[str, str | None] = {
    "app/main.py": "def main():\n    return 'v1'\n",
    "app/legacy.py": None,
    "app/feature.py": "FEATURE = 1\n",
}
SMALL_DOCS: dict[str, str | None] = {
    "README.md": "# Demo\n\nNew section.\n",
    "app/feature.py": "FEATURE = 1\n# documented\n",
}
SMALL_TOUCHED = ["README.md", "app/feature.py", "app/legacy.py", "app/main.py"]


# ------------------------------------------------------------------ helpers


def settled(tree: Mapping[str, str], desired: Mapping[str, str | None]) -> dict[str, str]:
    """``tree`` after ``desired`` is applied (``None`` deletes the path)."""
    out = dict(tree)
    for path, text in desired.items():
        if text is None:
            out.pop(path, None)
        else:
            out[path] = text
    return out


FINAL = settled(settled(BASELINE, INTEGRATED_FILES), DOCS_FILES)
SMALL_FINAL = settled(settled(SMALL, SMALL_INTEGRATED), SMALL_DOCS)
CHANGED = sorted(p for p in {*BASELINE, *FINAL} if BASELINE.get(p) != FINAL.get(p))


def manifest_hash(files: Mapping[str, str]) -> str:
    """Same definition as ``BaselineSnapshot.manifest_hash``: every file, ``sha256_text`` each."""
    return hash_jsonable({path: sha256_text(text) for path, text in files.items()})


def expected_patch(baseline: Mapping[str, str], final: Mapping[str, str]) -> str:
    """The release patch computed independently: baseline -> final tree."""
    desired: dict[str, str | None] = {path: final.get(path) for path in {*baseline, *final}}
    return to_unified_diff(MemoryView(baseline), compute_ops(MemoryView(baseline), desired))


def readiness(verdict: str = "GO") -> dict[str, Any]:
    return {
        "schema": 1,
        "verdict": verdict,
        "checks": [],
        "risk": {"level": "LOW", "reasons": []},
        "rollback_plan": "x",
        "change_reference": "CHG-1",
        "open_items": [],
    }


def build(
    root: Path,
    integrated: Mapping[str, str | None] | None = None,
    docs: Mapping[str, str | None] | None = None,
    *,
    baseline: Mapping[str, str] = BASELINE,
    report: Mapping[str, Any] | None = None,
    clock_start: datetime | None = None,
) -> Rig:
    """A rig with the three declared inputs published: integrated is made against the baseline,
    docs against baseline + integrated (the engine's view semantics); ``report`` may be forged."""
    rig = Rig(root, baseline_files=dict(baseline), clock_start=clock_start)
    rig.publish_changeset(Kind.INTEGRATED_CHANGESET, integrated or {})
    rig.publish_changeset(Kind.DOCS_CHANGESET, docs or {}, base=settled(baseline, integrated or {}))
    rig.publish(
        Kind.READINESS_REPORT, readiness() if report is None else report, validate=report is None
    )
    return rig


def promote(rig: Rig, **kwargs: Any) -> AgentResult:
    return rig.run(PromoterAgent(), "promote", **kwargs)


def patch_of(result: AgentResult) -> str:
    return text_of(result, Kind.RELEASE_PATCH)


def bundle_of(result: AgentResult) -> dict[str, Any]:
    return payload(result, Kind.RELEASE_BUNDLE)


def assert_fatal(caught: pytest.ExceptionInfo[AgentError], *words: str) -> None:
    """The attempt fails FATAL (never retried, skip to the fallback) and says why."""
    error = caught.value
    assert (error.kind, error.retryable) == ("fatal", False), str(error)
    for word in words:
        assert word.lower() in str(error).lower(), (word, str(error))


class StubPort:
    """An ``EffectPort`` that records what it is asked to do and fails on demand."""

    def __init__(
        self, *, error: BaseException | None = None, receipt: ApplyReceipt | None = None
    ) -> None:
        self.error = error
        self.receipt = receipt or ApplyReceipt("dir:stub", (), {}, {}, ChangeSet())
        self.applied: list[ChangeSet] = []
        self.reverted: list[ApplyReceipt] = []

    def apply(self, cs: ChangeSet, *, dry_run: bool = False) -> ApplyReceipt:
        self.applied.append(cs)
        if self.error is not None:
            raise self.error
        return self.receipt

    def revert(self, receipt: ApplyReceipt) -> None:
        self.reverted.append(receipt)


def compensate(receipt: Any, effects: EffectPort) -> str:
    return ApplyBundleCompensator().compensate(receipt, effects=effects, run_view=FakeRunView())


def lenient_views(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the rig build the node view WITHOUT verifying base hashes.  The engine's strict view
    would already refuse change sets that drifted from the baseline; the promoter must not rely on
    that (defence in depth), so these tests give it the lenient view and a drifted input."""

    def view_for(spec: NodeSpec) -> FileView:
        sets = [
            ChangeSet.from_dict(rig.store.read_json(artifact))
            for kind in spec.overlay_from
            for artifact in rig.store.active(kind)
        ]
        return OverlayView(rig.baseline, sets, strict=False)

    monkeypatch.setattr(rig, "view_for", view_for)


# ---- the independent mini applier


def apply_patch(base: Mapping[str, str], patch: str) -> dict[str, str]:
    """Apply a git-style patch to ``base`` with a plain line walker (no production code).

    Every hunk must match the old text exactly and its header counts must add up: a patch that
    is malformed, or was made against another tree, raises ``AssertionError``.
    """
    assert patch.endswith("\n") and "\r" not in patch
    tree = dict(base)
    lines = patch[:-1].split("\n")
    pos = 0
    while pos < len(lines):
        head = re.fullmatch(r"diff --git a/(.+) b/\1", lines[pos])
        assert head is not None, lines[pos]
        path = head.group(1)
        end = pos + 1
        while end < len(lines) and not lines[end].startswith("diff --git "):
            end += 1
        new = apply_file(tree.get(path), path, lines[pos + 1 : end])
        if new is None:
            del tree[path]
        else:
            tree[path] = new
        pos = end
    return tree


def split_lines(text: str) -> list[str]:
    """Lines the way git sees them: split at LF only (``str.splitlines`` also splits at form
    feed, U+2028 and friends)."""
    return re.findall(r"[^\n]*\n|[^\n]+", text)


def apply_file(old: str | None, path: str, section: list[str]) -> str | None:
    """One file section of ``apply_patch``; returns the new text (``None``: file deleted)."""
    kind = "modify"
    if section and re.fullmatch(r"(new|deleted) file mode 100644", section[0]):
        kind = section.pop(0).split()[0]
    assert (old is None) == (kind == "new"), f"{path}: {kind} but exists={old is not None}"
    if not section:  # header only: an empty file is created / deleted
        assert kind != "modify" and not old, path
        return "" if kind == "new" else None
    old_name = "/dev/null" if kind == "new" else f"a/{path}"
    new_name = "/dev/null" if kind == "deleted" else f"b/{path}"
    assert section[:2] == [f"--- {old_name}", f"+++ {new_name}"], section[:2]
    source = split_lines(old or "")
    out: list[str] = []
    cursor = 0
    k = 2
    while k < len(section):
        hunk = HUNK.fullmatch(section[k])
        assert hunk is not None, section[k]
        old_start, old_len, new_len = int(hunk[1]), int(hunk[2] or 1), int(hunk[4] or 1)
        first = old_start - 1 if old_len else old_start
        out += source[cursor:first]
        cursor = first
        body: list[tuple[str, str]] = []
        k += 1
        while k < len(section) and not section[k].startswith("@@ "):
            line = section[k]
            if line == NO_EOL:
                tag, text = body[-1]
                assert text.endswith("\n")
                body[-1] = (tag, text[:-1])
            else:
                assert line[:1] in (" ", "-", "+"), repr(line)
                body.append((line[0], line[1:] + "\n"))
            k += 1
        removed = [text for tag, text in body if tag != "+"]
        added = [text for tag, text in body if tag != "-"]
        assert (len(removed), len(added)) == (old_len, new_len), (path, old_len, new_len)
        assert source[cursor : cursor + old_len] == removed, f"{path}: hunk does not match"
        cursor += old_len
        out += added
    out += source[cursor:]
    if kind == "deleted":
        assert not out, f"{path}: deleted file keeps content"
        return None
    return "".join(out)


# ---- real target trees


def write_tree(root: Path, files: Mapping[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def as_bytes(files: Mapping[str, str]) -> dict[str, bytes]:
    return {path: text.encode("utf-8") for path, text in files.items()}


def target_with(tmp_path: Path, files: Mapping[str, str]) -> Path:
    root = tmp_path / "target"
    root.mkdir()
    write_tree(root, files)
    return root


def make_port(root: Path, rig: Rig, **kwargs: Any) -> DirectoryEffectPort:
    kwargs.setdefault("git_guard", allow_any_worktree)
    return DirectoryEffectPort(
        root, protected_globs=PROTECTED, audit=rig.audit, node_id="promote", attempt=1, **kwargs
    )


@dataclass
class Applied:
    """An apply-mode run against a real target tree that holds the ``SMALL`` baseline."""

    rig: Rig
    target: Path
    port: DirectoryEffectPort
    result: AgentResult

    @property
    def bundle(self) -> dict[str, Any]:
        return bundle_of(self.result)

    @property
    def receipt(self) -> ApplyReceipt:
        return ApplyReceipt.from_dict(self.bundle["receipt"])


@pytest.fixture(autouse=True)
def fast_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    plain_writes(monkeypatch)  # artifact writes only: the effect port keeps its own real writer


@pytest.fixture
def applied(tmp_path: Path) -> Applied:
    rig = build(tmp_path / "rig", SMALL_INTEGRATED, SMALL_DOCS, baseline=SMALL)
    target = target_with(tmp_path, SMALL)
    port = make_port(target, rig)
    result = promote(rig, params={"mode": "apply"}, effects=port)
    return Applied(rig, target, port, result)


# ------------------------------------------------------------------ bundle mode


def test_the_mini_applier_rejects_a_patch_made_against_another_tree() -> None:
    """Guards the verification below against being vacuous."""
    patch = expected_patch(BASELINE, FINAL)
    assert apply_patch(BASELINE, patch) == FINAL
    edited = {**BASELINE, "app/main.py": "def main():\n    return 'other'\n"}
    without_legacy = {p: t for p, t in BASELINE.items() if p != "app/legacy.py"}
    with_feature = {**BASELINE, "app/feature.py": "FEATURE = 0\n"}
    for wrong in (edited, without_legacy, with_feature):
        with pytest.raises(AssertionError):
            apply_patch(wrong, patch)


def test_patch_is_the_independent_diff_of_baseline_to_final_tree(tmp_path: Path) -> None:
    result = promote(build(tmp_path, INTEGRATED_FILES, DOCS_FILES))
    patch = patch_of(result)
    assert patch == expected_patch(BASELINE, FINAL)
    assert apply_patch(BASELINE, patch) == FINAL  # an independent applier reaches the final tree
    assert "\r" not in patch and patch.endswith("\n")
    draft = result.artifacts[0]
    assert (draft.kind, draft.media_type, draft.filename) == (
        Kind.RELEASE_PATCH,
        "text/x-patch",
        "release.patch",
    )
    assert [d.kind for d in result.artifacts] == [Kind.RELEASE_PATCH, Kind.RELEASE_BUNDLE]
    assert_valid(result)


def test_bundle_describes_the_patch_the_files_and_the_baseline(tmp_path: Path) -> None:
    result = promote(build(tmp_path, INTEGRATED_FILES, DOCS_FILES))
    bundle, patch = bundle_of(result), patch_of(result)
    assert (bundle["schema"], bundle["mode"], bundle["receipt"]) == (1, "bundle", None)
    assert bundle["patch_sha256"] == sha256_text(patch)
    assert bundle["files"] == CHANGED == sorted(bundle["files"])
    assert "tests/test_main.py" not in bundle["files"]  # unchanged files are not released
    headers = re.findall(r"^diff --git a/(.+) b/\1$", patch, re.MULTILINE)
    assert headers == bundle["files"]  # the patch lists the files in the same sorted order
    assert bundle["base_manifest_hash"] == manifest_hash(BASELINE)  # ALL baseline files
    # and that is exactly what the real baseline snapshot of the same tree computes
    source = tmp_path / "source"
    write_tree(source, BASELINE)
    snapshot = BaselineSnapshot.create(source, tmp_path / "snapshot")
    assert bundle["base_manifest_hash"] == snapshot.manifest_hash


def test_bundle_mode_has_no_side_effect_and_needs_no_effect_port(tmp_path: Path) -> None:
    rig = build(tmp_path / "rig", SMALL_INTEGRATED, SMALL_DOCS, baseline=SMALL)
    target = target_with(tmp_path, SMALL)
    before = tree_bytes(target)
    stub = StubPort()
    runs = [
        promote(rig),  # bundle is the default mode; no effect port at all
        promote(rig, params={"mode": "bundle"}, effects=make_port(target, rig)),
        promote(rig, params={"mode": "bundle"}, effects=stub),
    ]
    assert tree_bytes(target) == before
    assert stub.applied == [] and stub.reverted == []
    assert rig.audit.of(EventType.EFFECT_STARTED) == []
    assert len({patch_of(run) for run in runs}) == 1
    assert all(bundle_of(run)["mode"] == "bundle" for run in runs)


def test_greenfield_is_all_creates_against_the_empty_baseline(tmp_path: Path) -> None:
    created: dict[str, str | None] = {"app/main.py": "print('hi')\n", "README.md": "# New\n"}
    docs: dict[str, str | None] = {"README.md": "# New\n\nDocs.\n", "docs/usage.md": "usage\n"}
    result = promote(build(tmp_path, created, docs, baseline={}))
    patch, bundle = patch_of(result), bundle_of(result)
    final = settled(settled({}, created), docs)
    assert patch == expected_patch({}, final)
    assert patch.count("new file mode 100644\n") == len(final) == 3
    assert patch.count("--- /dev/null\n") == 3
    assert "deleted file" not in patch and not re.search(r"^--- a/", patch, re.MULTILINE)
    assert apply_patch({}, patch) == final
    assert bundle["files"] == sorted(final)
    assert bundle["base_manifest_hash"] == hash_jsonable({})


def test_brownfield_hunks_reference_the_baseline_text(tmp_path: Path) -> None:
    patch = patch_of(promote(build(tmp_path, INTEGRATED_FILES, DOCS_FILES)))
    # changed twice (integrated v1, docs v2): one MODIFY from the BASELINE text to the final text
    assert "-    return 'v0'\n+    return 'v2'\n" in patch
    assert "return 'v1'" not in patch
    # deleted by integrated: a DELETE whose hunk is the baseline text
    section = "diff --git a/app/legacy.py b/app/legacy.py\ndeleted file mode 100644\n"
    assert section + "--- a/app/legacy.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-OLD = True\n" in patch
    # created by integrated and extended by docs: ONE create holding the final text
    assert patch.count("diff --git a/app/feature.py") == 1
    assert "+FEATURE = 1\n+\n+# documented\n" in patch
    assert "-FEATURE" not in patch


@pytest.mark.parametrize(
    ("integrated", "docs", "path", "absent"),
    [
        pytest.param(
            {"app/legacy.py": None},
            {"app/legacy.py": "NEW = 1\n"},
            "app/legacy.py",
            "deleted file",
            id="deleted-then-recreated",
        ),
        pytest.param(
            {"app/main.py": "def main():\n    return 'v1'\n"},
            {"app/main.py": "def main():\n    return 'v2'\n"},
            "app/main.py",
            "return 'v1'",
            id="modified-twice",
        ),
    ],
)
def test_docs_are_rebased_onto_the_baseline(
    tmp_path: Path,
    integrated: dict[str, str | None],
    docs: dict[str, str | None],
    path: str,
    absent: str,
) -> None:
    """The merged change set is relative to the BASELINE, whatever integrated did first."""
    result = promote(build(tmp_path, integrated, docs))
    patch = patch_of(result)
    final = settled(settled(BASELINE, integrated), docs)
    assert bundle_of(result)["files"] == [path]
    assert patch == expected_patch(BASELINE, final)
    assert apply_patch(BASELINE, patch) == final
    # one plain MODIFY of the baseline file: no create / delete header, hunks of the baseline text
    assert f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n" in patch
    assert absent not in patch


def test_created_then_extended_file_is_exactly_one_create_op(tmp_path: Path) -> None:
    rig = build(tmp_path, {"app/feature.py": "one\n"}, {"app/feature.py": "one\ntwo\n"})
    stub = StubPort()
    result = promote(rig, params={"mode": "apply"}, effects=stub)
    (merged,) = stub.applied
    assert [(op.path, op.kind, op.content) for op in merged.ops] == [
        ("app/feature.py", FileOpKind.CREATE, "one\ntwo\n")
    ]
    assert patch_of(result) == (
        "diff --git a/app/feature.py b/app/feature.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/app/feature.py\n@@ -0,0 +1,2 @@\n+one\n+two\n"
    )


def test_integrated_and_docs_on_different_paths_are_both_released_sorted(tmp_path: Path) -> None:
    integrated: dict[str, str | None] = {
        "z_last.py": "z\n",
        "app/util.py": "def helper():\n    pass\n",
    }
    docs: dict[str, str | None] = {"README.md": "# Demo v2\n", "docs/guide.md": "guide v0\nmore\n"}
    result = promote(build(tmp_path, integrated, docs))
    final = settled(settled(BASELINE, integrated), docs)
    assert bundle_of(result)["files"] == ["README.md", "app/util.py", "docs/guide.md", "z_last.py"]
    assert apply_patch(BASELINE, patch_of(result)) == final


def test_awkward_but_valid_file_names_survive_into_the_patch(tmp_path: Path) -> None:
    nfc = "donn\u00e9es/caf\u00e9 menu.txt"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc and unicodedata.is_normalized("NFC", nfc)
    spaced, dashed, odd = (
        "user guide/read me.md",
        "-dash/--flag file.txt",
        "odd(name) & co;$x/#1.txt",
    )
    baseline = {**BASELINE, spaced: "old\n"}
    integrated: dict[str, str | None] = {
        spaced: "new\n",
        nfd: "created\n",
        dashed: "d\n",
        odd: "o\n",
    }
    docs: dict[str, str | None] = {nfd: "created\nmore\n"}  # the decomposed spelling again
    result = promote(build(tmp_path, integrated, docs, baseline=baseline))
    patch, bundle = patch_of(result), bundle_of(result)
    final = {**baseline, spaced: "new\n", nfc: "created\nmore\n", dashed: "d\n", odd: "o\n"}
    assert bundle["files"] == sorted([spaced, nfc, dashed, odd])  # one NFC spelling, sorted
    assert nfd not in patch and f"diff --git a/{nfc} b/{nfc}\n" in patch
    assert f"--- a/{spaced}\n+++ b/{spaced}\n" in patch
    assert apply_patch(baseline, patch) == final
    assert bundle["base_manifest_hash"] == manifest_hash(baseline)


def test_empty_files_and_files_without_a_trailing_newline(tmp_path: Path) -> None:
    baseline = {
        "empty.txt": "",
        "to_empty.txt": "x\n",
        "from_empty.txt": "",
        "no_eol.txt": "last line",
        "add_eol.txt": "text",
        "drop_eol.txt": "text\n",
        "same.txt": "same\n",
    }
    integrated: dict[str, str | None] = {
        "empty.txt": None,
        "to_empty.txt": "",
        "from_empty.txt": "y\n",
        "no_eol.txt": "last line changed",
        "add_eol.txt": "text\n",
        "drop_eol.txt": "text",
        "new_no_eol.txt": "no newline",
        "new_empty.txt": "",
    }
    patch = patch_of(promote(build(tmp_path, integrated, None, baseline=baseline)))
    assert apply_patch(baseline, patch) == settled(baseline, integrated)
    # no_eol: old + new line, add_eol: old line, drop_eol: new line, new_no_eol: new line
    assert patch.count(NO_EOL + "\n") == 5
    assert "diff --git a/new_empty.txt b/new_empty.txt\nnew file mode 100644\ndiff --git" in patch
    assert "same.txt" not in patch


@pytest.mark.xfail(
    strict=False,
    reason="known bug in tools.changeset._split_lines (not owned by the promoter): it uses "
    "str.splitlines, which also splits at form feed, U+2028, U+0085 ...; the patch then carries "
    "spurious 'No newline at end of file' markers and git rejects it",
)
def test_files_with_a_form_feed_or_a_unicode_line_separator_give_an_applicable_patch(
    tmp_path: Path,
) -> None:
    baseline = {"a.txt": "one\n\x0c\ntwo\n", "c.txt": "p\u2028q\nr\n"}
    integrated: dict[str, str | None] = {
        "a.txt": "one\n\x0c\ntwo changed\n",
        "b.txt": "x\u2028y\n",
        "c.txt": "p\u2028q\nr changed\n",
    }
    result = promote(build(tmp_path, integrated, None, baseline=baseline))
    assert apply_patch(baseline, patch_of(result)) == settled(baseline, integrated)


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_git_apply_reproduces_the_final_tree(tmp_path: Path) -> None:
    nfc = "donn\u00e9es/caf\u00e9 menu.txt"
    baseline = {
        "user guide/read me.md": "old\n",
        "no_eol.txt": "last line",
        "empty.txt": "",
        "app/legacy.py": "OLD = True\n",
        "keep.txt": "keep\n",
    }
    integrated: dict[str, str | None] = {
        "user guide/read me.md": "new\n",
        nfc: "created\n",
        "app/legacy.py": None,
        "new_empty.txt": "",
        "no_eol.txt": "last line changed",
    }
    docs: dict[str, str | None] = {nfc: "created\nmore\n"}
    patch = patch_of(promote(build(tmp_path / "rig", integrated, docs, baseline=baseline)))
    tree = tmp_path / "tree"
    tree.mkdir()
    write_tree(tree, baseline)
    patch_file = tmp_path / "release.patch"
    patch_file.write_bytes(patch.encode("utf-8"))
    empty_config = tmp_path / "empty.gitconfig"
    empty_config.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(empty_config),
        "GIT_CONFIG_SYSTEM": str(empty_config),
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    argv = [str(GIT), "-c", "core.autocrlf=false", "-c", "core.quotepath=false"]
    argv += ["apply", "--whitespace=nowarn", str(patch_file)]
    done = subprocess.run(
        argv,
        cwd=tree,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    final = settled(settled(baseline, integrated), docs)
    assert tree_bytes(tree) == as_bytes(final)


# ------------------------------------------------------------------ baseline wiring


def test_baseline_is_read_through_the_guarded_baseline_view(tmp_path: Path) -> None:
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    agent = PromoterAgent()
    ctx = rig.context(agent, "promote")
    # the node's own view is baseline + integrated + docs: a created file is already visible ...
    assert ctx.toolbox.read_text("app/feature.py") == FINAL["app/feature.py"]
    # ... yet the patch is relative to the BASELINE, read only through the audited baseline view
    patch = text_of(agent.run(ctx), Kind.RELEASE_PATCH)
    assert patch == expected_patch(BASELINE, FINAL)
    tools = {e["tool"] for e in rig.audit.of(EventType.TOOL_CALL)}
    assert tools == {"read_text", "baseline_read_text", "baseline_list_files"}
    fresh = build(tmp_path / "fresh", INTEGRATED_FILES, DOCS_FILES)
    promote(fresh)
    used = {e["tool"] for e in fresh.audit.of(EventType.TOOL_CALL)}
    assert used == {"baseline_read_text", "baseline_list_files"}


@pytest.mark.parametrize(
    ("mode", "integrated", "docs"),
    [
        pytest.param("bundle", {"app/feature.py": "x\n"}, None, id="create"),
        pytest.param("bundle", {"app/main.py": "changed\n"}, None, id="modify"),
        pytest.param("bundle", {"app/legacy.py": None}, None, id="delete"),
        pytest.param("bundle", None, {"README.md": "# changed\n"}, id="docs-only"),
        pytest.param("apply", {"app/feature.py": "x\n"}, None, id="create-apply"),
    ],
)
def test_without_a_baseline_the_attempt_fails_fatal_and_emits_no_patch(
    tmp_path: Path,
    mode: str,
    integrated: dict[str, str | None] | None,
    docs: dict[str, str | None] | None,
) -> None:
    rig = build(tmp_path, integrated, docs)
    stub = StubPort()
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": mode}, effects=stub, baseline_handed_in=False)
    assert_fatal(caught, "baseline")
    assert isinstance(caught.value.__cause__, ChangeSetConflict)
    assert stub.applied == []  # nothing is applied against the wrong tree either


@pytest.mark.parametrize(
    "shape",
    ["modified-meanwhile", "created-meanwhile", "deleted-ghost", "docs-against-baseline"],
)
def test_changesets_that_do_not_apply_to_the_baseline_are_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    rig = Rig(tmp_path, baseline_files=BASELINE)
    lenient_views(rig, monkeypatch)
    older = {**BASELINE, "app/main.py": "def main():\n    return 'older'\n"}
    without_util = {p: t for p, t in BASELINE.items() if p != "app/util.py"}
    kind = Kind.INTEGRATED_CHANGESET
    path = "app/main.py"
    if shape == "modified-meanwhile":  # made against an older version of the file
        rig.publish_changeset(kind, {path: "new\n"}, base=older)
    elif shape == "created-meanwhile":  # the file was added to the baseline since
        path = "app/util.py"
        rig.publish_changeset(kind, {path: "new\n"}, base=without_util)
    elif shape == "deleted-ghost":  # deletes a file the baseline does not have
        path = "ghost.py"
        rig.publish_changeset(kind, {path: None}, base={**BASELINE, path: "x\n"})
    else:  # docs made against the baseline although integrated changed the same file
        rig.publish_changeset(kind, {path: "integrated\n"})
    docs = {path: "docs\n"} if shape == "docs-against-baseline" else {}
    rig.publish_changeset(Kind.DOCS_CHANGESET, docs)  # relative to the BASELINE (wrong for docs)
    rig.publish(Kind.READINESS_REPORT, readiness())
    with pytest.raises(AgentError) as caught:
        promote(rig)
    assert_fatal(caught, "do not apply to the baseline", path)


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize("verdict", ["NO_GO", "go", "GO ", None, "missing"])
def test_a_readiness_verdict_other_than_go_is_refused(tmp_path: Path, verdict: str | None) -> None:
    report = readiness("NO_GO" if verdict is None else verdict)
    if verdict == "missing":
        del report["verdict"]
    elif verdict is None:
        report["verdict"] = None
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES, report=report)
    stub = StubPort()
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": "apply"}, effects=stub)
    assert_fatal(caught, "not GO")
    assert stub.applied == []


@pytest.mark.parametrize("mode", ["apply ", "APPLY", "", None, 5, ["apply"]])
def test_an_unknown_mode_is_fatal_and_never_applies(tmp_path: Path, mode: Any) -> None:
    assert MODES == ("bundle", "apply")  # the only two modes there are
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    stub = StubPort()
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": mode}, effects=stub)
    assert_fatal(caught, "mode must be one of")
    assert stub.applied == []


@pytest.mark.parametrize("case", ["both-empty", "docs-revert-integrated", "docs-delete-created"])
def test_a_merged_change_set_without_any_effect_is_fatal(tmp_path: Path, case: str) -> None:
    integrated: dict[str, str | None] = {}
    docs: dict[str, str | None] = {}
    if case == "docs-revert-integrated":
        integrated = {"app/main.py": "temporary\n"}
        docs = {"app/main.py": BASELINE["app/main.py"]}
    elif case == "docs-delete-created":
        integrated = {"app/new.py": "x\n"}
        docs = {"app/new.py": None}
    stub = StubPort()
    with pytest.raises(AgentError) as caught:
        promote(build(tmp_path, integrated, docs), params={"mode": "apply"}, effects=stub)
    assert_fatal(caught, "nothing to release")
    assert stub.applied == []


def test_a_case_insensitive_path_clash_between_the_change_sets_is_refused(tmp_path: Path) -> None:
    """One file on Windows / macOS: integrated creates Guide2.md, docs creates guide2.md."""
    rig = build(tmp_path, {"docs/Guide2.md": "a\n"}, {"docs/guide2.md": "b\n"})
    stub = StubPort()
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": "apply"}, effects=stub)
    assert_fatal(caught, "not valid", "case-insensitive")
    assert stub.applied == []


def test_a_release_larger_than_one_change_set_can_hold_is_refused(tmp_path: Path) -> None:
    half = MAX_CHANGESET_OPS // 2 + 1  # each set is legal, the merged release is not
    integrated: dict[str, str | None] = {f"a/f{i}.txt": "x\n" for i in range(half)}
    docs: dict[str, str | None] = {f"b/f{i}.txt": "y\n" for i in range(half)}
    with pytest.raises(AgentError) as caught:
        promote(build(tmp_path, integrated, docs))
    assert_fatal(caught, "not valid", "too many ops")


@pytest.mark.parametrize(
    "path", ["../evil.txt", "/abs/evil.txt", ".git/hooks/pre-commit", "C:/evil.txt", "a\\b.txt"]
)
def test_a_forged_change_set_with_an_escaping_path_is_never_rendered(
    tmp_path: Path, path: str
) -> None:
    rig = Rig(tmp_path, baseline_files=BASELINE)
    op = {"path": path, "kind": "create", "content": "x\n", "base_hash": None}
    forged = {"description": "", "author": "", "ops": [op]}
    rig.publish(Kind.INTEGRATED_CHANGESET, forged, validate=False)
    rig.publish_changeset(Kind.DOCS_CHANGESET, {})
    rig.publish(Kind.READINESS_REPORT, readiness())
    stub = StubPort()
    with pytest.raises(AutonomyViolation):  # CRITICAL for the engine, not a swallowed error
        promote(rig, params={"mode": "apply"}, effects=stub)
    assert stub.applied == []


# ------------------------------------------------------------------ apply mode


def test_apply_mode_writes_the_release_through_the_port(applied: Applied) -> None:
    bundle, receipt = applied.bundle, applied.bundle["receipt"]
    patch = patch_of(applied.result)
    assert tree_bytes(applied.target) == as_bytes(SMALL_FINAL)  # real files, written correctly
    assert not (applied.target / "app" / "legacy.py").exists()
    assert bundle["mode"] == "applied" and bundle["files"] == SMALL_TOUCHED
    assert patch == expected_patch(SMALL, SMALL_FINAL)  # the patch is still produced
    assert apply_patch(SMALL, patch) == SMALL_FINAL
    assert bundle["patch_sha256"] == sha256_text(patch)
    assert bundle["base_manifest_hash"] == manifest_hash(SMALL)
    # the receipt is the port's proof, stored as plain JSON and loadable again
    assert ApplyReceipt.from_dict(receipt).to_dict() == receipt
    assert json.loads(json.dumps(receipt)) == receipt
    assert receipt["target_id"] == applied.port.target_id and receipt["applied"] == SMALL_TOUCHED
    assert receipt["pre_hashes"]["app/feature.py"] is None
    assert receipt["post_hashes"]["app/legacy.py"] is None
    assert receipt["post_hashes"]["README.md"] == sha256_text(SMALL_FINAL["README.md"])
    assert receipt["post_hashes"]["app/feature.py"] == sha256_text(SMALL_FINAL["app/feature.py"])
    assert_valid(applied.result)
    # audited by the port: STARTED before the first mutation, COMPLETED after the last
    audit = applied.rig.audit
    (started,), (completed,) = (
        audit.of(EventType.EFFECT_STARTED),
        audit.of(EventType.EFFECT_COMPLETED),
    )
    assert started["paths"] == SMALL_TOUCHED and completed["ok"] is True
    # no absolute path leaks into the persisted bundle
    dumped = json.dumps(bundle)
    assert applied.target.as_posix() not in dumped
    assert json.dumps(str(applied.target))[1:-1] not in dumped


def test_apply_hands_the_merged_change_set_to_the_port_and_stores_its_receipt(
    tmp_path: Path,
) -> None:
    expected = compute_ops(MemoryView(BASELINE), {p: FINAL.get(p) for p in {*BASELINE, *FINAL}})
    paths = tuple(op.path for op in expected.ops)
    receipt = ApplyReceipt(
        "dir:stub", paths, dict.fromkeys(paths), dict.fromkeys(paths), ChangeSet()
    )
    stub = StubPort(receipt=receipt)
    result = promote(
        build(tmp_path, INTEGRATED_FILES, DOCS_FILES), params={"mode": "apply"}, effects=stub
    )
    (merged,) = stub.applied
    assert merged.ops == expected.ops  # relative to the BASELINE, docs folded in
    assert (merged.author, stub.reverted) == ("promoter", [])
    bundle = bundle_of(result)
    assert bundle["mode"] == "applied" and bundle["receipt"] == receipt.to_dict()
    assert_valid(result)


def test_apply_mode_without_an_effect_port_is_fatal_not_a_bundle(tmp_path: Path) -> None:
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": "apply"})
    assert_fatal(caught, "effect port", "bundle")


@pytest.mark.parametrize("drift", ["edited", "deleted", "appeared"])
def test_a_drifted_target_is_refused_and_left_untouched(tmp_path: Path, drift: str) -> None:
    rig = build(tmp_path / "rig", SMALL_INTEGRATED, SMALL_DOCS, baseline=SMALL)
    target = target_with(tmp_path, SMALL)
    if drift == "edited":  # a baseline file the release modifies was edited after the snapshot
        write_tree(target, {"app/main.py": "def main():\n    return 'human'\n"})
    elif drift == "deleted":
        (target / "app" / "main.py").unlink()
    else:  # a file the release wants to create appeared in the target
        write_tree(target, {"app/feature.py": "FEATURE = 'human'\n"})
    before = tree_bytes(target)
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": "apply"}, effects=make_port(target, rig))
    assert_fatal(caught, "target refused")
    assert isinstance(caught.value.__cause__, ChangeSetConflict)
    assert tree_bytes(target) == before
    assert rig.audit.of(EventType.EFFECT_STARTED) == []  # refused before anything was touched


@pytest.mark.parametrize("path", [".github/workflows/ci.yml", "Dockerfile", ".env.production"])
def test_a_change_set_touching_a_protected_path_is_refused_by_the_port(
    tmp_path: Path, path: str
) -> None:
    rig = build(tmp_path / "rig", {path: "x\n", "app/ok.py": "ok = 1\n"}, None, baseline=SMALL)
    target = target_with(tmp_path, SMALL)
    before = tree_bytes(target)
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": "apply"}, effects=make_port(target, rig))
    assert_fatal(caught, "protected")
    assert isinstance(caught.value.__cause__, ProtectedPathError)
    assert tree_bytes(target) == before  # the harmless op did not apply either
    assert rig.audit.of(EventType.EFFECT_STARTED) == []


def test_an_unsafe_target_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    def dirty(root: Path) -> None:
        raise UnsafeTargetError("refusing to modify a git working tree with uncommitted changes")

    rig = build(tmp_path / "rig", SMALL_INTEGRATED, SMALL_DOCS, baseline=SMALL)
    target = target_with(tmp_path, SMALL)
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": "apply"}, effects=make_port(target, rig, git_guard=dirty))
    assert_fatal(caught, "uncommitted")
    assert isinstance(caught.value.__cause__, UnsafeTargetError)
    assert tree_bytes(target) == as_bytes(SMALL)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ChangeSetConflict("app/main.py", "drift"), id="conflict"),
        pytest.param(ProtectedPathError("protected"), id="protected"),
        pytest.param(UnsafeTargetError("dirty tree"), id="unsafe-target"),
        pytest.param(RevertRefused("changed"), id="revert-refused"),
        pytest.param(RollbackIncompleteError("manual reconciliation"), id="rollback-incomplete"),
    ],
)
def test_every_refusal_of_the_effect_port_is_fatal(tmp_path: Path, error: Exception) -> None:
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": "apply"}, effects=StubPort(error=error))
    assert_fatal(caught, "target refused", str(error))
    assert caught.value.__cause__ is error


def test_refusal_messages_are_single_line_and_bounded(tmp_path: Path) -> None:
    hostile = EffectError("first line\nsecond \x1b[31mred " + "x" * 500)
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    with pytest.raises(AgentError) as caught:
        promote(rig, params={"mode": "apply"}, effects=StubPort(error=hostile))
    message = str(caught.value)
    assert "\n" not in message and "\x1b" not in message and len(message) < 300


def test_a_cancel_before_the_effect_propagates_and_the_target_is_untouched(tmp_path: Path) -> None:
    token = CancelToken()
    rig = build(tmp_path / "rig", SMALL_INTEGRATED, SMALL_DOCS, baseline=SMALL)
    target = target_with(tmp_path, SMALL)
    port = make_port(target, rig, cancel=token)

    class CancelThenApply:
        def apply(self, cs: ChangeSet, *, dry_run: bool = False) -> ApplyReceipt:
            token.cancel("kill switch")  # the kill switch fires right before the effect
            return port.apply(cs, dry_run=dry_run)

        def revert(self, receipt: ApplyReceipt) -> None:
            port.revert(receipt)

    with pytest.raises(AgentCancelled):  # not mapped to a fatal refusal
        promote(rig, params={"mode": "apply"}, effects=CancelThenApply(), cancel=token)
    assert tree_bytes(target) == as_bytes(SMALL)
    assert rig.audit.of(EventType.EFFECT_STARTED) == []


def test_a_run_cancelled_before_it_starts_reads_nothing(tmp_path: Path) -> None:
    token = CancelToken()
    token.cancel("kill switch")
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    with pytest.raises(AgentCancelled):
        promote(rig, cancel=token)
    assert rig.audit.of(EventType.TOOL_CALL) == []


# ------------------------------------------------------------------ the compensator


@pytest.mark.parametrize("form", ["receipt", "dict"])
def test_the_compensator_reverts_from_the_receipt_or_its_persisted_form(
    applied: Applied, form: str
) -> None:
    receipt: Any = applied.receipt if form == "receipt" else applied.bundle["receipt"]
    later_process = make_port(applied.target, applied.rig)  # a new port: same root, same target id
    note = compensate(receipt, later_process)
    assert tree_bytes(applied.target) == as_bytes(SMALL)  # exactly the baseline again
    assert not (applied.target / "app" / "feature.py").exists()  # the created file is gone
    assert note.startswith(f"reverted {len(SMALL_TOUCHED)} file(s) applied to ")
    assert applied.port.target_id in note and "app/feature.py" in note


def test_compensating_twice_is_a_no_op(applied: Applied) -> None:
    first = compensate(applied.bundle["receipt"], applied.port)
    second = compensate(applied.receipt, applied.port)  # nothing left to do, returns normally
    assert first == second
    assert tree_bytes(applied.target) == as_bytes(SMALL)
    started = applied.rig.audit.of(EventType.EFFECT_STARTED)
    assert [s["action"] for s in started] == ["apply", "revert", "revert"]
    assert [s["noop"] for s in started[1:]] == [False, True]


def test_a_file_edited_after_the_apply_makes_the_compensation_refuse(applied: Applied) -> None:
    (applied.target / "app" / "main.py").write_bytes(b"human edit after the release\n")
    after_edit = tree_bytes(applied.target)
    with pytest.raises(RevertRefused, match="changed since"):
        compensate(applied.receipt, applied.port)
    assert tree_bytes(applied.target) == after_edit  # nothing at all was reverted


def test_a_receipt_of_another_target_is_refused(applied: Applied) -> None:
    after_apply = tree_bytes(applied.target)
    foreign = {**applied.bundle["receipt"], "target_id": "dir:" + "0" * 16}
    with pytest.raises(RevertRefused, match="different target"):
        compensate(foreign, applied.port)
    assert tree_bytes(applied.target) == after_apply


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({}, id="empty"),
        pytest.param({"target_id": "dir:x"}, id="no-reverse"),
        pytest.param({"target_id": "dir:x", "reverse": "not a change set"}, id="reverse-str"),
        pytest.param({"target_id": "dir:x", "reverse": ["op"]}, id="reverse-list"),
        pytest.param({"target_id": "dir:x", "reverse": {"ops": 5}}, id="ops-int"),
        pytest.param(
            {"target_id": "dir:x", "reverse": {"ops": [{"path": "a", "kind": "explode"}]}},
            id="bad-kind",
        ),
        pytest.param({"target_id": "dir:x", "applied": 5, "reverse": {}}, id="applied-int"),
        pytest.param({"target_id": "dir:x", "pre_hashes": [1, 2], "reverse": {}}, id="hashes"),
        pytest.param("not a mapping", id="str"),
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param(5, id="int"),
    ],
)
def test_a_malformed_persisted_receipt_is_fatal_and_reverts_nothing(raw: Any) -> None:
    stub = StubPort()
    with pytest.raises(AgentError) as caught:
        compensate(raw, stub)
    assert_fatal(caught, "receipt")
    assert stub.reverted == []


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", ".git/hooks/pre-commit", "C:/x"])
def test_a_forged_receipt_with_an_escaping_path_is_refused(path: str) -> None:
    forged = {
        "target_id": "dir:x",
        "applied": [path],
        "pre_hashes": {path: None},
        "post_hashes": {path: "0" * 64},
        "reverse": {
            "ops": [{"path": path, "kind": "delete", "content": None, "base_hash": "0" * 64}]
        },
    }
    stub = StubPort()
    with pytest.raises(AgentError):
        compensate(forged, stub)
    assert stub.reverted == []


def test_a_forged_receipt_cannot_rewrite_a_protected_file(tmp_path: Path) -> None:
    current = "FROM scratch\n"
    rig = Rig(tmp_path / "rig")
    target = target_with(tmp_path, {"Dockerfile": current})
    port = make_port(target, rig)
    forged = {
        "target_id": port.target_id,
        "applied": ["Dockerfile"],
        "pre_hashes": {"Dockerfile": sha256_text("older")},
        "post_hashes": {"Dockerfile": sha256_text(current)},
        "reverse": {
            "ops": [
                {
                    "path": "Dockerfile",
                    "kind": "modify",
                    "content": "FROM evil\n",
                    "base_hash": sha256_text(current),
                }
            ]
        },
    }
    with pytest.raises(ProtectedPathError):
        compensate(forged, port)
    assert tree_bytes(target) == as_bytes({"Dockerfile": current})


def test_the_description_names_the_file_count_and_samples_at_most_eight_paths() -> None:
    paths = tuple(f"pkg/m{i:02d}.py" for i in range(11))
    receipt = ApplyReceipt(
        "dir:stub", paths, dict.fromkeys(paths), dict.fromkeys(paths, "0" * 64), ChangeSet()
    )
    stub = StubPort()
    note = compensate(receipt.to_dict(), stub)
    assert stub.reverted == [receipt]  # the dict form is turned into a real ApplyReceipt
    assert note.startswith("reverted 11 file(s) applied to dir:stub: pkg/m00.py, pkg/m01.py")
    assert "pkg/m07.py" in note and "pkg/m08.py" not in note and note.endswith("(+3 more)")


def test_the_compensator_satisfies_the_engine_protocol() -> None:
    assert isinstance(ApplyBundleCompensator(), Compensator)


# ------------------------------------------------------------------ decisions, determinism


@pytest.mark.parametrize(("mode", "outcome"), [("bundle", "bundle"), ("apply", "applied")])
def test_exactly_one_release_decision_whose_outcome_is_the_bundle_mode(
    tmp_path: Path, mode: str, outcome: str
) -> None:
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    result = promote(rig, params={"mode": mode}, effects=StubPort())
    bundle = bundle_of(result)
    (decision,) = result.decisions
    assert decision.kind is DecisionKind.RELEASE
    assert decision.outcome == outcome == bundle["mode"]
    assert decision.data == {"patch_sha256": bundle["patch_sha256"], "files": len(CHANGED)}
    assert f"{len(CHANGED)} file(s)" in decision.summary
    assert decision.parents == () and decision.refs == ()
    assert result.plan_expansion is None and result.clarifications == []


def test_two_runs_with_different_clocks_produce_identical_artifact_content(tmp_path: Path) -> None:
    early = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    late = datetime(2031, 12, 30, 23, 59, 58, tzinfo=UTC)
    first = promote(build(tmp_path / "a", INTEGRATED_FILES, DOCS_FILES, clock_start=early))
    second = promote(build(tmp_path / "b", INTEGRATED_FILES, DOCS_FILES, clock_start=late))
    assert content_hashes(first) == content_hashes(second)
    assert patch_of(first).encode("utf-8") == patch_of(second).encode("utf-8")
    assert first.decisions == second.decisions


def test_the_agent_works_with_only_its_declared_inputs(tmp_path: Path) -> None:
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    rig.publish(Kind.REQUIREMENTS, requirements_payload())  # in the store, but not declared
    agent = PromoterAgent()
    ctx = rig.context(agent, "promote")
    for kind in (Kind.REQUIREMENTS, Kind.TASK_PLAN, Kind.TEST_REPORT):
        with pytest.raises(UndeclaredInputError):
            ctx.has(kind)
    assert_valid(agent.run(ctx))


def test_the_drafts_are_valid_and_survive_the_artifact_store_byte_for_byte(tmp_path: Path) -> None:
    rig = build(tmp_path, INTEGRATED_FILES, DOCS_FILES)
    result = promote(rig)
    assert_valid(result)
    stored_patch, stored_bundle = (rig.store.put(d, producer="promote") for d in result.artifacts)
    text = rig.store.read_text(stored_patch)
    assert text == patch_of(result)
    assert sha256_text(text) == bundle_of(result)["patch_sha256"]
    assert rig.store.read_json(stored_bundle) == bundle_of(result)


# ------------------------------------------------------------------ budget and autonomy


def test_a_big_baseline_is_read_once_per_file_within_the_tool_call_budget(tmp_path: Path) -> None:
    baseline = {f"pkg/mod_{i:03d}.py": f"VALUE = {i}\n" for i in range(300)}
    integrated: dict[str, str | None] = {
        f"pkg/mod_{i:03d}.py": f"VALUE = {i + 1}\n" for i in range(0, 300, 15)
    }
    integrated.update({"pkg/mod_007.py": None, "pkg/new_a.py": "A = 1\n"})
    docs: dict[str, str | None] = {
        "pkg/new_a.py": "A = 1\nB = 2\n",  # extends a file integrated created
        "pkg/mod_000.py": "VALUE = 'docs'\n",  # touches a file integrated modified
        "pkg/mod_001.py": "VALUE = 'one'\n",
    }
    rig = build(tmp_path, integrated, docs, baseline=baseline)
    result = promote(rig)
    final = settled(settled(baseline, integrated), docs)
    assert apply_patch(baseline, patch_of(result)) == final
    assert bundle_of(result)["base_manifest_hash"] == manifest_hash(baseline)
    calls = rig.audit.of(EventType.TOOL_CALL)
    reads = Counter(c["target"] for c in calls if c["tool"] == "baseline_read_text")
    assert set(reads) == {*baseline, "pkg/new_a.py"} and set(reads.values()) == {1}
    assert [c["tool"] for c in calls].count("baseline_list_files") == 1
    assert len(calls) == len(baseline) + 2 == result.usage["tool_calls"]
    assert len(calls) <= PromoterAgent.autonomy.max_tool_calls == MAX_BASELINE_READS


def test_a_budget_that_is_too_small_fails_loudly_without_a_partial_patch(tmp_path: Path) -> None:
    baseline = {f"f{i:02d}.txt": "x\n" for i in range(30)}
    rig = build(tmp_path, {"f01.txt": "y\n"}, None, baseline=baseline)
    tight = {"autonomy": AutonomyProfile(max_tool_calls=10)}
    with pytest.raises(BudgetExhausted) as caught:
        promote(rig, node_overrides=tight)
    assert caught.value.retryable is False


def test_the_promoter_is_read_only_and_cannot_execute(tmp_path: Path) -> None:
    agent = PromoterAgent()
    assert (agent.agent_id, agent.role) == ("promoter", "promoter")
    assert agent.autonomy.write_globs == () and agent.autonomy.can_execute is False
    assert agent.autonomy.read_globs == ("**",)
    ctx: AgentContext = build(tmp_path, INTEGRATED_FILES, DOCS_FILES).context(agent, "promote")
    with pytest.raises(ToolDenied):  # the real toolbox enforces the (empty) write scope
        ctx.toolbox.write_text("x.txt", "y")
    with pytest.raises(ToolDenied):  # ... and the execution switch
        ctx.toolbox.run_python(["-c", "print(1)"])
    assert ctx.toolbox.usage()["bytes_written"] == 0.0
