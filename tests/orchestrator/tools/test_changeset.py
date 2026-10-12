"""Tests for ``orchestrator.tools.changeset``: build, apply, merge, reverse, diff, digest."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.core.errors import ConfigError, PathViolation
from orchestrator.core.fsutil import normalize_newlines, sha256_text
from orchestrator.core.types import ChangeSet, FileOp, FileOpKind
from orchestrator.tools.changeset import (
    ChangeSetConflict,
    DiffStats,
    apply_changeset,
    compute_ops,
    diff_stats,
    digest,
    make_create,
    make_delete,
    make_modify,
    merge_changesets,
    reverse_changeset,
    to_unified_diff,
)
from orchestrator.tools.workspace import MemoryView, materialize

BASE = {
    "app/main.py": "print('hello')\n",
    "app/util.py": "def f():\n    return 1\n",
    "README.md": "# demo\n",
}


def view() -> MemoryView:
    return MemoryView(BASE)


def cs_of(*ops: FileOp, author: str = "") -> ChangeSet:
    return ChangeSet(tuple(ops), "", author)


# ------------------------------ builders


def test_make_create_normalises_newlines_and_has_no_base_hash() -> None:
    op = make_create("new.txt", "a\r\nb\rc\n")
    assert op.kind == FileOpKind.CREATE
    assert op.content == "a\nb\nc\n"
    assert op.base_hash is None


def test_make_modify_binds_the_hash_of_the_current_content() -> None:
    op = make_modify(view(), "README.md", "# changed\r\n")
    assert op.kind == FileOpKind.MODIFY
    assert op.content == "# changed\n"
    assert op.base_hash == sha256_text(BASE["README.md"])


def test_make_modify_and_make_delete_refuse_missing_files() -> None:
    with pytest.raises(ChangeSetConflict, match="does not exist"):
        make_modify(view(), "nope.py", "x")
    with pytest.raises(ChangeSetConflict, match="does not exist"):
        make_delete(view(), "nope.py")


def test_make_delete_binds_the_hash_and_carries_no_content() -> None:
    op = make_delete(view(), "app/util.py")
    assert op.kind == FileOpKind.DELETE
    assert op.content is None
    assert op.base_hash == sha256_text(BASE["app/util.py"])


@pytest.mark.parametrize("bad", ["../x", "/abs", "C:x", "a:b", "CON", ".git/config", "a\\b"])
def test_builders_reject_hostile_paths(bad: str) -> None:
    with pytest.raises(PathViolation):
        make_create(bad, "x")


# ------------------------------ compute_ops


def test_compute_ops_detects_create_modify_delete_and_drops_noops() -> None:
    cs = compute_ops(
        view(),
        {
            "app/main.py": "print('hello')\n",  # unchanged -> dropped
            "app/util.py": "def f():\n    return 2\n",  # modified
            "README.md": None,  # deleted
            "docs/new.md": "new\n",  # created
            "ghost.txt": None,  # delete of a file that does not exist -> dropped
        },
        description="d",
        author="a1",
    )
    by_path = {op.path: op for op in cs.ops}
    assert sorted(by_path) == ["README.md", "app/util.py", "docs/new.md"]
    assert by_path["README.md"].kind == FileOpKind.DELETE
    assert by_path["app/util.py"].kind == FileOpKind.MODIFY
    assert by_path["app/util.py"].base_hash == sha256_text(BASE["app/util.py"])
    assert by_path["docs/new.md"].kind == FileOpKind.CREATE
    assert (cs.description, cs.author) == ("d", "a1")


def test_compute_ops_is_sorted_by_path_regardless_of_input_order() -> None:
    cs = compute_ops(MemoryView(), {"b.txt": "b", "a.txt": "a", "c/d.txt": "d"})
    assert cs.paths() == ("a.txt", "b.txt", "c/d.txt")


def test_compute_ops_treats_crlf_only_difference_as_no_change() -> None:
    cs = compute_ops(MemoryView({"a.txt": "x\ny\n"}), {"a.txt": "x\r\ny\r\n"})
    assert cs.ops == ()


def test_compute_ops_rejects_hostile_paths_and_case_collisions() -> None:
    with pytest.raises(PathViolation):
        compute_ops(MemoryView(), {"../escape.txt": "x"})
    with pytest.raises(ConfigError, match="case-insensitive"):
        compute_ops(MemoryView(), {"Foo.py": "x", "foo.py": "y"})


def test_compute_ops_rejects_a_hostile_path_even_when_the_op_would_be_dropped() -> None:
    """Regression: ``{"../x": None}`` used to vanish silently because the file 'does not exist'."""
    with pytest.raises(PathViolation):
        compute_ops(MemoryView(), {"../x": None})


def test_compute_ops_normalises_the_path_before_looking_it_up() -> None:
    """Regression: an NFD spelling of an existing NFC file used to become a CREATE of its twin."""
    nfc, nfd = "caf\u00e9.txt", "cafe\u0301.txt"
    existing = MemoryView({nfc: "old\n"})
    cs = compute_ops(existing, {nfd: "new\n"})
    (op,) = cs.ops
    assert (op.path, op.kind) == (nfc, FileOpKind.MODIFY)
    assert apply_changeset(existing, cs) == {nfc: "new\n"}


# ------------------------------ apply


def test_apply_returns_new_content_and_never_mutates_the_view() -> None:
    v = view()
    cs = cs_of(
        make_create("new.txt", "n\n"),
        make_modify(v, "README.md", "# x\n"),
        make_delete(v, "app/util.py"),
    )
    result = apply_changeset(v, cs)
    assert result == {"new.txt": "n\n", "README.md": "# x\n", "app/util.py": None}
    assert v.read_text("README.md") == BASE["README.md"]
    assert v.exists("app/util.py")
    assert not v.exists("new.txt")


def test_strict_apply_refuses_create_over_existing_file() -> None:
    with pytest.raises(ChangeSetConflict, match="already exists"):
        apply_changeset(view(), cs_of(make_create("README.md", "x")))


def test_strict_apply_refuses_modify_or_delete_of_missing_file() -> None:
    modify = FileOp("ghost.py", FileOpKind.MODIFY, "x", sha256_text("old"))
    delete = FileOp("ghost.py", FileOpKind.DELETE, None, sha256_text("old"))
    for op in (modify, delete):
        with pytest.raises(ChangeSetConflict, match="does not exist"):
            apply_changeset(view(), cs_of(op))


def test_strict_apply_detects_drift_through_the_base_hash() -> None:
    stale = FileOp("README.md", FileOpKind.MODIFY, "# x\n", sha256_text("something else\n"))
    with pytest.raises(ChangeSetConflict, match="hash mismatch") as excinfo:
        apply_changeset(view(), cs_of(stale))
    assert excinfo.value.path == "README.md"
    stale_delete = FileOp("README.md", FileOpKind.DELETE, None, sha256_text("something else\n"))
    with pytest.raises(ChangeSetConflict, match="hash mismatch"):
        apply_changeset(view(), cs_of(stale_delete))


def test_strict_apply_is_all_or_nothing_in_its_result() -> None:
    good = make_create("ok.txt", "ok")
    bad = make_create("README.md", "dup")
    with pytest.raises(ChangeSetConflict):
        apply_changeset(view(), cs_of(good, bad))


def test_non_strict_apply_skips_every_precondition() -> None:
    stale_modify = FileOp("README.md", FileOpKind.MODIFY, "# x\n", sha256_text("zzz"))
    missing_modify = FileOp("ghost.py", FileOpKind.MODIFY, "g\n", sha256_text("zzz"))
    missing_delete = FileOp("ghost2.py", FileOpKind.DELETE, None, sha256_text("zzz"))
    overwrite = make_create("app/main.py", "replaced\n")
    cs = cs_of(stale_modify, missing_modify, missing_delete, overwrite)
    assert apply_changeset(view(), cs, strict=False) == {
        "README.md": "# x\n",
        "ghost.py": "g\n",
        "ghost2.py": None,
        "app/main.py": "replaced\n",
    }


def test_apply_to_a_crlf_view_hashes_newline_insensitively() -> None:
    crlf_view = MemoryView({"a.txt": "x\r\ny\r\n"})
    op = make_modify(crlf_view, "a.txt", "x\ny\nz\n")
    assert apply_changeset(crlf_view, cs_of(op)) == {"a.txt": "x\ny\nz\n"}


# ------------------------------ digest


def test_digest_ignores_order_author_and_description_but_not_content() -> None:
    a, b = make_create("a.txt", "a"), make_create("b.txt", "b")
    one = ChangeSet((a, b), "first", "agent-1")
    two = ChangeSet((b, a), "second", "agent-2")
    assert digest(one) == digest(two)
    assert digest(one) != digest(cs_of(a, make_create("b.txt", "B")))
    assert digest(cs_of()) == digest(ChangeSet())


# ------------------------------ merge


def test_merge_of_disjoint_changesets_is_relative_to_the_base() -> None:
    v = view()
    c1 = ChangeSet((make_modify(v, "README.md", "# one\n"),), "", "impl.a")
    c2 = ChangeSet(
        (make_create("app/new.py", "x = 1\n"), make_delete(v, "app/util.py")), "", "impl.b"
    )
    merged = merge_changesets(v, [c1, c2])
    assert merged.conflicts == ()
    assert merged.merged_paths == ("README.md", "app/new.py", "app/util.py")
    assert merged.changeset.author == "integrator"
    assert apply_changeset(v, merged.changeset) == {
        "README.md": "# one\n",
        "app/new.py": "x = 1\n",
        "app/util.py": None,
    }
    modify = next(op for op in merged.changeset.ops if op.path == "README.md")
    assert modify.base_hash == sha256_text(BASE["README.md"])


def test_merge_accepts_identical_edits_to_the_same_path() -> None:
    v = view()
    c1 = ChangeSet((make_modify(v, "README.md", "# same\n"),), "", "a")
    c2 = ChangeSet((make_modify(v, "README.md", "# same\n"),), "", "b")
    merged = merge_changesets(v, [c1, c2])
    assert merged.conflicts == ()
    assert [op.path for op in merged.changeset.ops] == ["README.md"]


def test_merge_reports_conflicts_but_still_merges_unrelated_paths() -> None:
    v = view()
    c1 = ChangeSet((make_modify(v, "README.md", "# a\n"), make_create("only_a.txt", "a")), "", "a")
    c2 = ChangeSet((make_modify(v, "README.md", "# b\n"), make_create("only_b.txt", "b")), "", "b")
    merged = merge_changesets(v, [c1, c2])
    assert [(c.path, c.authors) for c in merged.conflicts] == [("README.md", ("a", "b"))]
    assert "different" in merged.conflicts[0].reason
    assert merged.merged_paths == ("only_a.txt", "only_b.txt")
    assert {op.path for op in merged.changeset.ops} == {"only_a.txt", "only_b.txt"}


def test_merge_detects_case_insensitive_path_collisions() -> None:
    """``Foo.py`` / ``foo.py`` collide on Windows and macOS even with equal content."""
    v = MemoryView()
    c1 = ChangeSet((make_create("pkg/Foo.py", "x = 1\n"),), "", "a")
    c2 = ChangeSet((make_create("pkg/foo.py", "x = 1\n"),), "", "b")
    merged = merge_changesets(v, [c1, c2])
    assert len(merged.conflicts) == 1
    assert merged.conflicts[0].authors == ("a", "b")
    assert merged.changeset.ops == ()
    assert merged.merged_paths == ()


def test_merge_flags_a_source_that_is_stale_against_the_base() -> None:
    v = view()
    stale = FileOp("README.md", FileOpKind.MODIFY, "# x\n", sha256_text("not the base"))
    fine = make_create("ok.txt", "ok\n")
    merged = merge_changesets(v, [ChangeSet((stale, fine), "", "late")])
    assert [(c.path, c.authors) for c in merged.conflicts] == [("README.md", ("late",))]
    assert "hash mismatch" in merged.conflicts[0].reason
    assert merged.merged_paths == ("ok.txt",)


def test_merge_delete_versus_modify_conflicts() -> None:
    v = view()
    c1 = ChangeSet((make_delete(v, "README.md"),), "", "a")
    c2 = ChangeSet((make_modify(v, "README.md", "# b\n"),), "", "b")
    merged = merge_changesets(v, [c1, c2])
    assert [c.path for c in merged.conflicts] == ["README.md"]
    assert merged.changeset.ops == ()


def test_merge_is_deterministic_and_handles_empty_input() -> None:
    v = view()
    cs = [ChangeSet((make_create("z.txt", "z"), make_create("a.txt", "a")), "", "x")]
    assert digest(merge_changesets(v, cs).changeset) == digest(merge_changesets(v, cs).changeset)
    empty = merge_changesets(v, [])
    assert (empty.changeset.ops, empty.conflicts, empty.merged_paths) == ((), (), ())


def test_merge_names_anonymous_sources_by_index() -> None:
    v = view()
    c1 = ChangeSet((make_create("n.txt", "1"),))
    c2 = ChangeSet((make_create("n.txt", "2"),))
    merged = merge_changesets(v, [c1, c2])
    assert merged.conflicts[0].authors == ("changeset[0]", "changeset[1]")


# ------------------------------ reverse


def after(v: MemoryView, cs: ChangeSet) -> MemoryView:
    """The view that results from applying ``cs`` to ``v``."""
    files = {p: v.read_text(p) for p in v.list_files("**")}
    for path, content in apply_changeset(v, cs).items():
        if content is None:
            files.pop(path, None)
        else:
            files[path] = content
    return MemoryView(files)


def mixed_changeset(before: MemoryView) -> ChangeSet:
    return cs_of(
        make_create("new/file.txt", "n\n"),
        make_modify(before, "README.md", "# changed\n"),
        make_delete(before, "app/util.py"),
    )


def test_reverse_round_trips_create_modify_delete() -> None:
    before = view()
    cs = mixed_changeset(before)
    changed = after(before, cs)
    assert changed.list_files("**") != before.list_files("**")

    reverse = reverse_changeset(before, cs)
    restored = after(changed, reverse)  # strict apply: base hashes must match the *after* tree
    assert {p: restored.read_text(p) for p in restored.list_files("**")} == BASE


def test_reverse_of_reverse_is_the_original_change_set() -> None:
    before = view()
    cs = mixed_changeset(before)
    reverse = reverse_changeset(before, cs)
    assert digest(reverse_changeset(after(before, cs), reverse)) == digest(cs)
    assert reverse.description == "reverse of: "


def test_reverse_of_a_non_strict_create_over_an_existing_file_restores_it() -> None:
    """Regression: the inverse used to be a DELETE, losing the overwritten content."""
    before = MemoryView({"a.txt": "original\n"})
    cs = cs_of(FileOp("a.txt", FileOpKind.CREATE, "overwritten\n"))
    changed = MemoryView({"a.txt": "overwritten\n"})
    assert apply_changeset(before, cs, strict=False) == {"a.txt": "overwritten\n"}
    reverse = reverse_changeset(before, cs)
    (op,) = reverse.ops
    assert op.kind == FileOpKind.MODIFY
    assert apply_changeset(changed, reverse) == {"a.txt": "original\n"}


def test_reverse_cannot_be_applied_to_a_tree_that_did_not_get_the_change() -> None:
    before = view()
    reverse = reverse_changeset(before, mixed_changeset(before))
    with pytest.raises(ChangeSetConflict):
        apply_changeset(before, reverse)


def test_reverse_fails_when_the_original_content_is_unknown() -> None:
    ghost = FileOp("ghost.py", FileOpKind.MODIFY, "x", sha256_text("y"))
    with pytest.raises(ChangeSetConflict, match="original content unknown"):
        reverse_changeset(view(), cs_of(ghost))
    ghost_delete = FileOp("ghost.py", FileOpKind.DELETE, None, sha256_text("y"))
    with pytest.raises(ChangeSetConflict, match="original content unknown"):
        reverse_changeset(view(), cs_of(ghost_delete))


# ------------------------------ diff stats


def test_diff_stats_counts_files_and_lines() -> None:
    v = view()
    cs = cs_of(
        make_create("new.txt", "a\nb\nc\n"),  # +3
        make_modify(v, "app/util.py", "def f():\n    return 2\n    # x\n"),  # -1 +2
        make_delete(v, "README.md"),  # -1
    )
    stats = diff_stats(v, cs)
    assert stats == DiffStats(1, 1, 1, 5, 2)
    assert stats.files_changed == 3
    assert stats.to_dict() == {
        "files_added": 1,
        "files_modified": 1,
        "files_deleted": 1,
        "lines_added": 5,
        "lines_removed": 2,
    }


def test_diff_stats_of_an_empty_change_set_is_zero() -> None:
    assert diff_stats(view(), ChangeSet()) == DiffStats()


def test_diff_stats_for_an_empty_file_has_no_lines() -> None:
    stats = diff_stats(MemoryView(), cs_of(make_create("empty.txt", "")))
    assert (stats.files_added, stats.lines_added, stats.lines_removed) == (1, 0, 0)


# ------------------------------ unified diff


def test_unified_diff_format_basics() -> None:
    v = view()
    cs = cs_of(
        make_modify(v, "README.md", "# changed\n"),
        make_create("a_new.txt", "x\n"),
        make_delete(v, "app/util.py"),
    )
    patch = to_unified_diff(v, cs)
    assert "\r" not in patch
    headers = [line for line in patch.splitlines() if line.startswith("diff --git")]
    assert headers == [  # sorted by path
        "diff --git a/README.md b/README.md",
        "diff --git a/a_new.txt b/a_new.txt",
        "diff --git a/app/util.py b/app/util.py",
    ]
    assert "new file mode 100644" in patch
    assert "deleted file mode 100644" in patch
    assert "--- /dev/null" in patch and "+++ /dev/null" in patch


def test_unified_diff_skips_no_op_modifications() -> None:
    v = view()
    noop = FileOp("README.md", FileOpKind.MODIFY, BASE["README.md"], sha256_text(BASE["README.md"]))
    assert to_unified_diff(v, cs_of(noop)) == ""
    assert to_unified_diff(v, ChangeSet()) == ""


GIT = shutil.which("git")


def _git_apply(base: MemoryView, cs: ChangeSet, workdir: Path) -> dict[str, str | None]:
    """Materialise ``base``, apply the generated patch with real git, return the resulting files."""
    workdir.mkdir()
    materialize(base, workdir)
    patch = workdir.parent / "change.patch"
    patch.write_bytes(to_unified_diff(base, cs).encode("utf-8"))
    git = [str(GIT), "-c", "core.autocrlf=false", "-c", "core.quotepath=false"]
    check = subprocess.run(
        [*git, "apply", "--check", str(patch)], cwd=workdir, capture_output=True, text=True
    )
    assert check.returncode == 0, check.stderr
    applied = subprocess.run(
        [*git, "apply", str(patch)], cwd=workdir, capture_output=True, text=True
    )
    assert applied.returncode == 0, applied.stderr
    result: dict[str, str | None] = {}
    for op in cs.ops:
        target = workdir / op.path
        result[op.path] = (
            normalize_newlines(target.read_bytes().decode("utf-8")) if target.exists() else None
        )
    return result


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_unified_diff_is_accepted_by_real_git_apply(tmp_path: Path) -> None:
    base = MemoryView(
        {
            "a.txt": "one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\n",
            "nonl.txt": "no newline at end",
            "gone.txt": "bye\n",
            "empty_gone.txt": "",
            "sp ace/f.txt": "x\n",
            "café.txt": "café\n",
        }
    )
    desired: dict[str, str | None] = {
        "a.txt": "ONE\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nTEN\nextra\n",  # two hunks
        "nonl.txt": "no newline at end!\n",
        "gone.txt": None,
        "empty_gone.txt": None,
        "new.txt": "fresh\n",
        "empty_new.txt": "",
        "nonl_new.txt": "tail without newline",
        "sp ace/f.txt": "y\n",
        "sp ace/new one.txt": "n\n",
        "café.txt": "café!\n",
        "näme.txt": "z\n",
        "tricky.txt": "diff --git a/x b/x\n@@ -1 +1 @@\n--- a/x\n+++ b/x\n-minus\n+plus\n",
    }
    cs = compute_ops(base, desired)
    applied = _git_apply(base, cs, tmp_path / "tree")
    assert applied == apply_changeset(base, cs)


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_unified_diff_of_a_trailing_newline_only_change_applies(tmp_path: Path) -> None:
    base = MemoryView({"a.txt": "line\n", "b.txt": "line"})
    cs = compute_ops(base, {"a.txt": "line", "b.txt": "line\n"})
    assert _git_apply(base, cs, tmp_path / "tree") == apply_changeset(base, cs)


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_git_rejects_the_patch_when_the_tree_drifted(tmp_path: Path) -> None:
    """Sanity check of the harness: the patch really is anchored to the base content."""
    base = MemoryView({"a.txt": "one\ntwo\n"})
    cs = compute_ops(base, {"a.txt": "one\nTWO\n"})
    patch = tmp_path / "p.patch"
    patch.write_bytes(to_unified_diff(base, cs).encode("utf-8"))
    tree = tmp_path / "tree"
    materialize(MemoryView({"a.txt": "completely different\n"}), tree)
    res = subprocess.run(
        [str(GIT), "apply", "--check", str(patch)], cwd=tree, capture_output=True, text=True
    )
    assert res.returncode != 0
