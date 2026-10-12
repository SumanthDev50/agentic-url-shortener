"""Tests for ``orchestrator.tools.workspace``: views, baseline snapshot, materialisation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from orchestrator.core.errors import ConfigError, IntegrityError, PathViolation
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import ChangeSet
from orchestrator.tools.changeset import (
    ChangeSetConflict,
    make_create,
    make_delete,
    make_modify,
)
from orchestrator.tools.workspace import (
    DEFAULT_EXCLUDES,
    BaselineSnapshot,
    DirectoryView,
    MemoryView,
    OverlayView,
    StagedView,
    materialize,
)

# ------------------------------ helpers


def write(root: Path, rel: str, data: str | bytes) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return target


def make_dir_link(link: Path, target: Path) -> bool:
    """Directory symlink, or a Windows junction; ``False`` when this platform / user cannot."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
        )
        return done.returncode == 0 and link.exists()
    return False


def make_file_link(link: Path, target: Path) -> bool:
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False


def tree(view: DirectoryView | MemoryView | StagedView) -> dict[str, str]:
    return {p: view.read_text(p) for p in view.list_files("**")}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "src_tree"
    write(root, "app/main.py", "print('hi')\r\n")
    write(root, "README.md", "# demo\n")
    return root


# ------------------------------ MemoryView


def test_memory_view_basics_and_crlf_normalisation() -> None:
    v = MemoryView({"b.txt": "x\r\ny\r\n", "a/c.py": "1"})
    assert v.exists("b.txt") and not v.exists("nope")
    assert v.read_text("b.txt") == "x\ny\n"
    assert v.list_files() == ["a/c.py", "b.txt"]
    with pytest.raises(FileNotFoundError):
        v.read_text("nope")


@pytest.mark.parametrize(
    "bad",
    [
        "../x",
        "/abs/x",
        "C:x",
        "a:b",
        "CON",
        "aux.txt",
        ".git/config",
        "a\\b",
        "a/./b",
        "",
        "x" * 300,
    ],
)
def test_memory_view_rejects_hostile_paths_at_construction(bad: str) -> None:
    with pytest.raises(PathViolation):
        MemoryView({bad: "x"})


def test_list_files_uses_anchored_segment_aware_glob_semantics() -> None:
    v = MemoryView(
        {
            "tests/a.py": "",
            "tests/sub/b.py": "",
            "evil/tests/c.py": "",
            "top.py": "",
            "src/pkg/mod.py": "",
        }
    )
    assert v.list_files("tests/*.py") == ["tests/a.py"]  # never right-anchored, '*' stops at '/'
    assert v.list_files("tests/**") == ["tests/a.py", "tests/sub/b.py"]
    assert v.list_files("**/*.py") == sorted(
        ["tests/a.py", "tests/sub/b.py", "evil/tests/c.py", "top.py", "src/pkg/mod.py"]
    )
    assert v.list_files("*.py") == ["top.py"]


@pytest.mark.parametrize("bad_glob", ["../**", "/abs/**", "a\\b", "a//b"])
def test_list_files_rejects_unsafe_globs(bad_glob: str) -> None:
    with pytest.raises(PathViolation):
        MemoryView({"a.py": ""}).list_files(bad_glob)


# ------------------------------ DirectoryView


def test_directory_view_lists_sorted_posix_paths_and_normalises_text(source: Path) -> None:
    write(source, "bom.txt", b"\xef\xbb\xbfhello\r\n")
    v = DirectoryView(source)
    assert v.list_files() == ["README.md", "app/main.py", "bom.txt"]
    assert v.read_text("app/main.py") == "print('hi')\n"  # CRLF -> LF
    assert v.read_text("bom.txt") == "hello\n"  # BOM stripped
    assert v.exists("README.md") and not v.exists("app")
    with pytest.raises(FileNotFoundError):
        v.read_text("missing.txt")
    assert v.report.skipped_binary == [] and v.report.skipped_links == []


def test_directory_view_skips_and_reports_binary_oversize_and_invalid_paths(tmp_path: Path) -> None:
    root = tmp_path / "t"
    write(root, "ok.txt", "fine\n")
    write(root, "blob.bin", b"\xff\xfe\x00\x80binary")
    write(root, "nul_inside.txt", b"abc\x00def\n")  # valid UTF-8 but not editable text
    write(root, "big.txt", "x" * 50)
    write(root, "evil\u202etxt.py", "spoofed extension (bidi override)\n")
    v = DirectoryView(root, max_file_bytes=40)
    assert v.list_files() == ["ok.txt"]
    assert sorted(v.report.skipped_binary) == ["blob.bin", "nul_inside.txt"]
    assert v.report.skipped_oversize == ["big.txt"]
    assert v.report.skipped_invalid_path == ["evil\u202etxt.py"]
    assert not v.exists("blob.bin")


def test_directory_view_enforces_file_count_and_byte_budgets(tmp_path: Path) -> None:
    root = tmp_path / "t"
    for i in range(3):
        write(root, f"f{i}.txt", "x" * 10)
    with pytest.raises(ConfigError, match="too large"):
        DirectoryView(root, max_files=2)
    with pytest.raises(ConfigError, match="too large"):
        DirectoryView(root, max_total_bytes=25)
    assert len(DirectoryView(root, max_files=3, max_total_bytes=30).list_files()) == 3


def test_directory_view_include_and_exclude_globs(source: Path) -> None:
    write(source, "docs/guide.md", "g\n")
    write(source, "docs/internal/secret_plan.md", "s\n")
    v = DirectoryView(source, include=("docs/**", "README.md"), exclude=("docs/internal/**",))
    assert v.list_files() == ["README.md", "docs/guide.md"]


def test_directory_view_never_follows_directory_links(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    write(outside, "secret.txt", "do not leak\n")
    root = tmp_path / "t"
    write(root, "ok.txt", "ok\n")
    if not make_dir_link(root / "linked", outside):
        pytest.skip("cannot create directory links (symlink / junction) here")
    v = DirectoryView(root)
    assert v.list_files() == ["ok.txt"]
    assert v.report.skipped_links == ["linked"]
    assert not v.exists("linked/secret.txt")
    with pytest.raises(FileNotFoundError):
        v.read_text("linked/secret.txt")


def test_directory_view_never_follows_file_links(tmp_path: Path) -> None:
    outside = write(tmp_path, "outside_secret.txt", "do not leak\n")
    root = tmp_path / "t"
    write(root, "ok.txt", "ok\n")
    if not make_file_link(root / "alias.txt", outside):
        pytest.skip("cannot create file symlinks here")
    v = DirectoryView(root)
    assert v.list_files() == ["ok.txt"]
    assert v.report.skipped_links == ["alias.txt"]


def test_default_excludes_apply_at_every_depth(tmp_path: Path) -> None:
    root = tmp_path / "t"
    keep = ["src/app.py", "README.md", "deploy/run.sh", ".envrc.example"]
    drop = [
        ".git/config",
        ".git/HEAD",
        "sub/.git/config",
        ".env",
        ".env.local",
        "deploy/.env",  # regression: nested env files used to be snapshotted
        "deploy/.env.production",
        "certs/server.pem",
        "id.key",
        "data.db",
        "var/app.sqlite3",
        "__pycache__/x.pyc",
        "pkg/__pycache__/m.cpython-313.pyc",
        ".venv/lib/site.py",
        "node_modules/dep/index.js",
        ".pytest_cache/v/cache",
        ".mypy_cache/x.json",
        "demo.egg-info/PKG-INFO",
        ".agentic/runs/r1/state.json",
    ]
    for rel in (*keep, *drop):
        write(root, rel, "x\n")
    assert DirectoryView(root).list_files() == sorted(keep)
    assert ".env" in "".join(DEFAULT_EXCLUDES)


# ------------------------------ StagedView / OverlayView


def test_staged_view_overlays_writes_and_deletions() -> None:
    base = MemoryView({"keep.txt": "k\n", "edit.txt": "old\n", "gone.txt": "g\n"})
    staged = StagedView(base, {"edit.txt": "new\n", "gone.txt": None, "added/new.py": "n\n"})
    assert tree(staged) == {"keep.txt": "k\n", "edit.txt": "new\n", "added/new.py": "n\n"}
    assert not staged.exists("gone.txt") and staged.exists("added/new.py")
    with pytest.raises(FileNotFoundError):
        staged.read_text("gone.txt")
    assert staged.list_files("added/**") == ["added/new.py"]
    assert base.read_text("edit.txt") == "old\n"  # the base is never touched


def test_staged_view_copies_the_initial_writes_but_sees_later_ones() -> None:
    writes: dict[str, str | None] = {"a.txt": "1"}
    staged = StagedView(MemoryView(), writes)
    writes["a.txt"] = "changed behind its back"
    assert staged.read_text("a.txt") == "1"
    staged.writes["b.txt"] = "2"
    assert staged.exists("b.txt")


def test_overlay_view_applies_changesets_in_order_each_relative_to_the_previous() -> None:
    base = MemoryView({"a.txt": "one\n"})
    first = ChangeSet((make_modify(base, "a.txt", "two\n"), make_create("b.txt", "b\n")))
    mid = OverlayView(base, [first])
    second = ChangeSet((make_modify(mid, "a.txt", "three\n"), make_delete(mid, "b.txt")))
    overlay = OverlayView(base, [first, second])
    assert tree(overlay) == {"a.txt": "three\n"}
    assert tree(base) == {"a.txt": "one\n"}


def test_overlay_view_strict_mode_refuses_conflicting_or_stale_changesets() -> None:
    base = MemoryView({"a.txt": "one\n"})
    a = ChangeSet((make_modify(base, "a.txt", "A\n"),))
    b = ChangeSet((make_modify(base, "a.txt", "B\n"),))  # made against the same base: stale after a
    with pytest.raises(ChangeSetConflict, match="hash mismatch"):
        OverlayView(base, [a, b])
    lenient = OverlayView(base, [a, b], strict=False)
    assert lenient.read_text("a.txt") == "B\n"


# ------------------------------ BaselineSnapshot


def populated_source(root: Path) -> None:
    write(root, "src/app.py", "print(1)\r\n")
    write(root, "README.md", "# demo\n")
    write(root, ".env", "FAKE_SETTING=1\n")
    write(root, "deploy/.env.production", "FAKE_SETTING=2\n")
    write(root, ".git/config", "[core]\n")
    write(root, "certs/server.pem", "pem\n")
    write(root, "local.db", "db\n")
    write(root, "pkg/__pycache__/m.pyc", "pyc\n")
    write(root, "bin/blob.bin", b"\xff\xfe\x00\x80")


def test_snapshot_create_copies_only_safe_text_and_records_what_was_skipped(tmp_path: Path) -> None:
    src = tmp_path / "src"
    populated_source(src)
    snap = BaselineSnapshot.create(src, tmp_path / "base")
    assert sorted(snap.manifest) == ["README.md", "src/app.py"]
    assert snap.manifest["src/app.py"] == sha256_text("print(1)\n")  # CRLF-insensitive hash
    assert (
        tmp_path / "base" / "files" / "src" / "app.py"
    ).read_bytes() == b"print(1)\n"  # LF on disk
    on_disk = sorted(
        p.relative_to(tmp_path / "base" / "files").as_posix()
        for p in (tmp_path / "base" / "files").rglob("*")
        if p.is_file()
    )
    assert on_disk == ["README.md", "src/app.py"]
    meta = json.loads((tmp_path / "base" / "manifest.json").read_text(encoding="utf-8"))
    assert meta["skipped"]["binary"] == ["bin/blob.bin"]
    assert snap.file_count == 2 and not snap.is_empty
    assert snap.source_root == src.resolve().as_posix()
    assert snap.verify() == []


def test_snapshot_never_contains_secret_bearing_or_vcs_files(tmp_path: Path) -> None:
    src = tmp_path / "src"
    populated_source(src)
    snap = BaselineSnapshot.create(src, tmp_path / "base")
    files_dir = tmp_path / "base" / "files"
    for secret_like in (
        ".env",
        "deploy/.env.production",
        ".git/config",
        "certs/server.pem",
        "local.db",
        "pkg/__pycache__/m.pyc",
    ):
        assert secret_like not in snap.manifest
        assert not (files_dir / secret_like).exists()
        assert not snap.view.exists(secret_like)


def test_snapshot_skips_and_reports_files_above_the_size_limit(tmp_path: Path) -> None:
    src = tmp_path / "src"
    write(src, "ok.txt", "ok\n")
    write(src, "huge.txt", b"a" * (1024 * 1024 + 1))
    snap = BaselineSnapshot.create(src, tmp_path / "base")
    assert list(snap.manifest) == ["ok.txt"]
    meta = json.loads((tmp_path / "base" / "manifest.json").read_text(encoding="utf-8"))
    assert meta["skipped"]["oversize"] == ["huge.txt"]


def test_snapshot_exclude_and_include_parameters_add_to_the_defaults(tmp_path: Path) -> None:
    src = tmp_path / "src"
    populated_source(src)
    write(src, "runs/r1/audit.jsonl", "{}\n")
    snap = BaselineSnapshot.create(src, tmp_path / "base", exclude=("runs/**",))
    assert "runs/r1/audit.jsonl" not in snap.manifest
    only_src = BaselineSnapshot.create(src, tmp_path / "base2", include=("src/**",))
    assert list(only_src.manifest) == ["src/app.py"]
    # an explicit include can never resurrect a default exclude
    forced = BaselineSnapshot.create(src, tmp_path / "base3", include=("**/.env*", "src/**"))
    assert list(forced.manifest) == ["src/app.py"]


def test_snapshot_manifest_hash_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    src = tmp_path / "src"
    populated_source(src)
    a = BaselineSnapshot.create(src, tmp_path / "a")
    b = BaselineSnapshot.create(src, tmp_path / "b")
    assert a.manifest_hash == b.manifest_hash == hash_jsonable(a.manifest)
    write(src, "README.md", "# changed\n")
    assert BaselineSnapshot.create(src, tmp_path / "c").manifest_hash != a.manifest_hash
    # line endings in the source tree do not change the baseline identity
    write(src, "README.md", "# demo\r\n")
    assert BaselineSnapshot.create(src, tmp_path / "d").manifest_hash == a.manifest_hash


def test_snapshot_never_follows_links(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    write(outside, "secret.txt", "do not leak\n")
    src = tmp_path / "src"
    write(src, "ok.txt", "ok\n")
    if not make_dir_link(src / "linked", outside):
        pytest.skip("cannot create directory links (symlink / junction) here")
    snap = BaselineSnapshot.create(src, tmp_path / "base")
    assert list(snap.manifest) == ["ok.txt"]
    assert not list((tmp_path / "base" / "files").rglob("secret.txt"))
    meta = json.loads((tmp_path / "base" / "manifest.json").read_text(encoding="utf-8"))
    assert meta["skipped"]["links"] == ["linked"]


def test_snapshot_respects_the_file_count_bound(tmp_path: Path) -> None:
    src = tmp_path / "src"
    for i in range(3):
        write(src, f"f{i}.txt", "x\n")
    with pytest.raises(ConfigError, match="too large"):
        BaselineSnapshot.create(src, tmp_path / "base", max_files=2)


def test_empty_snapshot(tmp_path: Path) -> None:
    snap = BaselineSnapshot.empty(tmp_path / "base")
    assert snap.is_empty and snap.manifest == {} and snap.file_count == 0
    assert snap.source_root is None
    assert snap.view.list_files() == []
    assert snap.verify() == []
    assert snap.manifest_hash == hash_jsonable({})
    assert BaselineSnapshot.load(tmp_path / "base").is_empty


def test_load_round_trips_the_manifest(tmp_path: Path) -> None:
    src = tmp_path / "src"
    populated_source(src)
    created = BaselineSnapshot.create(src, tmp_path / "base")
    loaded = BaselineSnapshot.load(tmp_path / "base")
    assert loaded.manifest == created.manifest
    assert loaded.manifest_hash == created.manifest_hash
    assert loaded.verify() == []
    assert loaded.view.read_text("src/app.py") == "print(1)\n"


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        '{"files": []}',
        '{"files": {"../escape.txt": "' + "a" * 64 + '"}}',
        '{"files": {"/abs.txt": "' + "a" * 64 + '"}}',
        '{"files": {"ok.txt": "not-a-hash"}}',
        '{"files": {"ok.txt": 5}}',
        '{"files": {"a": "' + "a" * 64 + '", "a": "' + "b" * 64 + '"}}',
        "{not json",
    ],
)
def test_load_rejects_a_corrupt_or_hostile_manifest(tmp_path: Path, payload: str) -> None:
    """Regression: a manifest that is no object / lists unsafe paths used to crash later or pass."""
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "manifest.json").write_text(payload, encoding="utf-8")
    with pytest.raises((IntegrityError, ValueError)):
        BaselineSnapshot.load(tmp_path / "base")


def test_load_of_a_missing_manifest_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BaselineSnapshot.load(tmp_path / "nothing")


@pytest.fixture
def snapshot(tmp_path: Path) -> Iterator[BaselineSnapshot]:
    src = tmp_path / "src"
    populated_source(src)
    BaselineSnapshot.create(src, tmp_path / "base")
    yield BaselineSnapshot.load(tmp_path / "base")


def test_verify_detects_modified_content(snapshot: BaselineSnapshot) -> None:
    write(snapshot.dest / "files", "README.md", "# tampered\n")
    assert snapshot.verify() == ["hash mismatch: README.md"]


def test_verify_detects_a_deleted_file(snapshot: BaselineSnapshot) -> None:
    (snapshot.dest / "files" / "src" / "app.py").unlink()
    assert snapshot.verify() == ["missing: src/app.py"]


def test_verify_detects_an_extra_file_even_after_the_view_was_built(
    snapshot: BaselineSnapshot,
) -> None:
    """Regression: ``verify`` used the cached view, so files added later went unnoticed."""
    assert "README.md" in snapshot.view.list_files()  # builds and caches the index
    write(snapshot.dest / "files", "planted/backdoor.py", "x = 1\n")
    assert snapshot.verify() == ["unexpected file: planted/backdoor.py"]


def test_verify_treats_a_file_replaced_by_binary_as_missing_not_as_a_crash(
    snapshot: BaselineSnapshot,
) -> None:
    write(snapshot.dest / "files", "README.md", b"\xff\xfe\x00\x80")
    assert snapshot.verify() == ["missing: README.md"]


def test_verify_reports_every_problem(snapshot: BaselineSnapshot) -> None:
    write(snapshot.dest / "files", "README.md", "# tampered\n")
    (snapshot.dest / "files" / "src" / "app.py").unlink()
    write(snapshot.dest / "files", "extra.txt", "e\n")
    assert sorted(snapshot.verify()) == [
        "hash mismatch: README.md",
        "missing: src/app.py",
        "unexpected file: extra.txt",
    ]


# ------------------------------ materialize


def test_materialize_writes_the_whole_view_with_lf_newlines(tmp_path: Path) -> None:
    view = MemoryView({"a/b/c.txt": "1\n2\n", "top.txt": "t"})
    assert materialize(view, tmp_path / "out") == 2
    assert (tmp_path / "out" / "a" / "b" / "c.txt").read_bytes() == b"1\n2\n"
    assert (tmp_path / "out" / "top.txt").read_bytes() == b"t"


def test_materialize_a_subset_and_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write(out, "a.txt", "old\n")
    view = MemoryView({"a.txt": "new\n", "b.txt": "b\n"})
    assert materialize(view, out, ["a.txt"]) == 1
    assert (out / "a.txt").read_text(encoding="utf-8") == "new\n"
    assert not (out / "b.txt").exists()
    with pytest.raises(FileNotFoundError):
        materialize(view, out, ["ghost.txt"])


class _AnyPathView:
    """A hostile view that returns content for *any* path, including traversal attempts."""

    def exists(self, path: str) -> bool:
        return True

    def read_text(self, path: str) -> str:
        return "pwned\n"

    def list_files(self, glob: str = "**") -> list[str]:
        return []


@pytest.mark.parametrize(
    "bad",
    [
        "../evil.txt",
        "a/../../evil.txt",
        "/abs/evil.txt",
        "C:evil.txt",
        "a:b",
        "CON",
        "nul.txt",
        ".git/config",
        "a\\..\\evil.txt",
    ],
)
def test_materialize_refuses_hostile_paths_and_writes_nothing_outside(
    tmp_path: Path, bad: str
) -> None:
    out = tmp_path / "scratch" / "out"
    with pytest.raises(PathViolation):
        materialize(_AnyPathView(), out, [bad])
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "scratch" / "evil.txt").exists()
    assert list(out.rglob("*")) == []


def test_materialize_refuses_to_write_through_a_link(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    if not make_dir_link(out / "sub", outside):
        pytest.skip("cannot create directory links (symlink / junction) here")
    with pytest.raises(PathViolation, match="link"):
        materialize(MemoryView({"sub/x.txt": "x\n"}), out)
    assert list(outside.iterdir()) == []


def test_materialize_is_bounded(tmp_path: Path) -> None:
    view = MemoryView({f"f{i}.txt": "x" * 10 for i in range(5)})
    with pytest.raises(ConfigError, match="too many files"):
        materialize(view, tmp_path / "a", max_files=4)
    with pytest.raises(ConfigError, match="byte budget"):
        materialize(view, tmp_path / "b", max_total_bytes=30)
    assert materialize(view, tmp_path / "c", max_files=5, max_total_bytes=50) == 5


def test_materialize_of_staged_and_overlay_views_round_trips(tmp_path: Path) -> None:
    base = MemoryView({"a.txt": "one\n", "gone.txt": "bye\n"})
    staged = StagedView(base, {"a.txt": "two\n", "gone.txt": None, "n/new.txt": "n\n"})
    materialize(staged, tmp_path / "out")
    assert tree(DirectoryView(tmp_path / "out", exclude=())) == {
        "a.txt": "two\n",
        "n/new.txt": "n\n",
    }
