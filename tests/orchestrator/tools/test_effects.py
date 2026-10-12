"""Tests for ``orchestrator.tools.effects``: all-or-nothing apply, drift detection, safe revert."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.audit import AuditLog
from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import AgentCancelled, ConfigError, PathViolation
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.types import (
    Actor,
    ApplyReceipt,
    CancelToken,
    ChangeSet,
    EffectPort,
    EventType,
    FileOp,
    FileOpKind,
)
from orchestrator.tools import effects
from orchestrator.tools.changeset import (
    ChangeSetConflict,
    apply_changeset,
    compute_ops,
    digest,
    make_create,
    make_delete,
    make_modify,
)
from orchestrator.tools.effects import (
    DirectoryEffectPort,
    ProtectedPathError,
    RevertRefused,
    RollbackIncompleteError,
    UnsafeTargetError,
    allow_any_worktree,
    check_git_worktree,
)
from orchestrator.tools.workspace import DirectoryView, MemoryView

PROTECTED = (".github/**", "Dockerfile", "requirements*.txt", ".env*", "pyproject.toml")

TREE = {
    "app/main.py": "print('v1')\n",
    "app/util.py": "def f():\n    return 1\n",
    "app/sub/deep.py": "deep = 1\n",
    "README.md": "# demo\n",
    ".github/workflows/ci.yml": "name: ci\n",
    "Dockerfile": "FROM scratch\n",
}

GIT = shutil.which("git")


# ------------------------------ helpers


@dataclass
class Emitted:
    type: EventType
    node_id: str | None
    attempt: int | None
    actor: Actor
    data: dict[str, Any]
    files_at_emit: dict[str, bytes]


class Journal:
    """An ``AuditSink`` that also snapshots the target tree at the moment of every event."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.events: list[Emitted] = []
        self.on_event: Callable[[EventType], None] | None = None

    def __call__(
        self,
        type: EventType,
        *,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: Mapping[str, Any],
    ) -> None:
        self.events.append(
            Emitted(type, node_id, attempt, actor, dict(data), tree_bytes(self.root))
        )
        if self.on_event is not None:
            self.on_event(type)

    def types(self) -> list[EventType]:
        return [e.type for e in self.events]


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def write_tree(root: Path, files: Mapping[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))


@pytest.fixture
def target(tmp_path: Path) -> Path:
    root = tmp_path / "target"
    root.mkdir()
    write_tree(root, TREE)
    return root


def make_port(
    root: Path,
    *,
    git_guard: Callable[[Path], None] = allow_any_worktree,
    protected: tuple[str, ...] = PROTECTED,
    cancel: CancelToken | None = None,
    audit: Journal | None = None,
) -> tuple[DirectoryEffectPort, Journal]:
    journal = audit or Journal(root)
    port = DirectoryEffectPort(
        root,
        protected_globs=protected,
        audit=journal,
        git_guard=git_guard,
        node_id="promote",
        attempt=1,
        cancel=cancel,
    )
    return port, journal


def view_of(root: Path) -> DirectoryView:
    return DirectoryView(root, exclude=())


def standard_changeset(root: Path) -> ChangeSet:
    """CREATE + MODIFY + DELETE (+ a delete that empties a directory), built against ``root``."""
    view = view_of(root)
    return ChangeSet(
        (
            make_create("app/new/feature.py", "feature = True\n"),
            make_modify(view, "app/main.py", "print('v2')\n"),
            make_delete(view, "app/util.py"),
            make_delete(view, "app/sub/deep.py"),
        ),
        "release patch",
        "agent-1",
    )


def fail_nth_write(monkeypatch: pytest.MonkeyPatch, nth: int) -> None:
    real = effects.atomic_write_bytes
    calls = {"n": 0}

    def flaky(path: Path | str, data: bytes, *, mode: int = 0o600) -> None:
        calls["n"] += 1
        if calls["n"] == nth:
            raise OSError("disk full (injected)")
        real(path, data, mode=mode)

    monkeypatch.setattr(effects, "atomic_write_bytes", flaky)


def make_dir_link(link: Path, target: Path) -> bool:
    from tests.orchestrator.tools.test_workspace import make_dir_link as _make

    return _make(link, target)


# ------------------------------ apply + revert


def test_apply_changes_the_tree_and_returns_a_proof(target: Path) -> None:
    port, _ = make_port(target)
    cs = standard_changeset(target)
    receipt = port.apply(cs)
    assert tree_bytes(target) == {
        **{k: v.encode() for k, v in TREE.items() if k not in ("app/util.py", "app/sub/deep.py")},
        "app/main.py": b"print('v2')\n",
        "app/new/feature.py": b"feature = True\n",
    }
    assert not (target / "app" / "sub").exists()  # the emptied directory is pruned
    assert target.is_dir() and (target / "app").is_dir()
    structural: EffectPort = (
        port  # (statically checked) DirectoryEffectPort implements the protocol
    )
    assert structural is port
    assert receipt.target_id == port.target_id and receipt.target_id.startswith("dir:")
    assert str(target) not in receipt.target_id  # no absolute path leaks into persisted receipts
    assert receipt.applied == cs.paths()
    assert receipt.pre_hashes == {
        "app/new/feature.py": None,
        "app/main.py": sha256_text(TREE["app/main.py"]),
        "app/util.py": sha256_text(TREE["app/util.py"]),
        "app/sub/deep.py": sha256_text(TREE["app/sub/deep.py"]),
    }
    assert receipt.post_hashes == {
        "app/new/feature.py": sha256_text("feature = True\n"),
        "app/main.py": sha256_text("print('v2')\n"),
        "app/util.py": None,
        "app/sub/deep.py": None,
    }
    # the reverse change set is exactly the inverse, applicable to the tree *after* the effect
    assert apply_changeset(view_of(target), receipt.reverse)["app/util.py"] == TREE["app/util.py"]


def test_revert_restores_the_original_tree_exactly(target: Path) -> None:
    original = tree_bytes(target)
    port, _ = make_port(target)
    receipt = port.apply(standard_changeset(target))
    assert tree_bytes(target) != original
    port.revert(receipt)
    assert tree_bytes(target) == original
    assert (target / "app" / "sub").is_dir()  # recreated with its file
    assert not (target / "app" / "new").exists()  # the created file's directory is gone again


def test_a_receipt_survives_json_persistence_and_can_be_reverted_by_a_new_port(
    target: Path,
) -> None:
    original = tree_bytes(target)
    port, _ = make_port(target)
    receipt = port.apply(standard_changeset(target))
    revived = ApplyReceipt.from_dict(json.loads(json.dumps(receipt.to_dict())))
    assert revived == receipt
    fresh, _ = make_port(
        target
    )  # e.g. a compensator in a later process: same root => same target id
    assert fresh.target_id == port.target_id
    fresh.revert(revived)
    assert tree_bytes(target) == original


def test_empty_changeset_is_a_no_op_without_events(target: Path) -> None:
    port, journal = make_port(target)
    receipt = port.apply(ChangeSet())
    assert receipt.applied == () and receipt.pre_hashes == {} and receipt.post_hashes == {}
    assert journal.events == []


def test_dry_run_verifies_everything_but_writes_and_audits_nothing(target: Path) -> None:
    original = tree_bytes(target)
    port, journal = make_port(target)
    cs = standard_changeset(target)
    preview = port.apply(cs, dry_run=True)
    assert tree_bytes(target) == original and journal.events == []
    assert port.apply(cs) == preview  # the real apply produces the very receipt that was previewed


def test_dry_run_still_detects_drift_protected_paths_and_a_refusing_guard(target: Path) -> None:
    port, _ = make_port(target)
    stale = ChangeSet((make_modify(view_of(target), "README.md", "# new\n"),))
    (target / "README.md").write_text("# edited by a human\n", encoding="utf-8")
    with pytest.raises(ChangeSetConflict):
        port.apply(stale, dry_run=True)
    with pytest.raises(ProtectedPathError):
        port.apply(ChangeSet((make_create("pyproject.toml", "[x]\n"),)), dry_run=True)

    def refuse(root: Path) -> None:
        raise UnsafeTargetError("dirty")

    guarded, _ = make_port(target, git_guard=refuse)
    with pytest.raises(UnsafeTargetError):
        guarded.apply(ChangeSet((make_create("ok.txt", "x"),)), dry_run=True)


def test_file_mode_and_target_newlines(target: Path) -> None:
    crlf = target / "windows.txt"
    crlf.write_bytes(b"one\r\ntwo\r\n")
    port, _ = make_port(target)
    cs = ChangeSet((make_modify(view_of(target), "windows.txt", "one\ntwo\nthree\n"),))
    port.apply(cs)
    assert crlf.read_bytes() == b"one\ntwo\nthree\n"  # rewritten with LF (documented)
    if sys.platform != "win32":
        script = target / "run.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        script.chmod(0o755)
        port.apply(ChangeSet((make_modify(view_of(target), "run.sh", "#!/bin/sh\necho hi\n"),)))
        assert script.stat().st_mode & 0o777 == 0o755  # the mode survives a rewrite


# ------------------------------ audit trail


def test_started_is_audited_before_the_first_write_and_completed_after_the_last(
    target: Path,
) -> None:
    original = tree_bytes(target)
    port, journal = make_port(target)
    cs = standard_changeset(target)
    port.apply(cs)
    started, completed = journal.events
    assert (started.type, completed.type) == (EventType.EFFECT_STARTED, EventType.EFFECT_COMPLETED)
    assert started.files_at_emit == original  # nothing touched yet
    assert completed.files_at_emit == tree_bytes(target)  # everything applied
    assert started.data["action"] == "apply" and started.data["paths"] == sorted(cs.paths())
    assert started.data["changeset_digest"] == digest(cs)
    assert started.data["target_id"] == port.target_id
    assert completed.data["ok"] is True and completed.data["files"] == 4
    assert (started.node_id, started.attempt, started.actor.label) == (
        "promote",
        1,
        "system:effects",
    )


def test_revert_is_audited_the_same_way(target: Path) -> None:
    port, journal = make_port(target)
    receipt = port.apply(standard_changeset(target))
    journal.events.clear()
    after_apply = tree_bytes(target)
    port.revert(receipt)
    started, completed = journal.events
    assert started.data["action"] == completed.data["action"] == "revert"
    assert started.files_at_emit == after_apply and completed.files_at_emit != after_apply
    assert completed.data["ok"] is True


def test_every_started_has_a_completed_even_when_the_effect_fails(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port, journal = make_port(target)
    fail_nth_write(monkeypatch, 2)
    with pytest.raises(OSError, match="injected"):
        port.apply(standard_changeset(target))
    assert journal.types() == [EventType.EFFECT_STARTED, EventType.EFFECT_COMPLETED]
    failed = journal.events[1].data
    assert failed["ok"] is False and failed["rolled_back"] is True and failed["error"] == "OSError"


def test_events_chain_in_the_real_audit_log(tmp_path: Path, target: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", run_id="run-1", clock=FakeClock())
    port = DirectoryEffectPort(
        target,
        protected_globs=PROTECTED,
        audit=log,
        git_guard=allow_any_worktree,
        node_id="promote",
        attempt=1,
    )
    port.revert(port.apply(standard_changeset(target)))
    assert log.verify().ok
    assert [e.type for e in log.events()] == [
        EventType.EFFECT_STARTED,
        EventType.EFFECT_COMPLETED,
        EventType.EFFECT_STARTED,
        EventType.EFFECT_COMPLETED,
    ]


# ------------------------------ drift detection


def test_drift_is_detected_before_anything_is_touched(target: Path) -> None:
    port, journal = make_port(target)
    cs = standard_changeset(target)
    (target / "app" / "util.py").write_text("def f():\n    return 'human edit'\n", encoding="utf-8")
    before = tree_bytes(target)
    with pytest.raises(ChangeSetConflict, match="hash mismatch") as excinfo:
        port.apply(cs)
    assert excinfo.value.path == "app/util.py"
    assert (
        tree_bytes(target) == before
    )  # even the ops that *would* have applied cleanly did not run
    assert journal.events == []


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda t: ChangeSet((make_create("README.md", "x"),)), "already exists"),
        (
            lambda t: ChangeSet((FileOp("ghost.py", FileOpKind.MODIFY, "x", sha256_text("y")),)),
            "does not exist",
        ),
        (
            lambda t: ChangeSet((FileOp("ghost.py", FileOpKind.DELETE, None, sha256_text("y")),)),
            "does not exist",
        ),
        (lambda t: ChangeSet((make_create("app", "x"),)), "not a regular file"),
    ],
)
def test_preconditions_are_verified_against_the_target_tree(
    target: Path, build: Callable[[Path], ChangeSet], message: str
) -> None:
    port, _ = make_port(target)
    before = tree_bytes(target)
    with pytest.raises(ChangeSetConflict, match=message):
        port.apply(build(target))
    assert tree_bytes(target) == before


def test_binary_and_oversize_target_files_cannot_be_verified_and_are_refused(target: Path) -> None:
    (target / "blob.dat").write_bytes(b"\xff\xfe\x00binary")
    (target / "huge.txt").write_bytes(b"x" * (1024 * 1024 + 1))
    port, _ = make_port(target)
    for name, message in (("blob.dat", "UTF-8"), ("huge.txt", "too large")):
        cs = ChangeSet((FileOp(name, FileOpKind.MODIFY, "text\n", sha256_text("anything")),))
        with pytest.raises(ChangeSetConflict, match=message):
            port.apply(cs)
    assert (target / "blob.dat").read_bytes() == b"\xff\xfe\x00binary"


def test_a_file_changed_between_verification_and_commit_is_never_clobbered(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check-to-use window is closed again right before each file is replaced."""
    port, journal = make_port(target)
    cs = standard_changeset(target)
    real = effects.atomic_write_bytes
    state = {"done": False}

    def racing(path: Path | str, data: bytes, *, mode: int = 0o600) -> None:
        if not state["done"]:
            state["done"] = True
            (target / "app" / "util.py").write_text("human edit mid-apply\n", encoding="utf-8")
        real(path, data, mode=mode)

    monkeypatch.setattr(effects, "atomic_write_bytes", racing)
    with pytest.raises(ChangeSetConflict, match="changed while the effect"):
        port.apply(cs)
    assert (target / "app" / "util.py").read_text(encoding="utf-8") == "human edit mid-apply\n"
    assert (target / "app" / "main.py").read_bytes() == TREE["app/main.py"].encode()  # rolled back
    assert not (target / "app" / "new").exists()
    assert journal.events[-1].data["ok"] is False


# ------------------------------ all-or-nothing


@pytest.mark.parametrize("failing_write", [1, 2, 3])
def test_a_failure_part_way_rolls_every_applied_file_back(
    target: Path, monkeypatch: pytest.MonkeyPatch, failing_write: int
) -> None:
    view = view_of(target)
    cs = ChangeSet(
        (
            make_create("a_first.py", "first\n"),
            make_modify(view, "app/main.py", "print('v2')\n"),
            make_delete(view, "app/util.py"),
            make_create("z_last.py", "last\n"),
            make_modify(view, "README.md", "# changed\n"),
        )
    )
    original = tree_bytes(target)
    port, journal = make_port(target)
    fail_nth_write(monkeypatch, failing_write)
    with pytest.raises(OSError, match="injected"):
        port.apply(cs)
    assert tree_bytes(target) == original
    assert not (target / "a_first.py").exists() and (target / "app" / "util.py").exists()
    assert journal.events[-1].data["rolled_back"] is True


def test_rollback_restores_the_original_bytes_including_crlf_newlines(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crlf = target / "app" / "windows.py"
    crlf.write_bytes(b"line one\r\nline two\r\n")
    view = view_of(target)
    cs = ChangeSet(
        (
            make_modify(view, "app/windows.py", "changed\n"),
            make_delete(view, "app/util.py"),
            make_create("z_boom.py", "x"),
        )
    )
    port, _ = make_port(target)
    fail_nth_write(monkeypatch, 2)  # the CREATE fails after the MODIFY and the DELETE
    with pytest.raises(OSError, match="injected"):
        port.apply(cs)
    assert crlf.read_bytes() == b"line one\r\nline two\r\n"  # byte-exact, not LF-normalised
    assert (target / "app" / "util.py").read_bytes() == TREE["app/util.py"].encode()


def test_a_rollback_that_cannot_complete_is_reported_loudly(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port, journal = make_port(target)
    fail_nth_write(monkeypatch, 2)

    def broken_restore(*args: object) -> None:
        raise OSError("cannot restore (injected)")

    monkeypatch.setattr(effects, "_restore", broken_restore)
    view = view_of(target)
    cs = ChangeSet((make_modify(view, "app/main.py", "x\n"), make_modify(view, "README.md", "y\n")))
    with pytest.raises(RollbackIncompleteError, match=r"app/main\.py|README\.md"):
        port.apply(cs)
    assert journal.events[-1].data == {
        **journal.events[-1].data,
        "ok": False,
        "rolled_back": False,
    }


def test_cancel_before_the_apply_prevents_any_write(target: Path) -> None:
    token = CancelToken()
    token.cancel("kill switch")
    port, journal = make_port(target, cancel=token)
    original = tree_bytes(target)
    with pytest.raises(AgentCancelled):
        port.apply(standard_changeset(target))
    assert tree_bytes(target) == original and journal.events == []


def test_cancel_between_files_rolls_back(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = CancelToken()
    port, journal = make_port(target, cancel=token)
    real = effects.atomic_write_bytes

    def write_then_cancel(path: Path | str, data: bytes, *, mode: int = 0o600) -> None:
        real(path, data, mode=mode)
        token.cancel("kill switch during apply")

    original = tree_bytes(target)
    monkeypatch.setattr(effects, "atomic_write_bytes", write_then_cancel)
    with pytest.raises(AgentCancelled):
        port.apply(standard_changeset(target))
    monkeypatch.undo()
    assert tree_bytes(target) == original
    assert journal.events[-1].data["ok"] is False and journal.events[-1].data["rolled_back"] is True


# ------------------------------ protected paths and containment


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ci.yml",
        ".github/workflows/new.yml",
        "Dockerfile",
        "requirements.txt",
        "requirements-dev.txt",
        ".env",
        ".env.production",
        "pyproject.toml",
        ".GitHub/workflows/ci.yml",  # case variants hit the same file on Windows / macOS
        ".GITHUB/WORKFLOWS/x.yml",
        "dockerfile",
        "DOCKERFILE",
        "PyProject.TOML",
        "Requirements-Dev.txt",
    ],
)
def test_protected_paths_are_refused_before_anything_is_written(target: Path, path: str) -> None:
    port, journal = make_port(target)
    cs = ChangeSet((make_create("zzz_fine.txt", "ok\n"), make_create(path, "x\n")))
    original = tree_bytes(target)
    with pytest.raises(ProtectedPathError):
        port.apply(cs)
    assert tree_bytes(target) == original  # the harmless op did not apply either
    assert journal.events == []


def test_modifying_or_deleting_an_existing_protected_file_is_refused(target: Path) -> None:
    port, _ = make_port(target)
    view = view_of(target)
    for op in (
        make_modify(view, "Dockerfile", "FROM evil\n"),
        make_delete(view, "Dockerfile"),
        make_modify(view, ".github/workflows/ci.yml", "name: pwned\n"),
    ):
        with pytest.raises(ProtectedPathError):
            port.apply(ChangeSet((op,)))
    assert (target / "Dockerfile").read_text(encoding="utf-8") == TREE["Dockerfile"]


def test_the_git_directory_can_never_be_an_op_path() -> None:
    for bad in (".git/config", ".git/hooks/pre-commit", "sub/.git/config"):
        with pytest.raises(PathViolation):
            FileOp(bad, FileOpKind.CREATE, "x")


def test_a_protected_path_in_a_forged_receipt_cannot_be_reverted(target: Path) -> None:
    port, _ = make_port(target)
    forged = ApplyReceipt(
        port.target_id,
        ("Dockerfile",),
        {"Dockerfile": sha256_text("old")},
        {"Dockerfile": sha256_text(TREE["Dockerfile"])},
        ChangeSet(
            (
                FileOp(
                    "Dockerfile", FileOpKind.MODIFY, "FROM evil\n", sha256_text(TREE["Dockerfile"])
                ),
            )
        ),
    )
    with pytest.raises(ProtectedPathError):
        port.revert(forged)
    assert (target / "Dockerfile").read_text(encoding="utf-8") == TREE["Dockerfile"]


def test_ops_cannot_write_through_a_link(target: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    if not make_dir_link(target / "linked", outside):
        pytest.skip("cannot create directory links (symlink / junction) here")
    port, _ = make_port(target)
    with pytest.raises(PathViolation, match="link"):
        port.apply(ChangeSet((make_create("linked/evil.txt", "pwned\n"),)))
    assert list(outside.iterdir()) == []


def test_root_validation(tmp_path: Path, target: Path) -> None:
    journal = Journal(target)
    with pytest.raises(ConfigError, match="does not exist"):
        DirectoryEffectPort(tmp_path / "missing", protected_globs=(), audit=journal)
    plain = tmp_path / "file.txt"
    plain.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="directory"):
        DirectoryEffectPort(plain, protected_globs=(), audit=journal)
    with pytest.raises(ConfigError, match="file-system root"):
        DirectoryEffectPort(Path(tmp_path.anchor), protected_globs=(), audit=journal)
    with pytest.raises(PathViolation):
        DirectoryEffectPort(target, protected_globs=("../**",), audit=journal)
    link = tmp_path / "rootlink"
    if make_dir_link(link, target):
        with pytest.raises(ConfigError, match="link"):
            DirectoryEffectPort(link, protected_globs=(), audit=journal)


def test_default_git_guard_is_the_real_check() -> None:
    import inspect

    assert (
        inspect.signature(DirectoryEffectPort.__init__).parameters["git_guard"].default
        is check_git_worktree
    )


# ------------------------------ revert semantics


def test_revert_is_idempotent(target: Path) -> None:
    original = tree_bytes(target)
    port, journal = make_port(target)
    receipt = port.apply(standard_changeset(target))
    port.revert(receipt)
    journal.events.clear()
    port.revert(receipt)  # nothing left to do
    port.revert(receipt)
    assert tree_bytes(target) == original
    assert [e.data["noop"] for e in journal.events if e.type == EventType.EFFECT_STARTED] == [
        True,
        True,
    ]
    assert all(e.data["ok"] for e in journal.events if e.type == EventType.EFFECT_COMPLETED)


@pytest.mark.parametrize(
    "what", ["modified", "deleted_modified_file", "recreated", "edited_created_file"]
)
def test_revert_is_refused_after_a_later_human_edit_and_changes_nothing(
    target: Path, what: str
) -> None:
    port, journal = make_port(target)
    receipt = port.apply(standard_changeset(target))
    if what == "modified":  # a human edited a file the effect had modified
        (target / "app" / "main.py").write_text("print('human v3')\n", encoding="utf-8")
    elif what == "deleted_modified_file":  # a human removed a file the effect had modified
        (target / "app" / "main.py").unlink()
    elif what == "recreated":  # a human put something else where the effect had deleted a file
        (target / "app" / "util.py").write_text("human replacement\n", encoding="utf-8")
    else:  # a human extended a file the effect had created
        (target / "app" / "new" / "feature.py").write_text("feature = 'human'\n", encoding="utf-8")
    after_human_edit = tree_bytes(target)
    journal.events.clear()
    with pytest.raises(RevertRefused, match="changed since the effect was applied"):
        port.revert(receipt)
    assert tree_bytes(target) == after_human_edit  # not even the untouched files were reverted
    assert journal.events == []  # refused before the effect started


def test_a_created_file_that_a_human_already_removed_is_simply_already_reverted(
    target: Path,
) -> None:
    """Nothing of anybody's is clobbered: the file is in the state the revert wants anyway."""
    original = tree_bytes(target)
    port, _ = make_port(target)
    receipt = port.apply(standard_changeset(target))
    (target / "app" / "new" / "feature.py").unlink()
    port.revert(receipt)
    assert tree_bytes(target) == original


def test_revert_finishes_a_partially_reverted_tree(target: Path) -> None:
    original = tree_bytes(target)
    port, _ = make_port(target)
    receipt = port.apply(standard_changeset(target))
    (target / "app" / "main.py").write_bytes(
        TREE["app/main.py"].encode()
    )  # already back at pre_hash
    port.revert(receipt)
    assert tree_bytes(target) == original


def test_revert_validates_the_receipt(target: Path, tmp_path: Path) -> None:
    port, _ = make_port(target)
    receipt = port.apply(standard_changeset(target))
    other_root = tmp_path / "other"
    other_root.mkdir()
    other, _ = make_port(other_root)
    with pytest.raises(RevertRefused, match="different target"):
        other.revert(receipt)
    dropped = dict(receipt.pre_hashes)
    dropped.pop("app/main.py")
    inconsistent = ApplyReceipt(
        receipt.target_id, receipt.applied, dropped, receipt.post_hashes, receipt.reverse
    )
    with pytest.raises(RevertRefused, match="inconsistent"):
        port.revert(inconsistent)
    shorter = ApplyReceipt(
        receipt.target_id,
        receipt.applied,
        receipt.pre_hashes,
        receipt.post_hashes,
        ChangeSet(receipt.reverse.ops[:1]),
    )
    with pytest.raises(RevertRefused, match="inconsistent"):
        port.revert(shorter)
    port.revert(receipt)  # the genuine receipt still works


def test_revert_ignores_the_cancel_token_so_compensation_can_always_finish(target: Path) -> None:
    token = CancelToken()
    port, _ = make_port(target, cancel=token)
    original = tree_bytes(target)
    receipt = port.apply(standard_changeset(target))
    token.cancel("run was killed")
    with pytest.raises(AgentCancelled):
        port.apply(ChangeSet((make_create("late.txt", "x"),)))
    port.revert(receipt)  # a rollback is never cancelled
    assert tree_bytes(target) == original


def test_revert_does_not_consult_the_git_guard(target: Path) -> None:
    """The tree is dirty *because of* our own effect: the guard would always refuse the undo."""
    calls: list[Path] = []

    def counting_guard(root: Path) -> None:
        calls.append(root)

    port, _ = make_port(target, git_guard=counting_guard)
    receipt = port.apply(standard_changeset(target))
    port.revert(receipt)
    assert len(calls) == 1 and calls[0] == target.resolve()


def test_revert_of_an_effect_that_was_rolled_back_mid_apply_has_no_receipt_to_use(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port, _ = make_port(target)
    fail_nth_write(monkeypatch, 1)
    with pytest.raises(OSError, match="injected"):
        port.apply(standard_changeset(target))
    assert tree_bytes(target) == {k: v.encode() for k, v in TREE.items()}


# ------------------------------ git guard


def run_git(repo: Path, *args: str) -> None:
    """Run git hermetically: ignore the developer machine's global / system config."""
    empty_config = repo.parent / "empty.gitconfig"
    empty_config.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(empty_config),
        "GIT_CONFIG_SYSTEM": str(empty_config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    done = subprocess.run(
        [str(GIT), "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert done.returncode == 0, done.stderr


@pytest.fixture(scope="module")
def repo_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One committed repository on branch 'feature' (git is slow to start: build it once)."""
    if GIT is None:
        pytest.skip("git is not installed")
    root = tmp_path_factory.mktemp("gitrepo") / "template"
    root.mkdir()
    run_git(root, "init", "-q", "-b", "feature")
    write_tree(root, {"app/main.py": "print('v1')\n", "README.md": "# demo\n"})
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "--no-verify", "-m", "initial")
    return root


@pytest.fixture
def repo(repo_template: Path, tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(repo_template, root)
    return root


@pytest.mark.slow
@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_a_clean_repo_on_a_feature_branch_is_accepted_with_the_real_guard(repo: Path) -> None:
    port, _ = make_port(repo, git_guard=check_git_worktree)
    receipt = port.apply(ChangeSet((make_modify(view_of(repo), "app/main.py", "print('v2')\n"),)))
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == "print('v2')\n"
    port.revert(receipt)  # the guard is not consulted on the way back
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == "print('v1')\n"


@pytest.mark.slow
@pytest.mark.skipif(GIT is None, reason="git is not installed")
@pytest.mark.parametrize("dirt", ["untracked", "modified", "staged"])
def test_a_dirty_working_tree_is_refused(repo: Path, dirt: str) -> None:
    if dirt == "untracked":
        (repo / "scratch.txt").write_text("x", encoding="utf-8")
    elif dirt == "modified":
        (repo / "README.md").write_text("# edited\n", encoding="utf-8")
    else:
        (repo / "README.md").write_text("# staged\n", encoding="utf-8")
        run_git(repo, "add", "README.md")
    port, journal = make_port(repo, git_guard=check_git_worktree)
    before = tree_bytes(repo)
    with pytest.raises(UnsafeTargetError, match="uncommitted"):
        port.apply(ChangeSet((make_create("new.txt", "x"),)))
    assert tree_bytes(repo) == before and journal.events == []


@pytest.mark.slow
@pytest.mark.skipif(GIT is None, reason="git is not installed")
@pytest.mark.parametrize("branch", ["main", "master"])
def test_the_default_branch_is_refused(repo: Path, branch: str) -> None:
    run_git(repo, "branch", "-m", branch)
    port, _ = make_port(repo, git_guard=check_git_worktree)
    with pytest.raises(UnsafeTargetError, match="default branch"):
        port.apply(ChangeSet((make_create("new.txt", "x"),)))


@pytest.mark.slow
@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_a_detached_head_is_refused(repo: Path) -> None:
    run_git(repo, "checkout", "-q", "--detach")
    port, _ = make_port(repo, git_guard=check_git_worktree)
    with pytest.raises(UnsafeTargetError, match="named branch"):
        port.apply(ChangeSet((make_create("new.txt", "x"),)))


@pytest.mark.slow
@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_the_remote_default_branch_name_is_refused_too(repo: Path) -> None:
    """``origin/HEAD`` (read locally, never fetched) names the default branch, whatever it is."""
    run_git(repo, "branch", "-m", "trunk-dev")
    run_git(repo, "update-ref", "refs/remotes/origin/trunk-dev", "HEAD")
    run_git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/trunk-dev")
    with pytest.raises(UnsafeTargetError, match="default branch"):
        check_git_worktree(repo)


def test_a_directory_without_git_needs_no_check(target: Path) -> None:
    check_git_worktree(target)  # no .git: nothing to verify, no subprocess


def test_a_git_directory_without_a_git_binary_is_refused(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (target / ".git").mkdir()
    monkeypatch.setattr(effects, "_find_git", lambda: None)
    with pytest.raises(UnsafeTargetError, match="not available"):
        check_git_worktree(target)


def test_git_failures_fail_closed(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (target / ".git").mkdir()  # not a repository: every git command fails
    if GIT is None:
        pytest.skip("git is not installed")
    with pytest.raises(UnsafeTargetError, match="cannot be verified"):
        check_git_worktree(target)

    def boom(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(effects.subprocess, "run", boom)
    with pytest.raises(UnsafeTargetError, match="cannot verify"):
        check_git_worktree(target)


def test_only_local_read_only_git_commands_are_ever_run(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (target / ".git").mkdir()
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(argv)
        if "refs/remotes/origin/HEAD" in argv:  # no remote default branch recorded
            return subprocess.CompletedProcess(argv, 1, b"", b"")
        out = b"feature\n" if "symbolic-ref" in argv else b""
        return subprocess.CompletedProcess(argv, 0, out, b"")

    monkeypatch.setattr(effects, "_find_git", lambda: "/fake/git")
    monkeypatch.setattr(effects.subprocess, "run", fake_run)
    check_git_worktree(target)
    verbs = {next(a for a in argv[1:] if a in {"status", "symbolic-ref"}) for argv in seen}
    assert verbs == {"status", "symbolic-ref"}
    for argv in seen:
        assert not {"fetch", "pull", "push", "clone", "remote", "ls-remote"} & set(argv)
        assert "core.fsmonitor=false" in argv  # a hostile repo config cannot run a program for us
    env = effects._git_env("/fake/git")
    assert not any("TOKEN" in k or "SECRET" in k or "KEY" in k for k in env)


def test_compute_ops_driven_changeset_round_trip(target: Path) -> None:
    """End to end with the producer-side API: desired state -> change set -> apply -> revert."""
    view = view_of(target)
    desired: dict[str, str | None] = {
        "README.md": "# v2\n",
        "docs/new.md": "new\n",
        "app/util.py": None,
    }
    cs = compute_ops(view, desired, description="x", author="a")
    port, _ = make_port(target)
    original = tree_bytes(target)
    receipt = port.apply(cs)
    assert (target / "docs" / "new.md").read_text(encoding="utf-8") == "new\n"
    port.revert(receipt)
    assert tree_bytes(target) == original
    assert MemoryView({"a": "b"}).exists("a")  # (keeps the MemoryView import honest in this module)
