"""Real side effects: apply a ``ChangeSet`` to a directory tree, and undo it later.

``DirectoryEffectPort`` is the only place the orchestrator mutates a tree outside its own run
directory (``promote --apply-to``).  The rules it enforces, in order:

1. every op path is re-validated (``validate_rel_path``) and matched, case-insensitively, against
   the protected globs (CI, Dockerfile, dependency manifests, env files ...): an agent-authored
   change set cannot reach them even if every earlier control was bypassed;
2. a target containing ``.git`` must be a clean working tree on a named, non-default branch
   (``check_git_worktree``; injectable, ON by default; only local, read-only git commands);
3. ``base_hash`` / non-existence of **every** touched path is verified against the *target tree*
   before anything is written (a tree that drifted since the change set was made is refused);
4. all-or-nothing: each file is written with temp-file + ``os.replace``; an in-memory journal of
   undo steps rolls back the already-applied files if a later op fails, a touched file changed
   under our feet, or the attempt is cancelled;
5. ``EFFECT_STARTED`` is audited before the first mutation and ``EFFECT_COMPLETED`` (``ok`` true
   or false) after, so a STARTED without a COMPLETED in the chain means a crash mid-effect.

``revert`` runs from the persisted ``ApplyReceipt`` only: it refuses (``RevertRefused``, nothing
touched) if any file's current hash is neither the recorded ``post_hash`` nor the original
``pre_hash`` - a later human edit is never clobbered - and is idempotent (files already back at
their ``pre_hash`` are left alone).  Revert deliberately ignores the cancel token and the git
guard: a compensation must be able to run to completion on the tree its own effect dirtied.

Files are text (UTF-8, LF).  A modified file is rewritten with LF newlines and keeps its mode.
A hard crash between two replaces cannot be rolled back in-process; the missing
``EFFECT_COMPLETED`` is the signal for manual reconciliation (documented limitation).
"""

from __future__ import annotations

import contextlib
import functools
import os
import stat
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.core.errors import ConfigError, OrchestratorError
from orchestrator.core.fsutil import (
    atomic_write_bytes,
    is_link_or_reparse,
    normalize_newlines,
    read_bytes_retry,
    safe_join,
    sha256_text,
)
from orchestrator.core.paths import casefold_key, match_any, validate_globs, validate_rel_path
from orchestrator.core.serde import hash_jsonable, sha256_hex
from orchestrator.core.types import (
    MAX_FILE_BYTES,
    Actor,
    ApplyReceipt,
    AuditSink,
    CancelToken,
    ChangeSet,
    EventType,
    FileOp,
    FileOpKind,
)
from orchestrator.tools.changeset import (
    ChangeSetConflict,
    apply_changeset,
    digest,
    reverse_changeset,
)
from orchestrator.tools.workspace import MemoryView

GitGuard = Callable[[Path], None]

_GIT_TIMEOUT_S = 15.0
_DEFAULT_BRANCHES = frozenset({"main", "master", "trunk"})
_NEW_FILE_MODE = 0o644
_MAX_REPORTED_PATHS = 8


class EffectError(OrchestratorError):
    """A real side effect was refused or failed."""


class ProtectedPathError(EffectError):
    """An op touches a protected path (CI, Dockerfile, manifests, env files, ...)."""


class UnsafeTargetError(EffectError):
    """The target tree is not in a state we are willing to modify (dirty / default branch)."""


class RevertRefused(EffectError):
    """Files no longer match the receipt, or the receipt is untrustworthy; nothing was touched."""


class RollbackIncompleteError(EffectError):
    """An apply failed *and* its rollback could not restore every file (manual reconciliation)."""


# ----------------------------------------------------------------------------- git guard


def allow_any_worktree(root: Path) -> None:
    """A ``GitGuard`` that accepts every tree: for tests and deliberate, explicit opt-out only."""


def _find_git() -> str | None:
    """First ``git`` on an *absolute* PATH entry (never the cwd, which Windows searches first)."""
    names = ("git.exe", "git") if os.name == "nt" else ("git",)
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or not os.path.isabs(entry):
            continue
        for name in names:
            candidate = Path(entry) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _git_env(git: str) -> dict[str, str]:
    env = {
        "PATH": str(Path(git).parent),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "LC_ALL": "C",
    }
    for key in ("SYSTEMROOT", "WINDIR", "HOME", "USERPROFILE", "XDG_CONFIG_HOME"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _run_git(git: str, root: Path, *args: str, required: bool = True) -> str | None:
    """One local, read-only git command.

    ``core.fsmonitor=false`` keeps a hostile repository config from executing a program during
    ``status``.  Any failure is fail-closed when ``required``.
    """
    argv = [git, "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(root), *args]
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
            stdin=subprocess.DEVNULL,
            env=_git_env(git),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UnsafeTargetError(f"cannot verify git state ({type(exc).__name__})") from exc
    if done.returncode != 0:
        if required:
            raise UnsafeTargetError(
                f"git {args[0]} failed: the working tree state cannot be verified"
            )
        return None
    return done.stdout.decode("utf-8", "replace")


def check_git_worktree(root: Path) -> None:
    """Refuse a target that contains ``.git`` unless it is clean and on a non-default branch.

    Only ``git status`` / ``symbolic-ref`` run (local, read-only; never fetch/pull/push).  Detached
    HEAD, a missing ``git`` binary or any git failure refuses: this check fails closed.
    """
    if not (root / ".git").exists():
        return
    git = _find_git()
    if git is None:
        raise UnsafeTargetError(
            "target contains .git but git is not available to verify the working tree"
        )
    status = _run_git(git, root, "status", "--porcelain", "--untracked-files=normal")
    if status and status.strip():
        raise UnsafeTargetError("refusing to modify a git working tree with uncommitted changes")
    head = _run_git(git, root, "symbolic-ref", "--quiet", "--short", "HEAD", required=False)
    if head is None or not head.strip():
        raise UnsafeTargetError(
            "refusing to modify a repository that is not on a named branch (detached HEAD)"
        )
    branch = head.strip()
    default = _run_git(
        git, root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD", required=False
    )
    defaults = set(_DEFAULT_BRANCHES)
    if default and default.strip():
        defaults.add(default.strip().partition("/")[2] or default.strip())
    if branch in defaults:
        raise UnsafeTargetError(
            f"refusing to modify the default branch ({branch}); apply on a feature branch"
        )


# ----------------------------------------------------------------------------- plan / journal


@dataclass(frozen=True)
class _Plan:
    """Everything verified about one change set against the target tree, before any write."""

    changeset: ChangeSet
    targets: Mapping[str, Path]
    raw_before: Mapping[str, bytes | None]
    modes: Mapping[str, int]
    texts: Mapping[str, str]  # current (LF-normalised) text of the files that exist
    result: Mapping[str, str | None]  # path -> new content (None = deleted)


def _prune_empty_parents(directory: Path, root: Path) -> None:
    """Remove now-empty directories above a deleted file (never ``root`` itself)."""
    while directory != root and directory.is_relative_to(root):
        try:
            directory.rmdir()
        except OSError:
            return
        directory = directory.parent


def _restore(target: Path, raw: bytes | None, mode: int, root: Path) -> None:
    """Idempotent undo of one step: put back the original bytes, or remove a file we created."""
    if raw is None:
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        _prune_empty_parents(target.parent, root)
    else:
        atomic_write_bytes(target, raw, mode=mode)


def _run_undo(steps: Sequence[tuple[str, Callable[[], None]]]) -> list[str]:
    """Newest first, best effort; returns the paths that could not be restored."""
    stuck: list[str] = []
    for rel, undo in reversed(steps):
        try:
            undo()
        except Exception:
            stuck.append(rel)
    return stuck


def _sample(paths: Sequence[str]) -> str:
    shown = ", ".join(sorted(paths)[:_MAX_REPORTED_PATHS])
    return shown + (
        f" (+{len(paths) - _MAX_REPORTED_PATHS} more)" if len(paths) > _MAX_REPORTED_PATHS else ""
    )


# ----------------------------------------------------------------------------- the port


class DirectoryEffectPort:
    """``types.EffectPort`` over a directory tree (see the module docstring for the guarantees)."""

    def __init__(
        self,
        target_root: Path | str,
        *,
        protected_globs: Sequence[str],
        audit: AuditSink,
        git_guard: GitGuard = check_git_worktree,
        node_id: str | None = None,
        attempt: int | None = None,
        cancel: CancelToken | None = None,
    ) -> None:
        root = Path(target_root)
        if is_link_or_reparse(root):
            raise ConfigError("effect target root must not be a link or reparse point")
        try:
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise ConfigError("effect target root does not exist") from exc
        if not resolved.is_dir() or resolved.parent == resolved:
            raise ConfigError("effect target root must be a directory below the file-system root")
        self._root = resolved
        self._protected = tuple(casefold_key(g) for g in validate_globs(tuple(protected_globs)))
        self._audit = audit
        self._git_guard = git_guard
        self._node_id = node_id
        self._attempt = attempt
        self._cancel = cancel
        self._lock = threading.Lock()  # one mutation of the tree at a time
        self.target_id = f"dir:{sha256_hex(resolved.as_posix())[:16]}"

    # -- public API -------------------------------------------------------------
    def apply(self, cs: ChangeSet, *, dry_run: bool = False) -> ApplyReceipt:
        """Apply ``cs`` to the target tree all-or-nothing and return the proof.

        ``dry_run`` performs every check and returns the receipt that *would* result, writing and
        auditing nothing.  Raises ``ProtectedPathError``, ``UnsafeTargetError``,
        ``ChangeSetConflict`` (drift), ``AgentCancelled`` or ``EffectError``; in every case the
        tree is left as it was (``RollbackIncompleteError`` if even the rollback failed).
        """
        with self._lock:
            self._check_cancel()
            self._git_guard(self._root)
            plan = self._plan(cs)
            receipt = self._receipt(plan)
            if dry_run or not cs.ops:
                return receipt
            self._journalled("apply", cs, receipt, lambda: self._commit(plan, cancellable=True))
            return receipt

    def revert(self, receipt: ApplyReceipt) -> None:
        """Undo ``receipt`` iff the files still hold what the effect left there (idempotent)."""
        with self._lock:
            pending = self._pending_reverts(receipt)
            ops = tuple(op for op in reversed(receipt.reverse.ops) if op.path in pending)
            reverse = ChangeSet(ops, f"revert of {receipt.target_id}", receipt.reverse.author)
            if not ops:
                self._journalled("revert", reverse, receipt, lambda: None, noop=True)
                return
            plan = self._plan(reverse)
            self._journalled(
                "revert", reverse, receipt, lambda: self._commit(plan, cancellable=False)
            )

    # -- verification -----------------------------------------------------------
    def _check_cancel(self) -> None:
        if self._cancel is not None:
            self._cancel.raise_if_cancelled()

    def _check_protected(self, rel: str) -> None:
        if match_any(casefold_key(rel), self._protected):
            raise ProtectedPathError(f"path is protected and cannot be changed by an effect: {rel}")

    def _resolve(self, rel: str) -> Path:
        """Contained, link-free target path; protected globs are also matched against the *real*
        name (case / 8.3 aliases of an existing file)."""
        target = safe_join(self._root, validate_rel_path(rel))
        real = target.resolve().relative_to(self._root).as_posix()
        if real != rel:
            self._check_protected(real)
        return target

    @staticmethod
    def _read_raw(rel: str, target: Path) -> tuple[bytes | None, int]:
        try:
            info = os.lstat(target)
        except FileNotFoundError:
            return None, _NEW_FILE_MODE
        except OSError as exc:
            raise ChangeSetConflict(
                rel, "target path is not accessible (a parent is not a directory?)"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise ChangeSetConflict(rel, "target is not a regular file")
        if info.st_size > MAX_FILE_BYTES:
            raise ChangeSetConflict(rel, "target file is too large to verify")
        return read_bytes_retry(target), stat.S_IMODE(info.st_mode)

    @staticmethod
    def _decode(rel: str, raw: bytes) -> str:
        body = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
        try:
            return normalize_newlines(body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ChangeSetConflict(rel, "target file is not UTF-8 text") from exc

    def _plan(self, cs: ChangeSet) -> _Plan:
        targets: dict[str, Path] = {}
        raw_before: dict[str, bytes | None] = {}
        modes: dict[str, int] = {}
        texts: dict[str, str] = {}
        for op in cs.ops:
            rel = validate_rel_path(op.path)
            self._check_protected(rel)
            target = self._resolve(rel)
            raw, mode = self._read_raw(rel, target)
            targets[rel], raw_before[rel], modes[rel] = target, raw, mode
            if raw is not None:
                texts[rel] = self._decode(rel, raw)
        result = apply_changeset(MemoryView(texts), cs, strict=True)  # drift -> ChangeSetConflict
        return _Plan(cs, targets, raw_before, modes, texts, result)

    def _receipt(self, plan: _Plan) -> ApplyReceipt:
        return ApplyReceipt(
            target_id=self.target_id,
            applied=tuple(op.path for op in plan.changeset.ops),
            pre_hashes={
                rel: sha256_text(plan.texts[rel]) if rel in plan.texts else None
                for rel in plan.targets
            },
            post_hashes={
                rel: None if new is None else sha256_text(new) for rel, new in plan.result.items()
            },
            reverse=reverse_changeset(MemoryView(dict(plan.texts)), plan.changeset),
        )

    # -- commit -------------------------------------------------------------------
    def _commit(self, plan: _Plan, *, cancellable: bool) -> None:
        undo: list[tuple[str, Callable[[], None]]] = []
        try:
            for op in plan.changeset.ops:
                if cancellable:
                    self._check_cancel()
                self._apply_op(op, plan, undo)
        except BaseException as exc:
            stuck = _run_undo(undo)
            if stuck:
                raise RollbackIncompleteError(
                    f"rollback incomplete; manual reconciliation needed for: {_sample(stuck)}"
                ) from exc
            raise

    def _apply_op(
        self, op: FileOp, plan: _Plan, undo: list[tuple[str, Callable[[], None]]]
    ) -> None:
        rel, target = op.path, plan.targets[op.path]
        before = plan.raw_before[rel]
        now, _ = self._read_raw(rel, target)
        if now != before:  # narrow the check-to-use window: a human edited the file meanwhile
            raise ChangeSetConflict(rel, "target changed while the effect was being applied")
        # the undo is registered first so a step that fails half-way is rolled back too
        undo.append((rel, functools.partial(_restore, target, before, plan.modes[rel], self._root)))
        if op.kind == FileOpKind.DELETE:
            target.unlink()
            _prune_empty_parents(target.parent, self._root)
        else:
            assert op.content is not None
            atomic_write_bytes(target, op.content.encode("utf-8"), mode=plan.modes[rel])

    # -- revert -------------------------------------------------------------------
    def _pending_reverts(self, receipt: ApplyReceipt) -> set[str]:
        """Paths that must be reverted; raises ``RevertRefused`` (touching nothing) on any drift."""
        if receipt.target_id != self.target_id:
            raise RevertRefused("receipt belongs to a different target")
        paths = set(receipt.post_hashes)
        if not (
            paths
            == set(receipt.pre_hashes)
            == set(receipt.applied)
            == {op.path for op in receipt.reverse.ops}
        ):
            raise RevertRefused("receipt is internally inconsistent")
        drifted: list[str] = []
        pending: set[str] = set()
        for rel in sorted(paths):
            self._check_protected(validate_rel_path(rel))
            try:
                raw, _ = self._read_raw(rel, self._resolve(rel))
                current = None if raw is None else sha256_text(self._decode(rel, raw))
            except ChangeSetConflict:
                drifted.append(rel)
                continue
            if current == receipt.post_hashes[rel]:
                pending.add(rel)
            elif current != receipt.pre_hashes[rel]:  # equal to pre_hash: already reverted
                drifted.append(rel)
        if drifted:
            raise RevertRefused(
                f"refusing to revert: {len(drifted)} file(s) changed since the effect was "
                f"applied: {_sample(drifted)}"
            )
        return pending

    # -- audit --------------------------------------------------------------------
    def _emit(self, etype: EventType, data: dict[str, Any]) -> None:
        self._audit(
            etype,
            node_id=self._node_id,
            attempt=self._attempt,
            actor=Actor.system("effects"),
            data=data,
        )

    def _journalled(
        self,
        action: str,
        cs: ChangeSet,
        receipt: ApplyReceipt,
        run: Callable[[], None],
        *,
        noop: bool = False,
    ) -> None:
        """STARTED before the first mutation, COMPLETED (ok or not) after."""
        base = {"action": action, "target_id": self.target_id}
        self._emit(
            EventType.EFFECT_STARTED,
            {**base, "paths": sorted(cs.paths()), "changeset_digest": digest(cs), "noop": noop},
        )
        try:
            run()
        except BaseException as exc:
            self._emit(
                EventType.EFFECT_COMPLETED,
                {
                    **base,
                    "ok": False,
                    "error": type(exc).__name__,
                    "rolled_back": not isinstance(exc, RollbackIncompleteError),
                },
            )
            raise
        self._emit(
            EventType.EFFECT_COMPLETED,
            {
                **base,
                "ok": True,
                "files": len(cs.ops),
                "post_hash": hash_jsonable(receipt.post_hashes),
            },
        )
