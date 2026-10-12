"""Change-set algebra: build, apply, merge, reverse, diff.

A ``ChangeSet`` is an immutable *proposal*; this module is the only place that knows how to
turn proposals into content, combine several of them, undo one, or render it as a patch.

Semantics (normative, see ``docs/engine-semantics.md`` §6):

* ``FileOp.base_hash`` is ``fsutil.sha256_text`` of the path's content in the *view the author
  worked against* at the first touch of the path (``None`` for CREATE).
  ``apply_changeset(strict=True)`` verifies it, so a change set cannot be applied to a tree that
  drifted since it was produced.
* ``merge_changesets`` applies change sets *in the given order* against ``base``; two change sets
  touching one path conflict unless they yield identical results. The merged change set is rebuilt
  **relative to ``base``** (the integrator rebases).
* Text only, UTF-8, LF. Every path is validated by ``FileOp`` itself.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from orchestrator.core.errors import OrchestratorError
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.paths import casefold_key, validate_rel_path
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import ChangeSet, FileOp, FileOpKind, FileView


class ChangeSetConflict(OrchestratorError):
    """A change set cannot be applied to / merged with the given tree."""

    def __init__(self, path: str, reason: str, authors: tuple[str, ...] = ()) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason
        self.authors = authors


@dataclass(frozen=True)
class DiffStats:
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    lines_added: int = 0
    lines_removed: int = 0

    @property
    def files_changed(self) -> int:
        return self.files_added + self.files_modified + self.files_deleted

    def to_dict(self) -> dict[str, int]:
        return {
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
        }


@dataclass(frozen=True)
class Conflict:
    path: str
    authors: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class MergeResult:
    changeset: ChangeSet  # relative to ``base``; contains only non-conflicting paths
    conflicts: tuple[Conflict, ...]
    merged_paths: tuple[str, ...]


# ----------------------------------------------------------------------------- builders


def _read(view: FileView, path: str) -> str | None:
    try:
        return view.read_text(path)
    except FileNotFoundError:
        return None


def make_create(path: str, content: str) -> FileOp:
    return FileOp(path, FileOpKind.CREATE, normalize_text(content))


def make_modify(view: FileView, path: str, new_content: str) -> FileOp:
    old = _read(view, path)
    if old is None:
        raise ChangeSetConflict(path, "cannot modify a file that does not exist in the view")
    return FileOp(path, FileOpKind.MODIFY, normalize_text(new_content), sha256_text(old))


def make_delete(view: FileView, path: str) -> FileOp:
    old = _read(view, path)
    if old is None:
        raise ChangeSetConflict(path, "cannot delete a file that does not exist in the view")
    return FileOp(path, FileOpKind.DELETE, None, sha256_text(old))


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def compute_ops(
    view: FileView,
    desired: Mapping[str, str | None],
    *,
    description: str = "",
    author: str = "",
) -> ChangeSet:
    """Build a change set from a desired post-state (``None`` = delete) relative to ``view``.

    Paths whose desired content equals the current content are dropped; a path that does not exist
    and is desired as ``None`` is dropped.
    """
    ops: list[FileOp] = []
    # Normalise (NFC) and validate *before* reading: a hostile or differently-normalised spelling
    # must neither be silently dropped nor mistaken for a new file next to its existing twin.
    normalised = sorted(
        ((validate_rel_path(raw), new) for raw, new in desired.items()), key=lambda item: item[0]
    )
    for path, new in normalised:
        old = _read(view, path)
        if new is None:
            if old is not None:
                ops.append(FileOp(path, FileOpKind.DELETE, None, sha256_text(old)))
        elif old is None:
            ops.append(FileOp(path, FileOpKind.CREATE, normalize_text(new)))
        elif normalize_text(new) != normalize_text(old):
            ops.append(FileOp(path, FileOpKind.MODIFY, normalize_text(new), sha256_text(old)))
    return ChangeSet(tuple(ops), description, author)


# ----------------------------------------------------------------------------- apply


def apply_changeset(view: FileView, cs: ChangeSet, *, strict: bool = True) -> dict[str, str | None]:
    """Return ``{path: new_content | None}`` for every path the change set touches.

    ``strict`` verifies that CREATE targets do not exist, MODIFY/DELETE targets exist and their
    current content hashes to ``base_hash``. Raises :class:`ChangeSetConflict` on any violation
    (nothing is partially applied: this function never mutates anything).
    """
    result: dict[str, str | None] = {}
    seen: set[str] = set()
    for op in cs.ops:
        key = casefold_key(op.path)
        if key in seen:
            raise ChangeSetConflict(op.path, "duplicate path in change set")
        seen.add(key)
        current = _read(view, op.path)
        if op.kind == FileOpKind.CREATE:
            if strict and current is not None:
                raise ChangeSetConflict(op.path, "file already exists (CREATE)")
            result[op.path] = op.content
        else:
            if current is None:
                if strict:
                    raise ChangeSetConflict(
                        op.path, f"file does not exist ({op.kind.value.upper()})"
                    )
            elif strict and sha256_text(current) != op.base_hash:
                raise ChangeSetConflict(
                    op.path,
                    "base content hash mismatch (file changed since the change set was made)",
                )
            result[op.path] = op.content if op.kind == FileOpKind.MODIFY else None
    return result


def digest(cs: ChangeSet) -> str:
    """Stable digest of the ops (author/description excluded)."""
    return hash_jsonable([op.to_dict() for op in sorted(cs.ops, key=lambda o: o.path)])


# ----------------------------------------------------------------------------- merge


def merge_changesets(base: FileView, changesets: Sequence[ChangeSet]) -> MergeResult:
    """Merge ``changesets`` (in order) into one change set relative to ``base``.

    Each source must itself be consistent with ``base`` for the paths it touches (``strict``
    apply); a source that is not (e.g. made against a different view) yields a conflict for that
    path.
    Two sources touching the same path conflict unless their resulting content is identical.
    """
    final: dict[str, str | None] = {}
    owners: dict[str, list[str]] = {}
    original_path: dict[str, str] = {}
    conflicts: dict[str, Conflict] = {}

    for idx, cs in enumerate(changesets):
        author = cs.author or f"changeset[{idx}]"
        try:
            outcome = apply_changeset(base, cs, strict=True)
        except ChangeSetConflict as exc:
            conflicts[casefold_key(exc.path)] = Conflict(exc.path, (author,), exc.reason)
            outcome = {}
            # fall back to per-path application so unrelated paths still merge
            for op in cs.ops:
                try:
                    outcome.update(apply_changeset(base, ChangeSet((op,)), strict=True))
                except ChangeSetConflict as inner:
                    conflicts.setdefault(
                        casefold_key(inner.path), Conflict(inner.path, (author,), inner.reason)
                    )
        for path, content in outcome.items():
            key = casefold_key(path)
            original_path.setdefault(key, path)
            if key in final:
                owners[key].append(author)
                if final[key] != content or original_path[key] != path:
                    conflicts[key] = Conflict(
                        original_path[key],
                        tuple(dict.fromkeys(owners[key])),
                        "path modified differently by more than one change set",
                    )
            else:
                final[key] = content
                owners[key] = [author]

    desired = {original_path[k]: v for k, v in final.items() if k not in conflicts}
    merged = compute_ops(base, desired, description="merged", author="integrator")
    return MergeResult(
        merged, tuple(sorted(conflicts.values(), key=lambda c: c.path)), tuple(sorted(desired))
    )


# ----------------------------------------------------------------------------- reverse


def reverse_changeset(view_before: FileView, cs: ChangeSet) -> ChangeSet:
    """The inverse of ``cs``, applicable to the tree *after* ``cs``.

    ``view_before`` is the tree ``cs`` was applied to.
    """
    ops: list[FileOp] = []
    for op in cs.ops:
        before = _read(view_before, op.path)
        if op.kind == FileOpKind.CREATE:
            assert op.content is not None
            if before is None:
                ops.append(FileOp(op.path, FileOpKind.DELETE, None, sha256_text(op.content)))
            else:  # a non-strict CREATE overwrote an existing file: restoring it, not deleting it
                ops.append(FileOp(op.path, FileOpKind.MODIFY, before, sha256_text(op.content)))
        elif op.kind == FileOpKind.MODIFY:
            assert op.content is not None
            if before is None:
                raise ChangeSetConflict(op.path, "cannot reverse MODIFY: original content unknown")
            ops.append(FileOp(op.path, FileOpKind.MODIFY, before, sha256_text(op.content)))
        else:
            if before is None:
                raise ChangeSetConflict(op.path, "cannot reverse DELETE: original content unknown")
            ops.append(FileOp(op.path, FileOpKind.CREATE, before))
    return ChangeSet(tuple(ops), f"reverse of: {cs.description}", cs.author)


# ----------------------------------------------------------------------------- diff / patch


def _split_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _hunk_lines(a: str, b: str, a_name: str, b_name: str) -> list[str]:
    a_lines, b_lines = _split_lines(a), _split_lines(b)
    out: list[str] = []
    for line in difflib.unified_diff(
        a_lines, b_lines, fromfile=a_name, tofile=b_name, n=3, lineterm="\n"
    ):
        out.append(line)
    # difflib omits the newline marker for a final line without "\n"; add git's marker
    fixed: list[str] = []
    for line in out:
        if line.endswith("\n"):
            fixed.append(line)
        else:
            fixed.append(line + "\n\\ No newline at end of file\n")
    return fixed


def to_unified_diff(view: FileView, cs: ChangeSet) -> str:
    """Render ``cs`` (relative to ``view``) as a git-style patch (LF, ``a/`` ``b/`` prefixes)."""
    chunks: list[str] = []
    for op in sorted(cs.ops, key=lambda o: o.path):
        old = _read(view, op.path)
        p = op.path
        if op.kind == FileOpKind.CREATE:
            assert op.content is not None
            chunks.append(f"diff --git a/{p} b/{p}\nnew file mode 100644\n")
            if op.content:
                chunks.extend(_hunk_lines("", op.content, "/dev/null", f"b/{p}"))
        elif op.kind == FileOpKind.DELETE:
            chunks.append(f"diff --git a/{p} b/{p}\ndeleted file mode 100644\n")
            if old:
                chunks.extend(_hunk_lines(old, "", f"a/{p}", "/dev/null"))
        else:
            assert op.content is not None
            if old == op.content:
                continue
            chunks.append(f"diff --git a/{p} b/{p}\n")
            chunks.extend(_hunk_lines(old or "", op.content, f"a/{p}", f"b/{p}"))
    return "".join(chunks)


def diff_stats(view: FileView, cs: ChangeSet) -> DiffStats:
    added = modified = deleted = lines_add = lines_rem = 0
    for op in cs.ops:
        old = _read(view, op.path) or ""
        new = op.content or ""
        if op.kind == FileOpKind.CREATE:
            added += 1
        elif op.kind == FileOpKind.DELETE:
            deleted += 1
        else:
            modified += 1
        matcher = difflib.SequenceMatcher(a=_split_lines(old), b=_split_lines(new), autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("replace", "delete"):
                lines_rem += i2 - i1
            if tag in ("replace", "insert"):
                lines_add += j2 - j1
    return DiffStats(added, modified, deleted, lines_add, lines_rem)
