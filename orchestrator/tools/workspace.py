"""File views, baseline snapshots and safe materialisation.

* ``MemoryView`` / ``DirectoryView`` / ``StagedView`` / ``OverlayView`` implement the read-only
  ``FileView`` protocol. They never follow symlinks/junctions, only expose UTF-8 text files
  (binary and oversize files are skipped and *recorded*), normalise CRLF -> LF and apply
  ``paths.match_glob`` semantics.
* ``BaselineSnapshot`` freezes the target tree at run start (``<run_dir>/baseline/``), with a
  manifest of ``sha256_text`` per file and a ``manifest_hash`` used in fingerprints and approvals.
* ``materialize`` writes a view to a scratch directory through ``fsutil.safe_join`` (containment,
  no links, validated paths) so tests can run against ``baseline + change sets``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchestrator.core.errors import ConfigError, IntegrityError, PathViolation
from orchestrator.core.fsutil import (
    atomic_write_json,
    is_link_or_reparse,
    read_text_lf,
    safe_join,
    sha256_text,
    write_text_lf,
)
from orchestrator.core.paths import match_any, match_glob, validate_rel_path
from orchestrator.core.serde import hash_jsonable, loads_strict
from orchestrator.core.types import ChangeSet, FileView
from orchestrator.tools.changeset import apply_changeset

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git/**",
    ".venv/**",
    "venv/**",
    ".agentic/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".mypy_cache/**",
    "node_modules/**",
    "**/.env",  # at any depth: a nested deploy/.env is as sensitive as the top-level one
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/*.db",
    "**/*.db-wal",
    "**/*.db-shm",
    "**/*.sqlite*",
    "*.egg-info/**",
)
MAX_FILE_BYTES = 1024 * 1024
MAX_FILES = 5000
MAX_TOTAL_BYTES = 64 * 1024 * 1024


# ----------------------------------------------------------------------------- views


class MemoryView:
    """In-memory ``FileView`` (tests, small trees)."""

    def __init__(self, files: Mapping[str, str] | None = None) -> None:
        self._files = {
            validate_rel_path(p): t.replace("\r\n", "\n") for p, t in (files or {}).items()
        }

    def exists(self, path: str) -> bool:
        return path in self._files

    def read_text(self, path: str) -> str:
        try:
            return self._files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    def list_files(self, glob: str = "**") -> list[str]:
        return sorted(p for p in self._files if match_glob(p, glob))


@dataclass
class ScanReport:
    skipped_binary: list[str] = field(default_factory=list)
    skipped_oversize: list[str] = field(default_factory=list)
    skipped_invalid_path: list[str] = field(default_factory=list)
    skipped_links: list[str] = field(default_factory=list)


class DirectoryView:
    """Read-only text view of a directory tree (links never followed; bounded)."""

    def __init__(
        self,
        root: Path | str,
        *,
        include: Sequence[str] = ("**",),
        exclude: Sequence[str] = DEFAULT_EXCLUDES,
        max_files: int = MAX_FILES,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ) -> None:
        self.root = Path(root).resolve()
        self.report = ScanReport()
        self._index: dict[str, Path] = {}
        include_globs, exclude_globs = tuple(include), tuple(exclude)
        total = 0
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            rel_dir = Path(dirpath).relative_to(self.root).as_posix()
            kept: list[str] = []
            for d in sorted(dirnames):
                rel = d if rel_dir == "." else f"{rel_dir}/{d}"
                if is_link_or_reparse(Path(dirpath) / d):
                    self.report.skipped_links.append(rel)
                elif not match_any(f"{rel}/_", exclude_globs):
                    kept.append(d)
            dirnames[:] = kept
            for name in sorted(filenames):
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                full = Path(dirpath) / name
                if is_link_or_reparse(full):
                    self.report.skipped_links.append(rel)
                    continue
                try:
                    rel = validate_rel_path(rel)
                except PathViolation:
                    self.report.skipped_invalid_path.append(rel)
                    continue
                if match_any(rel, exclude_globs) or not match_any(rel, include_globs):
                    continue
                try:
                    size = full.stat().st_size
                except OSError:
                    continue
                if size > max_file_bytes:
                    self.report.skipped_oversize.append(rel)
                    continue
                try:
                    is_text = "\x00" not in full.read_bytes().decode(
                        "utf-8"
                    )  # NUL: not editable text
                except (UnicodeDecodeError, OSError):
                    is_text = False
                if not is_text:
                    self.report.skipped_binary.append(rel)
                    continue
                total += size
                if len(self._index) + 1 > max_files or total > max_total_bytes:
                    raise ConfigError(
                        "directory too large for a snapshot "
                        f"(> {max_files} files / {max_total_bytes} bytes)"
                    )
                self._index[rel] = full

    def exists(self, path: str) -> bool:
        return path in self._index

    def read_text(self, path: str) -> str:
        try:
            return read_text_lf(self._index[path])
        except KeyError:
            raise FileNotFoundError(path) from None

    def list_files(self, glob: str = "**") -> list[str]:
        return sorted(p for p in self._index if match_glob(p, glob))


class StagedView:
    """``base`` + a dict of pending writes (``None`` = deleted). Used for staged agent writes."""

    def __init__(self, base: FileView, writes: Mapping[str, str | None] | None = None) -> None:
        self.base = base
        self.writes: dict[str, str | None] = dict(writes or {})

    def exists(self, path: str) -> bool:
        if path in self.writes:
            return self.writes[path] is not None
        return self.base.exists(path)

    def read_text(self, path: str) -> str:
        if path in self.writes:
            value = self.writes[path]
            if value is None:
                raise FileNotFoundError(path)
            return value
        return self.base.read_text(path)

    def list_files(self, glob: str = "**") -> list[str]:
        paths = {p for p in self.base.list_files("**") if self.writes.get(p, "") is not None}
        paths |= {p for p, v in self.writes.items() if v is not None}
        return sorted(p for p in paths if match_glob(p, glob))


class OverlayView(StagedView):
    """``base`` + change sets applied *in order*, each relative to the result so far.

    ``strict=True`` verifies base hashes (a conflict raises ``ChangeSetConflict``).
    """

    def __init__(
        self, base: FileView, changesets: Iterable[ChangeSet] = (), *, strict: bool = True
    ) -> None:
        super().__init__(base)
        for cs in changesets:
            self.writes.update(apply_changeset(self, cs, strict=strict))


# ----------------------------------------------------------------------------- snapshot

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _check_manifest(meta: Any) -> None:
    """Refuse a corrupt or hostile ``manifest.json`` (the trust anchor of ``verify``)."""
    files = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(files, dict):
        raise IntegrityError("baseline manifest is malformed")
    for rel, file_hash in files.items():
        try:
            validate_rel_path(rel)
        except PathViolation as exc:
            raise IntegrityError("baseline manifest lists an unsafe path") from exc
        if not isinstance(file_hash, str) or not _SHA256_RE.match(file_hash):
            raise IntegrityError("baseline manifest holds a malformed file hash")


class BaselineSnapshot:
    """The frozen target tree of a run (``<dest>/files`` + ``<dest>/manifest.json``)."""

    def __init__(self, dest: Path, manifest: dict[str, Any]) -> None:
        self.dest = Path(dest)
        self._manifest = manifest
        self._view: DirectoryView | None = None

    # -- construction ---------------------------------------------------------
    @classmethod
    def create(
        cls,
        source_root: Path | str,
        dest: Path | str,
        *,
        include: Sequence[str] = ("**",),
        exclude: Sequence[str] = (),
        max_files: int = MAX_FILES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ) -> BaselineSnapshot:
        """Snapshot ``source_root`` (never following links; default excludes always apply)."""
        src = DirectoryView(
            source_root,
            include=include,
            exclude=(*DEFAULT_EXCLUDES, *exclude),
            max_files=max_files,
            max_total_bytes=max_total_bytes,
        )
        dest_path = Path(dest)
        files_dir = dest_path / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, str] = {}
        for rel in src.list_files("**"):
            text = src.read_text(rel)
            write_text_lf(safe_join(files_dir, rel), text)
            manifest[rel] = sha256_text(text)
        meta = {
            "schema": 1,
            "source_root": Path(source_root).resolve().as_posix(),
            "files": manifest,
            "skipped": {
                "binary": src.report.skipped_binary,
                "oversize": src.report.skipped_oversize,
                "invalid_path": src.report.skipped_invalid_path,
                "links": src.report.skipped_links,
            },
        }
        atomic_write_json(dest_path / "manifest.json", meta)
        return cls(dest_path, meta)

    @classmethod
    def empty(cls, dest: Path | str) -> BaselineSnapshot:
        dest_path = Path(dest)
        (dest_path / "files").mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"schema": 1, "source_root": None, "files": {}, "skipped": {}}
        atomic_write_json(dest_path / "manifest.json", meta)
        return cls(dest_path, meta)

    @classmethod
    def load(cls, dest: Path | str) -> BaselineSnapshot:
        dest_path = Path(dest)
        meta = loads_strict((dest_path / "manifest.json").read_bytes())
        _check_manifest(meta)
        return cls(dest_path, meta)

    # -- accessors -------------------------------------------------------------
    @property
    def manifest(self) -> dict[str, str]:
        return dict(self._manifest["files"])

    @property
    def manifest_hash(self) -> str:
        return hash_jsonable(self._manifest["files"])

    @property
    def is_empty(self) -> bool:
        return not self._manifest["files"]

    @property
    def file_count(self) -> int:
        return len(self._manifest["files"])

    @property
    def source_root(self) -> str | None:
        return self._manifest.get("source_root")

    @property
    def view(self) -> DirectoryView:
        if self._view is None:
            self._view = DirectoryView(self.dest / "files", exclude=())
        return self._view

    def verify(self) -> list[str]:
        """Re-hash every file; returns a list of problems (empty = intact)."""
        problems: list[str] = []
        view = DirectoryView(self.dest / "files", exclude=())  # fresh scan: never the cached index
        manifest = self._manifest["files"]
        for rel, expected in manifest.items():
            try:
                if sha256_text(view.read_text(rel)) != expected:
                    problems.append(f"hash mismatch: {rel}")
            except FileNotFoundError:
                problems.append(f"missing: {rel}")
        for rel in view.list_files("**"):
            if rel not in manifest:
                problems.append(f"unexpected file: {rel}")
        return problems


# ----------------------------------------------------------------------------- materialise


def materialize(
    view: FileView,
    dest: Path | str,
    paths: Iterable[str] | None = None,
    *,
    max_files: int = MAX_FILES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> int:
    """Write ``view`` (or the subset ``paths``) under ``dest``; returns the number of files written.

    Every path goes through ``safe_join`` (validation, containment, no links). The write is bounded.
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    selected = list(paths) if paths is not None else view.list_files("**")
    if len(selected) > max_files:
        raise ConfigError(f"too many files to materialise ({len(selected)} > {max_files})")
    total = 0
    for rel in selected:
        text = view.read_text(rel)
        total += len(text.encode("utf-8"))
        if total > max_total_bytes:
            raise ConfigError("materialised tree exceeds the byte budget")
        target = safe_join(dest_path, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_lf(target, text)
    return len(selected)
