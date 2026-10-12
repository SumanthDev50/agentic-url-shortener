"""Portable, crash-safe file-system primitives (Windows + POSIX).

Every persistent writer in the orchestrator (state, artifact index, decision log,
audit log, kill switch, run lock) uses these helpers so the Windows pitfalls are
handled in exactly one place:

* ``os.replace`` onto a file another handle has open raises ``PermissionError``
  on Windows -> retried with backoff.
* ``Path.write_text`` emits CRLF on Windows -> all text I/O here is LF-only and
  **content hashing is CRLF-insensitive** (:func:`sha256_text`), so a baseline
  checked out with CRLF hashes identically to its LF form.
* ``os.kill(pid, 0)`` on Windows *terminates* the process -> :func:`pid_alive`
  uses the Win32 API instead.
* Symlinks/junctions are never followed when snapshotting or materialising.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import stat
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from orchestrator.core.errors import LockError, PathViolation

T = TypeVar("T")

_WIN = sys.platform == "win32"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_RETRIES = 12


# ----------------------------------------------------------------------------- hashing


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_text(text: str) -> str:
    """The ONLY hash used for file *content* (manifests, ``FileOp.base_hash``)."""
    return hashlib.sha256(normalize_newlines(text).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- text I/O


def read_text_lf(path: Path | str) -> str:
    """Read UTF-8 (BOM stripped), newlines normalised to LF. Raises ``UnicodeDecodeError`` on binary."""
    raw = read_bytes_retry(path)
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return normalize_newlines(raw.decode("utf-8"))


def write_text_lf(path: Path | str, text: str) -> None:
    atomic_write_bytes(path, normalize_newlines(text).encode("utf-8"))


def read_bytes_retry(path: Path | str) -> bytes:
    return _retry_permission(lambda: Path(path).read_bytes())


def _retry_permission(fn: Callable[[], T]) -> T:
    delay = 0.005
    for attempt in range(_RETRIES):
        try:
            return fn()
        except PermissionError:
            if not _WIN or attempt == _RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.7, 0.1)
    raise RuntimeError("unreachable")  # pragma: no cover


def atomic_write_bytes(path: Path | str, data: bytes, *, mode: int = 0o600) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same dir + fsync + replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if not _WIN:
            os.chmod(tmp_name, mode)
        _retry_permission(lambda: os.replace(tmp_name, target))
        _fsync_dir(target.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_write_json(path: Path | str, obj: Any) -> None:
    from orchestrator.core.serde import to_jsonable

    data = json.dumps(to_jsonable(obj), sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
    atomic_write_bytes(path, (data + "\n").encode("ascii"))


def append_line(path: Path | str, line: bytes, *, fsync: bool = False) -> None:
    """Append one ``\\n``-terminated line (O_APPEND). ``fsync`` for governance events."""
    if b"\n" in line:
        raise ValueError("line must not contain a newline")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "ab") as fh:
        fh.write(line + b"\n")
        fh.flush()
        if fsync:
            os.fsync(fh.fileno())


def _fsync_dir(directory: Path) -> None:
    if _WIN:
        return
    with contextlib.suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


# ----------------------------------------------------------------------------- links / containment


def is_link_or_reparse(path: Path | str) -> bool:
    """True for symlinks, Windows junctions/mount points and any reparse point."""
    p = Path(path)
    try:
        st = os.lstat(p)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & _FILE_ATTRIBUTE_REPARSE_POINT)


def safe_join(root: Path | str, rel: str) -> Path:
    """Join a *validated* relative POSIX path to ``root`` with containment + link checks."""
    from orchestrator.core.paths import validate_rel_path

    rel = validate_rel_path(rel)
    base = Path(root).resolve()
    candidate = base.joinpath(*rel.split("/"))
    current = base
    for part in rel.split("/"):
        current = current / part
        if current.exists() and is_link_or_reparse(current):
            raise PathViolation(f"link/reparse point in path: {rel}")
    if not candidate.resolve().is_relative_to(base):
        raise PathViolation(f"path escapes root: {rel}")
    return candidate


# ----------------------------------------------------------------------------- process liveness + run lock


def pid_alive(pid: int) -> bool:
    """Is ``pid`` a live process? (Never signals the process.)"""
    if pid <= 0:
        return False
    if _WIN:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """Exclusive, crash-tolerant lock on a run directory (``run.lock``).

    Created with ``O_CREAT | O_EXCL``. A lock is *stale* when its pid is dead or
    its heartbeat (mtime) is older than ``ttl_s``; stale locks are taken over.
    The holder must call :meth:`heartbeat` periodically while running.
    """

    def __init__(self, run_dir: Path | str, *, ttl_s: float = 60.0) -> None:
        self.path = Path(run_dir) / "run.lock"
        self.ttl_s = ttl_s
        self._held = False

    # context-manager sugar
    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def _read(self) -> dict[str, Any] | None:
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        # a foreign or garbled file means "no information", never a crash
        return info if isinstance(info, dict) else None

    def _is_stale(self) -> bool:
        info = self._read()
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return True
        if info is None:
            return age > self.ttl_s
        pid = info.get("pid")
        host_matches = info.get("host") == socket.gethostname()
        live = isinstance(pid, int) and not isinstance(pid, bool) and pid_alive(pid)
        if host_matches and not live:
            return True
        return age > self.ttl_s

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if self._is_stale():
                    with contextlib.suppress(OSError):
                        os.unlink(self.path)
                    continue
                return False
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started": time.time()}, fh)
            self._held = True
            return True
        return False

    def acquire(self) -> None:
        if not self.try_acquire():
            raise LockError(f"run is locked by another live process: {self.path}")

    def heartbeat(self) -> None:
        if self._held:
            with contextlib.suppress(OSError):
                os.utime(self.path, None)

    def release(self) -> None:
        if self._held:
            with contextlib.suppress(OSError):
                os.unlink(self.path)
            self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def is_locked_by_other(self) -> bool:
        """True if some *other* live process holds the lock."""
        if self._held:
            return False
        return self.path.exists() and not self._is_stale()
