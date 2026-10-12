"""Process-level command runner for agent-authored code (tests, helper scripts).

.. warning::
   This is **process-level isolation, not a security sandbox**.  The child runs as the same
   OS user with the same file-system and network access as the orchestrator; there are no
   memory, CPU or network limits.  What it *does* provide, and what the rest of the system
   relies on:

   * an **allow-listed environment built from scratch** (no inherited secrets: API keys, cloud
     credentials and tokens in the parent environment never reach the child);
   * **fixed argv grammars** (``pytest_argv`` / ``python_argv``): no ``-c``, no pytest plugin /
     rootdir / argfile injection, every path operand contained in the scratch directory;
   * a hard **wall-clock timeout** and **cooperative cancellation**, both of which kill the whole
     **process tree** (``taskkill /T /F`` on Windows, ``killpg`` on POSIX) - a test that forks a
     daemon cannot outlive its attempt;
   * **bounded output**: streams are drained by reader threads that keep only the head and tail
     of the output, so a child printing gigabytes cannot exhaust memory; what is kept is
     control-stripped and secret-redacted before anyone sees it.

   The real security boundary is structural (proposals only, autonomy checks, policy, human
   approval).  Do not run hostile code here and call it safe.

Known limitation: on Windows an orphan that outlives its parent *before* the tree is killed
cannot be found again by ``taskkill /T`` (the parent link is gone).
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from orchestrator.core.audit import Redactor
from orchestrator.core.errors import AgentCancelled, ConfigError, OrchestratorError
from orchestrator.core.sanitize import strip_controls
from orchestrator.core.types import CancelToken, CommandResult

Redact = Callable[[str], str]

DEFAULT_MAX_OUTPUT_BYTES = 262_144
MAX_OUTPUT_BYTES_LIMIT = 16 * 1024 * 1024
MAX_TIMEOUT_S = 3600.0

# Modules ``python_argv`` accepts after ``-m``.  ``pytest`` is deliberately absent: tests go
# through ``pytest_argv`` whose argv is fixed.
ALLOWED_MODULES = frozenset({"unittest", "py_compile", "compileall", "json.tool", "doctest"})

_OPERAND_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
_SAFE_FLAGS = frozenset({"-v", "-q"})
_MAX_ARG_CHARS = 240
_MAX_ARGV_ITEMS = 64
_READ_CHUNK = 65_536
_KILL_GRACE_S = 10.0
_READER_JOIN_S = 5.0
_TASKKILL_TIMEOUT_S = 15.0
_RMTREE_RETRIES = 6


class SandboxArgError(OrchestratorError, ValueError):
    """An argv, target or working directory was rejected by the sandbox grammar."""


# ----------------------------------------------------------------------------- bounded output


class _BoundedCollector:
    """Keeps the first and last ``cap // 2`` bytes of a stream and counts the rest.

    Memory is bounded by ``cap`` plus one read chunk regardless of how much the child prints.
    When output is cut, the head and tail are trimmed to whole lines so a secret is never split
    across the cut where a pattern-based redactor could no longer recognise its remains.
    """

    def __init__(self, cap: int) -> None:
        self._head_cap = cap // 2
        self._tail_cap = cap - self._head_cap
        self._cap = cap
        self._head = bytearray()
        self._tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self._head_cap - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail += chunk
            if len(self._tail) > 2 * self._tail_cap:  # amortised trim keeps appends O(1)
                del self._tail[: len(self._tail) - self._tail_cap]

    @property
    def truncated(self) -> bool:
        return self.total > self._cap

    def text(self) -> str:
        head = bytes(self._head)
        tail = bytes(self._tail[-self._tail_cap :]) if self._tail_cap else b""
        if not self.truncated:
            return (head + tail).decode("utf-8", "replace")
        cut = head.rfind(b"\n")
        head = head[: cut + 1] if cut >= 0 else b""
        newline = tail.find(b"\n")
        tail = tail[newline + 1 :] if newline >= 0 else b""
        omitted = self.total - len(head) - len(tail)
        marker = f"\n[... output truncated: {omitted} bytes omitted ...]\n"
        return head.decode("utf-8", "replace") + marker + tail.decode("utf-8", "replace")


def _drain(stream: Any, collector: _BoundedCollector) -> None:
    """Reader-thread body: read until EOF, never blocking the child on a full pipe."""
    try:
        while True:
            chunk = stream.read(_READ_CHUNK)
            if not chunk:
                return
            collector.feed(chunk)
    except (OSError, ValueError):  # pipe closed underneath us
        return


# ----------------------------------------------------------------------------- process tree


def _kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill ``proc`` and every descendant. Never raises."""
    if sys.platform == "win32":
        root = os.environ.get("SYSTEMROOT") or r"C:\Windows"
        taskkill = str(Path(root) / "System32" / "taskkill.exe")
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [taskkill, "/PID", str(proc.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_TASKKILL_TIMEOUT_S,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        with contextlib.suppress(OSError):  # fallback when taskkill is unavailable
            proc.kill()
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)


class _ProcessGuard:
    """Kill-once / finish-once state of one child.

    The cancel token keeps its callbacks after the run is over, so a late cancel must never
    signal a PID the OS may have recycled: once ``finish`` ran, ``kill`` is a no-op.
    """

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        self._proc = proc
        self._lock = threading.Lock()
        self._done = False
        self.cause: str | None = None

    def kill(self, cause: str) -> None:
        with self._lock:
            if self._done:
                return
            if self.cause is None:
                self.cause = cause
            _kill_process_tree(self._proc)

    def finish(self) -> None:
        """Mark the run over and make sure no descendant is left behind."""
        with self._lock:
            if self._done:
                return
            self._done = True
            # POSIX: the group still has members whenever stragglers exist, so its id cannot
            # have been recycled; Windows can only find descendants of a live parent.
            if self._proc.poll() is None or sys.platform != "win32":
                _kill_process_tree(self._proc)


def _popen_kwargs() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


# ----------------------------------------------------------------------------- validation helpers


def _validate_timeout(timeout_s: float) -> float:
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or not 0 < timeout_s <= MAX_TIMEOUT_S
    ):
        raise ConfigError(f"timeout_s must be a finite number in (0, {MAX_TIMEOUT_S:g}]")
    return float(timeout_s)


def _validate_output_cap(max_output_bytes: int) -> int:
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or not 2 <= max_output_bytes <= MAX_OUTPUT_BYTES_LIMIT
    ):
        raise ConfigError(f"max_output_bytes must be an int in [2, {MAX_OUTPUT_BYTES_LIMIT}]")
    return max_output_bytes


def _str_list(items: Sequence[str], what: str) -> list[str]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise SandboxArgError(f"{what} must be a sequence of strings")
    if len(items) > _MAX_ARGV_ITEMS:
        raise SandboxArgError(f"too many {what} ({len(items)} > {_MAX_ARGV_ITEMS})")
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise SandboxArgError(f"{what} must be strings")
        if "\x00" in item or len(item) > 4096:
            raise SandboxArgError(f"invalid {what} item")
        out.append(item)
    return out


def _same_file(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _system_dirs() -> list[str]:
    if sys.platform == "win32":
        root = os.environ.get("SYSTEMROOT") or r"C:\Windows"
        return [str(Path(root) / "System32"), root]
    return ["/usr/bin", "/bin"]


# ----------------------------------------------------------------------------- runner



@functools.lru_cache(maxsize=1)
def sandbox_pytest_ini() -> Path:
    """An empty pytest config in a private, randomly named temp directory (created once per process).

    Passing it with ``-c`` stops pytest from ascending into an ancestor directory, so a scratch dir
    that lives inside a repository does not inherit that repository's ``pyproject.toml``. It lives
    outside the work tree (so it never shows up in a materialised tree) and under an unpredictable
    name (a fixed path in a shared temp dir could be pre-created by another local user).
    """
    directory = Path(tempfile.mkdtemp(prefix="agentic-pytest-"))
    ini = directory / "pytest-sandbox.ini"
    ini.write_bytes(b"[pytest]" + bytes([10]))
    atexit.register(shutil.rmtree, directory, True)
    return ini

class SandboxRunner:
    """Runs the orchestrator's own interpreter in a scratch directory (see module docstring).

    Thread-safe: instances hold no per-run state.  Every scratch directory must live under
    ``scratch_root``; ``new_scratch`` hands out fresh ones.
    """

    def __init__(self, scratch_root: Path | str, *, redactor: Redact | None = None) -> None:
        root = Path(scratch_root)
        root.mkdir(parents=True, exist_ok=True)
        self.scratch_root = root.resolve()
        self._redact: Redact = redactor if redactor is not None else Redactor().redact_text

    # -- scratch management ---------------------------------------------------
    def new_scratch(self, prefix: str = "scratch") -> Path:
        """Create a fresh, empty, uniquely named directory under ``scratch_root``."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", prefix):
            raise SandboxArgError("invalid scratch prefix")
        return Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=self.scratch_root)).resolve()

    def discard(self, scratch: Path | str) -> None:
        """Delete a scratch directory from ``new_scratch`` (refuses anything outside the root)."""
        target = self._contain(scratch, allow_root=False)
        for attempt in range(_RMTREE_RETRIES):
            shutil.rmtree(target, ignore_errors=True)
            if not target.exists():
                return
            time.sleep(0.02 * (attempt + 1))  # Windows releases handles of killed children lazily

    def _contain(self, path: Path | str, *, allow_root: bool = True) -> Path:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.scratch_root) or (
            not allow_root and resolved == self.scratch_root
        ):
            raise SandboxArgError("path is outside the sandbox scratch root")
        return resolved

    # -- environment ----------------------------------------------------------
    def build_env(self, cwd: Path | str) -> dict[str, str]:
        """The child's whole environment: an allow-list built from nothing, never ``os.environ``."""
        work = Path(cwd)
        home = work / ".sandbox" / "home"
        tmp = work / ".sandbox" / "tmp"
        env: dict[str, str] = {
            "PATH": os.pathsep.join([str(Path(sys.executable).parent), *_system_dirs()]),
            "HOME": str(home),
            "USERPROFILE": str(home),
            "TEMP": str(tmp),
            "TMP": str(tmp),
            "TMPDIR": str(tmp),
            "PYTHONPATH": str(work),
            "PYTHONUTF8": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
        }
        if (
            sys.platform == "win32"
        ):  # Python (winsock, hashing, subprocess) fails to start without these
            root = os.environ.get("SYSTEMROOT") or r"C:\Windows"
            env["SYSTEMROOT"] = root
            env["WINDIR"] = os.environ.get("WINDIR") or root
            env["COMSPEC"] = os.environ.get("COMSPEC") or str(Path(root) / "System32" / "cmd.exe")
        return env

    # -- argv builders --------------------------------------------------------
    def pytest_argv(self, scratch: Path | str, targets: Sequence[str]) -> list[str]:
        """The fixed pytest command line; only ``targets`` vary and each is strictly validated."""
        base = self._contain(scratch)
        checked = [
            self._check_operand(base, t, allow_flags=False) for t in _str_list(targets, "targets")
        ]
        # Hermeticity: an explicit (empty) config + rootdir stop pytest from ascending into an
        # ancestor directory (the scratch dir may live inside a repository whose own pyproject.toml
        # would otherwise change the rootdir, test ids, markers and ``pythonpath``).
        ini = sandbox_pytest_ini()
        return [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(ini),
            "--rootdir",
            str(base),
            "--confcutdir",
            str(base),
            "-q",
            # one PASSED/FAILED/ERROR/SKIPPED line per test id (no captured-output sections, which would
            # overflow the output cap on a large suite): agents parse these lines
            "-rfEsxXp",
            "--tb=short",
            "--show-capture=no",
            "--basetemp",
            str(base / ".tmp"),
            "--junitxml",
            str(base / "junit.xml"),
            "--",
            *checked,
        ]

    def python_argv(self, args: Sequence[str], *, scratch: Path | str | None = None) -> list[str]:
        """``[-m, <allow-listed module>, ...]`` or ``[<script.py in scratch>, ...]``; never ``-c``.

        Operands after the module / script are positional paths or words only (plus ``-v``/``-q``).
        """
        items = _str_list(args, "python arguments")
        if not items:
            raise SandboxArgError("no python arguments")
        base = self._contain(scratch) if scratch is not None else self.scratch_root
        if items[0] == "-m":
            if len(items) < 2 or items[1] not in ALLOWED_MODULES:
                raise SandboxArgError(
                    f"module is not allow-listed (allowed: {', '.join(sorted(ALLOWED_MODULES))})"
                )
            head = [sys.executable, "-m", items[1]]
            operands = items[2:]
        else:
            script = self._check_operand(base, items[0], allow_flags=False)
            if not script.endswith(".py"):
                raise SandboxArgError(
                    "only '-m <module>' or a '*.py' script inside the scratch directory may run"
                )
            head = [sys.executable, script]
            operands = items[1:]
        return [*head, *(self._check_operand(base, a, allow_flags=True) for a in operands)]

    @staticmethod
    def _check_operand(base: Path, arg: str, *, allow_flags: bool) -> str:
        if allow_flags and arg in _SAFE_FLAGS:
            return arg
        if len(arg) > _MAX_ARG_CHARS or not _OPERAND_RE.match(arg):
            raise SandboxArgError(f"argument rejected by the sandbox grammar: {arg[:60]!r}")
        if arg.startswith(("-", "@", "/")):
            raise SandboxArgError(
                f"option, argfile and absolute operands are not allowed: {arg[:60]!r}"
            )
        if any(seg in ("", "..") for seg in arg.split("/")):
            raise SandboxArgError(f"path escapes the scratch directory: {arg[:60]!r}")
        if not (base / arg).resolve().is_relative_to(base):
            raise SandboxArgError(f"path escapes the scratch directory: {arg[:60]!r}")
        return arg

    # -- execution ------------------------------------------------------------
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | str,
        timeout_s: float,
        cancel: CancelToken | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> CommandResult:
        """Run ``argv`` (program: this interpreter) in ``cwd`` and collect bounded output.

        Raises ``AgentCancelled`` after killing the process tree when ``cancel`` trips; a timeout
        is reported as ``CommandResult.timed_out`` instead.
        """
        command = _str_list(argv, "argv")
        if not command or not _same_file(command[0], sys.executable):
            raise SandboxArgError("the sandbox only runs the orchestrator's own Python interpreter")
        workdir = self._contain(cwd)
        if not workdir.is_dir():
            raise SandboxArgError("cwd is not a directory")
        timeout = _validate_timeout(timeout_s)
        cap = _validate_output_cap(max_output_bytes)
        if cancel is not None:
            cancel.raise_if_cancelled()

        env = self.build_env(workdir)
        for sub in (env["HOME"], env["TMP"]):
            Path(sub).mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        proc = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            shell=False,
            close_fds=True,
            **_popen_kwargs(),
        )
        guard = _ProcessGuard(proc)
        out, err = _BoundedCollector(cap), _BoundedCollector(cap)
        readers = [
            threading.Thread(
                target=_drain, args=(proc.stdout, out), daemon=True, name="sandbox-stdout"
            ),
            threading.Thread(
                target=_drain, args=(proc.stderr, err), daemon=True, name="sandbox-stderr"
            ),
        ]
        timed_out = False
        try:  # from here on, whatever happens, the child's whole tree is gone before we return
            if cancel is not None:
                # runs immediately (killing the child) if the token tripped since the check above
                cancel.add_callback(lambda: guard.kill("cancel"))
            for reader in readers:
                reader.start()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                guard.kill("timeout")
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=_KILL_GRACE_S)
        finally:
            guard.finish()
            for reader in readers:
                if reader.ident is not None:  # never started if setup above failed
                    reader.join(timeout=_READER_JOIN_S)
            if not any(reader.is_alive() for reader in readers):
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
        if guard.cause == "cancel":
            raise AgentCancelled("sandbox command cancelled")
        returncode = proc.returncode if proc.returncode is not None else -1
        return CommandResult(
            command=tuple(command),
            returncode=returncode,
            stdout=self._render(out),
            stderr=self._render(err),
            duration_s=time.monotonic() - started,
            timed_out=timed_out,
            truncated=out.truncated or err.truncated,
        )

    def _render(self, collector: _BoundedCollector) -> str:
        """Controls are stripped *before* redaction so an escape sequence cannot hide a secret.

        Redaction runs line by line: the redactor caps a single string (about 8 KiB), which would cut
        the per-test ``-rA`` lines of a large suite; the whole output is already bounded by the
        collector, and every line is still redacted.
        """
        text = strip_controls(collector.text(), keep_newlines=True)
        return chr(10).join(self._redact(line) for line in text.split(chr(10)))
