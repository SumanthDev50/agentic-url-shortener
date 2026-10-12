"""Tests for ``orchestrator.tools.sandbox``: env, argv grammars, bounded output, tree kill."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from orchestrator.core.errors import AgentCancelled, ConfigError
from orchestrator.core.fsutil import pid_alive
from orchestrator.core.types import CancelToken, CommandResult
from orchestrator.tools import sandbox
from orchestrator.tools.sandbox import sandbox_pytest_ini
from orchestrator.tools.sandbox import ALLOWED_MODULES, SandboxArgError, SandboxRunner

# A fake credential assembled at runtime from fragments: no secret-shaped literal in the repo.
FAKE_KEY = "AK" + "IA" + "X" * 16
_KEY_RE = re.compile("AK" + "IA" + "[A-Z0-9]{16}")


def mask(text: str) -> str:
    return _KEY_RE.sub("[REDACTED]", text)


@pytest.fixture
def runner(tmp_path: Path) -> SandboxRunner:
    return SandboxRunner(tmp_path / "scratch", redactor=mask)


def script(runner: SandboxRunner, source: str, name: str = "agent.py") -> tuple[Path, list[str]]:
    """A fresh scratch dir holding ``name`` (dedented ``source``) and the argv that runs it."""
    scratch = runner.new_scratch()
    (scratch / name).write_text(textwrap.dedent(source), encoding="utf-8")
    return scratch, runner.python_argv([name], scratch=scratch)


def run(runner: SandboxRunner, source: str, **kwargs: object):  # type: ignore[no-untyped-def]
    scratch, argv = script(runner, source)
    kwargs.setdefault("timeout_s", 60)
    return scratch, runner.run(argv, cwd=scratch, **kwargs)  # type: ignore[arg-type]


def wait_for(path: Path, *, seconds: float = 90.0) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return path.read_text(encoding="utf-8").strip()
        time.sleep(0.01)
    raise AssertionError(f"{path.name} never appeared")


def run_to_timeout(
    runner: SandboxRunner,
    source: str,
    *,
    starts: int,
    ready: Callable[[Path, CommandResult], bool],
) -> tuple[Path, CommandResult]:
    """Run ``source`` until its timeout fires.

    A timeout test must not depend on how fast this machine starts Python: the timeout is scaled
    from the measured start-up time and widened (a few times at most) if the child was killed
    before it could do the observable thing the test needs (``ready``).
    """
    started = time.monotonic()
    run(runner, "pass")
    timeout = (starts + 1) * (time.monotonic() - started) + 1.5
    for _ in range(4):
        scratch, result = run(runner, source, timeout_s=timeout)
        assert result.timed_out and not result.ok
        if ready(scratch, result):
            return scratch, result
        timeout *= 2
    pytest.fail("the child could not start within the timeout (machine too slow?)")


def wait_gone(pid: int, *, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.02)
    return not pid_alive(pid)


# A child that spawns a grandchild; each records its own pid, then both sleep.
TREE_SCRIPT = """
    import os, subprocess, sys, time
    from pathlib import Path

    GRANDCHILD = (
        "import os, time\\n"
        "from pathlib import Path\\n"
        "Path('gc.pid').write_text(str(os.getpid()))\\n"
        "time.sleep(300)\\n"
    )
    subprocess.Popen([sys.executable, "-c", GRANDCHILD])
    Path("child.pid").write_text(str(os.getpid()))
    time.sleep(300)
"""


# ------------------------------ environment


def test_environment_is_built_from_scratch_and_inherits_no_secret(
    runner: SandboxRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    canaries = {
        "FAKE_SERVICE_TOKEN": "inherited-canary-1",
        "AWS_SESSION_TOKEN": "inherited-canary-2",
        "GITHUB_TOKEN": "inherited-canary-3",
        "ANTHROPIC_API_KEY": "inherited-canary-4",
        "PYTHONSTARTUP": "inherited-canary-5",
        "PYTHONPATH": "inherited-canary-6",
    }
    for key, value in canaries.items():
        monkeypatch.setenv(key, value)
    scratch, result = run(
        runner,
        """
        import json, os
        print(json.dumps(dict(os.environ)))
        """,
    )
    assert result.ok, result.stderr
    child_env = json.loads(result.stdout)
    assert not set(canaries.values()) & set(child_env.values())
    assert not set(canaries) - {"PYTHONPATH"} & set(child_env)
    expected = runner.build_env(scratch)
    # only what Python itself adds on some platforms may be extra
    assert set(child_env) - set(expected) <= {"LC_CTYPE", "__PYVENV_LAUNCHER__"}
    assert child_env["PYTHONPATH"] == str(scratch)  # ours, not the inherited one
    for key in ("PYTHONUTF8", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE"):
        assert child_env[key] == expected[key]


def test_build_env_contents(runner: SandboxRunner, tmp_path: Path) -> None:
    scratch = runner.new_scratch()
    env = runner.build_env(scratch)
    assert env["PYTHONUTF8"] == "1" and env["PYTHONHASHSEED"] == "0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1" and env["PYTEST_ADDOPTS"] == ""
    assert env["PYTHONPATH"] == str(scratch)
    for key in ("HOME", "USERPROFILE", "TEMP", "TMP", "TMPDIR"):
        assert Path(env[key]).is_relative_to(scratch)
    assert str(Path(sys.executable).parent) in env["PATH"].split(os.pathsep)
    if sys.platform == "win32":
        assert env["SYSTEMROOT"] and env["WINDIR"] and env["COMSPEC"]
    else:
        assert "SYSTEMROOT" not in env


def test_build_env_ignores_the_parent_path_and_secrets(
    runner: SandboxRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evil_dir = str(tmp_path / "evil-bin")
    monkeypatch.setenv("PATH", evil_dir)
    monkeypatch.setenv("MY_SECRET_PASSWORD", "inherited-canary")
    env = runner.build_env(runner.new_scratch())
    assert evil_dir not in env["PATH"]
    assert "MY_SECRET_PASSWORD" not in env and "inherited-canary" not in "".join(env.values())


# ------------------------------ scratch management


def test_scratch_directories_are_unique_and_confined_to_the_root(runner: SandboxRunner) -> None:
    a, b = runner.new_scratch(), runner.new_scratch("tests")
    assert a != b and a.is_dir() and b.is_dir()
    assert a.is_relative_to(runner.scratch_root) and b.name.startswith("tests-")
    with pytest.raises(SandboxArgError):
        runner.new_scratch("../escape")
    with pytest.raises(SandboxArgError):
        runner.new_scratch("")


def test_discard_removes_only_scratch_directories(runner: SandboxRunner, tmp_path: Path) -> None:
    scratch = runner.new_scratch()
    (scratch / "deep" / "er").mkdir(parents=True)
    (scratch / "deep" / "er" / "f.txt").write_text("x", encoding="utf-8")
    runner.discard(scratch)
    assert not scratch.exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(SandboxArgError):
        runner.discard(outside)
    with pytest.raises(SandboxArgError):
        runner.discard(runner.scratch_root)
    assert outside.exists() and runner.scratch_root.exists()


# ------------------------------ pytest argv


def test_pytest_argv_is_the_fixed_command_line(runner: SandboxRunner) -> None:
    scratch = runner.new_scratch()
    assert runner.pytest_argv(scratch, ["tests", "tests/test_a.py"]) == [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-c",
        str(sandbox_pytest_ini()),
        "--rootdir",
        str(scratch),
        "--confcutdir",
        str(scratch),
        "-q",
        "-rfEsxXp",
        "--tb=short",
        "--show-capture=no",
        "--basetemp",
        str(scratch / ".tmp"),
        "--junitxml",
        str(scratch / "junit.xml"),
        "--",
        "tests",
        "tests/test_a.py",
    ]
    assert runner.pytest_argv(scratch, [])[-1] == "--"


@pytest.mark.parametrize("target", [".", "tests", "a_b-c/d.e", "tests/test_x.py"])
def test_pytest_argv_accepts_plain_relative_paths(runner: SandboxRunner, target: str) -> None:
    assert runner.pytest_argv(runner.new_scratch(), [target])[-1] == target


@pytest.mark.parametrize(
    "target",
    [
        "-p evil",
        "-pevil",
        "--rootdir=/",
        "--rootdir",
        "-x",
        "--",
        "@argfile",
        "@/etc/passwd",
        "/etc/passwd",
        "/",
        "C:/Windows",
        "C:\\Windows",
        "..",
        "../x",
        "a/../../b",
        "a/../b",
        "a//b",
        "tests/",
        "tests/x y.py",
        "tests\\x.py",
        "tests/*.py",
        "a;b",
        "a$b",
        "a`b`",
        "x\x00y",
        "x\ny",
        "",
        " ",
        "t" * 300,
        "tests/test_a.py::test_x",  # node ids are outside the fixed grammar
    ],
)
def test_pytest_argv_rejects_hostile_targets(runner: SandboxRunner, target: str) -> None:
    with pytest.raises(SandboxArgError):
        runner.pytest_argv(runner.new_scratch(), [target])


def test_pytest_argv_rejects_non_sequences_non_strings_and_foreign_scratch(
    runner: SandboxRunner, tmp_path: Path
) -> None:
    scratch = runner.new_scratch()
    for bad in ("tests", b"tests", 5, None, [5], [None], ["ok", ("x",)]):
        with pytest.raises(SandboxArgError):
            runner.pytest_argv(scratch, bad)  # type: ignore[arg-type]
    with pytest.raises(SandboxArgError):
        runner.pytest_argv(tmp_path, ["tests"])
    with pytest.raises(SandboxArgError):
        runner.pytest_argv(runner.scratch_root / ".." / "elsewhere", ["tests"])
    with pytest.raises(SandboxArgError):
        runner.pytest_argv(scratch, ["t"] * 100)


def test_pytest_argv_rejects_a_target_that_leaves_the_scratch_through_a_link(
    runner: SandboxRunner, tmp_path: Path
) -> None:
    from tests.orchestrator.tools.test_workspace import make_dir_link

    outside = tmp_path / "outside"
    outside.mkdir()
    scratch = runner.new_scratch()
    if not make_dir_link(scratch / "link", outside):
        pytest.skip("cannot create directory links (symlink / junction) here")
    with pytest.raises(SandboxArgError, match="escapes"):
        runner.pytest_argv(scratch, ["link/test_x.py"])


# ------------------------------ python argv


def test_python_argv_accepts_allow_listed_modules_and_scripts(runner: SandboxRunner) -> None:
    scratch = runner.new_scratch()
    assert runner.python_argv(["-m", "unittest"], scratch=scratch) == [
        sys.executable,
        "-m",
        "unittest",
    ]
    assert runner.python_argv(["-m", "py_compile", "a.py", "pkg/b.py"], scratch=scratch) == [
        sys.executable,
        "-m",
        "py_compile",
        "a.py",
        "pkg/b.py",
    ]
    assert runner.python_argv(["tools/gen.py", "data.json", "-v"], scratch=scratch) == [
        sys.executable,
        "tools/gen.py",
        "data.json",
        "-v",
    ]
    for module in ALLOWED_MODULES:
        assert runner.python_argv(["-m", module])[:3] == [sys.executable, "-m", module]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["-c", "print(1)"],
        ["-c"],
        ["-ic", "x"],
        ["-m"],
        ["-m", "pytest"],  # tests go through pytest_argv
        ["-m", "http.server"],
        ["-m", "pip", "install", "x"],
        ["-m", "unittest.mock"],
        ["-m", "../evil"],
        ["-m", "unittest", "-s", "tests"],  # options other than -v / -q
        ["-m", "unittest", "--failfast"],
        ["-m", "py_compile", "../x.py"],
        ["-m", "py_compile", "/etc/passwd"],
        ["-m", "py_compile", "@list"],
        ["-X", "dev", "a.py"],
        ["-W", "error", "a.py"],
        ["-E", "a.py"],
        ["-i"],
        ["-"],
        ["--version"],
        ["a.txt"],  # not a script
        ["a"],
        ["/abs/a.py"],
        ["../a.py"],
        ["a/../../a.py"],
        ["C:/x/a.py"],
        ["sub\\a.py"],
        ["a.py", "--evil"],
        ["a.py", "@args"],
        ["a.py", "../secret"],
        ["a.py", "x y"],
        ["a.py\x00"],
        [5],
    ],
)
def test_python_argv_rejects_everything_outside_the_grammar(
    runner: SandboxRunner, args: list[object]
) -> None:
    with pytest.raises(SandboxArgError):
        runner.python_argv(args, scratch=runner.new_scratch())  # type: ignore[arg-type]


def test_python_argv_script_must_resolve_inside_the_scratch(
    runner: SandboxRunner, tmp_path: Path
) -> None:
    with pytest.raises(SandboxArgError):
        runner.python_argv(["a.py"], scratch=tmp_path)  # a scratch outside the runner's root
    with pytest.raises(SandboxArgError):
        runner.python_argv("a.py")  # type: ignore[arg-type]


# ------------------------------ run(): validation


def test_run_only_executes_the_orchestrators_own_interpreter(runner: SandboxRunner) -> None:
    scratch = runner.new_scratch()
    for argv in (["cmd", "/c", "echo", "hi"], ["git", "status"], ["/bin/sh", "-c", "id"], []):
        with pytest.raises(SandboxArgError):
            runner.run(argv, cwd=scratch, timeout_s=5)


def test_run_requires_a_scratch_directory_as_cwd(runner: SandboxRunner, tmp_path: Path) -> None:
    argv = [sys.executable, "-V"]
    with pytest.raises(SandboxArgError):
        runner.run(argv, cwd=tmp_path, timeout_s=5)
    with pytest.raises(SandboxArgError):
        runner.run(argv, cwd=runner.scratch_root / "does-not-exist", timeout_s=5)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "5", None, 10**9])
def test_run_validates_the_timeout(runner: SandboxRunner, timeout: object) -> None:
    with pytest.raises(ConfigError, match="timeout_s"):
        runner.run([sys.executable, "-V"], cwd=runner.new_scratch(), timeout_s=timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize("cap", [0, 1, -5, True, 1.5, 10**9])
def test_run_validates_the_output_cap(runner: SandboxRunner, cap: object) -> None:
    with pytest.raises(ConfigError, match="max_output_bytes"):
        runner.run(
            [sys.executable, "-V"], cwd=runner.new_scratch(), timeout_s=5, max_output_bytes=cap
        )  # type: ignore[arg-type]


# ------------------------------ run(): behaviour


def test_run_captures_output_exit_code_and_metadata(runner: SandboxRunner) -> None:
    scratch, result = run(
        runner,
        """
        import os, sys
        print("out line")
        print("err line", file=sys.stderr)
        print(os.getcwd())
        sys.exit(3)
        """,
    )
    assert result.returncode == 3 and not result.ok
    assert "out line" in result.stdout and "err line" in result.stderr
    assert Path(result.stdout.splitlines()[-1]).resolve() == scratch
    assert not result.timed_out and not result.truncated
    assert result.command[0] == sys.executable and result.duration_s > 0


def test_run_gives_the_child_no_stdin_and_can_import_from_the_scratch(
    runner: SandboxRunner,
) -> None:
    scratch = runner.new_scratch()
    (scratch / "helper_mod.py").write_text("VALUE = 41\n", encoding="utf-8")
    (scratch / "agent.py").write_text(
        "import sys, helper_mod\nprint(repr(sys.stdin.read()), helper_mod.VALUE + 1)\n",
        encoding="utf-8",
    )
    result = runner.run(
        runner.python_argv(["agent.py"], scratch=scratch), cwd=scratch, timeout_s=60
    )
    assert result.ok, result.stderr
    assert result.stdout.strip() == "'' 42"


def test_output_is_control_stripped_then_redacted(runner: SandboxRunner) -> None:
    half = len(FAKE_KEY) // 2
    _, result = run(
        runner,
        f"""
        import sys
        esc = chr(27)
        print(esc + "[31mcolour" + esc + "[0m and " + {FAKE_KEY!r})
        print({FAKE_KEY[:half]!r} + esc + "[0m" + {FAKE_KEY[half:]!r})  # escape splits the key
        print("title" + esc + "]0;spoofed-title" + chr(7) + " end")
        print({FAKE_KEY!r}, file=sys.stderr)
        """,
    )
    assert result.ok, result.stderr
    assert FAKE_KEY not in result.stdout and FAKE_KEY not in result.stderr
    assert result.stdout.count("[REDACTED]") == 2  # the split key is redacted too
    assert "[REDACTED]" in result.stderr
    assert "\x1b" not in result.stdout and "\x07" not in result.stdout
    assert "spoofed-title" not in result.stdout
    assert "colour" in result.stdout and "title end" in result.stdout


def test_default_redactor_is_the_audit_redactor(tmp_path: Path) -> None:
    default_runner = SandboxRunner(tmp_path / "s")
    scratch = default_runner.new_scratch()
    (scratch / "agent.py").write_text(f"print({FAKE_KEY!r})\n", encoding="utf-8")
    result = default_runner.run(
        default_runner.python_argv(["agent.py"], scratch=scratch), cwd=scratch, timeout_s=60
    )
    assert result.ok, result.stderr
    assert FAKE_KEY not in result.stdout


# ------------------------------ bounded output


def test_collector_keeps_small_output_untouched() -> None:
    collector = sandbox._BoundedCollector(100)
    collector.feed(b"hello ")
    collector.feed(b"world\n")
    assert collector.text() == "hello world\n"
    assert not collector.truncated and collector.total == 12


def test_collector_keeps_head_and_tail_on_line_boundaries() -> None:
    collector = sandbox._BoundedCollector(100)
    lines = [f"line-{i:04d}\n".encode() for i in range(500)]
    for line in lines:
        collector.feed(line)
    text = collector.text()
    assert collector.truncated and collector.total == sum(map(len, lines))
    assert text.startswith("line-0000\n") and text.endswith("line-0499\n")
    assert "output truncated" in text
    kept = [ln for ln in text.splitlines() if ln.startswith("line-")]
    assert all(len(ln) == 9 for ln in kept)  # whole lines only: a secret is never cut in half
    assert len(text) < 200


def test_collector_memory_is_bounded_regardless_of_input_volume() -> None:
    cap = 1000
    collector = sandbox._BoundedCollector(cap)
    chunk = b"x" * 4096
    for _ in range(2000):  # 8 MB through a 1 KB collector
        collector.feed(chunk)
        assert len(collector._head) + len(collector._tail) <= cap + 4096
    assert collector.truncated and collector.total == 2000 * 4096
    assert len(collector.text()) < cap + 200


def test_collector_without_any_newline_drops_the_ragged_edges() -> None:
    collector = sandbox._BoundedCollector(20)
    collector.feed(b"a" * 1000)
    text = collector.text()
    assert "aaa" not in text and "1000 bytes omitted" in text


def test_a_chatty_child_is_truncated_while_it_runs(runner: SandboxRunner) -> None:
    _, result = run(
        runner,
        """
        for i in range(20000):
            print("noisy line", i)
        print("the very last line")
        """,
        max_output_bytes=2000,
    )
    assert result.ok, result.stderr
    assert result.truncated
    assert len(result.stdout) < 2500
    assert result.stdout.startswith("noisy line 0\n") and result.stdout.rstrip().endswith(
        "the very last line"
    )
    assert "output truncated" in result.stdout


@pytest.mark.slow
def test_fifty_megabytes_of_output_cannot_exhaust_memory(runner: SandboxRunner) -> None:
    _, result = run(
        runner,
        """
        import sys
        line = "x" * 1023 + chr(10)
        for _ in range(50 * 1024):
            sys.stdout.write(line)
        """,
        timeout_s=120,
    )
    assert result.ok, result.stderr
    assert result.truncated
    assert (
        len(result.stdout) <= sandbox.DEFAULT_MAX_OUTPUT_BYTES + 200
    )  # 256 KiB + marker, not 50 MB


# ------------------------------ timeout / cancel


@pytest.mark.slow
def test_timeout_kills_the_whole_process_tree(runner: SandboxRunner) -> None:
    scratch, _ = run_to_timeout(
        runner,
        TREE_SCRIPT,
        starts=2,
        ready=lambda scratch, _result: (
            (scratch / "gc.pid").exists() and (scratch / "child.pid").exists()
        ),
    )
    child, grandchild = (
        int((scratch / name).read_text(encoding="utf-8")) for name in ("child.pid", "gc.pid")
    )
    assert wait_gone(child), "child survived the timeout"
    assert wait_gone(grandchild), "grandchild survived the timeout"


def test_cancel_mid_run_kills_the_whole_process_tree(runner: SandboxRunner) -> None:
    scratch, argv = script(runner, TREE_SCRIPT)
    token = CancelToken()
    outcome: list[BaseException] = []

    def target() -> None:
        try:
            runner.run(argv, cwd=scratch, timeout_s=120, cancel=token)
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    child = int(wait_for(scratch / "child.pid"))
    grandchild = int(wait_for(scratch / "gc.pid"))
    assert pid_alive(child) and pid_alive(grandchild)
    token.cancel("test cancel")
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], AgentCancelled)
    assert wait_gone(child), "child survived the cancel"
    assert wait_gone(grandchild), "grandchild survived the cancel"


def test_a_token_cancelled_before_the_run_prevents_the_spawn(runner: SandboxRunner) -> None:
    scratch, argv = script(runner, "from pathlib import Path\nPath('ran.marker').write_text('x')\n")
    token = CancelToken()
    token.cancel()
    with pytest.raises(AgentCancelled):
        runner.run(argv, cwd=scratch, timeout_s=30, cancel=token)
    assert not (scratch / "ran.marker").exists()


def test_a_cancel_after_the_run_finished_never_signals_a_process(
    runner: SandboxRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token keeps its callbacks: a late cancel must not hit a recycled PID."""
    token = CancelToken()
    _, result = run(runner, "print('done')", cancel=token)
    assert result.ok
    killed: list[int] = []
    monkeypatch.setattr(sandbox, "_kill_process_tree", lambda proc: killed.append(proc.pid))
    token.cancel("late")
    assert killed == []


def test_timeout_is_reported_not_raised_and_other_output_is_kept(runner: SandboxRunner) -> None:
    _, result = run_to_timeout(
        runner,
        """
        import sys, time
        print("before the hang", flush=True)
        time.sleep(300)
        """,
        starts=1,
        ready=lambda _scratch, result: "before the hang" in result.stdout,
    )
    assert "before the hang" in result.stdout
    assert result.duration_s < 60


# ------------------------------ real pytest through the runner


@pytest.mark.slow
def test_pytest_runs_in_the_sandbox_with_the_fixed_argv(runner: SandboxRunner) -> None:
    scratch = runner.new_scratch()
    (scratch / "tests").mkdir()
    (scratch / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    (scratch / "tests" / "test_bad.py").write_text(
        "def test_bad():\n    assert False\n", encoding="utf-8"
    )
    ok = runner.run(runner.pytest_argv(scratch, ["tests/test_ok.py"]), cwd=scratch, timeout_s=120)
    assert ok.ok, ok.stdout + ok.stderr
    assert "1 passed" in ok.stdout
    assert (scratch / "junit.xml").exists()
    assert not (scratch / ".pytest_cache").exists()  # cacheprovider is disabled
    bad = runner.run(runner.pytest_argv(scratch, ["tests"]), cwd=scratch, timeout_s=120)
    assert bad.returncode == 1 and "1 failed" in bad.stdout


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows cannot find orphans of an exited parent"
)
def test_orphans_of_a_normally_exiting_child_do_not_leak_on_posix(runner: SandboxRunner) -> None:
    scratch, result = run(
        runner,
        """
        import os, subprocess, sys
        from pathlib import Path
        code = (
            "import os, time; from pathlib import Path; "
            "Path('orphan.pid').write_text(str(os.getpid())); time.sleep(300)"
        )
        subprocess.Popen([sys.executable, "-c", code])
        while not Path("orphan.pid").exists():
            pass
        """,
        timeout_s=60,
    )
    assert result.ok
    assert wait_gone(int((scratch / "orphan.pid").read_text(encoding="utf-8")))


def test_popen_kwargs_are_platform_specific() -> None:
    kwargs = sandbox._popen_kwargs()
    if sys.platform == "win32":
        assert kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs == {"start_new_session": True}
