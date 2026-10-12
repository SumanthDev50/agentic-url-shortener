"""Kill switch (fail-closed file switch) and safe-stop budgets / circuit breaker."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import stat
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import ConfigError, StateError
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.state import Counters, NodeState, RunState
from orchestrator.core.types import (
    Epoch,
    FailureKind,
    KillSignal,
    NodeSpec,
    NodeStatus,
    Stage,
)
from orchestrator.engine import safestop
from orchestrator.engine.config import EngineConfig
from orchestrator.engine.safestop import (
    MAX_KILL_BYTES,
    KillClearance,
    KillSwitch,
    SafeStopController,
    SafeStopTrigger,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Durability is not under test and fsync dominates the run time on Windows."""
    monkeypatch.setattr(os, "fsync", lambda fd: None)


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    return tmp_path / "runs" / "KILL", tmp_path / "runs" / "r1" / "KILL"


@pytest.fixture
def switch(paths: tuple[Path, Path]) -> KillSwitch:
    return KillSwitch(paths[0], paths[1], clock=FakeClock())


def run_file(paths: tuple[Path, Path]) -> Path:
    return paths[1]


def symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not permitted in this environment")


def has_control_characters(text: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in text)


# =============================================================================
# KillSwitch: tripped / not tripped
# =============================================================================


def test_untripped_switch_reports_nothing(switch: KillSwitch) -> None:
    assert switch.check() is None
    assert switch.signals() == []


def test_missing_directories_are_not_tripped(tmp_path: Path) -> None:
    ks = KillSwitch(tmp_path / "no" / "KILL", tmp_path / "no" / "such" / "run" / "KILL")
    assert ks.check() is None


def test_existence_alone_trips_even_with_empty_contents(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    run_file(paths).write_bytes(b"")
    signal = switch.check()
    assert isinstance(signal, KillSignal)
    assert (signal.scope, signal.by) == ("run", "unknown")
    assert signal.reason  # a default text, never empty
    assert signal.ts.tzinfo is not None


def test_global_switch_wins_and_signals_lists_both(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    paths[0].write_text("global stop")
    paths[1].write_text("run stop")
    assert switch.check() is not None
    assert switch.check().scope == "global"  # type: ignore[union-attr]
    assert [s.scope for s in switch.signals()] == ["global", "run"]


def test_switch_without_global_path_only_checks_the_run_file(
    paths: tuple[Path, Path],
) -> None:
    paths[0].write_text("global stop")  # not configured: ignored
    ks = KillSwitch(None, paths[1])
    assert ks.check() is None
    paths[1].write_text("x")
    assert ks.check() is not None
    assert ks.check().scope == "run"  # type: ignore[union-attr]


def test_paths_may_be_given_as_strings(paths: tuple[Path, Path]) -> None:
    ks = KillSwitch(str(paths[0]), str(paths[1]))
    paths[1].write_text("x")
    assert ks.check() is not None


# =============================================================================
# KillSwitch: trip
# =============================================================================


def test_trip_is_attributed_atomic_and_round_trips(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    signal = switch.trip("run", "alice", "stop now")
    assert signal == KillSignal("alice", "stop now", T0, "run")
    assert switch.check() == signal
    document = json.loads(run_file(paths).read_text(encoding="ascii"))
    assert document == {"by": "alice", "reason": "stop now", "scope": "run", "ts": T0.isoformat()}
    assert len(run_file(paths).read_bytes()) <= MAX_KILL_BYTES
    assert sorted(p.name for p in run_file(paths).parent.iterdir()) == ["KILL"]  # no temp files


def test_trip_global_scope(switch: KillSwitch, paths: tuple[Path, Path]) -> None:
    signal = switch.trip("global", "ops", "maintenance")
    assert signal.scope == "global"
    assert paths[0].exists() and not paths[1].exists()
    assert switch.check() == signal


def test_trip_creates_missing_directories(tmp_path: Path) -> None:
    ks = KillSwitch(None, tmp_path / "fresh" / "run" / "KILL", clock=FakeClock())
    ks.trip("run", "alice", "x")
    assert ks.check() is not None


def test_trip_sanitises_actor_and_reason(switch: KillSwitch, paths: tuple[Path, Path]) -> None:
    signal = switch.trip("run", "al\x1b[31mice\n", "stop\r\n\x1b]0;evil\x07 now\x00")
    assert signal.by == "alice"
    assert not has_control_characters(signal.by + signal.reason)
    assert b"\x1b" not in run_file(paths).read_bytes()
    assert switch.check() == signal


def test_trip_requires_a_named_actor_and_a_known_scope(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    with pytest.raises(ConfigError, match="named actor"):
        switch.trip("run", " \x1b[0m ", "reason")
    with pytest.raises(ConfigError, match="scope"):
        switch.trip("galaxy", "alice", "reason")
    with pytest.raises(ConfigError, match="global"):
        KillSwitch(None, paths[1]).trip("global", "alice", "reason")
    assert not run_file(paths).exists()


def test_trip_keeps_the_file_within_one_kib_even_with_wide_characters(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    signal = switch.trip("run", "\U0001f600" * 100, "\U0001f600" * 500)
    assert len(run_file(paths).read_bytes()) <= MAX_KILL_BYTES
    assert signal.reason
    assert switch.check() == signal  # what trip returns is exactly what check reads back


def test_trip_without_reason_records_a_placeholder(switch: KillSwitch) -> None:
    assert switch.trip("run", "alice", "  ").reason == "no reason given"


def test_trip_over_a_directory_fails_loudly(
    switch: KillSwitch, paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "sleep", lambda seconds: None)  # skip fsutil's Windows retry back-off
    run_file(paths).mkdir()
    with pytest.raises(StateError, match="could not trip"):
        switch.trip("run", "alice", "x")
    assert run_file(paths).is_dir()
    assert sorted(p.name for p in run_file(paths).parent.iterdir()) == ["KILL"]  # temp cleaned up


def test_trip_replaces_a_planted_symlink_instead_of_writing_through(
    switch: KillSwitch, paths: tuple[Path, Path], tmp_path: Path
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    symlink_or_skip(run_file(paths), victim)
    switch.trip("run", "alice", "x")
    assert victim.read_text() == "precious"
    assert not run_file(paths).is_symlink()
    assert switch.check() is not None


# =============================================================================
# KillSwitch: contents are advisory and hostile
# =============================================================================


def test_free_text_and_binary_contents_are_shown_sanitised(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    run_file(paths).write_bytes(b"STOP\x1b[2J NOW\x00\x07")
    signal = switch.check()
    assert signal is not None and signal.by == "unknown"
    assert signal.reason == "STOP NOW"
    run_file(paths).write_bytes(b"\xff\xfe\x00\x01 garbage \x80")
    signal = switch.check()
    assert signal is not None and not has_control_characters(signal.reason)


def test_json_that_is_not_an_object_is_shown_as_text(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    run_file(paths).write_text('["halt", 3]')
    signal = switch.check()
    assert signal is not None and signal.by == "unknown" and "halt" in signal.reason


def test_json_object_with_wrong_types_falls_back_to_defaults(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    run_file(paths).write_text(json.dumps({"by": 5, "reason": ["x"], "ts": 7}))
    signal = switch.check()
    assert signal is not None and signal.by == "unknown" and signal.reason


def test_scope_inside_the_file_cannot_spoof_the_scope(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    run_file(paths).write_text(json.dumps({"by": "mallory", "reason": "x", "scope": "global"}))
    signal = switch.check()
    assert signal is not None and signal.scope == "run" and signal.by == "mallory"


def test_signal_identity_is_stable_for_an_unchanged_file(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    run_file(paths).write_text("plain text")  # no ts inside: the file's mtime is used
    first, second = switch.check(), switch.check()
    assert first is not None and first == second
    run_file(paths).write_text(json.dumps({"by": "a", "reason": "b", "ts": "2026-02-03T04:05:06Z"}))
    assert switch.check().ts == datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC)  # type: ignore[union-attr]
    run_file(paths).write_text(json.dumps({"by": "a", "reason": "b", "ts": "not a time"}))
    assert switch.check().ts.tzinfo is not None  # type: ignore[union-attr]


def test_oversized_file_is_tripped_and_its_contents_are_ignored(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    head = json.dumps({"by": "alice", "reason": ""})
    exact = json.dumps({"by": "alice", "reason": "r" * (MAX_KILL_BYTES - len(head))})
    assert len(exact.encode()) == MAX_KILL_BYTES
    run_file(paths).write_text(exact)
    assert switch.check().by == "alice"  # type: ignore[union-attr]  # exactly 1 KiB is fine
    run_file(paths).write_text(exact + " ")
    signal = switch.check()
    assert signal is not None and signal.by == "unknown"
    assert "larger than" in signal.reason


def test_huge_file_is_read_in_a_bounded_way(
    switch: KillSwitch, paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with run_file(paths).open("wb") as fh:
        fh.truncate(5 * 1024 * 1024)
    requested: list[int] = []
    real_read = os.read

    def spy(fd: int, n: int) -> bytes:
        requested.append(n)
        return real_read(fd, n)

    with monkeypatch.context() as scope:
        scope.setattr(os, "read", spy)
        signal = switch.check()
    assert signal is not None and "larger than" in signal.reason
    assert requested and sum(requested) <= MAX_KILL_BYTES + 1


def test_arbitrary_contents_never_break_check(switch: KillSwitch, paths: tuple[Path, Path]) -> None:
    rng = random.Random(7)
    for size in (1, 2, 17, 255, 1023, 1024, 1025, 3000):
        run_file(paths).write_bytes(bytes(rng.randrange(256) for _ in range(size)))
        signal = switch.check()
        assert signal is not None
        assert not has_control_characters(signal.by + signal.reason)


# =============================================================================
# KillSwitch: fail closed on anything but "not found"
# =============================================================================


def test_directory_in_place_of_the_file_is_tripped_and_not_cleared(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    run_file(paths).mkdir()
    (run_file(paths) / "inner").write_text("x")
    signal = switch.check()
    assert signal is not None and "not a regular file" in signal.reason
    with pytest.raises(StateError, match="not a regular file"):
        switch.clear("run", "alice", "cleaning up")
    assert run_file(paths).is_dir() and switch.check() is not None  # still tripped


def test_unreadable_file_is_tripped(
    switch: KillSwitch, paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_file(paths).write_text("x")

    def deny(path: Path, inspected: os.stat_result) -> bytes:
        raise PermissionError("denied")

    monkeypatch.setattr(safestop, "_read_head", deny)
    signal = switch.check()
    assert signal is not None and "could not be read" in signal.reason
    assert "PermissionError" in signal.reason and "denied" not in signal.reason


@pytest.mark.parametrize("error", [PermissionError("p"), OSError(5, "io error"), ValueError("nul")])
def test_inspection_errors_other_than_not_found_are_tripped(
    switch: KillSwitch, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def broken(path: Path) -> os.stat_result:
        raise error

    monkeypatch.setattr(safestop, "_lstat", broken)
    signal = switch.check()
    assert signal is not None and "could not be inspected" in signal.reason


def test_not_found_while_inspecting_or_opening_means_not_tripped(
    switch: KillSwitch, paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def gone(path: Path) -> os.stat_result:
        raise FileNotFoundError

    monkeypatch.setattr(safestop, "_lstat", gone)
    assert switch.check() is None
    monkeypatch.undo()
    run_file(paths).write_text("x")

    def vanished(path: Path, inspected: os.stat_result) -> bytes:
        raise FileNotFoundError

    monkeypatch.setattr(safestop, "_read_head", vanished)  # cleared between lstat and open
    assert switch.check() is None


def test_path_with_an_embedded_nul_is_tripped_not_a_crash(tmp_path: Path) -> None:
    ks = KillSwitch(None, tmp_path / "bad\x00name" / "KILL")
    signal = ks.check()
    assert signal is not None and "could not be inspected" in signal.reason


def test_file_swapped_after_the_inspection_is_tripped_not_trusted(
    switch: KillSwitch, paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_file(paths).write_text(json.dumps({"by": "alice", "reason": "x"}))
    real_lstat = safestop._lstat
    assert switch.check().by == "alice"  # type: ignore[union-attr]  # the genuine file is read

    def other_file(path: Path) -> os.stat_result:
        genuine = real_lstat(path)
        fields = list(genuine)
        fields[1] = genuine.st_ino + 1000  # as if the inspection saw another file index
        return os.stat_result(fields)

    monkeypatch.setattr(safestop, "_lstat", other_file)
    signal = switch.check()
    assert signal is not None and signal.by == "unknown" and "changed while" in signal.reason


def test_file_that_stops_being_regular_after_the_inspection_is_tripped(
    switch: KillSwitch, paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_file(paths).write_text("x")
    monkeypatch.setattr(safestop, "_read_head", lambda path, inspected: None)
    signal = switch.check()
    assert signal is not None and "changed while" in signal.reason


def test_links_are_tripped_without_being_read(
    switch: KillSwitch, paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def never_read(path: Path, inspected: os.stat_result) -> bytes:
        raise AssertionError("a link must not be read")

    monkeypatch.setattr(safestop, "_read_head", never_read)
    symlink_mode = types.SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_mtime=0.0)
    reparse = types.SimpleNamespace(
        st_mode=stat.S_IFREG | 0o644, st_mtime=0.0, st_file_attributes=0x400
    )
    for fake in (symlink_mode, reparse):
        monkeypatch.setattr(safestop, "_lstat", lambda path, fake=fake: fake)
        signal = switch.check()
        assert signal is not None and "not followed" in signal.reason


def test_real_symlink_is_not_followed(
    switch: KillSwitch, paths: tuple[Path, Path], tmp_path: Path
) -> None:
    secret = "target-contents-" + "x" * 8
    target = tmp_path / "elsewhere.txt"
    target.write_text(json.dumps({"by": "alice", "reason": secret}))
    symlink_or_skip(run_file(paths), target)
    signal = switch.check()
    assert signal is not None and signal.by == "unknown"
    assert secret not in signal.reason and "not followed" in signal.reason


# =============================================================================
# KillSwitch: clear is attributed
# =============================================================================


def test_clear_is_attributed_and_returns_what_was_cleared(
    switch: KillSwitch, paths: tuple[Path, Path]
) -> None:
    tripped = switch.trip("run", "alice", "stop now")
    cleared = switch.clear("run", "bob", "false alarm")
    assert cleared == KillClearance(tripped, "bob", "false alarm", T0)
    assert not run_file(paths).exists()
    assert switch.check() is None


def test_clear_without_an_actor_is_refused_and_keeps_the_switch_tripped(
    switch: KillSwitch,
) -> None:
    switch.trip("run", "alice", "stop now")
    with pytest.raises(ConfigError, match="attributed"):
        switch.clear("run", "  ", "reason")
    assert switch.check() is not None


def test_clear_of_an_untripped_switch_and_of_an_unknown_scope(switch: KillSwitch) -> None:
    assert switch.clear("run", "bob") is None
    with pytest.raises(ConfigError, match="scope"):
        switch.clear("galaxy", "bob")


def test_clear_only_touches_the_requested_scope(switch: KillSwitch) -> None:
    switch.trip("global", "ops", "x")
    switch.trip("run", "alice", "y")
    assert switch.clear("run", "bob") is not None
    assert [s.scope for s in switch.signals()] == ["global"]


def test_clear_removes_a_hostile_file_too(switch: KillSwitch, paths: tuple[Path, Path]) -> None:
    run_file(paths).write_bytes(b"\xff" * 4096)  # oversized garbage is still just a file
    cleared = switch.clear("run", "bob", "reset")
    assert cleared is not None and cleared.signal.by == "unknown"
    assert switch.check() is None


# =============================================================================
# SafeStopController: budgets
# =============================================================================


def make_state(**kwargs: Any) -> RunState:
    return RunState("r1", "scenario", "tester", T0, "trace-1", **kwargs)


def controller(**limits: Any) -> SafeStopController:
    return SafeStopController(EngineConfig(**limits))


def test_fresh_run_has_no_trigger() -> None:
    assert controller().check(make_state(), 0.0) is None


def test_trigger_is_an_immutable_value() -> None:
    trigger = SafeStopTrigger("budget", "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        trigger.reason = "y"  # type: ignore[misc]


def test_attempt_budget_is_a_maximum() -> None:
    ctl, state = controller(max_total_attempts=3), make_state()
    state.counters.budgeted_attempts = 3
    assert ctl.check(state, 0.0) is None  # three retries/fallbacks are allowed
    state.counters.budgeted_attempts = 4
    trigger = ctl.check(state, 0.0)
    assert trigger is not None and trigger.trigger == "budget"
    assert "attempt budget" in trigger.reason and "4 used, limit 3" in trigger.reason


@pytest.mark.parametrize(
    ("limit", "used", "expected"), [(0, 0, False), (0, 1, True), (2, 2, False)]
)
def test_replan_budget(limit: int, used: int, expected: bool) -> None:
    state = make_state()
    state.counters.replans = used
    trigger = controller(max_replans=limit).check(state, 0.0)
    assert (trigger is not None) is expected
    if trigger is not None:
        assert trigger.trigger == "budget" and "re-plan" in trigger.reason


def test_active_runtime_budget_uses_the_larger_of_live_and_persisted() -> None:
    ctl, state = controller(max_active_runtime_s=100.0), make_state()
    assert ctl.check(state, 100.0) is None
    trigger = ctl.check(state, 100.5)
    assert trigger is not None and trigger.trigger == "budget" and "runtime" in trigger.reason
    state.active_runtime_s = 250.0  # persisted accrual still counts if the live value is smaller
    assert ctl.check(state, 1.0) is not None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "soon", None])
def test_non_finite_runtime_counts_as_exhausted(bad: Any) -> None:
    trigger = controller().check(make_state(), bad)
    assert trigger is not None and "not a finite number" in trigger.reason


@pytest.mark.parametrize(
    ("cap", "zombies", "expected"), [(2, 2, False), (2, 3, True), (0, 0, False), (0, 1, True)]
)
def test_zombie_cap(cap: int, zombies: int, expected: bool) -> None:
    state = make_state()
    state.counters.zombies = zombies
    trigger = controller(max_zombie_threads=cap).check(state, 0.0)
    assert (trigger is not None) is expected
    if trigger is not None:
        assert trigger.trigger == "worker_exhaustion"


def zombie_state(zombies: int, *statuses: NodeStatus) -> RunState:
    state = make_state()
    state.counters.zombies = zombies
    state.nodes = {f"n{i}": NodeState(f"n{i}", status) for i, status in enumerate(statuses)}
    return state


def test_zombies_at_the_cap_stop_the_run_only_when_nothing_can_progress() -> None:
    ctl = controller(max_zombie_threads=2)
    stuck = ctl.check(zombie_state(2, NodeStatus.READY, NodeStatus.PENDING), 0.0)
    assert stuck == SafeStopTrigger(
        "worker_exhaustion",
        "2 zombie attempt thread(s) fill the cap and no other work can progress",
    )
    # dispatch is blocked at the cap, but running or human-owned work can still finish
    for busy in (NodeStatus.RUNNING, NodeStatus.AWAITING_APPROVAL, NodeStatus.AWAITING_INPUT):
        assert ctl.check(zombie_state(2, NodeStatus.READY, busy), 0.0) is None
    assert ctl.check(zombie_state(1, NodeStatus.READY), 0.0) is None  # below the cap
    assert ctl.check(zombie_state(2, NodeStatus.SUCCEEDED, NodeStatus.PENDING), 0.0) is None


def test_a_zero_zombie_cap_does_not_stop_a_run_that_has_no_zombies() -> None:
    ctl = controller(max_zombie_threads=0)
    assert ctl.check(zombie_state(0, NodeStatus.READY), 0.0) is None
    assert ctl.check(zombie_state(1, NodeStatus.RUNNING), 0.0) is not None  # over the cap


def test_human_inputs_are_refused_not_safe_stopped() -> None:
    ctl, state = controller(max_human_inputs=2), make_state()
    assert ctl.human_input_allowed(state)
    state.counters.human_inputs = 1
    assert ctl.human_input_allowed(state)
    state.counters.human_inputs = 2
    assert not ctl.human_input_allowed(state)
    state.counters.human_inputs = 50
    assert ctl.check(state, 0.0) is None  # exhausted human inputs never stop the run
    assert not controller(max_human_inputs=0).human_input_allowed(make_state())


def test_budgets_are_reported_in_a_fixed_order() -> None:
    ctl = controller(
        max_total_attempts=1, max_replans=0, max_active_runtime_s=10.0, max_zombie_threads=0
    )
    state = make_state()
    state.counters.budgeted_attempts, state.counters.replans = 2, 1
    state.counters.zombies, state.active_runtime_s = 1, 99.0
    seen = []
    for reset in (
        lambda: setattr(state.counters, "budgeted_attempts", 0),
        lambda: setattr(state.counters, "replans", 0),
        lambda: setattr(state, "active_runtime_s", 0.0),
        lambda: setattr(state.counters, "zombies", 0),
    ):
        trigger = ctl.check(state, 0.0)
        assert trigger is not None
        seen.append(trigger.reason.split(" ")[0])
        reset()
    assert ctl.check(state, 0.0) is None
    assert seen == ["retry/fallback", "re-plan", "active", "1"]


# =============================================================================
# SafeStopController: circuit breaker
# =============================================================================

COUNTED = [FailureKind.TRANSIENT, FailureKind.TIMEOUT, FailureKind.FATAL]
UNCOUNTED = [
    FailureKind.GATE_REJECTED,
    FailureKind.POLICY_VIOLATION,
    FailureKind.APPROVAL_REJECTED,
    FailureKind.AUTONOMY_VIOLATION,
    FailureKind.CANCELLED,
]


def test_breaker_counts_only_transient_timeout_and_fatal_failures() -> None:
    assert frozenset(COUNTED) == safestop.BREAKER_FAILURE_KINDS
    ctl, state = controller(circuit_breaker_failures=2), make_state()
    for kind in COUNTED:
        ctl.record_attempt(state, f"agent-{kind.value}", kind)
        assert state.counters.breaker_failures[f"agent-{kind.value}"] == 1
    for kind in UNCOUNTED:
        ctl.record_attempt(state, "agent-x", kind)
    assert "agent-x" not in state.counters.breaker_failures
    assert state.counters.breaker_open == []


def test_uncounted_failures_neither_count_nor_reset() -> None:
    ctl, state = controller(circuit_breaker_failures=3), make_state()
    ctl.record_attempt(state, "a", FailureKind.TRANSIENT)
    ctl.record_attempt(state, "a", FailureKind.GATE_REJECTED)
    ctl.record_attempt(state, "a", FailureKind.TIMEOUT)
    assert state.counters.breaker_failures == {"a": 2}


def test_breaker_opens_at_the_threshold_and_success_resets_it() -> None:
    ctl, state = controller(circuit_breaker_failures=3), make_state()
    assert [ctl.record_attempt(state, "a", FailureKind.TRANSIENT) for _ in range(3)] == [
        False,
        False,
        True,  # reports that this call opened the breaker
    ]
    assert ctl.breaker_open(state, "a") and state.counters.breaker_open == ["a"]
    assert ctl.record_attempt(state, "a", FailureKind.FATAL) is False  # already open: listed once
    assert state.counters.breaker_open == ["a"]
    ctl.record_attempt(state, "a", None)  # success
    assert not ctl.breaker_open(state, "a")
    assert state.counters.breaker_failures == {} and state.counters.breaker_open == []
    ctl.record_attempt(state, "a", FailureKind.TRANSIENT)  # the count starts from zero again
    assert state.counters.breaker_failures == {"a": 1}


def test_breaker_is_per_agent() -> None:
    ctl, state = controller(circuit_breaker_failures=2), make_state()
    for _ in range(2):
        ctl.record_attempt(state, "a", FailureKind.TIMEOUT)
    ctl.record_attempt(state, "b", FailureKind.TIMEOUT)
    assert ctl.breaker_open(state, "a") and not ctl.breaker_open(state, "b")
    ctl.record_attempt(state, "b", None)  # b's success does not close a's breaker
    assert ctl.breaker_open(state, "a")


def test_string_and_unknown_kinds_and_bad_agent_ids() -> None:
    ctl, state = controller(circuit_breaker_failures=3), make_state()
    ctl.record_attempt(state, "a", "transient")  # type: ignore[arg-type]
    ctl.record_attempt(state, "a", "mystery")  # type: ignore[arg-type]  # unknown => FATAL
    ctl.record_attempt(state, "a", "gate_rejected")  # type: ignore[arg-type]
    assert state.counters.breaker_failures == {"a": 2}
    for bad in ("", None, 5):
        with pytest.raises(ConfigError, match="agent_id"):
            ctl.record_attempt(state, bad, FailureKind.FATAL)  # type: ignore[arg-type]


def test_breaker_open_also_derives_from_the_counts() -> None:
    ctl, state = controller(circuit_breaker_failures=2), make_state()
    state.counters.breaker_failures = {"a": 2, "b": 1}  # e.g. a lower threshold after a resume
    assert ctl.breaker_open(state, "a") and not ctl.breaker_open(state, "b")
    state.counters.breaker_open = ["c"]
    assert ctl.breaker_open(state, "c")


def test_reset_breaker_for_one_agent_or_all() -> None:
    ctl, state = controller(circuit_breaker_failures=1), make_state()
    for agent in ("a", "b"):
        ctl.record_attempt(state, agent, FailureKind.FATAL)
    ctl.reset_breaker(state, "a")
    assert not ctl.breaker_open(state, "a") and ctl.breaker_open(state, "b")
    ctl.reset_breaker(state)
    assert state.counters.breaker_failures == {} and state.counters.breaker_open == []


def test_breaker_state_lives_in_the_run_state_and_survives_a_round_trip() -> None:
    ctl, state = controller(circuit_breaker_failures=2), make_state()
    for _ in range(2):
        ctl.record_attempt(state, "a", FailureKind.TRANSIENT)
    ctl.record_attempt(state, "b", FailureKind.TIMEOUT)
    restored = RunState.from_dict(state.to_dict())
    assert (
        restored.counters.to_dict()
        == Counters(breaker_failures={"a": 2, "b": 1}, breaker_open=["a"]).to_dict()
    )
    assert controller(circuit_breaker_failures=2).breaker_open(restored, "a")


def graph_document(*nodes: tuple[str, str, str | None]) -> dict[str, Any]:
    return {
        "schema": 1,
        "nodes": [{"id": n, "agent": a, "fallback_agent": f} for n, a, f in nodes],
    }


def breaker_state(
    statuses: dict[str, NodeStatus],
    nodes: tuple[tuple[str, str, str | None], ...],
    opened: list[str],
    epochs: dict[str, Epoch] | None = None,
) -> RunState:
    state = make_state(graph=graph_document(*nodes))
    state.nodes = {
        node_id: NodeState(node_id, status, epoch=(epochs or {}).get(node_id, Epoch.PRIMARY))
        for node_id, status in statuses.items()
    }
    state.counters.breaker_open = list(opened)
    return state


def test_open_breaker_without_a_fallback_stops_the_run() -> None:
    state = breaker_state(
        {"n1": NodeStatus.READY, "n2": NodeStatus.READY},
        (("n1", "agent-a", None), ("n2", "agent-b", "agent-c")),
        ["agent-a"],
    )
    trigger = controller().check(state, 0.0)
    assert trigger == SafeStopTrigger(
        "circuit_breaker",
        "circuit breaker open for agent(s) agent-a with no usable fallback (nodes: n1)",
    )


def test_open_breaker_with_a_usable_fallback_routes_instead_of_stopping() -> None:
    state = breaker_state({"n": NodeStatus.READY}, (("n", "agent-a", "agent-b"),), ["agent-a"])
    assert controller().check(state, 0.0) is None
    state.counters.breaker_open.append("agent-b")  # the fallback is dead too
    trigger = controller().check(state, 0.0)
    assert trigger is not None and trigger.trigger == "circuit_breaker"


def test_node_already_on_its_fallback_stops_when_that_agent_opens() -> None:
    state = breaker_state(
        {"n": NodeStatus.READY},
        (("n", "agent-a", "agent-b"),),
        ["agent-b"],
        epochs={"n": Epoch.FALLBACK},
    )
    trigger = controller().check(state, 0.0)
    assert trigger is not None and "agent-b" in trigger.reason
    state.counters.breaker_open = ["agent-a"]  # the primary is dead but this node left it
    assert controller().check(state, 0.0) is None


@pytest.mark.parametrize(
    "status",
    [s for s in NodeStatus if s != NodeStatus.READY],
)
def test_only_ready_nodes_can_be_blocked_by_a_breaker(status: NodeStatus) -> None:
    state = breaker_state({"n": status}, (("n", "agent-a", None),), ["agent-a"])
    assert controller().check(state, 0.0) is None


def test_breaker_reads_a_real_graph_document() -> None:
    graph = WorkflowGraph(
        [
            NodeSpec("a", "a", Stage.INTAKE, "agent-a"),
            NodeSpec("b", "b", Stage.DESIGN, "agent-b", fallback_agent="agent-c"),
        ]
    )
    state = make_state(graph=graph.to_dict())
    state.nodes = {"a": NodeState("a", NodeStatus.READY), "b": NodeState("b", NodeStatus.READY)}
    state.counters.breaker_open = ["agent-a", "agent-b"]
    trigger = controller().check(state, 0.0)
    assert trigger is not None and "(nodes: a)" in trigger.reason  # b can still use agent-c


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"nodes": "text"},
        {"nodes": [1, None, "x", {"id": 5, "agent": "a"}, {"id": "n"}, {"agent": "a"}]},
        [],
        "garbage",
    ],
)
def test_malformed_graph_documents_never_crash_the_check(document: Any) -> None:
    state = make_state(graph=document)
    state.nodes = {"n": NodeState("n", NodeStatus.READY)}
    state.counters.breaker_open = ["agent-a"]
    assert controller().check(state, 0.0) is None


def test_long_node_lists_are_summarised() -> None:
    names = [f"node{i}" for i in range(9)]
    state = breaker_state(
        dict.fromkeys(names, NodeStatus.READY),
        tuple((n, "agent-a", None) for n in names),
        ["agent-a"],
    )
    trigger = controller().check(state, 0.0)
    assert trigger is not None and "(9 in total)" in trigger.reason
    assert not has_control_characters(trigger.reason)


def test_hostile_agent_ids_are_sanitised_in_reasons() -> None:
    state = breaker_state(
        {"n": NodeStatus.READY}, (("n", "evil\x1b[31m\nagent", None),), ["evil\x1b[31m\nagent"]
    )
    trigger = controller().check(state, 0.0)
    assert trigger is not None and not has_control_characters(trigger.reason)


def test_default_config_matches_the_documented_budgets() -> None:
    cfg = EngineConfig()
    ctl = SafeStopController(cfg)
    assert ctl.config is cfg
    state = make_state()
    state.counters.budgeted_attempts = cfg.max_total_attempts
    state.counters.replans = cfg.max_replans
    state.counters.zombies = cfg.max_zombie_threads
    assert ctl.check(state, cfg.max_active_runtime_s) is None  # exactly at every limit
    state.counters.budgeted_attempts = cfg.max_total_attempts + 1
    assert ctl.check(state, 0.0) is not None
