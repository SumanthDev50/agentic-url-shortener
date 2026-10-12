"""Tests for ``orchestrator.tools.toolbox``: scoped, audited, cancel-aware agent tools."""

from __future__ import annotations

import dataclasses
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.audit import AuditLog
from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import (
    AgentCancelled,
    AgentError,
    AgentTimeout,
    AutonomyViolation,
    ConfigError,
    PathViolation,
    ToolDenied,
)
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.types import (
    ActionSubject,
    Actor,
    AutonomyProfile,
    CancelToken,
    Enforcement,
    EventType,
    FileOpKind,
    Finding,
    PolicyCategory,
    PolicyReport,
    RiskAssessment,
    RiskLevel,
    Severity,
    ToolAction,
    Toolbox,
)
from orchestrator.tools import toolbox as toolbox_module
from orchestrator.tools.changeset import apply_changeset
from orchestrator.tools.sandbox import SandboxRunner
from orchestrator.tools.toolbox import (
    BudgetExhausted,
    InvalidToolArgument,
    ScopedToolbox,
    compile_search_regex,
    make_toolbox_factory,
)
from orchestrator.tools.workspace import MemoryView

FAKE_KEY = "AK" + "IA" + "X" * 16
_KEY_RE = re.compile("AK" + "IA" + "[A-Z0-9]{16}")

BASE = {
    "app/main.py": "import os\n\ndef run():\n    return os.getcwd()\n\nclass Runner:\n    pass\n",
    "app/util.py": "def helper(x):\n    return x + 1\n",
    "tests/test_main.py": "def test_run():\n    assert True\n",
    "README.md": "# Demo run\n",
    "secret/notes.txt": "private\n",
}

PROFILE = AutonomyProfile(
    read_globs=("**",),
    write_globs=("app/**", "tests/**", "docs/**"),
    can_execute=True,
    max_tool_calls=200,
    max_runtime_s=120.0,
    max_changed_files=10,
    max_file_bytes=2048,
    max_total_bytes=8192,
)


# ------------------------------ test doubles


@dataclass
class Event:
    type: EventType
    node_id: str | None
    attempt: int | None
    actor: Actor
    data: dict[str, Any]


class Recorder:
    """An ``AuditSink`` that remembers everything it is given."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(
        self,
        type: EventType,
        *,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: Mapping[str, Any],
    ) -> None:
        self.events.append(Event(type, node_id, attempt, actor, dict(data)))

    def of(self, *types: EventType) -> list[Event]:
        return [e for e in self.events if e.type in types]


class StubPolicy:
    """A ``PolicyEvaluator`` whose verdict the test dictates."""

    def __init__(
        self,
        enforcement: Enforcement = Enforcement.ALLOW,
        *,
        rule_ids: tuple[str, ...] = (),
        raises: Exception | None = None,
        decide: Callable[[ActionSubject], Enforcement] | None = None,
    ) -> None:
        self.enforcement = enforcement
        self.rule_ids = rule_ids
        self.raises = raises
        self.decide = decide
        self.subjects: list[ActionSubject] = []

    def evaluate(self, subject: ActionSubject) -> PolicyReport:
        self.subjects.append(subject)
        if self.raises is not None:
            raise self.raises
        enforcement = self.decide(subject) if self.decide else self.enforcement
        findings = tuple(
            Finding(rid, PolicyCategory.SECURITY, Severity.HIGH, "stub finding")
            for rid in self.rule_ids
        )
        blocking = findings if enforcement in (Enforcement.DENY, Enforcement.SAFE_STOP) else ()
        needs = findings if enforcement == Enforcement.REQUIRE_APPROVAL else ()
        return PolicyReport(findings, (), enforcement, blocking, needs)

    def with_waivers(self, waivers: Any) -> StubPolicy:
        return self

    def assess_risk(self, subject: Any) -> RiskAssessment:
        return RiskAssessment(RiskLevel.LOW)


@dataclass
class Rig:
    box: ScopedToolbox
    audit: Recorder
    policy: StubPolicy
    cancel: CancelToken
    clock: FakeClock
    view: MemoryView
    extra: dict[str, Any] = field(default_factory=dict)


def build(
    files: Mapping[str, str] | None = None,
    *,
    profile: AutonomyProfile = PROFILE,
    policy: StubPolicy | None = None,
    runner: SandboxRunner | None = None,
    **profile_overrides: Any,
) -> Rig:
    if profile_overrides:
        profile = dataclasses.replace(profile, **profile_overrides)
    audit, cancel, clock = Recorder(), CancelToken(), FakeClock()
    stub = policy or StubPolicy()
    view = MemoryView(BASE if files is None else files)
    box = ScopedToolbox(
        run_id="run-1",
        node_id="implement.a",
        attempt=1,
        agent_id="impl-agent",
        profile=profile,
        view=view,
        cancel=cancel,
        clock=clock,
        policy=stub,  # type: ignore[arg-type]
        audit=audit,
        runner=runner,
    )
    return Rig(box, audit, stub, cancel, clock, view)


# ------------------------------ reads and staged writes


def test_reads_see_the_view_and_are_audited() -> None:
    rig = build()
    assert rig.box.read_text("app/util.py") == BASE["app/util.py"]
    assert rig.box.exists("app/util.py") and not rig.box.exists("app/nope.py")
    assert rig.box.list_files("app/**") == ["app/main.py", "app/util.py"]
    with pytest.raises(FileNotFoundError):
        rig.box.read_text("app/nope.py")
    calls = rig.audit.of(EventType.TOOL_CALL)
    assert [(e.data["tool"], e.data["target"]) for e in calls] == [
        ("read_text", "app/util.py"),
        ("exists", "app/util.py"),
        ("exists", "app/nope.py"),
        ("list_files", "app/**"),
        ("read_text", "app/nope.py"),
    ]
    assert calls[-1].data["ok"] is False and calls[0].data["ok"] is True
    assert {(e.node_id, e.attempt, e.actor.label) for e in calls} == {
        ("implement.a", 1, "agent:impl-agent")
    }


def test_staged_writes_are_visible_to_later_reads_but_never_touch_the_view() -> None:
    rig = build()
    rig.box.write_text("app/new.py", "x = 1\r\ny = 2\r\n")
    rig.box.write_text("app/util.py", "def helper(x):\n    return x + 2\n")
    rig.box.delete("app/main.py")
    assert rig.box.read_text("app/new.py") == "x = 1\ny = 2\n"  # CRLF normalised to LF
    assert "return x + 2" in rig.box.read_text("app/util.py")
    assert not rig.box.exists("app/main.py")
    with pytest.raises(FileNotFoundError):
        rig.box.read_text("app/main.py")
    assert rig.box.list_files("app/**") == ["app/new.py", "app/util.py"]
    assert rig.view.read_text("app/util.py") == BASE["app/util.py"]  # the start view is immutable


def test_staged_changeset_is_computed_against_the_attempt_start_view() -> None:
    rig = build()
    rig.box.write_text("app/util.py", "v1\n")  # first touch: base hash of the START content
    rig.box.write_text("app/util.py", "v2\n")  # a second write must not change the base hash
    rig.box.write_text("app/created.py", "c1\n")
    rig.box.write_text("app/created.py", "c2\n")  # create-then-modify stays a CREATE
    rig.box.write_text("app/ephemeral.py", "e\n")
    rig.box.delete("app/ephemeral.py")  # create-then-delete vanishes
    rig.box.delete("app/main.py")
    cs = rig.box.staged_changeset("my change")
    ops = {op.path: op for op in cs.ops}
    assert sorted(ops) == ["app/created.py", "app/main.py", "app/util.py"]
    assert ops["app/util.py"].kind == FileOpKind.MODIFY
    assert ops["app/util.py"].base_hash == sha256_text(BASE["app/util.py"])
    assert ops["app/util.py"].content == "v2\n"
    assert (ops["app/created.py"].kind, ops["app/created.py"].content) == (
        FileOpKind.CREATE,
        "c2\n",
    )
    assert ops["app/main.py"].kind == FileOpKind.DELETE
    assert ops["app/main.py"].base_hash == sha256_text(BASE["app/main.py"])
    assert (cs.description, cs.author) == ("my change", "impl-agent")
    assert (
        apply_changeset(rig.view, cs)["app/created.py"] == "c2\n"
    )  # strict apply to the start view


def test_a_write_that_restores_the_original_content_leaves_no_op() -> None:
    rig = build()
    rig.box.write_text("app/util.py", "changed\n")
    rig.box.write_text("app/util.py", BASE["app/util.py"])
    assert rig.box.staged_changeset().ops == ()


def test_staged_changeset_sanitises_the_description() -> None:
    rig = build()
    cs = rig.box.staged_changeset("line1\n\x1b[31mred\x1b[0m" + "x" * 1000)
    assert "\x1b" not in cs.description and "\n" not in cs.description
    assert len(cs.description) <= 520


def test_the_toolbox_satisfies_the_protocol_and_usage_counts_calls() -> None:
    rig = build()
    assert isinstance(rig.box, Toolbox)
    rig.box.read_text("README.md")
    rig.box.write_text("app/a.py", "12345")
    rig.box.write_text("app/a.py", "1234567890")
    rig.clock.advance(2.5)
    usage = rig.box.usage()
    assert usage["tool_calls"] == 3.0
    assert usage["bytes_written"] == 15.0  # cumulative bytes accepted
    assert usage["runtime_s"] == pytest.approx(2.5)


# ------------------------------ scope: read / write globs


def test_reads_outside_read_globs_are_denied_but_listings_are_filtered() -> None:
    rig = build(read_globs=("app/**", "README.md"))
    assert rig.box.read_text("README.md")
    with pytest.raises(ToolDenied, match="read_globs"):
        rig.box.read_text("secret/notes.txt")
    with pytest.raises(ToolDenied):
        rig.box.exists("secret/notes.txt")  # even existence is information
    assert rig.box.list_files("**") == ["README.md", "app/main.py", "app/util.py"]
    assert {h.path for h in rig.box.search("private", glob="**/*.txt")} == set()


def test_an_agent_can_read_back_what_it_staged_even_outside_read_globs() -> None:
    rig = build(read_globs=("app/**",), write_globs=("tests/**",))
    rig.box.write_text("tests/test_new.py", "def test_x():\n    pass\n")
    assert "test_x" in rig.box.read_text("tests/test_new.py")
    assert rig.box.list_files("tests/**") == ["tests/test_new.py"]
    with pytest.raises(ToolDenied):
        rig.box.read_text(
            "tests/test_main.py"
        )  # a pre-existing file outside read_globs stays hidden


def test_a_valid_path_outside_write_globs_is_denied_and_audited() -> None:
    rig = build()
    with pytest.raises(ToolDenied, match="write_globs") as excinfo:
        rig.box.write_text("src/other.py", "x")
    assert excinfo.value.kind == "policy_violation" and excinfo.value.retryable is True
    with pytest.raises(ToolDenied):
        rig.box.delete("secret/notes.txt")
    denied = rig.audit.of(EventType.TOOL_DENIED)
    assert [e.data["tool"] for e in denied] == ["write_text", "delete"]
    assert denied[0].data["denials"] == 1 and denied[1].data["denials"] == 2
    assert rig.box.staged_changeset().ops == ()  # nothing was staged


def test_the_third_denial_escalates_to_an_autonomy_violation_and_poisons_the_toolbox() -> None:
    rig = build(can_execute=False)
    with pytest.raises(ToolDenied):
        rig.box.write_text("src/a.py", "x")
    with pytest.raises(ToolDenied):
        rig.box.run_tests()  # a denial of another kind counts too
    with pytest.raises(AutonomyViolation) as excinfo:
        rig.box.write_text("src/b.py", "x")
    assert not isinstance(excinfo.value, ToolDenied)
    assert excinfo.value.kind == "autonomy_violation" and excinfo.value.retryable is False
    assert excinfo.value.details["denials"] == 3
    # an agent that swallows the exception cannot carry on: everything keeps failing
    for call in (
        lambda: rig.box.read_text("README.md"),
        lambda: rig.box.write_text("app/ok.py", "x"),
        lambda: rig.box.list_files(),
        lambda: rig.box.staged_changeset(),
    ):
        with pytest.raises(AutonomyViolation):
            call()
    assert rig.box.usage()["tool_calls"] >= 3  # usage stays readable for the engine
    assert rig.audit.of(EventType.TOOL_DENIED)[-1].data.get("critical") is True


# ------------------------------ hostile paths


HOSTILE_PATHS = [
    "../x.py",
    "app/../../x.py",
    "/etc/passwd",
    "/app/main.py",
    "C:x.py",
    "C:\\Windows\\win.ini",
    "a:b",
    "app/main.py:stream",
    "CON",
    "app/aux.txt",
    "NUL.py",
    "com1",
    ".git/config",
    "app/.git/hooks/pre-commit",
    ".GIT/config",
    "a\\b.py",
    "app\\main.py",
    "app//main.py",
    "app/./main.py",
    "",
    ".",
    "..",
    "app/",
    "app/name.",
    "app/name ",
    "evil\u202e.py",  # right-to-left override
    "zero\u200bwidth.py",  # zero-width space
    "bom\ufeff.py",
    "nul\x00byte.py",
    "tab\tname.py",
    "new\nline.py",
    "\u2028.py",
    "x" * 300,
    "/".join(["d"] * 20) + "/f.py",
]


@pytest.mark.parametrize("bad", HOSTILE_PATHS)
@pytest.mark.parametrize("operation", ["read_text", "exists", "write_text", "delete"])
def test_hostile_paths_raise_a_path_violation_on_every_method(bad: str, operation: str) -> None:
    rig = build(write_globs=("**",))
    call = {
        "read_text": lambda: rig.box.read_text(bad),
        "exists": lambda: rig.box.exists(bad),
        "write_text": lambda: rig.box.write_text(bad, "x"),
        "delete": lambda: rig.box.delete(bad),
    }[operation]
    with pytest.raises(PathViolation):
        call()
    assert rig.policy.subjects == []  # rejected before the policy was even consulted
    event = rig.audit.of(EventType.TOOL_DENIED)[-1]
    assert event.data["critical"] is True
    assert all(32 <= ord(ch) < 127 or ch.isprintable() for ch in event.data["target"])  # log-safe
    with pytest.raises(AutonomyViolation):  # CRITICAL => the toolbox is dead afterwards
        rig.box.read_text("README.md")
    with pytest.raises(AutonomyViolation):
        rig.box.staged_changeset()


@pytest.mark.parametrize("bad", ["../**", "/abs/**", "a\\b", "a//b", "a/../b/**", "\u202e**"])
def test_hostile_globs_are_path_violations(bad: str) -> None:
    for call in (
        lambda rig: rig.box.list_files(bad),
        lambda rig: rig.box.search("x", glob=bad),
    ):
        rig = build()
        with pytest.raises(PathViolation):
            call(rig)


def test_non_string_paths_are_rejected_as_violations() -> None:
    for bad in (None, 5, b"app/main.py", ["app/main.py"]):
        rig = build()
        with pytest.raises(PathViolation):
            rig.box.read_text(bad)  # type: ignore[arg-type]


def test_unicode_paths_are_normalised_and_cannot_alias_an_existing_file() -> None:
    nfc, nfd = "app/caf\u00e9.py", "app/cafe\u0301.py"
    rig = build()
    rig.box.write_text(nfd, "x = 1\n")
    assert rig.box.read_text(nfc) == "x = 1\n"  # the same file, whichever spelling is used
    (op,) = rig.box.staged_changeset().ops
    assert op.path == nfc  # the proposal carries the normalised path


def test_case_insensitive_collisions_are_denied() -> None:
    rig = build(write_globs=("**",))
    with pytest.raises(ToolDenied, match="case-insensitively"):
        rig.box.write_text("readme.md", "# shadow\n")  # collides with README.md on Windows / macOS
    rig.box.write_text("Docs/Guide.md", "g\n")
    with pytest.raises(ToolDenied, match="case-insensitively"):
        rig.box.write_text("docs/guide.md", "g2\n")
    rig.box.write_text("Docs/Guide.md", "g3\n")  # the same spelling may be rewritten
    assert [op.path for op in rig.box.staged_changeset().ops] == ["Docs/Guide.md"]


def test_a_path_cannot_be_both_a_file_and_a_directory() -> None:
    # Every denial counts towards the three-strikes escalation (see the test above), so the two
    # scenarios below use one toolbox each and stay under it (two denials per toolbox).
    rig = build(write_globs=("**",))
    with pytest.raises(ToolDenied, match="parent path is an existing file"):
        rig.box.write_text("README.md/inner.txt", "x")  # README.md is a file
    with pytest.raises(ToolDenied, match="existing directory"):
        rig.box.write_text("app", "x")  # app/ holds files
    rig.box.write_text("app/main.py", "rewrite of an existing file is fine\n")
    rig.box.delete("README.md")
    rig.box.write_text("README.md/now_fine.txt", "x")  # its file is gone: the name is free
    assert {op.path for op in rig.box.staged_changeset().ops} == {
        "app/main.py",
        "README.md/now_fine.txt",
        "README.md",
    }

    staged = build(write_globs=("**",))
    staged.box.write_text("fresh/dir/file.txt", "x")
    with pytest.raises(ToolDenied, match="parent path is an existing file"):
        staged.box.write_text("fresh/dir/file.txt/deeper", "x")  # a staged file is a file too
    with pytest.raises(ToolDenied, match="existing directory"):
        staged.box.write_text("fresh", "x")  # a staged directory is a directory too
    assert {op.path for op in staged.box.staged_changeset().ops} == {"fresh/dir/file.txt"}


def test_repeated_file_directory_clashes_are_denials_and_escalate_like_any_other() -> None:
    rig = build(write_globs=("**",))
    for clash in ("README.md/inner.txt", "app"):
        with pytest.raises(ToolDenied):
            rig.box.write_text(clash, "x")
    with pytest.raises(AutonomyViolation) as excinfo:  # the third clash is the third denial
        rig.box.write_text("app/util.py/deeper", "x")
    assert not isinstance(excinfo.value, ToolDenied) and excinfo.value.details["denials"] == 3
    with pytest.raises(AutonomyViolation):  # and the toolbox stays poisoned
        rig.box.write_text("app/ok.py", "x")
    assert rig.box.usage()["tool_calls"] >= 3


def test_a_denied_write_does_not_reserve_its_path_spelling() -> None:
    rig = build(write_globs=("**",), max_file_bytes=10)
    with pytest.raises(ToolDenied):
        rig.box.write_text("Foo.py", "x" * 100)  # too large: denied
    rig.box.write_text("foo.py", "small")  # a different case spelling is still free


# ------------------------------ content and size limits


def test_content_must_be_text_without_nul_and_valid_utf8() -> None:
    rig = build()
    for bad in ("nul\x00inside", "lone\ud800surrogate", 5, None, b"bytes"):
        with pytest.raises(InvalidToolArgument):
            rig.box.write_text("app/a.py", bad)  # type: ignore[arg-type]
    assert issubclass(InvalidToolArgument, ValueError) and InvalidToolArgument("x").kind == "fatal"
    assert rig.box.staged_changeset().ops == ()


def test_size_and_changed_file_limits_are_enforced_as_denials() -> None:
    rig = build(max_file_bytes=20, max_total_bytes=45, max_changed_files=2)
    with pytest.raises(ToolDenied, match="max_file_bytes"):
        rig.box.write_text("app/big.py", "x" * 21)
    rig.box.write_text("app/a.py", "x" * 20)
    rig.box.write_text("app/a.py", "y" * 20)  # rewriting one file does not double-count its bytes
    rig.box.write_text("app/b.py", "z" * 20)
    with pytest.raises(ToolDenied, match="max_total_bytes"):
        rig.box.write_text("app/c.py", "w" * 20)  # 60 > 45 staged bytes
    assert len(rig.box.staged_changeset().ops) == 2
    rig2 = build(max_changed_files=2)
    rig2.box.write_text("app/a.py", "1")
    rig2.box.write_text("app/b.py", "2")
    rig2.box.write_text("app/a.py", "1b")  # an already changed file may be edited again
    with pytest.raises(ToolDenied, match="max_changed_files"):
        rig2.box.write_text("app/c.py", "3")
    rig2.box.delete("app/b.py")  # un-creating one frees a slot
    rig2.box.write_text("app/c.py", "3")


def test_reading_a_file_above_max_file_bytes_is_denied() -> None:
    rig = build({"app/huge.py": "x" * 500}, max_file_bytes=100)
    with pytest.raises(ToolDenied, match="max_file_bytes"):
        rig.box.read_text("app/huge.py")


def test_delete_of_a_missing_file_raises_file_not_found() -> None:
    rig = build()
    with pytest.raises(FileNotFoundError):
        rig.box.delete("app/ghost.py")
    assert rig.audit.of(EventType.TOOL_CALL)[-1].data["ok"] is False
    rig.box.write_text("app/tmp.py", "x")
    rig.box.delete("app/tmp.py")
    with pytest.raises(FileNotFoundError):
        rig.box.delete("app/tmp.py")


# ------------------------------ budgets


def test_the_call_budget_is_exhausted_with_a_non_escalating_error() -> None:
    rig = build(max_tool_calls=3)
    for _ in range(3):
        rig.box.read_text("README.md")
    for _ in range(4):  # repeated attempts never escalate: this is not a denial
        with pytest.raises(BudgetExhausted) as excinfo:
            rig.box.read_text("README.md")
    assert not isinstance(excinfo.value, AutonomyViolation)
    assert excinfo.value.kind == "policy_violation" and excinfo.value.retryable is False
    assert rig.box.usage()["tool_calls"] == 3.0
    assert rig.box.staged_changeset().ops == ()  # results stay retrievable once the budget is spent
    assert rig.audit.of(EventType.TOOL_DENIED)[0].data["budget"] is True


def test_a_zero_budget_denies_the_very_first_call() -> None:
    with pytest.raises(BudgetExhausted):
        build(max_tool_calls=0).box.exists("README.md")


def test_every_io_method_consumes_budget() -> None:
    rig = build(max_tool_calls=1000)
    rig.box.read_text("README.md")
    rig.box.exists("README.md")
    rig.box.list_files()
    rig.box.search("run")
    rig.box.write_text("app/x.py", "x")
    rig.box.delete("app/x.py")
    rig.box.note("hello")
    assert rig.box.usage()["tool_calls"] == 7.0


def test_the_runtime_budget_stops_the_attempt() -> None:
    rig = build(max_runtime_s=10.0)
    rig.box.read_text("README.md")
    rig.clock.advance(10.5)
    with pytest.raises(AgentTimeout):
        rig.box.read_text("README.md")
    with pytest.raises(AgentTimeout):
        rig.box.staged_changeset()


# ------------------------------ cancellation


def _every_method(rig: Rig) -> list[Callable[[], Any]]:
    box = rig.box
    return [
        lambda: box.read_text("README.md"),
        lambda: box.exists("README.md"),
        lambda: box.list_files(),
        lambda: box.search("run"),
        lambda: box.write_text("app/x.py", "x"),
        lambda: box.delete("app/main.py"),
        lambda: box.staged_changeset(),
        lambda: box.run_tests(),
        lambda: box.run_python(["-m", "unittest"]),
        lambda: box.note("hi"),
        lambda: box.usage(),
    ]


def test_after_cancel_every_method_raises_and_nothing_more_is_audited() -> None:
    rig = build()
    rig.box.read_text("README.md")
    before = list(rig.audit.events)
    rig.cancel.cancel("kill switch")
    for call in _every_method(rig):
        with pytest.raises(AgentCancelled):
            call()
    assert rig.audit.events == before  # events after the cancel are dropped
    assert len(rig.policy.subjects) == 1  # the policy is not consulted after the cancel either


def test_cancel_between_the_check_and_the_audit_drops_the_event() -> None:
    cancel_in_policy = StubPolicy(decide=lambda subject: Enforcement.ALLOW)
    rig = build(policy=cancel_in_policy)
    original = cancel_in_policy.evaluate

    def evaluate_then_cancel(subject: ActionSubject) -> PolicyReport:
        report = original(subject)
        rig.cancel.cancel("cancelled while the call was in flight")
        return report

    cancel_in_policy.evaluate = evaluate_then_cancel  # type: ignore[method-assign]
    rig.box.read_text("README.md")  # the call itself completes ...
    assert rig.audit.events == []  # ... but leaves no trace once cancelled


# ------------------------------ policy


def test_every_io_call_is_evaluated_as_an_action_subject() -> None:
    rig = build()
    rig.box.read_text("app/util.py")
    rig.box.exists("README.md")
    rig.box.list_files("app/**")
    rig.box.search("run", glob="**/*.py")
    rig.box.write_text("app/new.py", "x")
    rig.box.delete("app/main.py")
    actions = [(s.action, s.target) for s in rig.policy.subjects]
    assert actions == [
        (ToolAction.READ, "app/util.py"),
        (ToolAction.READ, "README.md"),
        (ToolAction.READ, "app/**"),  # a glob, not a file path
        (ToolAction.READ, "**/*.py"),
        (ToolAction.WRITE, "app/new.py"),
        (ToolAction.DELETE, "app/main.py"),
    ]
    first = rig.policy.subjects[0]
    assert isinstance(first, ActionSubject)
    assert (first.agent_id, first.node_id, first.profile) == ("impl-agent", "implement.a", PROFILE)


def test_a_safe_stop_verdict_becomes_an_autonomy_violation() -> None:
    rig = build(policy=StubPolicy(Enforcement.SAFE_STOP, rule_ids=("SEC-005",)))
    with pytest.raises(AutonomyViolation, match="SEC-005") as excinfo:
        rig.box.read_text("README.md")
    assert not isinstance(excinfo.value, ToolDenied)
    with pytest.raises(AutonomyViolation):
        rig.box.exists("README.md")  # and the toolbox is poisoned
    assert rig.audit.of(EventType.TOOL_DENIED)[0].data["critical"] is True


@pytest.mark.parametrize("verdict", [Enforcement.DENY, Enforcement.REQUIRE_APPROVAL])
def test_deny_and_require_approval_verdicts_fail_closed_as_denials(verdict: Enforcement) -> None:
    rig = build(policy=StubPolicy(verdict, rule_ids=("SEC-005",)))
    with pytest.raises(ToolDenied, match="SEC-005"):
        rig.box.write_text("app/new.py", "x")
    assert rig.box.staged_changeset().ops == ()  # the write did not happen


def test_allow_and_warn_verdicts_let_the_call_through() -> None:
    for verdict in (Enforcement.ALLOW, Enforcement.WARN):
        rig = build(policy=StubPolicy(verdict))
        rig.box.write_text("app/new.py", "x")
        assert rig.box.staged_changeset().ops


def test_a_policy_that_blocks_only_some_actions() -> None:
    policy = StubPolicy(
        decide=lambda s: Enforcement.DENY if s.action == ToolAction.DELETE else Enforcement.ALLOW
    )
    rig = build(policy=policy)
    rig.box.write_text("app/new.py", "x")
    with pytest.raises(ToolDenied):
        rig.box.delete("app/main.py")


def test_a_crashing_policy_fails_closed() -> None:
    rig = build(policy=StubPolicy(raises=RuntimeError("boom")))
    with pytest.raises(AgentError) as excinfo:
        rig.box.write_text("app/new.py", "x")
    assert excinfo.value.kind == "fatal" and excinfo.value.retryable is False
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert rig.box.staged_changeset().ops == ()


# ------------------------------ search


def test_search_is_literal_by_default() -> None:
    rig = build()
    hits = rig.box.search("def run(")  # '(' would be a regex syntax error
    assert [(h.path, h.line, h.text) for h in hits] == [("app/main.py", 3, "def run():")]
    assert rig.box.search("def .*run") == []  # not interpreted as a regex
    assert rig.box.search("r.n") == []  # '.' is a literal dot
    assert {h.path for h in rig.box.search("run")} == {"app/main.py", "tests/test_main.py"}


def test_search_honours_glob_and_result_limit_and_sees_staged_writes() -> None:
    rig = build()
    assert [h.path for h in rig.box.search("run", glob="*.md")] == ["README.md"]
    rig.box.write_text("app/extra.py", "run_extra = 1\nrun_more = 2\n")
    hits = rig.box.search("run_", max_results=1)
    assert len(hits) == 1 and hits[0].path == "app/extra.py"
    assert len(rig.box.search("run_", max_results=10)) == 2
    for bad in (0, -1, True, "5", None, 1.5):
        with pytest.raises(InvalidToolArgument):
            rig.box.search("run", max_results=bad)  # type: ignore[arg-type]
    for bad_text in ("", "x" * 201, None, 5):
        with pytest.raises(InvalidToolArgument):
            rig.box.search(bad_text)  # type: ignore[arg-type]


def test_search_results_are_deterministic_bounded_and_clean() -> None:
    long_line = "needle " + "y" * 1000 + "\x1b[31m"
    rig = build(
        {"b.py": long_line + "\n", "a.py": "needle\n", "c.py": "needle\n"}, max_file_bytes=4096
    )
    hits = rig.box.search("needle")
    assert [h.path for h in hits] == ["a.py", "b.py", "c.py"]
    assert len(hits[1].text) <= toolbox_module.HIT_TEXT_CHARS and "\x1b" not in hits[1].text
    assert len(rig.box.search("needle", max_results=10_000)) == 3


def test_regex_search_with_a_restricted_grammar() -> None:
    rig = build()
    assert [(h.path, h.line) for h in rig.box.search(r"def \w+\(", regex=True)] == [
        ("app/main.py", 3),
        ("app/util.py", 1),
        ("tests/test_main.py", 1),
    ]
    assert [h.text for h in rig.box.search(r"^class (\w+):$", regex=True)] == ["class Runner:"]
    assert rig.audit.of(EventType.TOOL_CALL)[-1].data["regex"] is True


@pytest.mark.parametrize(
    "pattern",
    [
        r"def \w+\(",
        r"^class (\w+)",
        r"\d{1,3}\.\d+",
        r"(?:foo)?bar",
        r"(\w+)?x",
        r"(foo|bar)",
        r"a+?b",
        r"[a-z_]+",
        r"[^\]]+",
        r"x{3}",
        r"x{,5}",
        r"x{2,}",
        r"a{b}",  # not a repeat for re: literal braces
        r"a{}",
        r"{",
        r"\bword\b",
        r"a|b|c",
        r"\.\*\+\?",
        r"\s*def\s+\w+",  # three wide repetitions is the maximum
    ],
)
def test_regex_grammar_accepts(pattern: str) -> None:
    assert compile_search_regex(pattern).pattern == pattern


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        "(a+)+$",  # exponential
        "(a*)*",
        "(a+)*b",
        "(.*)*x",
        "(a|aa)+",  # exponential through alternation
        "(a|a)*",
        "((a+)b)+",
        "(?:a+)+",
        "(a{2,5})+",
        r"(\w+\s?)+",
        r"\1",  # backreference
        r"(a)\2",
        r"(?P<n>a)(?P=n)",
        r"(?=a)",  # lookarounds / inline flags
        r"(?!a)",
        r"(?<=a)b",
        r"(?i)abc",
        r"(?P<name>a)",
        "a{1000}",
        "a{1000,}",
        "a{,1000}",
        "(a+){1000}",  # a repeat the scanner must read exactly like re does
        "(a+){,5}",
        "(a+){2,}",
        "(a|b){2,3}",
        "a{0,101}",
        "a" * 201,
        "[[a]",
        "[a",
        "(a",
        "a)",
        "a**",
        "a+*",
        "a*+",
        "a{2}{3}",
        "*a",
        "+a",
        "?a",
        "a|*",
        r"\x41",
        r"\u0041",
        r"\A",
        r"\Z",
        r"\g<1>",
        "a\\",
        "a*a*a*a*b",  # four wide repetitions: polynomial blow-up
        r".*.*.*.*",
        "a{2,1}",  # rejected by re itself
        "a\x00b",
        "a\nb",
    ],
)
def test_regex_grammar_rejects(pattern: str) -> None:
    with pytest.raises(InvalidToolArgument):
        compile_search_regex(pattern)


def test_regex_input_must_be_a_string() -> None:
    for bad in (None, 5, b"abc"):
        with pytest.raises(InvalidToolArgument):
            compile_search_regex(bad)  # type: ignore[arg-type]


def test_regex_search_only_looks_at_the_first_characters_of_a_line() -> None:
    line = "x" * 250 + "needle"
    rig = build({"app/a.py": line + "\n"}, max_file_bytes=4096)
    assert rig.box.search("needle", regex=True) == []  # beyond REGEX_LINE_CHARS: bounded work
    assert len(rig.box.search("needle")) == 1  # the literal search sees the whole line


def test_search_work_is_bounded_by_scanned_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = build({f"app/f{i}.py": "needle\n" * 10 for i in range(30)}, max_file_bytes=4096)
    monkeypatch.setattr(toolbox_module, "LITERAL_SCAN_CHARS", 150)
    hits = rig.box.search("needle", max_results=1000)
    assert 0 < len(hits) < 300
    assert rig.audit.of(EventType.TOOL_CALL)[-1].data["complete"] is False


def test_search_work_is_bounded_by_a_wall_clock_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = build({"app/a.py": "needle\n" * 5})
    monkeypatch.setattr(toolbox_module, "_SEARCH_TIME_BUDGET_S", -1.0)
    assert rig.box.search("needle") == []
    assert rig.audit.of(EventType.TOOL_CALL)[-1].data["complete"] is False


def test_a_pathological_regex_cannot_hold_the_toolbox_hostage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three wide repetitions are the most the grammar admits (~0.2 s per 200-char line)."""
    rig = build({"app/a.py": ("a" * 200 + "\n") * 200}, max_file_bytes=1_000_000)
    monkeypatch.setattr(toolbox_module, "_SEARCH_TIME_BUDGET_S", 0.5)
    started = time.monotonic()
    assert rig.box.search(r"a*a*a*b", regex=True) == []
    assert time.monotonic() - started < 10
    assert rig.audit.of(EventType.TOOL_CALL)[-1].data["complete"] is False


# ------------------------------ note


def test_note_is_audited_sanitised_and_bounded() -> None:
    rig = build()
    rig.box.note("hello\n\x1b[31mworld\x1b[0m " + "x" * 2000, step=3, ok=True)
    rig.box.note("big", blob="y" * 10_000)
    rig.box.note("odd", thing=object())
    rig.box.note("secret", token=FAKE_KEY)
    notes = rig.audit.of(EventType.AGENT_NOTE)
    assert "\x1b" not in notes[0].data["message"] and "\n" not in notes[0].data["message"]
    assert len(notes[0].data["message"]) <= 500 and notes[0].data["data"] == {"step": 3, "ok": True}
    assert "_dropped" in notes[1].data["data"] and "_dropped" in notes[2].data["data"]
    assert notes[3].data["data"] == {
        "token": FAKE_KEY
    }  # redaction is the AuditLog's job (tested below)


# ------------------------------ execution


@pytest.fixture
def runner(tmp_path: Path) -> SandboxRunner:
    return SandboxRunner(tmp_path / "scratch", redactor=lambda s: _KEY_RE.sub("[REDACTED]", s))


def test_execution_requires_can_execute_and_a_runner(runner: SandboxRunner) -> None:
    with pytest.raises(ToolDenied, match="execution"):
        build(can_execute=False, runner=runner).box.run_python(["-m", "unittest"])
    with pytest.raises(ConfigError, match="sandbox runner"):
        build().box.run_tests()


@pytest.mark.parametrize(
    "targets",
    [
        ["-p evil"],
        ["-pevil"],
        ["--rootdir=/tmp"],
        ["@argfile"],
        ["tests x"],
        ["tests/*.py"],
        ["a;b"],
        ["tests::x"],
    ],
)
def test_run_tests_rejects_hostile_targets_as_denials(
    runner: SandboxRunner, targets: list[str]
) -> None:
    rig = build(runner=runner)
    with pytest.raises(AgentError) as excinfo:
        rig.box.run_tests(targets)
    assert isinstance(excinfo.value, (ToolDenied, PathViolation))
    assert list(runner.scratch_root.iterdir()) == []  # rejected before anything was materialised


@pytest.mark.parametrize(
    "targets", [["../x"], ["/abs"], ["C:x"], ["a:b"], ["CON"], [".git/config"], ["a\\b"]]
)
def test_run_tests_rejects_escaping_targets_as_path_violations(
    runner: SandboxRunner, targets: list[str]
) -> None:
    rig = build(runner=runner)
    with pytest.raises(PathViolation):
        rig.box.run_tests(targets)
    assert list(runner.scratch_root.iterdir()) == []


@pytest.mark.parametrize(
    "args",
    [
        ["-c", "print(1)"],
        ["-m", "pytest"],
        ["-m", "http.server"],
        ["-X", "dev", "a.py"],
        ["../x.py"],
        ["/abs/x.py"],
        [],
    ],
)
def test_run_python_rejects_everything_but_modules_and_scripts(
    runner: SandboxRunner, args: list[str]
) -> None:
    rig = build(runner=runner)
    with pytest.raises(ToolDenied):
        rig.box.run_python(args)
    assert list(runner.scratch_root.iterdir()) == []


def test_exec_arguments_are_validated(runner: SandboxRunner) -> None:
    rig = build(runner=runner)
    for bad in ("tests", b"x", 5, None):
        with pytest.raises(InvalidToolArgument):
            rig.box.run_tests(bad)  # type: ignore[arg-type]
    for bad in ("args", b"x", 5):
        with pytest.raises(InvalidToolArgument):
            rig.box.run_python(bad)  # type: ignore[arg-type]
    for timeout in (0, -1, float("nan"), float("inf"), True, "5"):
        with pytest.raises(InvalidToolArgument):
            rig.box.run_python(["-m", "unittest"], timeout_s=timeout)  # type: ignore[arg-type]


@pytest.mark.slow
def test_run_python_executes_view_plus_staged_writes_in_a_scratch_that_is_removed(
    runner: SandboxRunner,
) -> None:
    rig = build(runner=runner)
    rig.box.write_text("app/tool.py", "from app import util\nprint('result', util.helper(41))\n")
    rig.box.write_text("app/__init__.py", "")
    result = rig.box.run_python(["app/tool.py"], timeout_s=60)
    assert result.ok, result.stderr
    assert "result 42" in result.stdout
    assert list(runner.scratch_root.iterdir()) == []
    (call,) = [e for e in rig.audit.of(EventType.TOOL_CALL) if e.data["tool"] == "run_python"]
    assert call.data["returncode"] == 0 and call.data["timed_out"] is False
    assert rig.policy.subjects[-1].action == ToolAction.RUN_PYTHON


@pytest.mark.slow
def test_run_tests_runs_staged_tests_and_reports_failures(runner: SandboxRunner) -> None:
    rig = build(runner=runner)
    rig.box.write_text("tests/test_new.py", "def test_new():\n    assert 1 + 1 == 2\n")
    passed = rig.box.run_tests(["tests/test_new.py"], timeout_s=120)
    assert passed.ok and "1 passed" in passed.stdout
    rig.box.write_text("tests/test_new.py", "def test_new():\n    assert False\n")
    failed = rig.box.run_tests(["tests"], timeout_s=120)
    assert failed.returncode == 1 and not failed.ok
    assert rig.box.read_text("tests/test_new.py").endswith(
        "assert False\n"
    )  # the view never ran stale code
    assert list(runner.scratch_root.iterdir()) == []


@pytest.mark.slow
def test_run_python_times_out_and_reports_it(runner: SandboxRunner) -> None:
    rig = build(runner=runner)
    rig.box.write_text("app/slow.py", "import time\ntime.sleep(300)\n")
    result = rig.box.run_python(["app/slow.py"], timeout_s=2.0)
    assert result.timed_out and not result.ok
    assert list(runner.scratch_root.iterdir()) == []


def test_cancel_mid_run_kills_the_process_cleans_the_scratch_and_leaves_no_trace(
    runner: SandboxRunner,
) -> None:
    rig = build(runner=runner)
    rig.box.write_text(
        "app/sleeper.py",
        "from pathlib import Path\nPath('ready.txt').write_text('1')\n"
        "import time\ntime.sleep(300)\n",
    )
    events_before = len(rig.audit.events)
    outcome: list[BaseException] = []

    def target() -> None:
        try:
            rig.box.run_python(["app/sleeper.py"], timeout_s=120)
        except BaseException as exc:
            outcome.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    deadline = time.monotonic() + 60
    while not list(runner.scratch_root.glob("*/ready.txt")) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(runner.scratch_root.glob("*/ready.txt")), "the child never started"
    rig.cancel.cancel("kill switch")
    worker.join(timeout=60)
    assert not worker.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], AgentCancelled)
    assert list(runner.scratch_root.iterdir()) == []
    assert len(rig.audit.events) == events_before  # nothing was audited after the cancel


# ------------------------------ factory


def _factory_kwargs(rig_view: MemoryView, audit: Recorder) -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "node_id": "implement.a",
        "attempt": 2,
        "agent_id": "impl-agent",
        "profile": PROFILE,
        "view": rig_view,
        "cancel": CancelToken(),
        "clock": FakeClock(),
        "policy": StubPolicy(),
        "audit": audit,
    }


def test_the_factory_builds_per_attempt_toolboxes(tmp_path: Path) -> None:
    factory = make_toolbox_factory(materialise_dir=tmp_path / "scratch")
    audit = Recorder()
    first = factory(**_factory_kwargs(MemoryView(BASE), audit))
    second = factory(**_factory_kwargs(MemoryView(BASE), audit))
    assert isinstance(first, Toolbox) and first is not second
    first.write_text("app/a.py", "1")
    assert second.staged_changeset().ops == ()  # per-attempt state is never shared
    first.read_text("README.md")
    assert audit.events[0].attempt == 2


def test_the_factory_accepts_a_runner_or_a_directory_but_not_both(tmp_path: Path) -> None:
    runner = SandboxRunner(tmp_path / "s")
    make_toolbox_factory(runner=runner)
    make_toolbox_factory()
    with pytest.raises(ConfigError):
        make_toolbox_factory(runner=runner, materialise_dir=tmp_path / "other")


# ------------------------------ integration with the real audit log


def test_events_chain_correctly_in_the_real_audit_log(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", run_id="run-1", clock=FakeClock())
    box = ScopedToolbox(
        run_id="run-1",
        node_id="implement.a",
        attempt=1,
        agent_id="impl-agent",
        profile=PROFILE,
        view=MemoryView(BASE),
        cancel=CancelToken(),
        clock=FakeClock(),
        policy=StubPolicy(),  # type: ignore[arg-type]
        audit=log,
    )
    box.read_text("README.md")
    box.write_text("app/new.py", "x = 1\n")
    box.note("token " + FAKE_KEY)
    with pytest.raises(ToolDenied):
        box.write_text("secret/x.py", "x")
    with pytest.raises(PathViolation):
        box.read_text("../etc/passwd")
    assert log.verify().ok
    kinds = [e.type for e in log.events()]
    assert kinds == [
        EventType.TOOL_CALL,
        EventType.TOOL_CALL,
        EventType.AGENT_NOTE,
        EventType.TOOL_DENIED,
        EventType.TOOL_DENIED,
    ]
    assert FAKE_KEY not in (tmp_path / "audit.jsonl").read_text(
        encoding="utf-8"
    )  # the log redacts it
    assert {e.span_id for e in log.events()} == {"n:implement.a#1"}
