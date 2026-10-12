"""The per-attempt ``Toolbox``: the *only* way an agent touches files or runs code.

Every public method, in this order: (1) honours the cancel token (``AgentCancelled``; audit
events are dropped once cancelled), (2) honours the poison flag left by an earlier
``AutonomyViolation`` (an agent that swallows the exception cannot carry on), (3) counts against
the call / runtime budget, (4) validates paths with ``validate_rel_path`` (an unsafe path is a
``PathViolation`` - CRITICAL), (5) checks the autonomy profile (``ToolDenied`` for a valid but
out-of-scope request; the third denial in one attempt escalates to ``AutonomyViolation``),
(6) lets the policy engine evaluate an ``ActionSubject`` (``SAFE_STOP`` -> ``AutonomyViolation``,
``DENY`` / ``REQUIRE_APPROVAL`` -> ``ToolDenied``: nobody can approve in the middle of a call, so
the answer fails closed), and only then (7) performs the action and audits ``TOOL_CALL``.

Reads see ``view`` (``view_for(node)``) plus this attempt's own staged writes; writes are staged,
nothing on disk changes.  ``staged_changeset`` diffs the staged state against the attempt's
*start* view with ``compute_ops`` (first-touch ``base_hash``; create-then-modify stays CREATE;
create-then-delete vanishes).  ``run_tests`` / ``run_python`` materialise ``view + staged writes``
into a fresh scratch directory (always removed) and go through the ``SandboxRunner``.

Execution modes (fail-to-pass checks): ``mode="view"`` (default) materialises ``view + staged
writes``; ``mode="baseline_plus"`` materialises the run's BASELINE tree plus *only* the files named
in ``overlay_paths``, taken from the view (so "the new tests against the old code" is one call).
The baseline is handed in by the engine's factory; a toolbox built without one treats its start
view as the baseline.  ``baseline_view()`` (not part of the ``types.Toolbox`` protocol; reach it
through :func:`baseline_view_of`) gives agents that must render a patch against the baseline a
read-only view of it that obeys the same read scope, budget and audit rules as ``read_text``.

Instances are used by one worker thread (the agent's); the cancel token is the only object shared
with other threads.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.core.errors import (
    AgentError,
    AgentTimeout,
    AutonomyViolation,
    ConfigError,
    PathViolation,
    ToolDenied,
)
from orchestrator.core.paths import casefold_key, compile_glob, match_any, validate_rel_path
from orchestrator.core.sanitize import sanitize_for_terminal, strip_controls
from orchestrator.core.serde import canonical_json, to_jsonable
from orchestrator.core.types import (
    MAX_CHANGESET_OPS,
    MAX_FILE_BYTES,
    ActionSubject,
    Actor,
    AuditSink,
    AutonomyProfile,
    CancelToken,
    ChangeSet,
    Clock,
    CommandResult,
    Enforcement,
    EventType,
    FileView,
    PolicyEvaluator,
    SearchHit,
    ToolAction,
    Toolbox,
    ToolboxFactory,
)
from orchestrator.tools.changeset import compute_ops, normalize_text
from orchestrator.tools.sandbox import SandboxArgError, SandboxRunner
from orchestrator.tools.workspace import StagedView, materialize

MAX_DENIALS = 3  # the Nth denial in one attempt becomes an AutonomyViolation
MAX_PATTERN_CHARS = 200
MAX_REGEX_GROUPS = 10
MAX_REGEX_WIDE_REPEATS = 3  # *, +, {m,} and {m,n} with n > m: the only sources of polynomial cost
MAX_REGEX_OPTIONALS = 8
MAX_REGEX_BOUND = 100
MAX_SEARCH_RESULTS = 1000
LITERAL_LINE_CHARS = 2000
REGEX_LINE_CHARS = 200  # regexes only ever see the first 200 characters of a line
LITERAL_SCAN_CHARS = 8 * 1024 * 1024
REGEX_SCAN_CHARS = 2 * 1024 * 1024
HIT_TEXT_CHARS = 240
DEFAULT_EXEC_TIMEOUT_S = 120.0
MAX_EXEC_TIMEOUT_S = 600.0
EXEC_MODES = ("view", "baseline_plus")
MAX_OVERLAY_PATHS = 500
NOTE_MAX_CHARS = 500
NOTE_DATA_MAX_BYTES = 4096
DESCRIPTION_MAX_CHARS = 500
_AUDIT_TARGET_CHARS = 200

# Wall-clock safety net for ``search`` (independent of the injected clock): the regex grammar
# bounds the cost of one line, this bounds the total.
_SEARCH_TIME_BUDGET_S = 3.0


class BudgetExhausted(AgentError):
    """The attempt used up its tool-call budget.  Not retried on the same agent (the same budget
    would be exhausted again) and not a denial: it never escalates to ``AutonomyViolation``."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="policy_violation", retryable=False)


class InvalidToolArgument(AgentError, ValueError):
    """The agent called a tool with an unusable argument (bad type, unsupported regex, ...).

    An agent bug rather than a boundary probe: FATAL (skip to the fallback agent), not a denial.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="fatal", retryable=False)


# ----------------------------------------------------------------------------- regex grammar


@dataclass
class _Frame:
    """What one parenthesised group contains (needed to reject ``(a+)+`` and ``(a|aa)*``)."""

    repeated: bool = False
    alternated: bool = False


_BRACE_RE = re.compile(r"\{([0-9]*)(?:(,)([0-9]*))?\}")
_SAFE_ESCAPES = frozenset("dDwWsSbB")


def _brace_repeat(pattern: str, i: int) -> tuple[int, int, int | None] | None:
    """``(length, min, max)`` of a ``{...}`` repeat, read exactly the way ``re`` reads it.

    ``None`` when ``re`` treats the brace as a literal (``{}``, ``{abc}``, an unclosed ``{``).  Any
    disagreement with ``re`` here would let a repeat slip past the nesting checks, so the grammar is
    the one of ``sre_parse``: ASCII digits only, ``{,n}`` and ``{,}`` are repeats too.
    """
    found = _BRACE_RE.match(pattern, i)
    if found is None or found.group(0) == "{}":
        return None
    low = int(found.group(1)) if found.group(1) else 0
    if found.group(2) is None:
        high: int | None = low
    else:
        high = int(found.group(3)) if found.group(3) else None
    return found.end() - i, low, high


def _check_escape(pattern: str, i: int) -> None:
    if i + 1 >= len(pattern):
        raise InvalidToolArgument("regex ends with a lone backslash")
    nxt = pattern[i + 1]
    if nxt.isalnum() and nxt not in _SAFE_ESCAPES:
        raise InvalidToolArgument(
            f"regex escape \\{nxt} is not supported (no backreferences or numeric escapes)"
        )


def _skip_class(pattern: str, start: int) -> int:
    """Index just past the ``[...]`` class starting at ``start`` (no nested classes)."""
    i = start + 1
    if pattern[i : i + 1] == "^":
        i += 1
    if pattern[i : i + 1] == "]":
        i += 1  # a leading ']' is a literal
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            _check_escape(pattern, i)
            i += 2
            continue
        if ch == "[":
            raise InvalidToolArgument("nested character classes are not supported in regex")
        if ch == "]":
            return i + 1
        i += 1
    raise InvalidToolArgument("unterminated character class in regex")


def compile_search_regex(pattern: str) -> re.Pattern[str]:
    """Compile ``pattern`` if (and only if) it stays inside the bounded search grammar.

    Allowed: literals, ``.``, ``^``, ``$``, ``\\d \\w \\s \\b`` (and their negations), escaped
    punctuation, character classes, (non-)capturing groups, alternation and the repetition
    operators ``* + ? {m} {m,} {m,n}`` (each optionally lazy).  Rejected with
    ``InvalidToolArgument``: more than 200 characters, backreferences / numeric / other
    alphanumeric escapes, lookarounds and inline flags, a repeated group that itself contains a
    repetition or an alternation (the exponential shapes ``(a+)+`` / ``(a|a)*``), more than three
    wide repetitions (polynomial blow-up), repeat bounds above 100.
    """
    if not isinstance(pattern, str) or not pattern:
        raise InvalidToolArgument("regex pattern must be a non-empty string")
    if len(pattern) > MAX_PATTERN_CHARS:
        raise InvalidToolArgument(f"regex pattern longer than {MAX_PATTERN_CHARS} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in pattern):
        raise InvalidToolArgument("control character in regex pattern")

    stack = [_Frame()]
    target: _Frame | bool = False  # what a repetition operator would repeat
    after_repeat = False
    wide = optionals = groups = 0
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        step = 1
        quantifier: tuple[str, int, int | None] | None = None  # (kind, min, max)
        if ch == "\\":
            _check_escape(pattern, i)
            step, target, after_repeat = 2, True, False
        elif ch == "[":
            step, target, after_repeat = _skip_class(pattern, i) - i, True, False
        elif ch == "(":
            if pattern.startswith("(?", i):
                if not pattern.startswith("(?:", i):
                    raise InvalidToolArgument(
                        "lookarounds, named groups and inline flags are not supported in regex"
                    )
                step = 3
            else:
                groups += 1
                if groups > MAX_REGEX_GROUPS:
                    raise InvalidToolArgument(f"more than {MAX_REGEX_GROUPS} groups in regex")
            stack.append(_Frame())
            target, after_repeat = False, False
        elif ch == ")":
            if len(stack) == 1:
                raise InvalidToolArgument("unbalanced ')' in regex")
            closed = stack.pop()
            stack[-1].repeated |= closed.repeated
            stack[-1].alternated |= closed.alternated
            target, after_repeat = closed, False
        elif ch == "|":
            stack[-1].alternated = True
            target, after_repeat = False, False
        elif ch in "^$":
            target, after_repeat = False, False
        elif ch in "*+?":
            quantifier = (ch, 0 if ch != "+" else 1, None if ch != "?" else 1)
        elif ch == "{" and (brace := _brace_repeat(pattern, i)) is not None:
            step, low, high = brace
            if max(low, high or 0) > MAX_REGEX_BOUND:
                raise InvalidToolArgument(f"regex repeat bound above {MAX_REGEX_BOUND}")
            quantifier = ("{", low, high)
        else:
            target, after_repeat = True, False  # '.', a literal, or a '{' that is not a repeat
        if quantifier is not None:
            if after_repeat:
                if ch != "?":
                    raise InvalidToolArgument("stacked repetition operators in regex")
                target, after_repeat = False, False  # '?' after a repeat makes it lazy
            else:
                if target is False:
                    raise InvalidToolArgument("regex repetition operator has nothing to repeat")
                kind, low, high = quantifier
                many = high is None or high > 1  # '?' repeats at most once: no nesting hazard
                if many and isinstance(target, _Frame) and (target.repeated or target.alternated):
                    raise InvalidToolArgument(
                        "nested repetition or alternation under a repeat is not supported in regex"
                    )
                if kind == "?":
                    optionals += 1
                elif high is None or high > low:
                    wide += 1
                if wide > MAX_REGEX_WIDE_REPEATS or optionals > MAX_REGEX_OPTIONALS:
                    raise InvalidToolArgument("too many repetition operators in regex")
                stack[-1].repeated = True
                after_repeat = True
        i += step
    if len(stack) != 1:
        raise InvalidToolArgument("unbalanced '(' in regex")
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise InvalidToolArgument(f"invalid regex: {exc}") from exc


# ----------------------------------------------------------------------------- toolbox


def _clip(text: str, limit: int = _AUDIT_TARGET_CHARS) -> str:
    return sanitize_for_terminal(text, limit)


@dataclass(frozen=True)
class _ExecPlan:
    """What ``run_tests`` / ``run_python`` materialise: ``view`` or ``baseline_plus`` + overlay."""

    mode: str
    overlay: tuple[str, ...]


class ScopedToolbox:
    """Autonomy-bounded, audited, cancel-aware implementation of ``types.Toolbox``."""

    def __init__(
        self,
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        agent_id: str,
        profile: AutonomyProfile,
        view: FileView,
        cancel: CancelToken,
        clock: Clock,
        policy: PolicyEvaluator,
        audit: AuditSink,
        runner: SandboxRunner | None = None,
        baseline: FileView | None = None,
    ) -> None:
        self.run_id = run_id
        self._node_id = node_id
        self._attempt = attempt
        self._agent_id = agent_id
        self._profile = profile
        self._view = view  # the attempt's START view: the base of every proposed change
        self._stage = StagedView(view)
        self._cancel = cancel
        self._clock = clock
        self._policy = policy
        self._audit = audit
        self._runner = runner
        self._baseline = baseline  # None: the start view stands in for the baseline
        self._actor = Actor.agent(agent_id)
        self._started = clock.monotonic()
        self._calls = 0
        self._denials = 0
        self._bytes_written = 0
        self._violation: AutonomyViolation | None = None
        self._sizes: dict[str, int] = {}  # staged path -> bytes (None-writes excluded)
        self._changed: set[str] = set()  # staged paths whose content differs from the start view
        self._folded: dict[str, str] | None = None  # casefold_key -> path, built on first write

    # -- audit / failure plumbing ---------------------------------------------
    def _emit(self, etype: EventType, data: dict[str, Any]) -> None:
        if self._cancel.is_cancelled():  # a cancelled attempt leaves no further trace
            return
        self._audit(
            etype, node_id=self._node_id, attempt=self._attempt, actor=self._actor, data=data
        )

    def _record_call(self, tool: str, target: str, **facts: Any) -> None:
        self._emit(EventType.TOOL_CALL, {"tool": tool, "target": _clip(target), **facts})

    def _violate(self, tool: str, target: str, exc: AutonomyViolation) -> AutonomyViolation:
        """Poison the toolbox, audit and return ``exc`` (CRITICAL: the engine safe-stops)."""
        self._violation = exc
        self._emit(
            EventType.TOOL_DENIED,
            {
                "tool": tool,
                "target": _clip(target),
                "reason": _clip(str(exc), 300),
                "critical": True,
            },
        )
        return exc

    def _deny(self, tool: str, target: str, reason: str) -> AgentError:
        """Count a denial; the ``MAX_DENIALS``-th one is escalated to a violation."""
        self._denials += 1
        if self._denials >= MAX_DENIALS:
            return self._violate(
                tool,
                target,
                AutonomyViolation(
                    f"{self._denials} denied tool calls in one attempt (last: {reason})",
                    details={"denials": self._denials},
                ),
            )
        self._emit(
            EventType.TOOL_DENIED,
            {"tool": tool, "target": _clip(target), "reason": reason, "denials": self._denials},
        )
        return ToolDenied(f"{tool} denied: {reason}", details={"denials": self._denials})

    def _checkpoint(self) -> None:
        self._cancel.raise_if_cancelled()
        if self._violation is not None:
            raise self._violation
        if self._elapsed() > self._profile.max_runtime_s:
            raise AgentTimeout(
                f"attempt exceeded its runtime budget of {self._profile.max_runtime_s:g}s"
            )

    def _elapsed(self) -> float:
        return self._clock.monotonic() - self._started

    def _begin(self, tool: str) -> None:
        self._checkpoint()
        if self._calls >= self._profile.max_tool_calls:
            self._emit(
                EventType.TOOL_DENIED,
                {"tool": tool, "reason": "tool call budget exhausted", "budget": True},
            )
            raise BudgetExhausted(f"tool call budget of {self._profile.max_tool_calls} exhausted")
        self._calls += 1

    def _safe_path(self, tool: str, path: str) -> str:
        """``validate_rel_path`` with the violation recorded (and the toolbox poisoned)."""
        try:
            return validate_rel_path(path)
        except PathViolation as exc:
            raise self._violate(tool, path if isinstance(path, str) else repr(path), exc) from exc

    def _safe_glob(self, tool: str, glob: str) -> str:
        try:
            compile_glob(glob)
        except PathViolation as exc:
            raise self._violate(tool, glob if isinstance(glob, str) else repr(glob), exc) from exc
        return glob

    def _evaluate(self, tool: str, action: ToolAction, target: str) -> None:
        """Ask the policy engine; fail closed on anything but ALLOW / WARN."""
        subject = ActionSubject(self._agent_id, self._node_id, action, target, self._profile)
        try:
            report = self._policy.evaluate(subject)
        except Exception as exc:
            self._emit(
                EventType.TOOL_DENIED,
                {"tool": tool, "target": _clip(target), "reason": "policy evaluation failed"},
            )
            raise AgentError("policy evaluation failed", kind="fatal", retryable=False) from exc
        if report.enforcement in (Enforcement.ALLOW, Enforcement.WARN):
            return
        rules = (
            ", ".join(sorted({f.rule_id for f in (*report.blocking, *report.needs_approval)}))
            or "policy"
        )
        if report.enforcement == Enforcement.SAFE_STOP:
            raise self._violate(
                tool, target, AutonomyViolation(f"policy requires a safe stop ({rules})")
            )
        raise self._deny(tool, target, f"policy {report.enforcement.value} ({rules})")

    # -- scope checks -----------------------------------------------------------
    def _readable(self, rel: str) -> bool:
        """In ``read_globs``, or content this attempt itself staged (read-your-own-writes)."""
        return match_any(rel, self._profile.read_globs) or self._stage.writes.get(rel) is not None

    def _require_readable(self, tool: str, rel: str) -> None:
        if not self._readable(rel):
            raise self._deny(tool, rel, "path is outside read_globs")

    def _require_writable(self, tool: str, rel: str) -> None:
        if not match_any(rel, self._profile.write_globs):
            raise self._deny(tool, rel, "path is outside write_globs")

    # -- reads -------------------------------------------------------------------
    def read_text(self, path: str) -> str:
        self._begin("read_text")
        rel = self._safe_path("read_text", path)
        self._require_readable("read_text", rel)
        self._evaluate("read_text", ToolAction.READ, rel)
        try:
            text = self._stage.read_text(rel)
        except FileNotFoundError:
            self._record_call("read_text", rel, ok=False, error="not_found")
            raise
        size = len(text.encode("utf-8"))
        if size > self._profile.max_file_bytes:
            raise self._deny(
                "read_text",
                rel,
                f"file is larger than max_file_bytes ({self._profile.max_file_bytes})",
            )
        self._record_call("read_text", rel, ok=True, bytes=size)
        return text

    def exists(self, path: str) -> bool:
        self._begin("exists")
        rel = self._safe_path("exists", path)
        self._require_readable("exists", rel)
        self._evaluate("exists", ToolAction.READ, rel)
        found = self._stage.exists(rel)
        self._record_call("exists", rel, ok=True, exists=found)
        return found

    def list_files(self, glob: str = "**") -> list[str]:
        self._begin("list_files")
        pattern = self._safe_glob("list_files", glob)
        self._evaluate("list_files", ToolAction.READ, pattern)
        files = [p for p in self._stage.list_files(pattern) if self._readable(p)]
        self._record_call("list_files", pattern, ok=True, count=len(files))
        return files

    # -- baseline reads ----------------------------------------------------------
    def baseline_view(self) -> FileView:
        """Read-only view of the run's BASELINE tree (the start view when no baseline was given).

        Every read goes through the same checks as ``read_text``: cancel token, call budget, path
        validation, ``read_globs``, policy evaluation and audit.  Staged writes are never visible.
        """
        self._checkpoint()
        return _BaselineReader(self)

    def _baseline_source(self) -> FileView:
        return self._baseline if self._baseline is not None else self._view

    def _baseline_path(self, tool: str, path: str) -> str:
        rel = self._safe_path(tool, path)
        if not match_any(rel, self._profile.read_globs):
            raise self._deny(tool, rel, "path is outside read_globs")
        self._evaluate(tool, ToolAction.READ, rel)
        return rel

    def _baseline_exists(self, path: str) -> bool:
        self._begin("baseline_exists")
        rel = self._baseline_path("baseline_exists", path)
        found = self._baseline_source().exists(rel)
        self._record_call("baseline_exists", rel, ok=True, exists=found)
        return found

    def _baseline_text(self, path: str) -> str:
        self._begin("baseline_read_text")
        rel = self._baseline_path("baseline_read_text", path)
        try:
            text = self._baseline_source().read_text(rel)
        except FileNotFoundError:
            self._record_call("baseline_read_text", rel, ok=False, error="not_found")
            raise
        size = len(text.encode("utf-8"))
        if size > self._profile.max_file_bytes:
            raise self._deny(
                "baseline_read_text",
                rel,
                f"file is larger than max_file_bytes ({self._profile.max_file_bytes})",
            )
        self._record_call("baseline_read_text", rel, ok=True, bytes=size)
        return text

    def _baseline_files(self, glob: str) -> list[str]:
        self._begin("baseline_list_files")
        pattern = self._safe_glob("baseline_list_files", glob)
        self._evaluate("baseline_list_files", ToolAction.READ, pattern)
        files = [
            p
            for p in self._baseline_source().list_files(pattern)
            if match_any(p, self._profile.read_globs)
        ]
        self._record_call("baseline_list_files", pattern, ok=True, count=len(files))
        return files

    def search(
        self, pattern: str, glob: str = "**/*.py", max_results: int = 100, *, regex: bool = False
    ) -> list[SearchHit]:
        self._begin("search")
        line_cap, scan_cap = (
            (REGEX_LINE_CHARS, REGEX_SCAN_CHARS)
            if regex
            else (LITERAL_LINE_CHARS, LITERAL_SCAN_CHARS)
        )
        matches = self._matcher(pattern, regex, line_cap)
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise InvalidToolArgument("max_results must be a positive integer")
        limit = min(max_results, MAX_SEARCH_RESULTS)
        selector = self._safe_glob("search", glob)
        self._evaluate("search", ToolAction.READ, selector)
        hits, complete = self._scan(self._stage.list_files(selector), matches, limit, scan_cap)
        self._record_call(
            "search", selector, ok=True, hits=len(hits), regex=bool(regex), complete=complete
        )
        return hits

    @staticmethod
    def _matcher(pattern: str, regex: bool, line_cap: int) -> Callable[[str], bool]:
        if not regex:
            if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN_CHARS:
                raise InvalidToolArgument(f"search text must be 1-{MAX_PATTERN_CHARS} characters")
            return lambda line: pattern in line
        compiled = compile_search_regex(pattern)
        return lambda line: compiled.search(line[:line_cap]) is not None

    def _scan(
        self, candidates: Sequence[str], matches: Callable[[str], bool], limit: int, scan_cap: int
    ) -> tuple[list[SearchHit], bool]:
        """Bounded scan: result count, characters scanned and wall-clock time are all capped."""
        hits: list[SearchHit] = []
        scanned = 0
        deadline = time.monotonic() + _SEARCH_TIME_BUDGET_S
        for rel in candidates:
            if not self._readable(rel):
                continue
            try:
                text = self._stage.read_text(rel)
            except OSError:  # includes FileNotFoundError: a file vanished from the view
                continue
            scanned += len(text)
            if scanned > scan_cap:
                return hits, False
            for lineno, line in enumerate(text.split("\n"), 1):
                if time.monotonic() > deadline:
                    return hits, False
                if matches(line):
                    hits.append(SearchHit(rel, lineno, strip_controls(line[:HIT_TEXT_CHARS])))
                    if len(hits) >= limit:
                        return hits, True
        return hits, True

    # -- writes ------------------------------------------------------------------
    def write_text(self, path: str, content: str) -> None:
        self._begin("write_text")
        rel = self._safe_path("write_text", path)
        self._require_writable("write_text", rel)
        text, size = self._checked_content(content)
        self._require_no_case_collision("write_text", rel)
        self._require_no_file_directory_clash("write_text", rel)
        if size > min(self._profile.max_file_bytes, MAX_FILE_BYTES):
            raise self._deny("write_text", rel, "content exceeds max_file_bytes")
        sizes = {**self._sizes, rel: size}
        if sum(sizes.values()) > self._profile.max_total_bytes:
            raise self._deny("write_text", rel, "staged content exceeds max_total_bytes")
        changed = self._changed_after(rel, text)
        if len(changed) > min(self._profile.max_changed_files, MAX_CHANGESET_OPS):
            raise self._deny("write_text", rel, "too many changed files (max_changed_files)")
        self._evaluate("write_text", ToolAction.WRITE, rel)
        self._stage.writes[rel] = text
        self._fold_map()[casefold_key(rel)] = rel
        self._sizes, self._changed = sizes, changed
        self._bytes_written += size
        self._record_call("write_text", rel, ok=True, bytes=size)

    def delete(self, path: str) -> None:
        self._begin("delete")
        rel = self._safe_path("delete", path)
        self._require_writable("delete", rel)
        self._require_no_case_collision("delete", rel)
        if not self._stage.exists(rel):
            self._record_call("delete", rel, ok=False, error="not_found")
            raise FileNotFoundError(rel)
        changed = self._changed_after(rel, None)
        self._evaluate("delete", ToolAction.DELETE, rel)
        self._stage.writes[rel] = None
        self._sizes.pop(rel, None)
        self._changed = changed
        self._record_call("delete", rel, ok=True)

    @staticmethod
    def _checked_content(content: str) -> tuple[str, int]:
        if not isinstance(content, str):
            raise InvalidToolArgument("content must be a string")
        if "\x00" in content:
            raise InvalidToolArgument("content must not contain NUL")
        text = normalize_text(content)
        try:
            return text, len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise InvalidToolArgument("content is not valid UTF-8") from exc

    def _start_text(self, rel: str) -> str | None:
        try:
            return normalize_text(self._view.read_text(rel))
        except FileNotFoundError:
            return None

    def _changed_after(self, rel: str, new: str | None) -> set[str]:
        """The set of changed paths if ``rel`` were staged as ``new`` (a no-op leaves no op)."""
        changed = set(self._changed)
        if new == self._start_text(rel):
            changed.discard(rel)
        else:
            changed.add(rel)
        return changed

    def _require_no_case_collision(self, tool: str, rel: str) -> None:
        """Two spellings that differ only by case/normalisation collide on Windows and macOS."""
        if self._fold_map().get(casefold_key(rel), rel) != rel:
            raise self._deny(tool, rel, "path collides case-insensitively with an existing path")

    def _require_no_file_directory_clash(self, tool: str, rel: str) -> None:
        """A path cannot be a file and a directory at once (the algebra works on flat paths, so
        nothing downstream would notice until a patch or a real tree refuses it)."""
        parts = rel.split("/")
        if any(self._stage.exists("/".join(parts[:i])) for i in range(1, len(parts))):
            raise self._deny(tool, rel, "a parent path is an existing file")
        prefix = rel + "/"
        if any(p.startswith(prefix) for p in self._stage.list_files("**")):
            raise self._deny(tool, rel, "path is an existing directory")

    def _fold_map(self) -> dict[str, str]:
        if self._folded is None:
            self._folded = {casefold_key(p): p for p in self._view.list_files("**")}
        return self._folded

    # -- results -------------------------------------------------------------------
    def staged_changeset(self, description: str = "") -> ChangeSet:
        self._checkpoint()
        text = (
            sanitize_for_terminal(description, DESCRIPTION_MAX_CHARS)
            if isinstance(description, str)
            else ""
        )
        return compute_ops(
            self._view, dict(self._stage.writes), description=text, author=self._agent_id
        )

    def usage(self) -> Mapping[str, float]:
        self._cancel.raise_if_cancelled()
        return {
            "tool_calls": float(self._calls),
            "bytes_written": float(self._bytes_written),
            "runtime_s": self._elapsed(),
        }

    def note(self, message: str, **data: Any) -> None:
        self._begin("note")
        self._emit(
            EventType.AGENT_NOTE,
            {
                "message": sanitize_for_terminal(message, NOTE_MAX_CHARS),
                "data": _bounded_data(data),
            },
        )

    # -- execution -----------------------------------------------------------------
    def run_tests(
        self,
        targets: Sequence[str] = (),
        *,
        timeout_s: float | None = None,
        mode: str = "view",
        overlay_paths: Sequence[str] = (),
    ) -> CommandResult:
        self._begin("run_tests")
        checked = self._test_targets(targets)
        plan = self._exec_plan("run_tests", mode, overlay_paths)
        label = " ".join(checked) or "."
        runner = self._execution_runner("run_tests", label)
        self._pregrammar(
            "run_tests", label, lambda: runner.pytest_argv(runner.scratch_root, checked)
        )
        timeout = self._exec_timeout(timeout_s)
        self._evaluate("run_tests", ToolAction.RUN_TESTS, label)
        result = self._execute(
            runner, lambda scratch: runner.pytest_argv(scratch, checked), timeout, plan
        )
        self._record_run("run_tests", label, result, plan)
        return result

    def run_python(
        self,
        args: Sequence[str],
        *,
        timeout_s: float | None = None,
        mode: str = "view",
        overlay_paths: Sequence[str] = (),
    ) -> CommandResult:
        self._begin("run_python")
        if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
            raise InvalidToolArgument("args must be a sequence of strings")
        plan = self._exec_plan("run_python", mode, overlay_paths)
        label = _clip(" ".join(a if isinstance(a, str) else repr(a) for a in args))
        runner = self._execution_runner("run_python", label)
        self._pregrammar(
            "run_python", label, lambda: runner.python_argv(args, scratch=runner.scratch_root)
        )
        timeout = self._exec_timeout(timeout_s)
        self._evaluate("run_python", ToolAction.RUN_PYTHON, label)
        result = self._execute(
            runner, lambda scratch: runner.python_argv(args, scratch=scratch), timeout, plan
        )
        self._record_run("run_python", label, result, plan)
        return result

    def _exec_plan(self, tool: str, mode: str, overlay_paths: Sequence[str]) -> _ExecPlan:
        """Validate the execution mode and the overlay files *before* anything is materialised.

        Overlay paths are validated like any other path (an unsafe one is a ``PathViolation``), must
        be readable under ``read_globs`` and must exist in the view (view + staged writes).
        """
        if mode not in EXEC_MODES:
            raise InvalidToolArgument(f"mode must be one of {list(EXEC_MODES)}")
        if isinstance(overlay_paths, (str, bytes)) or not isinstance(overlay_paths, Sequence):
            raise InvalidToolArgument("overlay_paths must be a sequence of strings")
        if mode == "view":
            if overlay_paths:
                raise InvalidToolArgument("overlay_paths is only valid with mode='baseline_plus'")
            return _ExecPlan("view", ())
        if len(overlay_paths) > MAX_OVERLAY_PATHS:
            raise InvalidToolArgument(f"more than {MAX_OVERLAY_PATHS} overlay paths")
        checked: dict[str, None] = {}
        for raw in overlay_paths:
            rel = self._safe_path(tool, raw)
            self._require_readable(tool, rel)
            if not self._stage.exists(rel):
                raise InvalidToolArgument(f"overlay path does not exist in the view: {_clip(rel)}")
            checked.setdefault(rel, None)
        return _ExecPlan("baseline_plus", tuple(sorted(checked)))

    def _test_targets(self, targets: Sequence[str]) -> list[str]:
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
            raise InvalidToolArgument("targets must be a sequence of strings")
        return [t if t == "." else self._safe_path("run_tests", t) for t in targets]

    def _execution_runner(self, tool: str, label: str) -> SandboxRunner:
        if not self._profile.can_execute:
            raise self._deny(tool, label, "execution is not permitted by the autonomy profile")
        if self._runner is None:
            raise ConfigError("this toolbox was built without a sandbox runner")
        return self._runner

    def _pregrammar(self, tool: str, label: str, build: Callable[[], list[str]]) -> None:
        """Reject hostile arguments *before* anything is materialised."""
        try:
            build()
        except SandboxArgError as exc:
            raise self._deny(
                tool, label, f"rejected by the sandbox grammar: {_clip(str(exc), 160)}"
            ) from exc

    def _exec_timeout(self, requested: float | None) -> float:
        if requested is None:
            requested = DEFAULT_EXEC_TIMEOUT_S
        elif (
            isinstance(requested, bool)
            or not isinstance(requested, (int, float))
            or not 0 < requested < float("inf")
        ):
            raise InvalidToolArgument("timeout_s must be a positive finite number")
        remaining = self._profile.max_runtime_s - self._elapsed()
        return max(0.001, min(float(requested), MAX_EXEC_TIMEOUT_S, remaining))

    def _execute(
        self,
        runner: SandboxRunner,
        build_argv: Callable[[Path], list[str]],
        timeout_s: float,
        plan: _ExecPlan,
    ) -> CommandResult:
        scratch = runner.new_scratch()
        try:
            self._materialise(scratch, plan)
            return runner.run(
                build_argv(scratch), cwd=scratch, timeout_s=timeout_s, cancel=self._cancel
            )
        finally:
            runner.discard(scratch)

    def _materialise(self, scratch: Path, plan: _ExecPlan) -> None:
        """``view``: the view plus staged writes.  ``baseline_plus``: the baseline tree, then ONLY
        the overlay files (taken from the view plus staged writes) on top of it."""
        if plan.mode == "view":
            materialize(self._stage, scratch)
            return
        materialize(self._baseline_source(), scratch)
        try:
            materialize(self._stage, scratch, plan.overlay)
        except OSError as exc:  # e.g. an overlay file would have to replace a baseline directory
            raise InvalidToolArgument("an overlay path conflicts with the baseline tree") from exc

    def _record_run(self, tool: str, label: str, result: CommandResult, plan: _ExecPlan) -> None:
        self._record_call(
            tool,
            label,
            ok=result.ok,
            returncode=result.returncode,
            timed_out=result.timed_out,
            truncated=result.truncated,
            duration_s=round(result.duration_s, 3),
            mode=plan.mode,
            overlay=len(plan.overlay),
        )


class _BaselineReader:
    """``FileView`` over the baseline whose every call is a guarded toolbox call."""

    def __init__(self, box: ScopedToolbox) -> None:
        self._box = box

    def exists(self, path: str) -> bool:
        return self._box._baseline_exists(path)

    def read_text(self, path: str) -> str:
        return self._box._baseline_text(path)

    def list_files(self, glob: str = "**") -> list[str]:
        return self._box._baseline_files(glob)


def baseline_view_of(toolbox: object) -> FileView | None:
    """The guarded baseline view of ``toolbox``; ``None`` for implementations that have none
    (``baseline_view`` is an extension of ``ScopedToolbox``, not part of ``types.Toolbox``)."""
    getter = getattr(toolbox, "baseline_view", None)
    if not callable(getter):
        return None
    view = getter()
    return view if isinstance(view, FileView) else None


def _bounded_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe, size-capped copy of agent-supplied note data (never raises on odd values)."""
    try:
        encoded = canonical_json(data)
    except (TypeError, ValueError):
        return {"_dropped": "note data is not JSON-serialisable"}
    if len(encoded) > NOTE_DATA_MAX_BYTES:
        return {"_dropped": f"note data larger than {NOTE_DATA_MAX_BYTES} bytes"}
    result = to_jsonable(data)
    return result if isinstance(result, dict) else {}


def make_toolbox_factory(
    *, runner: SandboxRunner | None = None, materialise_dir: Path | str | None = None
) -> ToolboxFactory:
    """Build the engine's ``ToolboxFactory``.

    ``run_tests`` / ``run_python`` need a ``SandboxRunner``: pass one, or ``materialise_dir`` (the
    root under which per-call scratch directories are created).  With neither, execution tools
    raise ``ConfigError`` (only agents whose profile has ``can_execute`` ever reach that point).
    """
    if runner is not None and materialise_dir is not None:
        raise ConfigError("pass either a SandboxRunner or materialise_dir, not both")
    exec_runner = (
        runner
        if runner is not None
        else (SandboxRunner(materialise_dir) if materialise_dir else None)
    )

    def factory(
        *,
        run_id: str,
        node_id: str,
        attempt: int,
        agent_id: str,
        profile: AutonomyProfile,
        view: FileView,
        cancel: CancelToken,
        clock: Clock,
        policy: PolicyEvaluator,
        audit: AuditSink,
        baseline: FileView | None = None,
    ) -> Toolbox:
        return ScopedToolbox(
            run_id=run_id,
            node_id=node_id,
            attempt=attempt,
            agent_id=agent_id,
            profile=profile,
            view=view,
            cancel=cancel,
            clock=clock,
            policy=policy,
            audit=audit,
            runner=exec_runner,
            baseline=baseline,
        )

    return factory


__all__ = [
    "BudgetExhausted",
    "InvalidToolArgument",
    "ScopedToolbox",
    "baseline_view_of",
    "compile_search_regex",
    "make_toolbox_factory",
]
