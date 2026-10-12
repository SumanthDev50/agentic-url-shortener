"""Verifier agent (``verify`` node; with ``release-verifier`` the only agent that executes).

What it does, in order (all execution goes through ``ctx.toolbox``, i.e. the sandbox):

1. Runs the test suite of the integrated tree (view = baseline + ``integrated_changeset``) with
   ``toolbox.run_tests(["tests"])`` and parses the ``-rA`` output into ``tests[{nodeid, outcome,
   covers}]``.  Only the *last* ``short test summary info`` section is read and its entries must
   agree with the counts of pytest's final line, so text printed by a test cannot forge a pass
   (documented limit: agent code is trusted code and the sandbox is process isolation, not a
   security boundary).  ``covers`` comes from ``Covers: R-001, AC-001-2`` lines in the docstrings
   of the test functions (``ast``).
2. FAIL-TO-PASS.  *Authored* tests are the test files (``tests/**/test_*.py`` / ``*_test.py``)
   that the integrated change set CREATES.  They run with ``mode="baseline_plus"`` (baseline plus
   only those files).  A test is *fail-to-pass* when it fails (or cannot be collected) on the
   baseline and passes in the integrated run; ``ok`` needs every authored test to be one.  Tests
   that already pass on the baseline are listed as ``vacuous`` (an extension of the schema).
   Files whose tests were not reached because another file broke collection are re-run in further
   rounds (at most ``MAX_BASELINE_ROUNDS``).  In a greenfield run the baseline is empty, so every
   authored test fails there by construction.  ``baseline.passed`` is False when any authored test
   failed on the baseline, which is the normal, expected value.
3. ``api_compat``.  When a task-plan capability's recipe declares ``api_changes`` and the product
   (``params['app_import']``, default ``shortener.main:create_app``) is in the tree, the OpenAPI
   document is exported on the baseline and on the integrated tree with a staged helper script
   (``run_python``; self-contained, the toolbox cannot run ``-m`` helper modules) and diffed
   with ``tools.openapi.diff_openapi``.  The shortener refuses to start without API keys, so for
   ``shortener.*`` apps the helper hands it a throwaway key generated at run time inside the
   sandbox process (authentication stays enabled; no secret is ever a literal).  If the export is
   impossible (no product, a failing factory, a greenfield baseline, ...) the report says
   ``checked: false`` and the reason goes to the artifact *metadata*.
4. ``perf``.  With ``params['perf_budget_ms']`` (or a capability parameter of that name) and at
   least one test whose function name contains ``perf``: ``checked``, ``budget_ms``, ``ok``.

The payload holds outcomes only (no durations, no timestamps, no sandbox paths); durations and the
reasons for unchecked sections live in the artifact metadata.  Failure policy: a timeout is
``AgentTimeout`` and a crashed or unreachable sandbox is ``TransientAgentError`` (both retried by
the engine); inconsistent or truncated pytest output, pytest internal / usage errors and bad
parameters are FATAL (retrying identical inputs cannot help); failing tests are a normal report
with ``passed: false`` (the ``tests_passed`` gate routes that into the remediation loop).
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator.agents.base import BaseAgent, json_artifact
from orchestrator.agents.catalog import get_capability
from orchestrator.agents.recipes.base import validate_params
from orchestrator.core.errors import AgentError, AgentTimeout, TransientAgentError
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.schemas import Kind
from orchestrator.core.serde import loads_strict
from orchestrator.core.types import (
    MAX_FILE_BYTES,
    AgentContext,
    AgentResult,
    AutonomyProfile,
    ChangeSet,
    CommandResult,
    FileOpKind,
    Toolbox,
)
from orchestrator.tools.openapi import APP_IMPORT_RE, MAX_EXPORT_BYTES, diff_openapi

HELPER_DIR = "_agentic_verify"
OPENAPI_HELPER = f"{HELPER_DIR}/openapi_dump.py"
DEFAULT_APP_IMPORT = "shortener.main:create_app"
DEFAULT_TEST_TARGETS = ("tests",)
DEFAULT_TEST_TIMEOUT_S = 300.0
MAX_TEST_TIMEOUT_S = 600.0
MAX_TARGETS = 8
MAX_AUTHORED_FILES = 40
MAX_BASELINE_ROUNDS = 4
MAX_COVERS_FILES = 200
MAX_PERF_BUDGET_MS = 3_600_000.0

# pytest exit codes: 0 ok, 1 failures, 2 interrupted (collection errors), 3 internal, 4 usage, 5
# empty
_REPORTING_EXITS = frozenset({0, 1, 2, 5})
_DETERMINISTIC_BROKEN_EXITS = frozenset({3, 4})

_SUMMARY_HEADER = re.compile(r"^=+ short test summary info =+$")
_RESULT_LINE = re.compile(
    r"^(?P<outcome>PASSED|FAILED|ERROR|XFAIL|XPASS) "
    r"(?P<nodeid>[^\s:\[]+(?:::[^\s:\[]+)*(?:\[.*?\])?)(?:\s.*)?$"
)
_FINAL_LINE = re.compile(
    r"^(?:=+ )?(?P<body>\d+ \w+(?:, \d+ \w+)*) in [0-9.]+s(?: \([0-9:]+\))?(?: =+)?$"
)
_NO_TESTS_LINE = re.compile(r"^(?:=+ )?no tests ran in [0-9.]+s(?: =+)?$")
_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)")
_COVERS_LINE = re.compile(r"^\s*Covers:\s*(?P<ids>.+?)\s*$", re.MULTILINE)
_COVER_ID = re.compile(r"\b(?:R-\d{3}|AC-\d{3}-\d{1,2})\b")
_TARGET_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,120}$")
_OUTCOME_RANK = {"error": 3, "failed": 2, "passed": 1, "skipped": 0}
_OUTCOME_OF = {
    "PASSED": "passed",
    "FAILED": "failed",
    "ERROR": "error",
    "XFAIL": "skipped",
    "XPASS": "passed",
}
_TAIL_CHARS = 300

_HELPER_TEMPLATE = """import contextlib
import importlib
import json
import sys

TARGET = {target}
module_name, _, attr = TARGET.partition(":")
try:
    with contextlib.redirect_stdout(sys.stderr):
        obj = getattr(importlib.import_module(module_name), attr)
        app = obj if callable(getattr(obj, "openapi", None)) else obj()
        schema = app.openapi()
    # one entry per line: the sandbox redacts (and caps) output line by line
    text = json.dumps(schema, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False)
except (Exception, SystemExit) as exc:
    print("openapi dump failed:", type(exc).__name__, str(exc)[:300], file=sys.stderr)
    raise SystemExit(3)
print(text)
"""
# The shortener refuses to start without API keys; auth stays ENABLED for the schema export: the
# sandboxed helper process gives it a throwaway key generated at run time (never a literal).
_SHORTENER_ENV = """import os
import secrets

os.environ.setdefault("SHORTENER_API_KEYS", "schema-dump:" + secrets.token_urlsafe(32))
"""
_SHORTENER_PREFIX = "shortener."


def helper_source(app_import: str) -> str:
    """The staged OpenAPI helper script for a validated ``module:factory``."""
    env = _SHORTENER_ENV if app_import.startswith(_SHORTENER_PREFIX) else ""
    return env + _HELPER_TEMPLATE.format(target=json.dumps(app_import))


# ----------------------------------------------------------------------------- pytest output


@dataclass(frozen=True)
class ParsedRun:
    """What the last ``short test summary info`` section and pytest's final line say."""

    entries: tuple[
        tuple[str, str], ...
    ]  # (outcome, nodeid); nodeid may be a bare file (collection)
    counts: Mapping[str, int]  # passed / failed / error / skipped (xfail = skipped, xpass = passed)
    final_found: bool


def parse_pytest_output(stdout: str) -> ParsedRun:
    """Parse ``pytest -q -rA`` output (see the module docstring for what is trusted)."""
    lines = [line.rstrip() for line in stdout.splitlines()]
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    final_index = None
    final_found = False
    for index in range(len(lines) - 1, max(-1, len(lines) - 8), -1):
        text = lines[index].strip()
        if not text:
            continue
        found = _FINAL_LINE.match(text)
        if found:
            for number, word in _COUNT.findall(found.group("body")):
                key = {"errors": "error", "xfailed": "skipped", "xpassed": "passed"}.get(word, word)
                counts[key] += int(number)
            final_index, final_found = index, True
            break
        if _NO_TESTS_LINE.match(text):
            final_index, final_found = index, True
            break
    headers = [i for i, line in enumerate(lines) if _SUMMARY_HEADER.match(line.strip())]
    entries: list[tuple[str, str]] = []
    if headers:
        stop = final_index if final_index is not None else len(lines)
        for line in lines[headers[-1] + 1 : stop]:
            match = _RESULT_LINE.match(line)
            if match:
                entries.append((_OUTCOME_OF[match.group("outcome")], match.group("nodeid")))
    return ParsedRun(tuple(entries), counts, final_found)


def _merge_outcomes(entries: Sequence[tuple[str, str]]) -> dict[str, str]:
    """One outcome per node id (a test that passed but errored in teardown counts as an error)."""
    merged: dict[str, str] = {}
    for outcome, nodeid in entries:
        previous = merged.get(nodeid)
        if previous is None or _OUTCOME_RANK[outcome] > _OUTCOME_RANK[previous]:
            merged[nodeid] = outcome
    return merged


def _bare_nodeid(nodeid: str) -> str:
    """``tests/t.py::TestX::test_a[p]`` -> ``tests/t.py::TestX::test_a``."""
    bracket = nodeid.find("[")
    return nodeid if bracket < 0 else nodeid[:bracket]


def _file_of(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


# ----------------------------------------------------------------------------- covers (ast)


def _docstring_ids(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> set[str]:
    doc = ast.get_docstring(node, clean=True)
    ids: set[str] = set()
    for match in _COVERS_LINE.finditer(doc or ""):
        ids.update(_COVER_ID.findall(match.group("ids")))
    return ids


def covers_by_test(source: str) -> dict[str, frozenset[str]]:
    """``{"test_a": ids, "TestX::test_b": ids}`` from ``Covers:`` docstring lines.

    A class docstring applies to its methods unless a method has its own ``Covers:`` line.  Hostile
    or broken source (syntax error, recursion, memory) yields an empty mapping, never an exception.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {}
    found: dict[str, frozenset[str]] = {}
    functions = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, functions):
            found[node.name] = frozenset(_docstring_ids(node))
        elif isinstance(node, ast.ClassDef):
            inherited = _docstring_ids(node)
            for member in node.body:
                if isinstance(member, functions):
                    own = _docstring_ids(member)
                    found[f"{node.name}::{member.name}"] = frozenset(own or inherited)
    return found


class _CoversIndex:
    """Reads each test file once through the toolbox and answers ``covers`` queries."""

    def __init__(self, box: Toolbox) -> None:
        self._box = box
        self._by_file: dict[str, dict[str, frozenset[str]]] = {}

    def _file(self, path: str) -> dict[str, frozenset[str]]:
        if path not in self._by_file:
            if len(self._by_file) >= MAX_COVERS_FILES:
                self._by_file[path] = {}
            else:
                try:
                    self._by_file[path] = covers_by_test(self._box.read_text(path))
                except (FileNotFoundError, AgentError):
                    self._by_file[path] = {}
        return self._by_file[path]

    def covers(self, nodeid: str) -> list[str]:
        path = _file_of(nodeid)
        table = self._file(path)
        if "::" not in nodeid:  # a file-level (collection) error: everything the file claims
            return sorted(set().union(*table.values())) if table else []
        name = "::".join(_bare_nodeid(nodeid).split("::")[1:])
        return sorted(table.get(name, frozenset()))


# ----------------------------------------------------------------------------- helpers


def _fatal(message: str, **details: Any) -> AgentError:
    return AgentError(message, kind="fatal", retryable=False, details=dict(details))


def _tail(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return sanitize_for_terminal(lines[-1], _TAIL_CHARS) if lines else "no diagnostic output"


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path.startswith("tests/")
        and name.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


def _is_perf_test(nodeid: str) -> bool:
    return "perf" in _bare_nodeid(nodeid).rsplit("::", 1)[-1].lower()


def _timeout(params: Mapping[str, Any]) -> float:
    raw = params.get("test_timeout_s", DEFAULT_TEST_TIMEOUT_S)
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not 0 < raw <= MAX_TEST_TIMEOUT_S
    ):
        raise _fatal(f"test_timeout_s must be a number in (0, {MAX_TEST_TIMEOUT_S:g}]")
    return float(raw)


def _targets(params: Mapping[str, Any], box: Toolbox) -> list[str]:
    raw = params.get("test_targets", list(DEFAULT_TEST_TARGETS))
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) > MAX_TARGETS:
        raise _fatal(f"test_targets must be a list of at most {MAX_TARGETS} directories or files")
    targets: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not _TARGET_RE.match(item) or ".." in item.split("/"):
            raise _fatal("test_targets holds an invalid entry")
        if box.exists(item) or box.list_files(f"{item.rstrip('/')}/**"):
            targets.append(item)
    return sorted(set(targets))


def _perf_budget(params: Mapping[str, Any], requirements: Mapping[str, Any]) -> float | None:
    raw: Any = params.get("perf_budget_ms")
    if raw is None:
        for cap in sorted(requirements.get("capabilities") or [], key=lambda c: str(c.get("id"))):
            raw = (cap.get("params") or {}).get("perf_budget_ms")
            if raw is not None:
                break
    if raw is None:
        return None
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not 0 < raw <= MAX_PERF_BUDGET_MS
    ):
        raise _fatal(f"perf_budget_ms must be a number in (0, {MAX_PERF_BUDGET_MS:g}]")
    return float(raw)


class VerifierAgent(BaseAgent):
    agent_id = "verifier"
    role = "verifier"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=(
            f"{HELPER_DIR}/**",
        ),  # only the staged OpenAPI helper script, never the product
        can_execute=True,
        max_tool_calls=1000,
        max_runtime_s=1800.0,
        max_changed_files=4,
        max_file_bytes=MAX_FILE_BYTES,
    )

    # ------------------------------------------------------------------ entry point
    def run(self, ctx: AgentContext) -> AgentResult:
        box = ctx.toolbox
        params = ctx.params
        integrated = ChangeSet.from_dict(ctx.json(Kind.INTEGRATED_CHANGESET))
        task_plan = ctx.json(Kind.TASK_PLAN)
        requirements = ctx.json(Kind.REQUIREMENTS)
        timeout = _timeout(params)
        budget = _perf_budget(params, requirements)

        durations: list[float] = []
        targets = _targets(params, box)
        main = self._run_pytest(box, targets, timeout, durations) if targets else _empty_run()
        covers = _CoversIndex(box)
        merged = _merge_outcomes(main.entries)
        tests = [
            {"nodeid": nodeid, "outcome": outcome, "covers": covers.covers(nodeid)}
            for nodeid, outcome in sorted(merged.items())
        ]
        counts = main.counts
        total = sum(counts.values())
        suite_passed = (
            bool(targets)
            and main.returncode == 0
            and total > 0
            and counts["failed"] == 0
            and counts["error"] == 0
        )

        authored = sorted(
            op.path
            for op in integrated.ops
            if op.kind == FileOpKind.CREATE and _is_test_file(op.path)
        )
        f2p, baseline = self._fail_to_pass(box, authored, merged, timeout, durations)
        api, api_reason = self._api_compat(ctx, task_plan, timeout, durations)
        perf = _perf_section(tests, budget)

        payload = {
            "passed": suite_passed,
            "total": total,
            "failed": counts["failed"],
            "errors": counts["error"],
            "skipped": counts["skipped"],
            "tests": tests,
            "baseline": baseline,
            "fail_to_pass": f2p,
            "perf": perf,
            "api_compat": api,
        }
        metadata: dict[str, Any] = {
            "pytest_runs": len(durations),
            "duration_s": round(sum(durations), 3),
            "api_compat_reason": api_reason,
            "targets": targets,
        }
        rationale = (
            f"{counts['passed']} passed, {counts['failed']} failed, {counts['error']} errors, "
            f"{counts['skipped']} skipped; fail-to-pass "
            f"{'ok' if f2p['ok'] else 'NOT ok' if f2p['checked'] else 'not checked'}."
        )
        return AgentResult(
            artifacts=[json_artifact(Kind.TEST_REPORT, payload, metadata=metadata)],
            rationale=rationale,
            usage=dict(box.usage()),
        )

    # ------------------------------------------------------------------ pytest
    def _run_pytest(
        self,
        box: Toolbox,
        targets: Sequence[str],
        timeout: float,
        durations: list[float],
        *,
        mode: str = "view",
        overlay: Sequence[str] = (),
    ) -> _Run:
        try:
            result = box.run_tests(
                list(targets), timeout_s=timeout, mode=mode, overlay_paths=overlay
            )
        except OSError as exc:  # scratch creation / cleanup trouble: environmental, retry
            raise TransientAgentError(
                f"sandbox I/O failure: {sanitize_for_terminal(exc, 160)}"
            ) from exc
        durations.append(result.duration_s)
        return _classify(result)

    # ------------------------------------------------------------------ fail-to-pass
    def _fail_to_pass(
        self,
        box: Toolbox,
        authored: Sequence[str],
        integrated: Mapping[str, str],
        timeout: float,
        durations: list[float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unchecked_f2p = {"checked": False, "ok": False, "tests": [], "vacuous": []}
        unchecked_base: dict[str, Any] = {"ran": False, "passed": None, "failed_tests": []}
        if not authored:
            return unchecked_f2p, unchecked_base
        if len(authored) > MAX_AUTHORED_FILES:
            raise _fatal(f"more than {MAX_AUTHORED_FILES} authored test files cannot be verified")
        on_baseline: dict[str, str] = {}
        broken_files: set[str] = set()
        pending = list(authored)
        all_green = True
        ran = False
        for _ in range(MAX_BASELINE_ROUNDS):
            if not pending:
                break
            run = self._run_pytest(
                box, pending, timeout, durations, mode="baseline_plus", overlay=pending
            )
            ran = True
            all_green = all_green and run.returncode == 0
            reached: set[str] = set()
            for outcome, nodeid in run.entries:
                path = _file_of(nodeid)
                if "::" in nodeid:
                    if path in pending:
                        reached.add(path)
                        prior = on_baseline.get(nodeid)
                        if prior is None or _OUTCOME_RANK[outcome] > _OUTCOME_RANK[prior]:
                            on_baseline[nodeid] = outcome
                elif outcome in ("error", "failed"):
                    # a collection error of a file, or of a package directory holding pending files
                    prefix = path.rstrip("/") + "/"
                    broken = [p for p in pending if p == path or p.startswith(prefix)]
                    reached.update(broken)
                    broken_files.update(broken)
            if not reached:
                break  # nothing new learned: stop instead of looping
            pending = [p for p in pending if p not in reached]

        authored_set = set(authored)
        ids = sorted(n for n in integrated if "::" in n and _file_of(n) in authored_set)

        def failed_before(nodeid: str) -> bool:
            return (
                on_baseline.get(nodeid) in ("failed", "error") or _file_of(nodeid) in broken_files
            )

        good = [n for n in ids if integrated[n] == "passed" and failed_before(n)]
        vacuous = [n for n in ids if on_baseline.get(n) == "passed"]
        failing = [n for n in ids if failed_before(n)]
        covered_files = {_file_of(n) for n in ids}
        failing += sorted(f for f in broken_files if f not in covered_files)
        ok = bool(ids) and not pending and set(good) == set(ids)
        f2p = {"checked": ran, "ok": ok, "tests": good, "vacuous": vacuous}
        baseline = {"ran": ran, "passed": all_green if ran else None, "failed_tests": failing}
        return f2p, baseline

    # ------------------------------------------------------------------ api compatibility
    def _api_compat(
        self,
        ctx: AgentContext,
        task_plan: Mapping[str, Any],
        timeout: float,
        durations: list[float],
    ) -> tuple[dict[str, Any], str]:
        unchecked: dict[str, Any] = {"checked": False, "breaking": [], "additive": []}
        box = ctx.toolbox
        changes, reason = _declares_api_changes(task_plan)
        if not changes:
            return unchecked, reason
        app_import = ctx.params.get("app_import", DEFAULT_APP_IMPORT)
        if not isinstance(app_import, str) or not APP_IMPORT_RE.match(app_import):
            raise _fatal("app_import must look like 'package.module:factory'")
        module = app_import.split(":", 1)[0].replace(".", "/")
        if not (box.exists(f"{module}.py") or box.exists(f"{module}/__init__.py")):
            return unchecked, "the product is not part of the tree"
        box.write_text(OPENAPI_HELPER, helper_source(app_import))
        try:
            new = self._export(box, timeout, durations, mode="view")
            old = self._export(box, timeout, durations, mode="baseline_plus")
        except _ExportUnavailable as exc:
            return unchecked, str(exc)
        diff = diff_openapi(old, new)
        return {"checked": True, "breaking": diff.breaking, "additive": diff.additive}, "exported"

    def _export(
        self, box: Toolbox, timeout: float, durations: list[float], *, mode: str
    ) -> dict[str, Any]:
        overlay = [OPENAPI_HELPER] if mode == "baseline_plus" else []
        try:
            result = box.run_python(
                [OPENAPI_HELPER], timeout_s=timeout, mode=mode, overlay_paths=overlay
            )
        except OSError as exc:
            raise _ExportUnavailable(f"sandbox I/O failure during the {mode} export") from exc
        durations.append(result.duration_s)
        label = "integrated" if mode == "view" else "baseline"
        if result.timed_out:
            raise _ExportUnavailable(f"the {label} OpenAPI export timed out")
        if result.returncode != 0:
            raise _ExportUnavailable(
                f"the {label} OpenAPI export failed (exit {result.returncode}): "
                f"{_tail(result.stderr)}"
            )
        if result.truncated:
            raise _ExportUnavailable(f"the {label} OpenAPI export was truncated")
        try:
            spec = loads_strict(result.stdout.strip(), max_bytes=MAX_EXPORT_BYTES)
        except ValueError as exc:
            raise _ExportUnavailable(f"the {label} OpenAPI export is not JSON") from exc
        if not (
            isinstance(spec, dict)
            and isinstance(spec.get("openapi"), str)
            and isinstance(spec.get("paths"), dict)
        ):
            raise _ExportUnavailable(f"the {label} export is not an OpenAPI document")
        return spec


class _ExportUnavailable(Exception):
    """The OpenAPI export could not be produced (recorded as ``checked: false``, never fatal)."""


@dataclass(frozen=True)
class _Run:
    returncode: int
    entries: tuple[tuple[str, str], ...]
    counts: Mapping[str, int]


def _empty_run() -> _Run:
    return _Run(5, (), {"passed": 0, "failed": 0, "error": 0, "skipped": 0})


def _classify(result: CommandResult) -> _Run:
    """Turn a sandbox result into a trustworthy ``_Run`` or the right failure."""
    if result.timed_out:
        raise AgentTimeout("the test run timed out in the sandbox")
    code = result.returncode
    if code in _DETERMINISTIC_BROKEN_EXITS:
        raise _fatal(f"pytest failed to run (exit {code}): {_tail(result.stderr or result.stdout)}")
    if code not in _REPORTING_EXITS:
        raise TransientAgentError(f"the test process crashed (exit code {code})")
    if result.truncated:
        raise _fatal("the pytest output was truncated; the report cannot be trusted")
    parsed = parse_pytest_output(result.stdout)
    if code != 5 and not parsed.final_found:
        raise _fatal("the pytest summary line is missing; the report cannot be trusted")
    seen = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for outcome, _ in parsed.entries:
        seen[outcome] += 1
    counts = parsed.counts
    if (
        any(seen[k] != counts[k] for k in ("passed", "failed", "error"))
        or seen["skipped"] > counts["skipped"]
        or (code == 0 and (counts["failed"] or counts["error"]))
        or (code == 1 and not (counts["failed"] or counts["error"]))
    ):
        raise _fatal(
            "the pytest entries disagree with its summary line (possible output tampering): "
            f"entries {dict(seen)} against summary {dict(counts)}"
        )
    return _Run(code, parsed.entries, counts)


def _declares_api_changes(task_plan: Mapping[str, Any]) -> tuple[bool, str]:
    """Does any capability of the plan declare API changes in its recipe design?"""
    seen: dict[str, Mapping[str, Any]] = {}
    for group in task_plan.get("groups") or []:
        seen.setdefault(str(group.get("capability")), group.get("params") or {})
    if not seen:
        return False, "the plan has no capability groups"
    for capability_id, raw in sorted(seen.items()):
        try:
            capability = get_capability(capability_id)
            design = capability.recipe.design(validate_params(capability.recipe.params, raw), None)
        except (KeyError, AgentError) as exc:
            return (
                False,
                f"cannot read the API design of {sanitize_for_terminal(capability_id, 40)}: "
                f"{type(exc).__name__}",
            )
        if design.get("api_changes") or design.get("api_paths"):
            return True, "declared"
    return False, "no capability declares API changes"


def _perf_section(tests: Sequence[Mapping[str, Any]], budget: float | None) -> dict[str, Any]:
    perf = [t for t in tests if _is_perf_test(str(t["nodeid"]))]
    checked = budget is not None and bool(perf)
    return {
        "checked": checked,
        "budget_ms": budget,
        "ok": all(t["outcome"] == "passed" for t in perf) if checked else None,
    }
