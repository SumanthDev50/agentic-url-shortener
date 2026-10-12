"""Release verifier agent (``release-verify`` node; can execute).

Verifies what is about to be (or has just been) released, independently of the promoter:

1. **The patch applies cleanly to a fresh baseline copy.**  ``release_patch`` is parsed with a
   small,
   strict parser for exactly the git-style dialect ``tools.changeset.to_unified_diff`` emits
   (anything
   else is rejected, never guessed) and applied in memory to the BASELINE read through
   ``ScopedToolbox.baseline_view()``; every result must equal the reviewed tree (the toolbox view =
   baseline + integrated + docs).  The patch must not touch a path outside the reviewed change sets,
   must list the same files as ``release_bundle.files`` and must hash to ``patch_sha256``.
   Without a
   baseline view the check cannot be made and fails closed.
2. **The suite is green** on the reviewed tree (``run_tests`` over ``tests``; no tests = not green).
3. **/healthz smoke** when the product (``shortener/main.py`` + ``shortener/config.py``) is part
   of the
   tree: a static smoke test is staged under ``_agentic_release/`` and run in the sandbox (the app
   is
   built with auth disabled on loopback and its database inside the scratch directory).  ``None``
   when
   the product is absent.

``target`` is ``bundle-scratch`` for a bundle and ``applied-target`` for an applied release; in the
latter case the effect receipt's ``post_hashes`` must additionally equal the hashes of the verified
tree.  Honest limit: this agent has no handle on the live target tree (only the effect port of the
promote node has), so the verification always runs on a scratch reconstruction and compares the
receipt with it.  Timeouts and crashed sandboxes are retried (``AgentTimeout`` /
``TransientAgentError``); everything else is a recorded fact (``false``), which fails the
``release_verified`` gate and, in apply mode, triggers the saga rollback of ``promote``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator.agents.base import BaseAgent, json_artifact
from orchestrator.core.errors import AgentTimeout, PathViolation, TransientAgentError
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.paths import validate_rel_path
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    AutonomyProfile,
    ChangeSet,
    CommandResult,
    FileView,
    Toolbox,
)
from orchestrator.tools.toolbox import baseline_view_of

HELPER_DIR = "_agentic_release"
SMOKE_TEST = f"{HELPER_DIR}/test_healthz_smoke.py"
PRODUCT_FILES = ("shortener/main.py", "shortener/config.py")
DEFAULT_TIMEOUT_S = 300.0
MAX_PATCH_FILES = 600
MAX_PATCH_LINE_CHARS = 200_000
MAX_PROBLEMS = 20

# Static text, no interpolation: the smoke test only touches the scratch directory.
SMOKE_SOURCE = '''\
"""Release smoke test: the service starts with safe defaults and answers /healthz."""


def test_healthz_answers() -> None:
    from fastapi.testclient import TestClient

    from shortener.config import Settings
    from shortener.main import create_app

    settings = Settings(auth_disabled=True, database_path="healthz-smoke.db")
    with TestClient(create_app(settings)) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"
'''

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<path>.+) b/(?P=path)$")
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@$")
_NO_NEWLINE = "\\ No newline at end of file"
_FINAL_LINE = re.compile(
    r"^(?:=+ )?(?P<body>\d+ \w+(?:, \d+ \w+)*) in [0-9.]+s(?: \([0-9:]+\))?(?: =+)?$"
)
_COUNT = re.compile(r"(\d+) (passed|failed|errors?|xfailed|xpassed)")
_CRASH_FREE_EXITS = frozenset({0, 1, 2, 3, 4, 5})


class PatchError(ValueError):
    """The patch is not in the dialect ``to_unified_diff`` emits, or does not apply."""


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    old_len: int
    lines: tuple[
        tuple[str, str], ...
    ]  # (' ' | '+' | '-', text including its newline if it has one)


@dataclass(frozen=True)
class _FilePatch:
    path: str
    kind: str  # create | modify | delete
    hunks: tuple[_Hunk, ...]


def _parse_hunk(lines: Sequence[str], start: int) -> tuple[_Hunk, int]:
    header = _HUNK_HEADER.match(lines[start])
    if header is None:
        raise PatchError("malformed hunk header")
    old_start = int(header.group(1))
    old_len = 1 if header.group(2) is None else int(header.group(2))
    new_len = 1 if header.group(4) is None else int(header.group(4))
    body: list[tuple[str, str]] = []
    old_seen = new_seen = 0
    index = start + 1
    while old_seen < old_len or new_seen < new_len:
        if index >= len(lines) or not lines[index][:1] or lines[index][0] not in " +-":
            raise PatchError("hunk is shorter than its header says")
        marker, text = lines[index][0], lines[index][1:]
        old_seen += marker in " -"
        new_seen += marker in " +"
        body.append((marker, text + "\n"))
        index += 1
        if index < len(lines) and lines[index] == _NO_NEWLINE:
            body[-1] = (marker, text)  # that line has no trailing newline
            index += 1
    if old_seen != old_len or new_seen != new_len:
        raise PatchError("hunk line counts disagree with its header")
    return _Hunk(old_start, old_len, tuple(body)), index


def parse_patch(patch: str) -> list[_FilePatch]:
    """Parse the dialect of ``to_unified_diff`` strictly (``PatchError`` on anything else)."""
    if "\r" in patch:
        raise PatchError("patch must use LF line endings")
    lines = patch.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if any(len(line) > MAX_PATCH_LINE_CHARS for line in lines):
        raise PatchError("patch line too long")
    patches: list[_FilePatch] = []
    index = 0
    while index < len(lines):
        header = _DIFF_HEADER.match(lines[index]) if len(lines[index]) < 600 else None
        if header is None:
            raise PatchError("expected a 'diff --git' header")
        try:
            path = validate_rel_path(header.group("path"))
        except PathViolation as exc:
            raise PatchError("patch names an unsafe path") from exc
        if len(patches) >= MAX_PATCH_FILES:
            raise PatchError("patch touches too many files")
        index += 1
        kind = "modify"
        if index < len(lines) and lines[index] == "new file mode 100644":
            kind, index = "create", index + 1
        elif index < len(lines) and lines[index] == "deleted file mode 100644":
            kind, index = "delete", index + 1
        hunks: list[_Hunk] = []
        if index < len(lines) and lines[index].startswith("--- "):
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                raise PatchError("missing '+++' header")
            want_old = "/dev/null" if kind == "create" else f"a/{path}"
            want_new = "/dev/null" if kind == "delete" else f"b/{path}"
            if lines[index][4:] != want_old or lines[index + 1][4:] != want_new:
                raise PatchError("file headers disagree with the diff header")
            index += 2
            while index < len(lines) and lines[index].startswith("@@ "):
                hunk, index = _parse_hunk(lines, index)
                hunks.append(hunk)
        patches.append(_FilePatch(path, kind, tuple(hunks)))
    return patches


def apply_file_patch(patch: _FilePatch, old: str | None) -> str | None:
    """Result of applying ``patch`` to ``old`` (``None`` = file absent); ``None`` for a delete."""
    if patch.kind == "create":
        if old is not None:
            raise PatchError("a created file already exists in the baseline")
    elif old is None:
        raise PatchError("a modified or deleted file is missing from the baseline")
    old_lines = (old or "").splitlines(keepends=True)
    out: list[str] = []
    pos = 0
    for hunk in patch.hunks:
        start = hunk.old_start - 1 if hunk.old_len > 0 else hunk.old_start
        if start < pos or start > len(old_lines):
            raise PatchError("hunk is out of range or overlaps the previous one")
        out.extend(old_lines[pos:start])
        pos = start
        for marker, text in hunk.lines:
            if marker == "+":
                out.append(text)
                continue
            if pos >= len(old_lines) or old_lines[pos] != text:
                raise PatchError("context or removed lines do not match the baseline")
            if marker == " ":
                out.append(text)
            pos += 1
    out.extend(old_lines[pos:])
    result = "".join(out)
    if patch.kind == "delete":
        if result:
            raise PatchError("a delete patch leaves content behind")
        return None
    return result


# ----------------------------------------------------------------------------- the checks


def _problem(problems: list[str], message: str) -> None:
    if len(problems) < MAX_PROBLEMS:
        problems.append(sanitize_for_terminal(message, 240))


def _check_patch(
    box: Toolbox,
    patch_text: str,
    bundle: Mapping[str, Any],
    reviewed_paths: set[str],
    problems: list[str],
) -> tuple[bool, int]:
    """(applies cleanly and reproduces the reviewed tree, number of files checked)."""
    if sha256_text(patch_text) != bundle["patch_sha256"]:
        _problem(problems, "release_patch does not hash to the bundle's patch_sha256")
        return False, 0
    try:
        patches = parse_patch(patch_text)
    except PatchError as exc:
        _problem(problems, f"release_patch cannot be parsed: {exc}")
        return False, 0
    paths = {p.path for p in patches}
    clean = True
    if paths != set(bundle["files"]):
        _problem(problems, "release_patch and release_bundle list different files")
        clean = False
    if not paths <= reviewed_paths:
        extra = sorted(paths - reviewed_paths)[:3]
        _problem(problems, f"release_patch touches paths outside the reviewed change sets: {extra}")
        clean = False
    baseline = baseline_view_of(box)
    if baseline is None:
        _problem(problems, "the toolbox has no baseline view: cannot check the patch against it")
        return False, len(patches)
    for patch in patches:
        if not _reproduces(box, baseline, patch, problems):
            clean = False
    return clean, len(patches)


def _reproduces(box: Toolbox, baseline: FileView, patch: _FilePatch, problems: list[str]) -> bool:
    try:
        old = baseline.read_text(patch.path) if baseline.exists(patch.path) else None
        result = apply_file_patch(patch, old)
    except PatchError as exc:
        _problem(problems, f"{patch.path}: {exc}")
        return False
    if result is None:
        if box.exists(patch.path):
            _problem(
                problems, f"{patch.path}: deleted by the patch but present in the reviewed tree"
            )
            return False
        return True
    if not box.exists(patch.path) or box.read_text(patch.path) != result:
        _problem(problems, f"{patch.path}: the patched baseline differs from the reviewed tree")
        return False
    return True


def _check_receipt(box: Toolbox, bundle: Mapping[str, Any], problems: list[str]) -> bool:
    receipt = bundle.get("receipt")
    if not isinstance(receipt, Mapping):
        _problem(problems, "an applied bundle carries no receipt")
        return False
    post = receipt.get("post_hashes")
    if not isinstance(post, Mapping) or set(post) != set(bundle["files"]):
        _problem(problems, "the receipt does not cover exactly the bundle's files")
        return False
    clean = True
    for path in sorted(post):
        expected = sha256_text(box.read_text(path)) if box.exists(path) else None
        if post[path] != expected:
            _problem(problems, f"{path}: the applied file differs from the reviewed tree")
            clean = False
    return clean


def _green(result: CommandResult) -> bool:
    """Exit code 0 and pytest's own final line reporting at least one pass and no failure."""
    if result.returncode != 0 or result.timed_out:
        return False
    counts = {"passed": 0, "failed": 0, "error": 0}
    for raw in reversed(result.stdout.splitlines()[-8:]):
        found = _FINAL_LINE.match(raw.strip())
        if found:
            for number, word in _COUNT.findall(found.group("body")):
                key = {"errors": "error", "xfailed": "passed", "xpassed": "passed"}.get(word, word)
                counts[key] += int(number)
            break
    return counts["passed"] > 0 and counts["failed"] == 0 and counts["error"] == 0


def _run(box: Toolbox, targets: Sequence[str], timeout: float, durations: list[float]) -> bool:
    try:
        result = box.run_tests(list(targets), timeout_s=timeout)
    except OSError as exc:
        raise TransientAgentError(
            f"sandbox I/O failure: {sanitize_for_terminal(exc, 160)}"
        ) from exc
    durations.append(result.duration_s)
    if result.timed_out:
        raise AgentTimeout("the release suite timed out in the sandbox")
    if result.returncode not in _CRASH_FREE_EXITS:
        raise TransientAgentError(f"the test process crashed (exit code {result.returncode})")
    return _green(result)


class ReleaseVerifierAgent(BaseAgent):
    agent_id = "release-verifier"
    role = "release-verifier"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=(f"{HELPER_DIR}/**",),  # only the static smoke test
        can_execute=True,
        max_tool_calls=2000,
        max_runtime_s=1800.0,
        max_changed_files=2,
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        box = ctx.toolbox
        bundle = ctx.json(Kind.RELEASE_BUNDLE)
        patch_text = ctx.text(Kind.RELEASE_PATCH)
        integrated = ChangeSet.from_dict(ctx.json(Kind.INTEGRATED_CHANGESET))
        docs = ChangeSet.from_dict(ctx.json(Kind.DOCS_CHANGESET))
        timeout = _timeout(ctx.params)
        problems: list[str] = []
        durations: list[float] = []

        clean, checked = _check_patch(
            box, patch_text, bundle, {*integrated.paths(), *docs.paths()}, problems
        )
        applied = bundle["mode"] == "applied"
        if applied:
            clean = _check_receipt(box, bundle, problems) and clean
        has_tests = bool(box.list_files("tests/**"))
        tests_passed = _run(box, ["tests"], timeout, durations) if has_tests else False
        if not has_tests:
            _problem(problems, "the reviewed tree has no tests")
        healthz: bool | None = None
        if all(box.exists(path) for path in PRODUCT_FILES):
            box.write_text(SMOKE_TEST, SMOKE_SOURCE)
            healthz = _run(box, [SMOKE_TEST], timeout, durations)

        payload = {
            "target": "applied-target" if applied else "bundle-scratch",
            "applied_cleanly": clean,
            "tests_passed": tests_passed,
            "healthz_ok": healthz,
            "checked_files": checked,
        }
        return AgentResult(
            artifacts=[
                json_artifact(
                    Kind.RELEASE_VERIFICATION,
                    payload,
                    metadata={"problems": problems, "duration_s": round(sum(durations), 3)},
                )
            ],
            rationale=(
                f"patch {'applies cleanly' if clean else 'DOES NOT verify'}; "
                f"tests {'green' if tests_passed else 'NOT green'}; healthz {healthz}."
            ),
            usage=dict(box.usage()),
        )


def _timeout(params: Mapping[str, Any]) -> float:
    raw = params.get("test_timeout_s", DEFAULT_TIMEOUT_S)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 0 < raw <= 600:
        return DEFAULT_TIMEOUT_S
    return float(raw)
