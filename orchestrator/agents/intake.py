"""Intake agent: classifies the workspace and the kind of change (deterministic, read-only).

What it produces (``intake_report``):

* ``workspace`` - number of files, number of Python files, ``is_empty`` and ``manifest_hash``.
  The manifest hash is ``hash_jsonable({path: sha256_text(content)})`` over everything the toolbox
  can see, i.e. the same definition as ``BaselineSnapshot.manifest_hash``.
* ``classification.workspace_kind`` - ``greenfield`` iff the workspace has no files, else
  ``brownfield``.
* ``classification.change_type`` - keyword rules over the request text (:func:`classify_change`).
* ``request`` - character and line counts of the request (never its content).
* ``input_findings`` - always ``[]``: policy screening of human input happens in the engine at
  ingestion (engine-semantics section 16), not here.

Change-type rules, first match wins (heuristic, documented limitation: keywords, no understanding):

1. ``bugfix``           fix / fixes / fixed / fixing, bug, bugfix, regression
2. ``refactor``         refactor*, clean up / cleanup
3. ``docs``             documentation words and no behaviour words (implement, feature, ...)
4. ``test_improvement`` test words and no behaviour words
5. ``feature`` (greenfield) / ``enhancement`` (brownfield) otherwise; ``unknown`` for a blank
   request.

The request is hostile input: only a bounded prefix is scanned, every pattern is linear-time, and
nothing in it is ever interpreted as an instruction.
"""

from __future__ import annotations

import re

from orchestrator.agents.base import MAX_TEXT_CHARS, BaseAgent, json_artifact
from orchestrator.core.errors import AgentError
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.sanitize import strip_controls
from orchestrator.core.schemas import Kind
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import K_REQUEST, AgentContext, AgentResult, AutonomyProfile

MAX_WORKSPACE_FILES = 5000  # == tools.workspace.MAX_FILES: one read per file must fit the budget

_FLAGS = re.IGNORECASE
_BUGFIX = re.compile(
    r"\b(?:fix(?:es|ed|ing)?(?![- ](?:window|size|length|width|rate))|bugs?|bug-?fix(?:es)?|"
    r"regressions?)\b",
    _FLAGS,
)
_REFACTOR = re.compile(r"\b(?:refactor\w*|clean(?:[- ]?up))\b", _FLAGS)
_DOCS = re.compile(r"\b(?:docs?|documentation|readme|docstrings?|changelog)\b", _FLAGS)
_TESTS = re.compile(r"\b(?:tests?|test suite|unit[- ]tests?|coverage)\b", _FLAGS)
_BEHAVIOUR = re.compile(
    r"\b(?:implement\w*|features?|endpoints?|migrations?|introduce|allow|support|export|enable)\b",
    _FLAGS,
)
_LETTER = re.compile(r"[^\W\d_]")


def classify_change(request: str, workspace_kind: str) -> str:
    """Change type of ``request`` (see the module docstring for the rules)."""
    text = request[:MAX_TEXT_CHARS]
    if _LETTER.search(text) is None:
        return "unknown"
    if _BUGFIX.search(text):
        return "bugfix"
    if _REFACTOR.search(text):
        return "refactor"
    behaviour = _BEHAVIOUR.search(text) is not None
    if _DOCS.search(text) and not behaviour:
        return "docs"
    if _TESTS.search(text) and not behaviour:
        return "test_improvement"
    return "feature" if workspace_kind == "greenfield" else "enhancement"


class IntakeAgent(BaseAgent):
    agent_id = "intake"
    role = "intake"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=(),
        can_execute=False,
        max_tool_calls=MAX_WORKSPACE_FILES + 16,
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        request = strip_controls(ctx.text(K_REQUEST), keep_newlines=True)
        box = ctx.toolbox
        manifest: dict[str, str] = {}
        for path in box.list_files("**"):
            try:
                manifest[path] = sha256_text(box.read_text(path))
            except FileNotFoundError as exc:
                raise AgentError(
                    f"workspace changed while it was being read: {path}",
                    kind="fatal",
                    retryable=False,
                ) from exc
        workspace_kind = "brownfield" if manifest else "greenfield"
        payload = {
            "workspace": {
                "file_count": len(manifest),
                "python_files": sum(1 for p in manifest if p.endswith(".py")),
                "is_empty": not manifest,
                "manifest_hash": hash_jsonable(manifest),
            },
            "classification": {
                "workspace_kind": workspace_kind,
                "change_type": classify_change(request, workspace_kind),
            },
            "request": {
                "chars": len(request),
                "lines": request.count("\n") + 1 if request else 0,
            },
            "input_findings": [],
        }
        return AgentResult(
            artifacts=[json_artifact(Kind.INTAKE_REPORT, payload)],
            rationale=f"{workspace_kind} workspace with {len(manifest)} file(s).",
            usage=dict(box.usage()),
        )
