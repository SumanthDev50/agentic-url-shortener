"""Documenter agent (``docs`` node): stages documentation as a ``docs_changeset``.

For every capability of the task plan (sorted by id) it

1. runs ``recipe.build_docs(BuildContext(...))`` (the recipe stages its own pages through the
   toolbox);
2. adds a *design summary* block to ``docs/features/<capability>.md`` built from structured facts of
   ``design_doc`` and ``requirements`` only (ids, enums, counts and escaped text).  If the page
   already
   exists (baseline, or staged by the recipe) the block is appended, or replaced between its
   markers,
   so the recipe's own prose is never lost and a second run is idempotent;

and finally appends one entry to ``CHANGELOG.md`` in append-only style: existing text is preserved
byte for byte, the entry is added after one blank line, and an entry that is already present is not
repeated.  Entries carry no dates (determinism rule).

Security: all text that comes from requirements, designs or capabilities passes ``escape_markdown``
(no raw HTML, no active links, no markdown injection); capability ids are validated against a strict
grammar before they become a path component; the agent can only write ``docs/**``,
``CHANGELOG.md`` and
``README.md`` (the toolbox enforces the globs, SEC-005 re-checks every op of the change set).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from orchestrator.agents.base import BaseAgent, changeset_artifact
from orchestrator.agents.catalog import get_capability
from orchestrator.agents.recipes.base import BuildContext, validate_params
from orchestrator.core.errors import AgentError
from orchestrator.core.sanitize import escape_markdown
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    AutonomyProfile,
    Toolbox,
)

CHANGELOG = "CHANGELOG.md"
FEATURE_DIR = "docs/features"
BEGIN_MARKER = "<!-- agentic:design-summary:begin -->"
END_MARKER = "<!-- agentic:design-summary:end -->"
_CAPABILITY_ID = re.compile(r"^[a-z][a-z0-9_-]{0,40}\Z")
_TEXT = 300
_SHORT = 120


def _esc(value: object, limit: int = _TEXT) -> str:
    return escape_markdown(" ".join(str(value).split()), limit)


def _fatal(message: str) -> AgentError:
    return AgentError(message, kind="fatal", retryable=False)


def _capabilities(task_plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """capability id -> params of its first group (groups sorted by id)."""
    found: dict[str, Mapping[str, Any]] = {}
    for group in sorted(task_plan.get("groups") or [], key=lambda g: str(g.get("id"))):
        capability_id = str(group.get("capability"))
        if not _CAPABILITY_ID.match(capability_id):
            raise _fatal("the task plan names a capability with an invalid id")
        found.setdefault(capability_id, group.get("params") or {})
    return found


def _read(box: Toolbox, path: str) -> str | None:
    return box.read_text(path) if box.exists(path) else None


def _requirements_of(
    requirements: Mapping[str, Any], capability_id: str
) -> list[Mapping[str, Any]]:
    ids: set[str] = set()
    for entry in requirements.get("capabilities") or []:
        if entry.get("id") == capability_id:
            ids.update(str(i) for i in entry.get("requirement_ids") or [])
    return [
        r
        for r in requirements.get("requirements") or []
        if r.get("id") in ids or capability_id in (r.get("capabilities") or [])
    ]


def _design_summary(
    title: str,
    reqs: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
) -> str:
    wanted = {str(r["id"]) for r in reqs}
    components = [
        c for c in design.get("components") or [] if wanted & set(c.get("requirement_ids") or [])
    ] or list(design.get("components") or [])
    lines = [
        f"## Design summary: {_esc(title, _SHORT)}",
        "",
        "_Generated from the design document and the requirements; change those, not this block._",
        "",
        _esc(design.get("summary", ""), 600),
        "",
        "### Requirements",
        "",
    ]
    for req in reqs:
        lines.append(f"- **{req['id']}** ({req['priority']}, {req['status']}): {_esc(req['text'])}")
        lines.extend(
            f"  - {ac['id']}: {_esc(ac['text'])}" for ac in req.get("acceptance_criteria") or []
        )
    if components:
        lines += ["", "### Components", ""]
        for comp in components:
            touches = ", ".join(_esc(t, 80) for t in comp.get("touches") or [])
            suffix = f" (touches: {touches})" if touches else ""
            lines.append(
                f"- **{_esc(comp.get('name'), 80)}**: {_esc(comp.get('responsibility'))}{suffix}"
            )
    if design.get("adrs"):
        lines += ["", "### Decisions", ""]
        lines.extend(
            f"- {adr['id']}: {_esc(adr.get('title'), _SHORT)}. {_esc(adr.get('decision'))}"
            f" Rationale: {_esc(adr.get('rationale'))}"
            for adr in design["adrs"]
        )
    if design.get("api_changes"):
        lines += ["", "### API changes", ""]
        lines.extend(
            f"- {_esc(c.get('method'), 12)} {_esc(c.get('path'), 120)} ({c.get('change')}"
            f"{', breaking' if c.get('breaking') else ''})"
            for c in design["api_changes"]
        )
    if design.get("data_model_changes"):
        lines += ["", "### Data model changes", ""]
        lines.extend(
            f"- {_esc(c.get('table'), 80)} ({c.get('change')}): {_esc(c.get('detail'))}"
            for c in design["data_model_changes"]
        )
    risk = design.get("risk") or {}
    reasons = "; ".join(_esc(r, 160) for r in risk.get("reasons") or [])
    lines += [
        "",
        "### Rollback and risk",
        "",
        f"- Rollback plan: {_esc(design.get('rollback_plan', ''), 600)}",
        f"- Risk: {risk.get('level', 'UNKNOWN')}" + (f" ({reasons})" if reasons else ""),
        f"- Change reference: {_esc(design.get('change_reference', ''), 80)}",
    ]
    return "\n".join(lines)


def _with_block(existing: str | None, block: str, title: str) -> str:
    marked = f"{BEGIN_MARKER}\n{block}\n{END_MARKER}\n"
    if existing is None:
        return f"# {_esc(title, _SHORT)}\n\n{marked}"
    start = existing.find(BEGIN_MARKER)
    end = existing.find(END_MARKER, start + 1) if start >= 0 else -1
    if start >= 0 and end > start:
        start = existing.rfind(BEGIN_MARKER, start, end)  # nearest begin: an orphan one stays text
        return existing[:start] + marked + existing[end + len(END_MARKER) :].lstrip("\n")
    return existing.rstrip("\n") + "\n\n" + marked


def _changelog_entry(
    titles: Sequence[str], requirements: Mapping[str, Any], design: Mapping[str, Any]
) -> str:
    heading = ", ".join(_esc(t, 60) for t in titles) or "change"
    lines = [f"## Release notes: {heading}", ""]
    reference = design.get("change_reference")
    if reference:
        lines += [f"Change reference: {_esc(reference, 80)}", ""]
    for req in sorted(requirements.get("requirements") or [], key=lambda r: str(r["id"])):
        lines.append(
            f"- {req['id']} ({req['priority']}, {req['status']}): {_esc(req['text'], 200)}"
        )
    rollback = " ".join(str(design.get("rollback_plan", "")).split())
    if rollback:
        lines += ["", f"Rollback: {_esc(rollback, 300)}"]
    return "\n".join(lines) + "\n"


def _with_entry(existing: str | None, entry: str) -> str | None:
    """The new CHANGELOG text, or ``None`` when the entry is already there."""
    if existing is None:
        return f"# Changelog\n\n{entry}"
    if entry in existing:
        return None
    return existing.rstrip("\n") + "\n\n" + entry


class DocumenterAgent(BaseAgent):
    agent_id = "documenter"
    role = "documenter"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=("docs/**", CHANGELOG, "README.md"),
        can_execute=False,
        max_tool_calls=400,
        max_changed_files=25,
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        box = ctx.toolbox
        requirements = ctx.json(Kind.REQUIREMENTS)
        design = ctx.json(Kind.DESIGN_DOC)
        plan = ctx.json(Kind.TASK_PLAN)
        titles: list[str] = []
        for capability_id, raw in sorted(_capabilities(plan).items()):
            try:
                capability = get_capability(capability_id)
            except KeyError as exc:
                raise _fatal(f"unknown capability {_esc(capability_id, 40)}") from exc
            params = validate_params(capability.recipe.params, raw)
            capability.recipe.build_docs(
                BuildContext(
                    toolbox=box,
                    params=params,
                    requirements=requirements,
                    design=design,
                    anchors={},
                )
            )
            titles.append(capability.title)
            page = f"{FEATURE_DIR}/{capability_id}.md"
            block = _design_summary(
                capability.title, _requirements_of(requirements, capability_id), design
            )
            box.write_text(page, _with_block(_read(box, page), block, capability.title))
        changelog = _with_entry(
            _read(box, CHANGELOG), _changelog_entry(titles, requirements, design)
        )
        if changelog is not None:
            box.write_text(CHANGELOG, changelog)
        staged = box.staged_changeset("documentation for the integrated change")
        return AgentResult(
            artifacts=[changeset_artifact(Kind.DOCS_CHANGESET, staged)],
            rationale=f"documented {len(titles)} capability(ies) in {len(staged.ops)} file(s).",
            usage=dict(box.usage()),
        )
