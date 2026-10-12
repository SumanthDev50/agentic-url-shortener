"""Test author: generates the tests of every *active* capability from the acceptance criteria.

``TestAuthorAgent`` (``test-author``) runs in the ``author-tests`` node, parallel to
``implement.*``.  For each capability referenced by at least one active requirement it calls
``recipe.build_tests(BuildContext(...))`` exactly once through ``ctx.toolbox`` and returns the
staged files as one ``changeset`` artifact (tests only: the agent's autonomy is ``tests/**``).

* Active capability: referenced by a requirement with ``status == 'active'``.  Deferred and
  descoped requirements are removed from the requirements payload before a recipe sees it, so no
  test is ever generated for a feature that was deliberately not built.  Each recipe additionally
  only sees the requirements of its own capability, as a private deep copy (a recipe cannot alter
  shared state).
* Recipe parameters: the ``params`` of the capability's ``task_plan`` groups (the very values the
  implementers use, so tests and code cannot diverge), else the ``params`` of the capability entry
  in the requirements.  They are validated with ``recipes.base.validate_params`` before any recipe
  runs.
* Tests are derived from the requirements, not from the code, so no ``CodebaseModel`` is built
  (``BuildContext.model`` stays ``None``; recipes treat the model as optional).
* Traceability: the recipes write the ``Covers: R-001, AC-001-2`` docstring lines from the
  acceptance criteria; the ``tests_cover_acceptance_criteria`` gate (not this agent) enforces
  completeness.
* No active capability -> an empty change set (nothing to test).  Active capabilities whose
  recipes stage no file at all are a fatal error: silence would look like a pass.

``__test__ = False`` keeps pytest from collecting the class when it is imported into a test module.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from typing import Any

from orchestrator.agents.base import BaseAgent, changeset_artifact
from orchestrator.agents.catalog import get_capability
from orchestrator.agents.recipes.base import BuildContext, Capability, validate_params
from orchestrator.core.errors import AgentError, UndeclaredInputError
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.schemas import REQ_ID_RE, Kind
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    AutonomyProfile,
    DecisionDraft,
    DecisionKind,
)

CapabilityLookup = Callable[[str], Capability]

MAX_TOOL_CALLS = 1000  # recipe reads and writes only (no workspace-wide analysis)
MAX_REFS = 60
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}\Z")


def _fatal(message: str) -> AgentError:
    return AgentError(message, kind="fatal", retryable=False)


def _optional_json(ctx: AgentContext, kind: str) -> Mapping[str, Any] | None:
    """Payload of ``kind`` if the node declared it and it exists (an undeclared kind is absent)."""
    try:
        if not ctx.has(kind):
            return None
        payload = ctx.json(kind)
    except UndeclaredInputError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _requirement_entries(requirements: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = requirements.get("requirements")
    if not isinstance(entries, list) or not all(isinstance(r, Mapping) for r in entries):
        raise _fatal("the requirements artifact has no 'requirements' list of objects")
    return entries


def _capabilities_of(entry: Mapping[str, Any]) -> list[str]:
    caps = entry.get("capabilities", [])
    if not isinstance(caps, list):
        raise _fatal("a requirement has a malformed 'capabilities' list")
    return [c for c in caps if isinstance(c, str)]


def active_capabilities(requirements: Mapping[str, Any]) -> dict[str, list[str]]:
    """``{capability id: sorted ids of its active requirements}``, ordered by capability id."""
    found: dict[str, set[str]] = {}
    for entry in _requirement_entries(requirements):
        rid = entry.get("id")
        if entry.get("status") == "active" and isinstance(rid, str):
            for cap in _capabilities_of(entry):
                found.setdefault(cap, set()).add(rid)
    return {cap: sorted(found[cap]) for cap in sorted(found)}


def requirements_view(requirements: Mapping[str, Any], capability_id: str) -> dict[str, Any]:
    """Private deep copy holding only the active requirements that belong to ``capability_id``."""
    view = copy.deepcopy(dict(requirements))
    view["requirements"] = [
        copy.deepcopy(dict(r))
        for r in _requirement_entries(requirements)
        if r.get("status") == "active" and capability_id in _capabilities_of(r)
    ]
    return view


def _raw_params(
    capability_id: str, requirements: Mapping[str, Any], plan: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    """Recipe parameters of the capability: the plan's groups first, then the requirements."""
    groups = plan.get("groups") if plan else None
    plan_groups = [g for g in groups or [] if isinstance(g, Mapping)]
    for group in sorted(plan_groups, key=lambda g: str(g.get("id"))):
        params = group.get("params")
        if group.get("capability") == capability_id and isinstance(params, Mapping):
            return params
    for entry in requirements.get("capabilities") or []:
        if isinstance(entry, Mapping) and entry.get("id") == capability_id:
            params = entry.get("params")
            if isinstance(params, Mapping):
                return params
    return {}


class TestAuthorAgent(BaseAgent):
    """Builds the tests of each active capability (see the module docstring)."""

    __test__ = False  # not a pytest test class

    agent_id = "test-author"
    role = "test-author"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=("tests/**",),
        can_execute=False,
        max_tool_calls=MAX_TOOL_CALLS,
    )

    def __init__(self, capabilities: CapabilityLookup | None = None) -> None:
        self._lookup: CapabilityLookup = (
            capabilities if capabilities is not None else get_capability
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        box = ctx.toolbox
        requirements = ctx.json(Kind.REQUIREMENTS)
        if not isinstance(requirements, Mapping):
            raise _fatal("the requirements artifact is not an object")
        plan = _optional_json(ctx, Kind.TASK_PLAN)
        design = _optional_json(ctx, Kind.DESIGN_DOC)
        active = active_capabilities(requirements)
        if ctx.feedback:
            box.note(
                "re-running with engine feedback",
                rule_ids=sorted({f.rule_id for f in ctx.feedback}),
            )
        for cap_id in active:
            cap = self._capability(cap_id)
            params = validate_params(cap.recipe.params, _raw_params(cap_id, requirements, plan))
            cap.recipe.build_tests(
                BuildContext(
                    toolbox=box,
                    params=params,
                    requirements=requirements_view(requirements, cap_id),
                    design=design,
                    anchors={},
                )
            )
        names = ", ".join(active) or "none"
        proposal = box.staged_changeset(f"author tests for capabilities: {names}")
        if active and not proposal.ops:
            raise _fatal("the recipes of the active capabilities staged no test files")
        decisions: list[DecisionDraft] = []
        if active:
            refs = sorted({r for ids in active.values() for r in ids if REQ_ID_RE.match(r)})
            decisions.append(
                DecisionDraft(
                    kind=DecisionKind.DESIGN,
                    summary=f"author tests for capabilities: {names}",
                    rationale="one suite per active capability, derived from its criteria",
                    outcome=f"{len(proposal.ops)} test file(s) proposed",
                    refs=tuple(refs[:MAX_REFS]),
                    data={"capabilities": list(active), "paths": list(proposal.paths())},
                )
            )
        return AgentResult(
            artifacts=[changeset_artifact(Kind.CHANGESET, proposal)],
            decisions=decisions,
            rationale=f"{len(proposal.ops)} test file(s) for {len(active)} active capability(ies).",
            usage=dict(box.usage()),
        )

    def _capability(self, capability_id: str) -> Capability:
        if not _CAPABILITY_RE.match(capability_id):
            raise _fatal(f"malformed capability id {sanitize_for_terminal(capability_id, 60)!r}")
        try:
            return self._lookup(capability_id)
        except KeyError as exc:
            raise _fatal(f"unknown capability {capability_id!r}") from exc
