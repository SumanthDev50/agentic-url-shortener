"""Implementer agents: turn one task-plan group into a change-set proposal through a recipe.

``ImplementerAgent`` (``implementer``) is the primary agent of every
``implement.<capability>-<group>`` node and ``SafeImplementerAgent`` (``implementer.safe``) is its
conservative fallback.  Both are deterministic stand-ins for a model-backed engineer: they cannot
implement anything a recipe does not know, and they never write outside the toolbox.

Assignment (``ctx.params``, written by the planner's expansion template; this node's ``task_plan``
group fills any key that is missing):

* ``group``         the recipe group to build (``schemas.GROUP_ID_RE``);
* ``capability``    capability id in the catalog;
* ``recipe_params`` typed recipe parameters, validated with ``recipes.base.validate_params`` before
  any recipe code runs (unknown keys, wrong types and out-of-range values are fatal, never retried);
* ``write_scope``   globs the group may touch.  The toolbox already enforces the node's narrowed
  autonomy; the agent re-checks the finished change set against ``write_scope`` (defence in depth).

Flow: validate -> ``recipe.build_group(group, BuildContext(...))`` through ``ctx.toolbox`` (staged
writes, nothing on disk changes) -> ``toolbox.staged_changeset()`` -> a ``changeset`` artifact.  For
capabilities that need existing code (``requires_baseline``) the agent hands the recipe a
``CodebaseModel`` built from the toolbox view (skipped, with a note, when the node's tool-call
budget cannot afford one read per source file); recipes still resolve their anchors against file
content and raise ``RecipeAnchorError`` on drift (fatal: the safe agent would hit the same drift).
A group that produces no file change at all is a fatal error, never an empty success.

What "conservative" means for ``implementer.safe`` (concrete and testable):

1. ``BuildContext.anchors['conservative']`` is ``True`` (the primary passes ``False``).  A recipe
   that honours the flag drops optional extras and optimisations (for example a cache-bypass fast
   path) and keeps only the minimal anchored edits.  A recipe that ignores the flag yields the same
   change set for both agents, which is acceptable and documented: the fallback then only adds the
   two guards below.
2. The change set may not contain DELETE operations (the fallback never removes files).
3. Tighter autonomy budgets than the primary (``max_changed_files``, ``max_total_bytes``).

Everything in the output is deterministic: no times, durations or random ids reach artifact content.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator.agents.base import BaseAgent, changeset_artifact
from orchestrator.agents.catalog import get_capability
from orchestrator.agents.recipes.base import BuildContext, Capability, validate_params
from orchestrator.core.errors import AgentError, UndeclaredInputError
from orchestrator.core.paths import match_any
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.schemas import GROUP_ID_RE, REQ_ID_RE, TASK_ID_RE, Kind
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    AutonomyProfile,
    ChangeSet,
    DecisionDraft,
    DecisionKind,
    FileOpKind,
)
from orchestrator.tools.codebase import CodebaseModel, analyze_codebase

CapabilityLookup = Callable[[str], Capability]

MAX_TOOL_CALLS = 6000  # a whole-workspace read for the codebase model (<= 5000 files) + recipe I/O
MODEL_RESERVE = 64  # tool calls kept back for the recipe's own reads and writes
MAX_SCOPE_GLOBS = 50
MAX_REFS = 60
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}\Z")
_SHOWN = 80


def _fatal(message: str) -> AgentError:
    return AgentError(message, kind="fatal", retryable=False)


def _shown(value: object) -> str:
    return sanitize_for_terminal(value, _SHOWN)


@dataclass(frozen=True)
class GroupAssignment:
    """What one ``implement.*`` node has to build (type-checked, but values are still untrusted)."""

    group: str
    capability: str
    recipe_params: Mapping[str, Any]
    write_scope: tuple[str, ...]
    task_ids: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()


# ----------------------------------------------------------------------------- inputs


def _optional_json(ctx: AgentContext, kind: str) -> Mapping[str, Any] | None:
    """Payload of ``kind`` if the node declared it and it exists (an undeclared kind is absent)."""
    try:
        if not ctx.has(kind):
            return None
        payload = ctx.json(kind)
    except UndeclaredInputError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _plan_group(plan: Mapping[str, Any] | None, node_id: str) -> Mapping[str, Any] | None:
    """The ``task_plan`` group owned by ``node_id``."""
    groups = plan.get("groups") if plan else None
    for entry in groups if isinstance(groups, list) else []:
        if isinstance(entry, Mapping) and entry.get("node_id") == node_id:
            return entry
    return None


def _plan_tasks(plan: Mapping[str, Any] | None, task_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    listed = plan.get("tasks") if plan else None
    found = listed if isinstance(listed, list) else []
    return [t for t in found if isinstance(t, Mapping) and t.get("id") in task_ids]


def _recipe_group(tasks: Sequence[Mapping[str, Any]]) -> str | None:
    """The recipe's group id the planner recorded in the group's tasks (``params.recipe_group``)."""
    for task in tasks:
        task_params = task.get("params")
        recipe_group = task_params.get("recipe_group") if isinstance(task_params, Mapping) else None
        if isinstance(recipe_group, str):
            return recipe_group
    return None


def _strings(value: object, what: str, limit: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _fatal(f"{what} must be a list of strings")
    if len(value) > limit or not all(isinstance(v, str) and v for v in value):
        raise _fatal(f"{what} must hold at most {limit} non-empty strings")
    return tuple(value)


def _ids(value: object, pattern: re.Pattern[str]) -> list[str]:
    """The well-formed ids in ``value`` (anything else, including a non-list, is dropped)."""
    items = value if isinstance(value, list) else []
    return [v for v in items if isinstance(v, str) and pattern.match(v)]


def read_assignment(ctx: AgentContext) -> GroupAssignment:
    """Parse ``ctx.params`` (this node's ``task_plan`` group fills gaps); reject bad shapes.

    ``group`` is the *recipe's* group id.  When the node does not carry it, it is the
    ``recipe_group`` the planner recorded in the plan's tasks of this node's group (the plan's own
    group id is ``<capability>-<group>`` and names a node, not a recipe group).
    """
    params = ctx.params
    plan = _optional_json(ctx, Kind.TASK_PLAN)
    entry = _plan_group(plan, ctx.node.id)
    fallback: Mapping[str, Any] = entry if entry is not None else {}
    task_ids = _ids(params.get("task_ids", fallback.get("task_ids")), TASK_ID_RE)[:MAX_REFS]
    tasks = _plan_tasks(plan, task_ids)
    requirements = sorted({r for t in tasks for r in _ids(t.get("requirement_ids"), REQ_ID_RE)})
    group = params.get("group", _recipe_group(tasks) or fallback.get("id"))
    capability = params.get("capability", fallback.get("capability"))
    recipe_params = params.get("recipe_params", fallback.get("params"))
    scope = params.get("write_scope", fallback.get("write_scope", ()))
    if not isinstance(group, str) or not GROUP_ID_RE.match(group):
        raise _fatal(f"node {ctx.node.id} has no valid 'group' parameter")
    if not isinstance(capability, str) or not _CAPABILITY_RE.match(capability):
        raise _fatal(f"node {ctx.node.id} has no valid 'capability' parameter")
    if not isinstance(recipe_params, Mapping):
        raise _fatal(f"node {ctx.node.id} has no 'recipe_params' mapping")
    return GroupAssignment(
        group,
        capability,
        dict(recipe_params),
        _strings(scope, "write_scope", MAX_SCOPE_GLOBS),
        tuple(task_ids),
        tuple(requirements[:MAX_REFS]),
    )


# ----------------------------------------------------------------------------- agents


class ImplementerAgent(BaseAgent):
    """Primary implementer: builds one group of one capability through its recipe."""

    agent_id = "implementer"
    role = "implementer"
    version = "1"
    # ``**`` so that a node's narrower ``write_scope`` survives ``AutonomyProfile.narrow`` (a node
    # may only narrow, and a pattern is kept only when the agent's profile provably covers it).
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=("**",),
        can_execute=False,
        max_tool_calls=MAX_TOOL_CALLS,
    )
    conservative = False

    def __init__(self, capabilities: CapabilityLookup | None = None) -> None:
        self._lookup: CapabilityLookup = (
            capabilities if capabilities is not None else get_capability
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        box = ctx.toolbox
        work = read_assignment(ctx)
        cap = self._capability(work.capability)
        recipe = cap.recipe
        params = validate_params(recipe.params, work.recipe_params)
        spec = next((g for g in recipe.groups(params) if g.id == work.group), None)
        if spec is None:
            raise _fatal(f"capability {cap.id} has no group {_shown(work.group)!r}")
        if ctx.feedback:
            box.note(
                "re-running with engine feedback",
                rule_ids=sorted({f.rule_id for f in ctx.feedback}),
            )
        model = self._codebase_model(ctx) if cap.requires_baseline else None
        build = BuildContext(
            toolbox=box,
            params=params,
            model=model,
            design=_optional_json(ctx, Kind.DESIGN_DOC),
            anchors={"conservative": self.conservative},
        )
        recipe.build_group(work.group, build)
        proposal = box.staged_changeset(f"implement {cap.id}/{work.group}: {spec.title}")
        self._vet(proposal, work)
        mode = "conservative" if self.conservative else "standard"
        return AgentResult(
            artifacts=[changeset_artifact(Kind.CHANGESET, proposal)],
            decisions=[
                DecisionDraft(
                    kind=DecisionKind.DESIGN,
                    summary=(
                        f"implement group {work.group} of capability {cap.id} "
                        f"with recipe {recipe.id}"
                    ),
                    rationale=f"{mode} edits anchored on the baseline; parameters validated first",
                    outcome=f"{len(proposal.ops)} file(s) proposed",
                    refs=(*work.task_ids, *work.requirement_ids),
                    data={
                        "group": work.group,
                        "capability": cap.id,
                        "conservative": self.conservative,
                        "paths": list(proposal.paths()),
                    },
                )
            ],
            rationale=f"{cap.id}/{work.group}: {len(proposal.ops)} file(s) proposed ({mode} mode).",
            usage=dict(box.usage()),
        )

    def _codebase_model(self, ctx: AgentContext) -> CodebaseModel | None:
        """The model costs one tool call per Python / SQL file: skipped when the budget is short.

        Recipes treat the model as an optional verification aid (they still anchor every edit on
        file content), so a workspace too large for the node's tool-call budget degrades to
        unverified symbols instead of failing the node; the skip is audited with a note.
        """
        box = ctx.toolbox
        budget = self.autonomy.narrow(ctx.node.autonomy).max_tool_calls
        files = {*box.list_files("**/*.py"), *box.list_files("**/*.sql")}
        remaining = budget - box.usage()["tool_calls"]
        if len(files) + MODEL_RESERVE + 4 > remaining:
            box.note("codebase model skipped: tool-call budget too small", files=len(files))
            return None
        return analyze_codebase(box)

    def _capability(self, capability_id: str) -> Capability:
        try:
            return self._lookup(capability_id)
        except KeyError as exc:
            raise _fatal(f"unknown capability {_shown(capability_id)!r}") from exc

    def _vet(self, proposal: ChangeSet, work: GroupAssignment) -> None:
        """Checks every implementer applies to the finished change set."""
        if not proposal.ops:
            raise _fatal(f"group {work.group} produced no file changes")
        if work.write_scope:
            outside = [p for p in proposal.paths() if not match_any(p, work.write_scope)]
            if outside:
                shown = ", ".join(_shown(p) for p in outside[:3])
                raise _fatal(f"group {work.group} wrote outside its write_scope: {shown}")


class SafeImplementerAgent(ImplementerAgent):
    """Conservative fallback (the module docstring defines what *conservative* means)."""

    agent_id = "implementer.safe"
    conservative = True
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=("**",),
        can_execute=False,
        max_tool_calls=MAX_TOOL_CALLS,
        max_changed_files=24,
        max_total_bytes=2 * 1024 * 1024,
    )

    def _vet(self, proposal: ChangeSet, work: GroupAssignment) -> None:
        super()._vet(proposal, work)
        deleted = [op.path for op in proposal.ops if op.kind is FileOpKind.DELETE]
        if deleted:
            shown = ", ".join(_shown(p) for p in deleted[:3])
            raise _fatal(f"the conservative implementer never deletes files: {shown}")
