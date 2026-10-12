"""Planner agent: decomposes the design into tasks, groups and the ``implement.*`` expansion.

Per delivered capability (``requirements.capabilities``, sorted by id) the agent validates the
parameters and asks the recipe for its parallel groups (``recipe.groups(params)``). Group ids must
be unique, their ``depends_on`` must name sibling groups and must not be cyclic; groups are emitted
in a stable topological order (recipe order wherever dependencies allow it).

* ``task_plan.tasks``: ``T-001...``: one ``implement`` task per group (``depends_on`` from
  ``GroupSpec.depends_on``), then one ``test`` task (authored from the criteria, parallel to the
  implementation) and one ``doc`` task (after everything) when something is implemented.
* ``task_plan.groups``: ``{id: '<capability>-<group>', node_id: 'implement.<capability>-<group>'}``;
  ids longer than the schema allows (31 characters) are shortened to 22 characters + ``-`` + an 8
  hex digest.
* ``task_plan.critical_path``: the longest ``depends_on`` chain (ties: the earliest task).
* ``task_plan.unsupported``: the active requirements no capability covers (the
  ``plan_covers_requirements`` gate turns them into a clarification; they are never planned or
  invented).
* ``AgentResult.plan_expansion``: one ``implement.*`` node per group (``agent='implementer'``,
  autonomy narrowed to the group's ``write_scope``, ``params = {group, capability, recipe_params,
  task_ids, write_scope, expected_paths}`` where ``group`` is the *recipe's* group id) and
  ``rewire={'integrate': Rewire(add_depends_on=<all implement nodes>)}``. The engine overwrites
  every governance attribute from the owner's template; the node count is checked against
  ``expansion_policy.max_nodes`` when the owner has one.

The plan must follow the approved design: an active requirement that neither a design component
covers nor the requirements list as unsupported is a fatal inconsistency (the design gate should
have caught it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orchestrator.agents import catalog
from orchestrator.agents.base import BaseAgent, json_artifact, stable_hex, task_id
from orchestrator.agents.recipes.base import GroupSpec, validate_params
from orchestrator.core.errors import AgentError
from orchestrator.core.paths import NODE_ID_RE
from orchestrator.core.schemas import GROUP_ID_RE, Kind
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    AutonomyProfile,
    NodeSpec,
    PlanExpansion,
    Rewire,
    Stage,
)

INTEGRATE_NODE = "integrate"
IMPLEMENT_PREFIX = "implement."
MAX_GROUP_ID_CHARS = 31  # schemas.GROUP_ID_RE
TEST_SCOPE = ("tests/**",)
DOC_SCOPE = ("docs/**", "CHANGELOG.md")


def _fatal(message: str) -> AgentError:
    return AgentError(message, kind="fatal", retryable=False)


def _items(value: object) -> list[Mapping[str, Any]]:
    return [v for v in value if isinstance(v, Mapping)] if isinstance(value, list) else []


def _ordered(cap_id: str, specs: Sequence[GroupSpec]) -> list[GroupSpec]:
    """Validate one recipe's groups and return them in stable topological order."""
    if not specs:
        raise _fatal(f"capability {cap_id}: the recipe returned no groups")
    ids = [s.id for s in specs]
    if len(set(ids)) != len(ids):
        raise _fatal(f"capability {cap_id}: duplicate group ids")
    for spec in specs:
        unknown = [d for d in spec.depends_on if d not in ids]
        if unknown:
            raise _fatal(
                f"capability {cap_id}: group {spec.id} depends on unknown group(s) {unknown}"
            )
    ordered: list[GroupSpec] = []
    done: set[str] = set()
    pending = list(specs)
    while pending:
        ready = next((s for s in pending if all(d in done for d in s.depends_on)), None)
        if ready is None:
            raise _fatal(f"capability {cap_id}: cyclic group dependencies")
        ordered.append(ready)
        done.add(ready.id)
        pending.remove(ready)
    return ordered


def _plan_group_id(cap_id: str, group_id: str) -> str:
    gid = f"{cap_id}-{group_id}"
    if len(gid) > MAX_GROUP_ID_CHARS:
        gid = f"{gid[:22].rstrip('-_')}-{stable_hex(gid, n=8)}"
    if not GROUP_ID_RE.match(gid) or not NODE_ID_RE.match(f"{IMPLEMENT_PREFIX}{gid}"):
        raise _fatal(f"capability {cap_id}: group id {group_id!r} does not give a valid node id")
    return gid


def _critical_path(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    """Longest ``depends_on`` chain; ``tasks`` are in dependency order."""
    best: dict[str, list[str]] = {}
    for task in tasks:
        longest = max((best[d] for d in task["depends_on"]), key=len, default=[])
        best[task["id"]] = [*longest, task["id"]]
    return max(best.values(), key=len, default=[])


def _max_nodes(ctx: AgentContext) -> int | None:
    policy = ctx.node.params.get("expansion_policy")
    limit = policy.get("max_nodes") if isinstance(policy, Mapping) else None
    return limit if isinstance(limit, int) and not isinstance(limit, bool) else None


class PlannerAgent(BaseAgent):
    agent_id = "planner"
    role = "planner"
    version = "1"
    autonomy = AutonomyProfile(read_globs=(), write_globs=(), can_execute=False, max_tool_calls=8)

    def run(self, ctx: AgentContext) -> AgentResult:
        requirements = ctx.json(Kind.REQUIREMENTS)
        design = ctx.json(Kind.DESIGN_DOC)
        active_ids = [
            str(r.get("id"))
            for r in _items(requirements.get("requirements"))
            if r.get("status") == "active"
        ]
        unsupported = [
            {"requirement_id": str(u.get("requirement_id")), "reason": str(u.get("reason"))}
            for u in _items(requirements.get("unsupported"))
        ]
        covered = {
            i
            for comp in _items(design.get("components"))
            for i in comp.get("requirement_ids", ())
            if isinstance(i, str)
        }
        held = {u["requirement_id"] for u in unsupported}
        missing = [i for i in active_ids if i not in covered and i not in held]
        if missing:
            raise _fatal(
                f"the design does not cover active requirement(s): {', '.join(missing[:5])}"
            )

        tasks: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        nodes: list[NodeSpec] = []
        delivered: set[str] = set()
        entries = sorted(_items(requirements.get("capabilities")), key=lambda e: str(e.get("id")))
        for entry in entries:
            try:
                cap = catalog.get_capability(str(entry.get("id")))
            except KeyError as exc:
                raise _fatal(
                    f"the requirements name an unknown capability: {str(entry.get('id'))[:40]!r}"
                ) from exc
            params = validate_params(cap.recipe.params, entry.get("params") or {})
            ids = [i for i in entry.get("requirement_ids", ()) if i in active_ids]
            delivered.update(ids)
            task_of: dict[str, str] = {}
            node_of: dict[str, str] = {}
            for spec in _ordered(cap.id, cap.recipe.groups(params)):
                tid = task_id(len(tasks) + 1)
                gid = _plan_group_id(cap.id, spec.id)
                node_id = f"{IMPLEMENT_PREFIX}{gid}"
                if any(g["id"] == gid for g in groups):
                    raise _fatal(f"duplicate plan group id {gid!r}")
                task_of[spec.id], node_of[spec.id] = tid, node_id
                title = f"Implement {cap.title}: {spec.title}"
                tasks.append(
                    {
                        "id": tid,
                        "title": title,
                        "group": gid,
                        "requirement_ids": list(ids),
                        "depends_on": [task_of[d] for d in spec.depends_on],
                        "capability": cap.id,
                        "kind": "implement",
                        "write_scope": list(spec.write_scope),
                        "expected_paths": list(spec.expected_paths),
                        "params": {"recipe_group": spec.id, "recipe_params": dict(params)},
                    }
                )
                groups.append(
                    {
                        "id": gid,
                        "node_id": node_id,
                        "task_ids": [tid],
                        "capability": cap.id,
                        "params": dict(params),
                        "write_scope": list(spec.write_scope),
                    }
                )
                nodes.append(
                    NodeSpec(
                        id=node_id,
                        title=title,
                        stage=Stage.IMPLEMENTATION,
                        agent="implementer",
                        fallback_agent="implementer.safe",
                        depends_on=(ctx.node.id, *(node_of[d] for d in spec.depends_on)),
                        consumes=(Kind.TASK_PLAN, Kind.DESIGN_DOC),
                        optional_consumes=(Kind.IMPACT_REPORT,),
                        produces=(Kind.CHANGESET,),
                        params={
                            "group": spec.id,
                            "capability": cap.id,
                            "recipe_params": dict(params),
                            "task_ids": [tid],
                            "write_scope": list(spec.write_scope),
                            "expected_paths": list(spec.expected_paths),
                        },
                        autonomy=AutonomyProfile(write_globs=tuple(spec.write_scope)),
                    )
                )

        limit = _max_nodes(ctx)
        if limit is not None and len(nodes) > limit:
            raise _fatal(
                f"the plan needs {len(nodes)} implementation nodes but the template allows {limit}"
            )
        if tasks:
            ids = [i for i in active_ids if i in delivered]
            implement_ids = [t["id"] for t in tasks]
            test_id, doc_id = task_id(len(tasks) + 1), task_id(len(tasks) + 2)
            tasks.append(
                {
                    "id": test_id,
                    "title": "Author tests from the acceptance criteria",
                    "group": "tests",
                    "requirement_ids": ids,
                    "depends_on": [],
                    "capability": "",
                    "kind": "test",
                    "write_scope": list(TEST_SCOPE),
                    "expected_paths": list(TEST_SCOPE),
                    "params": {},
                }
            )
            tasks.append(
                {
                    "id": doc_id,
                    "title": "Document the delivered behaviour",
                    "group": "docs",
                    "requirement_ids": ids,
                    "depends_on": [*implement_ids, test_id],
                    "capability": "",
                    "kind": "doc",
                    "write_scope": list(DOC_SCOPE),
                    "expected_paths": list(DOC_SCOPE),
                    "params": {},
                }
            )
        payload = {
            "tasks": tasks,
            "groups": groups,
            "critical_path": _critical_path(tasks),
            "unsupported": unsupported,
        }
        rewire = (
            {INTEGRATE_NODE: Rewire(add_depends_on=tuple(n.id for n in nodes))} if nodes else {}
        )
        return AgentResult(
            artifacts=[json_artifact(Kind.TASK_PLAN, payload)],
            plan_expansion=PlanExpansion(ctx.node.id, tuple(nodes), rewire),
            rationale=(
                f"{len(tasks)} task(s) in {len(groups)} group(s); "
                f"{len(unsupported)} unsupported requirement(s)."
            ),
            usage=dict(ctx.toolbox.usage()),
        )
