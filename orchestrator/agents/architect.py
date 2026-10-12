"""Architect agent: assembles ``design_doc`` and ``api_contract`` from the capability recipes.

For every capability the ``requirements`` artifact says is delivered, the agent validates its
parameters (``recipes.base.validate_params``) and merges ``recipe.design(params, model)`` into one
design:

* components: the recipe components, each mapped to the requirement ids the capability delivers (the
  recipe cannot know them), plus generic ones: acceptance tests and documentation (every requirement
  that has a criterion), and an explicit "unsupported scope" component for requirements no
  capability covers, so they stay visible to the ``design_covers_requirements`` gate instead of
  being silently dropped;
* ADRs: the recipe ADRs, one ADR that ties the threat model to acceptance criteria and one on
  delivering as reviewed change sets; numbered ``ADR-001...`` in that order (a DESIGN decision per
  ADR);
* ``data_model_changes`` / ``api_changes`` / ``api_paths`` (-> ``api_contract``) /
  ``rollback_plan``: merged;
* ``risk``: HIGH when there is a data model change or a breaking API change (``breaking`` or
  ``remove``), MEDIUM when the API gains or modifies paths, else LOW; open threat risks and impact
  coupling risks are listed as reasons only;
* ``change_reference``: ``CR-`` + ``stable_hex(goal, requirement texts)``. The request itself is not
  an input of the design node, so the reference is derived from the goal and requirement texts (a
  deterministic function of the request that also changes when an amendment changes the
  requirements).

The model handed to ``recipe.design`` is ``analyze_codebase`` over the toolbox view, exactly what
the implementer's ``BuildContext`` gets, and only when an ``impact_report`` exists (brownfield);
greenfield recipes receive ``None``. The impact report itself contributes its coupling risks to the
risk reasons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from orchestrator.agents import catalog
from orchestrator.agents.base import BaseAgent, adr_id, json_artifact, stable_hex
from orchestrator.agents.recipes.base import Capability, validate_params
from orchestrator.core.errors import AgentError
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    AutonomyProfile,
    DecisionDraft,
    DecisionKind,
)
from orchestrator.tools.codebase import MAX_PY_FILES, MAX_SQL_FILES, analyze_codebase

MAX_REASON_ITEMS = 5
NO_CHANGE = "No change is delivered; nothing to roll back."


@dataclass
class _Design:
    components: list[dict[str, Any]] = field(default_factory=list)
    adrs: list[dict[str, Any]] = field(default_factory=list)
    data_model_changes: list[dict[str, Any]] = field(default_factory=list)
    api_changes: list[dict[str, Any]] = field(default_factory=list)
    api_paths: list[dict[str, Any]] = field(default_factory=list)
    rollback: list[str] = field(default_factory=list)


def _items(value: object) -> list[Mapping[str, Any]]:
    return [v for v in value if isinstance(v, Mapping)] if isinstance(value, list) else []


def _contribute(
    cap: Capability,
    params: Mapping[str, Any],
    ids: Sequence[str],
    model: Any | None,
    design: _Design,
) -> None:
    try:
        contribution = cap.recipe.design(params, model)
        for comp in contribution.get("components", ()):
            design.components.append(
                {
                    "name": comp["name"],
                    "responsibility": comp["responsibility"],
                    "touches": [str(t) for t in comp.get("touches", ())],
                    "requirement_ids": list(ids),
                }
            )
        design.adrs.extend(dict(a) for a in contribution.get("adrs", ()))
        design.data_model_changes.extend(
            dict(d) for d in contribution.get("data_model_changes", ())
        )
        design.api_changes.extend(dict(a) for a in contribution.get("api_changes", ()))
        design.api_paths.extend(dict(p) for p in contribution.get("api_paths", ()))
        plan = contribution.get("rollback_plan")
        if isinstance(plan, str) and plan:
            design.rollback.append(f"{cap.title}: {plan}")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise AgentError(
            f"capability {cap.id}: malformed design contribution ({type(exc).__name__}: {exc})",
            kind="fatal",
            retryable=False,
        ) from exc


def _threat_adr(threat: Mapping[str, Any]) -> dict[str, Any] | None:
    risks = _items(threat.get("risks"))
    if not risks:
        return None

    def named(status: str) -> str:
        found = [str(r.get("id")) for r in risks if r.get("status") == status]
        return ", ".join(found) if found else "none"

    return {
        "title": "Enforce threat mitigations through acceptance criteria",
        "decision": (
            f"Each mitigation is an acceptance criterion with a covering test. Mitigated: "
            f"{named('mitigated')}. Accepted: {named('accepted')}. Open: {named('open')}."
        ),
        "alternatives": [
            {
                "option": "Describe the mitigations in prose only",
                "rejected_because": "prose is not verified by a test, so a regression would go "
                "unnoticed",
            }
        ],
        "rationale": "A test whose docstring names the criterion (Covers:) is the evidence that "
        "release "
        "readiness checks.",
        "consequences": "Open risks stay visible in the threat model until a human accepts or "
        "mitigates them.",
    }


def _delivery_adr() -> dict[str, Any]:
    return {
        "title": "Deliver the change as reviewed change sets with disjoint write scopes",
        "decision": "Independent groups each propose a change set; nothing is written to the "
        "baseline "
        "before the release approval.",
        "alternatives": [
            {
                "option": "Apply edits directly to the baseline",
                "rejected_because": "it bypasses review and cannot be undone as an artifact",
            },
            {
                "option": "One monolithic change set",
                "rejected_because": "no parallel work and a larger review surface",
            },
        ],
        "rationale": "Proposals can be discarded, merged deterministically and verified before "
        "anything "
        "real changes.",
        "consequences": "Two groups that touch one file make the integration gate fail, so every "
        "file "
        "belongs to exactly one group.",
    }


def _risk(
    design: _Design, threat: Mapping[str, Any], impact: Mapping[str, Any] | None
) -> dict[str, Any]:
    breaking = [
        c for c in design.api_changes if c.get("breaking") is True or c.get("change") == "remove"
    ]
    surface = [c for c in design.api_changes if c.get("change") in ("add", "modify")]
    level = "LOW"
    reasons: list[str] = []
    if design.data_model_changes:
        level = "HIGH"
        reasons.append(f"{len(design.data_model_changes)} data model change(s) (migration)")
    if breaking:
        level = "HIGH"
        reasons.append(f"{len(breaking)} breaking API change(s)")
    if level != "HIGH" and surface:
        level = "MEDIUM"
        reasons.append(f"{len(surface)} API path(s) added or modified")
    if not reasons:
        reasons.append("no data model or API surface change")
    open_ids = [str(r.get("id")) for r in _items(threat.get("risks")) if r.get("status") == "open"]
    if open_ids:
        reasons.append(
            f"{len(open_ids)} open threat-model risk(s): {', '.join(open_ids[:MAX_REASON_ITEMS])}"
        )
    if impact is not None:
        reasons.extend(
            f"coupling risk in {c.get('component')}: {c.get('reason')}"
            for c in _items(impact.get("coupling_risks"))[:MAX_REASON_ITEMS]
        )
    return {"level": level, "reasons": reasons}


class ArchitectAgent(BaseAgent):
    agent_id = "architect"
    role = "architect"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=(),
        can_execute=False,
        max_tool_calls=MAX_PY_FILES
        + MAX_SQL_FILES
        + 16,  # the codebase model reads each analysable file
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        requirements = ctx.json(Kind.REQUIREMENTS)
        threat = ctx.json(Kind.THREAT_MODEL)
        impact = ctx.json(Kind.IMPACT_REPORT) if ctx.has(Kind.IMPACT_REPORT) else None
        entries = sorted(_items(requirements.get("capabilities")), key=lambda e: str(e.get("id")))
        model = analyze_codebase(ctx.toolbox) if impact is not None and entries else None

        active = [
            r for r in _items(requirements.get("requirements")) if r.get("status") == "active"
        ]
        active_ids = [str(r.get("id")) for r in active]
        design = _Design()
        for entry in entries:
            try:
                cap = catalog.get_capability(str(entry.get("id")))
            except KeyError as exc:
                raise AgentError(
                    f"the requirements name an unknown capability: {str(entry.get('id'))[:40]!r}",
                    kind="fatal",
                    retryable=False,
                ) from exc
            params = validate_params(cap.recipe.params, entry.get("params") or {})
            ids = [i for i in entry.get("requirement_ids", ()) if i in active_ids]
            _contribute(cap, params, ids, model, design)

        tested = [str(r.get("id")) for r in active if _items(r.get("acceptance_criteria"))]
        if tested:
            design.components.append(
                {
                    "name": "Acceptance tests",
                    "responsibility": "Executable tests derived from the acceptance criteria; "
                    "each test "
                    "names the criteria it covers.",
                    "touches": ["tests/**"],
                    "requirement_ids": tested,
                }
            )
            design.components.append(
                {
                    "name": "Documentation and changelog",
                    "responsibility": "Feature documentation and a changelog entry for the "
                    "delivered behaviour.",
                    "touches": ["docs/**", "CHANGELOG.md"],
                    "requirement_ids": tested,
                }
            )
        unsupported = sorted(
            {
                str(u.get("requirement_id"))
                for u in _items(requirements.get("unsupported"))
                if str(u.get("requirement_id")) in active_ids
            }
        )
        if unsupported:
            design.components.append(
                {
                    "name": "Unsupported scope (needs a human decision)",
                    "responsibility": "No capability can deliver these requirements; the plan "
                    "holds them "
                    "until a human descopes or defers them.",
                    "touches": [],
                    "requirement_ids": unsupported,
                }
            )

        extra = [a for a in (_threat_adr(threat), _delivery_adr()) if a is not None]
        adrs = [
            {**{k: v for k, v in adr.items() if k != "id"}, "id": adr_id(n)}
            for n, adr in enumerate([*design.adrs, *extra], 1)
        ]
        problem = requirements.get("problem")
        goal = str(problem.get("goal", "")) if isinstance(problem, Mapping) else ""
        texts = [str(r.get("text", "")) for r in _items(requirements.get("requirements"))]
        risk = _risk(design, threat, impact)
        titles = [catalog.get_capability(str(e.get("id"))).title for e in entries]
        payload: dict[str, Any] = {
            "summary": (
                f"Deliver {len(active_ids)} active requirement(s) through "
                f"{', '.join(titles) if titles else 'no capability'}: {len(design.components)} "
                f"component(s), {len(adrs)} ADR(s), risk {risk['level']}."
            ),
            "components": design.components,
            "adrs": adrs,
            "data_model_changes": design.data_model_changes,
            "api_changes": design.api_changes,
            "rollback_plan": "\n".join(design.rollback) if design.rollback else NO_CHANGE,
            "change_reference": "CR-" + stable_hex(goal, *texts),
            "risk": risk,
        }
        contract = {
            "paths": design.api_paths,
            "breaking": any(
                c.get("breaking") is True or c.get("change") == "remove" for c in design.api_changes
            ),
        }
        design_artifact = json_artifact(Kind.DESIGN_DOC, payload)
        contract_artifact = json_artifact(Kind.API_CONTRACT, contract)
        decisions = [
            DecisionDraft(
                DecisionKind.DESIGN,
                f"{adr['id']}: {adr['title']}"[:200],
                rationale=adr["rationale"],
                alternatives=tuple(a["option"] for a in adr["alternatives"]),
                outcome=adr["decision"],
                refs=(adr["id"],),
            )
            for adr in adrs
        ]
        return AgentResult(
            artifacts=[design_artifact, contract_artifact],
            decisions=decisions,
            rationale=payload["summary"],
            usage=dict(ctx.toolbox.usage()),
        )
