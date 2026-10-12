"""Threat-modeler agent: a STRIDE-lite rule table over the requirements (deterministic, read-only).

Each rule of :data:`_RULES` fires when one of its patterns matches the delivered scope (goal,
in-scope statements, the text and the acceptance criteria of every *active* requirement) or when one
of its capability ids is delivered. The text is hostile input: it is only matched by linear-time
patterns, never interpreted.

A fired rule becomes a ``RISK-nnn`` entry (rule-table order). Its mitigations are *real*:

* every acceptance criterion whose capability ``aspect`` is one of the rule's ``needs`` groups
  becomes a mitigation that cites the requirement id, the criterion id and the ``Covers:`` line a
  test must carry;
* all ``needs`` groups covered (each group is "any of these aspects")  =>  ``mitigated``;
* otherwise ``accepted`` when the rule documents an acceptable residual risk (the reason is recorded
  as a mitigation text), else ``open`` with a *proposed* mitigation (no requirement id: nothing
  covers it yet).

Limits (documented, not hidden): keyword rules over English text, no data-flow analysis; a risk that
no rule names is not found. The authoritative guardrails remain the policy engine and the human
approvals.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator.agents.base import BaseAgent, json_artifact, risk_id
from orchestrator.core.schemas import Kind
from orchestrator.core.types import AgentContext, AgentResult, AutonomyProfile

DEFAULT_ASSETS = ("delivered change set (integrity)",)
DEFAULT_BOUNDARIES = ("reviewed change proposal to baseline (human approval gate)",)


@dataclass(frozen=True)
class _Rule:
    key: str
    category: str  # STRIDE value of the ``threat_model`` schema
    title: str
    description: str
    patterns: tuple[re.Pattern[str], ...]
    capabilities: tuple[str, ...]
    likelihood: str
    impact: str
    needs: tuple[
        tuple[str, ...], ...
    ]  # every group needs one covering aspect (any-of inside a group)
    proposal: str
    accept_reason: str | None
    assets: tuple[str, ...]
    boundaries: tuple[str, ...]


def _rule(
    key: str,
    category: str,
    title: str,
    description: str,
    *,
    patterns: Sequence[str] = (),
    capabilities: Sequence[str] = (),
    likelihood: str,
    impact: str,
    needs: Sequence[Sequence[str]] = (),
    proposal: str,
    accept_reason: str | None = None,
    assets: Sequence[str],
    boundaries: Sequence[str],
) -> _Rule:
    return _Rule(
        key,
        category,
        title,
        description,
        tuple(re.compile(p, re.IGNORECASE) for p in patterns),
        tuple(capabilities),
        likelihood,
        impact,
        tuple(tuple(group) for group in needs),
        proposal,
        accept_reason,
        tuple(assets),
        tuple(boundaries),
    )


_ENDPOINT = (
    r"\bendpoints?\b",
    r"\broutes?\b",
    r"\b(?:get|post|put|patch|delete)\s+/\S*",
    r"\bhttp api\b",
)

_RULES: tuple[_Rule, ...] = (
    _rule(
        "ssrf",
        "spoofing",
        "SSRF or spoofed destination through user-supplied URLs",
        "A URL supplied by a client can name an internal service, a loopback or link-local "
        "address or a "
        "non-web scheme, so the server contacts or redirects to a destination the attacker chose.",
        patterns=(r"\b(?:urls?|uris?|destinations?|hyperlinks?)\b",),
        capabilities=("linkcheck",),
        likelihood="medium",
        impact="high",
        needs=(("ssrf-block",), ("scheme-allowlist",)),
        proposal="Add acceptance criteria that block private, loopback and link-local "
        "destinations and "
        "allow only http and https.",
        assets=("destination URLs of short links", "internal network reachable from the server"),
        boundaries=("untrusted destination URL to server-side request",),
    ),
    _rule(
        "csv_injection",
        "tampering",
        "Formula injection through exported CSV cells",
        "Exported cells that start with =, +, - or @ are executed as formulas by spreadsheet "
        "software, so "
        "attacker-controlled data (for example a referrer) can run on the reader's machine.",
        patterns=(r"\bcsv\b", r"\bspreadsheets?\b", r"\bxlsx?\b", r"\bexports?\b"),
        capabilities=("analytics_export",),
        likelihood="medium",
        impact="medium",
        needs=(("formula-injection",),),
        proposal="Neutralise cells that start with =, +, - or @ before they are written to the "
        "export.",
        assets=("exported analytics data", "spreadsheet user environment"),
        boundaries=("exported file to spreadsheet application",),
    ),
    _rule(
        "race",
        "tampering",
        "Concurrent requests bypass a counter or limit",
        "A check-then-update on a shared counter lets parallel requests all pass the check, so a "
        "limit "
        "(for example a click allowance) is exceeded.",
        patterns=(
            r"\bconcurren\w*",
            r"\brace conditions?\b",
            r"\bsimultaneous\w*",
            r"\bparallel\b",
            r"\batomic\w*",
            r"\bcounters?\b",
            r"\bmax_clicks\b",
            r"\bclick limits?\b",
            r"\bsingle-use\b",
            r"\blimited-use\b",
        ),
        capabilities=("max_clicks",),
        likelihood="high",
        impact="high",
        needs=(("concurrency-safe",),),
        proposal="Add an acceptance criterion that parallel requests cannot exceed the limit "
        "(atomic conditional update).",
        assets=("click counter and link limit",),
        boundaries=("concurrent public requests to shared counter storage",),
    ),
    _rule(
        "exhaustion",
        "denial_of_service",
        "A limited resource is exhausted by third parties",
        "Anyone who can follow a limited-use link can use up its allowance before the intended "
        "audience.",
        patterns=(
            r"\bsingle-use\b",
            r"\blimited-use\b",
            r"\bmax_clicks\b",
            r"\bclick limits?\b",
            r"\bquota\b",
        ),
        capabilities=("max_clicks",),
        likelihood="medium",
        impact="low",
        proposal="Operators can disable or recreate the link; add rate limiting if abuse is "
        "observed.",
        accept_reason="exhausting a limited-use link is inherent to a click limit; it affects one "
        "link, "
        "not the service",
        assets=("availability of limited-use links",),
        boundaries=("public redirect endpoint to limited-use link",),
    ),
    _rule(
        "endpoint_auth",
        "spoofing",
        "Anonymous access to a new or changed endpoint",
        "A new route that does not declare the authentication dependency of its sibling routes "
        "can be "
        "called by anyone.",
        patterns=_ENDPOINT,
        capabilities=("analytics_export",),
        likelihood="medium",
        impact="high",
        needs=(("auth-required",),),
        proposal="Require the existing API-key dependency on the route and test that an anonymous "
        "call "
        "is rejected.",
        assets=("API surface",),
        boundaries=("anonymous client to authenticated API",),
    ),
    _rule(
        "endpoint_authz",
        "elevation_of_privilege",
        "Access to another owner's data through a new or changed endpoint",
        "A route that authenticates but does not scope its queries to the caller lets one owner "
        "read or "
        "change the data of another.",
        patterns=_ENDPOINT,
        capabilities=("analytics_export",),
        likelihood="medium",
        impact="high",
        needs=(("owner-scoped",),),
        proposal="Scope every query to the authenticated owner and add a cross-owner test.",
        assets=("link data of each owner",),
        boundaries=("authenticated owner to data of another owner",),
    ),
    _rule(
        "migration",
        "tampering",
        "A schema migration corrupts or loses existing data",
        "A migration that rewrites or drops existing columns, or that existing clients cannot "
        "tolerate, "
        "damages data that cannot be recreated.",
        patterns=(
            r"\bmigrat\w*",
            r"\balter table\b",
            r"\bnew column\b",
            r"\bschema change\b",
            r"\bdatabase schema\b",
        ),
        capabilities=("max_clicks",),
        likelihood="low",
        impact="high",
        needs=(("backward-compatible",),),
        proposal="State that existing rows keep working unchanged and test the migration on a "
        "seeded "
        "database.",
        assets=("existing link records",),
        boundaries=("migration script to the production database",),
    ),
    _rule(
        "pii_logging",
        "information_disclosure",
        "Visitor identifiers are stored or logged",
        "IP addresses, user agents and referrers identify visitors; keeping them in logs or "
        "analytics "
        "without a purpose, masking or retention period discloses personal data.",
        patterns=(
            r"\bip address(?:es)?\b",
            r"\buser[- ]agents?\b",
            r"\breferr?ers?\b",
            r"\bpersonal data\b",
            r"\bpii\b",
            r"\bgeolocation\b",
        ),
        likelihood="medium",
        impact="medium",
        proposal="Define what is logged, hash or mask identifiers and set a retention period.",
        assets=("visitor identifiers in logs and analytics",),
        boundaries=("visitor to log and analytics storage",),
    ),
    _rule(
        "retry_amplification",
        "denial_of_service",
        "Retries amplify load on destinations and workers",
        "Unbounded or immediate retries multiply the number of requests sent to a failing "
        "destination "
        "and keep workers busy.",
        patterns=(r"\bretr(?:y|ies|ied|ying)\b", r"\bback-?off\b"),
        likelihood="medium",
        impact="medium",
        needs=(("retry",),),
        proposal="Bound the number of retries and require exponential backoff in an acceptance "
        "criterion.",
        assets=("availability of destination servers and checker workers",),
        boundaries=("checker to external destination servers",),
    ),
    _rule(
        "input_validation",
        "tampering",
        "Out-of-range or malformed client input is accepted",
        "A client-supplied number or window that is not range-checked can break limits, exhaust "
        "resources or corrupt stored data.",
        patterns=(
            r"\bclients? (?:may |can |must )?(?:optionally )?supply\b",
            r"\bselectable\b",
            r"\binteger between\b",
            r"\b(?:request|query) (?:parameter|param|field|string)s?\b",
        ),
        likelihood="medium",
        impact="medium",
        needs=(("range-validation", "window-bounds"),),
        proposal="Add an acceptance criterion that values outside the allowed range are rejected.",
        assets=("validated request parameters",),
        boundaries=("untrusted API client to request validation",),
    ),
)


@dataclass(frozen=True)
class _Criterion:
    req_id: str
    ac_id: str
    aspect: str
    text: str


def _items(value: object) -> list[Mapping[str, Any]]:
    return [v for v in value if isinstance(v, Mapping)] if isinstance(value, list) else []


def _scope_text(requirements: Mapping[str, Any], active: Sequence[Mapping[str, Any]]) -> str:
    problem = requirements.get("problem")
    parts: list[str] = []
    if isinstance(problem, Mapping):
        parts.append(str(problem.get("goal", "")))
        parts.extend(str(s) for s in problem.get("in_scope", ()) if isinstance(s, str))
    for req in active:
        parts.append(str(req.get("text", "")))
        parts.extend(str(c.get("text", "")) for c in _items(req.get("acceptance_criteria")))
    return "\n".join(parts)


def _criteria(active: Sequence[Mapping[str, Any]]) -> list[_Criterion]:
    out: list[_Criterion] = []
    for req in active:
        for crit in _items(req.get("acceptance_criteria")):
            aspect = crit.get("aspect")
            if isinstance(aspect, str):
                out.append(
                    _Criterion(
                        str(req.get("id")), str(crit.get("id")), aspect, str(crit.get("text", ""))
                    )
                )
    return out


def _assess(rule: _Rule, criteria: Sequence[_Criterion]) -> tuple[str, list[dict[str, Any]]]:
    """``(status, mitigations)`` of a fired rule (see the module docstring)."""
    mitigations: list[dict[str, Any]] = []
    missing = False
    for group in rule.needs:
        hits = [c for c in criteria if c.aspect in group]
        missing = missing or not hits
        mitigations.extend(
            {
                "text": f"{c.ac_id}: {c.text}",
                "requirement_id": c.req_id,
                "test_hint": f"Covers: {c.req_id}, {c.ac_id}",
            }
            for c in hits
        )
    if rule.needs and not missing:
        return "mitigated", mitigations
    if rule.accept_reason is not None:
        mitigations.append(
            {"text": f"Accepted: {rule.accept_reason}", "requirement_id": None, "test_hint": None}
        )
        return "accepted", mitigations
    mitigations.append(
        {"text": f"Proposed: {rule.proposal}", "requirement_id": None, "test_hint": None}
    )
    return "open", mitigations


class ThreatModelerAgent(BaseAgent):
    agent_id = "threat-modeler"
    role = "threat-modeler"
    version = "1"
    autonomy = AutonomyProfile(read_globs=(), write_globs=(), can_execute=False, max_tool_calls=8)

    def run(self, ctx: AgentContext) -> AgentResult:
        requirements = ctx.json(Kind.REQUIREMENTS)
        active = [
            r for r in _items(requirements.get("requirements")) if r.get("status") == "active"
        ]
        haystack = _scope_text(requirements, active)
        delivered = {str(c.get("id")) for c in _items(requirements.get("capabilities"))}
        criteria = _criteria(active)
        fired = [
            rule
            for rule in _RULES
            if delivered.intersection(rule.capabilities)
            or any(p.search(haystack) for p in rule.patterns)
        ]
        risks: list[dict[str, Any]] = []
        for n, rule in enumerate(fired, 1):
            status, mitigations = _assess(rule, criteria)
            risks.append(
                {
                    "id": risk_id(n),
                    "category": rule.category,
                    "title": rule.title,
                    "description": rule.description,
                    "likelihood": rule.likelihood,
                    "impact": rule.impact,
                    "mitigations": mitigations,
                    "status": status,
                }
            )
        payload = {
            "assets": list(dict.fromkeys(a for rule in fired for a in rule.assets))
            or list(DEFAULT_ASSETS),
            "trust_boundaries": list(dict.fromkeys(b for rule in fired for b in rule.boundaries))
            or list(DEFAULT_BOUNDARIES),
            "risks": risks,
            "requirement_ids": [str(r.get("id")) for r in active],
        }
        counts = {
            s: sum(1 for r in risks if r["status"] == s) for s in ("mitigated", "accepted", "open")
        }
        return AgentResult(
            artifacts=[json_artifact(Kind.THREAT_MODEL, payload)],
            rationale=(
                f"{len(risks)} risk(s): {counts['mitigated']} mitigated, "
                f"{counts['accepted']} accepted, {counts['open']} open."
            ),
            usage=dict(ctx.toolbox.usage()),
        )
