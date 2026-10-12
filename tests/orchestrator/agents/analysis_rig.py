"""Shared test rig for the analysis agents (intake .. analyst).

* ``make_rig`` builds a real ``AgentContext``: a real ``ScopedToolbox`` over a ``MemoryView`` (the
  real ``PolicyEngine`` evaluates every tool action), an in-memory pinned artifact reader and a fake
  governance view.  The agent's own ``AutonomyProfile`` is used unchanged, so a test also proves
  that the agent stays inside its tool-call budget.
* ``fake_linkcheck`` / ``fake_max_clicks`` / ``fake_export`` are small stand-ins for the real
  recipes, built from the real ``recipes.base`` dataclasses.  ``install_catalog`` swaps the
  process-wide catalog cache for them; ``activate`` can instead select the real recipes and the
  real product tree when they are installed.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents import catalog
from orchestrator.agents.recipes.base import (
    AspectSpec,
    BuildContext,
    Capability,
    GroupSpec,
    ParamSpec,
)
from orchestrator.core.clock import FakeClock
from orchestrator.core.schemas import Kind
from orchestrator.core.serde import canonical_json, hash_jsonable
from orchestrator.core.types import (
    Actor,
    ActorType,
    AgentContext,
    AgentResult,
    Artifact,
    ArtifactStatus,
    AutonomyProfile,
    CancelToken,
    Decision,
    DecisionKind,
    EventType,
    NodeSpec,
    RiskLevel,
    Stage,
)
from orchestrator.policies.engine import PolicyEngine
from orchestrator.tools.toolbox import ScopedToolbox
from orchestrator.tools.workspace import DirectoryView, MemoryView

REPO = Path(__file__).resolve().parents[3]
SCENARIOS = REPO / "orchestrator" / "scenarios"

ProductFiles = Mapping[str, str]

# A tiny brownfield product: enough structure for the codebase analyser (a route, a service, a
# repository, a migration) without depending on the real ``shortener`` package.
PRODUCT: ProductFiles = {
    "shop/__init__.py": "",
    "shop/api.py": (
        "from fastapi import APIRouter, Depends\n"
        "from shop.service import LinkService\n"
        "from shop.security import require_api_key\n\n"
        "router = APIRouter(prefix='/api/v1')\n"
        "service = LinkService()\n\n"
        "@router.post('/links', dependencies=[Depends(require_api_key)])\n"
        "def create_link(body: dict) -> dict:\n"
        "    return service.create(body['url'])\n\n"
        "@router.get('/links/{code}')\n"
        "def link_details(code: str) -> dict:\n"
        "    return service.details(code)\n"
    ),
    "shop/security.py": "def require_api_key() -> None:\n    return None\n",
    "shop/service.py": (
        "from shop.repo import LinkRepository\n\n"
        "class LinkService:\n"
        "    def __init__(self) -> None:\n"
        "        self.repo = LinkRepository()\n\n"
        "    def create(self, url: str) -> dict:\n"
        "        return self.repo.insert(url)\n\n"
        "    def details(self, code: str) -> dict:\n"
        "        return self.repo.get(code)\n"
    ),
    "shop/repo.py": (
        "class LinkRepository:\n"
        "    def insert(self, url: str) -> dict:\n"
        "        return {'sql': 'INSERT INTO links (code, url) VALUES (?, ?)', 'url': url}\n\n"
        "    def get(self, code: str) -> dict:\n"
        "        return {'sql': 'SELECT code, url FROM links WHERE code = ?', 'code': code}\n"
    ),
    "shop/migrations/001_init.sql": "CREATE TABLE links (code TEXT PRIMARY KEY, url TEXT NOT "
    "NULL);\n",
    "tests/test_service.py": "from shop.service import LinkService\n\ndef test_create() -> "
    "None:\n    assert LinkService()\n",
}


# ----------------------------------------------------------------------------- context rig


class Recorder:
    """An ``AuditSink`` that remembers everything it is given."""

    def __init__(self) -> None:
        self.events: list[tuple[EventType, str | None, dict[str, Any]]] = []

    def __call__(
        self,
        type: EventType,
        *,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: Mapping[str, Any],
    ) -> None:
        self.events.append((type, node_id, dict(data)))


class MemoryReader:
    """A pinned, node-scoped ``ArtifactReader`` over in-memory payloads (JSON round-tripped)."""

    def __init__(self, items: Mapping[str, Any], clock: FakeClock) -> None:
        self._arts: dict[str, Artifact] = {}
        self._text: dict[str, str] = {}
        for n, (kind, content) in enumerate(items.items(), 1):
            text = content if isinstance(content, str) else canonical_json(content).decode("utf-8")
            art = Artifact(
                id=f"{kind}@v1",
                kind=kind,
                key="",
                version=1,
                content_hash=hash_jsonable(text),
                media_type="text/plain" if isinstance(content, str) else "application/json",
                size=len(text),
                path=f"{kind}/v1/content",
                created_at=clock.now(),
                producer="external" if kind in ("request", "clarification_answers") else f"node{n}",
                attempt=0,
                agent_id="",
                status=ArtifactStatus.ACTIVE,
            )
            self._arts[kind] = art
            self._text[art.id] = text

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        return self._arts.get(kind) if not key else None

    def active(self, kind: str) -> list[Artifact]:
        return [self._arts[kind]] if kind in self._arts else []

    def all(self) -> tuple[Artifact, ...]:
        return tuple(self._arts.values())

    def read_text(self, artifact: Artifact) -> str:
        return self._text[artifact.id]

    def read_bytes(self, artifact: Artifact) -> bytes:
        return self._text[artifact.id].encode("utf-8")

    def read_json(self, artifact: Artifact) -> Any:
        return json.loads(self._text[artifact.id])


class FakeGovernance:
    """A ``GovernanceView`` that only knows the decisions the test gives it."""

    def __init__(self, decisions: Sequence[Decision] = ()) -> None:
        self._decisions = tuple(decisions)

    def node_statuses(self) -> Mapping[str, Any]:
        return {}

    def decisions(self) -> tuple[Decision, ...]:
        return self._decisions

    def decisions_for(self, ref: Any) -> tuple[Decision, ...]:
        return ()

    def approvals(self) -> tuple[Any, ...]:
        return ()

    def waivers(self) -> tuple[Any, ...]:
        return ()

    def events(self) -> tuple[Any, ...]:
        return ()


class FakeRunView(FakeGovernance):
    """A ``RunView`` over the same in-memory artifacts (what the SDLC gates read through)."""

    def __init__(self, reader: MemoryReader, decisions: Sequence[Decision] = ()) -> None:
        super().__init__(decisions)
        self._reader = reader

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        return self._reader.latest(kind, key)

    def ancestors(self, node_id: str) -> frozenset[str]:
        return frozenset(f"node{n}" for n in range(1, 100))

    def read_json(self, artifact: Artifact) -> Any:
        return self._reader.read_json(artifact)

    def read_text(self, artifact: Artifact) -> str:
        return self._reader.read_text(artifact)

    def params(self) -> Mapping[str, Any]:
        return {}

    def node_specs(self) -> Mapping[str, NodeSpec]:
        return {}

    def mandatory_nodes(self) -> tuple[str, ...]:
        return ()

    def artifacts(self, include_inactive: bool = False) -> tuple[Artifact, ...]:
        return self._reader.all()


@dataclass
class Rig:
    ctx: AgentContext
    audit: Recorder
    cancel: CancelToken


def human_decision(
    n: int, kind: DecisionKind, qid: str, answer: str = "x", actor: Actor | None = None
) -> Decision:
    """A decision as the engine records it for a human answer (``data = {question_id, answer}``)."""
    return Decision(
        id=f"D-{n:04d}",
        run_id="run-1",
        ts=FakeClock().now(),
        kind=kind,
        actor=actor or Actor(ActorType.HUMAN, "carol"),
        node_id=None,
        summary="human answer",
        data={"question_id": qid, "answer": answer},
    )


def make_rig(
    agent: Any,
    *,
    files: ProductFiles | None = None,
    artifacts: Mapping[str, Any] | None = None,
    decisions: Sequence[Decision] = (),
    node_id: str = "node",
    stage: Stage = Stage.REQUIREMENTS,
    node_params: Mapping[str, Any] | None = None,
    profile: AutonomyProfile | None = None,
) -> Rig:
    clock, cancel, audit = FakeClock(), CancelToken(), Recorder()
    box = ScopedToolbox(
        run_id="run-1",
        node_id=node_id,
        attempt=1,
        agent_id=agent.agent_id,
        profile=profile or agent.autonomy,
        view=MemoryView(files or {}),
        cancel=cancel,
        clock=clock,
        policy=PolicyEngine(),
        audit=audit,
    )
    node = NodeSpec(
        id=node_id,
        title=node_id,
        stage=stage,
        agent=agent.agent_id,
        params=dict(node_params or {}),
    )
    ctx = AgentContext(
        run_id="run-1",
        node=node,
        attempt=1,
        attempt_no=1,
        generation=1,
        agent_id=agent.agent_id,
        is_fallback=False,
        params={},
        artifacts=MemoryReader(artifacts or {}, clock),
        toolbox=box,
        clock=clock,
        cancel=cancel,
        run_view=FakeGovernance(decisions),
    )
    return Rig(ctx, audit, cancel)


def run_agent(agent: Any, **kwargs: Any) -> AgentResult:
    return agent.run(make_rig(agent, **kwargs).ctx)


def payload(result: AgentResult, kind: str) -> dict[str, Any]:
    """The JSON content of the (single) artifact of ``kind`` in ``result``."""
    found = [a for a in result.artifacts if a.kind == kind]
    assert len(found) == 1, f"expected one {kind}, got {[a.kind for a in result.artifacts]}"
    content = found[0].content
    assert isinstance(content, Mapping)
    return json.loads(canonical_json(content))


def fingerprint(result: AgentResult) -> str:
    """Hash of everything an agent returns (content only: determinism tests compare two runs)."""
    return hash_jsonable(
        {
            "artifacts": [(a.kind, a.key, a.content) for a in result.artifacts],
            "decisions": [
                (d.kind.value, d.summary, d.rationale, d.outcome, d.parents, d.refs, d.data)
                for d in result.decisions
            ],
            "clarifications": [q.to_dict() for q in result.clarifications],
            "expansion": result.plan_expansion.to_dict() if result.plan_expansion else None,
        }
    )


def intake_for(files: ProductFiles | None, request: str, agent_cls: type) -> dict[str, Any]:
    """The real intake report for ``request`` over ``files``."""
    return payload(
        run_agent(agent_cls(), files=files, artifacts={Kind.REQUEST: request}), Kind.INTAKE_REPORT
    )


# ----------------------------------------------------------------------------- fake capabilities


@dataclass
class FakeRecipe:
    """Just enough of the ``Recipe`` protocol for the analysis agents (they never build files)."""

    id: str
    params: tuple[ParamSpec, ...]
    group_specs: tuple[GroupSpec, ...]
    design_doc: Mapping[str, Any]
    design_calls: list[tuple[dict[str, Any], Any]] = field(default_factory=list)

    def groups(self, params: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
        return self.group_specs

    def design(self, params: Mapping[str, Any], model: Any | None) -> Mapping[str, Any]:
        self.design_calls.append((dict(params), model))
        return copy.deepcopy(dict(self.design_doc))

    def build_group(self, group_id: str, ctx: BuildContext) -> None:
        raise NotImplementedError

    def build_tests(self, ctx: BuildContext) -> None:
        raise NotImplementedError

    def build_docs(self, ctx: BuildContext) -> None:
        raise NotImplementedError


def _adr(title: str, decision: str, option: str, why_not: str) -> dict[str, Any]:
    return {
        "title": title,
        "decision": decision,
        "alternatives": [{"option": option, "rejected_because": why_not}],
        "rationale": f"{decision} keeps the change reviewable.",
        "consequences": "Documented in the design review.",
    }


def fake_linkcheck() -> Capability:
    recipe = FakeRecipe(
        "linkcheck",
        (
            ParamSpec("timeout_seconds", "int", 5, 1, 60),
            ParamSpec("max_retries", "int", 3, 0, 5),
        ),
        (
            GroupSpec(
                "guard", "URL safety guard", ("linkcheck/guard.py",), ("linkcheck/guard.py",)
            ),
            GroupSpec(
                "checker",
                "Checker with retry",
                ("linkcheck/checker.py",),
                ("linkcheck/checker.py",),
                depends_on=("guard",),
            ),
            GroupSpec(
                "report", "Summary report", ("linkcheck/report.py",), ("linkcheck/report.py",)
            ),
        ),
        {
            "components": [
                {
                    "name": "URL guard",
                    "responsibility": "Blocks private destinations and non-http schemes",
                    "touches": ["linkcheck/guard.py"],
                    "requirement_ids": [],
                },
                {
                    "name": "Checker",
                    "responsibility": "Classifies results with retry and backoff",
                    "touches": ["linkcheck/checker.py"],
                    "requirement_ids": [],
                },
            ],
            "adrs": [
                _adr(
                    "Inject the fetcher",
                    "Use an injected Fetcher",
                    "Patch urllib in tests",
                    "brittle",
                ),
                _adr(
                    "Do not follow redirects",
                    "Classify redirects",
                    "Follow redirects",
                    "hides moves",
                ),
            ],
            "data_model_changes": [],
            "api_changes": [],
            "rollback_plan": "Delete the linkcheck package; no data impact.",
        },
    )
    return Capability(
        id="linkcheck",
        title="Destination health checker",
        kind="new_module",
        triggers=(
            r"\bhealth[- ]?check",
            r"\bdestinations?\b",
            r"\bclassif",
            r"\bsummary report\b",
            r"\bunit-testable\b",
        ),
        aspects=(
            AspectSpec(
                "classification",
                "Classification",
                (r"\bclassif", r"\bhealthy\b"),
                "Each destination check returns exactly one classification: healthy, redirecting, "
                "client_error, server_error, timeout or blocked.",
            ),
            AspectSpec(
                "timeout",
                "Timeout",
                (r"\btime ?out after\b",),
                "A single check times out after {timeout_seconds} seconds and is classified as "
                "timeout.",
            ),
            AspectSpec(
                "retry",
                "Retry",
                (r"\bretr(?:y|ied|ies)\b", r"\bbackoff\b"),
                "A timeout or 5xx response is retried up to {max_retries} times with exponential "
                "backoff.",
            ),
            AspectSpec(
                "ssrf-block",
                "SSRF block",
                (r"\bnever contact\b", r"\bprivate-network\b", r"\bloopback\b"),
                "A private, loopback or link-local destination is never contacted and is reported "
                "as blocked.",
            ),
            AspectSpec(
                "scheme-allowlist",
                "Scheme allow-list",
                (r"\bonly http",),
                "A destination with a scheme other than http or https is rejected.",
            ),
            AspectSpec(
                "report",
                "Summary report",
                (r"\bsummary report\b",),
                "The summary report lists the number of links per classification and the "
                "unhealthy links.",
            ),
            AspectSpec(
                "offline-testable",
                "Offline tests",
                (r"\bunit-testable\b", r"\busable as a library\b"),
                "The checker runs in a unit test with an injected fetcher and opens no network "
                "connection.",
            ),
        ),
        extractors={
            "timeout_seconds": r"time ?out after (\d+) seconds?",
            "max_retries": r"up to (\d+) times",
        },
        impact_keywords=(),
        risk=RiskLevel.LOW,
        recipe=recipe,
    )


def fake_max_clicks() -> Capability:
    recipe = FakeRecipe(
        "max_clicks",
        (
            ParamSpec("min_value", "int", 1, 1, 1_000_000),
            ParamSpec("max_value", "int", 1_000_000, 1, 1_000_000),
        ),
        (
            GroupSpec(
                "storage",
                "Migration and repository",
                ("shop/migrations/**", "shop/repo.py"),
                ("shop/migrations/003_add_max_clicks.sql", "shop/repo.py"),
            ),
            GroupSpec("domain-api", "Schemas and routes", ("shop/api.py",), ("shop/api.py",)),
            GroupSpec(
                "service",
                "Resolve logic",
                ("shop/service.py",),
                ("shop/service.py",),
                depends_on=("storage",),
            ),
        ),
        {
            "components": [
                {
                    "name": "Link repository",
                    "responsibility": "Atomic conditional click consumption",
                    "touches": ["shop/repo.py"],
                    "requirement_ids": [],
                },
                {
                    "name": "Link service",
                    "responsibility": "Resolve with limit",
                    "touches": ["shop/service.py"],
                    "requirement_ids": [],
                },
            ],
            "adrs": [
                _adr("Enforce in SQL", "Atomic UPDATE", "In-memory counter", "lost on restart")
            ],
            "data_model_changes": [
                {
                    "table": "links",
                    "change": "add_column",
                    "detail": "max_clicks INTEGER NULL",
                    "migration": "shop/migrations/003_add_max_clicks.sql",
                }
            ],
            "api_changes": [
                {"method": "POST", "path": "/api/v1/links", "change": "modify", "breaking": False}
            ],
            "api_paths": [
                {
                    "method": "POST",
                    "path": "/api/v1/links",
                    "summary": "Create a link with an optional click limit",
                    "request_fields": [
                        {"name": "max_clicks", "required": False, "type": "integer"}
                    ],
                    "responses": [201, 422],
                }
            ],
            "rollback_plan": "Forward-fix; SQLite cannot drop the column safely.",
        },
    )
    return Capability(
        id="max_clicks",
        title="Click limit for short links",
        kind="enhancement",
        triggers=(
            r"\bmax_clicks\b",
            r"\bclicks?\b",
            r"\b410\b",
            r"\bshort (?:link|code)",
        ),
        aspects=(
            AspectSpec(
                "optional-field",
                "Optional field",
                (r"\boptionally supply\b",),
                "A link created without max_clicks has no limit and keeps working.",
            ),
            AspectSpec(
                "range-validation",
                "Range validation",
                (r"\binteger between\b",),
                "A max_clicks value outside {min_value}..{max_value} is rejected with status 422.",
            ),
            AspectSpec(
                "gone-after-limit",
                "Gone after the limit",
                (r"\b410\b",),
                "After max_clicks follows, the short code returns 410 Gone.",
            ),
            AspectSpec(
                "backward-compatible",
                "Backward compatible",
                (r"\bnot be affected\b", r"\bexactly as today\b"),
                "Creating a link without max_clicks still returns status 201 and the OpenAPI "
                "diff reports no breaking change.",
            ),
            AspectSpec(
                "remaining-visible",
                "Remaining visible",
                (r"\bremaining click\b",),
                "The link details response contains remaining_clicks.",
            ),
            AspectSpec(
                "concurrency-safe",
                "Concurrency safe",
                (r"\bconcurrent\b",),
                "With max_clicks of 1, two parallel requests produce exactly one redirect.",
            ),
        ),
        extractors={
            "min_value": r"between (\d[\d,_]*) and",
            "max_value": r"between \d[\d,_]* and (\d[\d,_]*)",
        },
        impact_keywords=("link", "click", "redirect"),
        risk=RiskLevel.HIGH,
        recipe=recipe,
        requires_baseline=True,
    )


def fake_export() -> Capability:
    recipe = FakeRecipe(
        "analytics_export",
        (
            ParamSpec("max_window_days", "int", 90, 1, 730),
            ParamSpec("perf_budget_ms", "int", 250, 50, 5000),
        ),
        (
            GroupSpec("service", "Export service", ("shop/service.py",), ("shop/service.py",)),
            GroupSpec(
                "csv-encoding", "CSV encoding", ("shop/csv_export.py",), ("shop/csv_export.py",)
            ),
            GroupSpec(
                "route",
                "Export route",
                ("shop/api.py",),
                ("shop/api.py",),
                depends_on=("service", "csv-encoding"),
            ),
        ),
        {
            "components": [
                {
                    "name": "Export service",
                    "responsibility": "Aggregates daily clicks",
                    "touches": ["shop/service.py"],
                    "requirement_ids": [],
                },
                {
                    "name": "CSV encoder",
                    "responsibility": "Neutralises formula cells",
                    "touches": ["shop/csv_export.py"],
                    "requirement_ids": [],
                },
            ],
            "adrs": [
                _adr("Buffered CSV", "Build the CSV in memory", "Stream rows", "not required")
            ],
            "data_model_changes": [],
            "api_changes": [
                {
                    "method": "GET",
                    "path": "/api/v1/links/{code}/analytics/export.csv",
                    "change": "add",
                    "breaking": False,
                }
            ],
            "api_paths": [
                {
                    "method": "GET",
                    "path": "/api/v1/links/{code}/analytics/export.csv",
                    "summary": "Export daily click counts as CSV",
                    "request_fields": [{"name": "days", "required": False, "type": "integer"}],
                    "responses": [200, 401, 404, 422],
                }
            ],
            "rollback_plan": "Remove the route and the csv module; no data change.",
        },
    )
    return Capability(
        id="analytics_export",
        title="Analytics CSV export",
        kind="enhancement",
        triggers=(
            r"\bcsv\b",
            r"\bexport",
            r"\bdaily click",
            r"\banalytics\b",
            r"\bwindow\b",
        ),
        aspects=(
            AspectSpec(
                "csv-export",
                "CSV export",
                (r"\bcsv\b", r"\bexport"),
                "The export returns a CSV document with one row per day and link and the click "
                "count.",
            ),
            AspectSpec(
                "window-bounds",
                "Window bounds",
                (r"\bwindow\b",),
                "A window of more than {max_window_days} days is rejected with status 422.",
            ),
            AspectSpec(
                "performance",
                "Performance",
                (r"\bp95\b", r"\bwithin \d+ ?ms\b"),
                "The export responds within {perf_budget_ms} ms at p95 for a "
                "{max_window_days}-day window.",
            ),
            AspectSpec(
                "formula-injection",
                "Formula injection",
                (r"\bformula\b",),
                "A cell starting with =, +, - or @ is prefixed with a single quote.",
            ),
        ),
        extractors={
            "max_window_days": r"up to (\d[\d,]*) days?",
            "perf_budget_ms": r"within (\d[\d,]*) ?ms",
        },
        impact_keywords=("analytics", "click", "export"),
        risk=RiskLevel.MEDIUM,
        recipe=recipe,
        requires_baseline=True,
    )


def install_catalog(monkeypatch: pytest.MonkeyPatch, *caps: Capability) -> None:
    """Replace the process-wide capability registry (restored by ``monkeypatch``)."""
    monkeypatch.setattr(catalog, "_cache", {c.id: c for c in caps})


def real_recipes_installed() -> bool:
    import importlib.util

    return all(
        importlib.util.find_spec(f"orchestrator.agents.recipes.{name}") is not None
        for name in ("linkcheck", "max_clicks", "analytics_export")
    )


def real_product_files() -> dict[str, str]:
    """The baseline of the brownfield scenarios: the real product tree of this repository."""
    view = DirectoryView(REPO, include=("shortener/**", "tests/shortener/**", "pyproject.toml"))
    return {path: view.read_text(path) for path in view.list_files("**")}


def scenario_request(name: str, filename: str = "request.md") -> str:
    return (SCENARIOS / name / filename).read_text(encoding="utf-8")


# The scripted human answers of the ambiguous scenario (its ``scenario.json``):
# first matching rule wins, applied to the question text.
AMBIGUOUS_SCRIPT: tuple[tuple[str, str], ...] = (
    (
        "report",
        "CSV export of daily click counts per link for a selectable window of up to 90 days",
    ),
    (
        "fast|latenc|perform|speed",
        "The export must respond within 250 ms (p95) for 90 days of data",
    ),
    ("safe|secur", "Out of scope for this iteration; tracked separately"),
    (".*", "@default"),
)


def scripted_answers(
    questions: Iterable[Any], script: Sequence[tuple[str, str]] = AMBIGUOUS_SCRIPT
) -> dict[str, Any]:
    """``clarification_answers`` payload answering ``questions`` like the scenario script."""
    import re

    answers: dict[str, dict[str, Any]] = {}
    for q in questions:
        for pattern, answer in script:
            if re.search(pattern, q.text, re.IGNORECASE):
                answers[q.id] = {"answer": answer, "by": "carol", "round": 1}
                break
    return {"schema": 1, "answers": answers}


# ----------------------------------------------------------------------------- payload builders


def req_entry(
    n: int,
    text: str,
    *,
    aspects: Sequence[str] = (),
    status: str = "active",
    capabilities: Sequence[str] = (),
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A schema-valid ``requirements[]`` entry with one criterion per aspect."""
    return {
        "id": f"R-{n:03d}",
        "text": text,
        "kind": "functional",
        "priority": "must",
        "status": status,
        "status_decision": f"SCOPE:R-{n:03d}" if status != "active" else None,
        "capabilities": list(capabilities),
        "parameters": dict(parameters or {}),
        "verification": "test",
        "acceptance_criteria": [
            {
                "id": f"AC-{n:03d}-{k}",
                "text": f"The {a} behaviour returns the documented result.",
                "aspect": a,
            }
            for k, a in enumerate(aspects, 1)
        ],
    }


def requirements_payload(
    entries: Sequence[Mapping[str, Any]],
    *,
    capabilities: Sequence[Mapping[str, Any]] = (),
    unsupported: Sequence[Mapping[str, Any]] = (),
    goal: str = "A goal",
    in_scope: Sequence[str] = (),
) -> dict[str, Any]:
    body = {
        "schema": 1,
        "problem": {
            "goal": goal,
            "in_scope": list(in_scope),
            "out_of_scope": [],
            "constraints": [],
        },
        "classification": {"workspace_kind": "greenfield", "change_type": "feature"},
        "requirements": [dict(e) for e in entries],
        "assumptions": [],
        "ambiguities": [],
        "open_questions": [],
        "unsupported": [dict(u) for u in unsupported],
        "capabilities": [dict(c) for c in capabilities],
    }
    from orchestrator.core.schemas import validate_payload

    assert validate_payload(Kind.REQUIREMENTS, body) == []
    return body


def threat_payload(risks: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    return {
        "schema": 1,
        "assets": [],
        "trust_boundaries": [],
        "risks": [dict(r) for r in risks],
        "requirement_ids": [],
    }


def capability_entry(
    cap: Capability, requirement_ids: Sequence[str], **params: Any
) -> dict[str, Any]:
    defaults = {p.name: p.default for p in cap.recipe.params}
    return {
        "id": cap.id,
        "title": cap.title,
        "requirement_ids": list(requirement_ids),
        "params": {**defaults, **params},
    }


# ----------------------------------------------------------------------------- pipeline


@dataclass
class Chain:
    """Payloads (by kind) and results (by agent id) of an intake -> ... -> architect run."""

    artifacts: dict[str, Any]
    results: dict[str, AgentResult]

    def node_inputs(self, *kinds: str) -> dict[str, Any]:
        return {k: self.artifacts[k] for k in kinds if k in self.artifacts}


def run_chain(
    request: str,
    files: ProductFiles | None = None,
    *,
    answers: Mapping[str, Any] | None = None,
    decisions: Sequence[Decision] = (),
    through: str = "threat-modeler",
) -> Chain:
    """intake -> requirements [-> analyst when brownfield] -> threat-modeler [-> architect]."""
    from orchestrator.agents.analyst import AnalystAgent
    from orchestrator.agents.architect import ArchitectAgent
    from orchestrator.agents.intake import IntakeAgent
    from orchestrator.agents.requirements import RequirementsAgent
    from orchestrator.agents.threat_model import ThreatModelerAgent

    arts: dict[str, Any] = {Kind.REQUEST: request}
    results: dict[str, AgentResult] = {}

    def step(agent: Any, kind: str, *, extra: str | None = None, **kw: Any) -> None:
        result = run_agent(agent, files=files, artifacts=dict(arts), **kw)
        results[agent.agent_id] = result
        arts[kind] = payload(result, kind)
        if extra is not None:
            arts[extra] = payload(result, extra)

    step(IntakeAgent(), Kind.INTAKE_REPORT)
    if answers is not None:
        arts[Kind.CLARIFICATION_ANSWERS] = dict(answers)
    step(RequirementsAgent(), Kind.REQUIREMENTS, decisions=decisions)
    if files:
        step(AnalystAgent(), Kind.IMPACT_REPORT, node_id="codebase-analysis", stage=Stage.ANALYSIS)
    step(ThreatModelerAgent(), Kind.THREAT_MODEL, node_id="threat-model", stage=Stage.SECURITY)
    if through == "architect":
        step(
            ArchitectAgent(),
            Kind.DESIGN_DOC,
            extra=Kind.API_CONTRACT,
            node_id="design",
            stage=Stage.DESIGN,
        )
    return Chain(arts, results)


def ambiguous_inputs(files: ProductFiles | None) -> tuple[Mapping[str, Any], tuple[Decision, ...]]:
    """The scripted answers and the human decisions the engine would record for them."""
    from orchestrator.agents.intake import IntakeAgent
    from orchestrator.agents.requirements import RequirementsAgent

    text = scenario_request("ambiguous")
    arts = {Kind.REQUEST: text, Kind.INTAKE_REPORT: intake_for(files, text, IntakeAgent)}
    first = run_agent(RequirementsAgent(), files=files, artifacts=arts)
    answers = scripted_answers(first.clarifications)
    decisions = tuple(
        human_decision(n, DecisionKind.CLARIFICATION, qid, entry["answer"])
        for n, (qid, entry) in enumerate(answers["answers"].items(), 1)
    )
    return answers, decisions


def activate(mode: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Install the fake or the real catalog and return the matching brownfield product files."""
    if mode == "fake":
        install_catalog(monkeypatch, fake_linkcheck(), fake_max_clicks(), fake_export())
        return dict(PRODUCT)
    monkeypatch.setattr(catalog, "_cache", None)  # force the real lazy import
    return real_product_files()


def real_mode_available() -> bool:
    return real_recipes_installed() and (REPO / "shortener").is_dir()


MODES = (
    "fake",
    pytest.param(
        "real",
        marks=pytest.mark.skipif(
            not real_mode_available(), reason="real recipes or product missing"
        ),
    ),
)
