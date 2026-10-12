"""Shared rig for the implementer / test-author / integrator / chaos agent tests.

``Rig`` builds an ``AgentContext`` the way the engine does:

* a real ``ArtifactStore`` and a ``PinnedReader`` restricted to the kinds the node declares in
  docs/sdlc-contract.md (``NODES``): reading an undeclared kind raises ``UndeclaredInputError``;
* a real ``ScopedToolbox`` over the baseline ``MemoryView``, the real ``PolicyEngine`` (tests may
  import policies), a ``FakeClock`` and a recording audit sink;
* the narrowed autonomy profile ``agent.autonomy.narrow(node.autonomy)``.

``FakeRecipe`` / ``fake_capability`` are deterministic stand-ins for the real recipes (which are
written by other agents): three groups, one typed ``limit`` parameter and switches for every
behaviour the agents have to cope with (drift, deletions, out-of-scope writes, ...).  Payload
builders return schema-valid artifact payloads; ``Rig.publish`` validates every payload so a broken
fixture fails loudly.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.implementer import ImplementerAgent
from orchestrator.agents.integrator import IntegratorAgent
from orchestrator.agents.recipes.base import (
    AspectSpec,
    BuildContext,
    Capability,
    GroupSpec,
    ParamSpec,
    RecipeAnchorError,
)
from orchestrator.agents.test_author import TestAuthorAgent
from orchestrator.core.artifacts import ArtifactStore
from orchestrator.core.clock import FakeClock
from orchestrator.core.schemas import Kind, validate_payload
from orchestrator.core.types import (
    Actor,
    Agent,
    AgentContext,
    AgentResult,
    Artifact,
    ArtifactDraft,
    AutonomyProfile,
    CancelToken,
    ChangeSet,
    ChangeSetSubject,
    EventType,
    Finding,
    NodeSpec,
    PolicyCategory,
    PolicyReport,
    RiskLevel,
    Severity,
    Stage,
)
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.engine import PolicyEngine
from orchestrator.tools.changeset import compute_ops
from orchestrator.tools.toolbox import ScopedToolbox
from orchestrator.tools.workspace import DirectoryView, MemoryView
from tests.orchestrator.policies.policy_builders import FakeRunView

RUN_ID = "run-1"
GROUPS = ("alpha", "beta", "gamma")


# ----------------------------------------------------------------------------- audit


class AuditRecorder:
    """An ``AuditSink`` that remembers everything it is given."""

    def __init__(self) -> None:
        self.events: list[tuple[EventType, dict[str, Any]]] = []

    def __call__(
        self,
        type: EventType,
        *,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: Mapping[str, Any],
    ) -> None:
        self.events.append((type, dict(data)))

    def of(self, kind: EventType) -> list[dict[str, Any]]:
        return [data for etype, data in self.events if etype == kind]


# ----------------------------------------------------------------------------- fake recipe


class FakeRecipe:
    """Deterministic recipe: groups ``alpha`` / ``beta`` / ``gamma`` writing ``pkg/<group>/``.

    Switches (all off by default): ``honours_conservative`` (skip the optional ``fast_path.py``
    when ``anchors['conservative']``), ``anchor=(path, marker)`` (anchored MODIFY of an existing
    file, ``RecipeAnchorError`` on drift), ``delete`` / ``outside`` (a DELETE / a write outside
    ``pkg/<group>/``), ``extra_files`` (N more files per group), ``writes_nothing``,
    ``mutate_requirements`` (``build_tests`` vandalises its requirements view).
    """

    params: tuple[ParamSpec, ...] = (ParamSpec("limit", "int", default=3, minimum=1, maximum=10),)

    def __init__(
        self,
        recipe_id: str = "fake",
        *,
        honours_conservative: bool = True,
        anchor: tuple[str, str] | None = None,
        delete: str | None = None,
        outside: str | None = None,
        extra_files: int = 0,
        writes_nothing: bool = False,
        mutate_requirements: bool = False,
    ) -> None:
        self.id = recipe_id
        self.honours_conservative = honours_conservative
        self.anchor = anchor
        self.delete = delete
        self.outside = outside
        self.extra_files = extra_files
        self.writes_nothing = writes_nothing
        self.mutate_requirements = mutate_requirements
        self.group_calls: list[tuple[str, BuildContext, dict[str, Any]]] = []
        self.test_calls: list[tuple[BuildContext, dict[str, Any]]] = []

    def groups(self, params: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
        return tuple(
            GroupSpec(g, f"{g} group", (f"pkg/{g}/**",), (f"pkg/{g}/core.py",)) for g in GROUPS
        )

    def design(self, params: Mapping[str, Any], model: Any | None) -> Mapping[str, Any]:
        return {}

    def build_group(self, group_id: str, ctx: BuildContext) -> None:
        self.group_calls.append((group_id, ctx, dict(ctx.params)))
        if self.writes_nothing:
            return
        box = ctx.toolbox
        box.write_text(f"pkg/{group_id}/__init__.py", "")
        source = f"LIMIT = {ctx.params['limit']!r}\n\n\ndef limit() -> int:\n    return LIMIT\n"
        box.write_text(f"pkg/{group_id}/core.py", source)
        if not (self.honours_conservative and ctx.anchors.get("conservative")):
            box.write_text(f"pkg/{group_id}/fast_path.py", "FAST = True\n")
        for index in range(self.extra_files):
            box.write_text(f"pkg/{group_id}/extra_{index}.py", f"VALUE = {index}\n")
        if self.anchor is not None:
            path, marker = self.anchor
            try:
                text = box.read_text(path)
            except FileNotFoundError as exc:
                raise RecipeAnchorError(f"{path} is missing") from exc
            if marker not in text:
                raise RecipeAnchorError(f"anchor {marker!r} not found in {path}")
            box.write_text(path, text.replace(marker, marker + "  # patched", 1))
        if self.delete is not None:
            box.delete(self.delete)
        if self.outside is not None:
            box.write_text(self.outside, "STRAY = 1\n")

    def build_tests(self, ctx: BuildContext) -> None:
        assert ctx.requirements is not None
        self.test_calls.append((ctx, dict(ctx.params)))
        lines = [f'"""Generated tests of capability {self.id}."""', ""]
        for req in ctx.requirements["requirements"]:
            for criterion in req["acceptance_criteria"]:
                name = criterion["id"].lower().replace("-", "_")
                lines += [
                    "",
                    f"def test_{name}() -> None:",
                    f"    '''Covers: {req['id']}, {criterion['id']}'''",
                    "    assert True",
                ]
        if self.mutate_requirements:
            ctx.requirements["requirements"].clear()
        if not self.writes_nothing:
            ctx.toolbox.write_text(f"tests/{self.id}/test_{self.id}.py", "\n".join(lines) + "\n")

    def build_docs(self, ctx: BuildContext) -> None:
        ctx.toolbox.write_text(f"docs/features/{self.id}.md", f"# {self.id}\n")


def fake_capability(
    recipe: FakeRecipe | None = None, *, requires_baseline: bool = False
) -> Capability:
    recipe = recipe or FakeRecipe()
    return Capability(
        id=recipe.id,
        title=f"Fake capability {recipe.id}",
        kind="enhancement" if requires_baseline else "new_module",
        triggers=(recipe.id,),
        aspects=(AspectSpec("alpha", "Alpha", (recipe.id,), "The alpha aspect holds"),),
        extractors={},
        impact_keywords=(recipe.id,),
        risk=RiskLevel.LOW,
        recipe=recipe,
        requires_baseline=requires_baseline,
    )


def lookup(*capabilities: Capability) -> Callable[[str], Capability]:
    """A capability lookup over exactly ``capabilities`` (``KeyError`` for anything else)."""
    table = {c.id: c for c in capabilities}

    def find(capability_id: str) -> Capability:
        return table[capability_id]

    return find


# ----------------------------------------------------------------------------- payload builders


def requirement(
    index: int,
    *,
    status: str = "active",
    capabilities: Sequence[str] = ("fake",),
    criteria: Sequence[tuple[str, str]] = (("alpha", "The alpha aspect holds"),),
) -> dict[str, Any]:
    rid = f"R-{index:03d}"
    return {
        "id": rid,
        "text": f"Requirement {index}",
        "kind": "functional",
        "priority": "must",
        "status": status,
        "status_decision": None if status == "active" else f"SCOPE:{rid}",
        "capabilities": list(capabilities),
        "parameters": {},
        "verification": "test",
        "acceptance_criteria": [
            {"id": f"AC-{index:03d}-{j}", "text": text, "aspect": aspect}
            for j, (aspect, text) in enumerate(criteria, 1)
        ],
    }


def requirements_payload(
    requirements: Sequence[Mapping[str, Any]],
    *,
    capability_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    ids: dict[str, list[str]] = {}
    for req in requirements:
        for cap in req["capabilities"]:
            ids.setdefault(cap, []).append(req["id"])
    params = capability_params or {}
    return {
        "schema": 1,
        "problem": {"goal": "goal", "in_scope": [], "out_of_scope": [], "constraints": []},
        "classification": {"workspace_kind": "greenfield", "change_type": "feature"},
        "requirements": [dict(r) for r in requirements],
        "assumptions": [],
        "ambiguities": [],
        "open_questions": [],
        "unsupported": [],
        "capabilities": [
            {"id": cap, "title": cap, "requirement_ids": rids, "params": dict(params.get(cap, {}))}
            for cap, rids in ids.items()
        ],
    }


def plan_group(
    group: str,
    *,
    capability: str = "fake",
    params: Mapping[str, Any] | None = None,
    node_id: str | None = None,
    task_ids: Sequence[str] = ("T-001",),
    scope: Sequence[str] | None = None,
) -> dict[str, Any]:
    """A ``task_plan`` group the way the planner writes it: ``<capability>-<recipe group>``."""
    plan_id = f"{capability}-{group}"
    return {
        "id": plan_id,
        "node_id": node_id or f"implement.{plan_id}",
        "task_ids": list(task_ids),
        "capability": capability,
        "params": dict(params or {}),
        "write_scope": list(scope if scope is not None else [f"pkg/{group}/**"]),
    }


def plan_task(
    task_id: str,
    group: str,
    *,
    capability: str = "fake",
    requirement_ids: Sequence[str] = ("R-001",),
) -> dict[str, Any]:
    """A ``task_plan`` task; ``params.recipe_group`` is the recipe group, like the planner's."""
    return {
        "id": task_id,
        "title": f"Task {task_id}",
        "group": f"{capability}-{group}",
        "requirement_ids": list(requirement_ids),
        "depends_on": [],
        "capability": capability,
        "kind": "implement",
        "write_scope": [f"pkg/{group}/**"],
        "expected_paths": [f"pkg/{group}/core.py"],
        "params": {"recipe_group": group, "recipe_params": {}},
    }


def task_plan_payload(
    groups: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    return {
        "schema": 1,
        "tasks": [dict(t) for t in tasks],
        "groups": [dict(g) for g in groups],
        "critical_path": [],
        "unsupported": [],
    }


def design_doc_payload() -> dict[str, Any]:
    return {
        "schema": 1,
        "summary": "design summary",
        "components": [],
        "adrs": [],
        "data_model_changes": [],
        "api_changes": [],
        "rollback_plan": "revert the change",
        "change_reference": "CHG-1",
        "risk": {"level": "LOW", "reasons": []},
    }


def impact_report_payload(expected_paths: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": 1,
        "keywords": [],
        "modules": [],
        "routes": [],
        "tables": [],
        "data_flows": [],
        "tests": [],
        "coupling_risks": [],
        "risk_signals": [],
        "expected_paths": list(expected_paths),
    }


# --------------------------------------------------------------------------- nodes (sdlc)

NODES: Mapping[str, dict[str, Any]] = {
    "implement": {
        "stage": Stage.IMPLEMENTATION,
        "consumes": (Kind.TASK_PLAN, Kind.DESIGN_DOC),
        "optional": (Kind.IMPACT_REPORT,),
        "produces": (Kind.CHANGESET,),
    },
    "author-tests": {
        "stage": Stage.TESTING,
        "consumes": (Kind.REQUIREMENTS, Kind.TASK_PLAN, Kind.DESIGN_DOC),
        "optional": (),
        "produces": (Kind.CHANGESET,),
    },
    "integrate": {
        "stage": Stage.IMPLEMENTATION,
        "consumes": (Kind.CHANGESET, Kind.TASK_PLAN),
        "optional": (Kind.IMPACT_REPORT,),
        "produces": (Kind.INTEGRATED_CHANGESET, Kind.INTEGRATION_REPORT),
    },
}


# ----------------------------------------------------------------------------- the rig


@dataclass
class Rig:
    """Everything an agent needs, wired like the engine does (see the module docstring)."""

    root: Path
    baseline_files: Mapping[str, str] = field(default_factory=dict)
    clock_start: datetime | None = None

    def __post_init__(self) -> None:
        self.clock = FakeClock(self.clock_start)
        self.audit = AuditRecorder()
        self.policy = PolicyEngine()
        self.store = ArtifactStore(self.root / "store", clock=self.clock)
        self.baseline = MemoryView(self.baseline_files)

    # -- artifacts ------------------------------------------------------------------------
    def publish(
        self,
        kind: str,
        payload: Any,
        *,
        key: str = "",
        producer: str = "upstream",
        validate: bool = True,
    ) -> Artifact:
        """Store + activate an artifact (dict -> JSON, validated; ``ChangeSet`` -> its dict)."""
        content = payload.to_dict() if isinstance(payload, ChangeSet) else payload
        if validate and isinstance(content, Mapping):
            problems = validate_payload(kind, content)
            assert not problems, f"invalid {kind} fixture: {problems}"
        return self.store.put(ArtifactDraft(kind=kind, content=content, key=key), producer=producer)

    def publish_changeset(
        self,
        node_id: str,
        desired: Mapping[str, str | None],
        *,
        base: Mapping[str, str] | None = None,
        author: str = "implementer",
    ) -> ChangeSet:
        """Publish ``compute_ops(base or baseline, desired)`` as the change set of ``node_id``."""
        cs = compute_ops(
            MemoryView(self.baseline_files if base is None else base),
            desired,
            description=f"changes of {node_id}",
            author=author,
        )
        self.publish(Kind.CHANGESET, cs, key=node_id, producer=node_id)
        return cs

    # -- contexts -------------------------------------------------------------------------
    def node(self, flavour: str, node_id: str, **overrides: Any) -> NodeSpec:
        shape = NODES[flavour]
        fields: dict[str, Any] = {
            "id": node_id,
            "title": node_id,
            "stage": shape["stage"],
            "agent": "agent-under-test",
            "consumes": shape["consumes"],
            "optional_consumes": shape["optional"],
            "produces": shape["produces"],
        }
        fields.update(overrides)
        return NodeSpec(**fields)

    def context(
        self,
        agent: Agent,
        flavour: str,
        node_id: str,
        *,
        params: Mapping[str, Any] | None = None,
        node_autonomy: AutonomyProfile | None = None,
        feedback: Sequence[Finding] = (),
        is_fallback: bool = False,
        attempt_no: int = 1,
        cancel: CancelToken | None = None,
        undeclared: Sequence[str] = (),
    ) -> AgentContext:
        spec = self.node(flavour, node_id, params=dict(params or {}), autonomy=node_autonomy)
        cancel = cancel or CancelToken()
        profile: AutonomyProfile = agent.autonomy.narrow(spec.autonomy)
        toolbox = ScopedToolbox(
            run_id=RUN_ID,
            node_id=node_id,
            attempt=attempt_no,
            agent_id=agent.agent_id,
            profile=profile,
            view=self.baseline,
            cancel=cancel,
            clock=self.clock,
            policy=self.policy,
            audit=self.audit,
            baseline=self.baseline,
        )
        kinds = [k for k in spec.input_kinds if k not in undeclared]
        return AgentContext(
            run_id=RUN_ID,
            node=spec,
            attempt=attempt_no,
            attempt_no=attempt_no,
            generation=1,
            agent_id=agent.agent_id,
            is_fallback=is_fallback,
            params=spec.params,
            artifacts=self.store.snapshot(kinds, lambda _producer: True),
            toolbox=toolbox,
            clock=self.clock,
            cancel=cancel,
            run_view=FakeRunView(),
            feedback=tuple(feedback),
        )

    def run(self, agent: Agent, flavour: str, node_id: str, **kwargs: Any) -> AgentResult:
        return agent.run(self.context(agent, flavour, node_id, **kwargs))


# ----------------------------------------------------------------------------- results / policy


def drafts(result: AgentResult, kind: str) -> list[ArtifactDraft]:
    return [d for d in result.artifacts if d.kind == kind]


def payload_of(result: AgentResult, kind: str) -> dict[str, Any]:
    """The decoded content of the single JSON artifact draft of ``kind``."""
    found = drafts(result, kind)
    assert len(found) == 1, f"expected exactly one {kind} draft, got {len(found)}"
    content = found[0].content
    assert isinstance(content, dict), f"{kind} draft is not a JSON object"
    return content


def changeset_of(result: AgentResult, kind: str = Kind.CHANGESET) -> ChangeSet:
    return ChangeSet.from_dict(payload_of(result, kind))


def content_of(cs: ChangeSet) -> dict[str, str | None]:
    """``{path: content}`` of the ops (``None`` for deletions)."""
    return {op.path: op.content for op in cs.ops}


@functools.lru_cache(maxsize=1)
def engine() -> PolicyEngine:
    return PolicyEngine()


def evaluate(
    cs: ChangeSet,
    baseline: MemoryView,
    *,
    node_id: str = "implement.fake-alpha",
    profile: AutonomyProfile | None = None,
    impact: Mapping[str, Any] | None = None,
    sources: Sequence[ChangeSet] = (),
    config: PolicyConfig | None = None,
) -> PolicyReport:
    """The real policy engine over ``cs`` (author, profile and sources as the engine supplies)."""
    judge = engine() if config is None else PolicyEngine(config=config)
    return judge.evaluate(
        ChangeSetSubject(
            cs,
            baseline,
            node_id=node_id,
            author=cs.author or "agent",
            impact=impact,
            profile=profile or AutonomyProfile(write_globs=("**",)),
            sources=tuple(sources),
        )
    )


def rule_ids(report: PolicyReport) -> set[str]:
    return {f.rule_id for f in report.findings}


def feedback_finding(rule_id: str, severity: Severity = Severity.HIGH) -> Finding:
    """Engine-style feedback: rule id, severity and location only."""
    return Finding(rule_id, PolicyCategory.SECURITY, severity, "feedback", location="x.py:1")


# ----------------------------------------------------------------------------- real recipes

REPO_ROOT = Path(__file__).resolve().parents[3]
# The default fetcher of the linkcheck recipe is network code: a scenario allow-lists it (SEC-008).
LINKCHECK_POLICY = PolicyConfig(egress_modules=("linkcheck/fetcher.py",))


def real_capability(name: str) -> Capability:
    """``CAPABILITY`` of the real recipe module ``name`` (skip when it is not importable)."""
    try:
        module = importlib.import_module(f"orchestrator.agents.recipes.{name}")
    except ImportError:
        pytest.skip(f"recipe {name} is not available")
    capability: Capability = module.CAPABILITY
    return capability


def default_params(capability: Capability) -> dict[str, Any]:
    return {p.name: p.default for p in capability.recipe.params}


def product_files() -> dict[str, str]:
    """The real product tree (``shortener/``, its tests, ``pyproject.toml``) as a baseline."""
    if not (REPO_ROOT / "shortener").is_dir():
        pytest.skip("the product tree is not available")
    view = DirectoryView(
        REPO_ROOT, include=("shortener/**", "tests/shortener/**", "pyproject.toml")
    )
    return {path: view.read_text(path) for path in view.list_files("**")}


def group_node(capability: Capability, group: GroupSpec) -> str:
    return f"implement.{capability.id}-{group.id}"


def group_assignment(
    capability: Capability, group: GroupSpec, params: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "group": group.id,
        "capability": capability.id,
        "recipe_params": dict(params),
        "write_scope": list(group.write_scope),
    }


def run_group(rig: Rig, agent: Agent, capability: Capability, group: GroupSpec) -> AgentResult:
    """One ``implement.*`` node: the group's own write scope narrows the agent's autonomy."""
    params = default_params(capability)
    return rig.run(
        agent,
        "implement",
        group_node(capability, group),
        params=group_assignment(capability, group, params),
        node_autonomy=AutonomyProfile(write_globs=group.write_scope),
    )


@dataclass
class Chain:
    """The delivery chain of one capability: implementers, test author, integrator."""

    rig: Rig
    capability: Capability
    sources: dict[str, ChangeSet]
    integration: AgentResult

    @property
    def merged(self) -> ChangeSet:
        return changeset_of(self.integration, Kind.INTEGRATED_CHANGESET)

    @property
    def report(self) -> dict[str, Any]:
        return payload_of(self.integration, Kind.INTEGRATION_REPORT)


def build_chain(
    root: Path,
    capability: Capability,
    baseline: Mapping[str, str],
    *,
    wrap: Callable[[GroupSpec, Agent], Agent] | None = None,
) -> Chain:
    """Run every group's implementer, the test author and the integrator on ``capability``.

    ``wrap(group, implementer)`` may substitute another agent (for example a chaos wrapper).
    """
    rig = Rig(root, baseline)
    params = default_params(capability)
    sources: dict[str, ChangeSet] = {}
    for group in capability.recipe.groups(params):
        agent: Agent = ImplementerAgent(lookup(capability))
        if wrap is not None:
            agent = wrap(group, agent)
        node = group_node(capability, group)
        cs = changeset_of(run_group(rig, agent, capability, group))
        rig.publish(Kind.CHANGESET, cs, key=node, producer=node)
        sources[node] = cs
    criteria = [(a.id, a.criterion.format_map(params)) for a in capability.aspects]
    rig.publish(
        Kind.REQUIREMENTS,
        requirements_payload([requirement(1, capabilities=(capability.id,), criteria=criteria)]),
    )
    tests = changeset_of(
        rig.run(TestAuthorAgent(lookup(capability)), "author-tests", "author-tests")
    )
    rig.publish(Kind.CHANGESET, tests, key="author-tests", producer="author-tests")
    sources["author-tests"] = tests
    integration = rig.run(IntegratorAgent(), "integrate", "integrate")
    return Chain(rig, capability, sources, integration)
