"""Shared test support for the delivery agents (verifier ... release-verifier).

``Rig`` builds an ``AgentContext`` the way the engine does:

* a real ``ArtifactStore`` and a ``PinnedReader`` restricted to the kinds the node declares in
  docs/sdlc-contract.md (``NODES``): reading an undeclared kind raises ``UndeclaredInputError``;
* a real ``ScopedToolbox`` (``make_toolbox_factory``) over ``baseline + overlay_from change sets``
  (``MemoryView`` / ``OverlayView``), the real ``PolicyEngine`` (tests may import policies), a
  ``FakeClock`` and a recording audit sink; the baseline is handed to the factory;
* the narrowed autonomy profile ``agent.autonomy.narrow(node.autonomy)``.

Payload builders (``requirements_payload`` ...) return schema-valid artifact payloads;
``Rig.publish``
validates every payload with ``schemas.validate_payload`` so a broken fixture fails loudly.
Fake secrets are assembled at runtime from fragments (organisation rule).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.recipes.base import (
    AspectSpec,
    BuildContext,
    Capability,
    GroupSpec,
    ParamSpec,
)
from orchestrator.core.artifacts import ArtifactStore
from orchestrator.core.clock import FakeClock
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.schemas import Kind, validate_payload
from orchestrator.core.serde import hash_jsonable
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
    CommandResult,
    EffectPort,
    EventType,
    FileView,
    Finding,
    GovernanceView,
    NodeSpec,
    RiskLevel,
    Stage,
)
from orchestrator.policies.engine import PolicyEngine
from orchestrator.tools.changeset import compute_ops
from orchestrator.tools.sandbox import SandboxRunner
from orchestrator.tools.toolbox import ScopedToolbox, make_toolbox_factory
from orchestrator.tools.workspace import MemoryView, OverlayView
from tests.orchestrator.policies.policy_builders import FakeRunView

RUN_ID = "run-1"


# ----------------------------------------------------------------------------- fake secrets


def fake_aws_key_id() -> str:
    """A secret-shaped string assembled at runtime (never a literal)."""
    return "AK" + "IA" + "Q" * 16


def fake_github_token() -> str:
    return "gh" + "p_" + "aB3dE5" * 6


# ----------------------------------------------------------------------------- nodes
# (sdlc-contract)


def _node(
    node_id: str,
    stage: Stage,
    agent: str,
    *,
    consumes: tuple[str, ...],
    produces: tuple[str, ...],
    optional: tuple[str, ...] = (),
    overlay: tuple[str, ...] = (),
    **kwargs: Any,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        title=node_id,
        stage=stage,
        agent=agent,
        consumes=consumes,
        optional_consumes=optional,
        overlay_from=overlay,
        produces=produces,
        **kwargs,
    )


INTEGRATED = Kind.INTEGRATED_CHANGESET
DOCS = Kind.DOCS_CHANGESET

NODES: Mapping[str, NodeSpec] = {
    "verify": _node(
        "verify",
        Stage.TESTING,
        "verifier",
        consumes=(Kind.REQUIREMENTS, INTEGRATED, Kind.TASK_PLAN),
        optional=(Kind.IMPACT_REPORT,),
        overlay=(INTEGRATED,),
        produces=(Kind.TEST_REPORT,),
    ),
    "security-scan": _node(
        "security-scan",
        Stage.SECURITY,
        "security-scanner",
        consumes=(INTEGRATED,),
        optional=(Kind.IMPACT_REPORT,),
        overlay=(INTEGRATED,),
        produces=(Kind.SECURITY_REPORT,),
    ),
    "docs": _node(
        "docs",
        Stage.DOCUMENTATION,
        "documenter",
        consumes=(Kind.REQUIREMENTS, Kind.DESIGN_DOC, INTEGRATED, Kind.TASK_PLAN),
        optional=(Kind.API_CONTRACT,),
        overlay=(INTEGRATED,),
        produces=(DOCS,),
    ),
    "release-readiness": _node(
        "release-readiness",
        Stage.RELEASE,
        "release-manager",
        consumes=(
            Kind.REQUIREMENTS,
            Kind.DESIGN_DOC,
            Kind.TASK_PLAN,
            INTEGRATED,
            Kind.INTEGRATION_REPORT,
            Kind.TEST_REPORT,
            Kind.SECURITY_REPORT,
            DOCS,
        ),
        optional=(Kind.IMPACT_REPORT, Kind.THREAT_MODEL, Kind.INTAKE_REPORT),
        produces=(Kind.READINESS_REPORT, Kind.TRACEABILITY_MATRIX),
        cacheable=False,
    ),
    "promote": _node(
        "promote",
        Stage.RELEASE,
        "promoter",
        consumes=(INTEGRATED, DOCS, Kind.READINESS_REPORT),
        overlay=(INTEGRATED, DOCS),
        produces=(Kind.RELEASE_PATCH, Kind.RELEASE_BUNDLE),
    ),
    "release-verify": _node(
        "release-verify",
        Stage.RELEASE,
        "release-verifier",
        consumes=(Kind.RELEASE_BUNDLE, Kind.RELEASE_PATCH),
        overlay=(INTEGRATED, DOCS),
        produces=(Kind.RELEASE_VERIFICATION,),
    ),
}


# ----------------------------------------------------------------------------- audit / execution


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


def command_result(
    stdout: str = "",
    returncode: int = 0,
    *,
    stderr: str = "",
    timed_out: bool = False,
    truncated: bool = False,
) -> CommandResult:
    return CommandResult(
        ("python", "-m", "scripted"), returncode, stdout, stderr, 0.0, timed_out, truncated
    )


ExecHandler = Callable[[Sequence[str], str, Sequence[str]], CommandResult]


class ScriptedToolbox(ScopedToolbox):
    """Real reads, writes, policy and audit; ``run_tests`` / ``run_python`` call a scripted handler.

    ``handler(targets_or_args, mode, overlay_paths) -> CommandResult`` (it may also raise).  Every
    call is recorded in ``calls`` as ``(tool, targets_or_args, mode, overlay_paths)`` and the staged
    state of the toolbox at that moment is available through ``read_text`` like in production.
    """

    def __init__(
        self,
        *,
        tests: ExecHandler | None = None,
        python: ExecHandler | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._on_tests = tests
        self._on_python = python
        self.calls: list[tuple[str, tuple[str, ...], str, tuple[str, ...]]] = []

    def run_tests(
        self,
        targets: Sequence[str] = (),
        *,
        timeout_s: float | None = None,
        mode: str = "view",
        overlay_paths: Sequence[str] = (),
    ) -> CommandResult:
        self._begin("run_tests")
        self.calls.append(("run_tests", tuple(targets), mode, tuple(overlay_paths)))
        assert self._on_tests is not None, "no scripted run_tests handler"
        return self._on_tests(targets, mode, overlay_paths)

    def run_python(
        self,
        args: Sequence[str],
        *,
        timeout_s: float | None = None,
        mode: str = "view",
        overlay_paths: Sequence[str] = (),
    ) -> CommandResult:
        self._begin("run_python")
        self.calls.append(("run_python", tuple(args), mode, tuple(overlay_paths)))
        assert self._on_python is not None, "no scripted run_python handler"
        return self._on_python(args, mode, overlay_paths)


def plain_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speed-up for tests: skip the temp-file + replace dance of ``atomic_write_bytes`` (hundreds of
    ms per file on some Windows set-ups).  Use it from an autouse fixture; tests that exercise the
    real writer (sandbox, effect port) must not."""

    def write(path: Path | str, data: bytes, *, mode: int = 0o600) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    monkeypatch.setattr("orchestrator.core.fsutil.atomic_write_bytes", write)
    monkeypatch.setattr("orchestrator.core.artifacts.atomic_write_bytes", write)


# ----------------------------------------------------------------------------- the rig


@dataclass
class Rig:
    """Everything an agent needs, wired like the engine does (see the module docstring)."""

    root: Path
    baseline_files: Mapping[str, str] = field(default_factory=dict)
    clock_start: datetime | None = None
    with_runner: bool = False

    def __post_init__(self) -> None:
        self.clock = FakeClock(self.clock_start)
        self.audit = AuditRecorder()
        self.policy = PolicyEngine()
        self.store = ArtifactStore(self.root / "store", clock=self.clock)
        self.baseline = MemoryView(self.baseline_files)
        self.runner: SandboxRunner | None = (
            SandboxRunner(self.root / "sandbox") if self.with_runner else None
        )
        self.factory = make_toolbox_factory(runner=self.runner)

    # -- artifacts ----------------------------------------------------------------
    def publish(
        self,
        kind: str,
        payload: Any,
        *,
        key: str = "",
        producer: str = "upstream",
        validate: bool = True,
    ) -> Artifact:
        """Store + activate an artifact (dict: JSON, validated; ChangeSet: its dict; str: text)."""
        content = payload.to_dict() if isinstance(payload, ChangeSet) else payload
        if validate and isinstance(content, Mapping):
            problems = validate_payload(kind, content)
            assert not problems, f"invalid {kind} fixture: {problems}"
        return self.store.put(ArtifactDraft(kind=kind, content=content, key=key), producer=producer)

    def publish_changeset(
        self,
        kind: str,
        desired: Mapping[str, str | None],
        *,
        base: Mapping[str, str] | None = None,
        key: str = "",
        producer: str = "upstream",
        author: str = "",
    ) -> ChangeSet:
        """Publish ``compute_ops(base or baseline, desired)``; returns the change set."""
        cs = compute_ops(
            MemoryView(self.baseline_files if base is None else base),
            desired,
            description=kind,
            author=author,
        )
        self.publish(kind, cs, key=key, producer=producer)
        return cs

    # -- views / contexts -----------------------------------------------------------
    def node(self, name: str, **overrides: Any) -> NodeSpec:
        import dataclasses

        spec = NODES[name]
        return dataclasses.replace(spec, **overrides) if overrides else spec

    def view_for(self, spec: NodeSpec) -> FileView:
        """``baseline + ACTIVE overlay_from change sets`` (engine-semantics section 6)."""
        sets = [
            ChangeSet.from_dict(self.store.read_json(artifact))
            for kind in spec.overlay_from
            for artifact in self.store.active(kind)
        ]
        return OverlayView(self.baseline, sets)

    def context(
        self,
        agent: Agent,
        node_name: str,
        *,
        params: Mapping[str, Any] | None = None,
        feedback: Sequence[Finding] = (),
        effects: EffectPort | None = None,
        run_view: GovernanceView | None = None,
        is_fallback: bool = False,
        cancel: CancelToken | None = None,
        tests: ExecHandler | None = None,
        python: ExecHandler | None = None,
        node_overrides: Mapping[str, Any] | None = None,
        baseline_handed_in: bool = True,
    ) -> AgentContext:
        spec = self.node(node_name, **dict(node_overrides or {}))
        cancel = cancel or CancelToken()
        profile: AutonomyProfile = agent.autonomy.narrow(spec.autonomy)
        common: dict[str, Any] = {
            "run_id": RUN_ID,
            "node_id": spec.id,
            "attempt": 1,
            "agent_id": agent.agent_id,
            "profile": profile,
            "view": self.view_for(spec),
            "cancel": cancel,
            "clock": self.clock,
            "policy": self.policy,
            "audit": self.audit,
        }
        baseline = self.baseline if baseline_handed_in else None
        if tests is not None or python is not None:
            toolbox: Any = ScriptedToolbox(tests=tests, python=python, baseline=baseline, **common)
        else:
            toolbox = self.factory(baseline=baseline, **common)
        return AgentContext(
            run_id=RUN_ID,
            node=spec,
            attempt=1,
            attempt_no=1,
            generation=1,
            agent_id=agent.agent_id,
            is_fallback=is_fallback,
            params=dict(spec.params if params is None else params),
            artifacts=self.store.snapshot(spec.input_kinds, lambda _producer: True),
            toolbox=toolbox,
            clock=self.clock,
            cancel=cancel,
            run_view=run_view or FakeRunView(),
            feedback=tuple(feedback),
            effects=effects,
        )

    def run(self, agent: Agent, node_name: str, **kwargs: Any) -> AgentResult:
        return agent.run(self.context(agent, node_name, **kwargs))


# ----------------------------------------------------------------------------- results


def drafts(result: AgentResult, kind: str) -> list[ArtifactDraft]:
    return [d for d in result.artifacts if d.kind == kind]


def payload(result: AgentResult, kind: str) -> dict[str, Any]:
    """The decoded content of the single JSON artifact draft of ``kind``."""
    found = drafts(result, kind)
    assert len(found) == 1, f"expected exactly one {kind} draft, got {len(found)}"
    content = found[0].content
    assert isinstance(content, dict), f"{kind} draft is not a JSON object"
    return content


def text_of(result: AgentResult, kind: str) -> str:
    found = drafts(result, kind)
    assert len(found) == 1, f"expected exactly one {kind} draft, got {len(found)}"
    assert isinstance(found[0].content, str)
    return found[0].content


def changeset_of(result: AgentResult, kind: str) -> ChangeSet:
    return ChangeSet.from_dict(payload(result, kind))


def content_hashes(result: AgentResult) -> dict[str, str]:
    """Hash of every draft's *content* (metadata is excluded, like the engine's content hash)."""
    out: dict[str, str] = {}
    for draft in result.artifacts:
        content = draft.content
        out[draft.kind] = (
            sha256_text(content) if isinstance(content, str) else hash_jsonable(content)
        )
    return out


def assert_valid(result: AgentResult) -> None:
    """Every JSON draft of ``result`` validates against its kind's schema."""
    for draft in result.artifacts:
        if isinstance(draft.content, Mapping):
            problems = validate_payload(draft.kind, draft.content)
            assert not problems, f"{draft.kind}: {problems}"


# ----------------------------------------------------------------------------- payload builders


def requirement(
    rid: str = "R-001",
    *,
    text: str = "The widget shall be exported.",
    status: str = "active",
    verification: str = "test",
    capability: str = "widgets",
    criteria: int = 2,
    status_decision: str | None = None,
    priority: str = "must",
) -> dict[str, Any]:
    number = int(rid.split("-")[1])
    return {
        "id": rid,
        "text": text,
        "kind": "functional",
        "priority": priority,
        "status": status,
        "status_decision": status_decision
        if status_decision is not None or status == "active"
        else f"SCOPE:{rid}",
        "capabilities": [capability],
        "parameters": {},
        "verification": verification,
        "acceptance_criteria": [
            {"id": f"AC-{number:03d}-{j}", "text": f"Criterion {j} of {rid} holds."}
            for j in range(1, criteria + 1)
        ],
    }


def requirements_payload(*reqs: Mapping[str, Any], capability: str = "widgets") -> dict[str, Any]:
    listed = [dict(r) for r in (reqs or (requirement(capability=capability),))]
    return {
        "schema": 1,
        "problem": {
            "goal": "Deliver widgets.",
            "in_scope": [],
            "out_of_scope": [],
            "constraints": [],
        },
        "classification": {"workspace_kind": "greenfield", "change_type": "feature"},
        "requirements": listed,
        "assumptions": [],
        "ambiguities": [],
        "open_questions": [],
        "unsupported": [],
        "capabilities": [
            {
                "id": capability,
                "title": capability.title(),
                "requirement_ids": [r["id"] for r in listed],
                "params": {},
            }
        ],
    }


def task(
    tid: str = "T-001",
    *,
    group: str = "core",
    requirement_ids: Sequence[str] = ("R-001",),
    capability: str = "widgets",
    kind: str = "implement",
    expected_paths: Sequence[str] = ("widgets/core.py",),
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": tid,
        "title": f"Task {tid}",
        "group": group,
        "requirement_ids": list(requirement_ids),
        "depends_on": [],
        "capability": capability,
        "kind": kind,
        "write_scope": ["widgets/**"],
        "expected_paths": list(expected_paths),
        "params": dict(params or {}),
    }


def task_plan_payload(
    *tasks: Mapping[str, Any], capability: str = "widgets", params: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    listed = [dict(t) for t in (tasks or (task(capability=capability),))]
    groups: dict[str, dict[str, Any]] = {}
    for t in listed:
        g = groups.setdefault(
            t["group"],
            {
                "id": t["group"],
                "node_id": f"implement.{t['group']}",
                "task_ids": [],
                "capability": t["capability"],
                "params": dict(params or {}),
                "write_scope": ["widgets/**"],
            },
        )
        g["task_ids"].append(t["id"])
    return {
        "schema": 1,
        "tasks": listed,
        "groups": list(groups.values()),
        "critical_path": [t["id"] for t in listed],
        "unsupported": [],
    }


def design_doc_payload(
    *,
    risk: str = "LOW",
    rollback_plan: str = "Revert the release patch; no data is migrated.",
    change_reference: str = "CHG-100",
    api_changes: Sequence[Mapping[str, Any]] = (),
    data_model_changes: Sequence[Mapping[str, Any]] = (),
    summary: str = "Add widgets to the service.",
    requirement_ids: Sequence[str] = ("R-001",),
) -> dict[str, Any]:
    return {
        "schema": 1,
        "summary": summary,
        "components": [
            {
                "name": "widgets",
                "responsibility": "Own the widget behaviour.",
                "touches": ["widgets/core.py"],
                "requirement_ids": list(requirement_ids),
            }
        ],
        "adrs": [
            {
                "id": "ADR-001",
                "title": "Keep widgets in one module",
                "decision": "A single module owns the behaviour.",
                "alternatives": [{"option": "Split per feature", "rejected_because": "Too small."}],
                "rationale": "The surface is tiny.",
                "consequences": "Easy to revert.",
            }
        ],
        "data_model_changes": [dict(c) for c in data_model_changes],
        "api_changes": [dict(c) for c in api_changes],
        "rollback_plan": rollback_plan,
        "change_reference": change_reference,
        "risk": {"level": risk, "reasons": ["small, additive change"]},
    }


def report_row(nodeid: str, outcome: str = "passed", covers: Sequence[str] = ()) -> dict[str, Any]:
    return {"nodeid": nodeid, "outcome": outcome, "covers": sorted(covers)}


def build_test_report(
    tests: Sequence[Mapping[str, Any]],
    *,
    fail_to_pass_ok: bool = True,
    fail_to_pass_checked: bool = True,
    api_breaking: Sequence[str] = (),
    api_checked: bool = False,
) -> dict[str, Any]:
    listed = [dict(t) for t in tests]
    count = {
        o: sum(1 for t in listed if t["outcome"] == o)
        for o in ("passed", "failed", "error", "skipped")
    }
    return {
        "schema": 1,
        "passed": bool(listed) and count["failed"] == 0 and count["error"] == 0,
        "total": len(listed),
        "failed": count["failed"],
        "errors": count["error"],
        "skipped": count["skipped"],
        "tests": listed,
        "baseline": {"ran": fail_to_pass_checked, "passed": False, "failed_tests": []},
        "fail_to_pass": {
            "checked": fail_to_pass_checked,
            "ok": fail_to_pass_ok,
            "tests": [t["nodeid"] for t in listed if t["outcome"] == "passed"],
        },
        "perf": {"checked": False, "budget_ms": None, "ok": None},
        "api_compat": {"checked": api_checked, "breaking": list(api_breaking), "additive": []},
    }


def security_report_payload(
    findings: Sequence[Mapping[str, Any]] = (), *, files_scanned: int = 2
) -> dict[str, Any]:
    listed = [dict(f) for f in findings]
    return {
        "schema": 1,
        "rules_evaluated": ["SCN-001", "SCN-002", "SCN-003", "SCN-004"],
        "findings": listed,
        "files_scanned": files_scanned,
        "passed": not any(f["severity"] in ("HIGH", "CRITICAL") for f in listed),
    }


def integration_report_payload(
    *, conflicts: Sequence[Mapping[str, Any]] = (), merged_paths: Sequence[str] = ()
) -> dict[str, Any]:
    return {
        "schema": 1,
        "sources": [],
        "merged_paths": list(merged_paths),
        "conflicts": [dict(c) for c in conflicts],
        "stats": {
            "files_added": len(merged_paths),
            "files_modified": 0,
            "files_deleted": 0,
            "lines_added": 0,
            "lines_removed": 0,
        },
        "impact_delta": {"predicted_only": [], "actual_only": [], "precision": 1.0, "recall": 1.0},
    }


# ----------------------------------------------------------------------------- fake capabilities


class FakeRecipe:
    """A scripted ``Recipe``: each ``build_*`` stages the configured files through the toolbox."""

    id = "fake"
    params: tuple[ParamSpec, ...] = (ParamSpec("limit", "int", default=5, minimum=1, maximum=100),)

    def __init__(
        self,
        *,
        group_files: Mapping[str, str] | None = None,
        test_files: Mapping[str, str] | None = None,
        doc_files: Mapping[str, str] | None = None,
        api_changes: Sequence[Mapping[str, Any]] = (),
        docs_error: Exception | None = None,
    ) -> None:
        self.group_files = dict(group_files or {})
        self.test_files = dict(test_files or {})
        self.doc_files = dict(doc_files or {})
        self.api_changes = [dict(c) for c in api_changes]
        self.docs_error = docs_error
        self.doc_contexts: list[BuildContext] = []

    def groups(self, params: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
        return (GroupSpec("core", "Core", ("widgets/**",), ("widgets/core.py",)),)

    def design(self, params: Mapping[str, Any], model: Any | None) -> Mapping[str, Any]:
        return {
            "components": [],
            "adrs": [],
            "data_model_changes": [],
            "api_changes": list(self.api_changes),
            "rollback_plan": "Revert the patch.",
        }

    def build_group(self, group_id: str, ctx: BuildContext) -> None:
        for path, text in self.group_files.items():
            ctx.toolbox.write_text(path, text)

    def build_tests(self, ctx: BuildContext) -> None:
        for path, text in self.test_files.items():
            ctx.toolbox.write_text(path, text)

    def build_docs(self, ctx: BuildContext) -> None:
        self.doc_contexts.append(ctx)
        if self.docs_error is not None:
            raise self.docs_error
        for path, text in self.doc_files.items():
            ctx.toolbox.write_text(path, text)


def fake_capability(
    capability_id: str = "widgets",
    recipe: FakeRecipe | None = None,
    *,
    title: str | None = None,
    risk: RiskLevel = RiskLevel.LOW,
) -> Capability:
    return Capability(
        id=capability_id,
        title=title or capability_id.title(),
        kind="new_module",
        triggers=("widget",),
        aspects=(AspectSpec("basic", "Basic behaviour", ("widget",), "Widgets work."),),
        extractors={},
        impact_keywords=("widget",),
        risk=risk,
        recipe=recipe or FakeRecipe(),
    )


def install_capabilities(monkeypatch: pytest.MonkeyPatch, *caps: Capability) -> None:
    """Make ``catalog.get_capability`` resolve exactly ``caps`` (no real recipe is imported)."""
    monkeypatch.setattr("orchestrator.agents.catalog._cache", {c.id: c for c in caps})
