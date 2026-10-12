"""Engineering-summary finalizer: success, failed (missing artifacts) and safe-stopped runs.

The finalizer reads a ``RunView``; the tests use a hand-written view double (no scheduler, no
stores) with schema-valid payloads, so each expected line can be derived on paper.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.artifacts import ArtifactStore
from orchestrator.core.clock import FakeClock
from orchestrator.core.decisions import DecisionLog
from orchestrator.core.errors import IntegrityError
from orchestrator.core.schemas import validate_payload
from orchestrator.core.types import (
    Actor,
    ApprovalPosition,
    ApprovalPurpose,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalVote,
    Artifact,
    ArtifactDraft,
    AuditEvent,
    Decision,
    DecisionKind,
    EventType,
    NodeSpec,
    NodeStatus,
    RiskLevel,
    RunStatus,
    Stage,
    Waiver,
)
from orchestrator.engine.scheduler import FinalizerContext, RunSnapshot
from orchestrator.workflows.summary import (
    engineering_summary_finalizer,
    render_engineering_summary,
)

ET = EventType
T0 = datetime(2026, 1, 1, tzinfo=UTC)
HEAD = (6, "a" * 64)
HOSTILE = (
    "\x1b[31mred\x1b[0m <script>alert(1)</script> [click](https://evil.example/x) "
    "![img](http://evil.example/i.png) https://evil.example/a www.evil.example "
    "mail@evil.example | cell\nnew line `code` # heading \u202eRTL\u202c"
)
HEADINGS = [
    "# Engineering summary",
    "## 1. Plan and rationale",
    "## 2. Artifacts produced",
    "## 3. Risks, trade-offs and validation",
    "## 4. Assumptions",
    "## 5. Limitations",
]
CITATION = re.compile(r"`(?:[a-z_]+(?:\.[A-Za-z0-9_-]+)?@v\d+|D-\d{4}|audit:\d+)`")


# ----------------------------------------------------------------------------- payloads


def requirements_payload() -> dict[str, Any]:
    def req(number: int, text: str, status: str, decision: str | None) -> dict[str, Any]:
        return {
            "id": f"R-{number:03d}",
            "text": text,
            "kind": "functional",
            "priority": "must",
            "status": status,
            "status_decision": decision,
            "capabilities": [],
            "parameters": {},
            "verification": "test",
            "acceptance_criteria": [{"id": f"AC-{number:03d}-1", "text": "measurable"}],
        }

    return {
        "schema": 1,
        "problem": {
            "goal": "Add click limits to short links",
            "in_scope": ["limits"],
            "out_of_scope": [],
            "constraints": [],
        },
        "classification": {"workspace_kind": "brownfield", "change_type": "enhancement"},
        "requirements": [
            req(1, "Links must stop redirecting after max clicks", "active", None),
            req(2, "Export analytics as CSV", "deferred", "SCOPE:R-002"),
        ],
        "assumptions": [
            {"id": "ASM-001", "text": "Clicks are counted per redirect", "question_id": None}
        ],
        "ambiguities": [],
        "open_questions": [],
        "unsupported": [
            {"requirement_id": "R-003", "reason": "no capability matches weekly report"}
        ],
        "capabilities": [],
    }


def task_plan_payload() -> dict[str, Any]:
    def task(number: int, kind: str, after: list[str]) -> dict[str, Any]:
        return {
            "id": f"T-{number:03d}",
            "title": f"{kind} the click counter",
            "group": "core",
            "requirement_ids": ["R-001"],
            "depends_on": after,
            "capability": "max_clicks",
            "kind": kind,
            "write_scope": [],
            "expected_paths": [],
            "params": {},
        }

    return {
        "schema": 1,
        "tasks": [task(1, "implement", []), task(2, "test", ["T-001"])],
        "groups": [
            {
                "id": "core",
                "node_id": "implement.core",
                "task_ids": ["T-001"],
                "capability": "max_clicks",
                "params": {},
                "write_scope": [],
            }
        ],
        "critical_path": ["T-001", "T-002"],
        "unsupported": [
            {"requirement_id": "R-003", "reason": "no capability matches weekly report"}
        ],
    }


def design_payload() -> dict[str, Any]:
    return {
        "schema": 1,
        "summary": "Counter column plus a guard in the redirect path",
        "components": [
            {
                "name": "links",
                "responsibility": "counts clicks",
                "touches": ["shortener/services/links.py"],
                "requirement_ids": ["R-001"],
            }
        ],
        "adrs": [
            {
                "id": "ADR-001",
                "title": "Count in the database",
                "decision": "Use an atomic UPDATE",
                "alternatives": [
                    {"option": "In-memory counter", "rejected_because": "lost on restart"}
                ],
                "rationale": "Durable and race free",
                "consequences": "One more write per redirect",
            }
        ],
        "data_model_changes": [
            {
                "table": "links",
                "change": "add_column",
                "detail": "max_clicks INTEGER",
                "migration": "003_max_clicks.sql",
            }
        ],
        "api_changes": [
            {"method": "POST", "path": "/api/v1/links", "change": "modify", "breaking": False}
        ],
        "rollback_plan": "Revert the migration and the patch",
        "change_reference": "CR-abc123",
        "risk": {"level": "HIGH", "reasons": ["schema migration"]},
    }


def threat_payload() -> dict[str, Any]:
    return {
        "schema": 1,
        "assets": ["links"],
        "trust_boundaries": ["public redirect"],
        "risks": [
            {
                "id": "RISK-001",
                "category": "denial_of_service",
                "title": "Counter race",
                "description": "two redirects race",
                "likelihood": "medium",
                "impact": "high",
                "mitigations": [{"text": "atomic update", "requirement_id": "R-001"}],
                "status": "mitigated",
            }
        ],
        "requirement_ids": ["R-001"],
    }


def make_report_payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": 1,
        "passed": True,
        "total": 3,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "tests": [
            {"nodeid": "tests/test_a.py::test_x", "outcome": "passed", "covers": ["R-001"]},
            {"nodeid": "tests/test_a.py::test_y", "outcome": "passed", "covers": ["R-001"]},
            {"nodeid": "tests/test_a.py::test_z", "outcome": "passed", "covers": []},
        ],
        "baseline": {"ran": True, "passed": True, "failed_tests": []},
        "fail_to_pass": {"checked": True, "ok": True, "tests": ["tests/test_a.py::test_x"]},
        "perf": {"checked": False, "budget_ms": None, "ok": None},
        "api_compat": {"checked": True, "breaking": [], "additive": ["POST /api/v1/links"]},
    }
    body.update(overrides)
    return body


def bundle_payload(mode: str = "bundle", files: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema": 1,
        "mode": mode,
        "patch_sha256": "a" * 64,
        "files": files
        if files is not None
        else ["shortener/services/links.py", "migrations/003.sql"],
        "base_manifest_hash": "b" * 64,
        "receipt": None,
    }


PATCH_TEXT = (
    "diff --git a/shortener/services/links.py b/shortener/services/links.py\n"
    "--- a/shortener/services/links.py\n+++ b/shortener/services/links.py\n@@ -1 +1 @@\n-a\n+b\n"
    "diff --git a/docs/new.md b/docs/new.md\nnew file mode 100644\n"
)


def payloads() -> dict[str, Any]:
    """One valid payload per kind of a fully successful run."""
    out: dict[str, Any] = {
        "requirements": requirements_payload(),
        "task_plan": task_plan_payload(),
        "design_doc": design_payload(),
        "threat_model": threat_payload(),
        "test_report": make_report_payload(),
        "security_report": {
            "schema": 1,
            "rules_evaluated": ["SEC-001", "SEC-002"],
            "findings": [],
            "files_scanned": 12,
            "passed": True,
        },
        "readiness_report": {
            "schema": 1,
            "verdict": "GO",
            "checks": [
                {"id": "tests", "title": "Tests pass", "passed": True, "evidence": [], "detail": ""}
            ],
            "risk": {"level": "HIGH", "reasons": ["schema migration"]},
            "rollback_plan": "revert",
            "change_reference": "CR-abc123",
            "open_items": ["Announce the change to API users"],
        },
        "traceability_matrix": {"schema": 1, "requirements": [], "complete": True, "gaps": []},
        "release_bundle": bundle_payload(),
        "release_verification": {
            "schema": 1,
            "target": "bundle-scratch",
            "applied_cleanly": True,
            "tests_passed": True,
            "healthz_ok": True,
            "checked_files": 2,
        },
        "api_contract": {"schema": 1, "paths": [], "breaking": False},
        "clarification_answers": {
            "schema": 1,
            "answers": {
                "Q-unquantified-aaaaaa": {"answer": "Up to 90 days", "by": "carol", "round": 1}
            },
        },
    }
    return out


# ----------------------------------------------------------------------------- the view double


def artifact(kind: str, version: int = 1, **overrides: Any) -> Artifact:
    values: dict[str, Any] = {
        "id": f"{kind}@v{version}",
        "kind": kind,
        "key": "",
        "version": version,
        "content_hash": hashlib.sha256(f"{kind}{version}".encode()).hexdigest(),
        "media_type": "application/json",
        "size": 10,
        "path": f"{kind}/v{version}.json",
        "created_at": T0,
        "producer": "design",
        "attempt": 1,
        "agent_id": "architect",
    }
    values.update(overrides)
    return Artifact(**values)


class FakeView:
    """A ``RunView`` double holding exactly what the finalizer reads."""

    def __init__(
        self,
        docs: dict[str, Any] | None = None,
        texts: dict[str, str] | None = None,
        errors: dict[str, Exception] | None = None,
        **parts: Any,
    ) -> None:
        self._docs = docs or {}
        self._texts = texts or {}
        self._errors = errors or {}
        self._artifacts = {k: artifact(k) for k in (*self._docs, *self._texts)}
        self._parts = parts

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        return self._artifacts.get(kind)

    def read_json(self, art: Artifact) -> Any:
        if art.kind in self._errors:
            raise self._errors[art.kind]
        return self._docs[art.kind]

    def read_text(self, art: Artifact) -> str:
        return self._texts[art.kind]

    def artifacts(self, include_inactive: bool = False) -> tuple[Artifact, ...]:
        return tuple(self._artifacts.values())

    def decisions(self) -> tuple[Decision, ...]:
        return tuple(self._parts.get("decisions", ()))

    def approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(self._parts.get("approvals", ()))

    def waivers(self) -> tuple[Waiver, ...]:
        return tuple(self._parts.get("waivers", ()))

    def node_statuses(self) -> dict[str, NodeStatus]:
        return dict(self._parts.get("statuses", {}))

    def node_specs(self) -> dict[str, NodeSpec]:
        return dict(self._parts.get("specs", {}))

    def params(self) -> dict[str, Any]:
        return dict(self._parts.get("params", {}))

    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._parts.get("events", ()))


class MinimalView:
    """The accessors the assignment guarantees; ``artifacts`` / ``waivers`` / ``node_specs`` /
    ``params`` are missing on purpose."""

    def __init__(self, docs: dict[str, Any]) -> None:
        self._docs = docs

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        return artifact(kind) if kind in self._docs else None

    def read_json(self, art: Artifact) -> Any:
        return self._docs[art.kind]

    def read_text(self, art: Artifact) -> str:
        raise KeyError(art.kind)

    def decisions(self) -> tuple[Decision, ...]:
        return ()

    def approvals(self) -> tuple[ApprovalRequest, ...]:
        return ()

    def node_statuses(self) -> dict[str, NodeStatus]:
        return {"requirements": NodeStatus.SUCCEEDED}

    def events(self) -> tuple[AuditEvent, ...]:
        return ()


@dataclass
class FakeCtx:
    """Duck-typed ``FinalizerContext`` capturing what is published."""

    view: Any
    status: RunStatus = RunStatus.SUCCEEDED
    reason: str = ""
    run_id: str = "run-1"
    run_dir: Path = Path("unused")
    audit_head: tuple[int, str] = HEAD
    published: list[ArtifactDraft] = field(default_factory=list)

    def publish_artifact(self, draft: ArtifactDraft) -> Artifact:
        self.published.append(draft)
        return artifact(draft.kind)

    def record_decision(self, *args: Any, **kwargs: Any) -> Decision:
        raise AssertionError("the summary must not record decisions")


def render(ctx: Any) -> str:
    return render_engineering_summary(ctx)


def finalize(ctx: Any) -> None:
    engineering_summary_finalizer(ctx)


def event(seq: int, etype: EventType, node: str | None = None, **data: Any) -> AuditEvent:
    return AuditEvent(
        seq=seq,
        ts=T0,
        run_id="run-1",
        trace_id="trace-1",
        span_id="run" if node is None else f"n:{node}",
        parent_span_id=None,
        type=etype,
        node_id=node,
        attempt=None,
        actor=Actor.system(),
        data=data,
        prev_hash="0" * 64,
        hash=f"{seq:064x}",
    )


def decision(
    number: int,
    kind: DecisionKind,
    summary: str,
    actor: Actor | None = None,
    node: str | None = None,
    **fields: Any,
) -> Decision:
    return Decision(
        id=f"D-{number:04d}",
        run_id="run-1",
        ts=T0,
        kind=kind,
        actor=actor or Actor.agent("requirements"),
        node_id=node,
        summary=summary,
        **fields,
    )


def vote(approver: str, channel: str = "cli", comment: str = "ok") -> ApprovalVote:
    return ApprovalVote(approver, True, comment, T0, channel)


def approval(
    votes: list[ApprovalVote], request_id: str = "A-0001", node: str = "design"
) -> ApprovalRequest:
    return ApprovalRequest(
        id=request_id,
        run_id="run-1",
        node_id=node,
        position=ApprovalPosition.AFTER,
        purpose=ApprovalPurpose.NODE_OUTPUT,
        risk=RiskLevel.HIGH,
        reasons=("migration",),
        summary="engine summary",
        evidence=(),
        node_fingerprint="f" * 64,
        evidence_hash="e" * 64,
        required_approvals=2,
        attempt=1,
        generation=1,
        created_at=T0,
        status=ApprovalStatus.APPROVED,
        votes=votes,
    )


def spec(node_id: str, produces: tuple[str, ...]) -> NodeSpec:
    return NodeSpec(
        id=node_id, title=node_id, stage=Stage.TESTING, agent=f"{node_id}-agent", produces=produces
    )


def success_view(**overrides: Any) -> FakeView:
    decisions = (
        decision(
            1,
            DecisionKind.CLARIFICATION,
            "Answer to Q-unquantified-aaaaaa",
            Actor.human("carol"),
            data={"question_id": "Q-unquantified-aaaaaa", "answer": "Up to 90 days"},
        ),
        decision(
            2,
            DecisionKind.SCOPE,
            "Deferred R-002: Export analytics as CSV",
            refs=("R-002",),
            parents=("D-0001",),
            data={"question_id": "Q-unquantified-aaaaaa", "requirement_id": "R-002"},
        ),
        decision(3, DecisionKind.ASSUMPTION, "ASM-001: per redirect", refs=("ASM-001", "R-001")),
        decision(
            4,
            DecisionKind.DESIGN,
            "Count clicks in the database",
            Actor.agent("architect"),
            "design",
            rationale="durable",
            alternatives=("in-memory counter",),
            parents=("D-0003",),
        ),
        decision(5, DecisionKind.APPROVAL, "Approved the design", Actor.human("bob"), "design"),
        decision(6, DecisionKind.WAIVER, "Waived SEC-006", Actor.human("bob"), "design"),
    )
    events = (
        event(1, ET.RUN_CREATED),
        event(2, ET.NODE_STARTED, "requirements", agent_id="requirements", stage="requirements"),
        event(3, ET.NODE_SKIPPED, "codebase-analysis", reason="greenfield workspace"),
        event(
            4,
            ET.POLICY_VIOLATION,
            "implement.core",
            rule_id="SEC-001",
            severity="HIGH",
            enforcement="deny",
            location="app.py:3",
        ),
        event(5, ET.APPROVAL_VOTE, "design", approval_id="A-0001", channel="scripted"),
        event(6, ET.RUN_SUCCEEDED),
    )
    parts: dict[str, Any] = {
        "decisions": decisions,
        "approvals": (approval([vote("bob", "scripted"), vote("carol", "scripted")]),),
        "waivers": (
            Waiver(
                "W-0001",
                "SEC-006",
                "k",
                "design",
                "e" * 64,
                "reviewed",
                ("bob",),
                "A-0001",
                "D-0006",
                T0,
            ),
        ),
        "statuses": {
            "requirements": NodeStatus.SUCCEEDED,
            "design": NodeStatus.SUCCEEDED,
            "codebase-analysis": NodeStatus.SKIPPED,
        },
        "events": events,
        "params": {},
    }
    parts.update(overrides)
    return FakeView(payloads(), {"release_patch": PATCH_TEXT}, **parts)


def failed_view(**overrides: Any) -> FakeView:
    docs = {k: v for k, v in payloads().items() if k in ("requirements", "task_plan")}
    events = (
        event(1, ET.RUN_CREATED),
        event(2, ET.NODE_STARTED, "verify", agent_id="verifier"),
        event(3, ET.NODE_FAILED, "verify", kind="gate_rejected", message="2 tests failed"),
        event(4, ET.NODE_BLOCKED, "docs", reason="dependency verify failed"),
        event(5, ET.RUN_FAILED, reason="verify failed"),
    )
    parts: dict[str, Any] = {
        "decisions": (
            decision(1, DecisionKind.REJECTION, "Rejected verify output", node="verify"),
        ),
        "statuses": {
            "requirements": NodeStatus.SUCCEEDED,
            "verify": NodeStatus.FAILED,
            "docs": NodeStatus.BLOCKED,
            "release-readiness": NodeStatus.BLOCKED,
        },
        "specs": {
            "verify": spec("verify", ("test_report",)),
            "security-scan": spec("security-scan", ("security_report",)),
            "release-readiness": spec("release-readiness", ("readiness_report",)),
            "design": spec("design", ("design_doc",)),
        },
        "events": events,
    }
    parts.update(overrides)
    return FakeView(docs, **parts)


def bullets(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- ")]


def section(text: str, heading: str) -> str:
    lines = text.splitlines()
    start = lines.index(heading) + 1
    out: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        out.append(line)
    return "\n".join(out)


def outside_fences(text: str) -> list[str]:
    """Lines that are not inside a fenced code block (CommonMark closing-fence rule)."""
    out: list[str] = []
    fence = 0
    for line in text.splitlines():
        stripped = line.strip()
        ticks = len(stripped) - len(stripped.lstrip("`"))
        if fence == 0 and ticks >= 3 and not stripped.rstrip("`").strip().count("`"):
            fence = ticks
        elif fence and ticks >= fence and stripped == "`" * ticks:
            fence = 0
        elif not fence:
            out.append(line)
    return out


# ----------------------------------------------------------------------------- fixtures sanity


@pytest.mark.parametrize("kind", sorted(k for k in payloads() if k != "api_contract"))
def test_the_fixture_payloads_are_schema_valid(kind: str) -> None:
    assert validate_payload(kind, payloads()[kind]) == []


# ----------------------------------------------------------------------------- success run


def test_success_summary_has_the_five_sections_in_order() -> None:
    text = render(FakeCtx(success_view()))
    positions = [text.index(heading + "\n") for heading in HEADINGS]
    assert positions == sorted(positions)
    assert [
        line for line in outside_fences(text) if line.startswith("#") and "###" not in line
    ] == (HEADINGS)
    assert text.endswith("\n")


def test_plan_section_has_requirements_tasks_adrs_with_alternatives_and_decisions() -> None:
    plan = section(render(FakeCtx(success_view())), "## 1. Plan and rationale")
    assert "- Goal: Add click limits to short links (`requirements@v1`)" in plan
    assert (
        "- `R-001` Links must stop redirecting after max clicks - must, functional, active, "
        "1 acceptance criteria (`requirements@v1`)"
    ) in plan
    assert "- `T-001` implement the click counter - group `core`, implement; covers `R-001`" in plan
    assert "depends on `T-001` (`task_plan@v1`)" in plan
    assert "- Groups and their nodes: `core` -> `implement.core` (`task_plan@v1`)" in plan
    assert "- Critical path: `T-001` -> `T-002` (`task_plan@v1`)" in plan
    assert "- `ADR-001` Count in the database (`design_doc@v1`)" in plan
    assert "  - Decision: Use an atomic UPDATE" in plan
    assert (
        "  - Alternatives considered: In-memory counter (rejected because: lost on restart)" in plan
    )
    assert "  - Rationale: Durable and race free" in plan
    assert (
        "- `D-0004` `design` by `agent:architect` on `design`: Count clicks in the database" in plan
    )
    assert "why: durable; alternatives: in-memory counter; builds on `D-0003`" in plan
    assert "- `D-0001` `clarification` by `human:carol`: Answer to Q-unquantified-aaaaaa" in plan


def test_every_decision_is_listed_with_its_id() -> None:
    plan = section(render(FakeCtx(success_view())), "## 1. Plan and rationale")
    for number in range(1, 7):
        assert f"- `D-{number:04d}` " in plan


def test_artifacts_section_lists_kind_id_hash_prefix_and_patch_files() -> None:
    text = render(FakeCtx(success_view()))
    block = section(text, "## 2. Artifacts produced")
    digest = hashlib.sha256(b"requirements1").hexdigest()[:12]
    assert f"| `requirements` | `requirements@v1` | `design` | `{digest}` |" in block
    assert hashlib.sha256(b"requirements1").hexdigest() not in text
    assert (
        "- 2 file(s) in the patch (release mode bundle) (`release_bundle@v1`, `release_patch@v1`):"
        in block
    )
    fence = block[block.index("```") :]
    assert "shortener/services/links.py\nmigrations/003.sql" in fence


def test_patch_files_come_from_the_patch_text_without_a_bundle() -> None:
    docs = {k: v for k, v in payloads().items() if k != "release_bundle"}
    view = FakeView(docs, {"release_patch": PATCH_TEXT})
    block = section(render(FakeCtx(view)), "## 2. Artifacts produced")
    assert "- 2 file(s) in the patch (`release_patch@v1`):" in block
    assert "shortener/services/links.py\ndocs/new.md" in block


def test_validation_section_covers_threats_policy_tests_security_readiness_and_approvals() -> None:
    risks = section(render(FakeCtx(success_view())), "## 3. Risks, trade-offs and validation")
    assert (
        "- `RISK-001` Counter race - denial\\_of\\_service, likelihood medium, impact high, "
        "status mitigated; 1 mitigation(s) covering `R-001` (`threat_model@v1`)"
    ) in risks
    assert "- Design risk `HIGH`: schema migration (`design_doc@v1`)" in risks
    assert "- Rollback plan: Revert the migration and the patch (`design_doc@v1`)" in risks
    assert (
        "- Trade-off `ADR-001` Count in the database: chosen over In-memory counter; accepted "
        "consequence: One more write per redirect (`design_doc@v1`)"
    ) in risks
    assert (
        "- `SEC-001` x1: severity HIGH, enforcement deny; nodes `implement.core` (`audit:4`)"
        in risks
    )
    assert (
        "- Waiver `W-0001`: `SEC-006` on `design` accepted by bob: reviewed (`D-0006`, `A-0001`)"
    ) in risks
    assert (
        "- Suite passed: 3 test(s), 3 passed, 0 failed, 0 errors, 0 skipped (`test_report@v1`)"
        in risks
    )
    assert "- Baseline suite ran: yes (`test_report@v1`)" in risks
    assert (
        "every authored test fails on the baseline and passes on the integrated view: yes" in risks
    )
    assert (
        "- Compared with the baseline OpenAPI: checked yes, 0 breaking, 1 additive change(s)"
        in risks
    )
    assert "- Declared contract: 0 path(s), breaking no (`api_contract@v1`)" in risks
    assert "- Scan clean: 12 file(s) scanned, 0 finding(s), 2 rule(s) evaluated" in risks
    assert "- Verdict `GO`, release risk `HIGH` (`readiness_report@v1`)" in risks
    assert "- 1 of 1 readiness check(s) passed (`readiness_report@v1`)" in risks
    assert "- Traceability complete: yes, 0 gap(s) (`traceability_matrix@v1`)" in risks
    assert "- Release verification: applied cleanly yes, tests passed yes" in risks
    assert (
        "- `A-0001` on `design` (node_output, risk `HIGH`): `approved`, 2 distinct approver(s) of "
        "2 required; bob approved via scripted (ok); carol approved via scripted (ok) "
        "(`D-0005`, `D-0006`)"
    ).replace("node_output", "node\\_output") in risks
    assert "four-eyes is ATTESTED-ONLY" in risks


def test_assumptions_section_cites_the_decisions() -> None:
    block = section(render(FakeCtx(success_view())), "## 4. Assumptions")
    assert "- `ASM-001` Clicks are counted per redirect (`requirements@v1`, `D-0003`)" in block
    assert (
        "- `R-002` is deferred: Export analytics as CSV (`requirements@v1`, `D-0002`, `D-0001`)"
    ) in block
    assert (
        "- `Q-unquantified-aaaaaa` answered by carol in round 1: Up to 90 days "
        "(`clarification_answers@v1`, `D-0001`)"
    ) in block


def test_limitations_are_computed_from_the_facts_of_a_success() -> None:
    block = section(render(FakeCtx(success_view())), "## 5. Limitations")
    assert (
        "- Unsupported requirement `R-003`: no capability matches weekly report - the reference "
        "agents cannot deliver it (`requirements@v1`)"
    ) in block
    assert block.count("Unsupported requirement `R-003`") == 1  # requirements and plan agree
    assert (
        "- Not delivered by this run: `R-002` was deferred by a scope decision "
        "(`requirements@v1`, `D-0002`)"
    ) in block
    assert (
        "- Approvals were scripted (assurance `DEMO_SCRIPTED`), not a human review: scripted "
        "votes on `A-0001` (`D-0005`)"
    ) in block
    assert "- Skipped node `codebase-analysis`: greenfield workspace (`audit:3`)" in block
    assert "- Manual steps left: review the patch and apply it to your tree yourself" in block
    assert "(`release_patch@v1`, `release_bundle@v1`)" in block
    assert "- Open item: Announce the change to API users (`readiness_report@v1`)" in block
    assert "four-eyes: ATTESTED-ONLY" in block
    assert "- Work was done by deterministic reference agents (`requirements`)" in block
    assert f"- Final status `succeeded`; audit head seq 6, hash `{'a' * 64}` (`audit:6`)" in block
    assert "not produced" not in block
    assert "Failed node" not in block


def test_every_bullet_cites_an_artifact_a_decision_or_an_audit_event() -> None:
    for view in (success_view(), failed_view(), FakeView()):
        for line in bullets(render(FakeCtx(view))):
            assert CITATION.search(line), line


def test_assurance_is_human_unless_a_vote_was_scripted() -> None:
    human = success_view(approvals=(approval([vote("bob"), vote("carol")]),), events=())
    text = render(FakeCtx(human))
    assert "- Assurance: `HUMAN` (no scripted approval was recorded)" in text
    assert "Approvals were scripted" not in section(text, "## 5. Limitations")
    assert "- Assurance: `DEMO_SCRIPTED` (approvals were scripted, not a human review)" in render(
        FakeCtx(success_view())
    )


def test_a_scripted_vote_in_the_audit_log_alone_is_enough() -> None:
    view = success_view(approvals=())
    assert "Approvals were scripted" in section(render(FakeCtx(view)), "## 5. Limitations")


def test_applied_release_asks_for_a_review_of_the_target_tree() -> None:
    docs = payloads()
    docs["release_bundle"] = bundle_payload("applied")
    view = FakeView(docs, {"release_patch": PATCH_TEXT})
    block = section(render(FakeCtx(view)), "## 5. Limitations")
    assert "the patch was applied to the target tree by the promote node" in block
    assert "apply it to your tree yourself" not in block


def test_checks_that_did_not_run_are_limitations() -> None:
    report = make_report_payload(
        baseline={"ran": False, "passed": None, "failed_tests": []},
        fail_to_pass={"checked": False, "ok": False, "tests": []},
        api_compat={"checked": False, "breaking": [], "additive": []},
    )
    docs = {**payloads(), "test_report": report}
    view = FakeView(docs, params={"perf_budget_ms": 250})
    block = section(render(FakeCtx(view)), "## 5. Limitations")
    for text in (
        "the baseline suite did not run",
        "fail-to-pass was not checked",
        "API compatibility was not compared with the baseline",
        "the performance budget was not checked",
    ):
        assert f"- Check not performed: {text} (`test_report@v1`)" in block


def test_injected_faults_from_the_run_params_are_a_limitation() -> None:
    view = success_view(
        params={"faults": [{"node": "implement.*", "fault": "secret_leak", "attempts": 2}]},
        statuses={"implement.core": NodeStatus.SUCCEEDED, "design": NodeStatus.SUCCEEDED},
        decisions=(
            decision(1, DecisionKind.RETRY, "Retry implement.core", node="implement.core"),
            decision(2, DecisionKind.FALLBACK, "Fallback", node="implement.core"),
        ),
    )
    block = section(render(FakeCtx(view)), "## 5. Limitations")
    assert (
        "- Injected fault `secret_leak` on nodes matching `implement.*` (up to 2 attempt(s)); "
        "affected: `implement.core` (`D-0001`, `D-0002`)"
    ) in block


def test_injected_faults_without_decisions_cite_the_failed_attempts() -> None:
    failure = event(7, ET.NODE_ATTEMPT_FAILED, "implement.core", kind="transient", message="x")
    view = success_view(
        params={"faults": [{"node": "implement.core", "fault": "transient", "attempts": 1}]},
        statuses={"implement.core": NodeStatus.SUCCEEDED},
        decisions=(),
        events=(event(1, ET.NODE_STARTED, "implement.core", agent_id="implementer"), failure),
    )
    block = section(render(FakeCtx(view)), "## 5. Limitations")
    assert (
        "- Injected fault `transient` on nodes matching `implement.core` (up to 1 attempt(s)); "
        "affected: `implement.core` (`audit:7`)"
    ) in block


def test_llm_backed_agents_are_reported_as_such() -> None:
    view = success_view(
        events=(event(1, ET.NODE_STARTED, "requirements", agent_id="llm-requirements"),)
    )
    block = section(render(FakeCtx(view)), "## 5. Limitations")
    assert "- LLM-backed agent(s) took part (`llm-requirements`)" in block
    assert "deterministic reference agents" not in block


# ----------------------------------------------------------------------------- failed run


def failed_ctx() -> FakeCtx:
    return FakeCtx(failed_view(), RunStatus.FAILED, "verify failed", audit_head=(5, "b" * 64))


def test_a_failed_run_still_produces_a_complete_summary() -> None:
    text = render(failed_ctx())
    for heading in HEADINGS:
        assert heading in text
    assert "- Final status: `failed` - verify failed (`audit:5`)" in text
    assert "- Run: `run-1` (`audit:5`)" in text


def test_missing_artifacts_say_which_node_did_not_produce_them_and_why() -> None:
    text = render(failed_ctx())
    assert (
        "- not produced: `test_report`: node `verify` failed - gate\\_rejected: 2 tests failed "
        "(`audit:3`)"
    ) in section(text, "## 3. Risks, trade-offs and validation")
    risks = section(text, "## 3. Risks, trade-offs and validation")
    assert "not produced: `security_report`: node `security-scan` has no recorded status" in risks
    assert "not produced: `readiness_report`: node `release-readiness` was blocked" in risks
    # no node of this (custom) workflow produces these kinds
    assert "- not produced: no node of this run produces `threat_model` (`audit:5`)" in risks
    assert "- not produced: `design_doc`: node `design` has no recorded status" in risks
    assert "- not produced: `design_doc`" in section(text, "## 1. Plan and rationale")
    assert "- not produced: no node of this run produces `release_patch`" in section(
        text, "## 2. Artifacts produced"
    )


def test_without_node_specs_the_standard_workflow_tells_who_produces_a_kind() -> None:
    view = FakeView(
        {"requirements": requirements_payload()},
        statuses={"design": NodeStatus.FAILED, "promote": NodeStatus.CANCELLED},
        events=(event(1, ET.NODE_FAILED, "design", kind="fatal", message="no model"),),
    )
    text = render(FakeCtx(view, RunStatus.FAILED, "x", audit_head=(1, "")))
    assert (
        "- not produced: `design_doc`: node `design` failed - fatal: no model (`audit:1`)" in text
    )
    assert "- not produced: `release_patch`: node `promote` was cancelled (`audit:1`)" in text
    assert "- not produced: `test_report`: node `verify` has no recorded status" in text


def test_a_failed_run_still_shows_what_exists() -> None:
    plan = section(render(failed_ctx()), "## 1. Plan and rationale")
    assert "- `R-001` Links must stop redirecting after max clicks" in plan
    assert "- `T-002` test the click counter" in plan


def test_failed_and_blocked_nodes_are_limitations_with_their_reasons() -> None:
    block = section(render(failed_ctx()), "## 5. Limitations")
    assert "- Failed node `verify`: gate\\_rejected: 2 tests failed (`D-0001`)" in block
    assert "- Blocked node `docs`: dependency verify failed (`audit:4`)" in block
    assert "- Blocked node `release-readiness` (`audit:5`)" in block
    assert "- Manual steps: no patch was produced, so there is nothing to review or apply" in block
    assert "- Final status `failed` - verify failed; audit head seq 5" in block


def test_a_failed_run_without_events_or_decisions_still_renders() -> None:
    view = failed_view(events=(), decisions=())
    text = render(FakeCtx(view, RunStatus.FAILED, "x", audit_head=(0, "")))
    assert "- no decision was recorded (`audit:0`)" in text
    assert "audit head seq 0, hash `-`" in text


# ----------------------------------------------------------------------------- safe-stopped run


def test_a_safe_stopped_run_reports_the_stop_and_the_cancelled_nodes() -> None:
    events = (
        event(1, ET.RUN_CREATED),
        event(2, ET.NODE_STARTED, "implement.core", agent_id="implementer"),
        event(3, ET.SAFE_STOP_TRIGGERED, trigger="kill_switch", reason="operator pressed stop"),
        event(4, ET.NODE_CANCELLED, "implement.core", reason="run safe-stopped"),
        event(5, ET.RUN_SAFE_STOPPED, trigger="kill_switch"),
    )
    view = FakeView(
        {"requirements": requirements_payload()},
        statuses={"implement.core": NodeStatus.CANCELLED, "requirements": NodeStatus.SUCCEEDED},
        decisions=(decision(1, DecisionKind.SAFE_STOP, "Safe-stop: kill switch", Actor.system()),),
        events=events,
    )
    ctx = FakeCtx(view, RunStatus.SAFE_STOPPED, "kill switch", audit_head=(5, "c" * 64))
    text = render(ctx)
    block = section(text, "## 5. Limitations")
    assert (
        "- Safe-stop triggered by `kill_switch`: operator pressed stop (`D-0001`, `audit:3`)"
    ) in block
    assert "- Cancelled node `implement.core`: run safe-stopped" in block
    assert "- Final status `safe_stopped` - kill switch" in block
    assert "- Manual steps: no patch was produced" in block
    assert "- Final status: `safe_stopped` - kill switch (`audit:5`)" in text


# ----------------------------------------------------------------------------- tolerance


def test_an_unreadable_artifact_is_reported_not_raised() -> None:
    errors: dict[str, Exception] = {"test_report": IntegrityError("hash mismatch")}
    view = FakeView(payloads(), {"release_patch": PATCH_TEXT}, errors=errors)
    text = render(FakeCtx(view))
    risks = section(text, "## 3. Risks, trade-offs and validation")
    assert (
        "- `test_report` could not be read: integrity or format check failed (`test_report@v1`)"
    ) in risks
    block = section(text, "## 5. Limitations")
    assert "- Artifact `test_report@v1` could not be read (IntegrityError)" in block
    assert block.count("could not be read (IntegrityError)") == 1  # reported once, not per section


def test_a_payload_of_the_wrong_shape_is_unreadable() -> None:
    docs = {**payloads(), "design_doc": ["not", "an", "object"]}
    text = render(FakeCtx(FakeView(docs)))
    assert "- Artifact `design_doc@v1` could not be read (malformed)" in text


def test_a_view_without_the_optional_accessors_is_tolerated() -> None:
    docs = {"requirements": requirements_payload()}
    ctx = FakeCtx(MinimalView(docs))
    text = render(ctx)
    assert "- `R-001` Links must stop redirecting after max clicks" in text
    assert "| `requirements` | `requirements@v1` |" in text  # from the fallback over known kinds
    assert "- no waiver was granted" in text


def test_an_empty_view_still_gives_the_five_sections() -> None:
    text = render(FakeCtx(FakeView(), RunStatus.FAILED, "", audit_head=(0, "")))
    for heading in HEADINGS:
        assert heading in text
    assert "- no artifact is active: the run produced nothing (`audit:0`)" in text
    assert "- Agents used: unknown, no node start was recorded" in text


# ----------------------------------------------------------------------------- safety


def hostile_view() -> FakeView:
    docs = payloads()
    docs["requirements"]["requirements"][0]["text"] = HOSTILE
    docs["design_doc"]["adrs"][0]["title"] = HOSTILE
    docs["design_doc"]["adrs"][0]["decision"] = HOSTILE
    docs["threat_model"]["risks"][0]["title"] = HOSTILE
    docs["security_report"] = {
        "schema": 1,
        "rules_evaluated": [],
        "findings": [
            {"rule_id": "SEC-001", "category": "security", "severity": "HIGH", "message": HOSTILE}
        ],
        "files_scanned": 1,
        "passed": False,
    }
    docs["release_bundle"] = bundle_payload(files=["a.py\n```\n# Injected heading\n```", HOSTILE])
    docs["readiness_report"]["open_items"] = [HOSTILE]
    return success_view_with(docs)


def success_view_with(docs: dict[str, Any]) -> FakeView:
    base = success_view()
    return FakeView(
        docs,
        {"release_patch": PATCH_TEXT},
        decisions=(decision(1, DecisionKind.DESIGN, HOSTILE, Actor.agent("x"), "design"),),
        approvals=(approval([vote(HOSTILE, HOSTILE, HOSTILE)]),),
        waivers=(
            Waiver(
                "W-0001",
                "SEC-006",
                "k",
                "design",
                "e" * 64,
                HOSTILE,
                (HOSTILE,),
                "A-0001",
                "D-0001",
                T0,
            ),
        ),
        statuses={"verify": NodeStatus.FAILED},
        events=(event(1, ET.NODE_FAILED, "verify", kind="x", message=HOSTILE),),
        params=base.params(),
    )


def test_hostile_content_cannot_inject_markup_links_html_headings_or_terminal_codes() -> None:
    text = render(FakeCtx(hostile_view(), RunStatus.FAILED, HOSTILE))
    visible = "\n".join(outside_fences(text))
    spans_removed = re.sub(r"(?<!\\)`[A-Za-z0-9._@:+*?-]*`", "", visible)
    assert "\x1b" not in text
    assert "\u202e" not in text
    assert not re.search(r"(?<!\\)<", spans_removed)
    assert not re.search(r"(?<!\\)\[(?!truncated )", spans_removed)
    assert "](" not in spans_removed
    assert "://" not in spans_removed
    assert "www." not in spans_removed.lower()
    assert not re.search(r"(?<!\\)@", spans_removed)
    # the only headings are ours; the injected one stays inside its fence
    assert [ln for ln in outside_fences(text) if ln.startswith("# ")] == ["# Engineering summary"]
    assert [ln for ln in outside_fences(text) if ln.startswith("## ")] == HEADINGS[1:]
    assert "# Injected heading" not in outside_fences(text)
    assert "# Injected heading" in text


def test_the_patch_file_fence_cannot_be_closed_from_the_inside() -> None:
    text = render(FakeCtx(hostile_view()))
    opening = [ln for ln in text.splitlines() if ln.startswith("````")]
    assert len(opening) == 2  # a fence longer than the longest backtick run inside the list


def test_untrusted_text_is_one_line() -> None:
    for line in render(FakeCtx(hostile_view())).splitlines():
        assert "\r" not in line


# --------------------------------------------------------------------- bounds and determinism


def test_the_summary_is_deterministic_and_has_no_times_or_paths(tmp_path: Path) -> None:
    ctx = FakeCtx(success_view(), run_dir=tmp_path)
    first = render(ctx)
    assert render(FakeCtx(success_view(), run_dir=tmp_path)) == first
    assert not re.search(r"\d{4}-\d{2}-\d{2}", first)  # no dates
    assert not re.search(r"\d{2}:\d{2}:\d{2}", first)  # no clock times
    assert not re.search(r"\b\d+(\.\d+)?\s?(ms|s|sec|seconds)\b", first)  # no durations
    assert str(tmp_path) not in first
    assert tmp_path.name not in first


def test_long_lists_are_capped_and_the_rest_is_counted() -> None:
    payload = requirements_payload()
    template = payload["requirements"][0]
    payload["requirements"] = [
        {**template, "id": f"R-{n:03d}", "text": f"requirement {n}"} for n in range(1, 201)
    ]
    decisions = tuple(decision(n, DecisionKind.DESIGN, f"choice {n}") for n in range(1, 151))
    view = FakeView({"requirements": payload}, decisions=decisions)
    text = render(FakeCtx(view))
    assert "- `R-080` requirement 80" in text
    assert "`R-081`" not in text
    assert "- ... 120 more requirements not listed (`requirements@v1`)" in text
    assert "- `D-0080` " in text
    assert "`D-0081`" not in text
    assert "- ... 70 more decisions not listed (`audit:6`)" in text
    assert len(text) < 200_000


def test_the_finalizer_publishes_one_markdown_artifact_draft() -> None:
    ctx = FakeCtx(success_view())
    finalize(ctx)
    assert len(ctx.published) == 1
    draft = ctx.published[0]
    assert (draft.kind, draft.media_type) == ("engineering_summary", "text/markdown")
    assert isinstance(draft.content, str)
    assert draft.content == render(ctx)
    json.dumps(draft.content)  # plain text


@pytest.mark.parametrize(
    ("status", "view"),
    [
        (RunStatus.SUCCEEDED, success_view()),
        (RunStatus.FAILED, failed_view()),
        (RunStatus.SAFE_STOPPED, FakeView()),
    ],
)
def test_it_runs_for_every_terminal_state(status: RunStatus, view: FakeView) -> None:
    ctx = FakeCtx(view, status)
    finalize(ctx)
    assert len(ctx.published) == 1
    assert f"- Final status `{status.value}`" in ctx.published[0].content


# ----------------------------------------------------------------- the real scheduler types


def test_it_works_on_a_real_run_snapshot_and_finalizer_context(tmp_path: Path) -> None:
    """The scheduler's own ``RunSnapshot`` and ``FinalizerContext`` over real stores."""
    clock = FakeClock()
    store = ArtifactStore(tmp_path / "artifacts", clock=clock)
    store.put(
        ArtifactDraft(kind="requirements", content=requirements_payload()),
        producer="requirements",
        attempt=1,
        agent_id="requirements",
    )
    log = DecisionLog(tmp_path / "decisions.jsonl", run_id="run-1", clock=clock)
    log.record(
        DecisionKind.DESIGN,
        "Count clicks in the database",
        actor=Actor.agent("architect"),
        node_id="design",
    )
    failure = event(1, ET.NODE_FAILED, "verify", kind="gate_rejected", message="boom")
    snapshot = RunSnapshot(
        run_id="run-1",
        status=RunStatus.FAILED,
        reason="verify failed",
        params={},
        specs={"verify": spec("verify", ("test_report",))},
        ancestors={"verify": frozenset()},
        mandatory=(),
        statuses={"requirements": NodeStatus.SUCCEEDED, "verify": NodeStatus.FAILED},
        approvals=[],
        waivers=[],
        artifacts=store.all(include_inactive=True),
        decision_log=log,
        decision_count=len(log),
        events_fn=lambda: [failure],
        event_count=1,
        store=store,
    )
    published: list[ArtifactDraft] = []

    def publish(draft: ArtifactDraft) -> Artifact:
        published.append(draft)
        return artifact(draft.kind)

    def record(*args: Any, **kwargs: Any) -> Decision:
        raise AssertionError("the summary must not record decisions")

    ctx = FinalizerContext(
        run_id="run-1",
        status=RunStatus.FAILED,
        reason="verify failed",
        view=snapshot,
        run_dir=tmp_path,
        audit_head=(1, "d" * 64),
        publish_artifact=publish,
        record_decision=record,
    )
    engineering_summary_finalizer(ctx)
    assert len(published) == 1
    text = published[0].content
    assert isinstance(text, str)
    assert "- `R-001` Links must stop redirecting after max clicks" in text
    assert (
        "- `D-0001` `design` by `agent:architect` on `design`: Count clicks in the database" in text
    )
    assert (
        "- not produced: `test_report`: node `verify` failed - gate\\_rejected: boom (`audit:1`)"
    ) in text
    assert "| `requirements` | `requirements@v1` | `requirements` |" in text
