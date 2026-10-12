"""Tests for the release manager agent (``release-readiness``): readiness report + matrix.

The node runs through the shared delivery rig (real artifact store, a pinned reader that serves only
the kinds the node declares, real toolbox and policy engine, fake clock); governance (human
decisions, approvals, waivers) comes from ``FakeRunView``.  The traceability rules are compared with
the *real* CMP-001 rule, the approval rules with CMP-003 and the change reference / rollback plan
with CHG-005: a verdict of ``GO`` must never be one the node's own exit gates would refuse.
"""

from __future__ import annotations

import dataclasses
import json
import unicodedata
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.release_manager import (
    MAX_HUMAN_PARENTS,
    MAX_OPEN_ITEMS,
    MAX_PROVENANCE_NODES,
    ReleaseManagerAgent,
)
from orchestrator.core.errors import ToolDenied, UndeclaredInputError
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    Actor,
    Agent,
    AgentResult,
    ApprovalRequest,
    ApprovalSpec,
    ApprovalStatus,
    ArtifactStatus,
    Decision,
    DecisionDraft,
    DecisionKind,
    NodeSpec,
    RiskLevel,
    Stage,
    Waiver,
)
from orchestrator.policies.rules_change import ReleaseDocumentation
from orchestrator.policies.rules_compliance import DecisionCompleteness, Traceability
from tests.orchestrator.agents.delivery_support import (
    Rig,
    assert_valid,
    build_test_report,
    content_hashes,
    design_doc_payload,
    integration_report_payload,
    payload,
    plain_writes,
    report_row,
    requirement,
    requirements_payload,
    security_report_payload,
    task,
    task_plan_payload,
)
from tests.orchestrator.policies.policy_builders import (
    T0,
    FakeRunView,
    approval,
    make_finding,
    node,
    release_chain,
    run_subject,
    waiver_for,
)
from tests.orchestrator.policies.policy_builders import check as run_rule

NODE = "release-readiness"
INPUT_KINDS = (
    Kind.REQUIREMENTS,
    Kind.DESIGN_DOC,
    Kind.TASK_PLAN,
    Kind.INTEGRATED_CHANGESET,
    Kind.DOCS_CHANGESET,
    Kind.INTEGRATION_REPORT,
    Kind.TEST_REPORT,
    Kind.SECURITY_REPORT,
)
GREEN_CHECKS = [
    "tests_passed",
    "fail_to_pass",
    "api_compat",
    "security_clean",
    "docs_present",
    "no_conflicts",
    "traceability_complete",
    "rollback_plan",
    "change_reference",
    "approvals_recorded",
]
SCOPE = DecisionKind.SCOPE
CLARIFY = DecisionKind.CLARIFICATION
NO_APPROVAL = "no approval request is recorded"


@pytest.fixture(autouse=True)
def _fast_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    plain_writes(monkeypatch)


# ----------------------------------------------------------------------------- inputs and rig


@dataclasses.dataclass
class Inputs:
    """The payloads of one release-readiness context (mutate them, then publish)."""

    requirements: dict[str, Any]
    design: dict[str, Any]
    plan: dict[str, Any]
    tests: dict[str, Any]
    security: dict[str, Any]
    integration: dict[str, Any]
    integrated: dict[str, str | None]
    docs: dict[str, str | None]


def cover(rid: str, n: int = 2, outcome: str = "passed") -> list[dict[str, Any]]:
    """One test row per criterion of ``rid``; the first one also cites the requirement id."""
    number = int(rid.split("-")[1])
    return [
        report_row(
            f"tests/test_r{number:03d}.py::test_ac{j}",
            outcome,
            covers=[f"AC-{number:03d}-{j}", *([rid] if j == 1 else [])],
        )
        for j in range(1, n + 1)
    ]


def green() -> Inputs:
    """A complete, consistent context: R-001 (two criteria) covered by passed tests, all clean."""
    return Inputs(
        requirements=requirements_payload(requirement("R-001")),
        design=design_doc_payload(),
        plan=task_plan_payload(task("T-001", requirement_ids=["R-001"])),
        tests=build_test_report(cover("R-001")),
        security=security_report_payload(),
        integration=integration_report_payload(merged_paths=["widgets/core.py"]),
        integrated={"widgets/core.py": "VALUE = 1\n"},
        docs={"docs/widgets.md": "# Widgets\n"},
    )


def finding(severity: str, rule_id: str = "SEC-001") -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "category": "security",
        "severity": severity,
        "message": "a finding",
        "location": "widgets/core.py:1",
    }


def decision(
    did: str,
    kind: DecisionKind = CLARIFY,
    *,
    actor: Actor | None = None,
    refs: Sequence[str] = (),
    parents: Sequence[str] = (),
) -> Decision:
    """A decision made by a human (default) or by ``actor``."""
    return Decision(
        id=did,
        run_id="run-1",
        ts=T0,
        kind=kind,
        actor=actor or Actor.human("alice"),
        node_id=None,
        summary="s",
        refs=tuple(refs),
        parents=tuple(parents),
    )


def by_agent(
    did: str,
    kind: DecisionKind = SCOPE,
    *,
    refs: Sequence[str] = (),
    parents: Sequence[str] = (),
) -> Decision:
    return decision(did, kind, actor=Actor.agent("requirements"), refs=refs, parents=parents)


class GovView(FakeRunView):
    """``FakeRunView`` that also takes ready-made decisions (for example with parents)."""

    def add_raw(self, *decisions: Decision) -> GovView:
        self._decisions.extend(decisions)
        return self


def governed(*decisions: Decision) -> GovView:
    return GovView().add_raw(*decisions)


@dataclasses.dataclass
class Release:
    """What one run of the agent produced, decoded."""

    result: AgentResult
    readiness: dict[str, Any]
    matrix: dict[str, Any]

    @property
    def verdict(self) -> str:
        return str(self.readiness["verdict"])

    @property
    def failing(self) -> list[str]:
        return [c["id"] for c in self.readiness["checks"] if not c["passed"]]

    @property
    def decision(self) -> DecisionDraft:
        assert len(self.result.decisions) == 1, "exactly one decision draft is expected"
        return self.result.decisions[0]

    def check(self, check_id: str) -> dict[str, Any]:
        found = [c for c in self.readiness["checks"] if c["id"] == check_id]
        assert len(found) == 1, f"expected exactly one {check_id!r} check, got {len(found)}"
        return found[0]

    def row(self, rid: str) -> dict[str, Any]:
        found = [r for r in self.matrix["requirements"] if r["id"] == rid]
        assert len(found) == 1, f"expected exactly one matrix row for {rid}, got {len(found)}"
        return found[0]


class Harness:
    """A rig with the inputs published, the way the engine hands them to the node."""

    def __init__(
        self,
        root: Path,
        inp: Inputs | None = None,
        *,
        clock_start: datetime | None = None,
        kinds: Sequence[str] = INPUT_KINDS,
    ) -> None:
        self.rig = Rig(root, clock_start=clock_start)
        self.inp = inp if inp is not None else green()
        self.publish(*kinds)

    def _content(self, kind: str) -> Any:
        inp = self.inp
        return {
            Kind.REQUIREMENTS: inp.requirements,
            Kind.DESIGN_DOC: inp.design,
            Kind.TASK_PLAN: inp.plan,
            Kind.TEST_REPORT: inp.tests,
            Kind.SECURITY_REPORT: inp.security,
            Kind.INTEGRATION_REPORT: inp.integration,
            Kind.INTEGRATED_CHANGESET: inp.integrated,
            Kind.DOCS_CHANGESET: inp.docs,
        }[kind]

    def publish(self, *kinds: str) -> None:
        """(Re)publish ``kinds`` from the current ``inp``; a new version supersedes the old."""
        for kind in kinds:
            content = self._content(kind)
            if kind in (Kind.INTEGRATED_CHANGESET, Kind.DOCS_CHANGESET):
                self.rig.publish_changeset(kind, content)
            else:
                self.rig.publish(kind, content)

    def run(self, view: FakeRunView | None = None) -> Release:
        result = self.rig.run(ReleaseManagerAgent(), NODE, run_view=view)
        return Release(
            result,
            payload(result, Kind.READINESS_REPORT),
            payload(result, Kind.TRACEABILITY_MATRIX),
        )


def run_release(
    tmp_path: Path, inp: Inputs | None = None, view: FakeRunView | None = None
) -> Release:
    return Harness(tmp_path, inp).run(view)


# ----------------------------------------------------------------------------- the GO path


def test_green_inputs_give_go_with_every_check_passing(tmp_path: Path) -> None:
    out = run_release(tmp_path)
    checks = out.readiness["checks"]
    assert out.verdict == "GO"
    assert [c["id"] for c in checks] == GREEN_CHECKS
    assert all(c["passed"] and c["evidence"] and c["title"] and c["detail"] for c in checks)
    assert out.readiness["open_items"] == []
    assert out.matrix["complete"] is True and out.matrix["gaps"] == []
    assert out.check("approvals_recorded")["evidence"] == [NO_APPROVAL]
    assert out.result.rationale == "GO: 0 failing check(s), 0 gap(s)."


def test_go_copies_risk_rollback_plan_and_change_reference_from_the_design(tmp_path: Path) -> None:
    inp = green()
    inp.design = design_doc_payload(
        risk="MEDIUM",
        rollback_plan="Revert the patch, then run the down migration.",
        change_reference="CR-4f2a9c",
    )
    inp.design["risk"]["reasons"] = ["touches the public API", "adds a table"]
    out = run_release(tmp_path, inp)
    assert out.verdict == "GO"
    assert out.readiness["risk"] == {
        "level": "MEDIUM",
        "reasons": ["touches the public API", "adds a table"],
    }
    assert out.readiness["rollback_plan"] == "Revert the patch, then run the down migration."
    assert out.readiness["change_reference"] == "CR-4f2a9c"
    assert out.check("change_reference")["evidence"] == ["design_doc.change_reference: CR-4f2a9c"]


def test_go_matrix_is_complete_with_tests_results_tasks_files_and_decision_refs(
    tmp_path: Path,
) -> None:
    inp = green()
    inp.requirements = requirements_payload(requirement("R-001"), requirement("R-002", criteria=1))
    inp.plan = task_plan_payload(
        task("T-001", requirement_ids=["R-001"], expected_paths=["widgets/core.py", "gone.py"]),
        task("T-002", requirement_ids=["R-001"], expected_paths=["widgets/extra.py"]),
        task("T-003", requirement_ids=["R-002"], expected_paths=["widgets/other.py"]),
    )
    inp.integrated = {"widgets/core.py": "A = 1\n", "widgets/extra.py": "B = 1\n"}
    inp.integrated["widgets/other.py"] = "C = 1\n"
    inp.tests = build_test_report([*cover("R-001"), *cover("R-002", 1)])
    view = governed(
        decision("D-0001", DecisionKind.DESIGN, refs=["AC-001-2"]),
        by_agent("D-0002", DecisionKind.ASSUMPTION, refs=["R-001"]),
        decision("D-0003", DecisionKind.DESIGN, refs=["R-009"]),
    )
    out = run_release(tmp_path, inp, view)
    assert out.verdict == "GO" and out.matrix["complete"] is True and out.matrix["gaps"] == []
    assert out.row("R-001") == {
        "id": "R-001",
        "status": "active",
        "acceptance_criteria": [
            {
                "id": "AC-001-1",
                "tests": ["tests/test_r001.py::test_ac1"],
                "results": ["passed"],
                "covered": True,
            },
            {
                "id": "AC-001-2",
                "tests": ["tests/test_r001.py::test_ac2"],
                "results": ["passed"],
                "covered": True,
            },
        ],
        "tasks": ["T-001", "T-002"],
        "files": ["tests/test_r001.py", "widgets/core.py", "widgets/extra.py"],
        "decision_refs": ["D-0001", "D-0002"],
    }
    row = out.row("R-002")
    assert row["tasks"] == ["T-003"] and row["decision_refs"] == []
    assert row["files"] == ["tests/test_r002.py", "widgets/other.py"]  # 'gone.py' was not changed


def test_go_records_one_release_decision_that_cites_nothing(tmp_path: Path) -> None:
    out = run_release(tmp_path)
    draft = out.decision
    assert draft.kind is DecisionKind.RELEASE and draft.outcome == "GO"
    assert draft.summary == "Release readiness GO: 10 of 10 checks passed"
    assert draft.rationale == "every readiness check passed"
    assert draft.parents == () and draft.refs == () and dict(draft.data) == {"failed_checks": []}


def test_both_artifacts_validate_against_their_schemas(tmp_path: Path) -> None:
    broken = green()
    broken.tests = build_test_report(cover("R-001", 1))
    broken.security = security_report_payload([finding("CRITICAL")])
    for name, inp in (("go", None), ("no-go", broken)):
        out = run_release(tmp_path / name, inp)
        assert_valid(out.result)
        assert {d.kind for d in out.result.artifacts} == {
            Kind.READINESS_REPORT,
            Kind.TRACEABILITY_MATRIX,
        }
        assert out.readiness["schema"] == 1 and out.matrix["schema"] == 1
    assert out.verdict == "NO_GO"


def test_identity_autonomy_and_a_read_only_toolbox(tmp_path: Path) -> None:
    agent = ReleaseManagerAgent()
    assert isinstance(agent, Agent)
    assert (agent.agent_id, agent.role, agent.version) == (
        "release-manager",
        "release-manager",
        "1",
    )
    h = Harness(tmp_path)
    profile = agent.autonomy.narrow(h.rig.node(NODE).autonomy)
    assert profile.write_globs == () and profile.can_execute is False
    with pytest.raises(ToolDenied):
        h.rig.context(agent, NODE).toolbox.write_text("notes.txt", "x")
    usage = agent.run(h.rig.context(agent, NODE)).usage
    assert usage["tool_calls"] == 0 and usage["bytes_written"] == 0


# ----------------------------------------------------------------------------- failing inputs


def _failed_test(inp: Inputs) -> None:
    boom = report_row("tests/test_x.py::test_boom", "failed")
    inp.tests = build_test_report([*cover("R-001"), boom])


def _no_tests(inp: Inputs) -> None:
    inp.tests = build_test_report([])
    inp.tests["passed"] = True  # a summary flag that lies about an empty run


def _lying_flag(inp: Inputs) -> None:
    _failed_test(inp)
    inp.tests["passed"] = True  # the flag says passed although a row failed


def _f2p_unchecked(inp: Inputs) -> None:
    inp.tests = build_test_report(cover("R-001"), fail_to_pass_checked=False)


def _f2p_not_ok(inp: Inputs) -> None:
    inp.tests = build_test_report(cover("R-001"), fail_to_pass_ok=False)


def _high(inp: Inputs) -> None:
    inp.security = security_report_payload([finding("HIGH")])


def _critical(inp: Inputs) -> None:
    inp.security = security_report_payload([finding("CRITICAL")])


def _clean_flag_lies(inp: Inputs) -> None:
    _high(inp)
    inp.security["passed"] = True


def _no_docs(inp: Inputs) -> None:
    inp.docs = {}


def _conflict(inp: Inputs) -> None:
    clash = {"path": "widgets/core.py", "authors": ["implement.a", "implement.b"], "reason": "both"}
    inp.integration = integration_report_payload(conflicts=[clash], merged_paths=[])


def _no_rollback(inp: Inputs) -> None:
    inp.design = design_doc_payload(rollback_plan="")


def _no_reference(inp: Inputs) -> None:
    inp.design = design_doc_payload(change_reference="")


FLIPS = [
    pytest.param(_failed_test, {"tests_passed"}, ("tests_passed", "1 failed"), id="test-failed"),
    pytest.param(
        _no_tests,
        {"tests_passed", "traceability_complete"},
        ("tests_passed", "0 failed"),
        id="no-test-ran",
    ),
    pytest.param(
        _lying_flag, {"tests_passed"}, ("tests_passed", "1 failed"), id="flag-passed-row-failed"
    ),
    pytest.param(
        _f2p_unchecked, {"fail_to_pass"}, ("fail_to_pass", "not checked"), id="f2p-not-checked"
    ),
    pytest.param(_f2p_not_ok, {"fail_to_pass"}, ("fail_to_pass", "checked"), id="f2p-not-ok"),
    pytest.param(_high, {"security_clean"}, ("security_clean", "1 HIGH"), id="security-high"),
    pytest.param(
        _critical, {"security_clean"}, ("security_clean", "1 HIGH"), id="security-critical"
    ),
    pytest.param(
        _clean_flag_lies, {"security_clean"}, ("security_clean", "1 HIGH"), id="flag-clean-high"
    ),
    pytest.param(_no_docs, {"docs_present"}, ("docs_present", "0 documentation"), id="no-docs"),
    pytest.param(_conflict, {"no_conflicts"}, ("no_conflicts", "1 conflict"), id="conflict"),
    pytest.param(_no_rollback, {"rollback_plan"}, ("rollback_plan", "empty"), id="no-rollback"),
    pytest.param(
        _no_reference, {"change_reference"}, ("change_reference", "empty"), id="no-reference"
    ),
]


@pytest.mark.parametrize(("mutate", "expected", "detail"), FLIPS)
def test_each_failing_input_flips_the_verdict_and_names_the_check(
    tmp_path: Path,
    mutate: Callable[[Inputs], None],
    expected: set[str],
    detail: tuple[str, str],
) -> None:
    inp = green()
    mutate(inp)
    out = run_release(tmp_path, inp)
    assert out.verdict == "NO_GO"
    assert set(out.failing) == expected
    for check_id in expected:
        assert any(i.startswith(f"{check_id}: ") for i in out.readiness["open_items"])
    assert detail[1] in out.check(detail[0])["detail"]
    assert out.decision.outcome == "NO_GO"
    assert dict(out.decision.data) == {"failed_checks": out.failing}
    assert out.decision.rationale == "; ".join(out.failing)


def test_findings_below_high_do_not_flip_the_verdict(tmp_path: Path) -> None:
    inp = green()
    inp.security = security_report_payload([finding("MEDIUM"), finding("LOW"), finding("INFO")])
    out = run_release(tmp_path, inp)
    assert out.verdict == "GO"
    assert out.check("security_clean")["evidence"] == ["2 file(s) scanned, 3 finding(s)"]


def test_the_evidence_of_a_check_is_bounded(tmp_path: Path) -> None:
    inp = green()
    inp.docs = {f"docs/page{n:02d}.md": "# page\n" for n in range(15)}
    out = run_release(tmp_path, inp)
    got = out.check("docs_present")
    assert out.verdict == "GO" and got["detail"] == "15 documentation file(s)"
    assert got["evidence"] == [f"docs/page{n:02d}.md" for n in range(10)]


def test_a_missing_declared_input_raises_instead_of_producing_a_verdict(tmp_path: Path) -> None:
    kinds = [k for k in INPUT_KINDS if k != Kind.SECURITY_REPORT]
    h = Harness(tmp_path, kinds=kinds)
    with pytest.raises(KeyError, match="security_report"):
        h.run()


def test_undeclared_kinds_are_invisible_to_the_node(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    before = content_hashes(h.run().result)
    h.rig.publish(Kind.API_CONTRACT, {"schema": 1, "paths": [], "breaking": True})
    with pytest.raises(UndeclaredInputError):
        h.rig.context(ReleaseManagerAgent(), NODE).json(Kind.API_CONTRACT)
    assert content_hashes(h.run().result) == before


# ----------------------------------------------------------------------------- api / perf


def test_api_compat_follows_the_openapi_diff_and_the_declared_design(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    added = {"method": "GET", "path": "/widgets", "change": "add", "breaking": False}
    removed = {"method": "DELETE", "path": "/widgets/{id}", "change": "remove", "breaking": True}
    gone = ["DELETE /widgets/{id}"]
    cases: list[tuple[str, bool, list[str], list[dict[str, Any]], bool, str]] = [
        ("diff checked and clean", True, [], [], True, "0 breaking change(s), 0 additive"),
        ("breaking, not declared", True, gone, [], False, "1 breaking change(s), 0 additive"),
        ("breaking, declared", True, gone, [removed], True, "declared in the design"),
        ("breaking, only additive declared", True, gone, [added], False, "1 breaking change(s)"),
        ("not checked, API change declared", False, [], [added], False, "was not verified"),
        ("not checked, none declared", False, [], [], True, "declares no API changes"),
    ]
    for name, checked, breaking, declared, ok, detail in cases:
        h.inp.tests = build_test_report(cover("R-001"), api_checked=checked, api_breaking=breaking)
        h.inp.design = design_doc_payload(api_changes=declared)
        h.publish(Kind.TEST_REPORT, Kind.DESIGN_DOC)
        out = h.run()
        assert out.check("api_compat")["passed"] is ok, name
        assert detail in out.check("api_compat")["detail"], name
        assert (out.verdict == "GO") is ok, name


def test_perf_budget_check_exists_only_when_a_budget_was_checked(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    assert "perf_budget" not in [c["id"] for c in h.run().readiness["checks"]]
    for ok, passed in ((True, True), (False, False), (None, False)):
        h.inp.tests["perf"] = {"checked": True, "budget_ms": 250.0, "ok": ok}
        h.publish(Kind.TEST_REPORT)
        out = h.run()
        assert out.check("perf_budget")["passed"] is passed
        assert out.check("perf_budget")["evidence"] == ["budget 250.0 ms"]
        assert out.failing == ([] if passed else ["perf_budget"])


# ----------------------------------------------------------------------------- approvals


def approvals_of(
    h: Harness,
    *requests: ApprovalRequest,
    waivers: Sequence[Waiver] = (),
    decisions: Sequence[DecisionKind] = (),
) -> dict[str, Any]:
    view = GovView()
    for request in requests:
        view.add_approval(request)
    for waiver in waivers:
        view.add_waiver(waiver)
    for kind in decisions:
        view.add_decision(kind)
    return h.run(view).check("approvals_recorded")


def test_no_recorded_approval_passes_with_an_explicit_note(tmp_path: Path) -> None:
    got = approvals_of(Harness(tmp_path))  # completeness against the node specs is CMP-003's job
    assert got["passed"] is True and got["evidence"] == [NO_APPROVAL]


def test_a_current_and_sufficient_approval_is_evidence(tmp_path: Path) -> None:
    got = approvals_of(Harness(tmp_path), approval("design", voters=("alice", "bob"), required=2))
    assert got["passed"] is True
    assert got["evidence"] == ["design: A-0001 approved by 2"]
    assert got["detail"] == "1 approval(s) and waiver(s) verified"


def test_a_latest_request_that_is_not_approved_fails(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    for status in (ApprovalStatus.PENDING, ApprovalStatus.REJECTED, ApprovalStatus.INVALIDATED):
        got = approvals_of(h, approval("design", status=status, voters=()))
        assert got["passed"] is False, status
        assert f"design: latest request A-0001 is {status.value}" in got["detail"]
    both = approvals_of(
        h,
        approval("verify", request_id="A-0002", status=ApprovalStatus.PENDING, voters=()),
        approval("design", status=ApprovalStatus.REJECTED, voters=()),
    )
    assert both["detail"] == (
        "design: latest request A-0001 is rejected; verify: latest request A-0002 is pending"
    )


def test_only_the_latest_non_cancelled_request_of_a_node_counts(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    old = approval("design", request_id="A-0001")
    pending = approval(
        "design", request_id="A-0002", status=ApprovalStatus.PENDING, voters=(), created_offset=5
    )
    assert approvals_of(h, old, pending)["passed"] is False  # the old approval is stale
    invalidated = approval(
        "design", request_id="A-0001", status=ApprovalStatus.INVALIDATED, voters=()
    )
    recovered = approval("design", request_id="A-0002", created_offset=5)
    assert approvals_of(h, invalidated, recovered)["passed"] is True
    assert approvals_of(h, recovered, invalidated)["passed"] is True  # recording order is moot
    cancelled = approval(
        "design", request_id="A-0003", status=ApprovalStatus.CANCELLED, voters=(), created_offset=9
    )
    kept = approvals_of(h, old, cancelled)
    assert kept["passed"] is True and kept["evidence"] == ["design: A-0001 approved by 1"]
    only = approvals_of(h, cancelled)
    assert only["passed"] is True and only["evidence"] == [NO_APPROVAL]


def test_approvals_need_enough_distinct_approvers(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    few = approvals_of(h, approval("design", voters=("alice",), required=2))
    assert few["passed"] is False and "design: 1 approval(s) of 2 required" in few["detail"]
    same = approvals_of(h, approval("design", voters=("alice", "alice"), required=2))
    assert same["passed"] is False and "1 approval(s) of 2 required" in same["detail"]
    assert approvals_of(h, approval("design", voters=("alice", "bob"), required=2))["passed"]


def test_high_risk_needs_two_required_approvers(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    weak = approvals_of(h, approval("design", risk=RiskLevel.HIGH, required=1))
    assert weak["passed"] is False
    assert "design: HIGH risk with fewer than two required approvers" in weak["detail"]
    strong = approval("design", risk=RiskLevel.HIGH, required=2, voters=("alice", "bob"))
    assert approvals_of(h, strong)["passed"] is True


def test_the_nodes_own_requests_are_ignored(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    own = approval(NODE, request_id="A-0009", status=ApprovalStatus.PENDING, voters=())
    alone = approvals_of(h, own)
    assert alone["passed"] is True and alone["evidence"] == [NO_APPROVAL]
    mixed = approvals_of(h, own, approval("design"))
    assert mixed["passed"] is True and mixed["evidence"] == ["design: A-0001 approved by 1"]


def test_waivers_need_a_waiver_decision_and_an_approved_request(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    waiver = waiver_for(make_finding(), node_id="verify", evidence="ev-1")  # A-0001 / D-0001
    granted = approval("verify")
    backed = approvals_of(h, granted, waivers=[waiver], decisions=[DecisionKind.WAIVER])
    assert backed["passed"] is True
    assert f"waiver {waiver.id}: decision D-0001" in backed["evidence"]
    broken = {
        "no WAIVER decision": approvals_of(h, granted, waivers=[waiver]),
        "a decision of another kind": approvals_of(
            h, granted, waivers=[waiver], decisions=[DecisionKind.DESIGN]
        ),
        "no such approved request": approvals_of(
            h,
            approval("verify", request_id="A-0002"),
            waivers=[waiver],
            decisions=[DecisionKind.WAIVER],
        ),
        "request not approved": approvals_of(
            h,
            approval("verify", status=ApprovalStatus.PENDING, voters=()),
            waivers=[waiver],
            decisions=[DecisionKind.WAIVER],
        ),
    }
    for name, got in broken.items():
        assert got["passed"] is False, name
        assert (
            f"waiver {waiver.id} is not backed by a WAIVER decision and an approval"
            in got["detail"]
        ), name


def approval_chain() -> list[NodeSpec]:
    """requirements -> design (approval spec) -> the gated node, as CMP-003 sees a run."""
    return [
        node("requirements", produces=("requirements",), stage=Stage.REQUIREMENTS),
        node(
            "design",
            depends_on=("requirements",),
            produces=("design_doc",),
            stage=Stage.DESIGN,
            approval=ApprovalSpec(always=True),
        ),
        node(
            "release-readiness",
            depends_on=("design",),
            produces=("readiness_report",),
            stage=Stage.RELEASE,
        ),
    ]


def test_approvals_check_agrees_with_the_real_cmp003_rule(tmp_path: Path) -> None:
    """Where both look at the same request the agent never says 'fine' when CMP-003 refuses."""
    h = Harness(tmp_path)
    pending = ApprovalStatus.PENDING
    waiver = waiver_for(make_finding(), node_id="design", evidence="ev-1")
    waive = DecisionKind.WAIVER
    cases: list[tuple[str, list[ApprovalRequest], list[Waiver], list[DecisionKind], bool]] = [
        ("approved", [approval("design")], [], [], False),
        ("pending", [approval("design", status=pending, voters=())], [], [], True),
        (
            "rejected",
            [approval("design", status=ApprovalStatus.REJECTED, voters=())],
            [],
            [],
            True,
        ),
        (
            "invalidated",
            [approval("design", status=ApprovalStatus.INVALIDATED, voters=())],
            [],
            [],
            True,
        ),
        (
            "too few distinct voters",
            [approval("design", voters=("a", "a"), required=2)],
            [],
            [],
            True,
        ),
        ("high risk, one approver", [approval("design", risk=RiskLevel.HIGH)], [], [], True),
        (
            "high risk, two approvers",
            [approval("design", risk=RiskLevel.HIGH, required=2, voters=("a", "b"))],
            [],
            [],
            False,
        ),
        (
            "stale: a newer request is pending",
            [
                approval("design"),
                approval(
                    "design", request_id="A-0002", status=pending, voters=(), created_offset=5
                ),
            ],
            [],
            [],
            True,
        ),
        (
            "a newer approved request recovers",
            [
                approval("design", status=ApprovalStatus.INVALIDATED, voters=()),
                approval("design", request_id="A-0002", created_offset=5),
            ],
            [],
            [],
            False,
        ),
        (
            "cancelled requests are ignored",
            [
                approval("design"),
                approval(
                    "design",
                    request_id="A-0002",
                    status=ApprovalStatus.CANCELLED,
                    voters=(),
                    created_offset=5,
                ),
            ],
            [],
            [],
            False,
        ),
        (
            "nobody is required to approve",
            [approval("design", required=0, voters=())],
            [],
            [],
            True,
        ),
        ("backed waiver", [approval("design")], [waiver], [waive], False),
        ("waiver without a WAIVER decision", [approval("design")], [waiver], [], True),
        (
            "waiver, decision of another kind",
            [approval("design")],
            [waiver],
            [DecisionKind.DESIGN],
            True,
        ),
        (
            "waiver without its approved request",
            [approval("design", request_id="A-0009")],
            [waiver],
            [waive],
            True,
        ),
    ]
    for name, requests, waivers, kinds, problem in cases:
        view = GovView(approval_chain())
        for request in requests:
            view.add_approval(request)
        for item in waivers:
            view.add_waiver(item)
        for kind in kinds:
            view.add_decision(kind)
        flagged = bool(run_rule(DecisionCompleteness, run_subject(view)))
        got = h.run(view).check("approvals_recorded")
        assert flagged is problem, f"{name}: CMP-003 expectation is wrong"
        assert got["passed"] is (not problem), f"{name}: agent {got['detail']}"


# ----------------------------------------------------------------------------- traceability


def test_only_passed_tests_cover_a_criterion(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    first = report_row("tests/test_r001.py::test_ac1", covers=["R-001", "AC-001-1"])
    second = "tests/test_r001.py::test_ac2"
    unpassed = "covered only by tests that did not pass"
    cases = [
        ("failed", [report_row(second, "failed", covers=["AC-001-2"])], unpassed),
        ("error", [report_row(second, "error", covers=["AC-001-2"])], unpassed),
        ("skipped", [report_row(second, "skipped", covers=["AC-001-2"])], unpassed),
        ("absent", [], "no passed covering test"),
    ]
    for name, rows, why in cases:
        h.inp.tests = build_test_report([first, *rows])
        h.publish(Kind.TEST_REPORT)
        out = h.run()
        assert out.matrix["gaps"] == [f"R-001/AC-001-2: {why}"], name
        assert out.matrix["complete"] is False, name
        assert out.row("R-001")["acceptance_criteria"][1] == {
            "id": "AC-001-2",
            "tests": [r["nodeid"] for r in rows],
            "results": [r["outcome"] for r in rows],
            "covered": False,
        }, name
        assert out.verdict == "NO_GO" and "traceability_complete" in out.failing, name
    mixed = [
        report_row(f"{second}_a", "failed", covers=["AC-001-2"]),
        report_row(f"{second}_b", covers=["AC-001-2"]),
    ]
    h.inp.tests = build_test_report([first, *mixed])
    h.publish(Kind.TEST_REPORT)
    out = h.run()
    assert out.matrix["complete"] is True and out.matrix["gaps"] == []
    assert out.row("R-001")["acceptance_criteria"][1] == {
        "id": "AC-001-2",
        "tests": [f"{second}_a", f"{second}_b"],
        "results": ["failed", "passed"],
        "covered": True,
    }
    assert out.failing == ["tests_passed"]  # covered, but the failed test still blocks the release


def test_an_uncovered_criterion_is_a_gap_that_blocks_the_release(tmp_path: Path) -> None:
    inp = green()
    inp.tests = build_test_report(cover("R-001", 1))
    out = run_release(tmp_path, inp)
    assert out.matrix["gaps"] == ["R-001/AC-001-2: no passed covering test"]
    assert out.matrix["complete"] is False
    assert out.verdict == "NO_GO" and out.failing == ["traceability_complete"]
    assert out.check("traceability_complete")["detail"] == "1 gap(s)"
    assert "traceability_complete: 1 gap(s)" in out.readiness["open_items"]
    assert out.row("R-001")["acceptance_criteria"][1]["covered"] is False
    assert out.decision.refs == ("R-001",)


def test_a_requirement_without_criteria_needs_a_passed_test_citing_its_id(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.inp.requirements = requirements_payload(
        requirement("R-001"), requirement("R-005", criteria=0)
    )
    cited = report_row("tests/test_r005.py::test_it", covers=["R-005"])
    h.inp.tests = build_test_report([*cover("R-001"), cited])
    h.publish(Kind.REQUIREMENTS, Kind.TEST_REPORT)
    covered = h.run()
    assert covered.verdict == "GO" and covered.matrix["complete"] is True
    assert covered.row("R-005")["acceptance_criteria"] == []
    assert covered.row("R-005")["files"] == ["tests/test_r005.py"]
    h.inp.tests = build_test_report(cover("R-001"))
    h.publish(Kind.TEST_REPORT)
    bare = h.run()
    assert bare.matrix["gaps"] == ["R-005: no passed covering test"]
    assert bare.decision.refs == ("R-005",)


def test_human_decisions_evidence_a_criterion_but_agent_decisions_do_not(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    h.inp.tests = build_test_report(cover("R-001", 1))  # AC-001-2 has no test
    h.publish(Kind.TEST_REPORT)
    human = h.run(governed(decision("D-0001", DecisionKind.DESIGN, refs=["AC-001-2"])))
    assert human.matrix["complete"] is True
    assert human.row("R-001")["acceptance_criteria"][1]["covered"] is True
    assert human.row("R-001")["decision_refs"] == ["D-0001"]
    others = {
        "an agent decision": governed(by_agent("D-0001", DecisionKind.DESIGN, refs=["AC-001-2"])),
        "a human decision citing only the requirement": governed(
            decision("D-0001", DecisionKind.DESIGN, refs=["R-001"])
        ),
        "no decision": GovView(),
    }
    for name, view in others.items():
        out = h.run(view)
        assert out.matrix["gaps"] == ["R-001/AC-001-2: no passed covering test"], name


def inspection_inputs(*, component: bool, path: str | None) -> Inputs:
    """R-001 (test) and R-002 (inspection); R-002 is evidenced by the design and/or a task path."""
    inp = green()
    inp.requirements = requirements_payload(
        requirement("R-001"), requirement("R-002", verification="inspection", criteria=1)
    )
    inp.design = design_doc_payload(requirement_ids=("R-001", "R-002") if component else ("R-001",))
    tasks = [task("T-001", requirement_ids=["R-001"])]
    if path:
        tasks.append(task("T-002", requirement_ids=["R-002"], expected_paths=[path]))
    inp.plan = task_plan_payload(*tasks)
    return inp


INSPECTED = [
    pytest.param(True, None, ["component: widgets"], [], id="design-component"),
    pytest.param(
        False, "widgets/core.py", ["file: widgets/core.py"], ["widgets/core.py"], id="changed-file"
    ),
    pytest.param(
        False, "docs/widgets.md", ["file: docs/widgets.md"], ["docs/widgets.md"], id="docs-file"
    ),
    pytest.param(
        True,
        "widgets/core.py",
        ["component: widgets", "file: widgets/core.py"],
        ["widgets/core.py"],
        id="component-and-file",
    ),
]


@pytest.mark.parametrize(("component", "path", "evidence", "files"), INSPECTED)
def test_an_inspection_requirement_is_evidenced_by_the_design_and_the_changed_files(
    tmp_path: Path, component: bool, path: str | None, evidence: list[str], files: list[str]
) -> None:
    out = run_release(tmp_path, inspection_inputs(component=component, path=path))
    got = out.check("R-002")
    assert got["passed"] is True and got["evidence"] == evidence
    assert got["detail"] == "design component and changed files found"
    assert out.verdict == "GO" and out.matrix["complete"] is True
    assert out.row("R-002")["acceptance_criteria"] == [
        {"id": "AC-002-1", "tests": [], "results": ["inspection"], "covered": True}
    ]
    assert out.row("R-002")["files"] == files


def test_an_inspection_requirement_without_evidence_fails_its_check_and_is_a_gap(
    tmp_path: Path,
) -> None:
    out = run_release(tmp_path, inspection_inputs(component=False, path="widgets/not_changed.py"))
    got = out.check("R-002")
    assert got["passed"] is False and got["evidence"] == []
    assert got["detail"] == "no design component or changed file evidences the requirement"
    assert out.matrix["gaps"] == ["R-002: verification=inspection without inspection evidence"]
    assert out.row("R-002")["acceptance_criteria"] == [
        {"id": "AC-002-1", "tests": [], "results": [], "covered": False}
    ]
    assert set(out.failing) == {"R-002", "traceability_complete"} and out.verdict == "NO_GO"
    assert (
        "R-002: no design component or changed file evidences the requirement"
        in (out.readiness["open_items"])
    )
    assert out.decision.refs == ("R-002",)


def test_a_human_decision_traces_an_inspection_but_the_inspection_check_still_fails(
    tmp_path: Path,
) -> None:
    view = governed(decision("D-0001", DecisionKind.DESIGN, refs=["R-002"]))
    out = run_release(tmp_path, inspection_inputs(component=False, path=None), view)
    assert out.matrix["complete"] is True  # CMP-001: a human decision citing the id is evidence
    assert out.check("R-002")["passed"] is False  # the check itself needs design or file evidence
    assert out.failing == ["R-002"] and out.verdict == "NO_GO"


def deferred_inputs(marker: str | None, status: str = "deferred") -> Inputs:
    """R-001 (covered) plus R-004, a deferred/descoped requirement with one criterion."""
    inp = green()
    parked = requirement("R-004", status=status, criteria=1)
    parked["status_decision"] = marker
    inp.requirements = requirements_payload(requirement("R-001"), parked)
    return inp


SCOPE_CASES: list[tuple[str, str | None, list[Decision], bool]] = [
    ("human SCOPE citing the requirement", None, [decision("D-1", SCOPE, refs=["R-004"])], True),
    ("human SCOPE citing a criterion", None, [decision("D-1", SCOPE, refs=["AC-004-1"])], True),
    (
        "marker SCOPE:<id> accepts a human SCOPE decision",
        "SCOPE:R-004",
        [decision("D-1", SCOPE, refs=["R-004"])],
        True,
    ),
    (
        "agent SCOPE whose parent is a human CLARIFICATION",
        "SCOPE:R-004",
        [decision("D-1"), by_agent("D-2", SCOPE, refs=["R-004"], parents=["D-1"])],
        True,
    ),
    (
        "agent SCOPE two hops away from a human",
        "SCOPE:R-004",
        [
            decision("D-1"),
            by_agent("D-2", CLARIFY, parents=["D-1"]),
            by_agent("D-3", SCOPE, refs=["R-004"], parents=["D-2"]),
        ],
        True,
    ),
    ("agent-only SCOPE", "SCOPE:R-004", [by_agent("D-1", SCOPE, refs=["R-004"])], False),
    (
        "agent SCOPE with an unknown parent",
        None,
        [by_agent("D-1", SCOPE, refs=["R-004"], parents=["D-9999"])],
        False,
    ),
    (
        "human decision of another kind",
        None,
        [decision("D-1", DecisionKind.DESIGN, refs=["R-004"])],
        False,
    ),
    ("human SCOPE for another requirement", None, [decision("D-1", SCOPE, refs=["R-009"])], False),
    (
        "system SCOPE",
        None,
        [decision("D-1", SCOPE, actor=Actor.system(), refs=["R-004"])],
        False,
    ),
    ("no decision at all", "SCOPE:R-004", [], False),
    ("marker names a decision: it exists", "D-7", [decision("D-7", SCOPE, refs=["R-004"])], True),
    (
        "marker names a decision: another human SCOPE decision does not do",
        "D-7",
        [decision("D-1", SCOPE, refs=["R-004"])],
        False,
    ),
    (
        "marker names a decision: that one is agent-only",
        "D-7",
        [by_agent("D-7", SCOPE, refs=["R-004"]), decision("D-1", SCOPE, refs=["R-004"])],
        False,
    ),
    (
        "marker names a decision: that one is not a SCOPE decision",
        "D-7",
        [decision("D-7", CLARIFY, refs=["R-004"]), decision("D-1", SCOPE, refs=["R-004"])],
        False,
    ),
]


def test_a_deferred_requirement_needs_a_scope_decision_with_human_provenance(
    tmp_path: Path,
) -> None:
    h = Harness(tmp_path)
    for name, marker, decisions, covered in SCOPE_CASES:
        h.inp = deferred_inputs(marker)
        h.publish(Kind.REQUIREMENTS)
        out = h.run(governed(*decisions))
        row = out.row("R-004")
        assert row["status"] == "deferred", name
        if covered:
            assert out.matrix["gaps"] == [] and out.verdict == "GO", name
            assert row["acceptance_criteria"] == [
                {"id": "AC-004-1", "tests": [], "results": ["deferred"], "covered": True}
            ], name
        else:
            assert out.matrix["gaps"] == ["R-004: deferred without a human SCOPE decision"], name
            assert row["acceptance_criteria"] == [
                {"id": "AC-004-1", "tests": [], "results": [], "covered": False}
            ], name
            assert out.failing == ["traceability_complete"], name
            assert out.decision.refs == ("R-004",), name


def test_descoped_requirements_follow_the_same_rule(tmp_path: Path) -> None:
    h = Harness(tmp_path, deferred_inputs(None, "descoped"))
    assert h.run().matrix["gaps"] == ["R-004: descoped without a human SCOPE decision"]
    backed = h.run(governed(decision("D-0001", SCOPE, refs=["R-004"])))
    assert backed.matrix["gaps"] == []
    assert backed.row("R-004")["acceptance_criteria"][0]["results"] == ["descoped"]


def test_go_with_a_human_scoped_deferral_lists_it_as_an_open_item(tmp_path: Path) -> None:
    view = governed(decision("D-0001", SCOPE, refs=["R-004"]))
    out = run_release(tmp_path, deferred_inputs("SCOPE:R-004"), view)
    assert out.verdict == "GO" and out.readiness["open_items"] == ["R-004: deferred"]
    assert out.row("R-004")["decision_refs"] == ["D-0001"]
    assert out.decision.parents == ("D-0001",)


def agent_chain(length: int) -> list[Decision]:
    """C-0 (an agent's SCOPE decision citing R-004) -> C-1 -> ... -> H-1, a human decision."""
    links = [
        by_agent(
            f"C-{i}",
            SCOPE if i == 0 else CLARIFY,
            refs=["R-004"] if i == 0 else [],
            parents=[f"C-{i + 1}" if i + 1 < length else "H-1"],
        )
        for i in range(length)
    ]
    return [*links, decision("H-1")]


def test_the_provenance_walk_is_bounded_and_survives_cycles(tmp_path: Path) -> None:
    h = Harness(tmp_path, deferred_inputs(None))

    def gaps(*decisions: Decision) -> list[str]:
        return list(h.run(governed(*decisions)).matrix["gaps"])

    gap = ["R-004: deferred without a human SCOPE decision"]
    assert gaps(*agent_chain(25)) == []
    assert gaps(*agent_chain(MAX_PROVENANCE_NODES + 5)) == gap  # fail closed beyond the bound
    loop = [
        by_agent("D-1", SCOPE, refs=["R-004"], parents=["D-2"]),
        by_agent("D-2", CLARIFY, parents=["D-1"]),
    ]
    assert gaps(*loop) == gap  # a cycle without a human terminates
    way_out = by_agent("D-2", CLARIFY, parents=["D-1", "H-1"])
    assert gaps(loop[0], way_out, decision("H-1")) == []  # ... and a way out is found


def test_matrix_rows_gaps_tasks_and_files_are_sorted(tmp_path: Path) -> None:
    inp = green()
    inp.requirements = requirements_payload(
        requirement("R-003", criteria=1),
        requirement("R-001", criteria=1),
        requirement("R-002", criteria=1),
    )
    inp.tests = build_test_report([])
    inp.plan = task_plan_payload(
        task("T-003", requirement_ids=["R-001"], expected_paths=["widgets/b.py"]),
        task("T-001", requirement_ids=["R-001"], expected_paths=["widgets/a.py"]),
    )
    inp.integrated = {"widgets/b.py": "B = 1\n", "widgets/a.py": "A = 1\n"}
    out = run_release(tmp_path, inp)
    assert [r["id"] for r in out.matrix["requirements"]] == ["R-001", "R-002", "R-003"]
    assert len(out.matrix["gaps"]) == 3 and out.matrix["gaps"] == sorted(out.matrix["gaps"])
    assert out.row("R-001")["tasks"] == ["T-001", "T-003"]
    assert out.row("R-001")["files"] == ["widgets/a.py", "widgets/b.py"]


# ------------------------------------------------------------------------ real CMP-001 / CHG-005


def agree_with_cmp001(tmp_path: Path, inp: Inputs, *decisions: Decision) -> Release:
    """Run the agent and the real CMP-001 rule over the same evidence; the gaps must be the same.

    The artifacts are registered like the engine does: ancestors' outputs as active, the readiness
    report of the gated attempt as staged.
    """
    view = GovView(release_chain()).add_raw(*decisions)
    out = Harness(tmp_path, inp).run(view)
    view.add_artifact(Kind.REQUIREMENTS, inp.requirements, producer="requirements")
    view.add_artifact(Kind.TEST_REPORT, inp.tests, producer="verify")
    view.add_artifact(
        Kind.READINESS_REPORT, out.readiness, producer=NODE, status=ArtifactStatus.STAGED
    )
    policy = {str(f.location) for f in run_rule(Traceability, run_subject(view))}
    agent = {g.split(":")[0] for g in out.matrix["gaps"]}
    assert policy == agent, f"CMP-001 reports {sorted(policy)}, the matrix {sorted(agent)}"
    assert (not policy) is out.matrix["complete"]
    return out


def _uncovered() -> Inputs:
    inp = green()
    inp.tests = build_test_report(cover("R-001", 1))
    return inp


def _failed_only() -> Inputs:
    inp = green()
    failed = report_row("tests/test_r001.py::test_ac2", "failed", covers=["AC-001-2"])
    inp.tests = build_test_report([cover("R-001", 1)[0], failed])
    return inp


def _inspection_ok() -> Inputs:
    return inspection_inputs(component=True, path=None)


def _inspection_missing() -> Inputs:
    return inspection_inputs(component=False, path=None)


CONSISTENT = [
    pytest.param(green, set(), id="green"),
    pytest.param(_uncovered, {"R-001/AC-001-2"}, id="uncovered-criterion"),
    pytest.param(_failed_only, {"R-001/AC-001-2"}, id="covered-only-by-a-failed-test"),
    pytest.param(_inspection_ok, set(), id="inspection-with-evidence"),
    pytest.param(_inspection_missing, {"R-002"}, id="inspection-without-evidence"),
]


@pytest.mark.parametrize(("build", "gaps"), CONSISTENT)
def test_the_matrix_agrees_with_the_real_cmp001_rule(
    tmp_path: Path, build: Callable[[], Inputs], gaps: set[str]
) -> None:
    out = agree_with_cmp001(tmp_path, build())
    assert {g.split(":")[0] for g in out.matrix["gaps"]} == gaps


def test_a_mixed_run_gets_the_same_gaps_from_the_agent_and_cmp001(tmp_path: Path) -> None:
    def parked(rid: str, status: str = "deferred") -> dict[str, Any]:
        item = requirement(rid, status=status, criteria=1)
        item["status_decision"] = None
        return item

    inp = green()
    inp.requirements = requirements_payload(
        requirement("R-001"),  # AC-001-2 is covered only by a failed test
        requirement("R-002", criteria=1),  # a human decision cites AC-002-1
        requirement("R-003", criteria=1),  # only an agent decision cites AC-003-1
        requirement("R-004", verification="inspection", criteria=1),  # a design component
        requirement("R-005", verification="inspection", criteria=1),  # no evidence
        requirement("R-006", criteria=0),  # a passed test cites R-006
        requirement("R-007", criteria=0),  # nothing covers it
        parked("R-008"),  # a human SCOPE decision
        parked("R-009"),  # nothing
        parked("R-010", "descoped"),  # an agent-only SCOPE decision
        parked("R-011"),  # a human decision, but not a SCOPE one
        requirement("R-012", criteria=1),  # a human decision cites only R-012
    )
    inp.design = design_doc_payload(requirement_ids=("R-004",))
    failed = report_row("tests/test_r001.py::test_ac2", "failed", covers=["AC-001-2"])
    inp.tests = build_test_report(
        [cover("R-001", 1)[0], failed, report_row("tests/test_r006.py::test_it", covers=["R-006"])]
    )
    out = agree_with_cmp001(
        tmp_path,
        inp,
        decision("D-0001", DecisionKind.DESIGN, refs=["AC-002-1"]),
        by_agent("D-0002", DecisionKind.ASSUMPTION, refs=["AC-003-1"]),
        decision("D-0003", SCOPE, refs=["R-008"]),
        by_agent("D-0004", SCOPE, refs=["R-010"]),
        decision("D-0005", DecisionKind.DESIGN, refs=["R-011"]),
        decision("D-0006", DecisionKind.ASSUMPTION, refs=["R-012"]),
    )
    assert {g.split(":")[0] for g in out.matrix["gaps"]} == {
        "R-001/AC-001-2",
        "R-003/AC-003-1",
        "R-005",
        "R-007",
        "R-009",
        "R-010",
        "R-011",
        "R-012/AC-012-1",
    }
    assert out.matrix["complete"] is False


_MARKER_REASON = (
    "known divergence: CMP-001 reads any non-empty status_decision as a decision id, so the "
    "'SCOPE:<id>' marker the requirements agent writes can never match"
)
_ACTOR_REASON = (
    "known divergence: CMP-001 accepts only SCOPE decisions whose actor is human; the release "
    "manager also accepts an agent SCOPE decision whose parent chain reaches a human decision"
)
DIVERGENCES = [
    pytest.param(
        "SCOPE:R-004",
        [decision("D-0001", SCOPE, refs=["R-004"])],
        id="scope-marker",
        marks=pytest.mark.xfail(reason=_MARKER_REASON, strict=False),
    ),
    pytest.param(
        None,
        [decision("D-0001"), by_agent("D-0002", SCOPE, refs=["R-004"], parents=["D-0001"])],
        id="agent-scope-with-a-human-parent",
        marks=pytest.mark.xfail(reason=_ACTOR_REASON, strict=False),
    ),
]


@pytest.mark.parametrize(("marker", "decisions"), DIVERGENCES)
def test_deferrals_as_the_requirements_agent_records_them_also_satisfy_cmp001(
    tmp_path: Path, marker: str | None, decisions: list[Decision]
) -> None:
    out = agree_with_cmp001(tmp_path, deferred_inputs(marker), *decisions)
    assert out.matrix["complete"] is True


def test_rollback_plan_and_change_reference_agree_with_chg005(tmp_path: Path) -> None:
    h = Harness(tmp_path)
    values = ["", "   \n\t", "TBD", " n/a ", "None", "todo", "-", "?", "Unknown", "Revert it."]
    for field in ("rollback_plan", "change_reference"):
        for value in values:
            h.inp.design = design_doc_payload()
            h.inp.design[field] = value
            h.publish(Kind.DESIGN_DOC)
            out = h.run()
            view = GovView(release_chain())
            view.add_artifact(Kind.DESIGN_DOC, h.inp.design, producer="design")
            flagged = field in {
                f.evidence for f in run_rule(ReleaseDocumentation, run_subject(view))
            }
            assert out.check(field)["passed"] is (not flagged), f"{field}={value!r}"
            assert (out.verdict == "NO_GO") is flagged, f"{field}={value!r}"


# ----------------------------------------------------------------------------- the decision


def test_release_decision_parents_are_exactly_the_sorted_human_decision_ids(
    tmp_path: Path,
) -> None:
    view = governed(
        decision("D-0900"),
        decision("D-0100", SCOPE, refs=["R-001"]),
        by_agent("D-0500", DecisionKind.DESIGN),
        decision("D-0300", DecisionKind.DESIGN, actor=Actor.system()),
        decision("D-0400", DecisionKind.POLICY, actor=Actor.policy("auto")),
    )
    out = run_release(tmp_path / "with", None, view)
    assert out.decision.parents == ("D-0100", "D-0900")
    assert set(out.decision.parents) <= {d.id for d in view.decisions()}
    assert run_release(tmp_path / "without").decision.parents == ()  # none is invented


def test_release_decision_parents_are_capped_after_sorting(tmp_path: Path) -> None:
    newest_first = [decision(f"D-{n:04d}") for n in range(MAX_HUMAN_PARENTS + 10, 0, -1)]
    out = run_release(tmp_path, None, governed(*newest_first))
    assert out.decision.parents == tuple(f"D-{n:04d}" for n in range(1, MAX_HUMAN_PARENTS + 1))


def test_release_decision_refs_name_the_requirements_with_gaps_and_are_capped(
    tmp_path: Path,
) -> None:
    inp = green()
    inp.requirements = requirements_payload(
        requirement("R-001"),
        requirement("R-002"),  # two uncovered criteria, one reference
        *(requirement(f"R-{n:03d}", criteria=1) for n in range(3, 26)),
    )
    inp.tests = build_test_report(cover("R-001"))
    out = run_release(tmp_path, inp)
    assert len(out.matrix["gaps"]) == 2 + 23
    assert out.decision.refs == tuple(f"R-{n:03d}" for n in range(2, 22))  # sorted, first 20


def test_only_the_traceability_gaps_are_cited_by_the_decision(tmp_path: Path) -> None:
    inp = green()
    inp.security = security_report_payload([finding("HIGH")])
    out = run_release(tmp_path, inp)
    assert out.failing == ["security_clean"] and out.decision.refs == ()


# ----------------------------------------------------------------------------- determinism / text


def test_runs_with_different_clocks_produce_identical_content(tmp_path: Path) -> None:
    runs: list[Release] = []
    for name, start in (
        ("a", datetime(2026, 1, 1, tzinfo=UTC)),
        ("b", datetime(2031, 7, 9, 13, 5, tzinfo=UTC)),
    ):
        inp = inspection_inputs(component=False, path=None)
        inp.security = security_report_payload([finding("HIGH")])
        view = governed(decision("D-0001"), by_agent("D-0002", SCOPE, refs=["R-001"]))
        runs.append(Harness(tmp_path / name, inp, clock_start=start).run(view))
    assert runs[0].verdict == "NO_GO" and runs[0].matrix["gaps"]
    assert content_hashes(runs[0].result) == content_hashes(runs[1].result)
    assert runs[0].result.decisions == runs[1].result.decisions


HOSTILE = "\x1b[31mRED\x1b[0m\x1b]0;pwned\x07 sneaky\r\nnext\u202e\u200b\x00\x7f" + "X" * 3000
REQ_TOKEN = "REQTEXT-9f3a"
AC_TOKEN = "ACTEXT-7c1b"


def printable(text: str) -> bool:
    return not any(unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Co", "Cn"} for ch in text)


def test_hostile_text_is_sanitised_and_capped_in_open_items_and_details(tmp_path: Path) -> None:
    inp = green()
    hostile = requirement("R-001", text=f"{REQ_TOKEN} {HOSTILE}")
    hostile["acceptance_criteria"][0]["text"] = f"{AC_TOKEN} {HOSTILE}"
    inp.requirements = requirements_payload(
        hostile, requirement("R-002", verification="inspection", criteria=1)
    )
    inp.requirements["assumptions"] = [{"id": "ASM-001", "text": HOSTILE, "question_id": None}]
    inp.design = design_doc_payload(
        requirement_ids=("R-001", "R-002"), change_reference="CR-1" + HOSTILE
    )
    inp.design["components"][0]["name"] = HOSTILE
    out = run_release(tmp_path, inp)
    assert out.verdict == "GO"
    items = out.readiness["open_items"]
    assert len(items) == 1 and items[0].startswith("ASM-001: RED sneaky")
    assert len(items[0]) <= 300 and printable(items[0]) and "[truncated" in items[0]
    for item in out.readiness["checks"]:
        assert len(item["detail"]) <= 300 and printable(item["detail"]), item["id"]
        for line in item["evidence"]:
            assert len(line) <= 160 and printable(line), item["id"]
    assert out.check("R-002")["evidence"][0].startswith("component: RED sneaky")
    assert out.check("change_reference")["evidence"][0].startswith("design_doc.change_reference: ")
    emitted = json.dumps([out.readiness["checks"], out.readiness["open_items"], out.matrix])
    assert REQ_TOKEN not in emitted and AC_TOKEN not in emitted  # requirement text is never echoed


def test_open_items_are_capped_and_failing_checks_come_first(tmp_path: Path) -> None:
    inp = green()
    inp.requirements["assumptions"] = [
        {"id": f"ASM-{n:03d}", "text": f"assumption {n}", "question_id": None} for n in range(1, 61)
    ]
    inp.design = design_doc_payload(rollback_plan="")
    out = run_release(tmp_path, inp)
    items = out.readiness["open_items"]
    assert len(items) == MAX_OPEN_ITEMS
    assert items[0] == "rollback_plan: empty" and items[1] == "ASM-001: assumption 1"
