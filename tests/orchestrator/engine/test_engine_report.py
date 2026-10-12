"""Run report (report.md / report.json) and the sealed run outputs of ``engine/report.py``.

Runs are built by hand (hand-made audit events with a controllable time, hand-made state), so every
expected number can be derived on paper and nothing depends on the scheduler.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.artifacts import ArtifactStore
from orchestrator.core.audit import GENESIS_HASH, AuditLog, export_trace
from orchestrator.core.clock import FakeClock
from orchestrator.core.decisions import DecisionLog
from orchestrator.core.errors import IntegrityError
from orchestrator.core.metrics import RunMetrics, compute_run_metrics
from orchestrator.core.state import AttemptRecord, NodeState, RunState, SafeStopRecord, StateStore
from orchestrator.core.types import (
    Actor,
    ApprovalPosition,
    ApprovalPurpose,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalVote,
    Artifact,
    ArtifactDraft,
    ArtifactStatus,
    AuditEvent,
    Decision,
    DecisionKind,
    Epoch,
    EventType,
    NodeStatus,
    RiskLevel,
    RunStatus,
    Waiver,
)
from orchestrator.engine.report import (
    REPORT_FILES,
    md_cell,
    md_ident,
    md_text,
    render_report,
    write_run_outputs,
)

ET = EventType
T0 = datetime(2026, 1, 1, tzinfo=UTC)
HOSTILE = (
    "\x1b[31mred\x1b[0m <script>alert(1)</script> [click](https://evil.example/x) "
    "![img](http://evil.example/i.png) https://evil.example/a www.evil.example "
    "mail@evil.example | cell\nnew line `code` # heading \u202eRTL\u202c"
)


class Stream:
    """Hand-built audit event stream (time moves only through ``at``)."""

    def __init__(self, run_id: str = "run-1") -> None:
        self.run_id = run_id
        self.events: list[AuditEvent] = []
        self._now = 0.0

    def at(self, seconds: float) -> Stream:
        self._now = seconds
        return self

    def add(
        self,
        event_type: EventType,
        node: str | None = None,
        attempt: int | None = None,
        **data: Any,
    ) -> Stream:
        seq = len(self.events) + 1
        self.events.append(
            AuditEvent(
                seq=seq,
                ts=T0 + timedelta(seconds=self._now),
                run_id=self.run_id,
                trace_id="trace-1",
                span_id="run" if node is None else f"n:{node}",
                parent_span_id=None,
                type=event_type,
                node_id=node,
                attempt=attempt,
                actor=Actor.system(),
                data=data,
                prev_hash=GENESIS_HASH if seq == 1 else f"{seq - 1:064x}",
                hash=f"{seq:064x}",
            )
        )
        return self

    def start(self, node: str, attempt: int, stage: str = "implementation") -> Stream:
        return self.add(
            ET.NODE_STARTED, node, attempt, stage=stage, agent_id="agent", epoch="primary"
        )

    def done(self, node: str, attempt: int) -> Stream:
        return self.add(ET.NODE_SUCCEEDED, node, attempt, stage="x", outputs=[], cache_hit=False)

    def fail(self, node: str, attempt: int, kind: str = "transient", message: str = "m") -> Stream:
        return self.add(
            ET.NODE_ATTEMPT_FAILED,
            node,
            attempt,
            kind=kind,
            message=message,
            epoch="primary",
            retryable=True,
            rule_ids=[],
        )


def failed_run_stream(message: str = "boom") -> Stream:
    s = Stream()
    s.at(0).add(ET.RUN_CREATED, scenario="demo").start("intake", 1, "intake")
    s.at(5).done("intake", 1).start("design", 1, "design")
    s.at(10).add(ET.NODE_AWAITING_APPROVAL, "design", 1)
    s.add(ET.APPROVAL_REQUESTED, "design", approval_id="A-0001", purpose="node_output", risk="HIGH")
    s.add(ET.RUN_WAITING, reason="approval", approvals=["A-0001"], questions=[])
    s.at(70).add(ET.APPROVAL_VOTE, "design", approval_id="A-0001", approver="bob", approved=True)
    s.add(ET.APPROVAL_VOTE, "design", approval_id="A-0001", approver="carol", approved=True)
    s.add(ET.APPROVAL_GRANTED, "design", approval_id="A-0001").add(ET.RUN_RESUMED)
    s.at(71).done("design", 1).start("implement.core", 1)
    s.at(72).add(ET.PLAN_REPLANNED, decision_id="D-0004", added=["a", "b"], removed=[], retained=[])
    s.at(73).add(
        ET.POLICY_VIOLATION,
        "implement.core",
        1,
        rule_id="SEC-001",
        severity="HIGH",
        enforcement="deny",
        location="app.py:3",
    )
    s.fail("implement.core", 1, "policy_violation", "secret found")
    s.add(ET.NODE_RETRY_SCHEDULED, "implement.core", 1, delay_s=0.5, not_before="x")
    s.at(74).start("implement.core", 2)
    s.at(76).fail("implement.core", 2, "transient", "flaky")
    s.add(ET.NODE_FALLBACK_ACTIVATED, "implement.core", from_agent="implementer", to_agent="safe")
    s.at(77).start("implement.core", 3)
    s.at(80).done("implement.core", 3).start("verify", 1, "testing")
    s.at(84).fail("verify", 1, "gate_rejected", message)
    s.add(ET.NODE_FAILED, "verify", 1, kind="gate_rejected", message=message)
    s.add(ET.NODE_BLOCKED, "docs", reason="dependency verify failed")
    s.add(ET.ROLLBACK_STARTED, "implement.core", level="node")
    s.add(ET.NODE_ROLLED_BACK, "implement.core").add(ET.ROLLBACK_COMPLETED, "implement.core")
    s.at(85).add(ET.RUN_FAILED, reason=message)
    return s


def node(
    node_id: str,
    status: NodeStatus,
    attempts: int = 1,
    epoch: Epoch = Epoch.PRIMARY,
    agent: str = "agent",
    reason: str = "",
) -> NodeState:
    ns = NodeState(
        node_id=node_id, status=status, attempt_counter=attempts, epoch=epoch, reason=reason
    )
    ns.attempts = [
        AttemptRecord(number=i + 1, generation=i + 1, epoch=epoch, agent_id=agent, started_at=T0)
        for i in range(attempts)
    ]
    return ns


def vote(
    approver: str, channel: str = "cli", comment: str = "ok", approved: bool = True
) -> ApprovalVote:
    return ApprovalVote(approver, approved, comment, T0, channel)


def approval(
    request_id: str = "A-0001",
    node_id: str = "design",
    votes: list[ApprovalVote] | None = None,
    risk: RiskLevel = RiskLevel.HIGH,
) -> ApprovalRequest:
    return ApprovalRequest(
        id=request_id,
        run_id="run-1",
        node_id=node_id,
        position=ApprovalPosition.AFTER,
        purpose=ApprovalPurpose.NODE_OUTPUT,
        risk=risk,
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
        votes=votes if votes is not None else [vote("bob"), vote("carol")],
    )


def make_state(**overrides: Any) -> RunState:
    state = RunState(
        run_id="run-1",
        scenario="demo",
        created_by="alice",
        created_at=T0,
        trace_id="trace-1",
        status=RunStatus.FAILED,
        status_reason="verify failed",
        started_at=T0,
        finished_at=T0 + timedelta(seconds=85),
        assurance="HUMAN",
        graph_hash="a" * 64,
        baseline_manifest_hash="b" * 64,
    )
    state.graph = {
        "nodes": [
            {"id": "intake", "stage": "intake", "agent": "intake"},
            {"id": "design", "stage": "design", "agent": "architect"},
            {"id": "codebase-analysis", "stage": "analysis", "agent": "analyst"},
            {"id": "docs", "stage": "documentation", "agent": "documenter"},
        ]
    }
    state.nodes = {
        n.node_id: n
        for n in (
            node("intake", NodeStatus.SUCCEEDED, agent="intake"),
            node("design", NodeStatus.SUCCEEDED, agent="architect"),
            node("implement.core", NodeStatus.SUCCEEDED, 3, Epoch.FALLBACK, "safe"),
            node("verify", NodeStatus.FAILED, 1, agent="verifier", reason="gate_rejected: tests"),
            node("docs", NodeStatus.BLOCKED, 0, reason="dependency verify failed"),
            node("codebase-analysis", NodeStatus.SKIPPED, 0, reason="greenfield"),
        )
    }
    state.approvals = {"A-0001": approval()}
    state.audit_head = {"seq": 99, "hash": "c" * 64}
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def decision(
    number: int, kind: DecisionKind = DecisionKind.DESIGN, summary: str = "Use SQLite"
) -> Decision:
    return Decision(
        id=f"D-{number:04d}",
        run_id="run-1",
        ts=T0,
        kind=kind,
        actor=Actor.agent("architect"),
        node_id="design",
        summary=summary,
    )


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


class Run:
    """Everything ``render_report`` needs, with a metrics object derived from the events."""

    def __init__(
        self,
        state: RunState | None = None,
        stream: Stream | None = None,
        decisions: list[Decision] | None = None,
        artifacts: list[Artifact] | None = None,
        posture: str = "NOMINAL",
    ) -> None:
        self.state = state if state is not None else make_state()
        self.events = (stream if stream is not None else failed_run_stream()).events
        self.decisions = (
            decisions
            if decisions is not None
            else [decision(1), decision(2, DecisionKind.RETRY, "Retry implement.core")]
        )
        self.artifacts = (
            artifacts
            if artifacts is not None
            else [artifact("design_doc"), artifact("requirements")]
        )
        self.posture = posture
        # metrics cannot be computed from an empty log: a run without events borrows the default
        self.metrics: RunMetrics = compute_run_metrics(self.events or failed_run_stream().events)

    def kwargs(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "events": self.events,
            "decisions": self.decisions,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "posture": self.posture,
        }

    def render(self) -> tuple[str, dict[str, Any]]:
        return render_report(**self.kwargs())


def section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    start = lines.index(f"## {heading}") + 1
    body: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body)


def rows(markdown: str, heading: str) -> list[list[str]]:
    """Data rows (cells) of the table under ``heading``."""
    table = [ln for ln in section(markdown, heading).splitlines() if ln.startswith("|")]
    return [[c.strip() for c in re.split(r"(?<!\\)\|", ln)[1:-1]] for ln in table[2:]]


# ----------------------------------------------------------------------------- header


def ident(text: str) -> str:
    return f"`{text}`"


def test_render_report_returns_markdown_and_a_json_able_dict() -> None:
    markdown, facts = Run().render()
    assert markdown.startswith("# Run report: run-1\n")
    assert markdown.endswith("\n")
    assert "\r" not in markdown
    assert facts["schema"] == 1
    assert json.loads(json.dumps(facts, allow_nan=False)) == facts


def test_header_shows_scenario_status_reason_posture_and_audit_head() -> None:
    run = Run(posture="DEGRADED: max_parallel=16")
    markdown, facts = run.render()
    assert "- **Scenario:** demo" in markdown
    assert "- **Status:** `failed` - verify failed" in markdown
    assert "- **Governance posture:** DEGRADED: max\\_parallel=16" in markdown
    last = run.events[-1]
    assert f"- **Audit head:** seq {last.seq}, hash `{last.hash}`, {len(run.events)} events" in (
        markdown
    )
    assert facts["run"]["audit_head"] == {
        "seq": last.seq,
        "hash": last.hash,
        "count": len(run.events),
    }
    assert facts["run"]["posture"] == "DEGRADED: max_parallel=16"
    assert "- **Four-eyes:** ATTESTED-ONLY" in markdown


def test_audit_head_falls_back_to_the_state_when_there_are_no_events() -> None:
    _, facts = Run(stream=Stream()).render()
    assert facts["run"]["audit_head"] == {"seq": 99, "hash": "c" * 64, "count": 0}


@pytest.mark.parametrize(
    ("assurance", "channel", "banner"),
    [
        ("DEMO_SCRIPTED", "scripted", True),
        ("HUMAN", "scripted", True),  # the votes are facts: a stale state cannot hide them
        ("DEMO_SCRIPTED", "cli", True),
        ("HUMAN", "cli", False),
    ],
)
def test_assurance_banner(assurance: str, channel: str, banner: bool) -> None:
    state = make_state(assurance=assurance)
    state.approvals = {"A-0001": approval(votes=[vote("bob", channel), vote("carol", channel)])}
    markdown, facts = Run(state=state).render()
    text = "approvals were scripted, not a human review"
    assert (text in markdown) is banner
    assert (facts["run"]["assurance"] == "DEMO_SCRIPTED") is banner
    if banner:
        assert markdown.splitlines()[2].startswith("> **ASSURANCE: DEMO_SCRIPTED**")


# ----------------------------------------------------------------------------- tables


def keyed(markdown: str, heading: str, column: int = 0) -> dict[str, list[str]]:
    return {r[column].strip("`"): r for r in rows(markdown, heading)}


def test_node_table_has_status_attempts_epoch_duration_and_agent() -> None:
    markdown, facts = Run().render()
    table = keyed(markdown, "Nodes")
    assert table["implement.core"][1:] == [
        "-",  # not in the persisted graph: no stage known
        ident("succeeded"),
        "3",
        ident("fallback"),
        "7.00",
        ident("safe"),
        "",
    ]
    assert table["verify"][2:] == [
        ident("failed"),
        "1",
        ident("primary"),
        "4.00",
        ident("verifier"),
        "gate\\_rejected: tests",
    ]
    assert table["design"][1] == ident("design")  # stage from the persisted graph
    assert table["docs"][2:5] == [ident("blocked"), "0", ident("primary")]
    assert table["docs"][5] == "-"  # never started: no duration
    assert table["docs"][6] == ident("documenter")  # a node that never ran: agent of the graph
    assert {n["node_id"] for n in facts["nodes"]} == set(table)


def test_nodes_are_listed_in_start_order_then_by_id() -> None:
    order = [r[0].strip("`") for r in rows(Run().render()[0], "Nodes")]
    assert order == ["intake", "design", "implement.core", "verify", "codebase-analysis", "docs"]


def test_epoch_number_is_shown_after_a_replan_or_reset() -> None:
    state = make_state()
    state.nodes["design"].epoch_no = 2
    table = keyed(Run(state=state).render()[0], "Nodes")
    assert table["design"][4] == "`primary` #2"


def test_timeline_has_the_key_events_and_not_the_routine_ones() -> None:
    markdown, facts = Run().render()
    kinds = [r[2].strip("`") for r in rows(markdown, "Timeline of key events")]
    for expected in (
        "node.attempt_failed",
        "node.retry_scheduled",
        "node.fallback_activated",
        "node.rolled_back",
        "rollback.started",
        "policy.violation",
        "approval.requested",
        "approval.vote",
        "approval.granted",
        "plan.replanned",
        "run.waiting",
        "run.resumed",
        "run.failed",
        "node.failed",
        "node.blocked",
    ):
        assert expected in kinds
    assert "node.started" not in kinds
    assert "node.succeeded" not in kinds
    assert [e["seq"] for e in facts["timeline"]] == sorted(e["seq"] for e in facts["timeline"])


def test_timeline_details_show_whitelisted_fields_only() -> None:
    markdown, _ = Run().render()
    table = keyed(markdown, "Timeline of key events", column=2)
    assert table["policy.violation"][5] == (
        "rule\\_id=SEC-001; severity=HIGH; enforcement=deny; location=app.py:3"
    )
    assert table["node.fallback_activated"][5] == "from\\_agent=implementer; to\\_agent=safe"
    assert table["plan.replanned"][5] == "decision\\_id=D-0004; added=2; removed=0; retained=0"
    assert table["policy.violation"][3:5] == [ident("implement.core"), "1"]
    assert table["approval.vote"][5] == "approval\\_id=A-0001; approver=carol; approved=yes"


def test_timeline_time_is_utc_with_milliseconds() -> None:
    markdown, facts = Run().render()
    assert facts["timeline"][0]["ts"] == "2026-01-01T00:00:10.000Z"
    assert rows(markdown, "Timeline of key events")[0][1] == "2026-01-01T00:00:10.000Z"


def test_approvals_list_request_node_risk_and_votes_with_channel() -> None:
    state = make_state()
    state.approvals = {
        "A-0001": approval(votes=[vote("bob", "scripted", ""), vote("carol", "cli", "looks fine")]),
        "A-0002": approval("A-0002", "verify", [vote("dave", "interactive", "no", approved=False)]),
    }
    markdown, facts = Run(state=state).render()
    table = rows(markdown, "Approvals")
    assert table[0] == [
        ident("A-0001"),
        ident("design"),
        ident("node_output"),
        ident("HIGH"),
        ident("approved"),
        "2",
        "bob approved via scripted; carol approved via cli: looks fine",
    ]
    assert table[1][6] == "dave rejected via interactive: no"
    assert facts["approvals"][0]["scripted"] is True
    assert facts["approvals"][1]["scripted"] is False
    assert "Four-eyes: ATTESTED-ONLY" in section(markdown, "Approvals")


def test_no_approvals_say_so() -> None:
    state = make_state()
    state.approvals = {}
    assert "No approval was requested." in section(Run(state=state).render()[0], "Approvals")


def test_decisions_list_id_kind_actor_and_summary_sorted_by_id() -> None:
    run = Run(decisions=[decision(2, DecisionKind.RETRY, "Retry it"), decision(1)])
    markdown, facts = run.render()
    assert rows(markdown, "Decisions") == [
        [ident("D-0001"), ident("design"), ident("agent:architect"), "Use SQLite"],
        [ident("D-0002"), ident("retry"), ident("agent:architect"), "Retry it"],
    ]
    assert [d["id"] for d in facts["decisions"]] == ["D-0001", "D-0002"]


def test_artifacts_show_a_hash_prefix_never_the_full_hash() -> None:
    run = Run(
        artifacts=[artifact("requirements"), artifact("design_doc", status=ArtifactStatus.STAGED)]
    )
    markdown, facts = run.render()
    table = rows(markdown, "Artifacts")
    assert [r[0] for r in table] == [ident("design_doc@v1"), ident("requirements@v1")]
    assert table[0][2] == ident("staged")
    full = artifact("requirements").content_hash
    assert table[1][4] == f"`{full[:12]}`"
    assert full not in markdown
    assert facts["artifacts"][1]["hash_prefix"] == full[:12]


def test_metrics_section_shows_the_reliability_numbers() -> None:
    markdown, facts = Run().render()
    table = {r[0]: r[1] for r in rows(markdown, "Reliability metrics")}
    assert table["Success"] == "no"
    assert table["Attempts (countable)"] == "6 \\(6\\)"
    assert table["Retries"] == "1"
    assert table["Retry frequency"] == "16.7%"
    assert table["Fallbacks (rate)"] == "1 \\(25.0%\\)"
    assert table["Rollbacks"] == "1"
    assert table["MTTR (s)"] == "7.00"
    assert table["End-to-end latency (s)"] == "85.00"
    assert table["Active latency (s)"] == "25.00"
    assert table["Human wait (s)"] == "60.00"
    assert table["Max concurrency observed"] == "1"
    assert table["Re-plan reuse"].startswith("-")
    assert table["Policy violations by rule"] == "SEC-001 x1"
    assert facts["metrics"] == Run().metrics.to_dict()


# ----------------------------------------------------------------------------- limitations


def limits(run: Run) -> list[str]:
    return run.render()[1]["limitations"]


def test_limitations_of_a_failed_run_are_computed_from_the_facts() -> None:
    text = "\n".join(limits(Run()))
    assert "final status failed - verify failed" in text
    assert "1 node(s) failed: verify (gate_rejected: tests)" in text
    assert "1 node(s) were blocked: docs (dependency verify failed)" in text
    assert "1 node(s) were skipped: codebase-analysis (greenfield)" in text
    assert "Unrecovered incident(s) on: verify" in text
    assert "SEC-001 x1" in text
    assert "ATTESTED-ONLY" in text
    assert "no external anchor" in text
    assert "scripted" not in text.lower()  # nothing scripted happened: nothing is claimed


def test_a_clean_success_has_no_failure_limitations() -> None:
    state = make_state(status=RunStatus.SUCCEEDED, status_reason="")
    state.nodes = {"intake": node("intake", NodeStatus.SUCCEEDED)}
    stream = Stream()
    stream.at(0).add(ET.RUN_CREATED).start("intake", 1)
    stream.at(2).done("intake", 1).add(ET.RUN_SUCCEEDED)
    text = "\n".join(limits(Run(state=state, stream=stream)))
    assert "did not succeed" not in text
    assert "node(s)" not in text
    assert "ATTESTED-ONLY" in text


def test_limitations_of_a_safe_stopped_run() -> None:
    state = make_state(status=RunStatus.SAFE_STOPPED, status_reason="kill switch")
    state.safe_stop = SafeStopRecord("kill_switch", "operator pressed stop")
    state.safe_stop.compensation_incomplete = ["promote"]
    state.nodes["implement.core"].status = NodeStatus.CANCELLED
    stream = Stream()
    stream.at(0).add(ET.RUN_CREATED).start("implement.core", 1)
    stream.at(3).add(ET.SAFE_STOP_TRIGGERED, trigger="kill_switch", reason="operator pressed stop")
    stream.add(ET.NODE_CANCELLED, "implement.core").add(ET.RUN_SAFE_STOPPED)
    run = Run(state=state, stream=stream)
    text = "\n".join(limits(run))
    assert "final status safe_stopped - kill switch" in text
    assert "Safe-stop was triggered by kill_switch: operator pressed stop." in text
    assert "Compensation is incomplete for: promote." in text
    assert "were cancelled: implement.core" in text
    kinds = [r[2].strip("`") for r in rows(run.render()[0], "Timeline of key events")]
    assert "safe_stop.triggered" in kinds
    assert "run.safe_stopped" in kinds


def test_limitations_for_scripted_approvals_waivers_posture_and_resumes() -> None:
    state = make_state(assurance="DEMO_SCRIPTED", resume_count=2)
    state.approvals = {
        "A-0001": approval(votes=[vote("bob", "scripted"), vote("carol", "scripted")])
    }
    state.waivers = [
        Waiver(
            "W-0001",
            "SEC-006",
            "k",
            "design",
            "e" * 64,
            "reviewed",
            ("bob",),
            "A-0001",
            "D-0003",
            T0,
        )
    ]
    text = "\n".join(limits(Run(state=state, posture="DEGRADED: breaker=20")))
    assert "Approvals were scripted (assurance DEMO_SCRIPTED), not a human review" in text
    assert "scripted votes on A-0001." in text
    assert "Governance posture is not nominal: DEGRADED: breaker=20." in text
    assert "1 policy finding(s) were waived by humans instead of fixed: W-0001 (SEC-006)." in text
    assert "resumed 2 time(s)" in text


def test_a_terminal_state_that_disagrees_with_the_audit_log_is_flagged() -> None:
    state = make_state(status=RunStatus.SUCCEEDED, status_reason="")
    text = "\n".join(limits(Run(state=state)))  # the events say run.failed
    assert "state.json says succeeded but the audit log says failed" in text


def test_a_paused_run_is_reported_as_a_snapshot() -> None:
    state = make_state(status=RunStatus.WAITING_HUMAN, status_reason="approval pending")
    stream = Stream()
    stream.at(0).add(ET.RUN_CREATED).add(ET.RUN_WAITING, reason="approval")
    text = "\n".join(limits(Run(state=state, stream=stream)))
    assert "The run is not finished (status waiting_human)" in text
    assert "does not match" not in text


# ----------------------------------------------------------------------------- sanitisation


def hostile_run() -> Run:
    state = make_state(
        scenario=HOSTILE,
        created_by=HOSTILE,
        status_reason=HOSTILE,
        autonomy_mode=HOSTILE,
    )
    state.nodes["verify"].reason = HOSTILE
    state.nodes["verify"].attempts[0].agent_id = HOSTILE
    state.approvals = {
        "A-0001": approval(votes=[vote(HOSTILE, HOSTILE, HOSTILE), vote("carol", "cli", HOSTILE)])
    }
    state.waivers = [
        Waiver("W-0001", HOSTILE, "k", "design", "e" * 64, HOSTILE, (HOSTILE,), "A-0001", "D-1", T0)
    ]
    return Run(
        state=state,
        stream=failed_run_stream(HOSTILE),
        decisions=[decision(1, DecisionKind.DESIGN, HOSTILE)],
        artifacts=[artifact("requirements", producer=HOSTILE)],
        posture=HOSTILE,
    )


def test_hostile_text_cannot_inject_markup_links_html_or_terminal_codes() -> None:
    markdown, _ = hostile_run().render()
    assert "\x1b" not in markdown
    assert "\u202e" not in markdown
    assert not re.search(r"(?<!\\)<", markdown)  # every angle bracket is escaped
    # an unescaped bracket may only open the ``[truncated N chars ...]`` marker of a clipped cell
    assert not re.search(r"(?<!\\)\[(?!truncated )", markdown)
    assert "](" not in markdown
    assert "://" not in markdown
    assert "www." not in markdown.lower()
    plain = re.sub(r"(?<!\\)`[A-Za-z0-9._@:+*?-]*`", "", markdown)  # code spans are literal
    assert not re.search(r"(?<!\\)@", plain)
    assert not re.search(r"(?<!\\)`code`", markdown)  # backticks of the payload are escaped
    assert "\\<script\\>" in markdown


def test_hostile_text_cannot_break_table_rows() -> None:
    markdown, _ = hostile_run().render()
    blocks: list[list[str]] = []
    previous_was_table = False
    for line in markdown.splitlines():
        if line.startswith("|"):
            if not previous_was_table:
                blocks.append([])
            blocks[-1].append(line)
            previous_was_table = True
        else:
            previous_was_table = False
    assert len(blocks) == 6
    for block in blocks:
        widths = {len(re.findall(r"(?<!\\)\|", line)) for line in block}
        assert len(widths) == 1, block  # every row has the header's number of cells


def test_hostile_text_is_plain_single_line_text_in_the_json() -> None:
    _, facts = hostile_run().render()
    flat = json.dumps(facts)
    assert "\\u001b" not in flat
    assert "\\u202e" not in flat
    assert "\\n" not in json.dumps(facts["run"]) and "\\n" not in json.dumps(facts["nodes"])
    assert "\\n" not in json.dumps(facts["timeline"]) and "\\n" not in json.dumps(
        facts["decisions"]
    )


def test_long_text_is_capped() -> None:
    state = make_state(status_reason="x" * 100_000)
    state.nodes["verify"].reason = "y" * 100_000
    markdown, facts = Run(state=state).render()
    assert len(markdown) < 40_000
    assert len(facts["run"]["status_reason"]) <= 200
    assert "[truncated" in facts["run"]["status_reason"]


def test_markdown_helpers() -> None:
    assert md_ident("release_patch@v1") == "`release_patch@v1`"
    assert md_ident("x`y|z\u202e") == "`x?y?z`"
    assert md_ident("") == "`-`"
    assert md_text("a_b *c* [d](e) <f> & g") == "a\\_b \\*c\\* \\[d\\]\\(e\\) \\<f\\> &amp; g"
    assert md_text("https://x.example/p mail@x.example www.x.example") == (
        "https:\\/\\/x.example/p mail\\@x.example www\\.x.example"
    )
    assert "\n" not in md_cell("a\nb|c")
    assert md_cell("a|b") == "a\\|b"
    assert md_text(None) == "None"


# --------------------------------------------------------------------- bounds and determinism


def test_output_is_deterministic_and_independent_of_input_order() -> None:
    run = Run()
    first = run.render()
    assert run.render() == first
    shuffled = Run()
    shuffled.decisions = list(reversed(run.decisions))
    shuffled.artifacts = list(reversed(run.artifacts))
    shuffled.state.approvals = dict(reversed(list(run.state.approvals.items())))
    shuffled.state.nodes = dict(reversed(list(run.state.nodes.items())))
    assert shuffled.render() == first


def test_long_lists_keep_head_and_tail_and_say_what_was_left_out() -> None:
    stream = Stream()
    stream.at(0).add(ET.RUN_CREATED)
    for i in range(700):
        stream.at(i).add(ET.NODE_RETRY_SCHEDULED, "a", i + 1, delay_s=i, not_before="x")
    stream.add(ET.RUN_FAILED)
    markdown, facts = Run(state=make_state(), stream=stream).render()
    assert len(facts["timeline"]) == 300
    assert facts["omitted"]["timeline"] == 401
    assert facts["timeline"][0]["seq"] == 2
    assert facts["timeline"][-1]["type"] == "run.failed"
    assert "401 more key events omitted" in markdown
    assert any(
        "Lists were shortened" in line and "timeline 401" in line for line in facts["limitations"]
    )


def test_a_run_without_nodes_events_decisions_or_artifacts_still_renders() -> None:
    state = make_state()
    state.nodes, state.approvals = {}, {}
    stream = Stream()
    stream.add(ET.RUN_CREATED)
    markdown, facts = Run(state=state, stream=stream, decisions=[], artifacts=[]).render()
    for text in (
        "No nodes were recorded.",
        "No key events were recorded.",
        "No approval was requested.",
        "No decision was recorded.",
        "No artifact was produced.",
    ):
        assert text in markdown
    assert facts["nodes"] == [] and facts["decisions"] == []


def test_the_report_contains_no_raw_html_and_no_links() -> None:
    markdown, _ = Run().render()
    assert not re.search(r"(?<!\\)<[A-Za-z/!]", markdown)
    assert not re.search(r"\]\(", markdown)
    assert "http" not in markdown


# ----------------------------------------------------------------------------- write_run_outputs


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_run_dir(tmp_path: Path, run: Run, *, index: bool = True, log: bool = True) -> Path:
    store = StateStore(tmp_path)
    directory = store.create(run.state.run_id)
    store.save(run.state)
    if index:
        (directory / "artifacts").mkdir()
        (directory / "artifacts" / "index.json").write_bytes(b'{"artifacts": []}\n')
    if log:
        (directory / "decisions.jsonl").write_bytes(b'{"id": "D-0001"}\n')
    return directory


def test_write_run_outputs_writes_the_five_files_and_returns_their_hashes(tmp_path: Path) -> None:
    run = Run()
    directory = prepared_run_dir(tmp_path, run)
    hashes = write_run_outputs(directory, **run.kwargs())
    assert tuple(hashes) == REPORT_FILES
    for name, digest in hashes.items():
        assert sha256_of(directory / name) == digest
    markdown, facts = run.render()
    assert (directory / "report.md").read_text(encoding="utf-8") == markdown
    assert json.loads((directory / "report.json").read_text(encoding="ascii")) == facts
    assert json.loads((directory / "metrics.json").read_text(encoding="ascii")) == (
        run.metrics.to_dict()
    )
    assert json.loads((directory / "trace.json").read_text(encoding="ascii")) == json.loads(
        json.dumps(export_trace(run.events))
    )
    assert b"\r\n" not in (directory / "report.md").read_bytes()


def test_seal_records_the_audit_head_and_the_hashes_of_the_record_files(tmp_path: Path) -> None:
    run = Run()
    directory = prepared_run_dir(tmp_path, run)
    hashes = write_run_outputs(directory, **run.kwargs())
    seal = json.loads((directory / "seal.json").read_text(encoding="ascii"))
    assert seal["head_seq"] == run.events[-1].seq
    assert seal["head_hash"] == run.events[-1].hash
    assert seal["count"] == len(run.events)
    assert seal["state_sha256"] == sha256_of(directory / "state.json")
    assert seal["artifact_index_sha256"] == sha256_of(directory / "artifacts" / "index.json")
    assert seal["decisions_sha256"] == sha256_of(directory / "decisions.jsonl")
    assert seal["files"] == {
        "artifacts/index.json": seal["artifact_index_sha256"],
        "decisions.jsonl": seal["decisions_sha256"],
        "metrics.json": hashes["metrics.json"],
        "report.json": hashes["report.json"],
        "report.md": hashes["report.md"],
        "state.json": seal["state_sha256"],
        "trace.json": hashes["trace.json"],
    }
    assert seal["run_id"] == "run-1"


def test_seal_marks_missing_optional_files_as_null(tmp_path: Path) -> None:
    run = Run()
    directory = prepared_run_dir(tmp_path, run, index=False, log=False)
    write_run_outputs(directory, **run.kwargs())
    seal = json.loads((directory / "seal.json").read_text(encoding="ascii"))
    assert seal["artifact_index_sha256"] is None
    assert seal["decisions_sha256"] is None
    assert "decisions.jsonl" not in seal["files"]
    assert "artifacts/index.json" not in seal["files"]
    assert seal["state_sha256"] == sha256_of(directory / "state.json")


def test_seal_of_an_empty_log_uses_the_genesis_head(tmp_path: Path) -> None:
    run = Run(stream=Stream())
    directory = prepared_run_dir(tmp_path, run)
    write_run_outputs(directory, **run.kwargs())
    seal = json.loads((directory / "seal.json").read_text(encoding="ascii"))
    assert (seal["head_seq"], seal["head_hash"], seal["count"]) == (0, GENESIS_HASH, 0)


def test_a_changed_state_file_changes_the_seal(tmp_path: Path) -> None:
    run = Run()
    directory = prepared_run_dir(tmp_path, run)
    write_run_outputs(directory, **run.kwargs())
    before = json.loads((directory / "seal.json").read_text(encoding="ascii"))
    run.state.status_reason = "changed afterwards"
    StateStore(tmp_path).save(run.state)
    write_run_outputs(directory, **run.kwargs())
    after = json.loads((directory / "seal.json").read_text(encoding="ascii"))
    assert after["state_sha256"] != before["state_sha256"]
    assert after["head_hash"] == before["head_hash"]


def test_rewriting_identical_inputs_gives_identical_files_and_leaves_no_temp_files(
    tmp_path: Path,
) -> None:
    run = Run()
    directory = prepared_run_dir(tmp_path, run)
    first = write_run_outputs(directory, **run.kwargs())
    assert write_run_outputs(directory, **run.kwargs()) == first
    leftovers = sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))
    assert leftovers == []


def test_sealing_refuses_a_link_where_a_record_file_should_be(tmp_path: Path) -> None:
    run = Run()
    directory = prepared_run_dir(tmp_path, run, log=False)
    target = tmp_path / "elsewhere.jsonl"
    target.write_bytes(b"{}\n")
    try:
        (directory / "decisions.jsonl").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    with pytest.raises(IntegrityError, match=r"decisions\.jsonl"):
        write_run_outputs(directory, **run.kwargs())
    assert not (directory / "seal.json").exists()  # the seal is written last


def test_hostile_run_is_written_without_error(tmp_path: Path) -> None:
    run = hostile_run()
    directory = prepared_run_dir(tmp_path, run)
    hashes = write_run_outputs(directory, **run.kwargs())
    text = (directory / "report.md").read_text(encoding="utf-8")
    assert "\x1b" not in text
    assert set(hashes) == set(REPORT_FILES)


def make_dir_link(link: Path, target: Path) -> None:
    """A symbolic link, or on Windows a junction (needs no privilege), or skip."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("symbolic links are not available here")
        made = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        if made.returncode != 0:
            pytest.skip("neither symbolic links nor junctions can be created here")


def test_sealing_refuses_a_linked_artifact_directory(tmp_path: Path) -> None:
    run = Run()
    directory = prepared_run_dir(tmp_path, run, index=False, log=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "index.json").write_bytes(b"{}\n")
    make_dir_link(directory / "artifacts", elsewhere)
    with pytest.raises(IntegrityError, match=r"artifacts/index\.json"):
        write_run_outputs(directory, **run.kwargs())
    assert not (directory / "seal.json").exists()


def test_outputs_of_a_run_over_real_audit_log_stores_and_state(tmp_path: Path) -> None:
    """The seal agrees with what the real audit log, artifact store and decision log wrote."""
    clock = FakeClock()
    states = StateStore(tmp_path)
    directory = states.create("run-real")
    audit = AuditLog(directory / "audit.jsonl", run_id="run-real", clock=clock)
    store = ArtifactStore(directory / "artifacts", clock=clock, audit=audit)
    log = DecisionLog(directory / "decisions.jsonl", run_id="run-real", clock=clock, audit=audit)
    audit.emit(ET.RUN_CREATED, data={"scenario": "demo"})
    clock.advance(1)
    started = {"stage": "intake", "agent_id": "intake", "epoch": "primary", "generation": 1}
    audit.emit(ET.NODE_STARTED, node_id="intake", attempt=1, data=started)
    clock.advance(2)
    store.put(
        ArtifactDraft(kind="intake_report", content={"schema": 1}),
        producer="intake",
        attempt=1,
        agent_id="intake",
    )
    log.record(DecisionKind.DESIGN, "Use SQLite", actor=Actor.agent("architect"), node_id="intake")
    done = {"stage": "intake", "outputs": [], "cache_hit": False}
    audit.emit(ET.NODE_SUCCEEDED, node_id="intake", attempt=1, data=done)
    audit.emit(ET.RUN_SUCCEEDED, data={"reason": "done"})
    state = make_state(run_id="run-real", status=RunStatus.SUCCEEDED, status_reason="")
    state.nodes = {"intake": node("intake", NodeStatus.SUCCEEDED)}
    state.approvals = {}
    states.save(state)
    events = audit.events()
    hashes = write_run_outputs(
        directory,
        state=state,
        events=events,
        decisions=log.all(),
        metrics=compute_run_metrics(events),
        artifacts=store.all(include_inactive=True),
        posture="NOMINAL",
    )
    seal = json.loads((directory / "seal.json").read_text(encoding="ascii"))
    assert (seal["head_seq"], seal["head_hash"]) == audit.head()
    assert seal["count"] == len(events)
    assert seal["artifact_index_sha256"] == sha256_of(directory / "artifacts" / "index.json")
    assert seal["decisions_sha256"] == sha256_of(directory / "decisions.jsonl")
    assert seal["state_sha256"] == sha256_of(directory / "state.json")
    assert hashes["seal.json"] == sha256_of(directory / "seal.json")
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "| `intake_report@v1` | `intake_report` | `active` | `intake` |" in report
    assert "| `D-0001` | `design` | `agent:architect` | Use SQLite |" in report
    assert "- **Status:** `succeeded`" in report
    assert AuditLog.verify_file(directory / "audit.jsonl").ok


def test_decision_ids_sort_numerically_beyond_four_digits() -> None:
    run = Run(decisions=[decision(10000), decision(9999), decision(2)])
    ids = [d["id"] for d in run.render()[1]["decisions"]]
    assert ids == ["D-0002", "D-9999", "D-10000"]


def test_limitations_count_every_node_even_when_the_table_is_clipped() -> None:
    state = make_state()
    state.nodes = {
        f"n{i:03d}": node(f"n{i:03d}", NodeStatus.FAILED, reason="boom") for i in range(350)
    }
    markdown, facts = Run(state=state).render()
    assert len(facts["nodes"]) == 300
    assert facts["omitted"]["nodes"] == 50
    text = "\n".join(facts["limitations"])
    assert "350 node(s) failed: n000 (boom), n001 (boom)" in text
    assert ", +342 more." in text
    assert "50 more nodes omitted" in markdown
