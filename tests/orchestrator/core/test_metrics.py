"""Run metrics (engine-semantics §17.8) computed from synthetic audit event streams.

The tests build ``AuditEvent`` streams by hand (time only moves when told to), so every expected
number can be derived on paper and nothing depends on the engine.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.audit import AUDIT_FILENAME, AuditLog
from orchestrator.core.clock import FakeClock
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.metrics import (
    AggregateMetrics,
    Incident,
    IncidentStatus,
    RunMetrics,
    aggregate,
    compute_run_metrics,
    load_all,
    percentile,
    render_table,
)
from orchestrator.core.types import Actor, AuditEvent, EventType, NodeSpec, Stage

ET = EventType
T0 = datetime(2026, 1, 1, tzinfo=UTC)


class Stream:
    """Builds an event stream; ``at(seconds)`` sets the absolute time of the next events."""

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
                prev_hash="0" * 64,
                hash=f"{seq:064x}",
            )
        )
        return self

    def start(self, node: str, attempt: int, stage: str = "implementation") -> Stream:
        return self.add(
            ET.NODE_STARTED,
            node,
            attempt,
            stage=stage,
            agent_id="agent",
            epoch="primary",
            generation=1,
        )

    def done(self, node: str, attempt: int) -> Stream:
        return self.add(ET.NODE_SUCCEEDED, node, attempt, stage="x", outputs=[], cache_hit=False)

    def fail(self, node: str, attempt: int, kind: str = "transient") -> Stream:
        return self.add(
            ET.NODE_ATTEMPT_FAILED,
            node,
            attempt,
            kind=kind,
            message="m",
            epoch="primary",
            retryable=True,
            rule_ids=[],
        )

    def retry(self, node: str, attempt: int) -> Stream:
        return self.add(ET.NODE_RETRY_SCHEDULED, node, attempt, delay_s=0.2, not_before="x")


def metrics(stream: Stream, **kwargs: Any) -> RunMetrics:
    return compute_run_metrics(stream.events, **kwargs)


# ----------------------------------------------------------------- scenario streams


def clean_success(run_id: str = "run-ok") -> Stream:
    s = Stream(run_id)
    s.at(0).add(ET.RUN_CREATED).start("a", 1, "intake")
    s.at(10).done("a", 1).start("b", 1, "design")
    s.at(25).done("b", 1).add(ET.RUN_SUCCEEDED)
    s.at(26).add(ET.RUN_SEALED)
    return s


def retry_then_success(run_id: str = "run-retry") -> Stream:
    s = Stream(run_id)
    s.at(0).add(ET.RUN_CREATED).start("a", 1)
    s.at(5).fail("a", 1).retry("a", 1)
    s.at(7).start("a", 2)
    s.at(12).done("a", 2).add(ET.RUN_SUCCEEDED)
    return s


def failure_unrecovered(run_id: str = "run-fail") -> Stream:
    s = Stream(run_id)
    s.at(0).add(ET.RUN_CREATED).start("a", 1)
    s.at(3).fail("a", 1).retry("a", 1)
    s.at(4).start("a", 2)
    s.at(6).fail("a", 2).add(ET.NODE_FAILED, "a", 2, kind="transient", message="gave up")
    s.at(7).add(ET.RUN_FAILED)
    return s


def safe_stopped(run_id: str = "run-stop") -> Stream:
    s = Stream(run_id)
    s.at(0).add(ET.RUN_CREATED).start("a", 1)
    s.at(4).done("a", 1).start("b", 1)
    s.at(6).fail("b", 1).retry("b", 1)
    s.at(7).start("b", 2)
    s.at(9).add(ET.SAFE_STOP_TRIGGERED, trigger="kill_switch", reason="operator")
    s.add(ET.NODE_CANCELLED, "b")
    s.at(10).add(ET.ROLLBACK_STARTED, "a", level="node").add(ET.NODE_ROLLED_BACK, "a").add(
        ET.ROLLBACK_COMPLETED, "a", level="node"
    )
    s.at(11).add(ET.RUN_SAFE_STOPPED)
    return s


def waiting(run_id: str = "run-wait") -> Stream:
    s = Stream(run_id)
    s.at(0).add(ET.RUN_CREATED).start("a", 1)
    s.at(4).add(ET.NODE_AWAITING_APPROVAL, "a", 1)
    s.at(5).add(ET.RUN_WAITING, reason="approvals", approvals=["A-0001"], questions=[])
    s.at(30).add(ET.COMMAND_QUEUED)
    return s


# ----------------------------------------------------------------- single-run definitions


class TestCleanSuccess:
    def test_totals(self) -> None:
        m = metrics(clean_success())
        assert (m.run_id, m.status, m.success) == ("run-ok", "succeeded", True)
        assert (
            m.e2e_latency_s == 25.0
        )  # up to the terminal event; the trailing run.sealed is bookkeeping
        assert (m.active_latency_s, m.human_wait_s) == (25.0, 0.0)
        assert (m.started_at, m.ended_at) == (T0, T0 + timedelta(seconds=25))

    def test_attempts_and_rates(self) -> None:
        m = metrics(clean_success())
        assert (m.nodes, m.attempts, m.countable_attempts) == (2, 2, 2)
        assert (m.retries, m.retry_frequency, m.fallbacks, m.fallback_rate) == (0, 0.0, 0, 0.0)
        assert (m.rollbacks, m.safe_stops, m.replans, m.cache_hits) == (0, 0, 0, 0)
        assert m.first_pass_yield == 1.0 and m.replan_reuse_ratio is None
        assert m.incidents == () and m.mttr_s is None

    def test_latencies_and_structure(self) -> None:
        m = metrics(clean_success())
        assert m.node_latency_s == {"a": 10.0, "b": 15.0}
        assert m.stage_latency_s == {"design": 15.0, "intake": 10.0}
        assert m.critical_path_s == 25.0
        assert m.max_concurrency_observed == 1


class TestRetries:
    def test_retry_then_success(self) -> None:
        m = metrics(retry_then_success())
        assert (m.attempts, m.retries, m.retry_frequency) == (2, 1, 0.5)
        assert m.first_pass_yield == 0.0  # the only node needed a retry
        assert m.node_latency_s == {
            "a": 10.0
        }  # 5 s + 5 s of execution; the back-off wait is not execution
        (incident,) = m.incidents
        assert incident == Incident(
            "a",
            T0 + timedelta(seconds=5),
            T0 + timedelta(seconds=12),
            7.0,
            IncidentStatus.RECOVERED,
        )
        assert m.mttr_s == 7.0

    def test_only_the_first_failure_opens_an_incident(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(2).fail("a", 1).retry("a", 1).start("a", 2)
        s.at(4).fail("a", 2).retry("a", 2).start("a", 3)
        s.at(9).done("a", 3).add(ET.RUN_SUCCEEDED)
        m = metrics(s)
        assert len(m.incidents) == 1 and m.mttr_s == 7.0
        assert (m.attempts, m.retries) == (3, 2) and m.retry_frequency == pytest.approx(2 / 3)

    def test_fallback(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1).start("b", 1)
        s.at(1).done("b", 1)
        s.at(2).fail("a", 1).retry("a", 1).start("a", 2)
        s.at(5).fail("a", 2).retry("a", 2).start("a", 3)
        s.at(9).fail("a", 3).add(ET.NODE_FALLBACK_ACTIVATED, "a", from_agent="p", to_agent="f")
        s.at(10).start("a", 4)
        s.at(20).done("a", 4).add(ET.RUN_SUCCEEDED)
        m = metrics(s)
        assert (m.fallbacks, m.nodes, m.fallback_rate) == (1, 2, 0.5)
        assert (m.attempts, m.retries) == (4 + 1, 2)
        (incident,) = m.incidents
        assert (
            incident.status is IncidentStatus.RECOVERED and incident.mttr_s == 18.0
        )  # first failure -> final success
        assert m.first_pass_yield == 0.5  # b passed first time, a did not

    def test_unrecovered_failure(self) -> None:
        m = metrics(failure_unrecovered())
        assert (m.status, m.success) == ("failed", False)
        (incident,) = m.incidents
        assert incident == Incident(
            "a", T0 + timedelta(seconds=3), None, None, IncidentStatus.UNRECOVERED
        )
        assert m.mttr_s is None and m.first_pass_yield == 0.0
        assert (m.attempts, m.retries) == (2, 1)

    def test_incident_still_open_when_the_log_ends_counts_as_unrecovered(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(3).fail("a", 1).retry("a", 1)
        (incident,) = metrics(s).incidents
        assert incident.status is IncidentStatus.UNRECOVERED

    def test_cancelled_attempts_never_open_incidents(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(2).fail("a", 1, kind="cancelled")
        assert metrics(s).incidents == ()

    @pytest.mark.parametrize("closer", [ET.NODE_RESET, ET.NODE_REMOVED, ET.NODE_CANCELLED])
    def test_resets_removals_and_cancellations_abandon_the_incident(
        self, closer: EventType
    ) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(2).fail("a", 1)
        s.at(3).add(closer, "a")
        s.at(4).start("a", 2)
        s.at(6).fail("a", 2).add(ET.NODE_FAILED, "a", 2, kind="transient", message="")
        m = metrics(s)
        assert [i.status for i in m.incidents] == [
            IncidentStatus.ABANDONED,
            IncidentStatus.UNRECOVERED,
        ]
        assert m.incidents[0].recovered_at is None and m.incidents[0].mttr_s is None

    def test_cache_hit_recovers_an_incident(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(2).fail("a", 1)
        s.at(8).add(ET.NODE_CACHE_HIT, "a", fingerprint="f").add(
            ET.NODE_SUCCEEDED, "a", stage="x", outputs=[], cache_hit=True
        )
        (incident,) = metrics(s).incidents
        assert incident.status is IncidentStatus.RECOVERED and incident.mttr_s == 6.0

    def test_mttr_is_the_mean_over_recovered_incidents_only(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1).start("b", 1).start("c", 1)
        s.at(1).fail("a", 1).fail("b", 1).fail("c", 1)
        s.at(5).done("a", 2).done("b", 2)  # recovered after 4 s each
        s.at(7).add(ET.NODE_FAILED, "c", 1, kind="fatal", message="")
        m = metrics(s)
        assert sorted(i.status.value for i in m.incidents) == [
            "recovered",
            "recovered",
            "unrecovered",
        ]
        assert m.mttr_s == 4.0


class TestHumanWait:
    def stream(self) -> Stream:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(5).fail("a", 1, kind="gate_rejected").retry("a", 1)
        s.at(6).start("a", 2)
        s.at(8).add(ET.NODE_AWAITING_APPROVAL, "a", 2)
        s.at(10).add(ET.RUN_WAITING, reason="approvals", approvals=["A-0001"], questions=[])
        s.at(1010).add(ET.RUN_RESUMED)
        s.at(1012).add(ET.NODE_PUBLISHING, "a", 2, staged=[], expansion_hash="").done("a", 2).add(
            ET.RUN_SUCCEEDED
        )
        return s

    def test_human_wait_is_excluded_from_active_latency_and_mttr(self) -> None:
        m = metrics(self.stream())
        assert m.e2e_latency_s == 1012.0
        assert m.human_wait_s == 1000.0 and m.active_latency_s == 12.0
        (incident,) = m.incidents
        assert (
            incident.mttr_s == 7.0
        )  # 1007 s wall clock minus the 1000 s the run waited for a human
        assert m.mttr_s == 7.0

    def test_waiting_for_approval_is_not_execution_time(self) -> None:
        m = metrics(self.stream())
        assert m.node_latency_s == {
            "a": 7.0
        }  # 5 s + 2 s: the approval wait does not count as node latency
        assert m.critical_path_s == 7.0

    def test_an_open_wait_at_the_end_runs_to_the_last_event(self) -> None:
        m = metrics(waiting())
        assert (m.status, m.success) == ("waiting_human", False)
        assert m.e2e_latency_s == 30.0 and m.human_wait_s == 25.0 and m.active_latency_s == 5.0

    def test_a_terminal_event_ends_the_wait(self) -> None:
        s = waiting()
        s.at(50).add(ET.RUN_SAFE_STOPPED)
        m = metrics(s)
        assert m.human_wait_s == 45.0 and m.e2e_latency_s == 50.0 and m.status == "safe_stopped"

    def test_several_waits_add_up_and_stray_events_are_ignored(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).add(ET.RUN_RESUMED)  # resumed without waiting: no interval
        s.at(10).add(ET.RUN_WAITING).add(ET.RUN_WAITING)  # duplicate wait while one is open
        s.at(20).add(ET.RUN_RESUMED)
        s.at(30).add(ET.RUN_WAITING)
        s.at(35).add(ET.RUN_RESUMED)
        s.at(40).add(ET.RUN_SUCCEEDED)
        m = metrics(s)
        assert m.human_wait_s == 15.0 and m.active_latency_s == 25.0

    def test_waits_are_clipped_to_the_run_window(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED)
        s.at(10).add(ET.RUN_WAITING)
        s.at(20).add(ET.RUN_SUCCEEDED)  # terminal event closes the wait; nothing after it counts
        s.at(500).add(ET.RUN_SEALED)
        m = metrics(s)
        assert m.e2e_latency_s == 20.0 and m.human_wait_s == 10.0


class TestSafeStop:
    def test_counts(self) -> None:
        m = metrics(safe_stopped())
        assert (m.status, m.success) == ("safe_stopped", False)
        assert (m.safe_stops, m.rollbacks) == (1, 1)
        assert m.e2e_latency_s == 11.0

    def test_cancelled_attempts_are_left_out_of_the_retry_denominator(self) -> None:
        m = metrics(safe_stopped())
        assert m.attempts == 3  # a#1, b#1, b#2
        assert m.countable_attempts == 2  # b#2 was cancelled by the stop
        assert (m.retries, m.retry_frequency) == (1, 0.5)

    def test_in_flight_incident_is_abandoned(self) -> None:
        m = metrics(safe_stopped())
        (incident,) = m.incidents
        assert incident.node_id == "b" and incident.status is IncidentStatus.ABANDONED

    def test_safe_stop_abandons_every_open_incident(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1).start("b", 1)
        s.at(1).fail("a", 1).fail("b", 1)
        s.at(2).add(ET.RUN_SAFE_STOPPED)
        assert [i.status for i in metrics(s).incidents] == [IncidentStatus.ABANDONED] * 2

    def test_only_node_and_run_compensations_count_as_rollbacks(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED)
        s.add(ET.ARTIFACT_QUARANTINED, data="x").add(
            ET.ARTIFACT_QUARANTINED
        )  # attempt-level: reported separately
        s.add(ET.NODE_ROLLED_BACK, "a").add(ET.NODE_ROLLED_BACK, "b").add(ET.RUN_SAFE_STOPPED)
        m = metrics(s)
        assert m.rollbacks == 2 and m.quarantines == 2


class TestAttemptAccounting:
    def test_discarded_cancelled_and_needs_input_attempts_are_excluded(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED)
        s.start("a", 1).add(ET.NODE_ATTEMPT_DISCARDED, "a", 1, generation=1, reason="stale")
        s.start("a", 2).add(ET.NODE_AWAITING_INPUT, "a", 2)
        s.start("a", 3).fail("a", 3, kind="cancelled")
        s.start("a", 4).fail("a", 4, kind="transient").retry("a", 4)
        s.start("a", 5).done("a", 5).add(ET.RUN_SUCCEEDED)
        m = metrics(s)
        assert m.attempts == 5 and m.countable_attempts == 2
        assert (m.retries, m.retry_frequency) == (1, 0.5)

    def test_an_attempt_that_published_or_awaited_approval_still_counts(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1).add(ET.NODE_AWAITING_APPROVAL, "a", 1)
        s.start("b", 1).add(ET.NODE_PUBLISHING, "b", 1, staged=[], expansion_hash="")
        assert metrics(s).countable_attempts == 2

    def test_no_attempts_means_zero_rates_not_division_errors(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).at(1).add(ET.RUN_SUCCEEDED)
        m = metrics(s)
        assert (m.attempts, m.retry_frequency, m.fallback_rate, m.first_pass_yield) == (
            0,
            0.0,
            0.0,
            0.0,
        )
        assert m.critical_path_s == 0.0 and m.max_concurrency_observed == 0

    def test_first_pass_yield_counts_nodes_that_ran(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1).start("b", 1).start("c", 1).start("d", 1)
        s.at(1).done("a", 1).done("b", 1).fail("c", 1).done("d", 1)
        s.at(2).done("c", 2).add(ET.NODE_ATTEMPT_DISCARDED, "d", 0, generation=0, reason="x")
        m = metrics(s)
        assert m.first_pass_yield == 0.75  # a, b, d first time; c needed a retry

    def test_events_without_a_stage_get_an_unknown_bucket(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).add(ET.NODE_STARTED, "a", 1)
        s.at(2).done("a", 1)
        assert metrics(s).stage_latency_s == {"unknown": 2.0}

    def test_hostile_stage_and_rule_names_are_sanitised(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1, stage="\x1b[31mred\x1b[0m" + "x" * 200)
        s.at(1).done("a", 1)
        s.add(
            ET.POLICY_VIOLATION,
            "a",
            rule_id="\x1b[2Jevil",
            severity="HIGH",
            enforcement="deny",
            location="x",
        )
        s.add(ET.POLICY_VIOLATION, "a", severity="HIGH", enforcement="deny", location="x")
        m = metrics(s)
        keys = [*m.stage_latency_s, *m.policy_violations_by_rule]
        assert all("\x1b" not in k and len(k) <= 80 for k in keys)
        assert m.policy_violations_by_rule == {"evil": 1, "unknown": 1}


class TestReplanAndCache:
    def stream(self) -> Stream:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1).start("b", 1).start("c", 1)
        s.at(1).done("a", 1).done("b", 1).done("c", 1)
        s.at(5).add(ET.INPUT_CHANGED, kind="request", key="", old_ref="", new_ref="", by="alice")
        s.add(
            ET.PLAN_REPLANNED,
            decision_id="D-1",
            added=[],
            removed=["c"],
            retained=["a", "b"],
            invalidated=["a", "b", "c"],
        )
        s.add(ET.NODE_STALE, "a").add(ET.NODE_STALE, "b").add(ET.NODE_STALE, "c")
        s.at(6).add(ET.NODE_CACHE_HIT, "a", fingerprint="f").add(
            ET.NODE_SUCCEEDED, "a", stage="x", outputs=[], cache_hit=True
        )
        s.add(ET.NODE_REMOVED, "c")
        s.at(7).start("b", 2)
        s.at(9).done("b", 2).add(ET.RUN_SUCCEEDED)
        return s

    def test_reuse_ratio_is_cache_hits_over_revalidated_nodes(self) -> None:
        m = metrics(self.stream())
        assert (m.replans, m.cache_hits) == (1, 1)
        assert (
            m.revalidated_nodes == 2
        )  # a hit the cache, b ran again; c was removed, not revalidated
        assert m.replan_reuse_ratio == 0.5

    def test_everything_reused(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).add(ET.NODE_STALE, "a").add(
            ET.NODE_CACHE_HIT, "a", fingerprint="f"
        )
        assert metrics(s).replan_reuse_ratio == 1.0

    def test_nothing_reused(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).add(ET.NODE_STALE, "a").start("a", 2)
        assert metrics(s).replan_reuse_ratio == 0.0

    def test_no_revalidation_means_no_ratio(self) -> None:
        assert metrics(clean_success()).replan_reuse_ratio is None

    def test_a_first_run_start_is_not_a_miss(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)  # never stale
        assert metrics(s).revalidated_nodes == 0


class TestStructure:
    def overlapping(self, count: int) -> Stream:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED)
        for i in range(count):
            s.start(f"n{i}", 1)  # every timestamp is identical: only the log order can show overlap
        for i in range(count):
            s.done(f"n{i}", 1)
        return s

    @pytest.mark.parametrize("count", [1, 2, 3, 5])
    def test_concurrency_comes_from_log_order_not_timestamps(self, count: int) -> None:
        assert metrics(self.overlapping(count)).max_concurrency_observed == count

    def test_sequential_attempts_never_overlap(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1).done("a", 1).start("b", 1).done("b", 1)
        assert metrics(s).max_concurrency_observed == 1

    def test_partial_overlap(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1).start("b", 1).done("a", 1).start("c", 1).done(
            "b", 1
        ).done("c", 1)
        assert metrics(s).max_concurrency_observed == 2

    def test_two_attempts_of_one_node_can_overlap(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(1).start("a", 2).add(ET.NODE_ATTEMPT_DISCARDED, "a", 1, generation=1, reason="stale")
        s.at(2).done("a", 2)
        assert metrics(s).max_concurrency_observed == 2

    def test_a_restart_closes_the_attempts_the_crash_left_open(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)  # the process died here
        s.at(30).add(ET.RUN_RESUMED)
        s.at(31).start("a", 2)
        s.at(35).done("a", 2).add(ET.RUN_SUCCEEDED)
        m = metrics(s)
        assert m.max_concurrency_observed == 1
        assert m.node_latency_s == {"a": 34.0}  # 30 s until the resume + 4 s

    def test_spans_still_open_at_the_end_are_closed_there(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(9).add(ET.RUN_SAFE_STOPPED)
        assert metrics(s).node_latency_s == {"a": 9.0}

    def test_stage_latency_is_the_union_of_parallel_spans(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1, "implementation")
        s.at(5).start("b", 1, "implementation")
        s.at(10).done("a", 1)
        s.at(15).done("b", 1)
        s.at(20).start("c", 1, "implementation")
        s.at(23).done("c", 1)
        m = metrics(s)
        assert m.stage_latency_s == {"implementation": 18.0}  # [0,15] merged + [20,23]
        assert sum(m.node_latency_s.values()) == 23.0  # the per-node view double counts the overlap

    def test_approval_requests_and_policy_violations(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED)
        for _ in range(2):
            s.add(
                ET.APPROVAL_REQUESTED,
                "design",
                approval_id="A",
                purpose="node_output",
                risk="HIGH",
                required=2,
                node_fingerprint="f",
            )
        s.add(ET.APPROVAL_AUTO, "docs", reason="low risk")
        for rule in ("SEC-001", "CHG-001", "SEC-001"):
            s.add(
                ET.POLICY_VIOLATION,
                "x",
                rule_id=rule,
                severity="HIGH",
                enforcement="deny",
                location="p",
            )
        m = metrics(s)
        assert (m.approvals_requested, m.approvals_auto) == (2, 1)
        assert m.policy_violations_by_rule == {"CHG-001": 1, "SEC-001": 2}


class TestCriticalPath:
    def diamond_events(self, *, serial: bool) -> Stream:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(10).done("a", 1).start("b", 1)
        if serial:  # a scheduler that ran the two branches one after the other
            s.at(15).done("b", 1).start("c", 1)
            s.at(35).done("c", 1).start("d", 1)
            s.at(40).done("d", 1).add(ET.RUN_SUCCEEDED)
        else:
            s.start("c", 1)
            s.at(15).done("b", 1)
            s.at(30).done("c", 1).start("d", 1)
            s.at(35).done("d", 1).add(ET.RUN_SUCCEEDED)
        return s

    @staticmethod
    def diamond_graph() -> WorkflowGraph:
        def node(node_id: str, deps: tuple[str, ...] = ()) -> NodeSpec:
            return NodeSpec(
                id=node_id, title=node_id, stage=Stage.IMPLEMENTATION, agent="x", depends_on=deps
            )

        return WorkflowGraph(
            [node("a"), node("b", ("a",)), node("c", ("a",)), node("d", ("b", "c"))]
        )

    def test_without_a_graph_it_is_the_longest_chain_of_non_overlapping_spans(self) -> None:
        m = metrics(self.diamond_events(serial=False))
        assert m.critical_path_s == 35.0  # a(10) -> c(20) -> d(5); b ran in parallel with c

    def test_with_a_graph_it_is_the_exact_dependency_path(self) -> None:
        m = metrics(self.diamond_events(serial=False), graph=self.diamond_graph())
        assert m.critical_path_s == 35.0

    def test_serial_execution_inflates_the_graphless_bound_only(self) -> None:
        stream = self.diamond_events(serial=True)
        assert metrics(stream).critical_path_s == 40.0  # upper bound: everything ran back to back
        assert metrics(stream, graph=self.diamond_graph()).critical_path_s == 35.0  # a -> c -> d

    def test_nodes_of_the_graph_that_never_ran_weigh_nothing(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(4).done("a", 1).add(ET.RUN_SUCCEEDED)
        assert metrics(s, graph=self.diamond_graph()).critical_path_s == 4.0

    def test_graph_never_changes_the_other_numbers(self) -> None:
        stream = self.diamond_events(serial=False)
        plain, with_graph = metrics(stream), metrics(stream, graph=self.diamond_graph())
        assert dataclasses.replace(with_graph, critical_path_s=plain.critical_path_s) == plain


class TestRunStatus:
    def test_status_follows_the_last_run_level_event(self) -> None:
        s = Stream()
        s.at(0).add(ET.RUN_CREATED)
        assert metrics(s).status == "running"
        s.add(ET.RUN_WAITING)
        assert metrics(s).status == "waiting_human"
        s.add(ET.RUN_SAFE_STOPPED).add(ET.RUN_RESUMED)
        assert metrics(s).status == "running"  # an override resumed it
        s.add(ET.RUN_SUCCEEDED)
        assert metrics(s).status == "succeeded"

    def test_streams_without_run_events(self) -> None:
        s = Stream()
        s.at(0).start("a", 1)
        assert metrics(s).status == "unknown" and not metrics(s).success

    def test_empty_streams_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_run_metrics([])


class TestSerialisation:
    @pytest.mark.parametrize(
        "make", [clean_success, retry_then_success, failure_unrecovered, safe_stopped, waiting]
    )
    def test_run_metrics_round_trip_through_json(self, make: Any) -> None:
        original = metrics(make())
        restored = RunMetrics.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored == original

    def test_to_dict_is_plain_json(self) -> None:
        document = metrics(retry_then_success()).to_dict()
        assert document["started_at"] == T0.isoformat()
        assert document["incidents"][0]["status"] == "recovered"
        json.dumps(document)  # raises on anything that is not plain JSON


# ----------------------------------------------------------------- aggregation


def all_runs() -> list[RunMetrics]:
    return [
        metrics(make())
        for make in (clean_success, retry_then_success, failure_unrecovered, safe_stopped, waiting)
    ]


class TestAggregate:
    def test_status_counts_and_finished_run_denominators(self) -> None:
        total = aggregate(all_runs())
        assert (
            total.runs,
            total.succeeded,
            total.failed,
            total.safe_stopped,
            total.unfinished,
        ) == (5, 2, 1, 1, 1)
        assert total.success_rate == 2 / 4  # the paused run is not (yet) a success or a failure
        assert total.safe_stop_rate == 1 / 4

    def test_rates_are_pooled_over_the_underlying_counts(self) -> None:
        runs = all_runs()
        total = aggregate(runs)
        assert total.retry_rate == sum(r.retries for r in runs) / sum(
            r.countable_attempts for r in runs
        )
        assert total.retry_rate != sum(r.retry_frequency for r in runs) / len(
            runs
        )  # not a mean of ratios
        assert total.fallback_rate == 0.0
        assert total.rollback_frequency == 1 / 5  # one compensation across five runs
        assert total.first_pass_yield == pytest.approx(sum(r.first_pass_yield for r in runs) / 5)

    def test_incident_statistics(self) -> None:
        total = aggregate(all_runs())
        assert (
            total.incidents,
            total.incidents_recovered,
            total.incidents_unrecovered,
            total.incidents_abandoned,
        ) == (3, 1, 1, 1)
        assert total.mttr_mean_s == 7.0 and total.mttr_p95_s == 7.0

    def test_mttr_percentiles_use_every_recovered_incident(self) -> None:
        a, b = metrics(retry_then_success("r1")), metrics(self.long_recovery())
        total = aggregate([a, b])
        assert sorted(i.mttr_s for r in (a, b) for i in r.incidents if i.mttr_s is not None) == [
            7.0,
            18.0,
        ]
        assert total.mttr_mean_s == 12.5
        assert total.mttr_p95_s == pytest.approx(7 + 11 * 0.95)

    @staticmethod
    def long_recovery() -> Stream:
        s = Stream("r2")
        s.at(0).add(ET.RUN_CREATED).start("a", 1)
        s.at(2).fail("a", 1).retry("a", 1).start("a", 2)
        s.at(20).done("a", 2).add(ET.RUN_SUCCEEDED)
        return s

    def test_latency_percentiles(self) -> None:
        total = aggregate(all_runs())  # e2e of the five runs: 25, 12, 7, 11, 30
        assert total.e2e_p50_s == 12.0
        assert total.e2e_p95_s == pytest.approx(25 + 5 * 0.8)
        assert total.active_p50_s is not None and total.active_p95_s is not None
        assert total.human_wait_total_s == 25.0

    def test_replan_reuse_is_pooled(self) -> None:
        reuse = metrics(TestReplanAndCache().stream())
        total = aggregate([reuse, metrics(clean_success())])
        assert total.replan_reuse_ratio == 0.5
        assert aggregate([metrics(clean_success())]).replan_reuse_ratio is None

    def test_policy_violations_and_concurrency_merge(self) -> None:
        s1, s2 = Stream("p1"), Stream("p2")
        for s, rules in ((s1, ["SEC-001"]), (s2, ["SEC-001", "CHG-001"])):
            s.at(0).add(ET.RUN_CREATED)
            for rule in rules:
                s.add(
                    ET.POLICY_VIOLATION,
                    "x",
                    rule_id=rule,
                    severity="HIGH",
                    enforcement="deny",
                    location="p",
                )
            s.add(ET.RUN_FAILED)
        total = aggregate([metrics(s1), metrics(s2)])
        assert total.policy_violations_by_rule == {"CHG-001": 1, "SEC-001": 2}
        assert aggregate([metrics(TestStructure().overlapping(3))]).max_concurrency_observed == 3

    def test_no_runs(self) -> None:
        total = aggregate([])
        assert total.runs == 0 and total.success_rate == 0.0 and total.rollback_frequency == 0.0
        assert (
            total.mttr_mean_s is None
            and total.e2e_p50_s is None
            and total.replan_reuse_ratio is None
        )
        assert isinstance(total, AggregateMetrics)
        json.dumps(total.to_dict())

    def test_accepts_any_iterable(self) -> None:
        assert aggregate(iter(all_runs())).runs == 5


class TestPercentile:
    @pytest.mark.parametrize(
        ("values", "q", "expected"),
        [
            ([7, 11, 12, 25, 30], 0.5, 12),
            ([1, 2, 3, 4], 0.5, 2.5),
            ([1, 2, 3, 4], 0.0, 1),
            ([1, 2, 3, 4], 1.0, 4),
            ([1, 2, 3, 4], 0.95, 3.85),
            ([5], 0.95, 5),
            ([4, 1, 3, 2], 0.25, 1.75),
        ],
    )
    def test_linear_interpolation(self, values: list[float], q: float, expected: float) -> None:
        assert percentile(values, q) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("values", "q"),
        [([], 0.5), ([1.0, float("nan")], 0.5), ([1.0, 2.0], -0.1), ([1.0, 2.0], 1.1)],
    )
    def test_unusable_inputs_give_none(self, values: list[float], q: float) -> None:
        assert percentile(values, q) is None


# ----------------------------------------------------------------- loading from disk


def write_run(runs_dir: Path, stream: Stream) -> Path:
    """Persist a synthetic stream through the real AuditLog (so the chain is genuine)."""
    clock = FakeClock()
    run_dir = runs_dir / stream.run_id
    log = AuditLog(run_dir / AUDIT_FILENAME, run_id=stream.run_id, clock=clock)
    for event in stream.events:
        clock.advance(
            max(0.0, (event.ts - T0).total_seconds() - (clock.now() - T0).total_seconds())
        )
        log.emit(event.type, node_id=event.node_id, attempt=event.attempt, data=event.data)
    return run_dir


class TestLoadAll:
    def test_loads_every_intact_run_sorted_by_name(self, tmp_path: Path) -> None:
        write_run(tmp_path, retry_then_success("run-b"))
        write_run(tmp_path, clean_success("run-a"))
        loaded = load_all(tmp_path)
        assert [m.run_id for m in loaded] == ["run-a", "run-b"]
        assert loaded[0].e2e_latency_s == 25.0 and loaded[1].retries == 1
        assert loaded[1] == metrics(retry_then_success("run-b"))

    def test_damaged_logs_are_skipped_and_reported(self, tmp_path: Path) -> None:
        write_run(tmp_path, clean_success("run-ok"))
        bad_dir = write_run(tmp_path, retry_then_success("run-bad"))
        log_file = bad_dir / AUDIT_FILENAME
        log_file.write_bytes(
            log_file.read_bytes().replace(b'"kind":"transient"', b'"kind":"fatal"')
        )
        problems: list[str] = []
        assert [m.run_id for m in load_all(tmp_path, problems=problems)] == ["run-ok"]
        assert len(problems) == 1 and problems[0].startswith("run-bad: ")
        assert [m.run_id for m in load_all(tmp_path)] == ["run-ok"]  # problems list is optional

    def test_non_run_entries_are_ignored(self, tmp_path: Path) -> None:
        write_run(tmp_path, clean_success("run-ok"))
        (tmp_path / "not a run id").mkdir()
        (tmp_path / "run-empty").mkdir()
        (tmp_path / "stray-file").write_text("x", encoding="utf-8")
        (tmp_path / "anchors.jsonl").write_text("{}\n", encoding="utf-8")
        assert [m.run_id for m in load_all(tmp_path)] == ["run-ok"]

    def test_missing_directory(self, tmp_path: Path) -> None:
        assert load_all(tmp_path / "nope") == []

    def test_a_run_with_a_new_segment_is_read_as_one_run(self, tmp_path: Path) -> None:
        run_dir = write_run(tmp_path, clean_success("run-seg"))
        log_file = run_dir / AUDIT_FILENAME
        lines = log_file.read_bytes().split(b"\n")
        lines[1] = lines[1].replace(b'"intake"', b'"hacked"')  # event 2 is damaged
        log_file.write_bytes(b"\n".join(lines))
        error = AuditLog.verify_file(log_file)
        assert not error.ok and error.broken_at == 2
        segment = AuditLog.open_new_segment(
            log_file,
            broken_at=2,
            last_good_hash=error.last_good_hash,
            run_id="run-seg",
            clock=FakeClock(),
        )
        segment.emit(ET.RUN_FAILED)
        (loaded,) = load_all(tmp_path)
        assert loaded.run_id == "run-seg" and loaded.status == "failed"


# ----------------------------------------------------------------- rendering


class TestRenderTable:
    def test_rows_and_aggregate_block(self) -> None:
        runs = all_runs()
        text = render_table(runs)
        lines = text.splitlines()
        assert lines[0].split() == [
            "run",
            "status",
            "e2e_s",
            "active_s",
            "wait_s",
            "att",
            "retry",
            "fb",
            "rb",
            "safe",
            "mttr_s",
            "conc",
        ]
        assert set(lines[1]) <= {"-", " "}
        for run in runs:
            assert any(line.startswith(run.run_id) for line in lines[2:7])
        row = next(line for line in lines if line.startswith("run-retry"))
        assert "succeeded" in row and "50.0%" in row and "7.00" in row
        assert "aggregate over 5 run(s)" in text
        assert "success rate: 50.0%" in text and "MTTR mean / p95 (s): 7.00 / 7.00" in text
        assert text.endswith("\n")

    def test_columns_are_aligned(self) -> None:
        lines = render_table(all_runs()).splitlines()[:7]
        assert len({len(line) for line in lines}) == 1

    def test_empty_input(self) -> None:
        text = render_table([])
        assert "aggregate over 0 run(s)" in text and "replan reuse: -" in text

    def test_explicit_summary_is_used(self) -> None:
        summary = aggregate(all_runs())
        text = render_table(all_runs()[:1], summary)
        assert "aggregate over 5 run(s)" in text

    def test_hostile_run_ids_cannot_inject_terminal_sequences(self) -> None:
        evil = dataclasses.replace(
            metrics(clean_success()), run_id="\x1b[31mred\x1b]0;title\x07" + "x" * 300 + "\u202e"
        )
        text = render_table([evil])
        assert "\x1b" not in text and "\x07" not in text and "\u202e" not in text
        assert max(len(line) for line in text.splitlines()[:3]) < 200
