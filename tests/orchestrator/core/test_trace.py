"""export_trace (run -> node -> attempt span tree), run_status_of and AuditLog.normalize_events."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.audit import AUDIT_FILENAME, AuditLog, export_trace, run_status_of
from orchestrator.core.clock import FakeClock
from orchestrator.core.types import Actor, AuditEvent, EventType

ET = EventType
T0 = datetime(2026, 1, 1, tzinfo=UTC)


class Stream:
    """Hand-built event stream (time moves only through ``at``)."""

    def __init__(self, run_id: str = "run-1", trace_id: str = "trace-1") -> None:
        self.run_id, self.trace_id = run_id, trace_id
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
        actor: Actor | None = None,
        **data: Any,
    ) -> Stream:
        seq = len(self.events) + 1
        self.events.append(
            AuditEvent(
                seq=seq,
                ts=T0 + timedelta(seconds=self._now),
                run_id=self.run_id,
                trace_id=self.trace_id,
                span_id="run" if node is None else f"n:{node}",
                parent_span_id=None,
                type=event_type,
                node_id=node,
                attempt=attempt,
                actor=actor or Actor.system(),
                data=data,
                prev_hash="0" * 64,
                hash=f"{seq:064x}",
            )
        )
        return self


def find(span: dict[str, Any], span_id: str) -> dict[str, Any]:
    if span["span_id"] == span_id:
        return span
    for child in span.get("children", ()):
        try:
            return find(child, span_id)
        except KeyError:
            continue
    raise KeyError(span_id)


def sample_run() -> Stream:
    s = Stream()
    s.at(0).add(ET.RUN_CREATED, scenario="demo")
    s.at(1).add(ET.NODE_READY, "design").add(ET.NODE_STARTED, "design", 1, stage="design")
    s.at(2).add(ET.TOOL_CALL, "design", 1, tool="read")
    s.at(3).add(ET.NODE_ATTEMPT_FAILED, "design", 1, kind="gate_rejected", message="m")
    s.add(ET.NODE_RETRY_SCHEDULED, "design", 1)
    s.at(5).add(ET.NODE_STARTED, "design", 2, stage="design")
    s.at(7).add(ET.NODE_AWAITING_APPROVAL, "design", 2)
    s.at(8).add(ET.RUN_WAITING)
    s.at(20).add(ET.RUN_RESUMED)
    s.at(21).add(ET.NODE_PUBLISHING, "design", 2).add(ET.NODE_SUCCEEDED, "design", 2)
    s.at(22).add(ET.NODE_SKIPPED, "docs")
    s.at(30).add(ET.RUN_SUCCEEDED)
    return s


class TestExportTrace:
    def test_tree_shape_and_ordering(self) -> None:
        trace = export_trace(sample_run().events)
        assert (trace["trace_id"], trace["run_id"], trace["span_id"], trace["name"]) == (
            "trace-1",
            "run-1",
            "run",
            "run-1",
        )
        assert [c["span_id"] for c in trace["children"]] == ["n:design", "n:docs"]
        design = trace["children"][0]
        assert [c["span_id"] for c in design["children"]] == ["n:design#1", "n:design#2"]
        assert "children" not in trace["children"][1] or trace["children"][1]["children"] == []

    def test_run_span(self) -> None:
        trace = export_trace(sample_run().events)
        assert trace["start"] == T0.isoformat()
        assert trace["end"] == (T0 + timedelta(seconds=30)).isoformat()
        assert trace["duration_s"] == 30.0 and trace["status"] == "succeeded"
        assert trace["events"] == len(sample_run().events)

    def test_node_spans_cover_their_lifetime_and_carry_the_final_status(self) -> None:
        trace = export_trace(sample_run().events)
        design = find(trace, "n:design")
        assert design["status"] == "succeeded"
        assert (design["start"], design["end"]) == (
            (T0 + timedelta(seconds=1)).isoformat(),
            (T0 + timedelta(seconds=21)).isoformat(),
        )
        assert design["duration_s"] == 20.0
        docs = find(trace, "n:docs")
        assert docs["status"] == "skipped" and docs["duration_s"] == 0.0

    def test_attempt_spans(self) -> None:
        trace = export_trace(sample_run().events)
        first, second = find(trace, "n:design#1"), find(trace, "n:design#2")
        assert first["status"] == "failed" and first["failure_kind"] == "gate_rejected"
        assert first["duration_s"] == 2.0 and first["name"] == "design#1"
        assert second["status"] == "succeeded" and "failure_kind" not in second
        assert second["duration_s"] == 16.0  # the lifetime includes the wait for approval
        assert second["events"] == 4  # started, awaiting approval, publishing, succeeded

    @pytest.mark.parametrize(
        ("events", "expected"),
        [
            ([(ET.NODE_STARTED, 1)], "running"),
            ([(ET.NODE_STARTED, 1), (ET.NODE_ATTEMPT_DISCARDED, 1)], "discarded"),
            ([(ET.NODE_STARTED, 1), (ET.NODE_AWAITING_INPUT, 1)], "awaiting_input"),
            (
                [(ET.NODE_STARTED, 1), (ET.NODE_AWAITING_APPROVAL, 1), (ET.NODE_SUCCEEDED, 1)],
                "succeeded",
            ),
            ([(ET.NODE_STARTED, 1), (ET.NODE_FAILED, 1)], "failed"),
            ([(ET.NODE_STARTED, 1), (ET.NODE_CANCELLED, 1)], "cancelled"),
            ([(ET.TOOL_CALL, 1)], "unknown"),
        ],
    )
    def test_attempt_status_is_the_last_status_bearing_event(
        self, events: list[tuple[EventType, int]], expected: str
    ) -> None:
        s = Stream().at(0).add(ET.RUN_CREATED)
        for event_type, attempt in events:
            s.add(event_type, "n", attempt)
        assert find(export_trace(s.events), "n:n#1")["status"] == expected

    @pytest.mark.parametrize(
        ("event_type", "expected"),
        [
            (ET.NODE_READY, "ready"),
            (ET.NODE_BLOCKED, "blocked"),
            (ET.NODE_STALE, "stale"),
            (ET.NODE_RESET, "pending"),
            (ET.NODE_REMOVED, "removed"),
            (ET.NODE_CANCELLED, "cancelled"),
            (ET.NODE_ROLLED_BACK, "rolled_back"),
            (ET.NODE_AWAITING_INPUT, "awaiting_input"),
            (ET.NODE_RETRY_SCHEDULED, "ready"),
            (ET.NODE_FAILED, "failed"),
            (ET.AGENT_NOTE, "unknown"),
        ],
    )
    def test_node_status_mapping(self, event_type: EventType, expected: str) -> None:
        s = Stream().at(0).add(ET.RUN_CREATED).add(event_type, "n")
        assert find(export_trace(s.events), "n:n")["status"] == expected

    def test_spans_use_engine_derived_ids_and_the_payload_is_never_copied(self) -> None:
        leak = "gh" + "p_" + "Lk3j" * 9
        s = Stream().at(0).add(ET.RUN_CREATED, token=leak, note="private")
        s.add(ET.NODE_STARTED, "n", 1, stage="x", prompt=leak)
        s.at(1).add(ET.NODE_ATTEMPT_FAILED, "n", 1, kind="transient", message=leak)
        dumped = json.dumps(export_trace(s.events))
        assert leak not in dumped and "private" not in dumped

    def test_hostile_failure_kinds_are_sanitised_or_ignored(self) -> None:
        s = Stream().at(0).add(ET.RUN_CREATED).add(ET.NODE_STARTED, "a", 1)
        s.add(ET.NODE_ATTEMPT_FAILED, "a", 1, kind="\x1b[31m" + "k" * 300)
        s.add(ET.NODE_STARTED, "b", 1).add(ET.NODE_ATTEMPT_FAILED, "b", 1, kind={"x": 1})
        trace = export_trace(s.events)
        kind = find(trace, "n:a#1")["failure_kind"]
        assert "\x1b" not in kind and len(kind) < 120
        assert "failure_kind" not in find(trace, "n:b#1")

    def test_output_is_plain_json(self) -> None:
        document = export_trace(sample_run().events)
        assert json.loads(json.dumps(document)) == document

    def test_empty_stream(self) -> None:
        trace = export_trace([])
        assert trace["status"] == "empty" and trace["children"] == [] and trace["duration_s"] == 0.0
        json.dumps(trace)

    def test_clock_skew_never_yields_negative_durations(self) -> None:
        s = Stream().at(10).add(ET.RUN_CREATED).add(ET.NODE_STARTED, "a", 1)
        s.at(5).add(ET.NODE_ATTEMPT_FAILED, "a", 1, kind="transient")
        trace = export_trace(s.events)
        assert find(trace, "n:a#1")["duration_s"] == 0.0 and trace["duration_s"] == 0.0

    def test_attempt_children_exist_only_for_attempts_that_happened(self) -> None:
        s = Stream().at(0).add(ET.RUN_CREATED).add(ET.NODE_READY, "a").add(ET.NODE_STARTED, "a", 7)
        node = find(export_trace(s.events), "n:a")
        assert [c["span_id"] for c in node["children"]] == ["n:a#7"]

    def test_span_ids_match_the_ids_stored_in_a_real_log(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / AUDIT_FILENAME, run_id="run-1", clock=FakeClock())
        log.emit(ET.RUN_CREATED)
        log.emit(ET.NODE_STARTED, node_id="design", attempt=1)
        log.emit(ET.TOOL_CALL, node_id="design", attempt=1)
        log.emit(ET.NODE_SUCCEEDED, node_id="design", attempt=1)
        log.emit(ET.RUN_SUCCEEDED)
        events = log.events()
        trace = export_trace(events)
        stored = {e.span_id for e in events}
        assert {"run", "n:design#1"} == stored  # node spans are derived, attempt spans are stored
        assert find(trace, "n:design")["span_id"] == "n:design"
        assert (
            find(trace, "n:design#1")["status"] == "succeeded"
            and trace["trace_id"] == events[0].trace_id
        )


class TestRunStatusOf:
    @pytest.mark.parametrize(
        ("types", "expected"),
        [
            ([], "unknown"),
            ([ET.NODE_STARTED], "unknown"),
            ([ET.RUN_CREATED], "running"),
            ([ET.RUN_CREATED, ET.RUN_STARTED, ET.RUN_WAITING], "waiting_human"),
            ([ET.RUN_WAITING, ET.RUN_RESUMED], "running"),
            ([ET.RUN_CREATED, ET.RUN_SUCCEEDED, ET.RUN_SEALED], "succeeded"),
            ([ET.RUN_CREATED, ET.RUN_FAILED], "failed"),
            ([ET.RUN_CREATED, ET.RUN_SAFE_STOPPED], "safe_stopped"),
            ([ET.RUN_SAFE_STOPPED, ET.RUN_RESUMED, ET.RUN_SUCCEEDED], "succeeded"),
        ],
    )
    def test_status(self, types: list[EventType], expected: str) -> None:
        s = Stream().at(0)
        for event_type in types:
            s.add(event_type)
        assert run_status_of(s.events) == expected


# --------------------------------------------------------------------------- normalize_events


def normalized(stream: Stream) -> list[dict[str, Any]]:
    return AuditLog.normalize_events(stream.events)


class TestNormalizeEvents:
    def test_drops_everything_derived_from_time_or_the_chain(self) -> None:
        (record,) = normalized(Stream().at(5).add(ET.NODE_STARTED, "a", 1, stage="x"))
        assert set(record) == {"run_id", "type", "node_id", "attempt", "actor", "data"}
        assert record == {
            "run_id": "run-1",
            "type": "node.started",
            "node_id": "a",
            "attempt": 1,
            "actor": {"type": "system", "id": "engine"},
            "data": {"stage": "x"},
        }

    def test_golden_streams_from_different_runs_normalise_identically(self, tmp_path: Path) -> None:
        def run(directory: str, start: datetime, trace_id: str) -> list[dict[str, Any]]:
            clock = FakeClock(start)
            log = AuditLog(
                tmp_path / directory / AUDIT_FILENAME,
                run_id="run-1",
                trace_id=trace_id,
                clock=clock,
            )
            log.emit(ET.RUN_CREATED, data={"scenario": "demo"})
            for step, node in enumerate(("design", "build")):
                clock.advance(1.5 + step)
                log.emit(ET.NODE_STARTED, node_id=node, attempt=1, data={"stage": node})
                clock.advance(3)
                log.emit(ET.NODE_SUCCEEDED, node_id=node, attempt=1, data={"outputs": []})
            return AuditLog.normalize_events(log.events())

        first = run("a", datetime(2026, 1, 1, tzinfo=UTC), "trace-aaa")
        second = run("b", datetime(2031, 7, 9, 12, 30, tzinfo=UTC), "trace-bbb")
        assert first == second and len(first) == 5

    def test_tool_calls_move_behind_their_attempt_start_and_are_sorted(self) -> None:
        def build(order: list[str]) -> Stream:
            s = Stream().at(0).add(ET.RUN_CREATED)
            plan = {
                "Sa": lambda: s.add(ET.NODE_STARTED, "a", 1),
                "Sb": lambda: s.add(ET.NODE_STARTED, "b", 1),
                "Ta1": lambda: s.add(ET.TOOL_CALL, "a", 1, tool="write", path="z.py"),
                "Ta2": lambda: s.add(ET.TOOL_CALL, "a", 1, tool="read", path="a.py"),
                "Tb1": lambda: s.add(ET.TOOL_CALL, "b", 1, tool="read", path="b.py"),
                "Da": lambda: s.add(ET.TOOL_DENIED, "a", 1, tool="write", path="../x"),
                "Ea": lambda: s.add(ET.NODE_SUCCEEDED, "a", 1),
                "Eb": lambda: s.add(ET.NODE_SUCCEEDED, "b", 1),
            }
            for step in order:
                plan[step]()
            return s

        # non-tool events keep their relative order; tool events interleave differently in each run
        one = build(["Sa", "Ta1", "Sb", "Tb1", "Ta2", "Da", "Ea", "Eb"])
        two = build(["Sa", "Sb", "Tb1", "Da", "Ta2", "Ta1", "Ea", "Eb"])
        three = build(["Sa", "Sb", "Da", "Ta1", "Tb1", "Ta2", "Ea", "Eb"])
        assert normalized(one) == normalized(two) == normalized(three)
        order = [(r["type"], r["node_id"], r["data"].get("path")) for r in normalized(one)]
        assert order[:2] == [("run.created", None, None), ("node.started", "a", None)]
        assert order[2:5] == [
            ("tool.call", "a", "a.py"),
            ("tool.call", "a", "z.py"),
            ("tool.denied", "a", "../x"),
        ]
        assert order[5:7] == [("node.started", "b", None), ("tool.call", "b", "b.py")]
        assert order[7:] == [("node.succeeded", "a", None), ("node.succeeded", "b", None)]

    def test_tool_calls_of_an_attempt_that_never_started_stay_where_they_are(self) -> None:
        s = Stream().at(0).add(ET.RUN_CREATED)
        s.add(ET.TOOL_CALL, "ghost", 1, tool="read").add(ET.NODE_STARTED, "a", 1).add(
            ET.TOOL_CALL, "ghost", 1, tool="write"
        )
        assert [(r["type"], r["node_id"]) for r in normalized(s)] == [
            ("run.created", None),
            ("tool.call", "ghost"),
            ("node.started", "a"),
            ("tool.call", "ghost"),
        ]

    def test_other_attempts_of_the_same_node_are_not_mixed_up(self) -> None:
        s = Stream().at(0).add(ET.RUN_CREATED)
        s.add(ET.NODE_STARTED, "a", 1).add(ET.NODE_STARTED, "a", 2)
        s.add(ET.TOOL_CALL, "a", 2, tool="second").add(ET.TOOL_CALL, "a", 1, tool="first")
        got = [(r["type"], r["attempt"], r["data"].get("tool")) for r in normalized(s)]
        assert got == [
            ("run.created", None, None),
            ("node.started", 1, None),
            ("tool.call", 1, "first"),
            ("node.started", 2, None),
            ("tool.call", 2, "second"),
        ]

    def test_input_is_not_modified_and_output_is_plain_json(self) -> None:
        stream = sample_run()
        before = list(stream.events)
        result = normalized(stream)
        assert stream.events == before
        assert json.loads(json.dumps(result)) == result

    def test_accepts_any_iterable_and_empty_input(self) -> None:
        assert AuditLog.normalize_events(iter([])) == []
        assert len(AuditLog.normalize_events(iter(sample_run().events))) == len(sample_run().events)
