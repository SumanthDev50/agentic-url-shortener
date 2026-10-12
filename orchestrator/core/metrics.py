"""Run metrics: pure functions of audit events (engine-semantics §17.8).

Nothing here reads engine state; every number is derived from the (verified) event stream, so the
same log always yields the same metrics and a reader can recompute them.  Definitions:

* ``attempts`` = ``node.started`` events; ``retries`` = ``node.retry_scheduled``;
  ``retry_frequency`` = retries / *countable* attempts, where discarded, cancelled and needs-input
  attempts are excluded from the denominator (they are not evidence about reliability).
* ``fallback_rate`` = fallbacks / distinct nodes that ran; ``rollbacks`` counts compensated nodes
  (``node.rolled_back``: node- and run-level compensation; attempt quarantines are reported
  separately as ``quarantines``) and ``rollback_frequency`` = rollbacks / runs.
* An **incident** opens at the first non-cancelled ``node.attempt_failed`` of a node and closes at
  ``node.succeeded`` / ``node.cache_hit`` (*recovered*), ``node.failed`` (*unrecovered*), or is
  *abandoned* by a reset, removal, cancellation or safe stop.  An incident still open when the log
  ends counts as unrecovered.  **MTTR excludes human-wait intervals** (``run.waiting`` until the
  next ``run.resumed`` or terminal event).
* An **attempt span** runs from ``node.started`` to the first event that ends the *execution*
  (failure, discard, awaiting approval/input, publishing, success, cancel); a ``run.resumed``
  closes spans a crash left open.  ``node_latency_s`` sums a node's spans; ``stage_latency_s`` is
  the wall-clock *union* of a stage's spans (parallel nodes are not double counted).
* ``max_concurrency_observed`` is the peak number of overlapping spans in **log order** (event
  index), not wall time: virtual clocks and coarse timestamps cannot hide real overlap.
* ``critical_path_s``: with a ``graph`` the exact longest dependency chain weighted by node
  latency; without one, the longest chain of spans that did not overlap in time — an upper bound
  that is exact when parallel branches really ran in parallel.
* ``replan_reuse_ratio`` = cache hits / revalidated nodes, where a revalidated node is one that went
  stale and then either hit the cache or ran again (``None`` when nothing was revalidated).
* ``first_pass_yield`` = nodes that ran and succeeded without any failed attempt / nodes that ran.
"""

from __future__ import annotations

import dataclasses
import enum
import heapq
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.core.audit import read_run_events, run_status_of
from orchestrator.core.errors import OrchestratorError
from orchestrator.core.fsutil import is_link_or_reparse
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.paths import RUN_ID_RE
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.serde import parse_datetime
from orchestrator.core.types import AuditEvent, EventType

_ET = EventType
_TERMINAL = frozenset({_ET.RUN_SUCCEEDED, _ET.RUN_FAILED, _ET.RUN_SAFE_STOPPED})
_SPAN_ENDS = frozenset(
    {
        _ET.NODE_ATTEMPT_FAILED,
        _ET.NODE_ATTEMPT_DISCARDED,
        _ET.NODE_AWAITING_APPROVAL,
        _ET.NODE_AWAITING_INPUT,
        _ET.NODE_PUBLISHING,
        _ET.NODE_SUCCEEDED,
        _ET.NODE_FAILED,
        _ET.NODE_CANCELLED,
    }
)
_NOT_EVIDENCE_ENDS = frozenset(
    {_ET.NODE_ATTEMPT_DISCARDED, _ET.NODE_AWAITING_INPUT, _ET.NODE_CANCELLED}
)
_ABANDONING = frozenset({_ET.NODE_RESET, _ET.NODE_REMOVED, _ET.NODE_CANCELLED})
_MAX_KEY_CHARS = 48


class IncidentStatus(enum.StrEnum):
    RECOVERED = "recovered"
    UNRECOVERED = "unrecovered"
    ABANDONED = "abandoned"


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _opt_dt(value: str | None) -> datetime | None:
    return parse_datetime(value) if value else None


@dataclass(frozen=True)
class Incident:
    node_id: str
    failed_at: datetime
    recovered_at: datetime | None
    mttr_s: float | None
    status: IncidentStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "failed_at": self.failed_at.isoformat(),
            "recovered_at": _iso(self.recovered_at),
            "mttr_s": self.mttr_s,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Incident:
        return cls(
            d["node_id"],
            parse_datetime(d["failed_at"]),
            _opt_dt(d.get("recovered_at")),
            d.get("mttr_s"),
            IncidentStatus(d["status"]),
        )


@dataclass(frozen=True)
class RunMetrics:
    """Metrics of one run (see the module docstring for the exact definitions)."""

    run_id: str
    status: str  # running | waiting_human | succeeded | failed | safe_stopped | unknown
    success: bool
    started_at: datetime
    ended_at: datetime
    e2e_latency_s: float
    active_latency_s: float
    human_wait_s: float
    nodes: int
    attempts: int
    countable_attempts: int
    retries: int
    retry_frequency: float
    fallbacks: int
    fallback_rate: float
    rollbacks: int
    safe_stops: int
    replans: int
    cache_hits: int
    revalidated_nodes: int
    replan_reuse_ratio: float | None
    first_pass_yield: float
    incidents: tuple[Incident, ...]
    mttr_s: float | None
    stage_latency_s: dict[str, float]
    node_latency_s: dict[str, float]
    critical_path_s: float
    max_concurrency_observed: int
    approvals_requested: int
    approvals_auto: int
    policy_violations_by_rule: dict[str, int]
    quarantines: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        out["started_at"] = self.started_at.isoformat()
        out["ended_at"] = self.ended_at.isoformat()
        out["incidents"] = [i.to_dict() for i in self.incidents]
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunMetrics:
        values = {f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d}
        values["started_at"] = parse_datetime(d["started_at"])
        values["ended_at"] = parse_datetime(d["ended_at"])
        values["incidents"] = tuple(Incident.from_dict(i) for i in d.get("incidents", ()))
        return cls(**values)


@dataclass(frozen=True)
class AggregateMetrics:
    """Cross-run statistics. Rates use the denominators documented on each field."""

    runs: int
    succeeded: int
    failed: int
    safe_stopped: int
    unfinished: int
    success_rate: float  # succeeded / finished runs (succeeded + failed + safe_stopped)
    retry_rate: float  # total retries / total countable attempts
    fallback_rate: float  # total fallbacks / total nodes that ran
    rollback_frequency: float  # total compensations / runs
    safe_stop_rate: float  # runs with a safe stop / finished runs
    first_pass_yield: float  # mean of the per-run yields
    replan_reuse_ratio: float | None  # total cache hits / total revalidated nodes
    incidents: int
    incidents_recovered: int
    incidents_unrecovered: int
    incidents_abandoned: int
    mttr_mean_s: float | None  # over every recovered incident of every run
    mttr_p95_s: float | None
    e2e_p50_s: float | None
    e2e_p95_s: float | None
    active_p50_s: float | None
    active_p95_s: float | None
    human_wait_total_s: float = 0.0
    max_concurrency_observed: int = 0
    policy_violations_by_rule: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


# =============================================================================
# One run
# =============================================================================


@dataclass
class _Span:
    node_id: str
    attempt: int | None
    stage: str
    start_index: int
    start_ts: datetime
    end_index: int = -1
    end_ts: datetime | None = None
    countable: bool = True

    @property
    def seconds(self) -> float:
        return _seconds(self.start_ts, self.end_ts or self.start_ts)


def _seconds(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds())


def _key(value: object) -> str:
    return sanitize_for_terminal(value, _MAX_KEY_CHARS) or "unknown"


def _rule_key(event: AuditEvent) -> str:
    rule = event.data.get("rule_id")
    return _key(rule) if isinstance(rule, str) else "unknown"


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _end_of_run(stream: Sequence[AuditEvent]) -> datetime:
    """Timestamp of the last terminal run event (a trailing ``run.sealed`` is bookkeeping), else of
    the last event."""
    for event in reversed(stream):
        if event.type in _TERMINAL:
            return event.ts
    return stream[-1].ts


def _wait_intervals(stream: Sequence[AuditEvent], end: datetime) -> list[tuple[datetime, datetime]]:
    """Human-wait intervals: ``run.waiting`` until ``run.resumed`` / a terminal event / log end."""
    intervals: list[tuple[datetime, datetime]] = []
    opened: datetime | None = None
    for event in stream:
        if event.type is _ET.RUN_WAITING and opened is None:
            opened = event.ts
        elif opened is not None and (event.type is _ET.RUN_RESUMED or event.type in _TERMINAL):
            intervals.append((opened, event.ts))
            opened = None
    if opened is not None:
        intervals.append((opened, max(opened, end)))
    return intervals


def _overlap(
    start: datetime, end: datetime, intervals: Sequence[tuple[datetime, datetime]]
) -> float:
    return sum(
        _seconds(max(a, start), min(b, end)) for a, b in intervals if min(b, end) > max(a, start)
    )


def _build_spans(stream: Sequence[AuditEvent]) -> list[_Span]:
    spans: list[_Span] = []
    live: dict[tuple[str, int | None], _Span] = {}

    def close(span: _Span, index: int, ts: datetime, *, countable: bool = True) -> None:
        span.end_index, span.end_ts = index, max(ts, span.start_ts)
        span.countable = span.countable and countable
        live.pop((span.node_id, span.attempt), None)

    for index, event in enumerate(stream):
        node = event.node_id
        if event.type is _ET.RUN_RESUMED:  # a restarted process cannot own the old attempts
            for span in list(live.values()):
                close(span, index, event.ts)
        elif node is None:
            continue
        elif event.type is _ET.NODE_STARTED:
            stage = event.data.get("stage")
            span = _Span(
                node,
                event.attempt,
                _key(stage) if isinstance(stage, str) else "unknown",
                index,
                event.ts,
            )
            spans.append(span)
            live[(node, event.attempt)] = span
        elif event.type in _SPAN_ENDS:
            cancelled = (
                event.type is _ET.NODE_ATTEMPT_FAILED and event.data.get("kind") == "cancelled"
            )
            countable = event.type not in _NOT_EVIDENCE_ENDS and not cancelled
            targets = [
                s for (n, a), s in live.items() if n == node and (event.attempt in (None, a))
            ]
            for span in targets:
                close(span, index, event.ts, countable=countable)
    for span in list(live.values()):
        close(span, len(stream), stream[-1].ts)
    return spans


def _union_seconds(intervals: Iterable[tuple[datetime, datetime]]) -> float:
    total = 0.0
    current: tuple[datetime, datetime] | None = None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                total += _seconds(*current)
            current = (start, end)
        else:
            current = (current[0], max(current[1], end))
    return total + (_seconds(*current) if current else 0.0)


def _max_concurrency(spans: Sequence[_Span]) -> int:
    points = sorted([(s.start_index, 1) for s in spans] + [(s.end_index, -1) for s in spans])
    running = peak = 0
    for _, delta in points:
        running += delta
        peak = max(peak, running)
    return peak


def _longest_chain_s(spans: Sequence[_Span]) -> float:
    """Longest total duration of spans that follow one another without overlapping."""
    done_best = 0.0
    answer = 0.0
    pending: list[tuple[datetime, float]] = []
    for span in sorted(spans, key=lambda s: (s.start_ts, s.end_ts or s.start_ts)):
        while pending and pending[0][0] <= span.start_ts:
            done_best = max(done_best, heapq.heappop(pending)[1])
        value = done_best + span.seconds
        answer = max(answer, value)
        heapq.heappush(pending, (span.end_ts or span.start_ts, value))
    return answer


def _incidents(
    stream: Sequence[AuditEvent], waits: Sequence[tuple[datetime, datetime]]
) -> list[Incident]:
    out: list[Incident] = []
    open_at: dict[str, datetime] = {}

    def finish(node: str, status: IncidentStatus, ts: datetime | None) -> None:
        failed_at = open_at.pop(node)
        mttr = None
        if status is IncidentStatus.RECOVERED and ts is not None:
            mttr = max(0.0, _seconds(failed_at, ts) - _overlap(failed_at, ts, waits))
        out.append(
            Incident(
                node, failed_at, ts if status is IncidentStatus.RECOVERED else None, mttr, status
            )
        )

    for event in stream:
        node = event.node_id
        if event.type is _ET.RUN_SAFE_STOPPED:
            for pending in list(open_at):
                finish(pending, IncidentStatus.ABANDONED, None)
        elif node is None:
            continue
        elif event.type is _ET.NODE_ATTEMPT_FAILED and event.data.get("kind") != "cancelled":
            open_at.setdefault(node, event.ts)
        elif node in open_at:
            if event.type in (_ET.NODE_SUCCEEDED, _ET.NODE_CACHE_HIT):
                finish(node, IncidentStatus.RECOVERED, event.ts)
            elif event.type is _ET.NODE_FAILED:
                finish(node, IncidentStatus.UNRECOVERED, None)
            elif event.type in _ABANDONING:
                finish(node, IncidentStatus.ABANDONED, None)
    for pending in list(open_at):
        finish(pending, IncidentStatus.UNRECOVERED, None)
    return out


def _revalidated(stream: Sequence[AuditEvent]) -> int:
    """Nodes that went stale and then hit the cache or ran again."""
    stale: set[str] = set()
    revalidated = 0
    for event in stream:
        node = event.node_id
        if node is None:
            continue
        if event.type is _ET.NODE_STALE:
            stale.add(node)
        elif event.type in (_ET.NODE_REMOVED, _ET.NODE_CANCELLED):
            stale.discard(node)
        elif event.type is _ET.NODE_CACHE_HIT or (event.type is _ET.NODE_STARTED and node in stale):
            revalidated += 1
            stale.discard(node)
    return revalidated


def _first_pass_yield(stream: Sequence[AuditEvent], started: set[str]) -> float:
    succeeded = {e.node_id for e in stream if e.type is _ET.NODE_SUCCEEDED}
    troubled = {
        e.node_id
        for e in stream
        if e.type is _ET.NODE_FAILED
        or (e.type is _ET.NODE_ATTEMPT_FAILED and e.data.get("kind") != "cancelled")
    }
    return _ratio(len(started & (succeeded - troubled)), len(started))


def compute_run_metrics(
    events: Sequence[AuditEvent], *, graph: WorkflowGraph | None = None
) -> RunMetrics:
    """Metrics of one run from its events in log order.

    ``graph`` (optional) makes ``critical_path_s`` the exact dependency critical path; it never
    changes any other number.  Raises ``ValueError`` for an empty stream.
    """
    stream = list(events)
    if not stream:
        raise ValueError("cannot compute metrics of an empty event stream")
    counts = Counter(e.type for e in stream)
    started_at, ended_at = stream[0].ts, _end_of_run(stream)
    waits = _wait_intervals(stream, stream[-1].ts)
    e2e = _seconds(started_at, ended_at)
    human_wait = sum(_overlap(started_at, ended_at, [w]) for w in waits)
    spans = _build_spans(stream)
    started = {s.node_id for s in spans}
    node_latency: dict[str, float] = {}
    for span in spans:
        node_latency[span.node_id] = node_latency.get(span.node_id, 0.0) + span.seconds
    by_stage: dict[str, list[tuple[datetime, datetime]]] = {}
    for span in spans:
        by_stage.setdefault(span.stage, []).append((span.start_ts, span.end_ts or span.start_ts))
    incidents = _incidents(stream, waits)
    recovered = [i.mttr_s for i in incidents if i.mttr_s is not None]
    cache_hits = counts[_ET.NODE_CACHE_HIT]
    revalidated = _revalidated(stream)
    retries = counts[_ET.NODE_RETRY_SCHEDULED]
    countable = sum(1 for s in spans if s.countable)
    violations = Counter(_rule_key(e) for e in stream if e.type is _ET.POLICY_VIOLATION)
    critical = (
        graph.critical_path(node_latency).length_s if graph is not None else _longest_chain_s(spans)
    )
    status = run_status_of(stream)
    return RunMetrics(
        run_id=stream[0].run_id,
        status=status,
        success=status == "succeeded",
        started_at=started_at,
        ended_at=ended_at,
        e2e_latency_s=e2e,
        active_latency_s=max(0.0, e2e - human_wait),
        human_wait_s=human_wait,
        nodes=len(started),
        attempts=len(spans),
        countable_attempts=countable,
        retries=retries,
        retry_frequency=_ratio(retries, countable),
        fallbacks=counts[_ET.NODE_FALLBACK_ACTIVATED],
        fallback_rate=_ratio(counts[_ET.NODE_FALLBACK_ACTIVATED], len(started)),
        rollbacks=counts[_ET.NODE_ROLLED_BACK],
        safe_stops=counts[_ET.SAFE_STOP_TRIGGERED],
        replans=counts[_ET.PLAN_REPLANNED],
        cache_hits=cache_hits,
        revalidated_nodes=revalidated,
        replan_reuse_ratio=cache_hits / revalidated if revalidated else None,
        first_pass_yield=_first_pass_yield(stream, started),
        incidents=tuple(incidents),
        mttr_s=sum(recovered) / len(recovered) if recovered else None,
        stage_latency_s={k: _union_seconds(v) for k, v in sorted(by_stage.items())},
        node_latency_s=dict(sorted(node_latency.items())),
        critical_path_s=critical,
        max_concurrency_observed=_max_concurrency(spans),
        approvals_requested=counts[_ET.APPROVAL_REQUESTED],
        approvals_auto=counts[_ET.APPROVAL_AUTO],
        policy_violations_by_rule=dict(sorted(violations.items())),
        quarantines=counts[_ET.ARTIFACT_QUARANTINED],
    )


# =============================================================================
# Many runs
# =============================================================================


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (``q`` in [0, 1]); ``None`` for no values."""
    if not values or not 0.0 <= q <= 1.0 or any(not math.isfinite(v) for v in values):
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def aggregate(runs: Iterable[RunMetrics]) -> AggregateMetrics:
    """Combine per-run metrics (rates are pooled over the underlying counts, not averaged)."""
    rows = list(runs)
    status = Counter(r.status for r in rows)
    finished = status["succeeded"] + status["failed"] + status["safe_stopped"]
    incidents = [i for r in rows for i in r.incidents]
    mttrs = [i.mttr_s for i in incidents if i.mttr_s is not None]
    revalidated = sum(r.revalidated_nodes for r in rows)
    violations: Counter[str] = Counter()
    for r in rows:
        violations.update(r.policy_violations_by_rule)
    kinds = Counter(i.status for i in incidents)
    return AggregateMetrics(
        runs=len(rows),
        succeeded=status["succeeded"],
        failed=status["failed"],
        safe_stopped=status["safe_stopped"],
        unfinished=len(rows) - finished,
        success_rate=_ratio(status["succeeded"], finished),
        retry_rate=_ratio(sum(r.retries for r in rows), sum(r.countable_attempts for r in rows)),
        fallback_rate=_ratio(sum(r.fallbacks for r in rows), sum(r.nodes for r in rows)),
        rollback_frequency=_ratio(sum(r.rollbacks for r in rows), len(rows)),
        safe_stop_rate=_ratio(sum(1 for r in rows if r.safe_stops), finished),
        first_pass_yield=_ratio(sum(r.first_pass_yield for r in rows), len(rows)),
        replan_reuse_ratio=sum(r.cache_hits for r in rows) / revalidated if revalidated else None,
        incidents=len(incidents),
        incidents_recovered=kinds[IncidentStatus.RECOVERED],
        incidents_unrecovered=kinds[IncidentStatus.UNRECOVERED],
        incidents_abandoned=kinds[IncidentStatus.ABANDONED],
        mttr_mean_s=sum(mttrs) / len(mttrs) if mttrs else None,
        mttr_p95_s=percentile(mttrs, 0.95),
        e2e_p50_s=percentile([r.e2e_latency_s for r in rows], 0.5),
        e2e_p95_s=percentile([r.e2e_latency_s for r in rows], 0.95),
        active_p50_s=percentile([r.active_latency_s for r in rows], 0.5),
        active_p95_s=percentile([r.active_latency_s for r in rows], 0.95),
        human_wait_total_s=sum(r.human_wait_s for r in rows),
        max_concurrency_observed=max((r.max_concurrency_observed for r in rows), default=0),
        policy_violations_by_rule=dict(sorted(violations.items())),
    )


def load_all(runs_dir: Path | str, *, problems: list[str] | None = None) -> list[RunMetrics]:
    """Metrics of every run under ``runs_dir`` (one sub-directory per run, sorted by name).

    Only plain directories with a valid run-id name are read (links are never followed).  A run
    whose audit log is damaged is skipped — metrics of a tampered log would be meaningless — and
    described in ``problems`` when a list is given; runs without audit events are ignored.
    """
    root = Path(runs_dir)
    out: list[RunMetrics] = []
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not RUN_ID_RE.fullmatch(entry.name) or is_link_or_reparse(entry) or not entry.is_dir():
            continue
        try:
            events = read_run_events(entry)
            if events:
                out.append(compute_run_metrics(events))
        except (OrchestratorError, OSError, ValueError) as exc:
            if problems is not None:
                problems.append(f"{entry.name}: {sanitize_for_terminal(exc, 160)}")
    return out


# =============================================================================
# Rendering
# =============================================================================


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def render_table(runs: Sequence[RunMetrics], summary: AggregateMetrics | None = None) -> str:
    """Plain-text table of ``runs`` plus an aggregate block (computed when ``summary`` is None).

    Every cell derived from log content passes ``sanitize_for_terminal``: a hostile run id cannot
    inject escape sequences into the operator's terminal.
    """
    header = (
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
    )
    rows = [
        (
            sanitize_for_terminal(r.run_id, 24),
            r.status,
            _fmt(r.e2e_latency_s),
            _fmt(r.active_latency_s),
            _fmt(r.human_wait_s),
            str(r.attempts),
            _pct(r.retry_frequency),
            str(r.fallbacks),
            str(r.rollbacks),
            str(r.safe_stops),
            _fmt(r.mttr_s),
            str(r.max_concurrency_observed),
        )
        for r in runs
    ]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines.extend("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows)
    total = summary if summary is not None else aggregate(runs)
    lines.extend(["", f"aggregate over {total.runs} run(s)"])
    lines.extend(f"  {name}: {value}" for name, value in _summary_lines(total))
    return "\n".join(lines) + "\n"


def _summary_lines(total: AggregateMetrics) -> list[tuple[str, str]]:
    return [
        ("success rate", _pct(total.success_rate)),
        ("retry rate", _pct(total.retry_rate)),
        ("fallback rate", _pct(total.fallback_rate)),
        ("rollback frequency", _fmt(total.rollback_frequency)),
        ("safe-stop rate", _pct(total.safe_stop_rate)),
        ("first-pass yield", _pct(total.first_pass_yield)),
        ("replan reuse", _pct(total.replan_reuse_ratio)),
        (
            "incidents (rec/unrec/aband)",
            f"{total.incidents_recovered}/{total.incidents_unrecovered}/{total.incidents_abandoned}",
        ),
        ("MTTR mean / p95 (s)", f"{_fmt(total.mttr_mean_s)} / {_fmt(total.mttr_p95_s)}"),
        ("e2e p50 / p95 (s)", f"{_fmt(total.e2e_p50_s)} / {_fmt(total.e2e_p95_s)}"),
        ("active p50 / p95 (s)", f"{_fmt(total.active_p50_s)} / {_fmt(total.active_p95_s)}"),
        ("human wait total (s)", _fmt(total.human_wait_total_s)),
    ]
