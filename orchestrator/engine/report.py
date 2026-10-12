"""Run report and sealed run outputs (engine-semantics sections 17.2 and 18).

``render_report`` turns the facts of a run into ``report.md`` and its JSON twin ``report.json``;
``write_run_outputs`` writes ``report.md``, ``report.json``, ``metrics.json``, ``trace.json`` and
``seal.json`` into the run directory (each file atomically) and returns ``{file name: sha256}``.

Trust model
-----------
Everything that comes from agents, humans or artifacts (failure messages, node reasons, decision
summaries, approval comments, artifact kinds ...) is hostile until proven otherwise.  The facts are
collected first into one JSON-able structure (``report.json``): every free-text value passes
``sanitize_for_terminal`` there (no control / bidi characters, one line, length capped).  The
Markdown is rendered from those facts only: free text additionally goes through
``escape_markdown`` / ``table_cell`` (no raw HTML, no active links, no table or heading injection)
and bare ``scheme://``, ``www.`` and e-mail autolinks are defused, because Markdown escaping alone
does not stop GitHub-style autolinking.  Identifiers are shown as code spans restricted to a small
alphabet, so nothing can break out of them.

Determinism
-----------
Identical inputs give byte-identical outputs: no clock reads, no random values, no absolute paths,
and every collection is sorted or in input order.  Long lists are clipped (head and tail are kept,
the omitted count is stated) so a hostile log cannot inflate the report.

Seal
----
``seal.json`` records the audit head (``head_seq``, ``head_hash``, ``count``) and the SHA-256 of
the files that make up the run's record: ``state.json``, ``artifacts/index.json`` and
``decisions.jsonl`` (``null`` when a file does not exist) plus the four report files written just
before it.  The hashes are computed from the files on disk, so call ``write_run_outputs`` after the
final ``state.json`` save; any later save of ``state.json`` makes ``state_sha256`` stale, which is
the point of a seal.  ``seal.json`` is written last: its absence marks an interrupted finalizer.
A link or reparse point where one of those files should be is refused (``IntegrityError``).

Limits (documented, not hidden): the seal and the audit chain detect accidents and casual edits;
a privileged same-user attacker can rewrite the log, the files and the seal together (the fix would
be a remote anchor, see engine-semantics section 17.2).  Approver identities are attested only.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator.core.audit import GENESIS_HASH, export_trace
from orchestrator.core.errors import IntegrityError
from orchestrator.core.fsutil import (
    atomic_write_bytes,
    atomic_write_json,
    is_link_or_reparse,
    read_bytes_retry,
    sha256_bytes,
)
from orchestrator.core.metrics import RunMetrics
from orchestrator.core.sanitize import escape_markdown, sanitize_for_terminal, table_cell
from orchestrator.core.state import RunState
from orchestrator.core.types import (
    ApprovalRequest,
    Artifact,
    AuditEvent,
    Decision,
    EventType,
    NodeStatus,
    RunStatus,
)

__all__ = [
    "ASSURANCE_SCRIPTED",
    "REPORT_FILES",
    "REPORT_SCHEMA",
    "md_cell",
    "md_ident",
    "md_text",
    "render_report",
    "write_run_outputs",
]

REPORT_SCHEMA = 1
REPORT_FILES = ("report.md", "report.json", "metrics.json", "trace.json", "seal.json")
ASSURANCE_SCRIPTED = "DEMO_SCRIPTED"
SCRIPTED_CHANNEL = "scripted"

_ET = EventType
_TEXT = 200  # cap of one free-text fact
_CELL = 160  # cap of one table cell
_PART = 400  # cap of one untrusted part of a limitation sentence
_HASH_PREFIX = 12
_MAX_ROWS = 300  # per list; the first and the last half are kept when a list is longer
_FIRST_SEEN_LAST = 1 << 62
_ID_CHARS = re.compile(r"[^A-Za-z0-9._@:+*-]")
_WWW = re.compile(r"(?i)www\.")
_HEX = frozenset("0123456789abcdef")

# Key events of the timeline and the ``AuditEvent.data`` keys shown for each.  A trailing ``#``
# shows the length of a list instead of its content.
_TIMELINE: Mapping[EventType, tuple[str, ...]] = {
    _ET.RUN_WAITING: ("reason",),
    _ET.RUN_RESUMED: (),
    _ET.RUN_SUCCEEDED: (),
    _ET.RUN_FAILED: ("reason",),
    _ET.RUN_SAFE_STOPPED: ("trigger", "reason"),
    _ET.NODE_ATTEMPT_FAILED: ("kind", "message", "rule_ids"),
    _ET.NODE_RETRY_SCHEDULED: ("delay_s",),
    _ET.NODE_FALLBACK_ACTIVATED: ("from_agent", "to_agent"),
    _ET.NODE_CACHE_HIT: (),
    _ET.NODE_FAILED: ("kind", "message"),
    _ET.NODE_BLOCKED: ("reason",),
    _ET.NODE_CANCELLED: ("reason",),
    _ET.NODE_ROLLED_BACK: ("reason",),
    _ET.ROLLBACK_STARTED: ("level",),
    _ET.ROLLBACK_COMPLETED: ("level",),
    _ET.ROLLBACK_FAILED: ("level", "reason"),
    _ET.POLICY_VIOLATION: ("rule_id", "severity", "enforcement", "location"),
    _ET.POLICY_WAIVED: ("rule_id", "waiver_id"),
    _ET.APPROVAL_REQUESTED: ("approval_id", "purpose", "risk", "required"),
    _ET.APPROVAL_VOTE: ("approval_id", "approver", "approved", "channel"),
    _ET.APPROVAL_GRANTED: ("approval_id",),
    _ET.APPROVAL_REJECTED: ("approval_id",),
    _ET.APPROVAL_AUTO: ("risk", "reason"),
    _ET.APPROVAL_INVALIDATED: ("approval_id", "reason"),
    _ET.APPROVAL_CANCELLED: ("approval_id", "reason"),
    _ET.APPROVAL_REFUSED: ("approval_id", "reason"),
    _ET.CLARIFICATION_REQUESTED: ("round", "questions#"),
    _ET.CLARIFICATION_ANSWERED: ("round", "by"),
    _ET.INPUT_CHANGED: ("kind", "key", "by"),
    _ET.PLAN_EXPANDED: ("added#",),
    _ET.PLAN_REPLANNED: ("decision_id", "added#", "removed#", "retained#", "invalidated#"),
    _ET.SAFE_STOP_TRIGGERED: ("trigger", "reason"),
    _ET.SAFE_STOP_OVERRIDDEN: ("by", "reason"),
    _ET.KILL_SWITCH_DETECTED: ("scope", "by", "reason"),
    _ET.AUDIT_TAIL_REPAIRED: ("repair",),
}


# =============================================================================
# Markdown-safe text (shared with workflows/summary.py)
# =============================================================================


def _defang(text: str) -> str:
    """Break bare ``scheme://``, ``www.`` and e-mail autolinks (``escape_markdown`` keeps them)."""
    text = text.replace("://", ":\\/\\/").replace("@", "\\@")
    return _WWW.sub(lambda match: match.group(0)[:-1] + "\\.", text)


def md_text(value: object, limit: int = _TEXT) -> str:
    """Untrusted text as one line of literal Markdown: no controls, HTML, links or markup."""
    return _defang(escape_markdown(sanitize_for_terminal(value, limit), limit * 6))


def md_cell(value: object, limit: int = _CELL) -> str:
    """Untrusted text for one table cell (no pipes, newlines, HTML or links)."""
    return _defang(table_cell(value, limit))


def md_ident(value: object, limit: int = 80) -> str:
    """An identifier as an inline code span; only ``A-Za-z0-9._@:+*-`` survive, so no markup can
    escape the span (anything else becomes ``?``)."""
    cleaned = _ID_CHARS.sub("?", sanitize_for_terminal(value, limit))
    return f"`{cleaned}`" if cleaned else "`-`"


# =============================================================================
# Plain facts (report.json)
# =============================================================================


def _t(value: object, limit: int = _TEXT) -> str:
    return sanitize_for_terminal("" if value is None else value, limit)


def _number(identifier: str) -> int:
    """The digits of an id such as ``D-0042`` as a number (``D-10000`` sorts after ``D-9999``)."""
    digits = "".join(ch for ch in identifier if ch in "0123456789")
    return int(digits) if digits else 0


def _hex(value: object, limit: int = 64) -> str:
    """Hex digits only (hashes are hex by contract; anything else is dropped)."""
    if not isinstance(value, str):
        return ""
    return "".join(ch for ch in value.lower() if ch in _HEX)[:limit]


def _ts(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _num(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value * 100:.1f}%"


def _fact(value: object, *, count: bool = False) -> str:
    """One event data value as short plain text (scalars, short lists, list sizes)."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value) if math.isfinite(value) else ""
    if isinstance(value, str):
        return sanitize_for_terminal(value, 80)
    if isinstance(value, (list, tuple)):
        if count:
            return str(len(value))
        shown = [_fact(item) for item in value[:5] if isinstance(item, (str, int, float))]
        extra = f",+{len(value) - 5}" if len(value) > 5 else ""
        return ",".join(shown) + extra
    return ""


def _detail(event: AuditEvent) -> str:
    parts: list[str] = []
    data = event.data if isinstance(event.data, Mapping) else {}
    for key in _TIMELINE[event.type]:
        name = key.rstrip("#")
        text = _fact(data.get(name), count=key.endswith("#"))
        if text:
            parts.append(f"{name}={text}")
    return sanitize_for_terminal("; ".join(parts), _TEXT)


def _clip_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """At most ``_MAX_ROWS`` rows: the first and the last half, plus how many were left out."""
    if len(rows) <= _MAX_ROWS:
        return rows, 0
    half = _MAX_ROWS // 2
    return [*rows[:half], *rows[-half:]], len(rows) - 2 * half


def _graph_index(graph: object) -> dict[str, Mapping[str, Any]]:
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
    out: dict[str, Mapping[str, Any]] = {}
    if isinstance(nodes, list):
        for item in nodes:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                out[item["id"]] = item
    return out


def _audit_head(state: RunState, events: Sequence[AuditEvent]) -> dict[str, Any]:
    if events:
        return {"seq": events[-1].seq, "hash": _hex(events[-1].hash), "count": len(events)}
    recorded = state.audit_head if isinstance(state.audit_head, Mapping) else {}
    seq = recorded.get("seq")
    return {
        "seq": seq if isinstance(seq, int) and not isinstance(seq, bool) else 0,
        "hash": _hex(recorded.get("hash")),
        "count": 0,
    }


def _node_facts(
    state: RunState, metrics: RunMetrics, events: Sequence[AuditEvent]
) -> list[dict[str, Any]]:
    specs = _graph_index(state.graph)
    first_start: dict[str, int] = {}
    for event in events:
        if event.type is _ET.NODE_STARTED and event.node_id is not None:
            first_start.setdefault(event.node_id, event.seq)
    ordered = sorted(
        state.nodes.items(), key=lambda item: (first_start.get(item[0], _FIRST_SEEN_LAST), item[0])
    )
    rows: list[dict[str, Any]] = []
    for node_id, ns in ordered:
        spec = specs.get(node_id, {})
        agent = ns.attempts[-1].agent_id if ns.attempts else spec.get("agent", "")
        rows.append(
            {
                "node_id": _t(node_id, 80),
                "stage": _t(spec.get("stage"), 32),
                "status": ns.status.value,
                "attempts": max(ns.attempt_counter, len(ns.attempts)),
                "epoch": ns.epoch.value,
                "epoch_no": ns.epoch_no,
                "duration_s": metrics.node_latency_s.get(node_id),
                "agent": _t(agent, 80),
                "cache_hits": ns.cache_hits,
                "reason": _t(ns.reason),
            }
        )
    return rows


def _timeline_facts(events: Sequence[AuditEvent]) -> list[dict[str, Any]]:
    return [
        {
            "seq": event.seq,
            "ts": _ts(event.ts),
            "type": event.type.value,
            "node_id": _t(event.node_id, 80) if event.node_id is not None else None,
            "attempt": event.attempt,
            "actor": _t(event.actor.label, 80),
            "detail": _detail(event),
        }
        for event in events
        if event.type in _TIMELINE
    ]


def _approval_facts(approvals: Sequence[ApprovalRequest]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request in sorted(approvals, key=lambda r: (r.created_at, r.id)):
        votes = [
            {
                "approver": _t(vote.approver, 64),
                "approved": bool(vote.approved),
                "channel": _t(vote.channel, 24),
                "comment": _t(vote.comment),
            }
            for vote in request.votes
        ]
        rows.append(
            {
                "id": _t(request.id, 32),
                "node_id": _t(request.node_id, 80),
                "purpose": request.purpose.value,
                "position": request.position.value,
                "risk": request.risk.name,
                "status": request.status.value,
                "required": request.required_approvals,
                "scripted": any(v["channel"] == SCRIPTED_CHANNEL for v in votes),
                "votes": votes,
            }
        )
    return rows


def _decision_facts(decisions: Sequence[Decision]) -> list[dict[str, Any]]:
    return [
        {
            "id": _t(d.id, 32),
            "kind": d.kind.value,
            "actor": _t(d.actor.label, 80),
            "node_id": _t(d.node_id, 80) if d.node_id is not None else None,
            "summary": _t(d.summary),
        }
        for d in sorted(decisions, key=lambda d: (_number(d.id), d.id))
    ]


def _artifact_facts(artifacts: Sequence[Artifact]) -> list[dict[str, Any]]:
    return [
        {
            "id": _t(a.id, 120),
            "kind": _t(a.kind, 48),
            "key": _t(a.key, 80),
            "version": a.version,
            "status": a.status.value,
            "producer": _t(a.producer, 80),
            "size": a.size,
            "hash_prefix": _hex(a.content_hash, _HASH_PREFIX),
        }
        for a in sorted(artifacts, key=lambda a: (a.kind, a.key, a.version, a.id))
    ]


# =============================================================================
# Limitations (computed from the facts, never boilerplate about things that did not happen)
# =============================================================================


@dataclass(frozen=True)
class _Line:
    """One sentence in two renderings: plain (report.json) and Markdown-safe (report.md)."""

    plain: str
    markdown: str


def _line(template: str, *parts: object) -> _Line:
    """``template`` is trusted static text with one ``{}`` per untrusted part."""
    return _Line(
        template.format(*(_t(part, _PART) for part in parts)),
        template.format(*(md_text(part, _PART) for part in parts)),
    )


def _named(rows: Sequence[Mapping[str, Any]], limit: int = 8) -> str:
    items = [f"{r['node_id']} ({r['reason']})" if r["reason"] else str(r["node_id"]) for r in rows]
    extra = f", +{len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(items[:limit]) + extra


def _limitations(
    state: RunState,
    metrics: RunMetrics,
    posture: str,
    nodes: Sequence[Mapping[str, Any]],
    approvals: Sequence[Mapping[str, Any]],
    omitted: Mapping[str, int],
    *,
    scripted: bool,
) -> list[_Line]:
    out: list[_Line] = []
    if state.status is not RunStatus.SUCCEEDED:
        if state.status.is_terminal:
            out.append(
                _line(
                    "The run did not succeed: final status {} - {}.",
                    state.status.value,
                    state.status_reason or "no reason recorded",
                )
            )
        else:
            out.append(
                _line(
                    "The run is not finished (status {}): this report is a snapshot, not a final "
                    "record.",
                    state.status.value,
                )
            )
    if state.status.is_terminal and metrics.status != state.status.value:
        out.append(
            _line(
                "state.json says {} but the audit log says {}: treat this run as unverified.",
                state.status.value,
                metrics.status,
            )
        )
    if state.safe_stop is not None:
        out.append(
            _line(
                "Safe-stop was triggered by {}: {}.",
                state.safe_stop.trigger,
                state.safe_stop.reason,
            )
        )
        if state.safe_stop.compensation_incomplete:
            out.append(
                _line(
                    "Compensation is incomplete for: {}.",
                    ", ".join(state.safe_stop.compensation_incomplete),
                )
            )
    for status, label in (
        (NodeStatus.FAILED, "failed"),
        (NodeStatus.BLOCKED, "were blocked"),
        (NodeStatus.CANCELLED, "were cancelled"),
        (NodeStatus.ROLLED_BACK, "were rolled back"),
        (NodeStatus.SKIPPED, "were skipped"),
    ):
        rows = [r for r in nodes if r["status"] == status.value]
        if rows:
            out.append(_line("{} node(s) " + label + ": {}.", len(rows), _named(rows)))
    if scripted:
        ids = ", ".join(str(a["id"]) for a in approvals if a["scripted"])
        out.append(
            _line(
                "Approvals were scripted (assurance DEMO_SCRIPTED), not a human review"
                + (": scripted votes on {}." if ids else "."),
                *((ids,) if ids else ()),
            )
        )
    out.append(
        _line(
            "Approver identity is attested, not authenticated (four-eyes: ATTESTED-ONLY): "
            "names are asserted by the caller."
        )
    )
    if posture != "NOMINAL":
        out.append(_line("Governance posture is not nominal: {}.", posture))
    if state.waivers:
        out.append(
            _line(
                "{} policy finding(s) were waived by humans instead of fixed: {}.",
                len(state.waivers),
                ", ".join(sorted(f"{w.id} ({w.rule_id})" for w in state.waivers)),
            )
        )
    if metrics.policy_violations_by_rule:
        found = ", ".join(f"{rule} x{n}" for rule, n in metrics.policy_violations_by_rule.items())
        out.append(
            _line("Policy violations were raised ({}): see the timeline for each one.", found)
        )
    unrecovered = sorted(i.node_id for i in metrics.incidents if i.status.value == "unrecovered")
    if unrecovered:
        out.append(_line("Unrecovered incident(s) on: {}.", ", ".join(unrecovered)))
    abandoned = sorted(i.node_id for i in metrics.incidents if i.status.value == "abandoned")
    if abandoned:
        out.append(
            _line(
                "Incident(s) abandoned by a reset, removal, cancellation or safe-stop on: {}.",
                ", ".join(abandoned),
            )
        )
    if metrics.quarantines:
        out.append(
            _line(
                "{} artifact(s) were quarantined (rejected, discarded or rolled-back attempts).",
                metrics.quarantines,
            )
        )
    if state.resume_count:
        out.append(
            _line(
                "The run was resumed {} time(s), after waiting for humans or an interruption.",
                state.resume_count,
            )
        )
    if any(omitted.values()):
        shortened = ", ".join(f"{name} {count}" for name, count in omitted.items() if count)
        out.append(
            _line(
                "Lists were shortened in this report (omitted: {}); trace.json, decisions.jsonl "
                "and the audit log hold everything.",
                shortened,
            )
        )
    out.append(
        _line(
            "The audit head is anchored only in state.json and seal.json (no external anchor): "
            "the hash chain detects edits and truncation, not an attacker who rewrites the log "
            "together with both files."
        )
    )
    return out


# =============================================================================
# Markdown
# =============================================================================


def _cell(value: object) -> str:
    return md_cell("" if value is None else value)


def _id(value: object) -> str:
    """An identifier-like fact as a code span (``-`` when empty)."""
    return "-" if value is None or value == "" else md_ident(value)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return [*lines, ""]


def _omitted_note(count: int, what: str) -> list[str]:
    return [f"_{count} more {what} omitted from the middle of this list._", ""] if count else []


def _votes_cell(votes: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for vote in votes:
        text = f"{vote['approver']} {'approved' if vote['approved'] else 'rejected'}"
        text += f" via {vote['channel']}"
        if vote["comment"]:
            text += f": {vote['comment']}"
        parts.append(_cell(text))
    return "; ".join(parts) if parts else "-"


def _metric_rows(m: RunMetrics) -> list[tuple[str, str]]:
    violations = ", ".join(f"{rule} x{n}" for rule, n in m.policy_violations_by_rule.items())
    kinds: dict[str, int] = {}
    for incident in m.incidents:
        kinds[incident.status.value] = kinds.get(incident.status.value, 0) + 1
    return [
        ("Success", "yes" if m.success else "no"),
        ("Attempts (countable)", f"{m.attempts} ({m.countable_attempts})"),
        ("Retries", str(m.retries)),
        ("Retry frequency", _pct(m.retry_frequency)),
        ("Fallbacks (rate)", f"{m.fallbacks} ({_pct(m.fallback_rate)})"),
        ("Rollbacks", str(m.rollbacks)),
        ("Safe stops", str(m.safe_stops)),
        ("Re-plans", str(m.replans)),
        ("MTTR (s)", _num(m.mttr_s)),
        ("End-to-end latency (s)", _num(m.e2e_latency_s)),
        ("Active latency (s)", _num(m.active_latency_s)),
        ("Human wait (s)", _num(m.human_wait_s)),
        ("Critical path (s)", _num(m.critical_path_s)),
        ("Max concurrency observed", str(m.max_concurrency_observed)),
        (
            "Re-plan reuse",
            f"{_pct(m.replan_reuse_ratio)} ({m.cache_hits} cache hits / "
            f"{m.revalidated_nodes} revalidated)",
        ),
        ("First-pass yield", _pct(m.first_pass_yield)),
        ("Approvals requested / automatic", f"{m.approvals_requested} / {m.approvals_auto}"),
        (
            "Incidents (recovered / unrecovered / abandoned)",
            f"{kinds.get('recovered', 0)} / {kinds.get('unrecovered', 0)} / "
            f"{kinds.get('abandoned', 0)}",
        ),
        ("Policy violations by rule", violations or "none"),
    ]


def _markdown(facts: Mapping[str, Any], limitations: Sequence[_Line], metrics: RunMetrics) -> str:
    run = facts["run"]
    head = run["audit_head"]
    omitted = facts["omitted"]
    status = f"- **Status:** {_id(run['status'])}"
    if run["status_reason"]:
        status += f" - {md_text(run['status_reason'])}"
    out: list[str] = [f"# Run report: {md_text(run['run_id'], 80)}", ""]
    if run["assurance"] == ASSURANCE_SCRIPTED:
        out += [
            "> **ASSURANCE: DEMO_SCRIPTED** - approvals were scripted, not a human review.",
            "",
        ]
    out += [
        f"- **Scenario:** {md_text(run['scenario'], 80)}",
        status,
        f"- **Assurance:** {_id(run['assurance'])}",
        f"- **Governance posture:** {md_text(run['posture'], 300)}",
        f"- **Autonomy mode:** {_id(run['autonomy_mode'])}",
        f"- **Created by:** {md_text(run['created_by'], 80)}",
        f"- **Four-eyes:** {run['four_eyes']}",
        f"- **Audit head:** seq {head['seq']}, hash `{head['hash'] or '-'}`, "
        f"{head['count']} events",
        "",
        "## Nodes",
        "",
    ]
    if facts["nodes"]:
        out += _table(
            ["Node", "Stage", "Status", "Attempts", "Epoch", "Duration (s)", "Agent", "Note"],
            [
                [
                    _id(n["node_id"]),
                    _id(n["stage"]),
                    _id(n["status"]),
                    str(n["attempts"]),
                    _id(n["epoch"]) + (f" #{n['epoch_no']}" if n["epoch_no"] else ""),
                    _num(n["duration_s"]),
                    _id(n["agent"]),
                    _cell(n["reason"]),
                ]
                for n in facts["nodes"]
            ],
        )
        out += _omitted_note(omitted["nodes"], "nodes")
    else:
        out += ["No nodes were recorded.", ""]
    out += ["## Timeline of key events", ""]
    if facts["timeline"]:
        out += _table(
            ["Seq", "Time (UTC)", "Event", "Node", "Attempt", "Detail"],
            [
                [
                    str(e["seq"]),
                    _cell(e["ts"]),
                    _id(e["type"]),
                    _id(e["node_id"]),
                    "-" if e["attempt"] is None else str(e["attempt"]),
                    _cell(e["detail"]),
                ]
                for e in facts["timeline"]
            ],
        )
        out += _omitted_note(omitted["timeline"], "key events")
    else:
        out += ["No key events were recorded.", ""]
    out += [
        "## Approvals",
        "",
        f"Four-eyes: {run['four_eyes']} (identities are asserted names).",
        "",
    ]
    if facts["approvals"]:
        out += _table(
            ["Request", "Node", "Purpose", "Risk", "Status", "Required", "Votes"],
            [
                [
                    _id(a["id"]),
                    _id(a["node_id"]),
                    _id(a["purpose"]),
                    _id(a["risk"]),
                    _id(a["status"]),
                    str(a["required"]),
                    _votes_cell(a["votes"]),
                ]
                for a in facts["approvals"]
            ],
        )
        out += _omitted_note(omitted["approvals"], "approval requests")
    else:
        out += ["No approval was requested.", ""]
    out += ["## Decisions", ""]
    if facts["decisions"]:
        out += _table(
            ["Id", "Kind", "Actor", "Summary"],
            [
                [_id(d["id"]), _id(d["kind"]), _id(d["actor"]), _cell(d["summary"])]
                for d in facts["decisions"]
            ],
        )
        out += _omitted_note(omitted["decisions"], "decisions")
    else:
        out += ["No decision was recorded.", ""]
    out += ["## Artifacts", ""]
    if facts["artifacts"]:
        out += _table(
            ["Id", "Kind", "Status", "Producer", "Hash"],
            [
                [
                    _id(a["id"]),
                    _id(a["kind"]),
                    _id(a["status"]),
                    _id(a["producer"]),
                    f"`{a['hash_prefix'] or '-'}`",
                ]
                for a in facts["artifacts"]
            ],
        )
        out += _omitted_note(omitted["artifacts"], "artifacts")
    else:
        out += ["No artifact was produced.", ""]
    out += ["## Reliability metrics", ""]
    out += _table(
        ["Metric", "Value"], [[name, _cell(value)] for name, value in _metric_rows(metrics)]
    )
    out += ["## Limitations", ""]
    out += [f"- {line.markdown}" for line in limitations]
    return "\n".join(out) + "\n"


# =============================================================================
# Public API
# =============================================================================


def render_report(
    *,
    state: RunState,
    events: Sequence[AuditEvent],
    decisions: Sequence[Decision],
    metrics: RunMetrics,
    artifacts: Sequence[Artifact],
    posture: str,
) -> tuple[str, dict[str, Any]]:
    """``(report.md text, report.json dict)`` of a run; a pure function of its arguments."""
    nodes_all = _node_facts(state, metrics, events)
    nodes, nodes_omitted = _clip_rows(nodes_all)
    timeline, timeline_omitted = _clip_rows(_timeline_facts(events))
    approvals_all = _approval_facts(list(state.approvals.values()))
    approvals, approvals_omitted = _clip_rows(approvals_all)
    decision_rows, decisions_omitted = _clip_rows(_decision_facts(decisions))
    artifact_rows, artifacts_omitted = _clip_rows(_artifact_facts(artifacts))
    omitted = {
        "nodes": nodes_omitted,
        "timeline": timeline_omitted,
        "approvals": approvals_omitted,
        "decisions": decisions_omitted,
        "artifacts": artifacts_omitted,
    }
    scripted = state.assurance == ASSURANCE_SCRIPTED or any(a["scripted"] for a in approvals_all)
    limitations = _limitations(
        state, metrics, posture, nodes_all, approvals_all, omitted, scripted=scripted
    )
    facts: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "run": {
            "run_id": _t(state.run_id, 80),
            "scenario": _t(state.scenario, 80),
            "status": state.status.value,
            "status_reason": _t(state.status_reason),
            "created_by": _t(state.created_by, 80),
            "created_at": _ts(state.created_at),
            "started_at": _ts(state.started_at),
            "finished_at": _ts(state.finished_at),
            "assurance": ASSURANCE_SCRIPTED if scripted else _t(state.assurance, 32),
            "autonomy_mode": _t(state.autonomy_mode, 32),
            "posture": _t(posture, 300),
            "four_eyes": "ATTESTED-ONLY",
            "audit_head": _audit_head(state, events),
            "graph_hash": _hex(state.graph_hash),
            "baseline_manifest_hash": _hex(state.baseline_manifest_hash),
            "resume_count": state.resume_count,
        },
        "nodes": nodes,
        "timeline": timeline,
        "approvals": approvals,
        "decisions": decision_rows,
        "artifacts": artifact_rows,
        "metrics": metrics.to_dict(),
        "limitations": [line.plain for line in limitations],
        "omitted": omitted,
    }
    return _markdown(facts, limitations, metrics), facts


def _file_sha256(directory: Path, relative: str) -> str | None:
    """SHA-256 of ``directory/relative`` (``None`` if absent); links are refused, never followed."""
    path = directory
    for part in relative.split("/"):
        path = path / part
        if is_link_or_reparse(path):
            raise IntegrityError(f"refusing to seal through a link: {relative}")
    if not path.is_file():
        return None
    return sha256_bytes(read_bytes_retry(path))


def write_run_outputs(
    run_dir: Path,
    *,
    state: RunState,
    events: Sequence[AuditEvent],
    decisions: Sequence[Decision],
    metrics: RunMetrics,
    artifacts: Sequence[Artifact],
    posture: str,
) -> dict[str, str]:
    """Write the five run outputs into ``run_dir`` and return ``{file name: sha256}``.

    ``report.md`` / ``report.json`` come from :func:`render_report`, ``metrics.json`` is
    ``metrics.to_dict()``, ``trace.json`` is ``audit.export_trace(events)``; ``seal.json`` (written
    last) records the audit head and the hashes of the run's record files (see the module docs).
    """
    directory = Path(run_dir)
    markdown, report = render_report(
        state=state,
        events=events,
        decisions=decisions,
        metrics=metrics,
        artifacts=artifacts,
        posture=posture,
    )
    written: dict[str, str] = {}
    data = markdown.encode("utf-8", "replace")
    atomic_write_bytes(directory / "report.md", data)
    written["report.md"] = sha256_bytes(data)
    for name, document in (
        ("report.json", report),
        ("metrics.json", metrics.to_dict()),
        ("trace.json", export_trace(events)),
    ):
        atomic_write_json(directory / name, document)
        written[name] = sha256_bytes(read_bytes_retry(directory / name))
    covered = {
        name: _file_sha256(directory, name)
        for name in ("state.json", "artifacts/index.json", "decisions.jsonl")
    }
    head_seq, head_hash = (events[-1].seq, events[-1].hash) if events else (0, GENESIS_HASH)
    seal = {
        "schema": REPORT_SCHEMA,
        "run_id": _t(state.run_id, 80),
        "head_seq": head_seq,
        "head_hash": head_hash,
        "count": len(events),
        "state_sha256": covered["state.json"],
        "artifact_index_sha256": covered["artifacts/index.json"],
        "decisions_sha256": covered["decisions.jsonl"],
        "files": {**{n: h for n, h in covered.items() if h is not None}, **written},
    }
    atomic_write_json(directory / "seal.json", seal)
    written["seal.json"] = sha256_bytes(read_bytes_retry(directory / "seal.json"))
    return written
