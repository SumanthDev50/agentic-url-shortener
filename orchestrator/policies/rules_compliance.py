"""CMP-001 ... CMP-003: compliance guardrails (catalogue: docs/architecture.md section 1.5).

CMP-001 / CMP-003 are release-time rules over a ``RunSubject`` (evidence = artifacts, decisions,
approvals and waivers of the run, restricted to ancestors of the gate position - see ``runview``).
CMP-002 is a heuristic personal-data hygiene check over change sets.  All of them fail closed on
missing evidence.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    EXTERNAL_PRODUCER,
    ActorType,
    ApprovalRequest,
    ApprovalStatus,
    ChangeSetSubject,
    Decision,
    DecisionKind,
    Finding,
    NodeStatus,
    PolicyCategory,
    PolicyRule,
    RiskLevel,
    RunSubject,
    Severity,
    SubjectKind,
)
from orchestrator.policies import pyscan
from orchestrator.policies.base import RuleBase, python_ops, written_ops
from orchestrator.policies.config import CATALOGUE, PolicyConfig
from orchestrator.policies.hits import Hit, format_location, limit_hit
from orchestrator.policies.runview import considered_producers, latest_evidence, read_payload

_MAX_FINDINGS = 25


def _human_decisions(decisions: Sequence[Decision], *kinds: DecisionKind) -> list[Decision]:
    return [
        d for d in decisions if d.actor.type == ActorType.HUMAN and (not kinds or d.kind in kinds)
    ]


# ---------------------------------------------------------------------------------------- CMP-001


class Traceability(RuleBase):
    """Every active acceptance criterion needs one of: a *passed* covering test;
    ``verification=inspection`` with evidence (a passing readiness check with evidence, or a human
    decision citing the id in ``refs``); or - for a deferred/descoped requirement - a ``SCOPE``
    decision recorded by a human actor.
    """

    rule_id = "CMP-001"
    category = PolicyCategory.COMPLIANCE
    title = "Requirement traceability"
    severity = Severity.HIGH
    subjects = frozenset({SubjectKind.RUN})

    def check_run(self, subject: RunSubject) -> list[Finding]:
        requirements = read_payload(subject, latest_evidence(subject, Kind.REQUIREMENTS))
        listed = requirements.get("requirements") if requirements else None
        if not isinstance(listed, list):
            return [
                self.finding(
                    "no readable requirements artifact upstream of the gate: traceability is "
                    "unverifiable",
                    location=subject.node_id or None,
                )
            ]
        report = read_payload(subject, latest_evidence(subject, Kind.TEST_REPORT))
        passed, not_passed = self._coverage(report)
        decisions = subject.view.decisions()
        inspections = self._inspection_ids(subject, decisions)
        findings: list[Finding] = []
        for req in listed:
            if isinstance(req, Mapping):
                findings.extend(
                    self._check_requirement(
                        req, passed, not_passed, inspections, decisions, report is not None
                    )
                )
        return findings[:_MAX_FINDINGS]

    @staticmethod
    def _coverage(report: Mapping[str, Any] | None) -> tuple[set[str], set[str]]:
        """Ids covered by passed tests, and ids covered only by tests that did not pass."""
        passed: set[str] = set()
        other: set[str] = set()
        tests = report.get("tests") if report else None
        for test in tests if isinstance(tests, list) else []:
            if not isinstance(test, Mapping):
                continue
            covers = test.get("covers")
            ids = {c for c in covers if isinstance(c, str)} if isinstance(covers, list) else set()
            (passed if test.get("outcome") == "passed" else other).update(ids)
        return passed, other

    def _inspection_ids(self, subject: RunSubject, decisions: Sequence[Decision]) -> set[str]:
        ids: set[str] = set()
        readiness = read_payload(subject, latest_evidence(subject, Kind.READINESS_REPORT))
        for check in (readiness.get("checks") if readiness else None) or []:
            evidence = check.get("evidence") if isinstance(check, Mapping) else None
            if (
                isinstance(check, Mapping)
                and check.get("passed") is True
                and isinstance(evidence, list)
                and any(isinstance(e, str) and e.strip() for e in evidence)
            ):
                ids.add(str(check.get("id")))
        for decision in _human_decisions(decisions):
            ids.update(decision.refs)
        return ids

    def _check_requirement(
        self,
        req: Mapping[str, Any],
        passed: set[str],
        not_passed: set[str],
        inspections: set[str],
        decisions: Sequence[Decision],
        have_report: bool,
    ) -> list[Finding]:
        rid = str(req.get("id", "?"))
        status = req.get("status")
        criteria = [
            str(c.get("id")) for c in req.get("acceptance_criteria") or [] if isinstance(c, Mapping)
        ]
        if status in ("deferred", "descoped"):
            return (
                []
                if self._has_human_scope(req, rid, criteria, decisions)
                else [self._gap(rid, f"{status} without a human SCOPE decision")]
            )
        if status != "active":
            return [self._gap(rid, f"unknown requirement status {status!r}")]
        if req.get("verification") == "inspection":
            return (
                []
                if inspections & {rid, *criteria}
                else [self._gap(rid, "verification=inspection but no inspection evidence")]
            )
        targets = criteria or [rid]
        findings: list[Finding] = []
        for target in targets:
            if target in passed or target in inspections:
                continue
            if target in not_passed:
                reason = "covered only by tests that did not pass"
            elif not have_report:
                reason = "no test report upstream of the gate"
            else:
                reason = "no passed covering test"
            findings.append(self._gap(f"{rid}/{target}" if target != rid else rid, reason, target))
        return findings

    def _gap(self, location: str, reason: str, evidence: str | None = None) -> Finding:
        return self.finding(
            f"traceability gap: {reason}",
            location=location,
            evidence=evidence,
            remediation=(
                "Add a test with a 'Covers:' line for the criterion, record inspection evidence, "
                "or defer it with a human SCOPE decision."
            ),
        )

    @staticmethod
    def _has_human_scope(
        req: Mapping[str, Any], rid: str, criteria: list[str], decisions: Sequence[Decision]
    ) -> bool:
        scope = _human_decisions(decisions, DecisionKind.SCOPE)
        recorded = req.get("status_decision")
        if isinstance(recorded, str) and recorded:
            return any(d.id == recorded for d in scope)
        return any({rid, *criteria} & set(d.refs) for d in scope)


# ---------------------------------------------------------------------------------------- CMP-002

_PII_COLUMNS = frozenset(
    {
        "ip",
        "ip_address",
        "ipaddress",
        "client_ip",
        "remote_addr",
        "remote_ip",
        "user_agent",
        "useragent",
        "ua",
    }
)
_PII_NAMES = frozenset(
    {"ip_address", "client_ip", "remote_addr", "remote_ip", "user_agent", "x_forwarded_for"}
)
_PII_ATTRS = frozenset({"remote_addr", "remote_ip", "client_ip"})
_PII_HEADERS = frozenset({"x-forwarded-for", "x-real-ip", "user-agent", "forwarded"})
_SANITISER = re.compile(
    r"hash|hmac|anonym|truncat|mask|digest|sha|redact|pseudonym|salt", re.IGNORECASE
)
_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}
)
_PERSIST_METHODS = frozenset(
    {
        "execute",
        "executemany",
        "executescript",
        "insert",
        "insert_one",
        "insert_many",
        "write",
        "write_text",
    }
)
_INSERT_COLUMNS = re.compile(
    r"\binsert\s+(?:or\s+\w+\s+)?into\s+[\w.\"`\[\]]{1,128}\s*\(([^)]{1,500})\)", re.IGNORECASE
)
_ADD_COLUMN = re.compile(
    r"\balter\s+table\s+[\w.\"`\[\]]{1,128}\s+add\s+(?:column\s+)?[\"`\[]?(\w{1,64})", re.IGNORECASE
)


def _callee_text(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_sanitiser(call: ast.Call) -> bool:
    return bool(_SANITISER.search(_callee_text(call.func)))


def _pii_source(expr: ast.AST) -> str | None:
    """First raw personal-data source in ``expr`` not wrapped in a hashing/masking call."""
    stack: list[ast.AST] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Call) and _is_sanitiser(node):
            continue
        if isinstance(node, ast.Attribute):
            if node.attr in _PII_ATTRS:
                return node.attr
            if (
                node.attr == "host"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "client"
            ):
                return "client.host"
        elif isinstance(node, ast.Name) and node.id.lower() in _PII_NAMES:
            return node.id
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.strip().lower() in _PII_HEADERS
        ):
            return node.value.strip().lower()
        stack.extend(ast.iter_child_nodes(node))
    return None


def _sink_kind(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id == "print":
        return "logged"
    if isinstance(call.func, ast.Attribute):
        attr = call.func.attr
        if attr in _LOG_METHODS and "log" in _callee_text(call.func.value).lower():
            return "logged"
        if attr in _PERSIST_METHODS:
            return "persisted"
    return None


def _sql_columns(sql: str) -> Iterator[str]:
    for match in _INSERT_COLUMNS.finditer(sql):
        yield from (
            c.strip().strip('"`[] ').split(".")[-1].lower() for c in match.group(1).split(",")
        )
    for match in _ADD_COLUMN.finditer(sql):
        yield match.group(1).lower()


class PersonalDataHygiene(RuleBase):
    rule_id = "CMP-002"
    category = PolicyCategory.COMPLIANCE
    title = "Personal-data hygiene (raw IP / user-agent)"
    severity = Severity.MEDIUM
    subjects = frozenset({SubjectKind.CHANGESET})

    _FIX = (
        "Hash with a salted HMAC or truncate before logging or persisting client IP / user-agent."
    )

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        hits: list[Hit] = []
        for op in python_ops(subject):
            hits.extend(pyscan.analyse(op.content or "", op.path, self._python_hits))
        for op in written_ops(subject, ".sql"):
            hits.extend(self._sql_hits(op.content or "", op.path))
        return self.from_hits(hits, self._FIX)

    @staticmethod
    def _python_hits(tree: ast.AST, label: str) -> list[Hit]:
        hits: list[Hit] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                sink = _sink_kind(node)
                source = _first_source(node) if sink else None
                if sink and source:
                    hits.append(
                        Hit(
                            f"pii_{sink}",
                            format_location(label, node.lineno),
                            f"personal data ({source}) is {sink} without hashing or truncation",
                            source,
                        )
                    )
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and len(node.value) < 4000
            ):
                hits.extend(_column_hits(node.value, label, node.lineno))
        return hits

    @staticmethod
    def _sql_hits(text: str, label: str) -> list[Hit]:
        capped = limit_hit(text, label)
        if capped is not None:
            return [capped]
        return _column_hits(text, label, 0)


def _first_source(call: ast.Call) -> str | None:
    for arg in (*call.args, *(k.value for k in call.keywords)):
        found = _pii_source(arg)
        if found:
            return found
    return None


def _column_hits(sql: str, label: str, line: int) -> list[Hit]:
    return [
        Hit(
            "pii_column",
            format_location(label, line),
            f"column {col!r} stores raw personal data",
            col,
        )
        for col in _sql_columns(sql)
        if col in _PII_COLUMNS
    ]


# ---------------------------------------------------------------------------------------- CMP-003


class DecisionCompleteness(RuleBase):
    """High-impact nodes carry valid, non-stale, sufficient approvals; every waiver has a ``WAIVER``
    decision.

    Staleness: the *latest* non-cancelled request of a node must be the APPROVED one (an INVALIDATED
    or newer undecided request means the approval no longer covers the output).
    ``params()['node_fingerprints']`` (``{node_id: current fingerprint}``), when the engine supplies
    it, is compared with the request's ``node_fingerprint``; otherwise lazy invalidation
    (engine-semantics 10.5) is trusted via the status.
    """

    rule_id = "CMP-003"
    category = PolicyCategory.COMPLIANCE
    title = "Decision-record completeness"
    severity = Severity.HIGH
    subjects = frozenset({SubjectKind.RUN})

    def check_run(self, subject: RunSubject) -> list[Finding]:
        view = subject.view
        specs, statuses = view.node_specs(), view.node_statuses()
        approvals, decisions = view.approvals(), view.decisions()
        fingerprints = view.params().get("node_fingerprints")
        eligible = considered_producers(subject) - {EXTERNAL_PRODUCER, subject.node_id}
        findings: list[Finding] = []
        for node_id in sorted(eligible & set(specs)):
            spec = specs[node_id]
            if spec.approval is None or statuses.get(node_id) in (
                NodeStatus.SKIPPED,
                NodeStatus.REMOVED,
            ):
                continue
            requests = sorted(
                (
                    a
                    for a in approvals
                    if a.node_id == node_id and a.status != ApprovalStatus.CANCELLED
                ),
                key=lambda a: (a.created_at, a.id),
            )
            expected_fp = fingerprints.get(node_id) if isinstance(fingerprints, Mapping) else None
            findings.extend(
                self._node_findings(node_id, spec.approval.always, requests, decisions, expected_fp)
            )
        findings.extend(self._waiver_findings(subject, approvals, decisions))
        return findings[:_MAX_FINDINGS]

    def _node_findings(
        self,
        node_id: str,
        always: bool,
        requests: list[ApprovalRequest],
        decisions: Sequence[Decision],
        expected_fp: Any,
    ) -> list[Finding]:
        if not requests:
            auto = any(d.kind == DecisionKind.APPROVAL and d.node_id == node_id for d in decisions)
            if always or not auto:
                return [
                    self._gap(
                        node_id,
                        "approval required but no approval request or automatic-approval decision "
                        "recorded",
                    )
                ]
            return []
        latest = requests[-1]
        problems: list[str] = []
        if latest.status != ApprovalStatus.APPROVED:
            problems.append(
                f"latest approval request is {latest.status.value} (stale or undecided)"
            )
        distinct = {v.approver for v in latest.votes if v.approved}
        if latest.required_approvals < 1 or len(distinct) < latest.required_approvals:
            problems.append(
                f"{len(distinct)} distinct approvals, {latest.required_approvals} required"
            )
        if latest.risk >= RiskLevel.HIGH and latest.required_approvals < 2:
            problems.append("HIGH risk requires two distinct approvers")
        if any(not v.approved for v in latest.votes):
            problems.append("a rejecting vote exists on an approved request")
        if not latest.node_fingerprint:
            problems.append("approval is not bound to a node fingerprint")
        elif isinstance(expected_fp, str) and expected_fp != latest.node_fingerprint:
            problems.append(
                "approval fingerprint does not match the node's current fingerprint (stale)"
            )
        if latest.evidence_hash and any(
            v.approved and v.evidence_hash != latest.evidence_hash for v in latest.votes
        ):
            problems.append("a vote does not cover the approved evidence")
        return [self._gap(node_id, p, latest.id) for p in problems]

    def _waiver_findings(
        self,
        subject: RunSubject,
        approvals: Sequence[ApprovalRequest],
        decisions: Sequence[Decision],
    ) -> list[Finding]:
        waiver_decisions = {d.id for d in decisions if d.kind == DecisionKind.WAIVER}
        granted = {a.id for a in approvals if a.status == ApprovalStatus.APPROVED}
        findings: list[Finding] = []
        for waiver in subject.view.waivers():
            entry = CATALOGUE.get(waiver.rule_id)
            if waiver.decision_id not in waiver_decisions:
                findings.append(
                    self._gap(waiver.id, "waiver has no WAIVER decision", waiver.rule_id)
                )
            if waiver.approval_id not in granted:
                findings.append(
                    self._gap(
                        waiver.id, "waiver is not backed by an approved request", waiver.rule_id
                    )
                )
            if entry is None or entry.severity > Severity.MEDIUM:
                findings.append(
                    self._gap(
                        waiver.id, "waiver targets a rule that may not be waived", waiver.rule_id
                    )
                )
        return findings

    def _gap(self, location: str, reason: str, evidence: str | None = None) -> Finding:
        return self.finding(
            f"decision record incomplete: {reason}",
            location=location,
            evidence=evidence,
            remediation="Obtain a valid, current approval (and WAIVER decision) before release.",
        )


def default_rules(config: PolicyConfig | None = None) -> list[PolicyRule]:
    return [Traceability(config), PersonalDataHygiene(config), DecisionCompleteness(config)]
