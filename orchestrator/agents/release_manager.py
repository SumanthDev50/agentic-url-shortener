"""Release manager agent (``release-readiness`` node): ``readiness_report`` +
``traceability_matrix``.

Everything is derived from upstream artifacts and from the governance view of the run
(``ctx.run_view``: decisions, approvals, waivers); nothing is estimated.  The node is
``cacheable=False`` because the governance view changes while the run progresses.

Traceability mirrors policy rule CMP-001 on purpose (the ``traceability`` gate re-checks it):
an *active* acceptance criterion is covered by a **passed** test whose ``covers`` lists it, or - for
``verification=inspection`` - by inspection evidence (a passing readiness check with the
requirement's
id and non-empty evidence, which this agent emits when the design and the change set show the
requirement is implemented, or a human decision citing the id).  A *deferred* / *descoped*
requirement
is accounted for only by a SCOPE decision whose provenance includes a human (human actor, or a
parent
chain that reaches one - the requirements agent records its SCOPE decision with the human
CLARIFICATION as parent).  A criterion covered only by tests that did not pass is a gap.

Readiness checks (ids): ``tests_passed``, ``fail_to_pass``, ``api_compat``, ``security_clean``,
``docs_present``, ``no_conflicts``, ``traceability_complete``, ``rollback_plan``,
``change_reference``,
``approvals_recorded`` (every recorded request is currently approved by enough distinct approvers;
the
completeness against the node specs is the engine's CMP-003), ``perf_budget`` (only when a budget
was
checked) and one check per inspection requirement.  Verdict ``GO`` iff every check passed.  The
RELEASE
decision it records cites the failing requirements and has every human decision of the run as
parent,
so lineage from the release artifacts reaches the human decisions.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from orchestrator.agents.base import BaseAgent, json_artifact
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    ActorType,
    AgentContext,
    AgentResult,
    ApprovalRequest,
    ApprovalStatus,
    AutonomyProfile,
    ChangeSet,
    Decision,
    DecisionDraft,
    DecisionKind,
    RiskLevel,
    Waiver,
)

MAX_EVIDENCE = 10
MAX_HUMAN_PARENTS = 50
MAX_PROVENANCE_NODES = 200
MAX_OPEN_ITEMS = 50
_LINE = 160
# Text that is no plan / reference: the same set as policy rule CHG-005 (the release_ready gate),
# so the rollback_plan / change_reference checks never pass what that gate would refuse.
_PLACEHOLDERS = frozenset({"n/a", "na", "none", "tbd", "todo", "-", "?", "unknown"})


def _text(value: object, limit: int = _LINE) -> str:
    return sanitize_for_terminal(value, limit)


def _state(value: object) -> str:
    """``present`` / ``empty`` / ``placeholder``: what a design text field really says."""
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return "empty"
    return "placeholder" if text.lower() in _PLACEHOLDERS else "present"


def _check(
    check_id: str, title: str, passed: bool, evidence: Iterable[str], detail: str
) -> dict[str, Any]:
    return {
        "id": check_id,
        "title": title,
        "passed": passed,
        "evidence": [_text(e) for e in list(evidence)[:MAX_EVIDENCE]],
        "detail": _text(detail, 300),
    }


# ----------------------------------------------------------------------------- governance helpers


def _is_human(decision: Decision) -> bool:
    return decision.actor.type == ActorType.HUMAN


def _has_human_provenance(decision: Decision, by_id: Mapping[str, Decision]) -> bool:
    """The decision, or a transitive parent, was made by a human (bounded breadth-first walk)."""
    queue: deque[Decision] = deque([decision])
    seen: set[str] = set()
    while queue and len(seen) < MAX_PROVENANCE_NODES:
        current = queue.popleft()
        if current.id in seen:
            continue
        seen.add(current.id)
        if _is_human(current):
            return True
        queue.extend(by_id[p] for p in current.parents if p in by_id)
    return False


def _scoped(req: Mapping[str, Any], criteria: Sequence[str], decisions: Sequence[Decision]) -> bool:
    """Is a deferred / descoped requirement accounted for by a human-backed SCOPE decision?"""
    by_id = {d.id: d for d in decisions}
    ids = {str(req["id"]), *criteria}
    marker = req.get("status_decision")
    for decision in decisions:
        if decision.kind != DecisionKind.SCOPE or not (ids & set(decision.refs)):
            continue
        if marker and not str(marker).startswith("SCOPE:") and decision.id != marker:
            continue
        if _has_human_provenance(decision, by_id):
            return True
    return False


def _approval_problems(
    approvals: Sequence[ApprovalRequest],
    waivers: Sequence[Waiver],
    decisions: Sequence[Decision],
    own_node: str,
) -> tuple[list[str], list[str]]:
    """(problems, evidence) over the recorded approvals and waivers."""
    latest: dict[str, ApprovalRequest] = {}
    for request in sorted(approvals, key=lambda a: (a.created_at, a.id)):
        if request.status != ApprovalStatus.CANCELLED and request.node_id != own_node:
            latest[request.node_id] = request
    problems: list[str] = []
    evidence: list[str] = []
    for node_id, request in sorted(latest.items()):
        distinct = {v.approver for v in request.votes if v.approved}
        if request.status != ApprovalStatus.APPROVED:
            problems.append(f"{node_id}: latest request {request.id} is {request.status.value}")
        elif request.required_approvals < 1:  # nobody had to approve: no control (as CMP-003)
            problems.append(f"{node_id}: request {request.id} requires no approval")
        elif len(distinct) < request.required_approvals:
            problems.append(
                f"{node_id}: {len(distinct)} approval(s) of {request.required_approvals} required"
            )
        elif request.risk >= RiskLevel.HIGH and request.required_approvals < 2:
            problems.append(f"{node_id}: HIGH risk with fewer than two required approvers")
        else:
            evidence.append(f"{node_id}: {request.id} approved by {len(distinct)}")
    waiver_decisions = {d.id for d in decisions if d.kind == DecisionKind.WAIVER}
    granted = {a.id for a in approvals if a.status == ApprovalStatus.APPROVED}
    for waiver in waivers:
        if waiver.decision_id not in waiver_decisions or waiver.approval_id not in granted:
            problems.append(
                f"waiver {waiver.id} is not backed by a WAIVER decision and an approval"
            )
        else:
            evidence.append(f"waiver {waiver.id}: decision {waiver.decision_id}")
    return problems, evidence


# ----------------------------------------------------------------------------- the agent


class ReleaseManagerAgent(BaseAgent):
    agent_id = "release-manager"
    role = "release-manager"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",), write_globs=(), can_execute=False, max_tool_calls=16
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        requirements = ctx.json(Kind.REQUIREMENTS)
        design = ctx.json(Kind.DESIGN_DOC)
        plan = ctx.json(Kind.TASK_PLAN)
        integrated = ChangeSet.from_dict(ctx.json(Kind.INTEGRATED_CHANGESET))
        docs = ChangeSet.from_dict(ctx.json(Kind.DOCS_CHANGESET))
        integration = ctx.json(Kind.INTEGRATION_REPORT)
        tests = ctx.json(Kind.TEST_REPORT)
        security = ctx.json(Kind.SECURITY_REPORT)
        decisions = ctx.run_view.decisions()
        changed = {*integrated.paths(), *docs.paths()}

        inspections = self._inspection_checks(requirements, plan, design, changed)
        matrix = self._matrix(requirements, plan, tests, decisions, changed, inspections)
        checks = [
            *self._core_checks(tests, security, design, docs, integration, matrix),
            self._approvals_check(ctx, decisions),
            *inspections.values(),
        ]
        perf = tests.get("perf") or {}
        if perf.get("checked"):
            checks.append(
                _check(
                    "perf_budget",
                    "Performance budget respected",
                    perf.get("ok") is True,
                    [f"budget {perf.get('budget_ms')} ms"],
                    "the generated performance test " + ("passed" if perf.get("ok") else "failed"),
                )
            )
        failing = [c for c in checks if not c["passed"]]
        verdict = "NO_GO" if failing else "GO"
        open_items = [f"{c['id']}: {c['detail']}" for c in failing]
        open_items += [f"{a['id']}: {a['text']}" for a in requirements.get("assumptions") or []]
        open_items += [
            f"{r['id']}: {r['status']}"
            for r in requirements["requirements"]
            if r["status"] != "active"
        ]
        readiness = {
            "verdict": verdict,
            "checks": checks,
            "risk": design["risk"],
            "rollback_plan": design["rollback_plan"],
            "change_reference": design["change_reference"],
            "open_items": [_text(i, 300) for i in open_items[:MAX_OPEN_ITEMS]],
        }
        decision = DecisionDraft(
            DecisionKind.RELEASE,
            f"Release readiness {verdict}: "
            f"{len(checks) - len(failing)} of {len(checks)} checks passed",
            rationale="; ".join(_text(c["id"]) for c in failing) or "every readiness check passed",
            outcome=verdict,
            parents=tuple(sorted(d.id for d in decisions if _is_human(d))[:MAX_HUMAN_PARENTS]),
            refs=tuple(sorted({str(g).split(":")[0].split("/")[0] for g in matrix["gaps"]}))[:20],
            data={"failed_checks": [c["id"] for c in failing]},
        )
        return AgentResult(
            artifacts=[
                json_artifact(Kind.READINESS_REPORT, readiness),
                json_artifact(Kind.TRACEABILITY_MATRIX, matrix),
            ],
            decisions=[decision],
            rationale=f"{verdict}: {len(failing)} failing check(s), {len(matrix['gaps'])} gap(s).",
            usage=dict(ctx.toolbox.usage()),
        )

    # ------------------------------------------------------------------ traceability
    def _inspection_checks(
        self,
        requirements: Mapping[str, Any],
        plan: Mapping[str, Any],
        design: Mapping[str, Any],
        changed: set[str],
    ) -> dict[str, dict[str, Any]]:
        """One check per active inspection requirement; its id is the requirement id (CMP-001)."""
        checks: dict[str, dict[str, Any]] = {}
        for req in sorted(requirements["requirements"], key=lambda r: str(r["id"])):
            if req["status"] != "active" or req["verification"] != "inspection":
                continue
            rid = str(req["id"])
            evidence = [
                f"component: {c['name']}"
                for c in design.get("components") or []
                if rid in (c.get("requirement_ids") or [])
            ]
            for task in plan.get("tasks") or []:
                if rid in task.get("requirement_ids", []):
                    evidence += [
                        f"file: {p}" for p in task.get("expected_paths", []) if p in changed
                    ]
            evidence = sorted(set(evidence))
            checks[rid] = _check(
                rid,
                f"Inspection of {rid}",
                bool(evidence),
                evidence,
                "design component and changed files found"
                if evidence
                else "no design component or changed file evidences the requirement",
            )
        return checks

    def _matrix(
        self,
        requirements: Mapping[str, Any],
        plan: Mapping[str, Any],
        tests: Mapping[str, Any],
        decisions: Sequence[Decision],
        changed: set[str],
        inspections: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        by_id: dict[str, list[tuple[str, str]]] = {}
        for test in tests["tests"]:
            for cid in test["covers"]:
                by_id.setdefault(cid, []).append((test["nodeid"], test["outcome"]))
        passed = {cid for cid, hits in by_id.items() if any(o == "passed" for _, o in hits)}
        evidenced = {rid for rid, check in inspections.items() if check["passed"]}
        evidenced |= {ref for d in decisions if _is_human(d) for ref in d.refs}
        rows: list[dict[str, Any]] = []
        gaps: list[str] = []
        for req in sorted(requirements["requirements"], key=lambda r: str(r["id"])):
            rid, status = str(req["id"]), str(req["status"])
            criteria = [str(c["id"]) for c in req["acceptance_criteria"]]
            entries: list[dict[str, Any]] = []
            if status == "active" and req["verification"] == "test":
                for target in criteria or [rid]:
                    hits = by_id.get(target, [])
                    covered = target in passed or target in evidenced
                    if not covered:
                        why = (
                            "covered only by tests that did not pass"
                            if hits
                            else "no passed covering test"
                        )
                        gaps.append(f"{rid}/{target}: {why}" if target != rid else f"{rid}: {why}")
                    if target != rid:
                        entries.append(
                            {
                                "id": target,
                                "tests": sorted(n for n, _ in hits),
                                "results": [o for _, o in sorted(hits)],
                                "covered": covered,
                            }
                        )
            else:
                if status == "active":
                    covered = bool(evidenced & {rid, *criteria})
                    marker, reason = (
                        "inspection",
                        "verification=inspection without inspection evidence",
                    )
                else:
                    covered = _scoped(req, criteria, decisions)
                    marker, reason = status, f"{status} without a human SCOPE decision"
                if not covered:
                    gaps.append(f"{rid}: {reason}")
                entries = [
                    {
                        "id": c,
                        "tests": [],
                        "results": [marker] if covered else [],
                        "covered": covered,
                    }
                    for c in criteria
                ]
            ids = {rid, *criteria}
            tasks = [t for t in plan.get("tasks") or [] if rid in t.get("requirement_ids", [])]
            files = {p for t in tasks for p in t.get("expected_paths", []) if p in changed}
            files |= {n.split("::")[0] for c in ids for n, _ in by_id.get(c, [])}
            rows.append(
                {
                    "id": rid,
                    "status": status,
                    "acceptance_criteria": entries,
                    "tasks": sorted(str(t["id"]) for t in tasks),
                    "files": sorted(files),
                    "decision_refs": sorted(d.id for d in decisions if ids & set(d.refs)),
                }
            )
        return {"requirements": rows, "complete": not gaps, "gaps": sorted(gaps)}

    # ------------------------------------------------------------------ checks
    def _core_checks(
        self,
        tests: Mapping[str, Any],
        security: Mapping[str, Any],
        design: Mapping[str, Any],
        docs: ChangeSet,
        integration: Mapping[str, Any],
        matrix: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        counts = {
            o: sum(1 for t in tests["tests"] if t["outcome"] == o)
            for o in ("passed", "failed", "error")
        }
        f2p = tests["fail_to_pass"]
        api = tests["api_compat"]
        declared = design.get("api_changes") or []
        declared_breaking = any(c.get("breaking") for c in declared)
        if api["checked"]:
            api_ok = not api["breaking"] or declared_breaking
            api_detail = (
                f"{len(api['breaking'])} breaking change(s)"
                + (" declared in the design" if api["breaking"] and declared_breaking else "")
                + f", {len(api['additive'])} additive"
            )
        else:
            api_ok = not declared
            api_detail = (
                "the design declares API changes but the OpenAPI diff was not verified"
                if declared
                else "the design declares no API changes"
            )
        high = [f for f in security["findings"] if f["severity"] in ("HIGH", "CRITICAL")]
        doc_paths = sorted(docs.paths())
        rollback = _state(design.get("rollback_plan"))
        reference = _state(design.get("change_reference"))
        reference_text = str(design.get("change_reference", "")).strip()
        return [
            _check(
                "tests_passed",
                "Test suite passed",
                # the summary flag is cross-checked against the rows it summarises
                bool(tests["passed"])
                and tests["total"] > 0
                and not counts["failed"]
                and not counts["error"],
                [f"test_report: {counts['passed']} passed of {tests['total']}"],
                f"{counts['failed']} failed, {counts['error']} error(s), "
                f"{tests['skipped']} skipped",
            ),
            _check(
                "fail_to_pass",
                "Authored tests fail before the change and pass after it",
                bool(f2p["checked"] and f2p["ok"]),
                [f"{len(f2p['tests'])} fail-to-pass test(s)"],
                "checked" if f2p["checked"] else "not checked",
            ),
            _check("api_compat", "API compatibility", api_ok, [api_detail], api_detail),
            _check(
                "security_clean",
                "No HIGH or CRITICAL security finding",
                bool(security["passed"]) and not high,
                [
                    f"{security['files_scanned']} file(s) scanned, "
                    f"{len(security['findings'])} finding(s)"
                ],
                f"{len(high)} HIGH or above",
            ),
            _check(
                "docs_present",
                "Documentation is part of the release",
                bool(doc_paths),
                doc_paths,
                f"{len(doc_paths)} documentation file(s)",
            ),
            _check(
                "no_conflicts",
                "No unresolved integration conflicts",
                not integration["conflicts"],
                [f"{len(integration['merged_paths'])} path(s) merged"],
                f"{len(integration['conflicts'])} conflict(s)",
            ),
            _check(
                "traceability_complete",
                "Every acceptance criterion is traced",
                bool(matrix["complete"]),
                [f"{len(matrix['requirements'])} requirement(s) in the matrix"],
                f"{len(matrix['gaps'])} gap(s)",
            ),
            _check(
                "rollback_plan",
                "Rollback plan present",
                rollback == "present",
                ["design_doc.rollback_plan"] if rollback == "present" else [],
                rollback,
            ),
            _check(
                "change_reference",
                "Change reference present",
                reference == "present",
                [f"design_doc.change_reference: {reference_text}"]
                if reference == "present"
                else [],
                reference,
            ),
        ]

    def _approvals_check(self, ctx: AgentContext, decisions: Sequence[Decision]) -> dict[str, Any]:
        problems, evidence = _approval_problems(
            ctx.run_view.approvals(), ctx.run_view.waivers(), decisions, ctx.node.id
        )
        return _check(
            "approvals_recorded",
            "Recorded approvals are current and sufficient",
            not problems,
            evidence or ["no approval request is recorded"],
            "; ".join(problems)
            if problems
            else f"{len(evidence)} approval(s) and waiver(s) verified",
        )
