"""CMP-001 (traceability), CMP-002 (personal data), CMP-003 (decision records)."""

from __future__ import annotations

from typing import Any

import pytest

from orchestrator.core.types import (
    Finding,
    Actor,
    ApprovalSpec,
    ApprovalStatus,
    ArtifactStatus,
    DecisionKind,
    RiskLevel,
    Severity,
    Stage,
)
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.rules_compliance import (
    DecisionCompleteness,
    PersonalDataHygiene,
    Traceability,
    default_rules,
)
from tests.orchestrator.policies.policy_builders import (
    FakeRunView,
    approval,
    check,
    create,
    cs_subject,
    make_finding,
    node,
    py_subject,
    release_chain,
    run_subject,
    waiver_for,
)

CFG = PolicyConfig()


def test_default_rules_are_the_catalogue() -> None:
    assert sorted(r.rule_id for r in default_rules()) == ["CMP-001", "CMP-002", "CMP-003"]


# ===============================================================================================
# CMP-001


def requirement(
    rid: str = "R-001",
    *,
    status: str = "active",
    verification: str = "test",
    criteria: tuple[str, ...] = ("AC-001-1",),
    status_decision: str | None = None,
) -> dict[str, Any]:
    return {
        "id": rid,
        "text": "text",
        "kind": "functional",
        "priority": "must",
        "status": status,
        "status_decision": status_decision,
        "capabilities": [],
        "parameters": {},
        "verification": verification,
        "acceptance_criteria": [{"id": c, "text": "t"} for c in criteria],
    }


def covering(outcome: str, *covers: str) -> dict[str, Any]:
    return {
        "nodeid": f"tests/test_x.py::t_{'_'.join(covers) or 'none'}",
        "outcome": outcome,
        "covers": list(covers),
    }


def trace_view(
    requirements: list[dict[str, Any]], tests: list[dict[str, Any]] | None = None
) -> FakeRunView:
    rv = FakeRunView(release_chain())
    rv.add_artifact(
        "requirements", {"schema": 1, "requirements": requirements}, producer="requirements"
    )
    if tests is not None:
        rv.add_artifact("test_report", {"schema": 1, "tests": tests}, producer="verify")
    return rv


def test_cmp001_every_criterion_with_a_passed_covering_test_passes() -> None:
    rv = trace_view(
        [requirement(criteria=("AC-001-1", "AC-001-2"))],
        [covering("passed", "R-001", "AC-001-1"), covering("passed", "AC-001-2")],
    )
    assert check(Traceability, run_subject(rv)) == []


def test_cmp001_uncovered_criterion_is_high_with_a_precise_location() -> None:
    rv = trace_view(
        [requirement(criteria=("AC-001-1", "AC-001-2"))], [covering("passed", "AC-001-1")]
    )
    findings = check(Traceability, run_subject(rv))
    assert [(f.rule_id, f.severity, f.location, f.evidence) for f in findings] == [
        ("CMP-001", Severity.HIGH, "R-001/AC-001-2", "AC-001-2")
    ]


@pytest.mark.parametrize("outcome", ["failed", "error", "skipped"])
def test_cmp001_only_passed_tests_count(outcome: str) -> None:
    rv = trace_view([requirement()], [covering(outcome, "AC-001-1")])
    findings = check(Traceability, run_subject(rv))
    assert len(findings) == 1
    assert "did not pass" in findings[0].message


def test_cmp001_missing_test_report_fails_closed() -> None:
    findings = check(Traceability, run_subject(trace_view([requirement()], None)))
    assert [f.severity for f in findings] == [Severity.HIGH]
    assert "no test report" in findings[0].message


def test_cmp001_requirement_without_criteria_is_covered_at_requirement_level() -> None:
    rv = trace_view([requirement(criteria=())], [covering("passed", "R-001")])
    assert check(Traceability, run_subject(rv)) == []
    assert check(
        Traceability,
        run_subject(trace_view([requirement(criteria=())], [covering("passed", "R-002")])),
    )


def test_cmp001_inspection_needs_evidence() -> None:
    reqs = [requirement(verification="inspection")]
    assert check(Traceability, run_subject(trace_view(reqs, [])))  # no evidence at all
    with_check = trace_view(reqs, [])
    with_check.add_artifact(
        "readiness_report",
        {"checks": [{"id": "R-001", "passed": True, "evidence": ["screenshot reviewed"]}]},
        producer="verify",
    )
    assert check(Traceability, run_subject(with_check)) == []
    empty_evidence = trace_view(reqs, [])
    empty_evidence.add_artifact(
        "readiness_report",
        {"checks": [{"id": "R-001", "passed": True, "evidence": ["  "]}]},
        producer="verify",
    )
    assert check(Traceability, run_subject(empty_evidence))
    failed_check = trace_view(reqs, [])
    failed_check.add_artifact(
        "readiness_report",
        {"checks": [{"id": "R-001", "passed": False, "evidence": ["x"]}]},
        producer="verify",
    )
    assert check(Traceability, run_subject(failed_check))


def test_cmp001_inspection_by_human_decision_but_not_by_an_agent() -> None:
    reqs = [requirement(verification="inspection")]
    human = trace_view(reqs, [])
    human.add_decision(DecisionKind.RELEASE, refs=["AC-001-1"])
    assert check(Traceability, run_subject(human)) == []
    agent = trace_view(reqs, [])
    agent.add_decision(
        DecisionKind.RELEASE, actor=Actor.agent("release-manager"), refs=["AC-001-1"]
    )
    assert check(Traceability, run_subject(agent))


def test_cmp001_staged_readiness_report_of_the_gated_attempt_counts() -> None:
    rv = trace_view([requirement(verification="inspection")], [])
    rv.add_artifact(
        "readiness_report",
        {"checks": [{"id": "R-001", "passed": True, "evidence": ["reviewed"]}]},
        producer="release-readiness",
        status=ArtifactStatus.STAGED,
    )
    assert check(Traceability, run_subject(rv)) == []


@pytest.mark.parametrize("status", ["deferred", "descoped"])
def test_cmp001_deferral_needs_a_human_scope_decision(status: str) -> None:
    reqs = [requirement(status=status, status_decision="D-0007")]
    nobody = trace_view(reqs, [])
    assert check(Traceability, run_subject(nobody))
    human = trace_view(reqs, [])
    human.add_decision(DecisionKind.SCOPE, decision_id="D-0007")
    assert check(Traceability, run_subject(human)) == []
    agent = trace_view(reqs, [])
    agent.add_decision(DecisionKind.SCOPE, actor=Actor.agent("requirements"), decision_id="D-0007")
    assert check(Traceability, run_subject(agent))
    wrong_kind = trace_view(reqs, [])
    wrong_kind.add_decision(DecisionKind.DESIGN, decision_id="D-0007")
    assert check(Traceability, run_subject(wrong_kind))


def test_cmp001_deferral_without_recorded_decision_id_falls_back_to_refs() -> None:
    reqs = [requirement(status="deferred", status_decision=None)]
    rv = trace_view(reqs, [])
    assert check(Traceability, run_subject(rv))
    rv.add_decision(DecisionKind.SCOPE, refs=["R-001"])
    assert check(Traceability, run_subject(rv)) == []


def test_cmp001_a_wrong_decision_id_is_not_rescued_by_other_decisions() -> None:
    rv = trace_view([requirement(status="deferred", status_decision="D-9999")], [])
    rv.add_decision(DecisionKind.SCOPE, refs=["R-001"], decision_id="D-0001")
    assert check(Traceability, run_subject(rv))


def test_cmp001_unknown_status_and_missing_requirements_fail_closed() -> None:
    assert check(Traceability, run_subject(trace_view([requirement(status="weird")], [])))
    assert check(Traceability, run_subject(FakeRunView(release_chain())))
    broken = FakeRunView(release_chain())
    broken.add_artifact("requirements", {"requirements": "nope"}, producer="requirements")
    assert [f.severity for f in check(Traceability, run_subject(broken))] == [Severity.HIGH]
    unreadable = FakeRunView(release_chain())
    unreadable.add_artifact("requirements", ValueError("bad"), producer="requirements")
    assert check(Traceability, run_subject(unreadable))


def test_cmp001_evidence_from_non_ancestors_does_not_count() -> None:
    rv = FakeRunView([*release_chain(), node("docs", depends_on=("requirements",))])
    rv.add_artifact("requirements", {"requirements": [requirement()]}, producer="requirements")
    rv.add_artifact("test_report", {"tests": [covering("passed", "AC-001-1")]}, producer="docs")
    assert check(Traceability, run_subject(rv))


def test_cmp001_tolerates_junk_entries_and_bounds_the_findings() -> None:
    reqs: list[Any] = [
        "junk",
        5,
        None,
        {"id": "R-9", "status": "active", "acceptance_criteria": "nope"},
    ]
    reqs.extend(requirement(f"R-{i:03d}", criteria=(f"AC-{i:03d}-1",)) for i in range(1, 60))
    junk_tests: list[Any] = [{"covers": "oops"}, "junk", {"outcome": "passed", "covers": [1, None]}]
    rv = trace_view(reqs, junk_tests)
    findings = check(Traceability, run_subject(rv))
    assert 1 <= len(findings) <= 25


# ===============================================================================================
# CMP-002


@pytest.mark.parametrize(
    "line",
    [
        "logger.info('request from %s', request.client.host)",
        "logger.warning(f'ip={request.client.host}')",
        "log.debug(request.headers.get('user-agent'))",
        "logging.info(request.headers['X-Forwarded-For'])",
        "print(request.remote_addr)",
        "self.logger.error('x', extra={'ip': request.remote_addr})",
        "logger.info('ua %s', user_agent)",
        "logger.info('ip %s', client_ip)",
        "cur.execute('insert into clicks (code, note) values (?, ?)', (code, request.client.host))",
        "cur.executemany('x', [request.headers.get('User-Agent')])",
    ],
)
def test_cmp002_flags_raw_personal_data_in_logs_and_persistence(line: str) -> None:
    source = f"def handler(request, cur, logger, log, user_agent, client_ip, code):\n    {line}\n"
    findings = check(PersonalDataHygiene, py_subject("api.py", source))
    assert findings, line
    assert {(f.rule_id, f.severity) for f in findings} == {("CMP-002", Severity.MEDIUM)}
    assert findings[0].location == "api.py:2"


@pytest.mark.parametrize(
    "line",
    [
        "logger.info('request from %s', hash_ip(request.client.host))",
        "logger.info('ip %s', hashlib.sha256(request.client.host.encode()).hexdigest())",
        "logger.info('ip %s', truncate_ip(request.client.host))",
        "logger.info('ip %s', anonymise(request.remote_addr))",
        "logger.info('visitor %s', hmac_digest(request.headers.get('user-agent')))",
        "logger.info('path %s', request.url.path)",
        "logger.info('code %s', code)",
        "other.append(request.client.host)",
        "cur.execute('select * from t where code=?', (code,))",
    ],
)
def test_cmp002_hashed_or_unrelated_values_are_fine(line: str) -> None:
    source = f"def handler(request, cur, logger, other, code):\n    {line}\n"
    assert check(PersonalDataHygiene, py_subject("api.py", source)) == []


def test_cmp002_insert_and_alter_table_columns_named_like_personal_data() -> None:
    for col in ("ip", "ip_address", "client_ip", "remote_addr", "user_agent", "ua"):
        sql = f"insert into clicks (code, {col}) values (?, ?);\n"  # noqa: S608 - fixture text
        findings = check(
            PersonalDataHygiene, cs_subject([create("shortener/storage/migrations/0003.sql", sql)])
        )
        assert [f.evidence for f in findings] == [col], col
    alter = "ALTER TABLE clicks ADD COLUMN user_agent TEXT;\n"
    assert check(PersonalDataHygiene, cs_subject([create("m.sql", alter)]))
    quoted = 'INSERT INTO "clicks" ("code", "IP_Address") VALUES (?, ?);\n'
    assert check(PersonalDataHygiene, cs_subject([create("m.sql", quoted)]))


def test_cmp002_python_string_constants_are_checked_too() -> None:
    source = "SQL = 'INSERT INTO clicks (code, ip_address) VALUES (?, ?)'\n"
    findings = check(PersonalDataHygiene, py_subject("repo.py", source))
    assert [(f.evidence, f.location) for f in findings] == [("ip_address", "repo.py:1")]


def test_cmp002_hashed_columns_and_truncated_ip_are_fine() -> None:
    sql = (
        "insert into clicks (code, ip_prefix, visitor_hash, user_agent_family) values (?, ?, ?, "
        "?);\n"
    )
    assert check(PersonalDataHygiene, cs_subject([create("m.sql", sql)])) == []


def test_cmp002_ignores_prose_and_other_file_types() -> None:
    assert (
        check(
            PersonalDataHygiene,
            cs_subject([create("docs/privacy.md", "we never log request.client.host\n")]),
        )
        == []
    )
    assert (
        check(
            PersonalDataHygiene,
            cs_subject([create("notes.txt", "insert into t (ip) values (1)\n")]),
        )
        == []
    )


def test_cmp002_fails_closed_on_unparseable_python_and_oversize_sql() -> None:
    assert [f.severity for f in check(PersonalDataHygiene, py_subject("a.py", "def (:\n"))] == [
        Severity.HIGH
    ]
    long_line = "select " + "x" * 2500 + ";\n"
    assert [
        f.severity for f in check(PersonalDataHygiene, cs_subject([create("big.sql", long_line)]))
    ] == [Severity.HIGH]


# ===============================================================================================
# CMP-003


def approval_view(**spec_kwargs: Any) -> FakeRunView:
    specs = [
        node("requirements", produces=("requirements",), stage=Stage.REQUIREMENTS),
        node(
            "design",
            depends_on=("requirements",),
            produces=("design_doc",),
            stage=Stage.DESIGN,
            approval=ApprovalSpec(**spec_kwargs),
        ),
        node(
            "release-readiness",
            depends_on=("design",),
            produces=("readiness_report",),
            stage=Stage.RELEASE,
        ),
    ]
    return FakeRunView(specs)


def test_cmp003_valid_sufficient_current_approval_passes() -> None:
    rv = approval_view(always=True)
    rv.add_approval(approval("design"))
    assert check(DecisionCompleteness, run_subject(rv)) == []


def test_cmp003_insufficient_distinct_votes() -> None:
    rv = approval_view(always=True)
    rv.add_approval(approval("design", voters=("alice", "alice"), required=2))
    findings = check(DecisionCompleteness, run_subject(rv))
    assert [(f.rule_id, f.severity) for f in findings] == [("CMP-003", Severity.HIGH)]
    assert "1 distinct approvals, 2 required" in findings[0].message


def test_cmp003_high_risk_needs_two_approvers() -> None:
    rv = approval_view(always=True)
    rv.add_approval(approval("design", risk=RiskLevel.HIGH, required=1))
    assert any(
        "two distinct approvers" in f.message for f in check(DecisionCompleteness, run_subject(rv))
    )
    ok = approval_view(always=True)
    ok.add_approval(approval("design", risk=RiskLevel.HIGH, required=2, voters=("alice", "bob")))
    assert check(DecisionCompleteness, run_subject(ok)) == []


@pytest.mark.parametrize(
    "status", [ApprovalStatus.PENDING, ApprovalStatus.REJECTED, ApprovalStatus.INVALIDATED]
)
def test_cmp003_latest_request_must_be_approved(status: ApprovalStatus) -> None:
    rv = approval_view(always=True)
    rv.add_approval(approval("design", status=status, voters=()))
    assert check(DecisionCompleteness, run_subject(rv))


def test_cmp003_stale_approval_superseded_by_a_newer_request() -> None:
    rv = approval_view(always=True)
    rv.add_approval(approval("design", request_id="A-0001"))
    rv.add_approval(
        approval(
            "design",
            request_id="A-0002",
            status=ApprovalStatus.PENDING,
            voters=(),
            created_offset=5,
        )
    )
    findings = check(DecisionCompleteness, run_subject(rv))
    assert any("stale or undecided" in f.message for f in findings)
    recovered = approval_view(always=True)
    recovered.add_approval(
        approval("design", request_id="A-0001", status=ApprovalStatus.INVALIDATED, voters=())
    )
    recovered.add_approval(approval("design", request_id="A-0002", created_offset=5))
    assert check(DecisionCompleteness, run_subject(recovered)) == []


def test_cmp003_cancelled_requests_are_ignored() -> None:
    rv = approval_view(always=True)
    rv.add_approval(approval("design"))
    rv.add_approval(
        approval(
            "design",
            request_id="A-0002",
            status=ApprovalStatus.CANCELLED,
            voters=(),
            created_offset=5,
        )
    )
    assert check(DecisionCompleteness, run_subject(rv)) == []


def test_cmp003_missing_approval_records() -> None:
    always = approval_view(always=True)
    assert check(DecisionCompleteness, run_subject(always))
    threshold = approval_view(always=False)
    assert check(
        DecisionCompleteness, run_subject(threshold)
    )  # no request and no automatic-approval decision
    threshold.add_decision(DecisionKind.APPROVAL, actor=Actor.policy("auto"), node_id="design")
    assert check(DecisionCompleteness, run_subject(threshold)) == []


def test_cmp003_fingerprint_and_evidence_binding() -> None:
    unbound = approval_view(always=True)
    unbound.add_approval(approval("design", fingerprint=""))
    assert any(
        "fingerprint" in f.message for f in check(DecisionCompleteness, run_subject(unbound))
    )
    mismatch = approval_view(always=True)
    mismatch.add_approval(approval("design", evidence_hash="ev-1", vote_evidence="ev-OTHER"))
    assert any(
        "does not cover" in f.message for f in check(DecisionCompleteness, run_subject(mismatch))
    )
    current = FakeRunView(
        approval_view(always=True)._specs.values(),
        params={"node_fingerprints": {"design": "fp-NEW"}},
    )
    current.add_approval(approval("design", fingerprint="fp-1"))
    assert any(
        "current fingerprint" in f.message
        for f in check(DecisionCompleteness, run_subject(current))
    )
    matching = FakeRunView(
        approval_view(always=True)._specs.values(), params={"node_fingerprints": {"design": "fp-1"}}
    )
    matching.add_approval(approval("design", fingerprint="fp-1"))
    assert check(DecisionCompleteness, run_subject(matching)) == []


def test_cmp003_only_ancestors_with_an_approval_spec_that_ran_are_considered() -> None:
    rv = approval_view(always=True)  # 'design' has no approval record, but is skipped
    rv._statuses["design"] = rv._statuses["design"].SKIPPED
    assert check(DecisionCompleteness, run_subject(rv)) == []
    # the gated node's own approval is not yet granted when the gate runs
    gated = FakeRunView([node("release-readiness", approval=ApprovalSpec(always=True))])
    assert check(DecisionCompleteness, run_subject(gated)) == []
    # a node that is not an ancestor of the gate is out of scope
    sibling = FakeRunView(
        [
            *approval_view(always=True)._specs.values(),
            node("docs", approval=ApprovalSpec(always=True)),
        ]
    )
    sibling.add_approval(approval("design"))
    assert check(DecisionCompleteness, run_subject(sibling)) == []


def test_cmp003_waivers_need_a_decision_an_approved_request_and_a_waivable_rule() -> None:
    rv = approval_view(always=True)
    rv.add_approval(approval("design"))
    finding = make_finding("SEC-006", Severity.MEDIUM)
    rv.add_waiver(waiver_for(finding, node_id="design", evidence="e"))
    messages = [f.message for f in check(DecisionCompleteness, run_subject(rv))]
    assert any("no WAIVER decision" in m for m in messages)  # D-0001 does not exist yet
    rv.add_decision(DecisionKind.WAIVER, decision_id="D-0001")
    assert check(DecisionCompleteness, run_subject(rv)) == []


def test_cmp003_waiver_problems_are_each_reported() -> None:
    rv = approval_view(always=True)
    rv.add_approval(approval("design", request_id="A-0002"))
    rv.add_decision(DecisionKind.WAIVER, decision_id="D-0001")
    high_rule = make_finding("CHG-001", Severity.HIGH)
    unknown_rule = make_finding("XXX-001", Severity.MEDIUM)
    rv.add_waiver(waiver_for(high_rule, node_id="design", evidence="e"))
    rv.add_waiver(waiver_for(unknown_rule, node_id="design", evidence="e"))
    messages = " | ".join(f.message for f in check(DecisionCompleteness, run_subject(rv)))
    assert (
        "not backed by an approved request" in messages
    )  # approval A-0001 on the waiver does not exist
    assert "may not be waived" in messages
