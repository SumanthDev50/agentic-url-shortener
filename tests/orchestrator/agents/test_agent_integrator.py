"""Tests for the integrator agent: merge all change sets against the baseline, report conflicts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.integrator import IntegratorAgent, impact_delta
from orchestrator.core.errors import AgentError
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.paths import NODE_ID_RE
from orchestrator.core.schemas import Kind, validate_payload
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import (
    Agent,
    AgentResult,
    ChangeSet,
    EventType,
    FileOp,
    FileOpKind,
)
from orchestrator.tools.changeset import apply_changeset
from orchestrator.tools.sandbox import SandboxRunner
from orchestrator.tools.workspace import MemoryView, OverlayView, materialize
from tests.orchestrator.agents.support_impl_chain import (
    Rig,
    build_chain,
    changeset_of,
    content_of,
    drafts,
    evaluate,
    impact_report_payload,
    payload_of,
    product_files,
    real_capability,
    rule_ids,
)

NODE = "integrate"
BASE = {
    "pkg/app.py": "VALUE = 1\n",
    "pkg/util.py": "def helper() -> int:\n    return 1\n",
    "docs/readme.md": "# readme\n",
}


def integrate(rig: Rig, **kwargs: Any) -> AgentResult:
    return rig.run(IntegratorAgent(), "integrate", NODE, **kwargs)


def three_disjoint(rig: Rig) -> dict[str, ChangeSet]:
    return {
        "implement.fake-alpha": rig.publish_changeset(
            "implement.fake-alpha", {"pkg/alpha.py": "A = 1\n", "pkg/app.py": "VALUE = 2\n"}
        ),
        "implement.fake-beta": rig.publish_changeset(
            "implement.fake-beta", {"pkg/beta.py": "B = 1\n"}
        ),
        "author-tests": rig.publish_changeset(
            "author-tests", {"tests/test_alpha.py": "def test_a() -> None:\n    assert True\n"}
        ),
    }


# ----------------------------------------------------------------------------- the happy path


def test_merges_disjoint_change_sets_into_one_proposal(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    three_disjoint(rig)
    result = integrate(rig)
    merged = changeset_of(result, Kind.INTEGRATED_CHANGESET)
    assert content_of(merged) == {
        "pkg/alpha.py": "A = 1\n",
        "pkg/app.py": "VALUE = 2\n",
        "pkg/beta.py": "B = 1\n",
        "tests/test_alpha.py": "def test_a() -> None:\n    assert True\n",
    }
    assert {d.kind for d in result.artifacts} == {
        Kind.INTEGRATED_CHANGESET,
        Kind.INTEGRATION_REPORT,
    }
    assert (merged.author, merged.description) == ("integrator", "integrated 3 change set(s)")


def test_the_integrated_change_set_is_relative_to_the_baseline(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    three_disjoint(rig)
    merged = changeset_of(integrate(rig), Kind.INTEGRATED_CHANGESET)
    modify = next(op for op in merged.ops if op.path == "pkg/app.py")
    assert modify.kind is FileOpKind.MODIFY and modify.base_hash == sha256_text(BASE["pkg/app.py"])
    applied = apply_changeset(rig.baseline, merged)  # strict: hashes verified
    assert applied["pkg/beta.py"] == "B = 1\n"


def test_the_report_lists_sources_paths_stats_and_no_conflicts(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    three_disjoint(rig)
    report = payload_of(integrate(rig), Kind.INTEGRATION_REPORT)
    assert validate_payload(Kind.INTEGRATION_REPORT, report) == []
    assert [s["node_id"] for s in report["sources"]] == [
        "author-tests",
        "implement.fake-alpha",
        "implement.fake-beta",
    ]
    assert report["sources"][1] == {
        "node_id": "implement.fake-alpha",
        "artifact_id": "changeset.implement.fake-alpha@v1",
        "paths": ["pkg/alpha.py", "pkg/app.py"],
    }
    assert (
        report["merged_paths"] == sorted(report["merged_paths"])
        and len(report["merged_paths"]) == 4
    )
    assert report["conflicts"] == []
    assert report["stats"] == {
        "files_added": 3,
        "files_modified": 1,
        "files_deleted": 0,
        "lines_added": 5,
        "lines_removed": 1,
    }


def test_deletions_and_many_sources_merge(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    rig.publish_changeset("implement.fake-alpha", {"docs/readme.md": None})
    for index in range(12):
        rig.publish_changeset(f"implement.n{index:02d}", {f"gen/f{index:02d}.py": f"N = {index}\n"})
    result = integrate(rig)
    merged = changeset_of(result, Kind.INTEGRATED_CHANGESET)
    assert len(merged.ops) == 13
    assert next(o for o in merged.ops if o.path == "docs/readme.md").kind is FileOpKind.DELETE
    assert payload_of(result, Kind.INTEGRATION_REPORT)["stats"]["files_deleted"] == 1


def test_the_output_is_deterministic_across_runs_and_clocks(tmp_path: Path) -> None:
    hashes = []
    for index, start in enumerate(
        (datetime(2026, 1, 1, tzinfo=UTC), datetime(2032, 2, 2, tzinfo=UTC))
    ):
        rig = Rig(tmp_path / str(index), BASE, clock_start=start)
        three_disjoint(rig)
        hashes.append(hash_jsonable([d.content for d in integrate(rig).artifacts]))
    assert hashes[0] == hashes[1]


def test_each_baseline_path_is_read_from_the_toolbox_at_most_once(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    rig.publish_changeset("implement.a", {"pkg/app.py": "VALUE = 5\n", "pkg/new_a.py": "A = 1\n"})
    rig.publish_changeset("implement.b", {"pkg/app.py": "VALUE = 5\n", "pkg/new_b.py": "B = 1\n"})
    integrate(rig)
    targets = [e["target"] for e in rig.audit.of(EventType.TOOL_CALL) if e["tool"] == "read_text"]
    assert sorted(targets) == ["pkg/app.py", "pkg/new_a.py", "pkg/new_b.py"]  # each exactly once


# ----------------------------------------------------------------------------- conflicts


def test_identical_results_for_one_path_do_not_conflict(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    rig.publish_changeset("implement.a", {"pkg/shared.py": "S = 1\n", "pkg/a.py": "A = 1\n"})
    rig.publish_changeset("implement.b", {"pkg/shared.py": "S = 1\n", "pkg/b.py": "B = 1\n"})
    result = integrate(rig)
    assert payload_of(result, Kind.INTEGRATION_REPORT)["conflicts"] == []
    assert len(changeset_of(result, Kind.INTEGRATED_CHANGESET).ops) == 3


def test_conflicts_are_reported_and_the_clean_subset_is_still_produced(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    rig.publish_changeset("implement.a", {"pkg/shared.py": "S = 'a'\n", "pkg/a.py": "A = 1\n"})
    rig.publish_changeset("implement.b", {"pkg/shared.py": "S = 'b'\n", "pkg/b.py": "B = 1\n"})
    result = integrate(rig)
    report = payload_of(result, Kind.INTEGRATION_REPORT)
    assert report["conflicts"] == [
        {
            "path": "pkg/shared.py",
            "authors": ["implement.a", "implement.b"],  # node ids, not the agent id
            "reason": "path modified differently by more than one change set",
        }
    ]
    merged = changeset_of(result, Kind.INTEGRATED_CHANGESET)
    assert set(content_of(merged)) == {"pkg/a.py", "pkg/b.py"}
    assert report["merged_paths"] == ["pkg/a.py", "pkg/b.py"]


def test_a_change_set_made_against_another_baseline_is_a_conflict(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    stale_base = {**BASE, "pkg/app.py": "VALUE = 0\n"}
    rig.publish_changeset("implement.a", {"pkg/app.py": "VALUE = 9\n"}, base=stale_base)
    rig.publish_changeset("implement.b", {"pkg/ok.py": "OK = 1\n"})
    result = integrate(rig)
    (conflict,) = payload_of(result, Kind.INTEGRATION_REPORT)["conflicts"]
    assert conflict["path"] == "pkg/app.py" and conflict["authors"] == ["implement.a"]
    assert "hash" in conflict["reason"]
    assert set(content_of(changeset_of(result, Kind.INTEGRATED_CHANGESET))) == {"pkg/ok.py"}


def test_a_case_insensitive_collision_is_a_conflict(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    rig.publish_changeset("implement.a", {"Pkg/Thing.py": "X = 1\n"})
    rig.publish_changeset("implement.b", {"pkg/thing.py": "X = 2\n"})
    result = integrate(rig)
    (conflict,) = payload_of(result, Kind.INTEGRATION_REPORT)["conflicts"]
    assert conflict["authors"] == ["implement.a", "implement.b"]
    assert changeset_of(result, Kind.INTEGRATED_CHANGESET).ops == ()


# ----------------------------------------------------------------------------- failures


def test_no_change_set_at_all_is_fatal(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    with pytest.raises(AgentError, match="no change set to integrate") as err:
        integrate(rig)
    assert err.value.kind == "fatal" and not err.value.retryable


@pytest.mark.parametrize(
    "content",
    [
        {"ops": [{"path": "../escape.py", "kind": "create", "content": "x"}]},
        {"ops": [{"path": "a.py", "kind": "explode", "content": "x"}]},
        {"ops": "not a list"},
        ["not", "a", "mapping"],
    ],
)
def test_a_malformed_change_set_is_fatal(tmp_path: Path, content: Any) -> None:
    rig = Rig(tmp_path, BASE)
    rig.publish(Kind.CHANGESET, content, key="implement.a", producer="implement.a", validate=False)
    with pytest.raises(AgentError, match="is malformed") as err:
        integrate(rig)
    assert err.value.kind == "fatal" and not err.value.retryable


# ----------------------------------------------------------------------------- impact delta


def test_impact_delta_compares_existing_files_with_the_prediction() -> None:
    merged = ChangeSet(
        (
            FileOp("a.py", FileOpKind.MODIFY, "x\n", "h"),
            FileOp("d.py", FileOpKind.DELETE, None, "h"),
            FileOp("new.py", FileOpKind.CREATE, "n\n"),
        )
    )
    delta = impact_delta({"a.py", "b.py", "c.py"}, merged)
    assert delta == {
        "predicted_only": ["b.py", "c.py"],
        "actual_only": ["d.py"],
        "precision": 0.3333,
        "recall": 0.5,
    }


def test_the_report_carries_the_impact_delta_when_a_report_exists(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    three_disjoint(rig)
    rig.publish(Kind.IMPACT_REPORT, impact_report_payload(["pkg/app.py", "pkg/util.py"]))
    result = integrate(rig)
    delta = payload_of(result, Kind.INTEGRATION_REPORT)["impact_delta"]
    assert delta == {
        "predicted_only": ["pkg/util.py"],
        "actual_only": [],
        "precision": 0.5,
        "recall": 1.0,
    }
    assert drafts(result, Kind.INTEGRATION_REPORT)[0].metadata == {"impact_report_present": True}


def test_without_an_impact_report_nothing_is_predicted(tmp_path: Path) -> None:
    greenfield = Rig(tmp_path / "green")
    greenfield.publish_changeset("implement.a", {"pkg/a.py": "A = 1\n"})
    delta = payload_of(integrate(greenfield), Kind.INTEGRATION_REPORT)["impact_delta"]
    assert delta == {"predicted_only": [], "actual_only": [], "precision": 1.0, "recall": 1.0}
    brownfield = Rig(tmp_path / "brown", BASE)
    brownfield.publish_changeset("implement.a", {"pkg/app.py": "VALUE = 3\n"})
    result = integrate(brownfield)
    delta = payload_of(result, Kind.INTEGRATION_REPORT)["impact_delta"]
    assert delta["actual_only"] == ["pkg/app.py"] and delta["recall"] == 0.0
    assert delta["precision"] == 1.0
    assert drafts(result, Kind.INTEGRATION_REPORT)[0].metadata == {"impact_report_present": False}


def test_an_undeclared_impact_report_is_treated_as_absent(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    three_disjoint(rig)
    rig.publish(Kind.IMPACT_REPORT, impact_report_payload(["pkg/app.py"]))
    result = integrate(rig, undeclared=(Kind.IMPACT_REPORT,))
    delta = payload_of(result, Kind.INTEGRATION_REPORT)["impact_delta"]
    assert delta["predicted_only"] == [] and delta["recall"] == 0.0


# ----------------------------------------------------------------------------- governance


def test_the_integrated_change_set_passes_the_real_policy_with_its_sources(tmp_path: Path) -> None:
    rig = Rig(tmp_path, BASE)
    sources = list(three_disjoint(rig).values())
    agent = IntegratorAgent()
    ctx = rig.context(agent, "integrate", NODE)
    merged = changeset_of(agent.run(ctx), Kind.INTEGRATED_CHANGESET)
    profile = agent.autonomy.narrow(ctx.node.autonomy)
    report = evaluate(
        merged,
        rig.baseline,
        node_id=NODE,
        profile=profile,
        impact={"modules": [], "expected_paths": ["pkg/app.py"]},
        sources=sources,
    )
    assert not report.blocking and not report.needs_approval, report.findings
    assert not rule_ids(report) & {"SEC-005", "CHG-007"}


def test_identity_protocol_and_autonomy() -> None:
    agent = IntegratorAgent()
    assert isinstance(agent, Agent) and NODE_ID_RE.match(agent.agent_id)
    assert (agent.agent_id, agent.role, agent.version) == ("integrator", "integrator", "1")
    assert not agent.autonomy.can_execute and agent.autonomy.write_globs == ("**",)
    assert agent.autonomy.max_changed_files >= 500  # it merges everybody else's files


# ----------------------------------------------------------------------------- the real recipes


@pytest.mark.parametrize("name", ["linkcheck", "max_clicks"])
def test_the_change_sets_of_the_real_recipes_integrate_without_conflicts(
    tmp_path: Path, name: str
) -> None:
    capability = real_capability(name)
    chain = build_chain(tmp_path, capability, {} if name == "linkcheck" else product_files())
    assert chain.report["conflicts"] == []
    assert [s["node_id"] for s in chain.report["sources"]] == sorted(chain.sources)
    assert len(chain.sources) >= 4  # >= 3 implementer groups plus the author of the tests
    touched = set().union(*(cs.paths() for cs in chain.sources.values()))
    assert set(chain.merged.paths()) == touched == set(chain.report["merged_paths"])
    assert apply_changeset(chain.rig.baseline, chain.merged)  # strict: relative to the baseline
    report = evaluate(
        chain.merged,
        chain.rig.baseline,
        node_id=NODE,
        profile=IntegratorAgent.autonomy,
        sources=list(chain.sources.values()),
    )
    assert not rule_ids(report) & {"SEC-005", "CHG-007"}, report.findings


@pytest.mark.slow
def test_the_integrated_greenfield_chain_runs_green_in_the_sandbox(tmp_path: Path) -> None:
    chain = build_chain(tmp_path / "chain", real_capability("linkcheck"), {})
    runner = SandboxRunner(tmp_path / "sandbox")
    scratch = runner.new_scratch()
    try:
        materialize(OverlayView(MemoryView({}), [chain.merged]), scratch)
        result = runner.run(runner.pytest_argv(scratch, ["tests"]), cwd=scratch, timeout_s=180)
    finally:
        runner.discard(scratch)
    assert result.returncode == 0, result.stdout[-2000:]
    assert " passed" in result.stdout
