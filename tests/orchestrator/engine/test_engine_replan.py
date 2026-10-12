"""Fingerprints (engine-semantics 4.2), invalidation table (12.2), remediation targets (9.2)."""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.artifacts import ArtifactStore, PinnedReader
from orchestrator.core.clock import FakeClock
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.state import NodeState
from orchestrator.core.types import (
    EXTERNAL_PRODUCER,
    ArtifactDraft,
    Epoch,
    NodeSpec,
    NodeStatus,
    PlanExpansion,
    RemediationSpec,
    Rewire,
    Stage,
    assert_transition,
)
from orchestrator.engine.replan import (
    Transition,
    compute_fingerprint,
    invalidate,
    resolve_remediation_targets,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Durability is not under test and fsync dominates the run time on Windows."""
    monkeypatch.setattr(os, "fsync", lambda fd: None)


def spec(node_id: str, deps: Sequence[str] = (), **kwargs: Any) -> NodeSpec:
    fields: dict[str, Any] = {"title": node_id, "stage": Stage.IMPLEMENTATION, "agent": "agent"}
    fields.update(kwargs)
    return NodeSpec(id=node_id, depends_on=tuple(deps), **fields)


# =============================================================================
# fingerprints
# =============================================================================


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", clock=FakeClock())


def publish(
    store: ArtifactStore,
    kind: str,
    content: Any,
    *,
    key: str = "",
    producer: str = "a",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    attempt = 0 if producer == EXTERNAL_PRODUCER else 1
    draft = ArtifactDraft(kind, content, key=key, metadata=dict(metadata or {}))
    store.put(draft, producer=producer, attempt=attempt)


def snapshot(store: ArtifactStore, *kinds: str) -> PinnedReader:
    return store.snapshot(kinds, lambda producer: True)


def fingerprint_graph() -> WorkflowGraph:
    return WorkflowGraph(
        [
            spec("a", produces=("doc", "extra", "changeset")),
            spec(
                "b",
                ["a"],
                consumes=("doc",),
                optional_consumes=("extra",),
                fallback_agent="agent-fb",
                produces=("out",),
            ),
        ]
    )


BASE: dict[str, Any] = {
    "agent_version": "1",
    "epoch": Epoch.PRIMARY,
    "baseline_hash": "baseline-1",
    "policy_hash": "policy-1",
    "feedback_keys": (),
    "engine_version": 1,
}


def fingerprint(
    pinned: PinnedReader,
    *,
    graph: WorkflowGraph | None = None,
    node: NodeSpec | None = None,
    **overrides: Any,
) -> str:
    graph = graph or fingerprint_graph()
    return compute_fingerprint(node or graph.node("b"), graph, pinned, **{**BASE, **overrides})


@pytest.fixture
def pinned(store: ArtifactStore) -> PinnedReader:
    publish(store, "doc", {"v": 1})
    return snapshot(store, "doc", "extra")


def test_fingerprint_is_a_stable_sha256(pinned: PinnedReader) -> None:
    first, second = fingerprint(pinned), fingerprint(pinned)
    assert first == second
    assert len(first) == 64 and set(first) <= set("0123456789abcdef")


def test_fingerprint_follows_the_documented_formula(pinned: PinnedReader) -> None:
    graph = fingerprint_graph()
    node = graph.node("b")
    (ref,) = pinned.refs()
    expected = hash_jsonable(
        {
            "spec": node.spec_hash(),
            "agent": ["agent", "1"],
            "epoch": "primary",
            "deps": ["a"],
            "inputs": [["doc", "", ref.content_hash], ["extra", "", None]],
            "baseline": "baseline-1",
            "policy": "policy-1",
            "feedback": "",
            "engine": 1,
        }
    )
    assert fingerprint(pinned, graph=graph) == expected
    with_feedback = fingerprint(pinned, graph=graph, feedback_keys=["k2", "k1", "k2"])
    assert with_feedback != expected
    expected_feedback = hash_jsonable(["k1", "k2"])
    assert hash_jsonable(
        {**_documented(graph, ref.content_hash), "feedback": expected_feedback}
    ) == (with_feedback)


def _documented(graph: WorkflowGraph, content_hash: str) -> dict[str, Any]:
    node = graph.node("b")
    return {
        "spec": node.spec_hash(),
        "agent": ["agent", "1"],
        "epoch": "primary",
        "deps": ["a"],
        "inputs": [["doc", "", content_hash], ["extra", "", None]],
        "baseline": "baseline-1",
        "policy": "policy-1",
        "engine": 1,
    }


@pytest.mark.parametrize(
    "override",
    [
        {"agent_version": "2"},
        {"epoch": Epoch.FALLBACK},
        {"baseline_hash": "baseline-2"},
        {"policy_hash": "policy-2"},
        {"feedback_keys": ["finding-1"]},
        {"engine_version": 2},
    ],
    ids=lambda o: next(iter(o)),
)
def test_fingerprint_is_sensitive_to_every_governance_input(
    pinned: PinnedReader, override: dict[str, Any]
) -> None:
    assert fingerprint(pinned, **override) != fingerprint(pinned)


def test_fingerprint_is_sensitive_to_the_node_spec(pinned: PinnedReader) -> None:
    node = fingerprint_graph().node("b")
    base = fingerprint(pinned)
    assert fingerprint(pinned, node=dataclasses.replace(node, title="renamed")) != base
    assert fingerprint(pinned, node=dataclasses.replace(node, params={"x": 1})) != base
    assert fingerprint(pinned, node=dataclasses.replace(node, timeout_s=12.0)) != base


def test_fingerprint_uses_the_agent_of_the_epoch(pinned: PinnedReader) -> None:
    graph = fingerprint_graph()
    node = graph.node("b")
    fallback = fingerprint(pinned, graph=graph, epoch=Epoch.FALLBACK)
    # the same fallback agent id under the same spec: only the epoch distinguishes them
    assert fallback != fingerprint(pinned, graph=graph, epoch=Epoch.PRIMARY)
    # a node without a fallback agent keeps its own agent in the FALLBACK epoch (no crash)
    plain = dataclasses.replace(node, fallback_agent=None)
    assert fingerprint(pinned, graph=graph, node=plain, epoch=Epoch.FALLBACK)


def test_fingerprint_is_sensitive_to_effective_dependencies_including_rewires(
    pinned: PinnedReader,
) -> None:
    base = fingerprint(pinned)
    rewired = fingerprint_graph()
    rewired.apply_expansion(
        PlanExpansion("a", (spec("n", ["a"]),), {"b": Rewire(add_depends_on=("n",))})
    )
    assert rewired.deps("b") == ("a", "n")
    assert fingerprint(pinned, graph=rewired) != base


def test_fingerprint_is_sensitive_to_pinned_content_and_key_sets(store: ArtifactStore) -> None:
    publish(store, "doc", {"v": 1})
    before = fingerprint(snapshot(store, "doc"))
    publish(store, "doc", {"v": 2})
    assert fingerprint(snapshot(store, "doc")) != before
    publish(store, "changeset", {"ops": []}, key="x")
    one_key = fingerprint(snapshot(store, "doc", "changeset"))
    publish(store, "changeset", {"ops": []}, key="y")
    assert fingerprint(snapshot(store, "doc", "changeset")) != one_key


def test_fingerprint_ignores_artifact_metadata_ids_and_versions(store: ArtifactStore) -> None:
    publish(store, "doc", {"v": 1}, metadata={"note": "first"})
    first = fingerprint(snapshot(store, "doc"))
    publish(store, "doc", {"v": 1}, metadata={"note": "second", "attempt_ms": 99})  # version 2
    assert store.latest("doc").version == 2  # type: ignore[union-attr]
    assert fingerprint(snapshot(store, "doc")) == first


def test_absent_kinds_are_recorded_as_absent(store: ArtifactStore) -> None:
    publish(store, "doc", {"v": 1})
    declared_absent = fingerprint(snapshot(store, "doc", "extra"))
    assert declared_absent != fingerprint(snapshot(store, "doc"))
    publish(store, "extra", {"e": 1})
    assert fingerprint(snapshot(store, "doc", "extra")) != declared_absent


def test_fingerprint_ignores_feedback_key_order_and_repetition(pinned: PinnedReader) -> None:
    a = fingerprint(pinned, feedback_keys=["k1", "k2"])
    assert a == fingerprint(pinned, feedback_keys=("k2", "k1", "k1"))
    assert a != fingerprint(pinned, feedback_keys=["k1"])


def test_fingerprint_of_a_node_unknown_to_the_graph_is_an_error(pinned: PinnedReader) -> None:
    from orchestrator.core.errors import GraphError

    stranger = spec("stranger")
    with pytest.raises(GraphError):
        fingerprint(pinned, node=stranger)


# =============================================================================
# transitions
# =============================================================================


def test_transition_defaults_and_immutability() -> None:
    t = Transition("n", NodeStatus.PENDING, "why")
    assert (t.bump_epoch, t.cancel_running, t.quarantine_staged, t.cancel_approval) == (
        False,
        False,
        False,
        False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.reason = "other"  # type: ignore[misc]


# =============================================================================
# invalidation
# =============================================================================


def pair_graph() -> WorkflowGraph:
    return WorkflowGraph([spec("p", produces=("k",)), spec("c", ["p"], consumes=("k",))])


def states(**statuses: NodeStatus) -> dict[str, NodeState]:
    return {
        name.replace("__", "."): NodeState(name.replace("__", "."), s)
        for name, s in statuses.items()
    }


def only(transitions: list[Transition]) -> Transition:
    assert len(transitions) == 1
    return transitions[0]


def flags(t: Transition) -> tuple[NodeStatus, bool, bool, bool, bool]:
    return (t.to, t.bump_epoch, t.cancel_running, t.quarantine_staged, t.cancel_approval)


# (to, bump_epoch, cancel_running, quarantine_staged, cancel_approval); None = unchanged
TABLE: dict[NodeStatus, tuple[NodeStatus, bool, bool, bool, bool] | None] = {
    NodeStatus.SUCCEEDED: (NodeStatus.STALE, False, False, False, False),
    NodeStatus.SKIPPED: (NodeStatus.PENDING, False, False, False, False),
    NodeStatus.READY: (NodeStatus.PENDING, False, False, False, False),
    NodeStatus.RUNNING: (NodeStatus.PENDING, True, True, True, False),
    NodeStatus.AWAITING_APPROVAL: (NodeStatus.PENDING, True, False, True, True),
    NodeStatus.AWAITING_INPUT: (NodeStatus.PENDING, False, False, True, False),
    NodeStatus.FAILED: (NodeStatus.PENDING, True, False, False, False),
    NodeStatus.BLOCKED: (NodeStatus.PENDING, True, False, False, False),
    NodeStatus.CANCELLED: (NodeStatus.PENDING, True, False, False, False),
    NodeStatus.ROLLED_BACK: (NodeStatus.PENDING, True, False, False, False),
    NodeStatus.PENDING: None,
    NodeStatus.STALE: None,
    NodeStatus.REMOVED: None,
}


def test_the_table_covers_every_node_status() -> None:
    assert set(TABLE) == set(NodeStatus)


@pytest.mark.parametrize("status", list(NodeStatus))
def test_invalidation_table_row(status: NodeStatus) -> None:
    graph = pair_graph()
    result = invalidate(
        graph, {"p": NodeState("p", NodeStatus.SUCCEEDED), "c": NodeState("c", status)}, ["k"]
    )
    expected = TABLE[status]
    if expected is None:
        assert result == []
        return
    transition = only(result)
    assert transition.node_id == "c" and flags(transition) == expected
    assert "input kind k" in transition.reason
    assert_transition(status, transition.to)  # every intent is legal for the scheduler


def test_the_changed_producer_itself_is_not_invalidated() -> None:
    graph = pair_graph()
    result = invalidate(graph, states(p=NodeStatus.SUCCEEDED, c=NodeStatus.SUCCEEDED), ["k"], ["p"])
    assert [t.node_id for t in result] == ["c"]


def test_awaiting_input_wakes_to_ready_only_when_the_answers_kind_changed() -> None:
    graph = pair_graph()
    nodes = states(p=NodeStatus.SUCCEEDED, c=NodeStatus.AWAITING_INPUT)
    woken = only(invalidate(graph, nodes, ["k"], wake_kind="k"))
    assert flags(woken) == (NodeStatus.READY, False, False, True, False)
    assert_transition(NodeStatus.AWAITING_INPUT, woken.to)
    other = only(invalidate(graph, nodes, ["k"], wake_kind="clarification_answers"))
    assert other.to == NodeStatus.PENDING  # some other input changed: re-evaluate from scratch
    assert only(invalidate(graph, nodes, ["k"])).to == NodeStatus.PENDING
    by_producer = only(invalidate(graph, nodes, [], ["p"], wake_kind="k"))
    assert by_producer.to == NodeStatus.PENDING  # the awaited kind did not change


def pipeline() -> WorkflowGraph:
    graph = WorkflowGraph(
        [
            spec("intake", consumes=("request",), produces=("intake_report",)),
            spec(
                "requirements",
                ["intake"],
                consumes=("request", "intake_report"),
                optional_consumes=("clarification_answers",),
                produces=("requirements",),
            ),
            spec("design", ["requirements"], consumes=("requirements",), produces=("design_doc",)),
            spec(
                "decompose",
                ["design"],
                consumes=("requirements", "design_doc"),
                produces=("task_plan",),
            ),
            spec("implement.a", ["decompose"], consumes=("task_plan",), produces=("changeset",)),
            spec(
                "integrate",
                ["decompose", "implement.a"],
                consumes=("changeset", "task_plan"),
                produces=("integrated_changeset",),
            ),
            spec(
                "verify",
                ["integrate"],
                consumes=("integrated_changeset",),
                overlay_from=("integrated_changeset",),
                produces=("test_report",),
                mandatory=True,
            ),
        ]
    )
    graph.validate()
    return graph


def all_succeeded(graph: WorkflowGraph) -> dict[str, NodeState]:
    return {n: NodeState(n, NodeStatus.SUCCEEDED) for n in graph.nodes}


def affected(transitions: Iterable[Transition]) -> list[str]:
    return [t.node_id for t in transitions]


def test_affected_nodes_are_the_direct_consumers_of_the_changed_kinds() -> None:
    graph = pipeline()
    nodes = all_succeeded(graph)
    # an amended request: only its direct readers go stale; they cascade only if they republish
    assert affected(invalidate(graph, nodes, ["request"])) == ["intake", "requirements"]
    assert affected(invalidate(graph, nodes, ["changeset"])) == ["integrate"]
    assert affected(invalidate(graph, nodes, ["request", "changeset"])) == [
        "intake",
        "requirements",
        "integrate",
    ]


def test_optional_overlay_and_skip_inputs_count_as_reads() -> None:
    graph = WorkflowGraph(
        [
            spec("p", produces=("k1", "k2", "changeset")),
            spec(
                "x",
                ["p"],
                optional_consumes=("k1",),
                skip_inputs=("k2",),
                overlay_from=("changeset",),
                skip_if="never",
            ),
        ]
    )
    nodes = {"p": NodeState("p", NodeStatus.SUCCEEDED), "x": NodeState("x", NodeStatus.SKIPPED)}
    for kind in ("k1", "k2", "changeset"):
        assert only(invalidate(graph, nodes, [kind])).node_id == "x"
    assert only(invalidate(graph, nodes, ["k2"])).to == NodeStatus.PENDING  # skip_if re-evaluated


def test_affected_nodes_include_all_descendants_of_a_changed_producer() -> None:
    graph = pipeline()
    nodes = all_succeeded(graph)
    result = invalidate(graph, nodes, [], ["design"])
    assert affected(result) == ["decompose", "implement.a", "integrate", "verify"]
    assert {t.to for t in result} == {NodeStatus.STALE}


def test_kinds_and_producers_add_up_without_duplicates() -> None:
    graph = pipeline()
    result = invalidate(graph, all_succeeded(graph), ["task_plan"], ["design"])
    assert affected(result) == ["decompose", "implement.a", "integrate", "verify"]
    assert len(set(affected(result))) == len(result)


def test_unknown_kinds_producers_and_nodes_without_state_are_ignored() -> None:
    graph = pipeline()
    nodes = all_succeeded(graph)
    del nodes["verify"]  # an expansion node that has no state yet is simply skipped
    result = invalidate(graph, nodes, ["no_such_kind", 7], ["ghost", "external", None])  # type: ignore[list-item]
    assert result == []
    assert "verify" not in affected(invalidate(graph, nodes, [], ["design"]))


def test_cancelling_intents_come_first_then_topological_order() -> None:
    graph = pipeline()
    nodes = all_succeeded(graph)
    nodes["decompose"].status = NodeStatus.RUNNING
    nodes["integrate"].status = NodeStatus.PENDING  # nothing to do
    nodes["verify"].status = NodeStatus.FAILED
    result = invalidate(graph, nodes, [], ["requirements"])
    assert affected(result) == ["decompose", "design", "implement.a", "verify"]
    assert result[0].cancel_running and not any(t.cancel_running for t in result[1:])


def test_invalidate_is_pure_and_deterministic() -> None:
    graph = pipeline()
    nodes = all_succeeded(graph)
    nodes["design"].status = NodeStatus.RUNNING
    graph_before, nodes_before = copy.deepcopy(graph.to_dict()), copy.deepcopy(nodes)
    first = invalidate(graph, nodes, ["request"], ["requirements"])
    second = invalidate(graph, nodes, ["request"], ["requirements"])
    assert first == second and first is not second
    assert graph.to_dict() == graph_before
    assert nodes == nodes_before  # no status, epoch or generation was touched


def test_generators_are_accepted_and_consumed_once() -> None:
    graph = pipeline()
    nodes = all_succeeded(graph)
    result = invalidate(graph, nodes, (k for k in ["changeset"]), (p for p in ["design"]))
    assert affected(result) == ["decompose", "implement.a", "integrate", "verify"]


def test_reasons_name_the_cause_and_stay_short() -> None:
    graph = WorkflowGraph(
        [
            spec("p", produces=("k1", "k2", "k3", "k4", "k5")),
            spec("c", ["p"], consumes=("k1", "k2", "k3", "k4", "k5")),
        ]
    )
    nodes = states(p=NodeStatus.SUCCEEDED, c=NodeStatus.SUCCEEDED)
    reason = only(invalidate(graph, nodes, ["k1", "k2", "k3", "k4", "k5"])).reason
    assert reason.startswith("stale: input kind k1, input kind k2, input kind k3")
    assert reason.endswith("(+2 more)") and len(reason) < 200
    by_node = only(invalidate(graph, nodes, [], ["p"])).reason
    assert "upstream node p" in by_node


def test_every_produced_transition_is_legal_on_a_mixed_graph() -> None:
    graph = pipeline()
    cycle = list(NodeStatus)
    nodes = {n: NodeState(n, cycle[i % len(cycle)]) for i, n in enumerate(graph.nodes)}
    for transition in invalidate(graph, nodes, ["request"], ["requirements"], wake_kind="request"):
        assert_transition(nodes[transition.node_id].status, transition.to)


# =============================================================================
# remediation targets
# =============================================================================


def remediation_graph() -> WorkflowGraph:
    return WorkflowGraph(
        [
            spec("decompose"),
            spec("author-tests", ["decompose"]),
            spec("implement.b", ["decompose"]),
            spec("implement.a", ["decompose"]),
            spec("verify", ["author-tests", "implement.a", "implement.b"]),
        ]
    )


def test_remediation_targets_resolve_ids_and_prefix_patterns_against_the_live_graph() -> None:
    graph = remediation_graph()
    spec_ = RemediationSpec(route_to=("implement.*", "author-tests"))
    assert resolve_remediation_targets(graph, spec_) == [
        "implement.b",
        "implement.a",
        "author-tests",
    ]
    graph.remove_node("verify")
    graph.remove_node("implement.b")  # dynamic nodes come and go: patterns follow the live graph
    assert resolve_remediation_targets(graph, spec_) == ["implement.a", "author-tests"]


def test_remediation_targets_are_deduplicated_in_route_order() -> None:
    graph = remediation_graph()
    spec_ = RemediationSpec(route_to=("implement.a", "implement.*", "implement.a"))
    assert resolve_remediation_targets(graph, spec_) == ["implement.a", "implement.b"]


def test_unmatched_and_malformed_targets_resolve_to_nothing() -> None:
    graph = remediation_graph()
    assert resolve_remediation_targets(graph, RemediationSpec(route_to=("docs.*", "ghost"))) == []
    garbage = RemediationSpec(route_to=("*", "a*b", "implement.*"))  # bare / mid wildcards
    assert resolve_remediation_targets(graph, garbage) == ["implement.b", "implement.a"]
