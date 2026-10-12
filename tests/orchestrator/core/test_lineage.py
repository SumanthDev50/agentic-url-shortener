"""LineageGraph: upstream/downstream/explain/reaches over artifact inputs and decision parents."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core import lineage as lineage_module
from orchestrator.core.artifacts import ArtifactStore
from orchestrator.core.decisions import DecisionLog
from orchestrator.core.errors import ConfigError, StateError
from orchestrator.core.lineage import ARTIFACT, DECISION, LineageGraph, LineageNode
from orchestrator.core.types import (
    EXTERNAL_PRODUCER,
    Actor,
    Artifact,
    ArtifactDraft,
    ArtifactRef,
    Decision,
    DecisionKind,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HUMAN = Actor.human("alice")
AGENT = Actor.agent("architect")


class StepClock:
    def __init__(self) -> None:
        self._t = T0
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            self._t += timedelta(seconds=1)
            return self._t

    def monotonic(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        return None


@dataclass
class World:
    store: ArtifactStore
    log: DecisionLog
    graph: LineageGraph


def make_world(root: Path) -> World:
    store = ArtifactStore(root / "artifacts", clock=StepClock())
    log = DecisionLog(root / "decisions.jsonl", run_id="run1", clock=StepClock())
    return World(store, log, LineageGraph(store, log))


def publish(
    world: World,
    kind: str,
    content: Any,
    *,
    producer: str,
    inputs: tuple[ArtifactRef, ...] = (),
    key: str = "",
) -> Artifact:
    staged = world.store.put(
        ArtifactDraft(kind, content, key=key), producer=producer, inputs=inputs, staged=True
    )
    return world.store.activate([staged.id])[0]


@dataclass
class Pipeline:
    world: World
    request: Artifact
    answers: Artifact
    requirements: Artifact
    design: Artifact
    patch: Artifact
    clarification: Decision
    assumption: Decision
    design_decision: Decision
    release: Decision


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    """request + answers -> requirements -> design_doc -> release_patch, plus their decisions."""
    world = make_world(tmp_path_factory.mktemp("lineage"))
    request = world.store.put(ArtifactDraft("request", "build it"), producer=EXTERNAL_PRODUCER)
    answers = world.store.put(
        ArtifactDraft("clarification_answers", {"schema": 1, "answers": {}}),
        producer=EXTERNAL_PRODUCER,
    )
    clarification = world.log.record(
        DecisionKind.CLARIFICATION,
        "TTL is 24h",
        actor=HUMAN,
        outputs=(answers.ref,),
        refs=("R-001",),
    )
    requirements = publish(
        world,
        "requirements",
        {"schema": 1, "r": 1},
        producer="requirements",
        inputs=(request.ref, answers.ref),
    )
    assumption = world.log.record(
        DecisionKind.ASSUMPTION,
        "assume 24h",
        actor=AGENT,
        node_id="requirements",
        inputs=(request.ref, answers.ref),
        outputs=(requirements.ref,),
        parents=(clarification.id,),
    )
    design = publish(
        world, "design_doc", {"schema": 1, "d": 1}, producer="design", inputs=(requirements.ref,)
    )
    design_decision = world.log.record(
        DecisionKind.DESIGN,
        "use SQLite",
        actor=AGENT,
        node_id="design",
        outputs=(design.ref,),
        parents=(assumption.id,),
        alternatives=("Postgres",),
    )
    patch = publish(world, "release_patch", "diff --git", producer="promote", inputs=(design.ref,))
    release = world.log.record(
        DecisionKind.RELEASE,
        "ship it",
        actor=HUMAN,
        node_id="promote",
        outputs=(patch.ref,),
        parents=(design_decision.id,),
    )
    return Pipeline(
        world,
        request,
        answers,
        requirements,
        design,
        patch,
        clarification,
        assumption,
        design_decision,
        release,
    )


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "fsync", lambda fd: None)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


def ids(nodes: list[LineageNode] | tuple[LineageNode, ...]) -> list[str]:
    return [n.id for n in nodes]


# =============================================================================
# upstream
# =============================================================================


class TestUpstream:
    def test_artifact_expands_into_inputs_and_the_decisions_that_list_it(
        self, pipeline: Pipeline
    ) -> None:
        root = pipeline.world.graph.upstream(pipeline.patch.id)
        assert (root.kind, root.id, root.relation) == (ARTIFACT, pipeline.patch.id, "root")
        assert [(c.kind, c.relation, c.id) for c in root.children] == [
            (ARTIFACT, "input", pipeline.design.id),
            (DECISION, "decision", pipeline.release.id),
        ]

    def test_the_walk_reaches_the_clarification_through_inputs_and_parents(
        self, pipeline: Pipeline
    ) -> None:
        root = pipeline.world.graph.upstream(pipeline.patch.id)
        reached = {n.id for n in root.walk()}
        assert {
            pipeline.design.id,
            pipeline.requirements.id,
            pipeline.request.id,
            pipeline.answers.id,
            pipeline.clarification.id,
            pipeline.assumption.id,
            pipeline.design_decision.id,
            pipeline.release.id,
        } <= reached

    def test_decision_parents_are_followed_recursively(self, world: World) -> None:
        d1 = world.log.record(DecisionKind.SCOPE, "one", actor=HUMAN)
        d2 = world.log.record(DecisionKind.REPLAN, "two", actor=AGENT, parents=(d1.id,))
        d3 = world.log.record(DecisionKind.DESIGN, "three", actor=AGENT, parents=(d2.id,))
        art = publish(world, "design_doc", {"n": 1}, producer="design")
        d4 = world.log.record(
            DecisionKind.RELEASE, "four", actor=HUMAN, outputs=(art.ref,), parents=(d3.id,)
        )
        node = world.graph.upstream(art.id).children[0]
        chain = []
        while node.children:
            chain.append(node.id)
            node = node.children[0]
        chain.append(node.id)
        assert chain == [d4.id, d3.id, d2.id, d1.id]

    def test_a_node_reached_twice_is_expanded_once(self, pipeline: Pipeline) -> None:
        nodes = list(pipeline.world.graph.upstream(pipeline.patch.id).walk())
        expanded = [n.id for n in nodes if n.note != "seen"]
        assert len(expanded) == len(set(expanded))  # nothing is expanded twice
        seen = [n for n in nodes if n.note == "seen"]
        assert seen and all(n.children == () for n in seen)

    def test_unknown_artifact_raises_state_error(self, pipeline: Pipeline) -> None:
        with pytest.raises(StateError, match="unknown artifact"):
            pipeline.world.graph.upstream("nope@v1")
        with pytest.raises(StateError):
            pipeline.world.graph.explain("nope@v1\x1b[31m")

    def test_staged_and_quarantined_artifacts_can_be_inspected(self, world: World) -> None:
        a = publish(world, "request", "x", producer=EXTERNAL_PRODUCER)
        staged = world.store.put(
            ArtifactDraft("design_doc", {"n": 1}), producer="design", inputs=(a.ref,), staged=True
        )
        node = world.graph.upstream(staged.id)
        assert ids(node.children) == [a.id]
        assert "[staged]" in node.label
        world.store.quarantine([staged.id], "rejected")
        assert "[quarantined]" in world.graph.upstream(staged.id).label

    def test_several_decisions_listing_one_artifact_are_all_shown_in_order(
        self, world: World
    ) -> None:
        art = publish(world, "design_doc", {"n": 1}, producer="design")
        d1 = world.log.record(DecisionKind.DESIGN, "first", actor=AGENT, outputs=(art.ref,))
        d2 = world.log.record(DecisionKind.APPROVAL, "second", actor=HUMAN, outputs=(art.ref,))
        assert ids(world.graph.upstream(art.id).children) == [d1.id, d2.id]

    def test_diamonds_are_shown_once_and_marked(self, world: World) -> None:
        base = publish(world, "request", "x", producer=EXTERNAL_PRODUCER)
        left = publish(world, "requirements", {"l": 1}, producer="requirements", inputs=(base.ref,))
        right = publish(world, "intake_report", {"r": 1}, producer="intake", inputs=(base.ref,))
        top = publish(
            world, "design_doc", {"t": 1}, producer="design", inputs=(left.ref, right.ref)
        )
        root = world.graph.upstream(top.id)
        base_nodes = [n for n in root.walk() if n.id == base.id]
        assert [n.note for n in base_nodes] == ["", "seen"]
        assert base_nodes[1].children == ()
        assert "already shown above" in world.graph.explain(top.id)

    def test_cyclic_references_terminate(self, world: World) -> None:
        # input refs are not required to exist at put time, so a tampered pipeline can form a cycle
        a_hash = hashlib.sha256(b'{\n  "a": 1\n}\n').hexdigest()  # the bytes put() will store
        a_ref = ArtifactRef("design_doc@v1", "design_doc", "", 1, a_hash)
        b = world.store.put(
            ArtifactDraft("requirements", {"b": 1}),
            producer="requirements",
            inputs=(a_ref,),
            staged=True,
        )
        a = world.store.put(
            ArtifactDraft("design_doc", {"a": 1}), producer="design", inputs=(b.ref,), staged=True
        )
        assert a.content_hash == a_hash
        nodes = list(world.graph.upstream(a.id).walk())
        assert [n.id for n in nodes] == [a.id, b.id, a.id]
        assert nodes[-1].note == "seen" and nodes[-1].children == ()

    def test_missing_inputs_and_hash_mismatches_are_flagged_not_fatal(self, world: World) -> None:
        real = publish(world, "request", "x", producer=EXTERNAL_PRODUCER)
        stale = ArtifactRef(real.id, real.kind, real.key, real.version, "f" * 64)
        ghost = ArtifactRef("task_plan@v9", "task_plan", "", 9, "e" * 64)
        art = publish(world, "design_doc", {"n": 1}, producer="design", inputs=(stale, ghost))
        children = world.graph.upstream(art.id).children
        assert [(c.id, c.note) for c in children] == [
            (real.id, "hash-mismatch"),
            ("task_plan@v9", "missing"),
        ]
        text = world.graph.explain(art.id)
        assert "content hash differs from the recorded reference" in text
        assert "missing from the store" in text

    def test_depth_and_size_limits_truncate_instead_of_exploding(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous: tuple[ArtifactRef, ...] = ()
        chain: list[Artifact] = []
        for n in range(6):
            art = publish(world, "design_doc", {"n": n}, producer="design", inputs=previous)
            chain.append(art)
            previous = (art.ref,)
        monkeypatch.setattr(lineage_module, "MAX_DEPTH", 2)
        root = world.graph.upstream(chain[-1].id)
        assert [n.id for n in root.walk()] == [chain[i].id for i in (5, 4, 3, 2)]
        assert list(root.walk())[-1].note == "truncated" and list(root.walk())[-1].children == ()
        assert "not expanded: lineage limit reached" in world.graph.explain(chain[-1].id)
        monkeypatch.setattr(lineage_module, "MAX_DEPTH", 64)
        monkeypatch.setattr(lineage_module, "MAX_NODES", 3)
        limited = list(world.graph.upstream(chain[-1].id).walk())
        assert len(limited) <= 4 and limited[-1].note == "truncated"

    def test_wide_fan_in_is_bounded(self, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        refs = tuple(
            publish(
                world, "changeset", {"n": n}, producer=f"implement.g{n}", key=f"implement.g{n}"
            ).ref
            for n in range(8)
        )
        top = publish(world, "integrated_changeset", {"t": 1}, producer="integrate", inputs=refs)
        monkeypatch.setattr(lineage_module, "MAX_NODES", 4)
        nodes = list(world.graph.upstream(top.id).walk())
        assert len(nodes) == 9  # every input is listed ...
        assert sum(1 for n in nodes if n.note == "truncated") >= 5  # ... but most are not expanded


# =============================================================================
# reaches
# =============================================================================


class TestReaches:
    def test_the_release_patch_reaches_the_clarification_decision(self, pipeline: Pipeline) -> None:
        graph = pipeline.world.graph
        assert graph.reaches(pipeline.patch.id, DecisionKind.CLARIFICATION) is True
        assert graph.reaches(pipeline.patch.id, "clarification") is True
        assert (
            graph.reaches(pipeline.patch.id, DecisionKind.RELEASE) is True
        )  # decisions listing it directly
        assert graph.reaches(pipeline.patch.id, DecisionKind.DESIGN) is True

    def test_it_does_not_reach_decisions_that_are_not_upstream(self, pipeline: Pipeline) -> None:
        graph = pipeline.world.graph
        assert (
            graph.reaches(pipeline.request.id, DecisionKind.CLARIFICATION) is False
        )  # nothing upstream of a root input
        assert (
            graph.reaches(pipeline.design.id, DecisionKind.RELEASE) is False
        )  # downstream, not upstream
        assert graph.reaches(pipeline.patch.id, DecisionKind.SAFE_STOP) is False

    def test_it_follows_decision_parents_even_without_artifact_inputs(self, world: World) -> None:
        root_decision = world.log.record(DecisionKind.SCOPE, "scope", actor=HUMAN)
        middle = world.log.record(
            DecisionKind.REPLAN, "replan", actor=AGENT, parents=(root_decision.id,)
        )
        art = publish(world, "design_doc", {"n": 1}, producer="design")
        world.log.record(
            DecisionKind.DESIGN, "design", actor=AGENT, outputs=(art.ref,), parents=(middle.id,)
        )
        assert world.graph.reaches(art.id, DecisionKind.SCOPE) is True

    def test_invalid_input_is_rejected(self, pipeline: Pipeline) -> None:
        with pytest.raises(ConfigError, match="unknown decision kind"):
            pipeline.world.graph.reaches(pipeline.patch.id, "nonsense")
        with pytest.raises(StateError):
            pipeline.world.graph.reaches("nope@v1", DecisionKind.DESIGN)

    def test_it_terminates_on_cycles(self, world: World) -> None:
        a_ref = ArtifactRef("design_doc@v1", "design_doc", "", 1, "0" * 64)
        b = world.store.put(
            ArtifactDraft("requirements", {"b": 1}),
            producer="requirements",
            inputs=(a_ref,),
            staged=True,
        )
        a = world.store.put(
            ArtifactDraft("design_doc", {"a": 1}), producer="design", inputs=(b.ref,), staged=True
        )
        assert world.graph.reaches(a.id, DecisionKind.SAFE_STOP) is False


# =============================================================================
# downstream
# =============================================================================


class TestDownstream:
    def test_everything_derived_from_an_artifact_is_listed_sorted(self, pipeline: Pipeline) -> None:
        down = pipeline.world.graph.downstream(pipeline.request.id)
        assert [a.id for a in down] == [
            pipeline.design.id,
            pipeline.patch.id,
            pipeline.requirements.id,
        ]
        assert [a.id for a in pipeline.world.graph.downstream(pipeline.design.id)] == [
            pipeline.patch.id
        ]
        assert pipeline.world.graph.downstream(pipeline.patch.id) == []

    def test_staged_and_quarantined_artifacts_are_not_part_of_the_published_lineage(
        self, world: World
    ) -> None:
        base = publish(world, "request", "x", producer=EXTERNAL_PRODUCER)
        live = publish(world, "requirements", {"n": 1}, producer="requirements", inputs=(base.ref,))
        world.store.put(
            ArtifactDraft("design_doc", {"n": 1}),
            producer="design",
            inputs=(base.ref,),
            staged=True,
        )
        rejected = world.store.put(
            ArtifactDraft("task_plan", {"n": 1}),
            producer="decompose",
            inputs=(base.ref,),
            staged=True,
        )
        world.store.quarantine([rejected.id], "no")
        assert [a.id for a in world.graph.downstream(base.id)] == [live.id]

    def test_superseded_derivations_are_history_and_stay_listed(self, world: World) -> None:
        base = publish(world, "request", "x", producer=EXTERNAL_PRODUCER)
        old = publish(world, "requirements", {"v": 1}, producer="requirements", inputs=(base.ref,))
        new = publish(world, "requirements", {"v": 2}, producer="requirements", inputs=(base.ref,))
        assert [a.id for a in world.graph.downstream(base.id)] == [old.id, new.id]

    def test_downstream_terminates_on_cycles_and_excludes_the_root(self, world: World) -> None:
        a_ref = ArtifactRef("design_doc@v1", "design_doc", "", 1, "0" * 64)
        b = publish(world, "requirements", {"b": 1}, producer="requirements", inputs=(a_ref,))
        a = publish(world, "design_doc", {"a": 1}, producer="design", inputs=(b.ref,))
        assert [x.id for x in world.graph.downstream(a.id)] == [b.id]

    def test_unknown_artifact_raises(self, world: World) -> None:
        with pytest.raises(StateError):
            world.graph.downstream("nope@v1")


# =============================================================================
# explain / to_dict
# =============================================================================


class TestExplain:
    def test_the_explanation_is_an_indented_tree(self, pipeline: Pipeline) -> None:
        def art(a: Artifact) -> str:
            return f"artifact {a.id} [active] producer={a.producer} sha256={a.content_hash[:12]}"

        clarification = "decision D-0001 [clarification] TTL is 24h (by human:alice)"
        assumption = "decision D-0002 [assumption] assume 24h (by agent:architect)"
        design = "decision D-0003 [design] use SQLite (by agent:architect)"
        release = "decision D-0004 [release] ship it (by human:alice)"
        p = pipeline
        expected = [
            art(p.patch),
            f"  input: {art(p.design)}",
            f"    input: {art(p.requirements)}",
            f"      input: {art(p.request)}",
            f"      input: {art(p.answers)}",
            f"        decided by: {clarification}",
            f"      decided by: {assumption}",
            f"        parent: {clarification}  (already shown above)",
            f"    decided by: {design}",
            f"      parent: {assumption}  (already shown above)",
            f"  decided by: {release}",
            f"    parent: {design}  (already shown above)",
        ]
        assert p.world.graph.explain(p.patch.id).split("\n") == expected

    def test_the_explanation_mentions_the_clarification_for_the_release_patch(
        self, pipeline: Pipeline
    ) -> None:
        assert "[clarification]" in pipeline.world.graph.explain(pipeline.patch.id)

    def test_every_line_is_single_line_control_free_and_capped(self, world: World) -> None:
        """Defence in depth: even if a decision text were hostile, the rendering is sanitised."""
        art = publish(world, "design_doc", {"n": 1}, producer="design")
        hostile = Decision(
            id="D-0001",
            run_id="run1",
            ts=T0,
            kind=DecisionKind.DESIGN,
            actor=Actor.human("alice"),
            node_id=None,
            summary="ok\x1b[31m RED\x1b[0m\nFAKE LINE\r\n\u202eevil\u200b" + "z" * 1000,
            outputs=(art.ref,),
        )

        class ForgedLog:
            def decisions_for(self, ref: ArtifactRef) -> tuple[Decision, ...]:
                return (hostile,) if ref == art.ref else ()

            def get(self, decision_id: str) -> Decision:
                if decision_id != hostile.id:
                    raise StateError("unknown")
                return hostile

            def __contains__(self, decision_id: object) -> bool:
                return decision_id == hostile.id

        graph = LineageGraph(world.store, ForgedLog())  # type: ignore[arg-type]
        text = graph.explain(art.id)
        lines = text.split("\n")
        assert len(lines) == 2  # the embedded newlines did not create extra lines
        for line in lines:
            assert (
                "\x1b" not in line
                and "\r" not in line
                and "\u202e" not in line
                and "\u200b" not in line
            )
            assert len(line) <= 2 + len("decided by: ") + lineage_module.MAX_LABEL_CHARS
        assert "FAKE LINE" in lines[1] and "FAKE LINE" not in lines[0]

    def test_explain_is_deterministic(self, pipeline: Pipeline) -> None:
        graph = pipeline.world.graph
        assert graph.explain(pipeline.patch.id) == graph.explain(pipeline.patch.id)


class TestToDict:
    def test_to_dict_is_json_ready_and_deterministic(self, pipeline: Pipeline) -> None:
        graph = pipeline.world.graph
        d = graph.to_dict(pipeline.request.id)
        assert json.loads(json.dumps(d)) == d
        assert d == graph.to_dict(pipeline.request.id)
        assert d["artifact_id"] == pipeline.request.id
        assert d["upstream"]["type"] == ARTIFACT and d["upstream"]["relation"] == "root"
        assert [r["artifact_id"] for r in d["downstream"]] == [
            pipeline.design.id,
            pipeline.patch.id,
            pipeline.requirements.id,
        ]
        assert set(d["downstream"][0]) == {"artifact_id", "kind", "key", "version", "content_hash"}

    def test_node_dicts_nest_children(self, pipeline: Pipeline) -> None:
        d = pipeline.world.graph.to_dict(pipeline.patch.id)["upstream"]
        assert set(d) == {"type", "id", "relation", "label", "note", "children"}
        assert [c["id"] for c in d["children"]] == [pipeline.design.id, pipeline.release.id]
        assert d["children"][1]["type"] == DECISION and d["children"][1]["relation"] == "decision"

    def test_walk_is_pre_order(self, pipeline: Pipeline) -> None:
        root = pipeline.world.graph.upstream(pipeline.patch.id)
        order = [n.id for n in root.walk()]
        assert order[0] == pipeline.patch.id
        assert order.index(pipeline.design.id) < order.index(pipeline.requirements.id)
        assert order.index(pipeline.release.id) > order.index(pipeline.request.id)


class TestLiveness:
    def test_the_graph_sees_new_records_without_being_rebuilt(self, world: World) -> None:
        art = publish(world, "design_doc", {"n": 1}, producer="design")
        assert world.graph.reaches(art.id, DecisionKind.DESIGN) is False
        world.log.record(DecisionKind.DESIGN, "later", actor=AGENT, outputs=(art.ref,))
        assert world.graph.reaches(art.id, DecisionKind.DESIGN) is True
