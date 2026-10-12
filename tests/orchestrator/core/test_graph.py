"""WorkflowGraph: structure queries, validation, governed expansion, persistence, rendering."""

from __future__ import annotations

import copy
import itertools
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from orchestrator.core.errors import ConfigError, GraphError, PlanPatchError
from orchestrator.core.graph import CriticalPath, ExpansionDiff, WorkflowGraph
from orchestrator.core.types import (
    ApprovalSpec,
    NodeSpec,
    NodeStatus,
    PlanExpansion,
    RemediationSpec,
    Rewire,
    RiskLevel,
    Stage,
)

# --------------------------------------------------------------------------- builders


def spec(node_id: str, deps: Sequence[str] = (), **kwargs: Any) -> NodeSpec:
    fields: dict[str, Any] = {"title": node_id, "stage": Stage.IMPLEMENTATION, "agent": "agent"}
    fields.update(kwargs)
    return NodeSpec(id=node_id, depends_on=tuple(deps), **fields)


def sdlc_like() -> WorkflowGraph:
    """A miniature of the standard workflow: dynamic ``implement.*`` nodes come later."""
    return WorkflowGraph(
        [
            spec("intake", consumes=("request",), produces=("intake_report",)),
            spec(
                "requirements",
                ["intake"],
                consumes=("request", "intake_report"),
                optional_consumes=("clarification_answers",),
                produces=("requirements",),
            ),
            spec(
                "design",
                ["requirements"],
                consumes=("requirements",),
                produces=("design_doc",),
                approval=ApprovalSpec(),
                risk=RiskLevel.MEDIUM,
            ),
            spec(
                "decompose",
                ["design"],
                consumes=("requirements", "design_doc"),
                produces=("task_plan", "plan_expansion"),
                params={"expansion_template": {"retry": {"max_attempts": 2}}},
            ),
            spec("author-tests", ["decompose"], consumes=("task_plan",), produces=("changeset",)),
            spec(
                "integrate",
                ["decompose", "author-tests"],
                consumes=("changeset", "task_plan"),
                produces=("integrated_changeset", "integration_report"),
            ),
            spec(
                "verify",
                ["integrate"],
                consumes=("integrated_changeset",),
                overlay_from=("integrated_changeset",),
                produces=("test_report",),
                mandatory=True,
                remediation=RemediationSpec(route_to=("implement.*", "author-tests")),
            ),
            spec("finish", ["verify"], consumes=("test_report",)),
        ]
    )


def impl(name: str, owner: str = "decompose", **kwargs: Any) -> NodeSpec:
    fields: dict[str, Any] = {"consumes": ("task_plan",), "produces": ("changeset",)}
    fields.update(kwargs)
    return spec(f"implement.{name}", [owner], **fields)


def fan_out(*names: str) -> PlanExpansion:
    return PlanExpansion(
        owner="decompose",
        nodes=tuple(impl(n) for n in names),
        rewire={"integrate": Rewire(add_depends_on=tuple(f"implement.{n}" for n in names))},
    )


def snapshot(graph: WorkflowGraph) -> dict[str, Any]:
    return copy.deepcopy(graph.to_dict())


# --------------------------------------------------------------------------- nodes


class TestNodeManagement:
    def test_nodes_is_an_insertion_ordered_copy(self) -> None:
        graph = WorkflowGraph([spec("b"), spec("a")])
        view = graph.nodes
        assert list(view) == ["b", "a"]
        view.clear()
        assert len(graph) == 2

    def test_lookup_helpers(self) -> None:
        graph = WorkflowGraph([spec("a")])
        assert graph.has("a") and "a" in graph
        assert not graph.has("z") and "z" not in graph
        assert 3 not in graph and None not in graph
        assert graph.node("a").id == "a"
        with pytest.raises(GraphError, match="unknown node"):
            graph.node("z")

    def test_unhashable_lookup_is_a_graph_error_not_a_crash(self) -> None:
        with pytest.raises(GraphError):
            WorkflowGraph().node([1])  # type: ignore[arg-type]

    def test_duplicate_ids_are_rejected_at_construction_and_add(self) -> None:
        with pytest.raises(GraphError, match="duplicate"):
            WorkflowGraph([spec("a"), spec("a")])
        graph = WorkflowGraph([spec("a")])
        with pytest.raises(GraphError, match="duplicate"):
            graph.add_node(spec("a", title="other"))

    def test_replace_and_remove(self) -> None:
        graph = WorkflowGraph([spec("a"), spec("b", ["a"])])
        graph.replace_node(spec("b", ["a"], title="renamed"))
        assert graph.node("b").title == "renamed"
        graph.remove_node("b")
        assert "b" not in graph
        with pytest.raises(GraphError):
            graph.replace_node(spec("zzz"))
        with pytest.raises(GraphError):
            graph.remove_node("zzz")

    def test_remove_leaves_dangling_edges_for_validate_to_report(self) -> None:
        graph = WorkflowGraph([spec("a"), spec("b", ["a"])])
        graph.remove_node("a")
        with pytest.raises(GraphError, match="unknown node 'a'"):
            graph.validate()

    def test_max_nodes_is_enforced(self) -> None:
        graph = WorkflowGraph([spec("a"), spec("b")], max_nodes=2)
        with pytest.raises(GraphError, match="max_nodes"):
            graph.add_node(spec("c"))

    @pytest.mark.parametrize("bad", [0, -1, 10_001, True, "5", 2.5])
    def test_invalid_max_nodes_is_a_config_error(self, bad: Any) -> None:
        with pytest.raises(ConfigError):
            WorkflowGraph(max_nodes=bad)

    def test_external_kinds_must_be_a_collection_of_valid_kinds(self) -> None:
        with pytest.raises(ConfigError):
            WorkflowGraph(external_kinds="request")
        with pytest.raises(ConfigError):
            WorkflowGraph(external_kinds=["Not A Kind"])

    def test_non_nodespec_is_rejected(self) -> None:
        with pytest.raises(GraphError):
            WorkflowGraph().add_node({"id": "a"})  # type: ignore[arg-type]

    def test_id_with_trailing_newline_never_enters_the_graph(self) -> None:
        with pytest.raises((ConfigError, GraphError)):
            WorkflowGraph().add_node(spec("a\n"))

    def test_hostile_dependency_entries_are_rejected_at_the_boundary(self) -> None:
        junk_deps: Any = [{"x": 1}]
        junk = spec("a", junk_deps)
        with pytest.raises(GraphError, match="list of strings"):
            WorkflowGraph().add_node(junk)
        junk_compensates = spec("b", compensates=(3,))
        with pytest.raises(GraphError, match="list of strings"):
            WorkflowGraph().add_node(junk_compensates)

    def test_expansion_members_cannot_be_edited_or_removed_directly(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a"))
        with pytest.raises(GraphError, match="belongs to the expansion"):
            graph.remove_node("implement.a")
        with pytest.raises(GraphError, match="belongs to the expansion"):
            graph.replace_node(impl("a", title="x"))
        with pytest.raises(GraphError, match="owns an expansion"):
            graph.remove_node("decompose")

    def test_copy_is_independent(self) -> None:
        graph = sdlc_like()
        clone = graph.copy()
        clone.add_node(spec("extra", ["finish"]))
        assert "extra" not in graph
        assert graph.graph_hash() != clone.graph_hash()


# --------------------------------------------------------------------------- relations


@pytest.fixture
def diamond() -> WorkflowGraph:
    return WorkflowGraph([spec("a"), spec("b", ["a"]), spec("c", ["a"]), spec("d", ["b", "c"])])


class TestRelations:
    def test_deps_dependents_ancestors_descendants(self, diamond: WorkflowGraph) -> None:
        assert diamond.deps("d") == ("b", "c")
        assert diamond.dependents("a") == {"b", "c"}
        assert diamond.ancestors("d") == {"a", "b", "c"}
        assert diamond.descendants("a") == {"b", "c", "d"}
        assert diamond.ancestors("a") == set() and diamond.descendants("d") == set()

    def test_results_are_fresh_copies(self, diamond: WorkflowGraph) -> None:
        diamond.dependents("a").clear()
        diamond.ancestors("d").clear()
        assert diamond.dependents("a") == {"b", "c"}
        assert diamond.ancestors("d") == {"a", "b", "c"}

    def test_duplicate_declared_deps_collapse(self) -> None:
        graph = WorkflowGraph([spec("a"), spec("b", ["a", "a"])])
        assert graph.deps("b") == ("a",)
        assert graph.topological_order() == ["a", "b"]

    def test_unknown_node_queries_raise(self, diamond: WorkflowGraph) -> None:
        for call in (
            diamond.deps,
            diamond.overlay_of,
            diamond.dependents,
            diamond.ancestors,
            diamond.descendants,
        ):
            with pytest.raises(GraphError):
                call("nope")

    def test_rewires_extend_deps_and_overlays(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        assert graph.deps("integrate") == (
            "decompose",
            "author-tests",
            "implement.a",
            "implement.b",
        )
        assert graph.overlay_of("integrate") == ()
        serial = PlanExpansion(
            "integrate",
            rewire={
                "implement.b": Rewire(
                    add_depends_on=("implement.a",), add_overlay_from=("changeset",)
                )
            },
        )
        graph.apply_expansion(serial)
        assert graph.deps("implement.b") == ("decompose", "implement.a")
        assert graph.overlay_of("implement.b") == ("changeset",)
        assert "implement.b" in graph.descendants("implement.a")

    def test_producers_and_consumers(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a"))
        assert graph.producers_of("changeset") == ["author-tests", "implement.a"]
        assert graph.producers_of("nothing") == []
        assert graph.consumers_of("task_plan") == ["author-tests", "integrate", "implement.a"]
        # overlay and skip inputs count as consumption
        assert "verify" in graph.consumers_of("integrated_changeset")
        graph = WorkflowGraph(
            [
                spec("p", produces=("intake_report",)),
                spec("q", ["p"], skip_inputs=("intake_report",), skip_if="x"),
            ]
        )
        assert graph.consumers_of("intake_report") == ["q"]

    def test_consumers_include_rewired_overlay_kinds(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        graph.apply_expansion(
            PlanExpansion(
                "integrate",
                rewire={
                    "implement.b": Rewire(
                        add_depends_on=("implement.a",), add_overlay_from=("changeset",)
                    )
                },
            )
        )
        assert "implement.b" in graph.consumers_of("changeset")


class TestOrdering:
    def test_topological_order_breaks_ties_by_insertion_order(self) -> None:
        graph = WorkflowGraph([spec("x", ["y"]), spec("y"), spec("z")])
        assert graph.topological_order() == ["y", "x", "z"]  # x (pos 0) is ready after y, before z
        assert WorkflowGraph([spec("b"), spec("a"), spec("c")]).topological_order() == [
            "b",
            "a",
            "c",
        ]

    def test_topological_order_is_a_copy_and_cached_results_stay_valid(
        self, diamond: WorkflowGraph
    ) -> None:
        order = diamond.topological_order()
        order.reverse()
        assert diamond.topological_order() == ["a", "b", "c", "d"]
        diamond.add_node(spec("e", ["d"]))
        assert diamond.topological_order()[-1] == "e"

    def test_topo_index(self, diamond: WorkflowGraph) -> None:
        assert [diamond.topo_index(n) for n in "abcd"] == [0, 1, 2, 3]
        with pytest.raises(GraphError):
            diamond.topo_index("zzz")

    def test_layers(self, diamond: WorkflowGraph) -> None:
        assert diamond.layers() == [["a"], ["b", "c"], ["d"]]
        assert WorkflowGraph().layers() == []
        assert WorkflowGraph([spec("q"), spec("p")]).layers() == [["q", "p"]]

    def test_ordering_a_cyclic_graph_raises_with_the_path(self) -> None:
        graph = WorkflowGraph([spec("a", ["b"]), spec("b", ["a"])])
        for call in (graph.topological_order, graph.layers):
            with pytest.raises(GraphError, match="cycle detected: a -> b -> a"):
                call()

    def test_critical_path(self, diamond: WorkflowGraph) -> None:
        result = diamond.critical_path({"a": 1, "b": 5, "c": 2, "d": 1})
        assert result == CriticalPath(("a", "b", "d"), 7.0)
        assert tuple(result) == (("a", "b", "d"), 7.0)  # unpacks as (path, length)

    def test_critical_path_defaults_and_ties(self, diamond: WorkflowGraph) -> None:
        assert diamond.critical_path({}) == CriticalPath(
            ("a", "b", "d"), 0.0
        )  # earliest-declared dep wins ties
        assert diamond.critical_path({"c": 3}).path == ("a", "c", "d")
        assert WorkflowGraph().critical_path({}) == CriticalPath((), 0.0)

    def test_critical_path_end_node_need_not_be_a_sink(self) -> None:
        graph = WorkflowGraph([spec("a"), spec("b", ["a"]), spec("c")])
        assert graph.critical_path({"a": 1, "b": 1, "c": 9}) == CriticalPath(("c",), 9.0)

    @pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True, "3"])
    def test_critical_path_rejects_bad_durations(self, diamond: WorkflowGraph, bad: Any) -> None:
        with pytest.raises(GraphError, match="duration"):
            diamond.critical_path({"a": bad})


class TestResolvePattern:
    def test_exact_and_prefix_patterns(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        assert graph.resolve_pattern("author-tests") == ["author-tests"]
        assert graph.resolve_pattern("implement.*") == ["implement.a", "implement.b"]
        assert graph.resolve_pattern("ghost") == []
        assert graph.resolve_pattern("ghost.*") == []

    def test_prefix_pattern_does_not_match_the_bare_prefix(self) -> None:
        graph = WorkflowGraph([spec("implement"), spec("implement.x")])
        assert graph.resolve_pattern("implement.*") == ["implement.x"]

    @pytest.mark.parametrize(
        "pattern",
        ["*", "a*", "*.x", "impl.*.x", "../x.*", "x/.*", "Implement.*", "implement.*\n", "", ".*"],
    )
    def test_other_syntax_is_rejected_not_treated_as_a_wildcard(self, pattern: str) -> None:
        with pytest.raises(GraphError):
            WorkflowGraph([spec("a")]).resolve_pattern(pattern)

    def test_non_string_pattern(self) -> None:
        with pytest.raises(GraphError):
            WorkflowGraph().resolve_pattern(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- validation


class TestValidation:
    def test_sdlc_like_graph_is_valid_before_and_after_expansion(self) -> None:
        graph = sdlc_like()
        graph.validate()  # implement.* is still pending: decompose carries an expansion_template
        graph.apply_expansion(fan_out("a", "b"))
        graph.validate()

    def test_empty_graph_is_valid(self) -> None:
        WorkflowGraph().validate()

    def test_cycle_message_contains_the_cycle_path(self) -> None:
        graph = WorkflowGraph([spec("a", ["c"]), spec("b", ["a"]), spec("c", ["b"]), spec("ok")])
        with pytest.raises(GraphError) as info:
            graph.validate()
        assert "cycle detected: a -> c -> b -> a" in str(info.value)

    def test_self_dependency_is_a_cycle(self) -> None:
        with pytest.raises(GraphError, match="cycle detected: a -> a"):
            WorkflowGraph([spec("a", ["a"])]).validate()

    def test_cycle_hidden_behind_an_acyclic_prefix_is_found(self) -> None:
        graph = WorkflowGraph([spec("root"), spec("x", ["root", "y"]), spec("y", ["x"])])
        with pytest.raises(GraphError, match=r"cycle detected: x -> y -> x"):
            graph.validate()

    def test_dangling_dependency(self) -> None:
        with pytest.raises(GraphError, match="node 'a' depends on unknown node 'ghost'"):
            WorkflowGraph([spec("a", ["ghost"])]).validate()

    def test_hostile_identifiers_are_sanitised_in_messages(self) -> None:
        graph = WorkflowGraph([spec("a", ["\x1b[31mevil\x07\u202e"])])
        with pytest.raises(GraphError) as info:
            graph.validate()
        message = str(info.value)
        assert "\x1b" not in message and "\x07" not in message and "\u202e" not in message
        assert "evil" in message

    def test_non_optional_consume_needs_an_ancestor_producer(self) -> None:
        graph = WorkflowGraph([spec("a", consumes=("design_doc",))])
        with pytest.raises(GraphError, match="reads 'design_doc' but no ancestor produces it"):
            graph.validate()

    def test_producer_that_is_not_an_ancestor_does_not_count(self) -> None:
        graph = WorkflowGraph(
            [spec("p", produces=("design_doc",)), spec("c", consumes=("design_doc",))]
        )
        with pytest.raises(GraphError, match="no ancestor produces"):
            graph.validate()
        graph.replace_node(spec("c", ["p"], consumes=("design_doc",)))
        graph.validate()

    def test_optional_consumes_need_an_ancestor_producer_too(self) -> None:
        graph = WorkflowGraph([spec("a", optional_consumes=("impact_report",))])
        with pytest.raises(GraphError, match="'impact_report'"):
            graph.validate()

    def test_skip_inputs_need_an_ancestor_producer(self) -> None:
        graph = WorkflowGraph([spec("a", skip_inputs=("intake_report",), skip_if="g")])
        with pytest.raises(GraphError, match="'intake_report'"):
            graph.validate()

    def test_external_kinds_are_always_available(self) -> None:
        WorkflowGraph(
            [spec("a", consumes=("request",), optional_consumes=("clarification_answers",))]
        ).validate()
        custom = WorkflowGraph([spec("a", consumes=("ticket",))], external_kinds=["ticket"])
        custom.validate()

    def test_a_kind_has_at_most_one_producer_unless_keyed(self) -> None:
        graph = WorkflowGraph(
            [spec("p", produces=("design_doc",)), spec("q", produces=("design_doc",))]
        )
        with pytest.raises(GraphError, match="several producers"):
            graph.validate()
        WorkflowGraph(
            [spec("p", produces=("changeset",)), spec("q", produces=("changeset",))]
        ).validate()

    def test_nodes_may_not_produce_external_kinds(self) -> None:
        with pytest.raises(GraphError, match="produce external kind 'request'"):
            WorkflowGraph([spec("a", produces=("request",))]).validate()

    def test_forged_mandatory_node_with_skip_if_is_caught(self) -> None:
        forged = spec("m", mandatory=True)
        object.__setattr__(
            forged, "skip_if", "greenfield"
        )  # NodeSpec forbids this; a forged/decoded spec might not
        with pytest.raises(GraphError, match="mandatory node 'm' must not declare skip_if"):
            WorkflowGraph([forged]).validate()

    def test_forged_side_effecting_node_without_compensator_is_caught(self) -> None:
        forged = spec("s")
        object.__setattr__(forged, "side_effecting", True)
        with pytest.raises(GraphError, match="side_effecting and compensator"):
            WorkflowGraph([forged]).validate()

    def test_compensates_must_name_a_side_effecting_ancestor(self) -> None:
        promote = spec("promote", side_effecting=True, compensator="undo")
        good = WorkflowGraph([promote, spec("check", ["promote"], compensates=("promote",))])
        good.validate()
        not_effecting = WorkflowGraph([spec("p"), spec("check", ["p"], compensates=("p",))])
        with pytest.raises(GraphError, match="not a side-effecting ancestor"):
            not_effecting.validate()
        not_ancestor = WorkflowGraph([promote, spec("check", compensates=("promote",))])
        with pytest.raises(GraphError, match="not a side-effecting ancestor"):
            not_ancestor.validate()

    def test_remediation_exact_target_must_exist(self) -> None:
        graph = WorkflowGraph(
            [spec("w"), spec("v", ["w"], remediation=RemediationSpec(route_to=("ghost",)))]
        )
        with pytest.raises(GraphError, match="matches no node"):
            graph.validate()

    def test_remediation_pattern_without_match_needs_a_possible_expander(self) -> None:
        verify = spec("v", remediation=RemediationSpec(route_to=("implement.*",)))
        with pytest.raises(GraphError, match="matches no node"):
            WorkflowGraph([verify]).validate()
        owner = spec("planner", params={"expansion_template": {}})
        WorkflowGraph([owner, verify]).validate()

    def test_remediation_pattern_syntax_is_validated(self) -> None:
        graph = WorkflowGraph([spec("v", remediation=RemediationSpec(route_to=("*",)))])
        with pytest.raises(GraphError, match="invalid node pattern"):
            graph.validate()

    def test_remediation_may_not_route_to_itself_through_a_pattern(self) -> None:
        graph = WorkflowGraph([spec("fix.a", remediation=RemediationSpec(route_to=("fix.*",)))])
        with pytest.raises(GraphError, match="routes to itself"):
            graph.validate()

    def test_all_problems_are_reported_and_long_lists_are_truncated(self) -> None:
        nodes = [spec(f"n{i}", ["ghost"]) for i in range(20)]
        with pytest.raises(GraphError) as info:
            WorkflowGraph(nodes).validate()
        message = str(info.value)
        assert message.count("unknown node 'ghost'") == 12
        assert "and 8 more" in message

    def test_cycle_and_dangling_are_reported_together(self) -> None:
        graph = WorkflowGraph([spec("a", ["b", "ghost"]), spec("b", ["a"])])
        with pytest.raises(GraphError) as info:
            graph.validate()
        assert "cycle detected" in str(info.value) and "unknown node 'ghost'" in str(info.value)


# --------------------------------------------------------------------------- expansion


class TestExpansion:
    def test_apply_adds_nodes_in_order_and_reports_the_diff(self) -> None:
        graph = sdlc_like()
        diff = graph.apply_expansion(fan_out("a", "b"))
        assert diff == ExpansionDiff(added=("implement.a", "implement.b"), rewired=("integrate",))
        assert list(graph.nodes)[-2:] == ["implement.a", "implement.b"]
        assert graph.owner_of("implement.a") == "decompose"
        assert graph.owner_of("verify") is None
        graph.validate()

    def test_diff_is_by_id_and_spec_hash(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        changed_b = impl("b", title="a different title")
        diff = graph.apply_expansion(
            PlanExpansion(
                "decompose",
                (impl("a"), changed_b, impl("c")),
                {"integrate": Rewire(add_depends_on=("implement.a", "implement.b", "implement.c"))},
            )
        )
        assert diff.added == ("implement.c",)
        assert diff.retained == ("implement.a",)
        assert diff.changed == ("implement.b",)
        assert diff.removed == ()
        assert diff.rewired == ("integrate",)
        diff = graph.apply_expansion(PlanExpansion("decompose", (impl("a"),)))
        assert diff.removed == ("implement.b", "implement.c")
        assert diff.rewired == ("integrate",)  # the rewire was dropped
        assert "implement.b" not in graph and graph.deps("integrate") == (
            "decompose",
            "author-tests",
        )

    def test_reapplying_an_identical_expansion_is_a_pure_retain(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        before = snapshot(graph)
        diff = graph.apply_expansion(fan_out("a", "b"))
        assert diff == ExpansionDiff(retained=("implement.a", "implement.b"))
        assert snapshot(graph) == before

    def test_retained_nodes_keep_their_position(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        other = PlanExpansion(
            "author-tests", (spec("extra", ["author-tests"], produces=("changeset",)),)
        )
        graph.apply_expansion(other)
        graph.apply_expansion(fan_out("b", "a"))  # declared in another order
        assert list(graph.nodes)[-3:] == ["implement.a", "implement.b", "extra"]

    def test_every_new_node_must_depend_on_the_owner(self) -> None:
        graph = sdlc_like()
        before = snapshot(graph)
        stray = spec("implement.x", ["design"], consumes=("task_plan",), produces=("changeset",))
        with pytest.raises(PlanPatchError, match="must depend on its owner 'decompose'"):
            graph.apply_expansion(PlanExpansion("decompose", (impl("a"), stray)))
        assert snapshot(graph) == before

    def test_new_node_may_not_clash_with_other_nodes(self) -> None:
        graph = sdlc_like()
        before = snapshot(graph)
        with pytest.raises(PlanPatchError, match="clashes with an existing node"):
            graph.apply_expansion(PlanExpansion("decompose", (spec("verify", ["decompose"]),)))
        with pytest.raises(PlanPatchError, match="clashes with an existing node"):
            graph.apply_expansion(PlanExpansion("decompose", (spec("decompose", ["decompose"]),)))
        assert snapshot(graph) == before

    def test_clash_with_another_owners_expansion_node(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a"))
        with pytest.raises(PlanPatchError, match="clashes"):
            graph.apply_expansion(
                PlanExpansion("author-tests", (spec("implement.a", ["author-tests"]),))
            )

    def test_duplicate_ids_inside_one_expansion(self) -> None:
        with pytest.raises(PlanPatchError, match="duplicate node id"):
            sdlc_like().apply_expansion(PlanExpansion("decompose", (impl("a"), impl("a"))))

    def test_unknown_owner(self) -> None:
        with pytest.raises(PlanPatchError, match="owner 'ghost'"):
            sdlc_like().apply_expansion(PlanExpansion("ghost", (spec("x", ["ghost"]),)))

    def test_expansion_nodes_cannot_be_mandatory(self) -> None:
        graph = sdlc_like()
        with pytest.raises(PlanPatchError, match="cannot be mandatory"):
            graph.apply_expansion(PlanExpansion("decompose", (impl("a", mandatory=True),)))

    def test_mandatory_nodes_are_pinned(self) -> None:
        graph = sdlc_like()
        pinned = graph.mandatory_snapshot()
        assert list(pinned) == ["verify"]
        graph.apply_expansion(fan_out("a", "b"))
        assert graph.mandatory_snapshot() == pinned

    def test_rewiring_a_mandatory_node_is_refused(self) -> None:
        graph = sdlc_like()
        before = snapshot(graph)
        with pytest.raises(PlanPatchError, match="mandatory and may not be altered"):
            graph.apply_expansion(
                PlanExpansion(
                    "decompose", (impl("a"),), {"verify": Rewire(add_depends_on=("implement.a",))}
                )
            )
        assert snapshot(graph) == before

    @pytest.mark.parametrize(
        ("target", "message"),
        [
            ("decompose", "cannot rewire its own owner"),
            ("implement.a", "cannot rewire its own node"),
            ("ghost", "does not exist"),
            ("Bad Id", "not a valid node id"),
        ],
    )
    def test_rewire_targets_must_be_existing_non_owner_nodes_outside_the_expansion(
        self, target: str, message: str
    ) -> None:
        graph = sdlc_like()
        before = snapshot(graph)
        with pytest.raises(PlanPatchError, match=message):
            graph.apply_expansion(
                PlanExpansion(
                    "decompose", (impl("a"),), {target: Rewire(add_depends_on=("implement.a",))}
                )
            )
        assert snapshot(graph) == before

    def test_rewire_edge_to_unknown_node_is_rejected(self) -> None:
        graph = sdlc_like()
        with pytest.raises(PlanPatchError, match="unknown node 'nowhere'"):
            graph.apply_expansion(
                PlanExpansion("decompose", (), {"integrate": Rewire(add_depends_on=("nowhere",))})
            )

    def test_rewire_may_not_introduce_a_cycle(self) -> None:
        graph = sdlc_like()
        before = snapshot(graph)
        with pytest.raises(
            PlanPatchError, match="cycle detected: integrate -> finish -> verify -> integrate"
        ):
            graph.apply_expansion(
                PlanExpansion("decompose", (), {"integrate": Rewire(add_depends_on=("finish",))})
            )
        assert snapshot(graph) == before

    def test_rewired_overlays_must_be_change_set_kinds(self) -> None:
        graph = sdlc_like()
        with pytest.raises(PlanPatchError, match="change set kinds"):
            graph.apply_expansion(
                PlanExpansion(
                    "decompose", (), {"integrate": Rewire(add_overlay_from=("design_doc",))}
                )
            )
        graph.apply_expansion(
            PlanExpansion("decompose", (), {"integrate": Rewire(add_overlay_from=("changeset",))})
        )
        assert graph.overlay_of("integrate") == ("changeset",)

    def test_malformed_rewire_shapes(self) -> None:
        graph = sdlc_like()
        forged_rewire: Any = {"integrate": {"add_depends_on": []}}
        bad_edges = Rewire(add_depends_on=(1,))  # type: ignore[arg-type]
        with pytest.raises(PlanPatchError, match="not a Rewire"):
            graph.apply_expansion(PlanExpansion("decompose", (), forged_rewire))
        with pytest.raises(PlanPatchError, match="list of strings"):
            graph.apply_expansion(PlanExpansion("decompose", (), {"integrate": bad_edges}))
        with pytest.raises(PlanPatchError, match="not a PlanExpansion"):
            graph.apply_expansion({"owner": "decompose"})  # type: ignore[arg-type]

    def test_max_nodes_bounds_expansion(self) -> None:
        graph = WorkflowGraph([spec("own", params={"expansion_template": {}})], max_nodes=3)
        before = snapshot(graph)
        too_many = PlanExpansion("own", tuple(spec(f"w.{c}", ["own"]) for c in "abc"))
        with pytest.raises(PlanPatchError, match="max_nodes"):
            graph.apply_expansion(too_many)
        assert snapshot(graph) == before
        graph.apply_expansion(PlanExpansion("own", tuple(spec(f"w.{c}", ["own"]) for c in "ab")))
        assert len(graph) == 3

    def test_removal_is_rejected_while_live_nodes_still_depend_on_the_removed_ones(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        graph.apply_expansion(
            PlanExpansion("integrate", rewire={"finish": Rewire(add_depends_on=("implement.b",))})
        )
        before = snapshot(graph)
        with pytest.raises(PlanPatchError, match=r"unknown node 'implement\.b'"):
            graph.apply_expansion(fan_out("a"))
        assert snapshot(graph) == before
        with pytest.raises(PlanPatchError):
            graph.remove_expansion("decompose")
        assert snapshot(graph) == before

    def test_failed_expansions_are_atomic(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a"))
        before, digest = snapshot(graph), graph.graph_hash()
        bad = PlanExpansion(
            "decompose",
            (impl("a"), impl("b"), impl("c", consumes=("never_produced",))),
            {"integrate": Rewire(add_depends_on=("implement.a", "implement.b"))},
        )
        with pytest.raises(PlanPatchError, match="never_produced"):
            graph.apply_expansion(bad)
        assert snapshot(graph) == before and graph.graph_hash() == digest
        assert graph.topological_order() == sdlc_order_with(graph)  # caches were not poisoned

    def test_remove_expansion(self) -> None:
        graph = sdlc_like()
        original = graph.graph_hash()
        graph.apply_expansion(fan_out("a", "b"))
        diff = graph.remove_expansion("decompose")
        assert diff.removed == ("implement.a", "implement.b") and diff.rewired == ("integrate",)
        assert graph.expansion_of("decompose") is None and graph.expansions() == {}
        assert graph.graph_hash() == original
        assert graph.remove_expansion("decompose") == ExpansionDiff()  # idempotent
        assert graph.remove_expansion("ghost") == ExpansionDiff()

    def test_expansion_of_and_expansions_return_copies(self) -> None:
        graph = sdlc_like()
        original = fan_out("a")
        graph.apply_expansion(original)
        stored = graph.expansion_of("decompose")
        assert stored == original
        assert graph.expansion_of("author-tests") is None
        assert isinstance(stored, PlanExpansion)
        stored.rewire["integrate"] = Rewire(add_depends_on=("finish",))  # type: ignore[index]
        graph.expansions()["decompose"].rewire.clear()  # type: ignore[attr-defined]
        assert graph.expansion_of("decompose") == original
        graph.validate()

    def test_callers_cannot_alter_an_applied_expansion_through_their_reference(self) -> None:
        graph = sdlc_like()
        mutable: dict[str, Rewire] = {"integrate": Rewire(add_depends_on=("implement.a",))}
        graph.apply_expansion(PlanExpansion("decompose", (impl("a"),), mutable))
        mutable["integrate"] = Rewire(
            add_depends_on=("finish",)
        )  # would create a cycle if it leaked in
        mutable["verify"] = Rewire(add_depends_on=("implement.a",))
        assert graph.deps("integrate") == ("decompose", "author-tests", "implement.a")
        assert graph.deps("verify") == ("integrate",)
        graph.validate()

    def test_nested_ownership_is_supported_and_guarded(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a"))
        graph.apply_expansion(
            PlanExpansion("implement.a", (spec("sub.1", ["implement.a"], produces=("changeset",)),))
        )
        assert graph.owner_of("sub.1") == "implement.a"
        with pytest.raises(
            PlanPatchError
        ):  # removing implement.a would orphan sub.1 and its owner record
            graph.apply_expansion(PlanExpansion("decompose", ()))
        graph.remove_expansion("implement.a")
        graph.apply_expansion(PlanExpansion("decompose", ()))
        assert "implement.a" not in graph

    def test_serialising_replan_scenario(self) -> None:
        """Engine-semantics §6: integrate serialises two conflicting implementations."""
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        diff = graph.apply_expansion(
            PlanExpansion(
                "integrate",
                rewire={
                    "implement.b": Rewire(
                        add_depends_on=("implement.a",), add_overlay_from=("changeset",)
                    )
                },
            )
        )
        assert diff == ExpansionDiff(rewired=("implement.b",))
        layer_of = {n: i for i, layer in enumerate(graph.layers()) for n in layer}
        assert layer_of["implement.b"] == layer_of["implement.a"] + 1  # b now waits for a
        graph.validate()


def sdlc_order_with(graph: WorkflowGraph) -> list[str]:
    """Reference topological order: repeatedly take the first node whose deps are done."""
    remaining = list(graph.nodes)
    done: list[str] = []
    while remaining:
        ready = next(n for n in remaining if all(d in done for d in graph.deps(n)))
        done.append(ready)
        remaining.remove(ready)
    return done


# --------------------------------------------------------------------------- persistence


class TestPersistence:
    def test_round_trip_preserves_everything(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a", "b"))
        graph.apply_expansion(
            PlanExpansion(
                "integrate", rewire={"implement.b": Rewire(add_depends_on=("implement.a",))}
            )
        )
        restored = WorkflowGraph.from_dict(graph.to_dict())
        assert restored.to_dict() == graph.to_dict()
        assert restored.graph_hash() == graph.graph_hash()
        assert list(restored.nodes) == list(graph.nodes)
        assert restored.expansions() == graph.expansions()
        assert restored.deps("implement.b") == ("decompose", "implement.a")
        assert restored.mandatory_snapshot() == graph.mandatory_snapshot()

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        graph = sdlc_like()
        graph.apply_expansion(fan_out("a"))
        assert (
            WorkflowGraph.from_dict(json.loads(json.dumps(graph.to_dict()))).graph_hash()
            == graph.graph_hash()
        )

    def test_max_nodes_override_and_config_independent_hash(self) -> None:
        graph = sdlc_like()
        restored = WorkflowGraph.from_dict(graph.to_dict(), max_nodes=50)
        assert restored.max_nodes == 50
        assert restored.graph_hash() == graph.graph_hash()

    def test_hash_changes_with_content_and_order(self) -> None:
        a, b = spec("a"), spec("b")
        assert WorkflowGraph([a, b]).graph_hash() == WorkflowGraph([a, b]).graph_hash()
        assert WorkflowGraph([a, b]).graph_hash() != WorkflowGraph([b, a]).graph_hash()
        assert WorkflowGraph([a]).graph_hash() != WorkflowGraph([spec("a", title="x")]).graph_hash()

    def test_mandatory_snapshot_tracks_spec_hashes(self) -> None:
        graph = sdlc_like()
        snap = graph.mandatory_snapshot()
        graph.replace_node(
            spec(
                "verify",
                ["integrate"],
                mandatory=True,
                title="altered",
                consumes=("integrated_changeset",),
                produces=("test_report",),
            )
        )
        assert graph.mandatory_snapshot()["verify"] != snap["verify"]

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda d: d.update(schema=2),
            lambda d: d.pop("schema"),
            lambda d: d.pop("nodes"),
            lambda d: d.update(nodes="not a list"),
            lambda d: d["nodes"].append(d["nodes"][0]),  # duplicate id
            lambda d: d["nodes"].append({"id": "x"}),  # incomplete node
            lambda d: d["nodes"].append(42),
            lambda d: d.update(max_nodes=-5),
            lambda d: d.update(max_nodes="many"),
            lambda d: d.update(external_kinds="request"),
            lambda d: d.update(
                expansions=[{"owner": "decompose", "nodes": ["ghost"], "rewire": {}}]
            ),
            lambda d: d.update(expansions=[{"owner": 7, "nodes": [], "rewire": {}}]),
            lambda d: d.update(
                expansions=[{"owner": "decompose", "nodes": [], "rewire": {"integrate": 3}}]
            ),
            lambda d: d.update(
                expansions=[{"owner": "decompose", "nodes": ["finish"], "rewire": {}}]
            ),  # does not depend on owner
            lambda d: d.update(expansions=[{"owner": "ghost", "nodes": [], "rewire": {}}]),
            lambda d: d.update(
                expansions=[
                    {
                        "owner": "decompose",
                        "nodes": [],
                        "rewire": {"ghost": {"add_depends_on": ["a"]}},
                    }
                ]
            ),
            lambda d: d.update(
                expansions=[{"owner": "decompose", "nodes": [], "rewire": {}}] * 2
            ),  # owner twice
            lambda d: d["nodes"][1].update(depends_on=["finish"]),  # cycle
            lambda d: d["nodes"][1].update(depends_on=["ghost"]),  # dangling
        ],
    )
    def test_hostile_documents_raise_graph_error(self, mutate: Any) -> None:
        document = copy.deepcopy(sdlc_like().to_dict())
        mutate(document)
        with pytest.raises(GraphError):
            WorkflowGraph.from_dict(document)

    def test_non_mapping_document(self) -> None:
        with pytest.raises(GraphError):
            WorkflowGraph.from_dict([1, 2])  # type: ignore[arg-type]

    def test_validate_can_be_skipped_for_trusted_callers(self) -> None:
        document = WorkflowGraph([spec("a", ["b"]), spec("b", ["a"])]).to_dict()
        graph = WorkflowGraph.from_dict(document, validate=False)
        with pytest.raises(GraphError, match="cycle"):
            graph.validate()


# --------------------------------------------------------------------------- rendering


NODE_LINE = re.compile(r'^  n\d+\["[A-Za-z0-9 _.:/-]*"\]$')
EDGE_LINE = re.compile(r"^  n\d+ --> n\d+$")
CLASSDEF_LINE = re.compile(r"^  classDef st_[a-z_]+ [^;\n]+;$")
CLASS_LINE = re.compile(r"^  class n\d+(,n\d+)* st_[a-z_]+;$")

HOSTILE_TITLES = [
    '"]\n  click n0 call alert(1)\n  n9["',
    "<script>alert(1)</script>",
    'x"] --> n99["y',
    "a|b;c%%{init}%%",
    "back\\slash `tick` 'quote' {brace} (paren)",
    "\u202e\u2028evil\x00\x1b[31m",
    "click n0 href 'javascript:alert(1)'",
    "A" * 500,
]


class TestRendering:
    def test_mermaid_uses_generated_ids_and_declaration_order(self) -> None:
        graph = WorkflowGraph(
            [spec("alpha", title="First"), spec("beta", ["alpha"], title="Second")]
        )
        text = graph.to_mermaid()
        assert text.splitlines() == [
            "flowchart LR",
            '  n0["alpha: First"]',
            '  n1["beta: Second"]',
            "  n0 --> n1",
        ]
        assert text.endswith("\n") and "\r" not in text

    def test_empty_graph(self) -> None:
        assert WorkflowGraph().to_mermaid() == "flowchart LR\n"

    @pytest.mark.parametrize("title", HOSTILE_TITLES)
    def test_hostile_titles_cannot_inject_syntax(self, title: str) -> None:
        graph = WorkflowGraph([spec("alpha", title=title), spec("beta", ["alpha"], title=title)])
        for text in (graph.to_mermaid({"alpha": NodeStatus.FAILED}),):
            for line in text.splitlines():
                stripped = line.strip()
                assert not stripped.startswith("click")
                assert line == "flowchart LR" or any(
                    p.match(line) for p in (NODE_LINE, EDGE_LINE, CLASSDEF_LINE, CLASS_LINE)
                ), line
            assert "<" not in text and ">" not in text.replace("-->", "")
            assert "alert(" not in text  # parentheses are outside the label alphabet
            assert text.count("\n") == len(text.splitlines())

    def test_hostile_node_ids_never_reach_the_output_verbatim(self) -> None:
        graph = WorkflowGraph([spec("a", ["\n click n0 call x()"])])
        text = graph.to_mermaid()
        assert "call x()" not in text and "click" not in text  # dangling dep: no edge, no id leak

    def test_statuses_become_classes_from_a_fixed_palette(self) -> None:
        graph = WorkflowGraph([spec("a"), spec("b", ["a"]), spec("c", ["b"])])
        text = graph.to_mermaid(
            {"a": NodeStatus.SUCCEEDED, "b": "failed", "c": "no-such-status", "ghost": "failed"}
        )
        lines = text.splitlines()
        assert "  class n0 st_succeeded;" in lines
        assert "  class n1 st_failed;" in lines
        assert not any("n2 st_" in line or "ghost" in line for line in lines)
        assert all(CLASSDEF_LINE.match(ln) for ln in lines if ln.startswith("  classDef"))

    def test_status_values_cannot_inject(self) -> None:
        graph = WorkflowGraph([spec("a")])
        text = graph.to_mermaid({"a": "failed;\n click n0 call x()"})
        assert "click" not in text and "classDef" not in text

    def test_dot_output_is_safe_and_uses_generated_ids(self) -> None:
        graph = WorkflowGraph(
            [spec("alpha", title='x"];\nsubgraph evil {'), spec("beta", ["alpha"])]
        )
        dot = graph.to_dot()
        assert dot.splitlines()[0] == "digraph workflow {" and dot.splitlines()[-1] == "}"
        assert "  n0 -> n1;" in dot
        label_lines = [ln for ln in dot.splitlines() if "label=" in ln]
        assert all(
            re.fullmatch(r'  n\d+ \[label="[A-Za-z0-9 _.:/-]*"\];', ln) for ln in label_lines
        )
        assert "subgraph" not in dot.replace(
            "subgraph evil", ""
        )  # only the sanitised label text may remain

    def test_rewired_edges_are_drawn(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a"))
        text = graph.to_mermaid()
        index = {n: f"n{i}" for i, n in enumerate(graph.nodes)}
        assert f"  {index['implement.a']} --> {index['integrate']}" in text


# --------------------------------------------------------------------------- property-style


def random_dag(seed: int, size: int | None = None) -> tuple[list[NodeSpec], dict[str, list[str]]]:
    rng = random.Random(seed)
    count = size if size is not None else rng.randint(2, 40)
    ids = [f"n{i}" for i in range(count)]
    deps: dict[str, list[str]] = {}
    for i, node in enumerate(ids):
        k = rng.randint(0, min(i, 4)) if i else 0
        deps[node] = rng.sample(ids[:i], k)
    insertion = ids[:]
    rng.shuffle(
        insertion
    )  # insertion order is unrelated to topology: deps may point at later nodes
    return [spec(n, deps[n]) for n in insertion], deps


def reference_ancestors(deps: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def visit(node: str) -> set[str]:
        if node not in memo:
            found: set[str] = set()
            for d in deps[node]:
                found |= {d} | visit(d)
            memo[node] = found
        return memo[node]

    return {n: visit(n) for n in deps}


SEEDS = range(40)


class TestPropertiesOnRandomDags:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_topological_order_respects_dependencies_and_ties(self, seed: int) -> None:
        specs, deps = random_dag(seed)
        graph = WorkflowGraph(specs)
        graph.validate()
        order = graph.topological_order()
        assert sorted(order) == sorted(deps)
        position = {n: i for i, n in enumerate(order)}
        assert all(position[d] < position[n] for n, ds in deps.items() for d in ds)
        # determinism rule: the next node is always the earliest-inserted one among the ready nodes
        inserted = {s.id: i for i, s in enumerate(specs)}
        done: set[str] = set()
        for node in order:
            ready = [n for n in deps if n not in done and all(d in done for d in deps[n])]
            assert node == min(ready, key=inserted.__getitem__)
            done.add(node)
        assert [graph.topo_index(n) for n in order] == list(range(len(order)))
        assert WorkflowGraph(specs).topological_order() == order  # reproducible

    @pytest.mark.parametrize("seed", SEEDS)
    def test_layers_are_valid(self, seed: int) -> None:
        specs, deps = random_dag(seed)
        graph = WorkflowGraph(specs)
        layers = graph.layers()
        layer_of = {n: i for i, layer in enumerate(layers) for n in layer}
        assert sorted(layer_of) == sorted(deps) and sum(len(layer) for layer in layers) == len(deps)
        for node, ds in deps.items():
            assert layer_of[node] == (1 + max(layer_of[d] for d in ds) if ds else 0)
        for (
            layer
        ) in layers:  # nodes of one layer never depend on each other (they can run in parallel)
            assert not any(d in layer for n in layer for d in deps[n])

    @pytest.mark.parametrize("seed", SEEDS)
    def test_ancestors_descendants_dependents_are_consistent(self, seed: int) -> None:
        specs, deps = random_dag(seed)
        graph = WorkflowGraph(specs)
        expected = reference_ancestors(deps)
        for node in deps:
            assert graph.ancestors(node) == expected[node]
            assert node not in graph.ancestors(node)
            assert graph.dependents(node) == {n for n, ds in deps.items() if node in ds}
            assert graph.descendants(node) == {n for n, anc in expected.items() if node in anc}
            for other in graph.ancestors(node):
                assert node in graph.descendants(other)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_critical_path_matches_a_brute_force_longest_path(self, seed: int) -> None:
        specs, deps = random_dag(seed)
        rng = random.Random(seed + 1000)
        durations = {n: float(rng.randint(0, 9)) for n in deps}
        graph = WorkflowGraph(specs)
        result = graph.critical_path(durations)
        memo: dict[str, float] = {}

        def longest(node: str) -> float:
            if node not in memo:
                memo[node] = durations[node] + max((longest(d) for d in deps[node]), default=0.0)
            return memo[node]

        assert result.length_s == max(longest(n) for n in deps)
        assert sum(durations[n] for n in result.path) == result.length_s
        assert all(
            result.path[i] in deps[result.path[i + 1]] for i in range(len(result.path) - 1)
        )  # a real chain

    @pytest.mark.parametrize("seed", SEEDS)
    def test_a_back_edge_is_reported_as_a_real_cycle(self, seed: int) -> None:
        specs, deps = random_dag(seed, size=random.Random(seed).randint(3, 30))
        graph = WorkflowGraph(specs)
        rng = random.Random(seed)
        candidates = [n for n in deps if graph.descendants(n)]
        if not candidates:
            pytest.skip("no edges in this random graph")
        victim = rng.choice(candidates)
        target = rng.choice(sorted(graph.descendants(victim)))
        graph.replace_node(spec(victim, [*deps[victim], target]))
        with pytest.raises(GraphError) as info:
            graph.validate()
        match = re.search(r"cycle detected: ([a-z0-9 ->]+)", str(info.value))
        assert match
        path = match.group(1).strip().split(" -> ")
        assert path[0] == path[-1] and len(path) >= 2
        for left, right in itertools.pairwise(path):
            assert right in graph.deps(left)

    @pytest.mark.parametrize("seed", range(15))
    def test_random_expansions_keep_the_graph_valid_or_leave_it_untouched(self, seed: int) -> None:
        rng = random.Random(seed)
        graph = sdlc_like()
        for _ in range(8):
            names = rng.sample("abcdef", rng.randint(0, 4))
            nodes = tuple(impl(n, title=str(rng.random())) for n in names)
            rewire = (
                {"integrate": Rewire(add_depends_on=tuple(f"implement.{n}" for n in names))}
                if names
                else {}
            )
            if rng.random() < 0.3:
                rewire["finish"] = Rewire(
                    add_depends_on=(rng.choice(["verify", "ghost", "integrate"]),)
                )
            before = snapshot(graph)
            try:
                graph.apply_expansion(PlanExpansion("decompose", nodes, rewire))
            except PlanPatchError:
                assert snapshot(graph) == before
            graph.validate()
            assert graph.mandatory_snapshot().keys() == {"verify"}


# --------------------------------------------------------------------------- defensive checks


def mandatory_member_document() -> dict[str, Any]:
    """A persisted graph whose expansion node is mandatory (an agent cannot produce one; a doctored
    ``state.json`` could)."""
    graph = sdlc_like()
    graph.apply_expansion(fan_out("a"))
    document = copy.deepcopy(graph.to_dict())
    for node in document["nodes"]:
        if node["id"] == "implement.a":
            node["mandatory"] = True
    return document


class TestDefensiveChecks:
    def test_expansion_diff_serialises(self) -> None:
        diff = ExpansionDiff(added=("a",), removed=("b",), rewired=("c",))
        assert diff.to_dict() == {
            "added": ["a"],
            "removed": ["b"],
            "retained": [],
            "changed": [],
            "rewired": ["c"],
        }

    def test_forged_node_id_is_rejected_by_add_node(self) -> None:
        forged = spec("fine")
        object.__setattr__(forged, "id", "Not A Valid Id")
        with pytest.raises(GraphError, match="invalid node id"):
            WorkflowGraph().add_node(forged)

    def test_validate_reports_a_graph_larger_than_max_nodes(self) -> None:
        graph = WorkflowGraph([spec("a"), spec("b")])
        graph.max_nodes = 1  # e.g. the operator lowered the limit after the graph was loaded
        with pytest.raises(GraphError, match="exceed max_nodes=1"):
            graph.validate()

    def test_validate_reports_a_node_whose_id_no_longer_matches_its_key(self) -> None:
        graph = WorkflowGraph([spec("a")])
        object.__setattr__(graph.node("a"), "id", "zz")
        with pytest.raises(GraphError, match="does not match its spec id"):
            graph.validate()

    @pytest.mark.parametrize("field", ["nodes", "rewire"])
    def test_expansions_with_non_collection_fields_are_plan_patch_errors(self, field: str) -> None:
        broken = PlanExpansion("decompose", (), {})
        object.__setattr__(broken, field, 42)
        graph = sdlc_like()
        before = snapshot(graph)
        with pytest.raises(PlanPatchError, match="malformed PlanExpansion"):
            graph.apply_expansion(broken)
        assert snapshot(graph) == before

    def test_a_node_claimed_by_two_expansions_is_rejected_on_load(self) -> None:
        graph = sdlc_like()
        graph.apply_expansion(fan_out("a"))
        document = copy.deepcopy(graph.to_dict())
        document["expansions"].append(
            {"owner": "author-tests", "nodes": ["implement.a"], "rewire": {}}
        )
        with pytest.raises(GraphError, match="claimed by two expansions"):
            WorkflowGraph.from_dict(document)

    def test_hostile_new_node_shapes_become_plan_patch_errors(self) -> None:
        junk_deps: Any = [{"x": 1}, "decompose"]
        junk = spec("implement.x", junk_deps, consumes=("task_plan",), produces=("changeset",))
        graph = sdlc_like()
        before = snapshot(graph)
        with pytest.raises(PlanPatchError, match="list of strings"):
            graph.apply_expansion(PlanExpansion("decompose", (junk,)))
        assert snapshot(graph) == before

    def test_an_expansion_may_never_remove_a_mandatory_node_even_in_a_doctored_graph(self) -> None:
        graph = WorkflowGraph.from_dict(mandatory_member_document())
        assert set(graph.mandatory_snapshot()) == {"verify", "implement.a"}
        before = snapshot(graph)
        with pytest.raises(
            PlanPatchError, match=r"mandatory node 'implement\.a' may not be removed or altered"
        ):
            graph.apply_expansion(PlanExpansion("decompose", (), {}))
        with pytest.raises(PlanPatchError, match="may not be removed or altered"):
            graph.remove_expansion("decompose")
        assert snapshot(graph) == before

    def test_replacing_a_mandatory_expansion_node_with_a_non_mandatory_one_is_refused(self) -> None:
        graph = WorkflowGraph.from_dict(mandatory_member_document())
        with pytest.raises(PlanPatchError, match="may not be removed or altered"):
            graph.apply_expansion(PlanExpansion("decompose", (impl("a"),)))
