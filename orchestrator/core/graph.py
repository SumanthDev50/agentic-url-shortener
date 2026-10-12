"""The workflow DAG: structure queries, contract validation and governed dynamic expansion.

A :class:`WorkflowGraph` holds ``NodeSpec`` vertices plus the *expansions* (``PlanExpansion``) that
planning nodes grafted onto it at run time.  Effective edges are

* ``deps(n)``       = ``spec.depends_on`` plus the ``add_depends_on`` of every expansion rewire
  aimed at ``n``
* ``overlay_of(n)`` = ``spec.overlay_from`` plus the ``add_overlay_from`` of those rewires
  (artifact *kinds*, not node ids)

Design decisions
----------------
* **One gate for structure.**  Mutators only enforce local invariants (id grammar via ``NodeSpec``,
  unique ids, ``max_nodes``).  Graph-level invariants (acyclic, no dangling edges, artifact
  contracts, remediation targets, ...) are checked by :meth:`WorkflowGraph.validate`, which the
  engine calls at start and which :meth:`apply_expansion` runs on every candidate result.
* **Expansions are untrusted.**  ``apply_expansion`` is atomic (it mutates a *copy* and swaps on
  success), forces every new node to depend on its owner, never lets an expansion create, remove,
  alter or rewire a mandatory node, and only ever *adds* edges to existing nodes.
* **Deterministic.**  Every iteration order derives from node insertion order, so topological
  order, layers, diagrams and the graph hash are reproducible.
* **Single writer.**  Like the scheduler that owns it, a graph is mutated from one thread; readers
  must work on a :meth:`copy` / snapshot (derived indexes are cached until the next mutation).

Cycle messages print the cycle as ``a -> b -> a`` where ``x -> y`` reads "x depends on y".
"""

from __future__ import annotations

import heapq
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from orchestrator.core.errors import ConfigError, GraphError, PlanPatchError
from orchestrator.core.paths import KIND_RE, NODE_ID_RE
from orchestrator.core.sanitize import mermaid_id, mermaid_label, sanitize_for_terminal
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import (
    CHANGESET_KINDS,
    EXTERNAL_KINDS,
    KEYED_KINDS,
    NodeSpec,
    NodeStatus,
    PlanExpansion,
    Rewire,
)

GRAPH_SCHEMA = 1
MAX_NODES_CEILING = 10_000
_MAX_REPORTED_PROBLEMS = 12
_PREFIX_PATTERN_RE = re.compile(r"^[a-z][a-z0-9_-]{0,40}\.\*$")
_EMPTY_REWIRE = Rewire()

# Fixed palette: statuses come from the NodeStatus vocabulary, never from caller strings.
_MERMAID_STATUS_STYLES: Mapping[NodeStatus, str] = {
    NodeStatus.SUCCEEDED: "fill:#d4edda,stroke:#2e7d32",
    NodeStatus.RUNNING: "fill:#cce5ff,stroke:#1565c0",
    NodeStatus.FAILED: "fill:#f8d7da,stroke:#c62828",
    NodeStatus.BLOCKED: "fill:#f8d7da,stroke:#c62828",
    NodeStatus.AWAITING_APPROVAL: "fill:#fff3cd,stroke:#f9a825",
    NodeStatus.AWAITING_INPUT: "fill:#fff3cd,stroke:#f9a825",
    NodeStatus.SKIPPED: "fill:#e2e3e5,stroke:#6c757d",
    NodeStatus.STALE: "fill:#ffe5d0,stroke:#e8590c",
}


def _q(value: object) -> str:
    """Quote an untrusted identifier for an error message (control-free, length-capped)."""
    return repr(sanitize_for_terminal(value, 60))


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


class ExpansionDiff(NamedTuple):
    """What an expansion changed, by node id (declared order).

    ``rewired`` lists the *existing* nodes whose owner-contributed edges differ from the owner's
    previous expansion (gained, changed or lost) — the engine re-plans exactly those.
    """

    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    retained: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    rewired: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in self._fields}


class CriticalPath(NamedTuple):
    """Longest dependency chain: its node ids (dependency order) and summed duration."""

    path: tuple[str, ...]
    length_s: float


@dataclass(frozen=True)
class _Index:
    """Adjacency derived from the current nodes and expansions (rebuilt after every mutation)."""

    order: tuple[str, ...]
    pos: Mapping[str, int]
    deps: Mapping[str, tuple[str, ...]]
    overlays: Mapping[str, tuple[str, ...]]
    dependents: Mapping[str, tuple[str, ...]]


class _Cache:
    """Lazily filled derived data; replaced wholesale on mutation so it can never go stale."""

    def __init__(self) -> None:
        self.index: _Index | None = None
        self.topo: tuple[str, ...] | None = None
        self.topo_pos: Mapping[str, int] | None = None


def _check_spec_shape(spec: object) -> NodeSpec:
    """Reject specs whose id or *unvalidated* list fields are hostile (``from_dict`` input).

    ``NodeSpec`` validates artifact kinds but not ``depends_on`` / ``compensates`` /
    ``remediation.route_to`` entries (unhashable junk there would crash graph algorithms), and
    older builds of its id check let an id with a trailing newline through, so the id grammar is
    enforced again here with ``fullmatch``.
    """
    if not isinstance(spec, NodeSpec):
        raise GraphError(f"not a NodeSpec: {type(spec).__name__}")
    if not NODE_ID_RE.fullmatch(spec.id):
        raise GraphError(f"invalid node id {_q(spec.id)}")
    fields: list[tuple[str, object]] = [
        ("depends_on", spec.depends_on),
        ("compensates", spec.compensates),
    ]
    if spec.remediation is not None:
        fields.append(("remediation.route_to", spec.remediation.route_to))
    for name, value in fields:
        if not isinstance(value, (tuple, list)) or not all(isinstance(v, str) for v in value):
            raise GraphError(f"node {_q(spec.id)}: {name} must be a list of strings")
    return spec


def _check_rewire_shape(target: object, rewire: object) -> Rewire:
    if not isinstance(target, str) or not NODE_ID_RE.fullmatch(target):
        raise PlanPatchError(f"rewire target is not a valid node id: {_q(target)}")
    if not isinstance(rewire, Rewire):
        raise PlanPatchError(f"rewire for {_q(target)} is not a Rewire")
    for name in ("add_depends_on", "add_overlay_from"):
        value = getattr(rewire, name)
        if not isinstance(value, (tuple, list)) or not all(isinstance(v, str) for v in value):
            raise PlanPatchError(f"rewire for {_q(target)}: {name} must be a list of strings")
    return rewire


def _freeze(expansion: PlanExpansion) -> PlanExpansion:
    """Private copy of ``expansion`` (its ``rewire`` mapping is mutable in the caller's hands)."""
    return PlanExpansion(expansion.owner, tuple(expansion.nodes), dict(expansion.rewire))


class WorkflowGraph:
    """Directed acyclic graph of ``NodeSpec`` with governed dynamic expansion (see module doc)."""

    def __init__(
        self,
        nodes: Iterable[NodeSpec] = (),
        *,
        external_kinds: Iterable[str] = EXTERNAL_KINDS,
        max_nodes: int = 200,
    ) -> None:
        if (
            isinstance(max_nodes, bool)
            or not isinstance(max_nodes, int)
            or not 1 <= max_nodes <= MAX_NODES_CEILING
        ):
            raise ConfigError(f"max_nodes must be an integer in [1, {MAX_NODES_CEILING}]")
        if isinstance(external_kinds, str):
            raise ConfigError("external_kinds must be a collection of kinds, not a string")
        external = frozenset(external_kinds)
        if not all(isinstance(k, str) and KIND_RE.fullmatch(k) for k in external):
            raise ConfigError("external_kinds must be valid artifact kinds")
        self.max_nodes = max_nodes
        self.external_kinds: frozenset[str] = external
        self._nodes: dict[str, NodeSpec] = {}
        self._expansions: dict[str, PlanExpansion] = {}
        self._cache = _Cache()
        for spec in nodes:
            self.add_node(spec)

    # ------------------------------------------------------------------ nodes

    @property
    def nodes(self) -> dict[str, NodeSpec]:
        """Insertion-ordered copy of the id -> spec mapping."""
        return dict(self._nodes)

    def node(self, node_id: str) -> NodeSpec:
        try:
            return self._nodes[node_id]
        except (KeyError, TypeError):
            raise GraphError(f"unknown node {_q(node_id)}") from None

    def has(self, node_id: str) -> bool:
        return isinstance(node_id, str) and node_id in self._nodes

    def __contains__(self, node_id: object) -> bool:
        return isinstance(node_id, str) and node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def add_node(self, spec: NodeSpec) -> None:
        _check_spec_shape(spec)
        if spec.id in self._nodes:
            raise GraphError(f"duplicate node id {_q(spec.id)}")
        if len(self._nodes) >= self.max_nodes:
            raise GraphError(f"graph would exceed max_nodes={self.max_nodes}")
        self._nodes[spec.id] = spec
        self._touch()

    def replace_node(self, spec: NodeSpec) -> None:
        """Swap the spec of an existing node. Expansion-owned nodes change only via their owner."""
        _check_spec_shape(spec)
        if spec.id not in self._nodes:
            raise GraphError(f"unknown node {_q(spec.id)}")
        self._require_not_expansion_member(spec.id, "replace")
        self._nodes[spec.id] = spec
        self._touch()

    def remove_node(self, node_id: str) -> None:
        """Remove a node. Dangling edges left behind are reported by :meth:`validate`."""
        self.node(node_id)
        self._require_not_expansion_member(node_id, "remove")
        if node_id in self._expansions:
            raise GraphError(f"node {_q(node_id)} owns an expansion; remove_expansion first")
        del self._nodes[node_id]
        self._touch()

    def owner_of(self, node_id: str) -> str | None:
        """Owner of the expansion that created ``node_id`` (``None`` for base-graph nodes)."""
        for owner, expansion in self._expansions.items():
            if any(n.id == node_id for n in expansion.nodes):
                return owner
        return None

    def _require_not_expansion_member(self, node_id: str, verb: str) -> None:
        owner = self.owner_of(node_id)
        if owner is not None:
            raise GraphError(
                f"cannot {verb} {_q(node_id)} directly: it belongs to the expansion of {_q(owner)}"
            )

    def _touch(self) -> None:
        self._cache = _Cache()

    def copy(self) -> WorkflowGraph:
        """Independent graph with the same nodes and expansions (specs are immutable, shared)."""
        clone = WorkflowGraph((), external_kinds=self.external_kinds, max_nodes=self.max_nodes)
        clone._nodes = dict(self._nodes)
        clone._expansions = {owner: _freeze(exp) for owner, exp in self._expansions.items()}
        return clone

    # ------------------------------------------------------------------ edges and relations

    def _index(self) -> _Index:
        cache = self._cache
        if cache.index is None:
            cache.index = self._build_index()
        return cache.index

    def _build_index(self) -> _Index:
        extra_deps: dict[str, list[str]] = {}
        extra_overlays: dict[str, list[str]] = {}
        for expansion in self._expansions.values():
            for target, rewire in expansion.rewire.items():
                extra_deps.setdefault(target, []).extend(rewire.add_depends_on)
                extra_overlays.setdefault(target, []).extend(rewire.add_overlay_from)
        order = tuple(self._nodes)
        deps = {n: _dedupe((*self._nodes[n].depends_on, *extra_deps.get(n, ()))) for n in order}
        overlays = {
            n: _dedupe((*self._nodes[n].overlay_from, *extra_overlays.get(n, ()))) for n in order
        }
        dependents: dict[str, list[str]] = {n: [] for n in order}
        for n in order:
            for d in deps[n]:
                if d in dependents:
                    dependents[d].append(n)
        return _Index(
            order=order,
            pos={n: i for i, n in enumerate(order)},
            deps=deps,
            overlays=overlays,
            dependents={n: tuple(v) for n, v in dependents.items()},
        )

    def deps(self, node_id: str) -> tuple[str, ...]:
        """Effective dependencies (declared + rewired), declaration order, de-duplicated."""
        self.node(node_id)
        return self._index().deps[node_id]

    def overlay_of(self, node_id: str) -> tuple[str, ...]:
        """Effective ``overlay_from`` artifact kinds (declared + rewired)."""
        self.node(node_id)
        return self._index().overlays[node_id]

    def dependents(self, node_id: str) -> set[str]:
        """Nodes that directly depend on ``node_id``."""
        self.node(node_id)
        return set(self._index().dependents[node_id])

    def ancestors(self, node_id: str) -> set[str]:
        """Strict transitive dependencies (never contains ``node_id`` in an acyclic graph)."""
        self.node(node_id)
        return self._closure(node_id, self._index().deps)

    def descendants(self, node_id: str) -> set[str]:
        """Strict transitive dependents."""
        self.node(node_id)
        return self._closure(node_id, self._index().dependents)

    @staticmethod
    def _closure(start: str, edges: Mapping[str, tuple[str, ...]]) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            for nxt in edges.get(stack.pop(), ()):
                if nxt in edges and nxt not in seen:  # unknown ids (dangling edges) are skipped
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    # ------------------------------------------------------------------ ordering

    def topological_order(self) -> list[str]:
        """Kahn's algorithm; among ready nodes the earliest-inserted goes first (deterministic)."""
        return list(self._topo())

    def topo_index(self, node_id: str) -> int:
        self.node(node_id)
        self._topo()
        positions = self._cache.topo_pos
        assert positions is not None  # filled by _topo()
        return positions[node_id]

    def _topo(self) -> tuple[str, ...]:
        cache = self._cache
        if cache.topo is None:
            order, leftover = self._kahn()
            if leftover:
                raise GraphError(f"cycle detected: {' -> '.join(self._cycle_path(leftover))}")
            cache.topo = order
            cache.topo_pos = {n: i for i, n in enumerate(order)}
        return cache.topo

    def _kahn(self) -> tuple[tuple[str, ...], set[str]]:
        idx = self._index()
        indegree = {n: sum(1 for d in idx.deps[n] if d in idx.pos) for n in idx.order}
        ready = [idx.pos[n] for n in idx.order if indegree[n] == 0]
        heapq.heapify(ready)
        out: list[str] = []
        while ready:
            node = idx.order[heapq.heappop(ready)]
            out.append(node)
            for child in idx.dependents[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, idx.pos[child])
        return tuple(out), {n for n in idx.order if indegree[n] > 0}

    def _cycle_path(self, leftover: set[str]) -> list[str]:
        """One concrete cycle among the nodes Kahn could not order.

        Every leftover node has a dependency that is also leftover (otherwise its in-degree would
        have reached zero), so repeatedly following the first such dependency must revisit a node.
        """
        idx = self._index()
        node = min(leftover, key=lambda n: idx.pos[n])
        seen_at: dict[str, int] = {}
        path: list[str] = []
        while node not in seen_at:
            seen_at[node] = len(path)
            path.append(node)
            node = next(d for d in idx.deps[node] if d in leftover)
        return [*path[seen_at[node] :], node]

    def layers(self) -> list[list[str]]:
        """Nodes grouped by longest distance from a root; each layer can run in parallel."""
        idx = self._index()
        level: dict[str, int] = {}
        for node in self._topo():
            level[node] = 1 + max((level[d] for d in idx.deps[node] if d in level), default=-1)
        out: list[list[str]] = [[] for _ in range(1 + max(level.values(), default=-1))]
        for node in idx.order:
            out[level[node]].append(node)
        return out

    def critical_path(self, durations: Mapping[str, float]) -> CriticalPath:
        """Longest dependency chain weighted by ``durations`` (missing nodes weigh 0).

        Equal lengths prefer the chain with more nodes (so all-zero durations still yield a real
        chain), then the earlier-declared dependency / earlier-inserted end node.
        """
        for node_id, value in durations.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise GraphError(f"duration of {_q(node_id)} must be a finite, non-negative number")
        idx = self._index()
        best: dict[str, float] = {}
        hops: dict[str, int] = {}
        previous: dict[str, str | None] = {}
        for node in self._topo():
            via: str | None = None
            for d in idx.deps[node]:
                if d in best and (via is None or (best[d], hops[d]) > (best[via], hops[via])):
                    via = d
            best[node] = (best[via] if via else 0.0) + float(durations.get(node, 0.0))
            hops[node] = (hops[via] if via else 0) + 1
            previous[node] = via
        if not best:
            return CriticalPath((), 0.0)
        end = max(best, key=lambda n: (best[n], hops[n], -idx.pos[n]))
        path: list[str] = []
        cursor: str | None = end
        while cursor is not None:
            path.append(cursor)
            cursor = previous[cursor]
        return CriticalPath(tuple(reversed(path)), best[end])

    # ------------------------------------------------------------------ artifact contracts

    def producers_of(self, kind: str) -> list[str]:
        return [n for n, spec in self._nodes.items() if kind in spec.produces]

    def consumers_of(self, kind: str) -> list[str]:
        """Nodes reading ``kind`` (consumes, optional, effective overlay, skip inputs)."""
        idx = self._index()
        return [
            n
            for n, spec in self._nodes.items()
            if kind
            in (*spec.consumes, *spec.optional_consumes, *idx.overlays[n], *spec.skip_inputs)
        ]

    def resolve_pattern(self, pattern: str) -> list[str]:
        """Resolve an exact node id or a ``prefix.*`` pattern against the live graph.

        Any other syntax (bare ``*``, mid-string wildcards, ...) raises ``GraphError`` so a
        remediation target can never silently match "everything".
        """
        if not isinstance(pattern, str):
            raise GraphError("node pattern must be a string")
        if pattern.endswith(".*"):
            if not _PREFIX_PATTERN_RE.fullmatch(pattern):
                raise GraphError(f"invalid node pattern {_q(pattern)}")
            prefix = pattern[:-1]
            return [n for n in self._nodes if n.startswith(prefix) and len(n) > len(prefix)]
        if NODE_ID_RE.fullmatch(pattern):
            return [pattern] if pattern in self._nodes else []
        raise GraphError(f"invalid node pattern {_q(pattern)}")

    # ------------------------------------------------------------------ validation

    def validate(self) -> None:
        """Raise ``GraphError`` listing every violated invariant (cycle messages carry the path).

        Checked: size cap, id integrity, dangling edges and compensation targets, acyclicity,
        expansion bookkeeping, artifact contracts (every consumed kind — also optional and skip
        inputs — has an *ancestor* producer or is external; each non-keyed kind has at most one
        producer; external kinds are never produced by nodes), overlay kinds, mandatory/skip_if,
        side effects/compensators and remediation targets.
        """
        problems = self._problems()
        if problems:
            shown = "; ".join(problems[:_MAX_REPORTED_PROBLEMS])
            extra = len(problems) - _MAX_REPORTED_PROBLEMS
            more = f"; ... and {extra} more" if extra > 0 else ""
            raise GraphError(f"invalid workflow graph: {shown}{more}")

    def _problems(self) -> list[str]:
        problems = self._structural_problems()
        _, leftover = self._kahn()
        if leftover:
            problems.append(f"cycle detected: {' -> '.join(self._cycle_path(leftover))}")
            return problems  # ancestry (hence the contract checks) is meaningless with a cycle
        problems.extend(self._contract_problems())
        problems.extend(self._governance_problems())
        return problems

    def _structural_problems(self) -> list[str]:
        idx = self._index()
        problems: list[str] = []
        if len(self._nodes) > self.max_nodes:
            problems.append(f"{len(self._nodes)} nodes exceed max_nodes={self.max_nodes}")
        for key, spec in self._nodes.items():
            if spec.id != key or not NODE_ID_RE.fullmatch(key):
                problems.append(f"node key {_q(key)} does not match its spec id {_q(spec.id)}")
        for node_id in idx.order:
            problems.extend(
                f"node {_q(node_id)} depends on unknown node {_q(d)}"
                for d in idx.deps[node_id]
                if d not in idx.pos
            )
            bad = [k for k in idx.overlays[node_id] if k not in CHANGESET_KINDS]
            if bad:
                problems.append(
                    f"node {_q(node_id)}: overlay_from must be change set kinds, got {_q(bad)}"
                )
        problems.extend(self._expansion_problems())
        return problems

    def _expansion_problems(self) -> list[str]:
        problems: list[str] = []
        claimed: dict[str, str] = {}
        for owner, expansion in self._expansions.items():
            if owner not in self._nodes:
                problems.append(f"expansion owner {_q(owner)} is not a node")
            for spec in expansion.nodes:
                if spec.id in claimed:
                    problems.append(f"node {_q(spec.id)} is claimed by two expansions")
                claimed[spec.id] = owner
                if owner not in spec.depends_on:
                    problems.append(
                        f"expansion node {_q(spec.id)} does not depend on its owner {_q(owner)}"
                    )
            problems.extend(
                f"expansion of {_q(owner)} rewires unknown node {_q(target)}"
                for target in expansion.rewire
                if target not in self._nodes
            )
        return problems

    def _contract_problems(self) -> list[str]:
        problems: list[str] = []
        producers: dict[str, list[str]] = {}
        for node_id, spec in self._nodes.items():
            for kind in spec.produces:
                producers.setdefault(kind, []).append(node_id)
        for kind, makers in producers.items():
            if kind in self.external_kinds:
                problems.append(f"nodes {_q(makers)} produce external kind {_q(kind)}")
            elif kind not in KEYED_KINDS and len(makers) > 1:
                problems.append(f"kind {_q(kind)} has several producers: {_q(makers)}")
        for node_id, spec in self._nodes.items():
            ancestors = self.ancestors(node_id)
            for kind in _dedupe((*spec.consumes, *spec.optional_consumes, *spec.skip_inputs)):
                if kind in self.external_kinds:
                    continue
                if not any(maker in ancestors for maker in producers.get(kind, ())):
                    problems.append(
                        f"node {_q(node_id)} reads {_q(kind)} but no ancestor produces it"
                    )
        return problems

    def _governance_problems(self) -> list[str]:
        problems: list[str] = []
        can_expand = any("expansion_template" in spec.params for spec in self._nodes.values())
        for node_id, spec in self._nodes.items():
            if spec.mandatory and spec.skip_if:
                problems.append(f"mandatory node {_q(node_id)} must not declare skip_if")
            if bool(spec.side_effecting) != bool(spec.compensator):
                problems.append(
                    f"node {_q(node_id)}: side_effecting and compensator must go together"
                )
            ancestors = self.ancestors(node_id)
            for target in spec.compensates:
                if target not in ancestors or not self._nodes[target].side_effecting:
                    problems.append(
                        f"node {_q(node_id)} compensates {_q(target)}, "
                        "which is not a side-effecting ancestor"
                    )
            if spec.remediation is not None:
                problems.extend(
                    self._remediation_problems(node_id, spec.remediation.route_to, can_expand)
                )
        return problems

    def _remediation_problems(
        self, node_id: str, route_to: tuple[str, ...], can_expand: bool
    ) -> list[str]:
        problems: list[str] = []
        for entry in route_to:
            try:
                matches = self.resolve_pattern(entry)
            except GraphError as exc:
                problems.append(f"node {_q(node_id)} remediation: {exc}")
                continue
            if node_id in matches:
                problems.append(f"node {_q(node_id)} remediation routes to itself via {_q(entry)}")
            # A ``prefix.*`` pattern may be satisfied later by a planning node's expansion; those
            # owners are recognisable by the ``expansion_template`` they carry (engine E1).
            pending_ok = entry.endswith(".*") and can_expand
            if not matches and not pending_ok:
                problems.append(
                    f"node {_q(node_id)} remediation target {_q(entry)} matches no node"
                )
        return problems

    # ------------------------------------------------------------------ expansion

    def apply_expansion(self, expansion: PlanExpansion) -> ExpansionDiff:
        """Replace the owner's previous expansion by ``expansion`` (atomically).

        Structural rules (anything else is the engine's job, see ``engine/expansion.py``): the owner
        exists; new ids are unique and do not clash with nodes of other owners; every new node
        depends directly on the owner and is not ``mandatory``; rewire targets are existing,
        non-mandatory nodes outside this expansion and only gain edges/overlays; mandatory nodes
        stay byte-identical; ``max_nodes`` holds; the result is acyclic and contract-valid.
        The candidate is built and validated on a copy: on ``PlanPatchError`` this graph is
        untouched.
        """
        if not isinstance(expansion, PlanExpansion):
            raise PlanPatchError("not a PlanExpansion")
        try:
            frozen = _freeze(expansion)
        except (TypeError, ValueError) as exc:
            raise PlanPatchError(
                "malformed PlanExpansion (nodes or rewire are not collections)"
            ) from exc
        return self._swap_expansion(frozen.owner, frozen)

    def remove_expansion(self, owner: str) -> ExpansionDiff:
        """Drop the owner's expansion (its nodes and rewires). A no-op if it has none."""
        if owner not in self._expansions:
            return ExpansionDiff()
        return self._swap_expansion(owner, None)

    def _swap_expansion(self, owner: str, expansion: PlanExpansion | None) -> ExpansionDiff:
        pinned = self.mandatory_snapshot()
        trial = self.copy()
        try:
            diff = trial._install_expansion(owner, expansion)
            trial.validate()
            trial._check_pinned(pinned)
        except PlanPatchError:
            raise
        except GraphError as exc:
            raise PlanPatchError(str(exc)) from exc
        self._nodes, self._expansions = trial._nodes, trial._expansions
        self._touch()
        return diff

    def _install_expansion(self, owner: str, expansion: PlanExpansion | None) -> ExpansionDiff:
        """Mutating step of :meth:`apply_expansion` (only ever called on a private copy)."""
        if not isinstance(owner, str) or owner not in self._nodes:
            raise PlanPatchError(f"expansion owner {_q(owner)} is not a node of the graph")
        previous = self._expansions.get(owner)
        old_nodes = {n.id: n for n in previous.nodes} if previous else {}
        new_specs = expansion.nodes if expansion else ()
        new_ids = self._check_new_nodes(owner, new_specs, old_nodes)
        rewire = {
            t: _check_rewire_shape(t, r) for t, r in (expansion.rewire if expansion else {}).items()
        }
        for target in rewire:
            self._check_rewire_target(owner, target, new_ids, old_nodes)

        removed = tuple(i for i in old_nodes if i not in new_ids)
        for node_id in removed:
            del self._nodes[node_id]
        added: list[str] = []
        retained: list[str] = []
        changed: list[str] = []
        for spec in new_specs:
            if spec.id not in old_nodes:
                added.append(spec.id)
            elif old_nodes[spec.id].spec_hash() == spec.spec_hash():
                retained.append(spec.id)
            else:
                changed.append(spec.id)
            self._nodes[spec.id] = spec  # replacing keeps the dict position of retained nodes
        if expansion is None:
            self._expansions.pop(owner, None)
        else:
            self._expansions[owner] = expansion
        if len(self._nodes) > self.max_nodes:
            raise PlanPatchError(f"expansion would exceed max_nodes={self.max_nodes}")
        self._touch()
        return ExpansionDiff(
            tuple(added),
            removed,
            tuple(retained),
            tuple(changed),
            self._rewired_targets(previous, rewire),
        )

    def _check_new_nodes(
        self, owner: str, specs: tuple[NodeSpec, ...], old_nodes: Mapping[str, NodeSpec]
    ) -> set[str]:
        ids: set[str] = set()
        for spec in specs:
            try:
                _check_spec_shape(spec)
            except GraphError as exc:
                raise PlanPatchError(str(exc)) from exc
            if spec.id in ids:
                raise PlanPatchError(f"duplicate node id {_q(spec.id)} in expansion")
            ids.add(spec.id)
            if spec.id in self._nodes and spec.id not in old_nodes:
                raise PlanPatchError(f"expansion node {_q(spec.id)} clashes with an existing node")
            if owner not in spec.depends_on:
                raise PlanPatchError(
                    f"expansion node {_q(spec.id)} must depend on its owner {_q(owner)}"
                )
            if spec.mandatory:
                raise PlanPatchError(f"expansion node {_q(spec.id)} cannot be mandatory")
        return ids

    def _check_rewire_target(
        self, owner: str, target: str, new_ids: set[str], old_nodes: Mapping[str, NodeSpec]
    ) -> None:
        if target == owner:
            raise PlanPatchError(f"expansion of {_q(owner)} cannot rewire its own owner")
        if target in new_ids or target in old_nodes:
            raise PlanPatchError(
                f"expansion of {_q(owner)} cannot rewire its own node {_q(target)}"
            )
        spec = self._nodes.get(target)
        if spec is None:
            raise PlanPatchError(f"rewire target {_q(target)} does not exist")
        if spec.mandatory:
            raise PlanPatchError(f"rewire target {_q(target)} is mandatory and may not be altered")

    @staticmethod
    def _rewired_targets(
        previous: PlanExpansion | None, new_rewire: Mapping[str, Rewire]
    ) -> tuple[str, ...]:
        old = {t: r for t, r in (previous.rewire if previous else {}).items() if r != _EMPTY_REWIRE}
        new = {t: r for t, r in new_rewire.items() if r != _EMPTY_REWIRE}
        return tuple(sorted(t for t in {*old, *new} if old.get(t) != new.get(t)))

    def _check_pinned(self, pinned: Mapping[str, str]) -> None:
        now = self.mandatory_snapshot()
        for node_id, digest in pinned.items():
            if now.get(node_id) != digest:
                raise PlanPatchError(f"mandatory node {_q(node_id)} may not be removed or altered")

    def expansion_of(self, owner: str) -> PlanExpansion | None:
        """Copy of the owner's current expansion, or ``None``."""
        expansion = self._expansions.get(owner)
        return _freeze(expansion) if expansion else None

    def expansions(self) -> dict[str, PlanExpansion]:
        """Copies of every expansion keyed by owner (application order)."""
        return {owner: _freeze(exp) for owner, exp in self._expansions.items()}

    # ------------------------------------------------------------------ identity and persistence

    def mandatory_snapshot(self) -> dict[str, str]:
        """``{id: spec_hash}`` of every mandatory node (the engine pins it for the whole run)."""
        return {n: spec.spec_hash() for n, spec in self._nodes.items() if spec.mandatory}

    def _structure(self) -> dict[str, Any]:
        return {
            "nodes": [spec.to_dict() for spec in self._nodes.values()],
            "expansions": [
                {
                    "owner": owner,
                    "nodes": [n.id for n in exp.nodes],
                    "rewire": {t: r.to_dict() for t, r in exp.rewire.items()},
                }
                for owner, exp in self._expansions.items()
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form; ``from_dict`` rebuilds an identical graph (order, expansions)."""
        return {
            "schema": GRAPH_SCHEMA,
            "max_nodes": self.max_nodes,
            "external_kinds": sorted(self.external_kinds),
            **self._structure(),
        }

    def graph_hash(self) -> str:
        """Hash of the workflow *structure* (nodes in order + expansions); process config such as
        ``max_nodes`` is deliberately excluded so a resume under another config still matches."""
        return hash_jsonable(self._structure())

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        max_nodes: int | None = None,
        validate: bool = True,
    ) -> WorkflowGraph:
        """Rebuild a graph from untrusted JSON (``state.json``).

        Malformed input raises ``GraphError``; by default the result is also :meth:`validate`-d."""
        try:
            return cls._from_dict(data, max_nodes, validate)
        except (ConfigError, KeyError, ValueError, TypeError, AttributeError) as exc:
            raise GraphError(
                f"malformed graph document: {sanitize_for_terminal(exc, 200)}"
            ) from exc

    @classmethod
    def _from_dict(
        cls, data: Mapping[str, Any], max_nodes: int | None, validate: bool
    ) -> WorkflowGraph:
        if not isinstance(data, Mapping) or data.get("schema") != GRAPH_SCHEMA:
            raise GraphError("unsupported graph document")
        graph = cls(
            (NodeSpec.from_dict(n) for n in data["nodes"]),
            external_kinds=data.get("external_kinds", EXTERNAL_KINDS),
            max_nodes=max_nodes if max_nodes is not None else data.get("max_nodes", 200),
        )
        for record in data.get("expansions", ()):
            owner = record["owner"]
            if not isinstance(owner, str) or owner in graph._expansions:
                raise GraphError("malformed expansion record")
            members = tuple(graph.node(node_id) for node_id in record["nodes"])
            rewire = {t: Rewire.from_dict(r) for t, r in record.get("rewire", {}).items()}
            graph._expansions[owner] = PlanExpansion(owner, members, rewire)
        graph._touch()
        if validate:
            graph.validate()
        return graph

    # ------------------------------------------------------------------ rendering

    def to_mermaid(self, statuses: Mapping[str, NodeStatus | str] | None = None) -> str:
        """Mermaid flowchart. Node ids are generated (``n0``...), labels are reduced to a safe
        alphabet by ``sanitize.mermaid_label`` and no ``click`` directive is ever emitted, so
        agent-supplied titles cannot inject links or scripts."""
        ids = {node_id: mermaid_id(i) for i, node_id in enumerate(self._nodes)}
        lines = ["flowchart LR"]
        for node_id, spec in self._nodes.items():
            label = mermaid_label(f"{node_id}: {spec.title}")
            lines.append(f'  {ids[node_id]}["{label}"]')
        lines.extend(f"  {ids[dep]} --> {ids[n]}" for n, dep in self._edges())
        lines.extend(self._mermaid_status_lines(ids, statuses or {}))
        return "\n".join(lines) + "\n"

    def to_dot(self) -> str:
        """Graphviz DOT with generated ids and sanitised labels (same guarantees as Mermaid)."""
        ids = {node_id: mermaid_id(i) for i, node_id in enumerate(self._nodes)}
        lines = ["digraph workflow {", "  rankdir=LR;", "  node [shape=box];"]
        for node_id, spec in self._nodes.items():
            label = mermaid_label(f"{node_id}: {spec.title}")
            lines.append(f'  {ids[node_id]} [label="{label}"];')
        lines.extend(f"  {ids[dep]} -> {ids[n]};" for n, dep in self._edges())
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _edges(self) -> list[tuple[str, str]]:
        """``(node, dependency)`` pairs over known nodes, in declaration order."""
        idx = self._index()
        return [(n, d) for n in idx.order for d in idx.deps[n] if d in idx.pos]

    @staticmethod
    def _mermaid_status_lines(
        ids: Mapping[str, str], statuses: Mapping[str, NodeStatus | str]
    ) -> list[str]:
        by_status: dict[NodeStatus, list[str]] = {}
        for node_id, raw in statuses.items():
            try:
                status = NodeStatus(raw)
            except ValueError:
                continue
            if node_id in ids and status in _MERMAID_STATUS_STYLES:
                by_status.setdefault(status, []).append(ids[node_id])
        lines: list[str] = []
        for status, members in by_status.items():
            lines.append(f"  classDef st_{status.value} {_MERMAID_STATUS_STYLES[status]};")
            lines.append(f"  class {','.join(members)} st_{status.value};")
        return lines
