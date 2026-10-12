"""Lineage: how an artifact came to be, across both graphs the run records.

Two relations make up the picture:

* **data lineage** - ``Artifact.inputs`` (the dispatch-time input snapshot of the attempt that
  produced it), and
* **decision lineage** - the decisions that list the artifact in their ``outputs``, followed
  recursively through ``Decision.parents``.

``agentic lineage <artifact>`` walks both, so it can reach, for example, the CLARIFICATION
decision that shaped a release patch through ``release_patch -> ... -> requirements ->
clarification_answers <- D-0007``.

The graph is a DAG (inputs are earlier artifacts, parents earlier decisions) but a tampered
store could contain cycles or absurd fan-out, and artifact/decision text is untrusted. So every
traversal keeps a visited set, the rendered tree is bounded in depth and node count, every
label is control-stripped and length-capped, and a node reached a second time is shown once
as ``already shown`` instead of being expanded again (a diamond must not render exponentially).
Traversals are iterative where they are unbounded (``reaches``, ``downstream``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from orchestrator.core.artifacts import ArtifactStore
from orchestrator.core.decisions import DecisionLog
from orchestrator.core.errors import ConfigError, StateError
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.types import Artifact, ArtifactStatus, DecisionKind

MAX_DEPTH = 64
MAX_NODES = 2000
MAX_LABEL_CHARS = 200

ARTIFACT = "artifact"
DECISION = "decision"

_RELATION_TEXT = {"input": "input", "decision": "decided by", "parent": "parent"}
_NOTE_TEXT = {
    "seen": "already shown above",
    "missing": "missing from the store",
    "hash-mismatch": "content hash differs from the recorded reference",
    "truncated": "not expanded: lineage limit reached",
}


@dataclass(frozen=True)
class LineageNode:
    """One vertex of the rendered upstream tree (an artifact or a decision)."""

    kind: str  # ARTIFACT | DECISION
    id: str
    relation: str  # to the parent node: "root" | "input" | "decision" | "parent"
    label: str  # single line, sanitised, length-capped
    note: str = ""  # "" | "seen" | "missing" | "hash-mismatch" | "truncated"
    children: tuple[LineageNode, ...] = ()

    def walk(self) -> Iterator[LineageNode]:
        """Pre-order traversal of this subtree (iterative)."""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "id": self.id,
            "relation": self.relation,
            "label": self.label,
            "note": self.note,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass(frozen=True)
class _Edge:
    relation: str
    kind: str
    id: str
    content_hash: str | None = None  # what the referencing side recorded (artifact edges)


class LineageGraph:
    """Read-only view over an :class:`ArtifactStore` and a :class:`DecisionLog`."""

    def __init__(self, store: ArtifactStore, decisions: DecisionLog) -> None:
        self._store = store
        self._decisions = decisions

    # ------------------------------------------------------------------ graph access

    def _edges(self, kind: str, ident: str) -> list[_Edge]:
        """Upstream neighbours of a node (empty for a node that no longer exists)."""
        if kind == ARTIFACT:
            try:
                art = self._store.get(ident)
            except StateError:
                return []
            edges = [_Edge("input", ARTIFACT, r.artifact_id, r.content_hash) for r in art.inputs]
            edges.extend(
                _Edge("decision", DECISION, d.id) for d in self._decisions.decisions_for(art.ref)
            )
            return edges
        try:
            decision = self._decisions.get(ident)
        except StateError:
            return []
        return [_Edge("parent", DECISION, parent) for parent in decision.parents]

    def _describe(self, edge: _Edge) -> tuple[str, str]:
        """``(label, note)`` of the node an edge points at."""
        if edge.kind == ARTIFACT:
            try:
                art = self._store.get(edge.id)
            except StateError:
                return sanitize_for_terminal(f"artifact {edge.id}", MAX_LABEL_CHARS), "missing"
            label = (
                f"artifact {art.id} [{art.status.value}] producer={art.producer} "
                f"sha256={art.content_hash[:12]}"
            )
            mismatch = edge.content_hash is not None and edge.content_hash != art.content_hash
            return sanitize_for_terminal(
                label, MAX_LABEL_CHARS
            ), "hash-mismatch" if mismatch else ""
        try:
            decision = self._decisions.get(edge.id)
        except StateError:
            return sanitize_for_terminal(f"decision {edge.id}", MAX_LABEL_CHARS), "missing"
        label = (
            f"decision {decision.id} [{decision.kind.value}] {decision.summary} "
            f"(by {decision.actor.label})"
        )
        return sanitize_for_terminal(label, MAX_LABEL_CHARS), ""

    # ------------------------------------------------------------------ queries

    def upstream(self, artifact_id: str) -> LineageNode:
        """Everything that fed into ``artifact_id``: its inputs, the decisions that produced it
        and, recursively, their parents. Unknown ids raise :class:`StateError`."""
        root = self._store.get(artifact_id)
        seen = {(ARTIFACT, root.id)}
        budget = [MAX_NODES]
        return self._expand(_Edge("root", ARTIFACT, root.id), 0, seen, budget)

    def _expand(
        self, edge: _Edge, depth: int, seen: set[tuple[str, str]], budget: list[int]
    ) -> LineageNode:
        label, note = self._describe(edge)
        budget[0] -= 1
        if note == "missing":
            return LineageNode(edge.kind, edge.id, edge.relation, label, note)
        children: list[LineageNode] = []
        for child in self._edges(edge.kind, edge.id):
            key = (child.kind, child.id)
            if key in seen:
                child_label, child_note = self._describe(child)
                budget[0] -= 1
                children.append(
                    LineageNode(
                        child.kind, child.id, child.relation, child_label, child_note or "seen"
                    )
                )
            elif depth + 1 > MAX_DEPTH or budget[0] <= 0:
                child_label, _ = self._describe(child)
                children.append(
                    LineageNode(child.kind, child.id, child.relation, child_label, "truncated")
                )
            else:
                seen.add(key)
                children.append(self._expand(child, depth + 1, seen, budget))
        return LineageNode(edge.kind, edge.id, edge.relation, label, note, tuple(children))

    def downstream(self, artifact_id: str) -> list[Artifact]:
        """Published artifacts (ACTIVE or SUPERSEDED) derived, directly or transitively, from
        ``artifact_id``, sorted by ``(kind, key, version)``. STAGED and QUARANTINED artifacts
        were never part of the published record and are excluded."""
        root = self._store.get(artifact_id)
        published = (ArtifactStatus.ACTIVE, ArtifactStatus.SUPERSEDED)
        dependents: dict[str, list[Artifact]] = {}
        for art in self._store.all(include_inactive=True):
            if art.status in published:
                for ref in art.inputs:
                    dependents.setdefault(ref.artifact_id, []).append(art)
        found: dict[str, Artifact] = {}
        queue = deque([root.id])
        while queue:
            for art in dependents.get(queue.popleft(), ()):
                if art.id != root.id and art.id not in found:
                    found[art.id] = art
                    queue.append(art.id)
        return sorted(found.values(), key=lambda a: (a.kind, a.key, a.version))

    def _kind_of(self, decision_id: str) -> DecisionKind | None:
        return self._decisions.get(decision_id).kind if decision_id in self._decisions else None

    def reaches(self, artifact_id: str, decision_kind: DecisionKind | str) -> bool:
        """True iff some decision of ``decision_kind`` is upstream of ``artifact_id``."""
        try:
            wanted = DecisionKind(decision_kind)
        except ValueError as exc:
            raise ConfigError(
                f"unknown decision kind: {sanitize_for_terminal(decision_kind, 40)}"
            ) from exc
        root = self._store.get(artifact_id)
        seen = {(ARTIFACT, root.id)}
        queue = deque([_Edge("root", ARTIFACT, root.id)])
        while queue:
            edge = queue.popleft()
            if edge.kind == DECISION and self._kind_of(edge.id) == wanted:
                return True
            for child in self._edges(edge.kind, edge.id):
                key = (child.kind, child.id)
                if key not in seen:
                    seen.add(key)
                    queue.append(child)
        return False

    def explain(self, artifact_id: str) -> str:
        """Indented, human-readable upstream tree (every label sanitised and capped)."""
        lines: list[str] = []
        stack = [(self.upstream(artifact_id), 0)]
        while stack:
            node, depth = stack.pop()
            prefix = "" if node.relation == "root" else f"{_RELATION_TEXT[node.relation]}: "
            note = f"  ({_NOTE_TEXT[node.note]})" if node.note else ""
            lines.append(f"{'  ' * depth}{prefix}{node.label}{note}")
            stack.extend((child, depth + 1) for child in reversed(node.children))
        return "\n".join(lines)

    def to_dict(self, artifact_id: str) -> dict[str, Any]:
        """JSON-ready lineage of ``artifact_id`` (upstream tree and downstream refs)."""
        return {
            "artifact_id": artifact_id,
            "upstream": self.upstream(artifact_id).to_dict(),
            "downstream": [a.ref.to_dict() for a in self.downstream(artifact_id)],
        }
