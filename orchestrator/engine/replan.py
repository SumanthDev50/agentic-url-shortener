"""Re-planning building blocks: node fingerprints, the invalidation table and remediation targets.

Everything in this module is **pure**: it reads the graph / node states it is handed and returns
intents. ``Scheduler.transition`` is the only code that changes a node status (engine-semantics
3.1).

* ``compute_fingerprint`` implements section 4.2. It is sensitive to the node spec, the agent id
  and version of the current epoch, the epoch, the effective dependencies (rewires included), the
  pinned input snapshot (``kind``, ``key``, ``content_hash`` of every visible artifact; a declared
  kind with no artifact is recorded as absent), the baseline and policy hashes, the remediation
  feedback keys and the engine schema version. It is *not* sensitive to artifact metadata, ids or
  versions, to the order of the inputs or to the order / repetition of the feedback keys.
* ``invalidate`` implements the table of section 12.2 for every ``NodeStatus``. The affected set is
  the graph consumers of the changed artifact kinds plus the strict descendants of the changed
  producer nodes. Changed or rewired expansion nodes are re-planned by the caller through the
  producers / kinds it passes in. Producers that are not (or no longer) nodes of the graph and nodes
  without a ``NodeState`` are ignored. Returned transitions are always legal under
  ``types.ALLOWED_TRANSITIONS``; intents that cancel in-flight work come first, then the rest, each
  group in topological order.
* ``resolve_remediation_targets`` resolves a ``RemediationSpec`` against the live graph (9.2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from orchestrator.core.artifacts import PinnedReader
from orchestrator.core.errors import GraphError
from orchestrator.core.graph import WorkflowGraph
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.state import NodeState
from orchestrator.core.types import Epoch, NodeSpec, NodeStatus, RemediationSpec

_MAX_CAUSES = 3
_REASON_MAX = 200
_PAST_FAILURES = (
    NodeStatus.FAILED,
    NodeStatus.BLOCKED,
    NodeStatus.CANCELLED,
    NodeStatus.ROLLED_BACK,
)


@dataclass(frozen=True)
class Transition:
    """An intent to move ``node_id`` to ``to``; the flags tell the scheduler what else to do.

    ``bump_epoch`` -> ``NodeState.epoch_no += 1``; ``cancel_running`` -> trip the cancel token and
    ``generation += 1`` (the late result is discarded); ``quarantine_staged`` -> quarantine the
    staged artifacts of the held / in-flight attempt; ``cancel_approval`` -> cancel the node's
    pending approval request.
    """

    node_id: str
    to: NodeStatus
    reason: str
    bump_epoch: bool = False
    cancel_running: bool = False
    quarantine_staged: bool = False
    cancel_approval: bool = False


# --------------------------------------------------------------------------- fingerprint


def compute_fingerprint(
    node: NodeSpec,
    graph: WorkflowGraph,
    pinned: PinnedReader,
    *,
    agent_version: str,
    epoch: Epoch,
    baseline_hash: str,
    policy_hash: str,
    feedback_keys: Sequence[str],
    engine_version: int,
) -> str:
    """The input fingerprint of ``node`` for one attempt (engine-semantics 4.2)."""
    epoch = Epoch(epoch)
    agent_id = (
        node.fallback_agent if epoch == Epoch.FALLBACK and node.fallback_agent else node.agent
    )
    inputs: list[list[str | None]] = [
        [ref.kind, ref.key, ref.content_hash] for ref in pinned.refs()
    ]
    inputs.extend([kind, "", None] for kind in pinned.absent_kinds)
    inputs.sort(key=lambda entry: (entry[0] or "", entry[1] or "", entry[2] or ""))
    feedback = sorted({str(key) for key in feedback_keys})
    return hash_jsonable(
        {
            "spec": node.spec_hash(),
            "agent": [agent_id, agent_version],
            "epoch": epoch.value,
            "deps": sorted(graph.deps(node.id)),
            "inputs": inputs,
            "baseline": baseline_hash,
            "policy": policy_hash,
            "feedback": hash_jsonable(feedback) if feedback else "",
            "engine": engine_version,
        }
    )


# --------------------------------------------------------------------------- invalidation


def _why(causes: Sequence[str]) -> str:
    shown = ", ".join(causes[:_MAX_CAUSES])
    more = f" (+{len(causes) - _MAX_CAUSES} more)" if len(causes) > _MAX_CAUSES else ""
    return sanitize_for_terminal(shown + more, _REASON_MAX)


def _transition(node_id: str, status: NodeStatus, why: str, wake: bool) -> Transition | None:
    """The row of the section 12.2 table for ``status`` (``None`` = unchanged)."""
    if status == NodeStatus.SUCCEEDED:
        return Transition(node_id, NodeStatus.STALE, f"stale: {why}")
    if status == NodeStatus.SKIPPED:
        return Transition(node_id, NodeStatus.PENDING, f"skip condition re-evaluated: {why}")
    if status == NodeStatus.READY:
        return Transition(node_id, NodeStatus.PENDING, f"reset: {why}")
    if status == NodeStatus.RUNNING:
        return Transition(
            node_id,
            NodeStatus.PENDING,
            f"in-flight attempt cancelled: {why}",
            bump_epoch=True,
            cancel_running=True,
            quarantine_staged=True,
        )
    if status == NodeStatus.AWAITING_APPROVAL:
        return Transition(
            node_id,
            NodeStatus.PENDING,
            f"approval cancelled: {why}",
            bump_epoch=True,
            cancel_approval=True,
            quarantine_staged=True,
        )
    if status == NodeStatus.AWAITING_INPUT:
        # the held output is obsolete either way (section 11): the answers re-run the node
        if wake:
            return Transition(
                node_id, NodeStatus.READY, f"answers received: {why}", quarantine_staged=True
            )
        return Transition(
            node_id,
            NodeStatus.PENDING,
            f"inputs changed while waiting: {why}",
            quarantine_staged=True,
        )
    if status in _PAST_FAILURES:
        return Transition(node_id, NodeStatus.PENDING, f"reset: {why}", bump_epoch=True)
    return None  # PENDING (nothing to do), STALE and REMOVED stay as they are


def invalidate(
    graph: WorkflowGraph,
    nodes: Mapping[str, NodeState],
    changed_kinds: Iterable[str],
    changed_producers: Iterable[str] = (),
    *,
    wake_kind: str | None = None,
) -> list[Transition]:
    """Pure implementation of the engine-semantics 12.2 table (see the module docstring).

    ``wake_kind`` names the artifact kind whose arrival wakes a node that waits for human input
    (``clarification_answers``): an ``AWAITING_INPUT`` node goes straight to ``READY`` when that
    kind is among ``changed_kinds`` and back to ``PENDING`` otherwise.
    """
    kinds = sorted({k for k in changed_kinds if isinstance(k, str)})
    producers = sorted({p for p in changed_producers if isinstance(p, str)})
    causes: dict[str, list[str]] = {}
    for kind in kinds:
        label = f"input kind {sanitize_for_terminal(kind, 60)}"
        for node_id in graph.consumers_of(kind):
            causes.setdefault(node_id, []).append(label)
    for producer in producers:
        if producer not in graph:
            continue
        label = f"upstream node {sanitize_for_terminal(producer, 60)}"
        for node_id in graph.descendants(producer):
            causes.setdefault(node_id, []).append(label)
    wake = wake_kind is not None and wake_kind in kinds
    cancelling: list[Transition] = []
    others: list[Transition] = []
    for node_id in graph.topological_order():
        state = nodes.get(node_id)
        if node_id not in causes or state is None:
            continue
        intent = _transition(node_id, state.status, _why(causes[node_id]), wake)
        if intent is not None:
            (cancelling if intent.cancel_running else others).append(intent)
    return [*cancelling, *others]


# --------------------------------------------------------------------------- remediation


def resolve_remediation_targets(graph: WorkflowGraph, spec: RemediationSpec) -> list[str]:
    """Node ids named by ``spec.route_to`` (exact ids and ``prefix.*`` patterns) in the live graph.

    Order: ``route_to`` order, then graph order within a pattern; each node once. A malformed
    pattern resolves to nothing instead of failing the scheduler tick (graph validation already
    rejects such specs at start).
    """
    targets: dict[str, None] = {}
    for entry in spec.route_to:
        try:
            matches = graph.resolve_pattern(entry)
        except GraphError:
            continue
        targets.update(dict.fromkeys(matches))
    return list(targets)


__all__ = [
    "Transition",
    "compute_fingerprint",
    "invalidate",
    "resolve_remediation_targets",
]
