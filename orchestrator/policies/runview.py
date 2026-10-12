"""Evidence access for run-level rules (CMP-001, CMP-003, CHG-005).

A ``RunSubject`` is evaluated *at a gate position* (``node_id``): only artifacts produced by
**strict ancestors** of that node (plus ``external`` inputs and what the gated attempt itself
staged) count as evidence.  A sibling, a descendant or a stale artifact can therefore never satisfy
a release check. An empty ``node_id`` means "no position": the whole run is considered, which is the
stricter reading (more evidence is required, nothing is exempted).

Every accessor degrades to "absent" (``None`` / empty) on an unknown node or an unreadable payload;
the calling rule then reports missing evidence, i.e. fails closed.
"""

from __future__ import annotations

from typing import Any

from orchestrator.core.types import EXTERNAL_PRODUCER, Artifact, ArtifactStatus, RunSubject


def considered_producers(subject: RunSubject) -> frozenset[str]:
    """Producer ids whose artifacts count as evidence at this gate position."""
    view = subject.view
    try:
        nodes = (
            frozenset(view.node_specs()) if not subject.node_id else view.ancestors(subject.node_id)
        )
    except (KeyError, ValueError):
        nodes = frozenset()
    return nodes | {EXTERNAL_PRODUCER}


def latest_evidence(subject: RunSubject, kind: str) -> Artifact | None:
    """The ACTIVE artifact of ``kind`` from an eligible producer, else one the gate staged."""
    view = subject.view
    eligible = considered_producers(subject)
    active = view.latest(kind)
    if active is not None and active.producer in eligible:
        return active
    if not subject.node_id:
        return None
    staged = [
        a
        for a in view.artifacts(include_inactive=True)
        if a.kind == kind and a.status == ArtifactStatus.STAGED and a.producer == subject.node_id
    ]
    return max(staged, key=lambda a: a.version, default=None)


def read_payload(subject: RunSubject, artifact: Artifact | None) -> Any | None:
    """Decoded JSON payload, or ``None`` when absent / unreadable / not an object."""
    if artifact is None:
        return None
    try:
        payload = subject.view.read_json(artifact)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
