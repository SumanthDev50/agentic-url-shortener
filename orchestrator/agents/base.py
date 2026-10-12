"""Shared helpers for reference agents (L3).

Agents are *stateless*: the registry creates a fresh instance per attempt. They talk to the
world only through ``ctx.toolbox`` and return an ``AgentResult``. Payload builders here
validate against ``schemas.validate_payload`` — an invalid payload is an *agent bug* and is
raised as a FATAL, non-retryable ``AgentError`` so it surfaces immediately instead of being
retried into a loop.

Determinism rule (docs/artifact-schemas.md): never put wall-clock times, durations, pids,
random ids or absolute paths into artifact *content*; use ``metadata``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from orchestrator.core.errors import AgentError
from orchestrator.core.schemas import SCHEMA_VERSION, validate_payload
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    ArtifactDraft,
    AutonomyProfile,
    ChangeSet,
)

MAX_TEXT_CHARS = 64 * 1024

_WS = re.compile(r"\s+")


class BaseAgent:
    """Convenience base: subclasses set the class attributes and implement ``run``."""

    agent_id: str = "base"
    role: str = "base"
    version: str = "1"
    autonomy: AutonomyProfile = AutonomyProfile()

    def run(self, ctx: AgentContext) -> AgentResult:  # pragma: no cover - abstract
        raise NotImplementedError


def json_artifact(
    kind: str,
    payload: Mapping[str, Any],
    *,
    key: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactDraft:
    """Build a JSON artifact draft (adds ``schema: 1``) and validate it against its kind's shape."""
    body = dict(payload)
    body.setdefault("schema", SCHEMA_VERSION)
    problems = validate_payload(kind, body)
    if problems:
        raise AgentError(
            f"agent produced an invalid {kind} payload: {'; '.join(problems[:5])}",
            kind="fatal",
            retryable=False,
            details={"problems": problems[:20]},
        )
    return ArtifactDraft(kind=kind, content=body, key=key, metadata=dict(metadata or {}))


def text_artifact(
    kind: str,
    text: str,
    *,
    media_type: str = "text/markdown",
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactDraft:
    return ArtifactDraft(kind=kind, content=text, media_type=media_type, metadata=dict(metadata or {}))


def changeset_artifact(kind: str, cs: ChangeSet) -> ArtifactDraft:
    """Change sets are stored as ``ChangeSet.to_dict()``; the engine forces the key and the author."""
    return ArtifactDraft(kind=kind, content=cs.to_dict())


# ----------------------------------------------------------------------------- ids


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def stable_hex(*parts: str, n: int = 6) -> str:
    h = hashlib.sha1("\x1f".join(normalize_text(p) for p in parts).encode("utf-8"), usedforsecurity=False)
    return h.hexdigest()[:n]


def question_id(category: str, text: str) -> str:
    """Content-derived id: answers survive amendments and re-runs."""
    return f"Q-{category}-{stable_hex(category, text)}"


def ambiguity_id(category: str, span: str) -> str:
    return f"AMB-{stable_hex(category, span)}"


def req_id(i: int) -> str:
    return f"R-{i:03d}"


def ac_id(i: int, j: int) -> str:
    return f"AC-{i:03d}-{j}"


def task_id(i: int) -> str:
    return f"T-{i:03d}"


def adr_id(i: int) -> str:
    return f"ADR-{i:03d}"


def risk_id(i: int) -> str:
    return f"RISK-{i:03d}"


def asm_id(i: int) -> str:
    return f"ASM-{i:03d}"


def split_sentences(text: str) -> list[str]:
    """Split on sentence/bullet boundaries (deterministic, linear-time)."""
    text = text[:MAX_TEXT_CHARS]
    out: list[str] = []
    for block in re.split(r"\n\s*\n|\n\s*[-*•]\s+|\n\s*\d+[.)]\s+", text):
        for part in re.split(r"(?<=[.!?])\s+", block.replace("\n", " ")):
            part = _WS.sub(" ", part).strip(" -*•\t")
            if part:
                out.append(part)
    return out
