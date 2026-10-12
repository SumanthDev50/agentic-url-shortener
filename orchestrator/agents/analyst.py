"""Analyst agent: static impact analysis of the existing codebase (read-only, deterministic).

``analyze_codebase`` runs over the toolbox view itself (the toolbox satisfies ``FileView``), so
every file read is autonomy-checked and audited, and ``impact_analysis`` turns the resulting model
plus keywords into the ``impact_report`` artifact (``ImpactReport.to_dict()``: sorted, no
timestamps, no absolute paths).

Keywords, in this order and bounded (:data:`MAX_NOUNS` requirement tokens):

1. ``Capability.impact_keywords`` of every capability the ``requirements`` artifact says is
   delivered (precise: written by the recipe author against the real product);
2. identifier-like tokens of the *active* requirement texts: backticked names, ``snake_case`` and
   ``CamelCase`` words, which point at code the requirement names explicitly;
3. only when no capability contributes keywords (nothing is delivered yet): plain nouns of the
   active requirement texts, i.e. words of at least four letters that are not generic
   (:data:`_GENERIC`). They are noisy (``code`` matches every error code), so they never widen a
   curated keyword list.

Heuristic limits are those of ``tools.codebase`` (name-based call resolution, ``unresolved`` entries
instead of guesses). The requirement text is only tokenised, never interpreted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from orchestrator.agents import catalog
from orchestrator.agents.base import BaseAgent, json_artifact
from orchestrator.core.errors import AgentError
from orchestrator.core.schemas import Kind
from orchestrator.core.types import AgentContext, AgentResult, AutonomyProfile
from orchestrator.tools.codebase import (
    MAX_PY_FILES,
    MAX_SQL_FILES,
    analyze_codebase,
    impact_analysis,
)

MAX_NOUNS = 12
MAX_TEXT_CHARS = 1000  # per requirement text: the tokeniser never sees more

_WORD = re.compile(r"[a-z][a-z0-9_]{3,}")
_IDENTIFIER = re.compile(
    r"`([A-Za-z_][\w.]{1,40})`|\b([a-z]+(?:_[a-z0-9]+)+)\b|\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b"
)
_GENERIC = frozenset(
    [
        "this",
        "that",
        "these",
        "those",
        "with",
        "without",
        "from",
        "have",
        "must",
        "should",
        "shall",
        "will",
        "would",
        "could",
        "need",
        "needs",
        "needed",
        "want",
        "wants",
        "also",
        "into",
        "than",
        "then",
        "they",
        "them",
        "their",
        "when",
        "what",
        "which",
        "while",
        "where",
        "each",
        "every",
        "only",
        "over",
        "under",
        "after",
        "before",
        "between",
        "within",
        "about",
        "more",
        "most",
        "less",
        "such",
        "same",
        "other",
        "both",
        "some",
        "any",
        "all",
        "and",
        "the",
        "for",
        "are",
        "was",
        "were",
        "been",
        "being",
        "has",
        "had",
        "does",
        "did",
        "not",
        "but",
        "can",
        "may",
        "might",
        "never",
        "always",
        "system",
        "service",
        "request",
        "requests",
        "response",
        "responses",
        "client",
        "clients",
        "existing",
        "keep",
        "working",
        "result",
        "results",
        "user",
        "users",
        "data",
        "value",
        "values",
        "field",
        "fields",
        "support",
        "provide",
        "allow",
        "ensure",
        "make",
        "return",
        "returns",
        "returned",
        "able",
        "used",
        "using",
        "time",
        "times",
        "new",
        "old",
        "one",
        "two",
        "three",
        "maybe",
        "perhaps",
        "possibly",
        "better",
        "faster",
        "fast",
        "safer",
        "kind",
        "sort",
        "things",
        "thing",
        "nice",
        "soon",
        "creating",
        "followed",
        "further",
        "optionally",
        "supply",
    ]
)


def _nouns(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for word in _WORD.findall(text[:MAX_TEXT_CHARS].lower()):
        if word not in _GENERIC:
            seen.setdefault(word, None)
    return list(seen)


def _identifiers(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _IDENTIFIER.finditer(text[:MAX_TEXT_CHARS]):
        seen.setdefault(next(g for g in match.groups() if g).lower(), None)
    return list(seen)


def _keywords(requirements: Mapping[str, Any]) -> list[str]:
    words: dict[str, None] = {}
    for entry in requirements.get("capabilities", ()):
        capability_id = entry.get("id")
        try:
            capability = catalog.get_capability(str(capability_id))
        except KeyError as exc:
            raise AgentError(
                f"the requirements name an unknown capability: {str(capability_id)[:40]!r}",
                kind="fatal",
                retryable=False,
            ) from exc
        for keyword in capability.impact_keywords:
            words.setdefault(keyword, None)
    curated = bool(words)
    identifiers: dict[str, None] = {}
    nouns: dict[str, None] = {}
    for req in requirements.get("requirements", ()):
        if req.get("status") == "active" and isinstance(req.get("text"), str):
            for identifier in _identifiers(req["text"]):
                identifiers.setdefault(identifier, None)
            if not curated:
                for noun in _nouns(req["text"]):
                    nouns.setdefault(noun, None)
    for token in [*list(identifiers)[:MAX_NOUNS], *list(nouns)[:MAX_NOUNS]]:
        words.setdefault(token, None)
    return list(words)


class AnalystAgent(BaseAgent):
    agent_id = "analyst"
    role = "analyst"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=(),
        can_execute=False,
        max_tool_calls=MAX_PY_FILES
        + MAX_SQL_FILES
        + 16,  # one read per analysable file + the listings
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        requirements = ctx.json(Kind.REQUIREMENTS)
        keywords = _keywords(requirements)
        model = analyze_codebase(ctx.toolbox)
        report = impact_analysis(model, keywords)
        return AgentResult(
            artifacts=[json_artifact(Kind.IMPACT_REPORT, report.to_dict())],
            rationale=(
                f"{len(model.modules)} module(s) analysed, {len(report.modules)} impacted by "
                f"{len(report.keywords)} keyword(s)."
            ),
            usage=dict(ctx.toolbox.usage()),
        )
