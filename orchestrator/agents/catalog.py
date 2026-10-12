"""Registry of capabilities the reference agents can deliver.

Each recipe module exposes a module-level ``CAPABILITY: Capability``. They are imported lazily
from the fixed list below (no directory scanning, no plugin loading: agents are trusted code
that lives in this repository).
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from orchestrator.agents.recipes.base import Capability, ParamSpec, RecipeParamError

_MODULES: tuple[str, ...] = (
    "orchestrator.agents.recipes.linkcheck",
    "orchestrator.agents.recipes.max_clicks",
    "orchestrator.agents.recipes.analytics_export",
)
MAX_MATCH_CHARS = 64 * 1024

_cache: dict[str, Capability] | None = None


def all_capabilities() -> dict[str, Capability]:
    global _cache  # noqa: PLW0603 - process-wide immutable registry, built once
    if _cache is None:
        caps: dict[str, Capability] = {}
        for name in _MODULES:
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError as exc:
                if exc.name == name:  # recipe not installed yet (partial build): skip
                    continue
                raise
            cap: Capability = module.CAPABILITY
            caps[cap.id] = cap
        _cache = caps
    return _cache


def get_capability(capability_id: str) -> Capability:
    try:
        return all_capabilities()[capability_id]
    except KeyError:
        raise KeyError(f"unknown capability {capability_id!r}; known: {sorted(all_capabilities())}") from None


@dataclass(frozen=True)
class CapabilityMatch:
    capability: Capability
    score: int
    matched_terms: tuple[str, ...]


def match_capabilities(text: str, *, min_score: int = 2) -> list[CapabilityMatch]:
    """Score every capability against ``text`` (deterministic; best first, ties by id).

    A capability matches when at least ``min_score`` distinct trigger patterns hit.
    """
    haystack = text[:MAX_MATCH_CHARS]
    matches: list[CapabilityMatch] = []
    for cap in all_capabilities().values():
        terms: list[str] = []
        for pattern in cap.triggers:
            m = re.search(pattern, haystack, flags=re.IGNORECASE)
            if m:
                terms.append(m.group(0).lower())
        if len(terms) >= min_score:
            matches.append(CapabilityMatch(cap, len(terms), tuple(sorted(set(terms)))))
    return sorted(matches, key=lambda m: (-m.score, m.capability.id))


def extract_params(cap: Capability, text: str) -> dict[str, Any]:
    """Extract typed recipe parameters from requirement/answer text via the capability's extractors.

    Values are coerced and **validated by the recipe's ParamSpec** (ranges, types); anything that
    fails raises ``RecipeParamError`` (fatal, surfaces to a human). Missing parameters fall back to
    ``ParamSpec.default``.
    """
    specs: Mapping[str, ParamSpec] = {p.name: p for p in cap.recipe.params}
    found: dict[str, Any] = {}
    haystack = text[:MAX_MATCH_CHARS]
    for name, pattern in cap.extractors.items():
        spec = specs.get(name)
        if spec is None:
            raise RecipeParamError(f"extractor for unknown parameter {name!r}")
        m = re.search(pattern, haystack, flags=re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1).replace(",", "").replace("_", "")
        if spec.kind == "int":
            found[name] = int(raw)
        elif spec.kind == "float":
            found[name] = float(raw)
        elif spec.kind == "bool":
            found[name] = raw.lower() in ("true", "yes", "1")
        else:
            found[name] = raw
    return found
