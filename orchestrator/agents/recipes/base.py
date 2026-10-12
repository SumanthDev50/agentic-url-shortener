"""Recipe and capability protocol (the offline reference agents' *knowledge base*).

A **capability** is something the reference agents know how to deliver end to end. It bundles
(1) how to recognise it in a requirement (``triggers``), (2) which *aspects* (testable facets)
it has and the acceptance criterion template of each, (3) how to extract typed parameters from
the requirement text, and (4) a **recipe**: the vetted, parameterised code/test/doc generation.

Honesty: the reference agents are deterministic stand-ins for model-backed agents. A requirement
that matches no capability is reported as ``unsupported`` and goes to a human; it is never guessed.

Security (docs/module-reference.md L3): recipe parameters come from untrusted text. They are
**typed and range-checked before any recipe runs** (``validate_params``); templates are static
text with typed holes; values are rendered with ``repr``/``json.dumps`` and never f-stringed into
SQL, identifiers or paths; free-form strings are not accepted at all.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from orchestrator.core.errors import AgentError
from orchestrator.core.types import RiskLevel, Toolbox

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]{0,31}$")


class RecipeParamError(AgentError):
    """A recipe parameter is missing, mistyped or out of range (fatal, never retried)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="fatal", retryable=False)


@dataclass(frozen=True)
class ParamSpec:
    """Typed, bounded recipe parameter (``kind`` is one of int/float/bool/enum/ident)."""

    name: str
    kind: str
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    description: str = ""
    required: bool = False

    def validate(self, value: Any) -> Any:
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise RecipeParamError(f"{self.name}: expected int")
            self._range(value)
            return value
        if self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecipeParamError(f"{self.name}: expected number")
            self._range(float(value))
            return float(value)
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise RecipeParamError(f"{self.name}: expected bool")
            return value
        if self.kind == "enum":
            if value not in self.choices:
                raise RecipeParamError(f"{self.name}: must be one of {list(self.choices)}")
            return value
        if self.kind == "ident":
            if not isinstance(value, str) or not _IDENT_RE.match(value) or keyword.iskeyword(value):
                raise RecipeParamError(f"{self.name}: invalid identifier")
            return value
        raise RecipeParamError(f"{self.name}: unknown parameter kind {self.kind!r}")  # pragma: no cover

    def _range(self, value: float) -> None:
        if self.minimum is not None and value < self.minimum:
            raise RecipeParamError(f"{self.name}: must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise RecipeParamError(f"{self.name}: must be <= {self.maximum}")


def validate_params(specs: Sequence[ParamSpec], params: Mapping[str, Any]) -> dict[str, Any]:
    """Reject unknown keys, apply defaults, type/range-check everything."""
    known = {s.name for s in specs}
    unknown = sorted(set(params) - known)
    if unknown:
        raise RecipeParamError(f"unknown recipe parameter(s): {unknown}")
    out: dict[str, Any] = {}
    for spec in specs:
        if spec.name in params:
            out[spec.name] = spec.validate(params[spec.name])
        elif spec.required:
            raise RecipeParamError(f"missing required parameter: {spec.name}")
        else:
            out[spec.name] = spec.default
    return out


@dataclass(frozen=True)
class AspectSpec:
    """A testable facet of a capability; becomes an acceptance criterion and a test."""

    id: str  # e.g. "timeout"
    title: str
    triggers: tuple[str, ...]  # linear-time regexes (case-insensitive) matched per requirement sentence
    criterion: str  # template, formatted with the validated params (str.format_map; params only)
    verification: str = "test"  # "test" | "inspection"


@dataclass(frozen=True)
class GroupSpec:
    """One unit of parallel work => one ``implement.<id>`` node."""

    id: str  # [a-z][a-z0-9_-]{0,30}
    title: str
    write_scope: tuple[str, ...]  # globs the group may write (narrows the agent's autonomy)
    expected_paths: tuple[str, ...]  # what the group intends to touch (CHG-003 anchors / impact delta)
    depends_on: tuple[str, ...] = ()  # other group ids (build order; must not be cyclic)


@dataclass
class BuildContext:
    """Everything a recipe may use. All side effects go through ``toolbox`` (staged writes)."""

    toolbox: Toolbox
    params: Mapping[str, Any]
    model: Any | None = None  # CodebaseModel for brownfield capabilities
    requirements: Mapping[str, Any] | None = None
    design: Mapping[str, Any] | None = None
    anchors: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Recipe(Protocol):
    id: str
    params: tuple[ParamSpec, ...]

    def groups(self, params: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
        """Decomposition into parallel groups (≥ 1)."""
        ...

    def design(self, params: Mapping[str, Any], model: Any | None) -> Mapping[str, Any]:
        """Design contributions: ``components``, ``adrs`` (with alternatives), ``data_model_changes``,
        ``api_changes``, ``rollback_plan`` (keys of ``Kind.DESIGN_DOC``). Must be derived from the
        codebase model where one is given (brownfield) so the design reflects the real code."""
        ...

    def build_group(self, group_id: str, ctx: BuildContext) -> None:
        """Stage the files of one group through ``ctx.toolbox``. Missing anchors in the baseline must
        raise ``RecipeAnchorError`` (fatal, not retried): never guess."""
        ...

    def build_tests(self, ctx: BuildContext) -> None:
        """Stage the tests. Every test function carries a docstring line ``Covers: R-001, AC-001-2``
        derived from ``ctx.requirements`` aspects (traceability convention)."""
        ...

    def build_docs(self, ctx: BuildContext) -> None:
        """Stage feature documentation / changelog entries from real design and parameters."""
        ...


class RecipeAnchorError(AgentError):
    """The baseline no longer has the code location a recipe expects (drift): fail safe, ask a human."""

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="fatal", retryable=False)


@dataclass(frozen=True)
class Capability:
    id: str
    title: str
    kind: str  # "new_module" | "enhancement"
    triggers: tuple[str, ...]  # linear-time regexes (case-insensitive); each hit adds to the match score
    aspects: tuple[AspectSpec, ...]
    extractors: Mapping[str, str]  # param name -> regex with ONE capture group over requirements+answers text
    impact_keywords: tuple[str, ...]
    risk: RiskLevel
    recipe: Recipe
    requires_baseline: bool = False  # True: needs existing code (brownfield analysis keywords)
