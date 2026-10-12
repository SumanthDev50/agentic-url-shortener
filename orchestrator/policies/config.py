"""Policy configuration with floors: a config can make the policy stricter, never weaker.

* ``CATALOGUE`` is the manifest of every rule (id, category, default severity); the ``PolicyEngine``
  asserts at construction that its rule set equals it, and ``tests`` assert it equals the table in
  ``docs/architecture.md`` section 1.5.
* ``FLOORS`` is the lowest severity a rule may ever be configured to (SEC-*, CHG-001/006/007/008 and
  QLT-001 are HIGH or above; SEC-006 is the one MEDIUM exception because the catalogue defines it as
  a human-confirmation rule).  Overrides are **raise-only for every rule** (``ConfigError``
  otherwise) and a finding's severity is ``max(finding, floor, override)``, so CRITICAL can never be
  lowered.
* Everything is validated at construction (hostile or typo'd config fails loudly) and the whole
  object is hashable through ``config_hash()``, which the engine records in ``RUN_CREATED`` and in
  node fingerprints.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from orchestrator.core.errors import ConfigError, PathViolation
from orchestrator.core.paths import RULE_ID_RE, match_any, validate_globs
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import PolicyCategory, Severity


@dataclass(frozen=True)
class CatalogueEntry:
    category: PolicyCategory
    severity: Severity


_SEC, _CMP, _CHG, _QLT = (
    PolicyCategory.SECURITY,
    PolicyCategory.COMPLIANCE,
    PolicyCategory.CHANGE_CONTROL,
    PolicyCategory.QUALITY,
)
_H, _M = Severity.HIGH, Severity.MEDIUM

CATALOGUE: Mapping[str, CatalogueEntry] = {
    "SEC-001": CatalogueEntry(_SEC, _H),
    "SEC-002": CatalogueEntry(_SEC, _H),
    "SEC-003": CatalogueEntry(_SEC, _H),
    "SEC-004": CatalogueEntry(_SEC, _H),
    "SEC-005": CatalogueEntry(_SEC, _H),
    "SEC-006": CatalogueEntry(_SEC, _M),
    "SEC-007": CatalogueEntry(_SEC, _H),
    "SEC-008": CatalogueEntry(_SEC, _H),
    "CMP-001": CatalogueEntry(_CMP, _H),
    "CMP-002": CatalogueEntry(_CMP, _M),
    "CMP-003": CatalogueEntry(_CMP, _H),
    "CHG-001": CatalogueEntry(_CHG, _H),
    "CHG-002": CatalogueEntry(_CHG, _M),
    "CHG-003": CatalogueEntry(_CHG, _M),
    "CHG-004": CatalogueEntry(_CHG, _H),
    "CHG-005": CatalogueEntry(_CHG, _H),
    "CHG-006": CatalogueEntry(_CHG, _H),
    "CHG-007": CatalogueEntry(_CHG, _H),
    "CHG-008": CatalogueEntry(_CHG, _H),
    "QLT-001": CatalogueEntry(_QLT, _H),
}

FLOORS: Mapping[str, Severity] = {
    rule_id: (Severity.MEDIUM if rule_id == "SEC-006" else Severity.HIGH)
    for rule_id in CATALOGUE
    if rule_id.startswith(("SEC-", "QLT-"))
    or rule_id in ("CHG-001", "CHG-006", "CHG-007", "CHG-008")
}

DEFAULT_PUBLIC_ROUTES: frozenset[str] = frozenset({"GET /{code}", "GET /healthz", "GET /readyz"})
DEFAULT_AUTH_DEPENDENCIES: frozenset[str] = frozenset(
    {"require_api_key", "get_current_key", "authenticate"}
)
DEFAULT_MIGRATIONS_GLOB = "**/migrations/*.sql"
DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    ".github/**",
    "Dockerfile",
    "pyproject.toml",
    "requirements*.txt",
    ".env",
    ".env.*",
    "**/*.pem",
    "**/*.key",
)
DEFAULT_REWIRE_TARGETS: tuple[str, ...] = ("integrate", "implement.*")

_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS", "ANY"})
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_NODE_PATTERN_RE = re.compile(r"^[a-z][a-z0-9_.*-]{0,60}\Z")


def _int_in(value: object, low: int, high: int) -> bool:
    """A real ``int`` (``True`` is not a budget) within ``[low, high]``."""
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


@dataclass(frozen=True)
class ChangeBudget:
    """CHG-002 budget: a change set above either bound needs a human waiver."""

    max_files: int = 25
    max_lines: int = 800

    def __post_init__(self) -> None:
        if not _int_in(self.max_files, 1, 500):
            raise ConfigError("change_budget.max_files must be an integer in [1, 500]")
        if not _int_in(self.max_lines, 1, 100_000):
            raise ConfigError("change_budget.max_lines must be an integer in [1, 100000]")

    def to_dict(self) -> dict[str, int]:
        return {"max_files": self.max_files, "max_lines": self.max_lines}


@dataclass(frozen=True)
class ExpansionRules:
    """CHG-007 limits the owner template does not carry: size and allowed rewire targets."""

    max_new_nodes: int = 40
    rewire_targets: tuple[str, ...] = DEFAULT_REWIRE_TARGETS  # node ids or ``prefix.*`` patterns

    def __post_init__(self) -> None:
        if not _int_in(self.max_new_nodes, 0, 200):
            raise ConfigError("expansion_rules.max_new_nodes must be an integer in [0, 200]")
        object.__setattr__(self, "rewire_targets", tuple(self.rewire_targets))
        for pattern in self.rewire_targets:
            if not isinstance(pattern, str) or not _NODE_PATTERN_RE.match(pattern):
                raise ConfigError(f"invalid rewire target pattern: {pattern!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"max_new_nodes": self.max_new_nodes, "rewire_targets": list(self.rewire_targets)}


def _normalise_route(route: str) -> str:
    method, _, path = route.partition(" ")
    method = method.upper()
    if method not in _HTTP_METHODS or not path.startswith("/") or len(path) > 200 or " " in path:
        raise ConfigError(f"invalid public route {route!r} (expected 'METHOD /path')")
    return f"{method} {path}"


def _parse_severity(value: Any) -> Severity:
    if isinstance(value, Severity):
        return value
    try:
        return Severity[value] if isinstance(value, str) else Severity(value)
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"invalid severity: {value!r}") from exc


def _globs(patterns: Iterable[str], field_name: str) -> tuple[str, ...]:
    try:
        return validate_globs(tuple(patterns))
    except (PathViolation, TypeError) as exc:
        raise ConfigError(f"{field_name}: invalid glob ({exc})") from exc


@dataclass(frozen=True)
class PolicyConfig:
    """Immutable, validated policy parameters (see module docstring for the floor semantics)."""

    severity_overrides: Mapping[str, Severity] = field(default_factory=dict)
    public_routes: frozenset[str] = DEFAULT_PUBLIC_ROUTES
    auth_dependency_names: frozenset[str] = DEFAULT_AUTH_DEPENDENCIES
    migrations_glob: str = DEFAULT_MIGRATIONS_GLOB
    change_budget: ChangeBudget = field(default_factory=ChangeBudget)
    protected_paths: tuple[str, ...] = DEFAULT_PROTECTED_PATHS
    # Globs of modules allowed to import egress/execution libraries (SEC-008). Empty by default:
    # fail closed,
    # so a project must opt each networking/subprocess module in explicitly.
    egress_modules: tuple[str, ...] = ()
    expansion_rules: ExpansionRules = field(default_factory=ExpansionRules)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "severity_overrides", self._checked_overrides(self.severity_overrides)
        )
        object.__setattr__(
            self, "public_routes", frozenset(_normalise_route(r) for r in self.public_routes)
        )
        names = frozenset(self.auth_dependency_names)
        if not all(isinstance(n, str) and _IDENT_RE.match(n) for n in names):
            raise ConfigError("auth_dependency_names must be Python identifiers")
        object.__setattr__(self, "auth_dependency_names", names)
        object.__setattr__(
            self, "migrations_glob", _globs([self.migrations_glob], "migrations_glob")[0]
        )
        protected = _globs(self.protected_paths, "protected_paths")
        missing = [p for p in DEFAULT_PROTECTED_PATHS if p not in protected]
        if missing:
            raise ConfigError(
                f"protected_paths may only be extended, never reduced (missing: {missing})"
            )
        object.__setattr__(self, "protected_paths", protected)
        object.__setattr__(self, "egress_modules", _globs(self.egress_modules, "egress_modules"))

    @staticmethod
    def _checked_overrides(overrides: Mapping[str, Any]) -> dict[str, Severity]:
        checked: dict[str, Severity] = {}
        for rule_id, raw in overrides.items():
            if (
                not isinstance(rule_id, str)
                or not RULE_ID_RE.match(rule_id)
                or rule_id not in CATALOGUE
            ):
                raise ConfigError(f"severity override for unknown rule: {rule_id!r}")
            severity = _parse_severity(raw)
            floor = FLOORS.get(rule_id, CATALOGUE[rule_id].severity)
            if severity < floor or severity < CATALOGUE[rule_id].severity:
                raise ConfigError(
                    f"severity overrides are raise-only: {rule_id} cannot go below "
                    f"{max(floor, CATALOGUE[rule_id].severity).name} (got {severity.name})"
                )
            checked[rule_id] = severity
        return checked

    # -- behaviour ---------------------------------------------------------------
    def effective_severity(self, rule_id: str, emitted: Severity) -> Severity:
        """``max(emitted, floor, override)``: a finding is never reported below its floor or its
        override.
        """
        floor = FLOORS.get(rule_id, Severity.INFO)
        return max(emitted, floor, self.severity_overrides.get(rule_id, Severity.INFO))

    def is_public_route(self, method: str, path: str) -> bool:
        return (
            f"{method.upper()} {path}" in self.public_routes or f"ANY {path}" in self.public_routes
        )

    def is_egress_module(self, path: str) -> bool:
        return match_any(path, self.egress_modules)

    # -- serialisation -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "severity_overrides": {k: v.name for k, v in sorted(self.severity_overrides.items())},
            "public_routes": sorted(self.public_routes),
            "auth_dependency_names": sorted(self.auth_dependency_names),
            "migrations_glob": self.migrations_glob,
            "change_budget": self.change_budget.to_dict(),
            "protected_paths": list(self.protected_paths),
            "egress_modules": list(self.egress_modules),
            "expansion_rules": self.expansion_rules.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyConfig:
        """Strict inverse of ``to_dict``; unknown keys are an error (a typo must not silently weaken
        policy).
        """
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown policy config keys: {sorted(unknown)}")
        default = cls()
        budget = data.get("change_budget")
        rules = data.get("expansion_rules")
        try:
            return cls(
                severity_overrides=dict(data.get("severity_overrides", {})),
                public_routes=frozenset(data.get("public_routes", default.public_routes)),
                auth_dependency_names=frozenset(
                    data.get("auth_dependency_names", default.auth_dependency_names)
                ),
                migrations_glob=data.get("migrations_glob", default.migrations_glob),
                change_budget=ChangeBudget(**budget)
                if budget is not None
                else default.change_budget,
                protected_paths=tuple(data.get("protected_paths", default.protected_paths)),
                egress_modules=tuple(data.get("egress_modules", default.egress_modules)),
                expansion_rules=ExpansionRules(**rules)
                if rules is not None
                else default.expansion_rules,
            )
        except TypeError as exc:
            raise ConfigError(f"malformed policy config: {exc}") from exc

    def config_hash(self) -> str:
        return hash_jsonable(self.to_dict())
