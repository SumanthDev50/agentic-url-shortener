"""PolicyConfig: defaults, floors (raise-only overrides), hostile config, hashing."""

from __future__ import annotations

import pytest

from orchestrator.core.errors import ConfigError
from orchestrator.core.types import Severity
from orchestrator.policies.config import (
    CATALOGUE,
    DEFAULT_PROTECTED_PATHS,
    FLOORS,
    ChangeBudget,
    ExpansionRules,
    PolicyConfig,
)

_PROTECTED_RULES = {"CHG-001", "CHG-006", "CHG-007", "CHG-008"}


def test_defaults_match_the_documented_values() -> None:
    cfg = PolicyConfig()
    assert cfg.public_routes == {"GET /{code}", "GET /healthz", "GET /readyz"}
    assert cfg.auth_dependency_names == {"require_api_key", "get_current_key", "authenticate"}
    assert cfg.migrations_glob == "**/migrations/*.sql"
    assert (cfg.change_budget.max_files, cfg.change_budget.max_lines) == (25, 800)
    assert cfg.protected_paths == (
        ".github/**",
        "Dockerfile",
        "pyproject.toml",
        "requirements*.txt",
        ".env",
        ".env.*",
        "**/*.pem",
        "**/*.key",
    )
    assert cfg.egress_modules == ()  # fail closed: nothing may open sockets unless opted in
    assert cfg.severity_overrides == {}
    assert "implement.*" in cfg.expansion_rules.rewire_targets


def test_catalogue_covers_every_family_and_floors_are_consistent() -> None:
    assert len(CATALOGUE) == 20
    assert set(FLOORS) <= set(CATALOGUE)
    for rule_id in CATALOGUE:
        if rule_id.startswith(("SEC-", "QLT-")) or rule_id in _PROTECTED_RULES:
            assert rule_id in FLOORS
    assert FLOORS["SEC-006"] == Severity.MEDIUM  # the catalogue defines it as human confirmation
    assert all(FLOORS[r] == Severity.HIGH for r in FLOORS if r != "SEC-006")


@pytest.mark.parametrize("rule_id", sorted(FLOORS))
def test_floored_rules_cannot_be_lowered(rule_id: str) -> None:
    below = Severity(FLOORS[rule_id] - 1)
    with pytest.raises(ConfigError, match="raise-only"):
        PolicyConfig(severity_overrides={rule_id: below})
    with pytest.raises(ConfigError, match="raise-only"):
        PolicyConfig(severity_overrides={rule_id: Severity.INFO})


@pytest.mark.parametrize(
    "rule_id", ["CMP-002", "CHG-002", "CHG-003", "CMP-001", "CHG-004", "CHG-005"]
)
def test_overrides_are_raise_only_for_every_rule(rule_id: str) -> None:
    default = CATALOGUE[rule_id].severity
    with pytest.raises(ConfigError):
        PolicyConfig(severity_overrides={rule_id: Severity.LOW})
    assert (
        PolicyConfig(severity_overrides={rule_id: default}).severity_overrides[rule_id] == default
    )
    raised = PolicyConfig(severity_overrides={rule_id: Severity.CRITICAL})
    assert raised.effective_severity(rule_id, Severity.LOW) == Severity.CRITICAL


def test_raising_is_allowed_and_accepts_names_and_ints() -> None:
    cfg = PolicyConfig(severity_overrides={"CMP-002": "HIGH", "SEC-006": 3})  # type: ignore[dict-item]
    assert cfg.severity_overrides == {"CMP-002": Severity.HIGH, "SEC-006": Severity.HIGH}


@pytest.mark.parametrize("bad", ["LOUD", 9, None, 1.5, ["HIGH"]])
def test_invalid_severity_values_are_rejected(bad: object) -> None:
    with pytest.raises(ConfigError):
        PolicyConfig(severity_overrides={"CMP-002": bad})  # type: ignore[dict-item]


@pytest.mark.parametrize("rule_id", ["XXX-999", "SEC-1", "sec-001", "", "POL-000", "SEC-001 "])
def test_override_for_unknown_rule_is_rejected(rule_id: str) -> None:
    with pytest.raises(ConfigError, match="unknown rule"):
        PolicyConfig(severity_overrides={rule_id: Severity.CRITICAL})


def test_effective_severity_never_lowers_and_honours_the_floor() -> None:
    cfg = PolicyConfig(severity_overrides={"SEC-002": Severity.CRITICAL})
    assert cfg.effective_severity("SEC-002", Severity.HIGH) == Severity.CRITICAL
    assert cfg.effective_severity("SEC-001", Severity.CRITICAL) == Severity.CRITICAL
    assert cfg.effective_severity("SEC-001", Severity.LOW) == Severity.HIGH  # floor lifts emission
    assert cfg.effective_severity("CMP-002", Severity.LOW) == Severity.LOW  # unfloored, no override
    assert cfg.effective_severity("POL-000", Severity.INFO) == Severity.INFO


@pytest.mark.parametrize(
    "route", ["GET", "FETCH /x", "GET x", "GET /a b", "GET /" + "x" * 300, "", "get", "/x"]
)
def test_invalid_public_routes_are_rejected(route: str) -> None:
    with pytest.raises(ConfigError, match="public route"):
        PolicyConfig(public_routes=frozenset({route}))


def test_public_routes_are_normalised_and_matched() -> None:
    cfg = PolicyConfig(public_routes=frozenset({"get /ping", "ANY /status"}))
    assert cfg.is_public_route("GET", "/ping")
    assert cfg.is_public_route("get", "/ping")
    assert not cfg.is_public_route("POST", "/ping")
    assert cfg.is_public_route("DELETE", "/status")  # ANY covers every method
    assert not cfg.is_public_route("GET", "/{code}")


@pytest.mark.parametrize("name", ["bad name", "1abc", "a-b", "", "x" * 80, "a.b", "ünïcode"])
def test_auth_dependency_names_must_be_identifiers(name: str) -> None:
    with pytest.raises(ConfigError, match="identifiers"):
        PolicyConfig(auth_dependency_names=frozenset({name}))


@pytest.mark.parametrize("glob", ["../x/*.sql", "/abs/*.sql", "a\\b", "", "a//b", "x" * 300])
def test_unsafe_globs_are_rejected_everywhere(glob: str) -> None:
    with pytest.raises(ConfigError):
        PolicyConfig(migrations_glob=glob)
    with pytest.raises(ConfigError):
        PolicyConfig(egress_modules=(glob,))
    with pytest.raises(ConfigError):
        PolicyConfig(protected_paths=(*DEFAULT_PROTECTED_PATHS, glob))


def test_protected_paths_can_be_extended_but_never_reduced() -> None:
    extended = PolicyConfig(protected_paths=(*DEFAULT_PROTECTED_PATHS, "deploy/**"))
    assert "deploy/**" in extended.protected_paths
    for victim in DEFAULT_PROTECTED_PATHS:
        reduced = tuple(p for p in DEFAULT_PROTECTED_PATHS if p != victim)
        with pytest.raises(ConfigError, match="extended, never reduced"):
            PolicyConfig(protected_paths=reduced)
    with pytest.raises(ConfigError):
        PolicyConfig(protected_paths=())


@pytest.mark.parametrize(
    ("files", "lines"), [(0, 10), (501, 10), (10, 0), (10, 100_001), (-1, -1), (True, 5)]
)
def test_change_budget_bounds(files: int, lines: int) -> None:
    with pytest.raises(ConfigError):
        ChangeBudget(files, lines)


def test_change_budget_accepts_the_limits() -> None:
    assert ChangeBudget(1, 1).max_files == 1
    assert ChangeBudget(500, 100_000).max_lines == 100_000


def test_expansion_rules_validation() -> None:
    assert ExpansionRules(0, ()).max_new_nodes == 0
    with pytest.raises(ConfigError):
        ExpansionRules(201)
    with pytest.raises(ConfigError):
        ExpansionRules(5, ("Not A Pattern",))
    with pytest.raises(ConfigError):
        ExpansionRules(5, ("../x",))


def test_egress_module_matching_uses_the_safe_glob_matcher() -> None:
    cfg = PolicyConfig(egress_modules=("shortener/health/**", "tools/fetch.py"))
    assert cfg.is_egress_module("shortener/health/check.py")
    assert cfg.is_egress_module("tools/fetch.py")
    assert not cfg.is_egress_module("evil/shortener/health/check.py")  # anchored at both ends
    assert not PolicyConfig().is_egress_module("anything.py")


def test_round_trip_and_hash_stability() -> None:
    cfg = PolicyConfig(
        severity_overrides={"CMP-002": Severity.HIGH},
        public_routes=frozenset({"GET /a", "POST /b"}),
        change_budget=ChangeBudget(10, 100),
        egress_modules=("a/**",),
        expansion_rules=ExpansionRules(3, ("integrate",)),
    )
    again = PolicyConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.config_hash() == cfg.config_hash()
    assert len(cfg.config_hash()) == 64
    assert cfg.config_hash() != PolicyConfig().config_hash()


def test_hash_changes_with_every_field() -> None:
    base = PolicyConfig().config_hash()
    variants = [
        PolicyConfig(severity_overrides={"CMP-002": Severity.HIGH}),
        PolicyConfig(public_routes=frozenset({"GET /x"})),
        PolicyConfig(auth_dependency_names=frozenset({"other"})),
        PolicyConfig(migrations_glob="db/*.sql"),
        PolicyConfig(change_budget=ChangeBudget(24, 800)),
        PolicyConfig(protected_paths=(*DEFAULT_PROTECTED_PATHS, "x/**")),
        PolicyConfig(egress_modules=("a.py",)),
        PolicyConfig(expansion_rules=ExpansionRules(1)),
    ]
    hashes = {v.config_hash() for v in variants}
    assert base not in hashes
    assert len(hashes) == len(variants)


def test_to_dict_is_deterministic_and_json_friendly() -> None:
    first = PolicyConfig(public_routes=frozenset({"GET /b", "GET /a"})).to_dict()
    second = PolicyConfig(public_routes=frozenset({"GET /a", "GET /b"})).to_dict()
    assert first == second
    assert first["public_routes"] == ["GET /a", "GET /b"]


def test_from_dict_rejects_unknown_keys_and_wrong_types() -> None:
    with pytest.raises(ConfigError, match="unknown policy config keys"):
        PolicyConfig.from_dict({"public_route": ["GET /x"]})  # a typo must not weaken policy
    with pytest.raises(ConfigError, match="malformed"):
        PolicyConfig.from_dict({"change_budget": {"max_files": 5, "bogus": 1}})
    with pytest.raises(ConfigError, match="malformed"):
        PolicyConfig.from_dict({"expansion_rules": "nope"})
    with pytest.raises(ConfigError):
        PolicyConfig.from_dict({"severity_overrides": {"SEC-001": "LOW"}})


def test_from_dict_defaults_for_missing_keys() -> None:
    assert PolicyConfig.from_dict({}) == PolicyConfig()


def test_config_is_immutable() -> None:
    cfg = PolicyConfig()
    with pytest.raises(AttributeError):
        cfg.migrations_glob = "x"  # type: ignore[misc]
