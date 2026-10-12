"""assess_changeset_risk: migrations/auth/deletions HIGH, routes/size MEDIUM, docs+tests LOW."""

from __future__ import annotations

import json

import pytest

from orchestrator.core.serde import canonical_json, hash_jsonable
from orchestrator.core.types import RiskLevel, ToolAction
from orchestrator.policies.config import ChangeBudget, PolicyConfig
from orchestrator.policies.engine import PolicyEngine
from orchestrator.policies.risk import assess_changeset_risk
from tests.orchestrator.policies.policy_builders import (
    action_subject,
    artifact_subject,
    create,
    cs_subject,
    delete,
    modify,
    view,
)

ROUTE = (
    "from fastapi import APIRouter\nrouter = APIRouter()\n@router.get('/x')\ndef h():\n    "
    "return 1\n"
)


def test_empty_change_set_is_low() -> None:
    result = assess_changeset_risk(cs_subject([]))
    assert result.level == RiskLevel.LOW
    assert result.reasons == ("empty change set",)
    assert result.signals["files"] == 0
    assert result.signals["docs_tests_only"] is False


@pytest.mark.parametrize(
    "path",
    [
        "docs/guide.md",
        "README.md",
        "docs/api/index.rst",
        "tests/test_x.py",
        "pkg/tests/test_y.py",
        "tests/conftest.py",
        "test_top.py",
        "pkg/mod_test.py",
        "CHANGELOG.md",
    ],
)
def test_docs_and_tests_only_are_low(path: str) -> None:
    result = assess_changeset_risk(cs_subject([create(path, "x\n")]))
    assert result.level == RiskLevel.LOW
    assert result.reasons == ("documentation / tests only",)
    assert result.signals["docs_tests_only"] is True


def test_ordinary_code_change_is_low() -> None:
    result = assess_changeset_risk(cs_subject([create("shortener/domain/codes.py", "x = 1\n")]))
    assert result.level == RiskLevel.LOW
    assert result.reasons == ("ordinary code change",)
    assert result.signals["docs_tests_only"] is False


def test_migrations_are_high() -> None:
    result = assess_changeset_risk(
        cs_subject([create("shortener/storage/migrations/0003_x.sql", "select 1;\n")])
    )
    assert result.level == RiskLevel.HIGH
    assert result.signals["migrations"] == ["shortener/storage/migrations/0003_x.sql"]
    assert any("database migration" in r for r in result.reasons)


@pytest.mark.parametrize(
    "path",
    [
        "shortener/api/auth.py",
        "shortener/security/keys.py",
        "shortener/api/middleware/authentication.py",
        "shortener/api/permissions.py",
        "shortener/core/crypto.py",
        "shortener/api_keys.py",
        "shortener/api/api-key-store.py",
        "shortener/jwt_helpers.py",
        "shortener/settings/secrets.py",
        "shortener/password_policy.py",
    ],
)
def test_auth_and_security_modules_are_high(path: str) -> None:
    result = assess_changeset_risk(cs_subject([create(path, "x = 1\n")]))
    assert result.level == RiskLevel.HIGH, path
    assert result.signals["security_paths"] == [path]


@pytest.mark.parametrize(
    "path",
    [
        "shortener/authors.py",
        "docs/authentication.md",
        "tests/test_auth.py",
        "shortener/api/links.py",
        "shortener/apikeyless.py",
    ],
)
def test_look_alikes_and_docs_or_tests_about_auth_are_not_security_modules(path: str) -> None:
    result = assess_changeset_risk(cs_subject([create(path, "x = 1\n")]))
    assert result.level == RiskLevel.LOW, path
    assert result.signals["security_paths"] == []


def test_any_deletion_is_high() -> None:
    base = view({"docs/old.md": "x\n"})
    result = assess_changeset_risk(cs_subject([delete(base, "docs/old.md")], base))
    assert result.level == RiskLevel.HIGH  # even docs: deletions are never silent
    assert result.signals["deleted"] == ["docs/old.md"]


def test_new_or_changed_routes_are_medium() -> None:
    result = assess_changeset_risk(cs_subject([create("shortener/api/new.py", ROUTE)]))
    assert result.level == RiskLevel.MEDIUM
    assert result.signals["new_routes"] == ["GET /x"]
    base = view({"api.py": ROUTE})
    renamed = ROUTE.replace("'/x'", "'/y'")
    assert assess_changeset_risk(cs_subject([modify(base, "api.py", renamed)], base)).signals[
        "new_routes"
    ] == ["GET /y"]


def test_unchanged_routes_do_not_raise_risk() -> None:
    base = view({"api.py": ROUTE})
    body_only = ROUTE.replace("return 1", "return 2")
    result = assess_changeset_risk(cs_subject([modify(base, "api.py", body_only)], base))
    assert result.level == RiskLevel.LOW
    assert result.signals["new_routes"] == []


def test_many_files_and_large_diffs_are_medium_with_boundaries() -> None:
    ten = [create(f"shortener/m{i}.py", "x = 1\n") for i in range(10)]
    assert assess_changeset_risk(cs_subject(ten)).level == RiskLevel.LOW
    eleven = [*ten, create("shortener/m10.py", "x = 1\n")]
    result = assess_changeset_risk(cs_subject(eleven))
    assert result.level == RiskLevel.MEDIUM
    assert any("many files" in r for r in result.reasons)
    assert (
        assess_changeset_risk(cs_subject([create("shortener/a.py", "x = 1\n" * 300)])).level
        == RiskLevel.LOW
    )
    large = assess_changeset_risk(cs_subject([create("shortener/a.py", "x = 1\n" * 301)]))
    assert large.level == RiskLevel.MEDIUM
    assert large.signals["lines"] == 301


def test_config_budget_lowers_the_medium_thresholds() -> None:
    cfg = PolicyConfig(change_budget=ChangeBudget(max_files=2, max_lines=20))
    three = [create(f"shortener/m{i}.py", "x = 1\n") for i in range(3)]
    assert assess_changeset_risk(cs_subject(three), cfg).level == RiskLevel.MEDIUM
    assert (
        assess_changeset_risk(cs_subject([create("shortener/a.py", "x\n" * 21)]), cfg).level
        == RiskLevel.MEDIUM
    )


def test_the_highest_trigger_wins_and_all_reasons_are_listed() -> None:
    base = view({"old.py": "x\n"})
    ops = [
        create("shortener/storage/migrations/0004.sql", "select 1;\n"),
        create("shortener/api/new.py", ROUTE),
        delete(base, "old.py"),
    ]
    result = assess_changeset_risk(cs_subject(ops, base))
    assert result.level == RiskLevel.HIGH
    assert len(result.reasons) == 3
    assert result.reasons[0].startswith(
        "database migration"
    )  # fixed trigger order => deterministic text


def test_docs_only_with_other_triggers_does_not_hide_them() -> None:
    docs = [create(f"docs/p{i}.md", "x\n") for i in range(11)]
    result = assess_changeset_risk(cs_subject(docs))
    assert result.level == RiskLevel.MEDIUM
    assert result.signals["docs_tests_only"] is True


def test_assessment_is_deterministic_and_serialisable() -> None:
    ops = [create(f"shortener/storage/migrations/{i:04d}.sql", "select 1;\n") for i in range(30)]
    first = assess_changeset_risk(cs_subject(ops))
    second = assess_changeset_risk(cs_subject(list(reversed(ops))))
    assert first.to_dict() == second.to_dict()
    assert first.signals["migrations"] == sorted(first.signals["migrations"])
    assert len(first.signals["migrations"]) == 20  # capped
    json.dumps(first.to_dict())
    assert hash_jsonable(first) == hash_jsonable(first.to_dict())
    assert canonical_json(first.signals)


def test_engine_delegates_and_non_changesets_are_low() -> None:
    engine = PolicyEngine()
    subject = cs_subject([create("shortener/storage/migrations/0003_x.sql", "select 1;\n")])
    assert engine.assess_risk(subject) == assess_changeset_risk(subject, engine.config)
    other = engine.assess_risk(artifact_subject("hi"))
    assert other.level == RiskLevel.LOW
    assert engine.assess_risk(action_subject(ToolAction.READ, "a.py")).level == RiskLevel.LOW
