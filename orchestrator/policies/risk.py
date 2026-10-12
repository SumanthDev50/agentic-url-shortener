"""Change-set risk assessment (``PolicyEvaluator.assess_risk``): what a human approval should weigh.

``assess_changeset_risk`` is a pure, deterministic function of the change set and its baseline view:

* HIGH   - a database migration; an auth / security module; any file deletion
* MEDIUM - a new or changed HTTP route; more than ``min(10, budget.max_files)`` files; more than
  ``min(300, budget.max_lines)`` changed lines
* LOW    - everything else (documentation / tests only, or an ordinary code change)

The result is the maximum over the triggers.  ``reasons`` is a sorted-by-trigger list of short
sentences and ``signals`` a JSON-friendly dict (path lists are sorted and capped), so the same input
always yields the same assessment - it feeds the approval request and the evidence hash.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import pairwise

from orchestrator.core.paths import match_glob
from orchestrator.core.types import (
    ChangeSetSubject,
    FileOp,
    FileOpKind,
    RiskAssessment,
    RiskLevel,
)
from orchestrator.policies.base import changed_lines, read_or_empty
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.routes import routes_of_text

MEDIUM_FILES_THRESHOLD = 10
MEDIUM_LINES_THRESHOLD = 300
_LIST_CAP = 20

_DOC_TEST_GLOBS = (
    "docs/**",
    "**/*.md",
    "**/*.rst",
    "**/README*",
    "tests/**",
    "**/tests/**",
    "**/test_*.py",
    "**/*_test.py",
    "**/conftest.py",
)
_SECURITY_TOKENS = frozenset(
    [
        "auth",
        "authn",
        "authz",
        "authenticate",
        "authentication",
        "authorization",
        "authorize",
        "authorise",
        "security",
        "secure",
        "crypto",
        "cryptography",
        "secret",
        "secrets",
        "permission",
        "permissions",
        "rbac",
        "acl",
        "oauth",
        "jwt",
        "credential",
        "credentials",
        "password",
        "passwords",
        "passwd",
        "apikey",
        "apikeys",
        "hmac",
        "signing",
        "tls",
        "ssl",
        "cert",
        "certs",
    ]
)


def _is_doc_or_test(path: str) -> bool:
    return any(match_glob(path, g) for g in _DOC_TEST_GLOBS)


def _is_security_path(path: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", path.lower()) if t]
    if any(t in _SECURITY_TOKENS for t in tokens):
        return True
    return any(a == "api" and b in ("key", "keys") for a, b in pairwise(tokens))


def _route_keys(text: str, path: str, auth_names: frozenset[str]) -> set[tuple[str, str]]:
    routes, _ = routes_of_text(text, path, auth_names)
    return {(r.key, r.signature) for r in routes or []}


def _changed_route_keys(subject: ChangeSetSubject, auth_names: frozenset[str]) -> list[str]:
    changed: set[str] = set()
    for op in subject.changeset.ops:
        if not op.path.lower().endswith(".py"):
            continue
        before = _route_keys(read_or_empty(subject.baseline, op.path), op.path, auth_names)
        after = (
            set()
            if op.kind == FileOpKind.DELETE
            else _route_keys(op.content or "", op.path, auth_names)
        )
        changed.update(key for key, _ in after - before)
    return sorted(changed)


def _paths(ops: tuple[FileOp, ...], predicate: Callable[[FileOp], bool]) -> list[str]:
    return sorted(op.path for op in ops if predicate(op))[:_LIST_CAP]


def assess_changeset_risk(
    subject: ChangeSetSubject, config: PolicyConfig | None = None
) -> RiskAssessment:
    """Assess ``subject`` per the module docstring; never raises for a well-formed subject."""
    cfg = config if config is not None else PolicyConfig()
    ops = subject.changeset.ops
    paths = [op.path for op in ops]
    migrations = _paths(ops, lambda op: match_glob(op.path, cfg.migrations_glob))
    security = _paths(ops, lambda op: not _is_doc_or_test(op.path) and _is_security_path(op.path))
    deleted = _paths(ops, lambda op: op.kind == FileOpKind.DELETE)
    new_routes = _changed_route_keys(subject, cfg.auth_dependency_names)[:_LIST_CAP]
    lines = sum(changed_lines(subject.baseline, op) for op in ops)
    docs_tests_only = bool(paths) and all(_is_doc_or_test(p) for p in paths)

    high = [
        (migrations, "database migration"),
        (security, "authentication / security module changed"),
        (deleted, "file deletion"),
    ]
    medium = [
        (new_routes, "new or changed HTTP route"),
        (
            [f"{len(paths)} files"]
            if len(paths) > min(MEDIUM_FILES_THRESHOLD, cfg.change_budget.max_files)
            else [],
            "many files changed",
        ),
        (
            [f"{lines} lines"]
            if lines > min(MEDIUM_LINES_THRESHOLD, cfg.change_budget.max_lines)
            else [],
            "large diff",
        ),
    ]
    reasons = [f"{label}: {', '.join(found)}" for found, label in (*high, *medium) if found]
    level = (
        RiskLevel.HIGH
        if any(found for found, _ in high)
        else RiskLevel.MEDIUM
        if reasons
        else RiskLevel.LOW
    )
    if not reasons:
        reasons = [
            "documentation / tests only"
            if docs_tests_only
            else "ordinary code change"
            if paths
            else "empty change set"
        ]
    signals = {
        "files": len(paths),
        "lines": lines,
        "migrations": migrations,
        "security_paths": security,
        "deleted": deleted,
        "new_routes": new_routes,
        "docs_tests_only": docs_tests_only,
    }
    return RiskAssessment(level, tuple(reasons), signals)
