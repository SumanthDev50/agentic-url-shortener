"""Artifact kinds, identifier grammars and payload shapes (the inter-agent data contract).

Every JSON artifact payload carries ``"schema": 1``. ``validate_payload(kind, obj)`` returns a
list of human-readable problems (empty = valid); the ``json_shape`` gate uses it, and agents use
it in their own tests. The authoritative prose description lives in ``docs/artifact-schemas.md``.

**Determinism rule:** payloads must not contain wall-clock times, durations, pids, random ids or
absolute paths; put those in ``ArtifactDraft.metadata`` (excluded from content hash and
fingerprint). This is what makes fingerprint memoisation and re-plan reuse work.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from orchestrator.core.types import CHANGESET_KINDS, EXTERNAL_KINDS, KEYED_KINDS, ChangeSet

SCHEMA_VERSION = 1


class Kind:
    """Artifact kind names (one place; never use string literals elsewhere)."""

    REQUEST = "request"
    CLARIFICATION_ANSWERS = "clarification_answers"
    INTAKE_REPORT = "intake_report"
    REQUIREMENTS = "requirements"
    IMPACT_REPORT = "impact_report"
    THREAT_MODEL = "threat_model"
    DESIGN_DOC = "design_doc"
    API_CONTRACT = "api_contract"
    TASK_PLAN = "task_plan"
    PLAN_EXPANSION = "plan_expansion"
    CHANGESET = "changeset"
    INTEGRATED_CHANGESET = "integrated_changeset"
    INTEGRATION_REPORT = "integration_report"
    TEST_REPORT = "test_report"
    SECURITY_REPORT = "security_report"
    DOCS_CHANGESET = "docs_changeset"
    READINESS_REPORT = "readiness_report"
    TRACEABILITY_MATRIX = "traceability_matrix"
    RELEASE_PATCH = "release_patch"
    RELEASE_BUNDLE = "release_bundle"
    RELEASE_VERIFICATION = "release_verification"
    ENGINEERING_SUMMARY = "engineering_summary"


ALL_KINDS = frozenset(v for k, v in vars(Kind).items() if not k.startswith("_"))
TEXT_KINDS = frozenset({Kind.REQUEST, Kind.RELEASE_PATCH, Kind.ENGINEERING_SUMMARY})

__all__ = [
    "ALL_KINDS",
    "CHANGESET_KINDS",
    "EXTERNAL_KINDS",
    "KEYED_KINDS",
    "SCHEMA_VERSION",
    "TEXT_KINDS",
    "Kind",
    "validate_payload",
]

# ---------------------------------------------------------------------------- id grammars
# ``\Z`` (not ``$``): ``$`` would also accept a trailing line feed.

REQ_ID_RE = re.compile(r"^R-\d{3}\Z")
AC_ID_RE = re.compile(r"^AC-\d{3}-\d{1,2}\Z")
TASK_ID_RE = re.compile(r"^T-\d{3}\Z")
ADR_ID_RE = re.compile(r"^ADR-\d{3}\Z")
RISK_ID_RE = re.compile(r"^RISK-\d{3}\Z")
ASM_ID_RE = re.compile(r"^ASM-\d{3}\Z")
AMB_ID_RE = re.compile(r"^AMB-[0-9a-f]{6}\Z")
QUESTION_ID_RE = re.compile(r"^Q-[a-z_]{2,20}-[0-9a-f]{6}\Z")
GROUP_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}\Z")

# ---------------------------------------------------------------------------- tiny shape language


class Enum:
    def __init__(self, *values: str) -> None:
        self.values = values


class Opt:
    """Optional key marker wrapper: value may be absent or None."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec


class Re:
    def __init__(self, pattern: re.Pattern[str]) -> None:
        self.pattern = pattern


def _check(spec: Any, value: Any, path: str, errors: list[str]) -> None:
    if isinstance(spec, Opt):
        if value is not None:
            _check(spec.spec, value, path, errors)
        return
    if spec is Any:
        return
    if isinstance(spec, Enum):
        if value not in spec.values:
            errors.append(f"{path}: expected one of {list(spec.values)}, got {value!r}")
        return
    if isinstance(spec, Re):
        if not isinstance(value, str) or not spec.pattern.match(value):
            errors.append(f"{path}: {value!r} does not match {spec.pattern.pattern}")
        return
    if isinstance(spec, type):
        ok = isinstance(value, spec)
        if spec is int and isinstance(value, bool):
            ok = False
        if spec is float and isinstance(value, int) and not isinstance(value, bool):
            ok = True
        if not ok:
            errors.append(f"{path}: expected {spec.__name__}, got {type(value).__name__}")
        return
    if isinstance(spec, list):
        if not isinstance(value, list):
            errors.append(f"{path}: expected list, got {type(value).__name__}")
            return
        for i, item in enumerate(value):
            _check(spec[0], item, f"{path}[{i}]", errors)
        return
    if isinstance(spec, dict):
        if not isinstance(value, Mapping):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
            return
        for key, sub in spec.items():
            if key not in value:
                if not isinstance(sub, Opt):
                    errors.append(f"{path}.{key}: missing required key")
                continue
            _check(sub, value[key], f"{path}.{key}", errors)
        return
    raise TypeError(f"bad shape spec: {spec!r}")  # pragma: no cover


_FINDING = {
    "rule_id": str,
    "category": str,
    "severity": Enum("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"),
    "message": str,
    "location": Opt(str),
    "evidence": Opt(str),
    "remediation": Opt(str),
}
_QUESTION = {
    "id": Re(QUESTION_ID_RE),
    "text": str,
    "rationale": str,
    "options": [str],
    "default_assumption": Opt(str),
    "blocking": bool,
    "related_requirements": [str],
    "category": str,
}
_WORKSPACE_KIND = Enum("greenfield", "brownfield")
_CHANGE_TYPE = Enum("feature", "enhancement", "bugfix", "refactor", "docs", "test_improvement", "unknown")

SHAPES: dict[str, Any] = {
    Kind.CLARIFICATION_ANSWERS: {
        "schema": int,
        "answers": Any,  # {question_id: {"answer": str, "by": str, "round": int}} (checked below)
    },
    Kind.INTAKE_REPORT: {
        "schema": int,
        "workspace": {"file_count": int, "python_files": int, "is_empty": bool, "manifest_hash": str},
        "classification": {"workspace_kind": _WORKSPACE_KIND, "change_type": _CHANGE_TYPE},
        "request": {"chars": int, "lines": int},
        "input_findings": [_FINDING],
    },
    Kind.REQUIREMENTS: {
        "schema": int,
        "problem": {"goal": str, "in_scope": [str], "out_of_scope": [str], "constraints": [str]},
        "classification": {"workspace_kind": _WORKSPACE_KIND, "change_type": _CHANGE_TYPE},
        "requirements": [
            {
                "id": Re(REQ_ID_RE),
                "text": str,
                "kind": Enum("functional", "non_functional", "constraint"),
                "priority": Enum("must", "should", "could"),
                "status": Enum("active", "deferred", "descoped"),
                "status_decision": Opt(str),
                "capabilities": [str],
                "parameters": dict,
                "verification": Enum("test", "inspection"),
                "acceptance_criteria": [{"id": Re(AC_ID_RE), "text": str}],
            }
        ],
        "assumptions": [{"id": Re(ASM_ID_RE), "text": str, "question_id": Opt(str)}],
        "ambiguities": [
            {
                "id": Re(AMB_ID_RE),
                "category": str,
                "span": str,
                "question_id": Re(QUESTION_ID_RE),
                "disposition": Enum("open", "answered", "assumed", "deferred"),
                "answer": Opt(str),
            }
        ],
        "open_questions": [_QUESTION],
        "unsupported": [{"requirement_id": str, "reason": str}],
        "capabilities": [{"id": str, "title": str, "requirement_ids": [str], "params": dict}],
    },
    Kind.IMPACT_REPORT: {
        "schema": int,
        "keywords": [str],
        "modules": [{"module": str, "path": str, "reason": str, "role": Enum("seed", "dependent")}],
        "routes": [{"method": str, "path": str, "handler": str, "module": str, "line": int, "auth": [str]}],
        "tables": [{"table": str, "columns": [str], "migration": str}],
        "data_flows": [
            {"route": str, "handler": str, "calls": [str], "repositories": [str], "tables": [str], "unresolved": [str]}
        ],
        "tests": [str],
        "coupling_risks": [{"component": str, "reason": str}],
        "risk_signals": [str],
        "expected_paths": [str],
    },
    Kind.THREAT_MODEL: {
        "schema": int,
        "assets": [str],
        "trust_boundaries": [str],
        "risks": [
            {
                "id": Re(RISK_ID_RE),
                "category": Enum("spoofing", "tampering", "repudiation", "information_disclosure", "denial_of_service", "elevation_of_privilege"),
                "title": str,
                "description": str,
                "likelihood": Enum("low", "medium", "high"),
                "impact": Enum("low", "medium", "high"),
                "mitigations": [{"text": str, "requirement_id": Opt(str), "test_hint": Opt(str)}],
                "status": Enum("mitigated", "accepted", "open"),
            }
        ],
        "requirement_ids": [str],
    },
    Kind.DESIGN_DOC: {
        "schema": int,
        "summary": str,
        "components": [{"name": str, "responsibility": str, "touches": [str], "requirement_ids": [str]}],
        "adrs": [
            {
                "id": Re(ADR_ID_RE),
                "title": str,
                "decision": str,
                "alternatives": [{"option": str, "rejected_because": str}],
                "rationale": str,
                "consequences": str,
            }
        ],
        "data_model_changes": [{"table": str, "change": Enum("create", "add_column", "alter"), "detail": str, "migration": str}],
        "api_changes": [{"method": str, "path": str, "change": Enum("add", "modify", "remove"), "breaking": bool}],
        "rollback_plan": str,
        "change_reference": str,
        "risk": {"level": Enum("LOW", "MEDIUM", "HIGH"), "reasons": [str]},
    },
    Kind.API_CONTRACT: {
        "schema": int,
        "paths": [
            {
                "method": str,
                "path": str,
                "summary": str,
                "request_fields": [{"name": str, "required": bool, "type": str}],
                "responses": [int],
            }
        ],
        "breaking": bool,
    },
    Kind.TASK_PLAN: {
        "schema": int,
        "tasks": [
            {
                "id": Re(TASK_ID_RE),
                "title": str,
                "group": Re(GROUP_ID_RE),
                "requirement_ids": [str],
                "depends_on": [str],
                "capability": str,
                "kind": Enum("implement", "test", "doc"),
                "write_scope": [str],
                "expected_paths": [str],
                "params": dict,
            }
        ],
        "groups": [
            {
                "id": Re(GROUP_ID_RE),
                "node_id": str,
                "task_ids": [str],
                "capability": str,
                "params": dict,
                "write_scope": [str],
            }
        ],
        "critical_path": [str],
        "unsupported": [{"requirement_id": str, "reason": str}],
    },
    Kind.INTEGRATION_REPORT: {
        "schema": int,
        "sources": [{"node_id": str, "artifact_id": str, "paths": [str]}],
        "merged_paths": [str],
        "conflicts": [{"path": str, "authors": [str], "reason": str}],
        "stats": {"files_added": int, "files_modified": int, "files_deleted": int, "lines_added": int, "lines_removed": int},
        "impact_delta": {"predicted_only": [str], "actual_only": [str], "precision": float, "recall": float},
    },
    Kind.TEST_REPORT: {
        "schema": int,
        "passed": bool,
        "total": int,
        "failed": int,
        "errors": int,
        "skipped": int,
        "tests": [{"nodeid": str, "outcome": Enum("passed", "failed", "error", "skipped"), "covers": [str]}],
        "baseline": {"ran": bool, "passed": Opt(bool), "failed_tests": [str]},
        "fail_to_pass": {"checked": bool, "ok": bool, "tests": [str]},
        "perf": {"checked": bool, "budget_ms": Opt(float), "ok": Opt(bool)},
        "api_compat": {"checked": bool, "breaking": [str], "additive": [str]},
    },
    Kind.SECURITY_REPORT: {
        "schema": int,
        "rules_evaluated": [str],
        "findings": [_FINDING],
        "files_scanned": int,
        "passed": bool,
    },
    Kind.READINESS_REPORT: {
        "schema": int,
        "verdict": Enum("GO", "NO_GO"),
        "checks": [{"id": str, "title": str, "passed": bool, "evidence": [str], "detail": str}],
        "risk": {"level": Enum("LOW", "MEDIUM", "HIGH"), "reasons": [str]},
        "rollback_plan": str,
        "change_reference": str,
        "open_items": [str],
    },
    Kind.TRACEABILITY_MATRIX: {
        "schema": int,
        "requirements": [
            {
                "id": str,
                "status": Enum("active", "deferred", "descoped"),
                "acceptance_criteria": [{"id": str, "tests": [str], "results": [str], "covered": bool}],
                "tasks": [str],
                "files": [str],
                "decision_refs": [str],
            }
        ],
        "complete": bool,
        "gaps": [str],
    },
    Kind.RELEASE_BUNDLE: {
        "schema": int,
        "mode": Enum("bundle", "applied"),
        "patch_sha256": str,
        "files": [str],
        "base_manifest_hash": str,
        "receipt": Opt(dict),
    },
    Kind.RELEASE_VERIFICATION: {
        "schema": int,
        "target": Enum("bundle-scratch", "applied-target"),
        "applied_cleanly": bool,
        "tests_passed": bool,
        "healthz_ok": Opt(bool),
        "checked_files": int,
    },
}


def validate_payload(kind: str, obj: Any) -> list[str]:
    """Validate a *decoded* artifact payload against its kind's shape. ``[]`` means valid."""
    errors: list[str] = []
    if kind in CHANGESET_KINDS:
        try:
            ChangeSet.from_dict(obj)
        except Exception as exc:  # noqa: BLE001 - any construction error is a validation problem
            errors.append(f"$: invalid change set: {exc}")
        return errors
    if kind == "plan_expansion":
        from orchestrator.core.types import PlanExpansion

        try:
            PlanExpansion.from_dict(obj)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"$: invalid plan expansion: {exc}")
        return errors
    shape = SHAPES.get(kind)
    if shape is None:
        return errors
    _check(shape, obj, "$", errors)
    if not errors and isinstance(obj, Mapping) and obj.get("schema") != SCHEMA_VERSION:
        errors.append(f"$.schema: expected {SCHEMA_VERSION}, got {obj.get('schema')!r}")
    if kind == Kind.CLARIFICATION_ANSWERS and not errors:
        answers = obj["answers"]
        if not isinstance(answers, Mapping):
            errors.append("$.answers: expected object")
        else:
            for qid, entry in answers.items():
                if not QUESTION_ID_RE.match(str(qid)):
                    errors.append(f"$.answers.{qid}: invalid question id")
                if not isinstance(entry, Mapping) or not isinstance(entry.get("answer"), str):
                    errors.append(f"$.answers.{qid}: expected object with string 'answer'")
    return errors
