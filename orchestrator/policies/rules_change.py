"""CHG-001 ... CHG-008: change-control guardrails (catalogue: docs/architecture.md section 1.5).

Every rule fails closed: evidence that is required but missing (no impact report for a modification,
no OpenAPI diff for a changed route, no design document at release) is a finding, never a pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from orchestrator.core.errors import PathViolation
from orchestrator.core.paths import casefold_key, match_glob
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    CHANGESET_KINDS,
    ChangeSetSubject,
    FileOp,
    FileOpKind,
    Finding,
    PlanSubject,
    PolicyCategory,
    PolicyRule,
    RunSubject,
    Severity,
    SubjectKind,
)
from orchestrator.policies.base import RuleBase, changed_lines, read_or_empty
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.hits import cap_evidence
from orchestrator.policies.routes import routes_of_text
from orchestrator.policies.runview import latest_evidence, read_payload

_CS, _PLAN, _RUN = SubjectKind.CHANGESET, SubjectKind.PLAN, SubjectKind.RUN
_MODIFYING = (FileOpKind.MODIFY, FileOpKind.DELETE)
_MAX_PER_KIND = 20


def _any_match(path: str, patterns: Iterable[str]) -> bool:
    """Glob match per pattern; one unusable pattern never hides the others."""
    for pattern in patterns:
        try:
            if match_glob(path, pattern):
                return True
        except PathViolation:
            continue
    return False


# ---------------------------------------------------------------------------------------- CHG-001


class ImmutableMigrations(RuleBase):
    rule_id = "CHG-001"
    category = PolicyCategory.CHANGE_CONTROL
    title = "Applied DB migrations are immutable"
    severity = Severity.HIGH
    subjects = frozenset({_CS})

    _FIX = "Add a new, higher-numbered migration instead of editing or removing an applied one."

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        glob = self.config.migrations_glob
        existing = {casefold_key(p) for p in subject.baseline.list_files(glob)}
        findings: list[Finding] = []
        for op in subject.changeset.ops:
            if not match_glob(op.path, glob):
                continue
            exists = subject.baseline.exists(op.path)
            problem = self._problem(op, exists, casefold_key(op.path) in existing)
            if problem:
                findings.append(
                    self.finding(
                        problem, location=op.path, evidence=op.kind.value, remediation=self._FIX
                    )
                )
        return findings

    @staticmethod
    def _problem(op: FileOp, exists: bool, collides: bool) -> str | None:
        if op.kind in _MODIFYING:
            if exists:
                return f"{op.kind.value} of an applied migration (migrations are immutable)"
            return (
                f"{op.kind.value} of a migration that is not in the baseline view (immutability "
                f"cannot be verified)"
            )
        if exists or collides:
            return "create overwrites or shadows an existing migration"
        return None


# ---------------------------------------------------------------------------------------- CHG-002


class ChangeSizeBudget(RuleBase):
    rule_id = "CHG-002"
    category = PolicyCategory.CHANGE_CONTROL
    title = "Change-size budget"
    severity = Severity.MEDIUM
    subjects = frozenset({_CS})

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        budget = self.config.change_budget
        ops = subject.changeset.ops
        findings: list[Finding] = []
        if len(ops) > budget.max_files:
            findings.append(
                self.finding(
                    f"change set touches {len(ops)} files (budget {budget.max_files})",
                    evidence=f"files={len(ops)}",
                    remediation="Split the change into smaller, separately reviewable change sets.",
                )
            )
        lines = sum(changed_lines(subject.baseline, op) for op in ops)
        if lines > budget.max_lines:
            findings.append(
                self.finding(
                    f"change set alters about {lines} lines (budget {budget.max_lines})",
                    evidence=f"lines={lines}",
                    remediation="Split the change into smaller, separately reviewable change sets.",
                )
            )
        return findings


# ---------------------------------------------------------------------------------------- CHG-003


class ScopeWithinImpact(RuleBase):
    rule_id = "CHG-003"
    category = PolicyCategory.CHANGE_CONTROL
    title = "Modified files lie inside the impact set"
    severity = Severity.MEDIUM
    subjects = frozenset({_CS})

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        touched = [op for op in subject.changeset.ops if op.kind in _MODIFYING]
        if not touched:
            return []  # new files cannot leave a scope that only constrains *existing* files
        allowed = self._allowed(subject.impact)
        if allowed is None:
            return [
                self.finding(
                    "scope unverifiable: existing files are modified but no usable impact report "
                    "was supplied",
                    evidence=f"modified={len(touched)}",
                    remediation=(
                        "Run codebase analysis so the impact set can be checked, or waive with "
                        "justification."
                    ),
                )
            ]
        return [
            self.finding(
                f"{op.kind.value} of an existing file outside the impact set",
                location=op.path,
                evidence="outside impact set",
                remediation="Restrict the change to the analysed impact set or extend it.",
            )
            for op in touched[:_MAX_PER_KIND]
            if not _any_match(op.path, allowed)
        ]

    @staticmethod
    def _allowed(impact: Mapping[str, Any] | None) -> list[str] | None:
        """Paths/globs the impact report (plus ``recipe_anchors``) allows; ``None`` = no usable
        evidence.
        """
        if not isinstance(impact, Mapping):
            return None
        modules, expected = impact.get("modules"), impact.get("expected_paths")
        if not isinstance(modules, list) and not isinstance(expected, list):
            return None
        entries: list[Any] = (
            [m.get("path") for m in modules if isinstance(m, Mapping)]
            if isinstance(modules, list)
            else []
        )
        for key in ("expected_paths", "tests", "recipe_anchors"):
            value = impact.get(key)
            if isinstance(value, list):
                entries.extend(value)
        return [e for e in entries if isinstance(e, str) and e]


# ---------------------------------------------------------------------------------------- CHG-004


class ApiCompatibility(RuleBase):
    rule_id = "CHG-004"
    category = PolicyCategory.CHANGE_CONTROL
    title = "No breaking public API change"
    severity = Severity.HIGH
    subjects = frozenset({_CS})

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        impact = subject.impact if isinstance(subject.impact, Mapping) else {}
        diff = impact.get("openapi_diff")
        if diff is None:
            # An OpenAPI export only exists once the change is assembled and runnable: partial
            # implementation nodes cannot have one. Callers that can export it set
            # ``openapi_required`` (fail closed); otherwise the ``api_compat`` gate of the verify
            # node checks the real diff against the declared contract.
            if not impact.get("openapi_required"):
                return []
            changed = self._changed_routes(subject)
            if not changed:
                return []
            return [
                self.finding(
                    "HTTP routes changed but no OpenAPI diff was supplied, so API compatibility is "
                    "unverifiable",
                    evidence=cap_evidence(", ".join(changed[:5]), 70),
                    remediation=(
                        "Export the OpenAPI schema before/after and supply the diff as "
                        "impact['openapi_diff']."
                    ),
                )
            ]
        breaking = (
            diff.get("breaking") if isinstance(diff, Mapping) else getattr(diff, "breaking", None)
        )
        if not isinstance(breaking, (list, tuple)):
            return [
                self.finding(
                    "OpenAPI diff is malformed (no 'breaking' list), so compatibility is "
                    "unverifiable"
                )
            ]
        return [
            self.finding(
                "breaking public API change",
                evidence=cap_evidence(str(entry), 70),
                remediation="Make the change additive (optional field / new path) or version it.",
            )
            for entry in breaking[:_MAX_PER_KIND]
        ]

    def _changed_routes(self, subject: ChangeSetSubject) -> list[str]:
        names = self.config.auth_dependency_names
        changed: list[str] = []
        for op in subject.changeset.ops:
            if not op.path.lower().endswith(".py"):
                continue
            old, _ = routes_of_text(read_or_empty(subject.baseline, op.path), op.path, names)
            new, _ = (
                routes_of_text(op.content or "", op.path, names)
                if op.kind != FileOpKind.DELETE
                else ([], None)
            )
            if new is None:
                continue  # unparseable: QLT-001 rejects it
            before = {(r.key, r.signature) for r in old or []}
            after = {(r.key, r.signature) for r in new}
            changed.extend(sorted(key for key, _ in before ^ after))
        return sorted(set(changed))


# ---------------------------------------------------------------------------------------- CHG-005


_PLACEHOLDER_TEXT = frozenset({"", "n/a", "na", "none", "tbd", "todo", "-", "?", "unknown"})


class ReleaseDocumentation(RuleBase):
    rule_id = "CHG-005"
    category = PolicyCategory.CHANGE_CONTROL
    title = "Release requires a change reference and a rollback plan"
    severity = Severity.HIGH
    subjects = frozenset({_RUN})

    def check_run(self, subject: RunSubject) -> list[Finding]:
        artifact = latest_evidence(subject, Kind.DESIGN_DOC)
        payload = read_payload(subject, artifact)
        if payload is None:
            return [
                self.finding(
                    "no readable design document upstream of the release gate: change reference "
                    "and rollback plan are unverifiable",
                    location=subject.node_id or None,
                )
            ]
        findings: list[Finding] = []
        for key, label in (
            ("change_reference", "change reference"),
            ("rollback_plan", "rollback plan"),
        ):
            value = payload.get(key)
            if not isinstance(value, str) or value.strip().lower() in _PLACEHOLDER_TEXT:
                findings.append(
                    self.finding(
                        f"design document has no {label}",
                        location=artifact.id if artifact else None,
                        evidence=key,
                        remediation=f"Record a concrete {label} before release.",
                    )
                )
        return findings


# ---------------------------------------------------------------------- CHG-006 / CHG-007 (plans)


class MandatoryNodesPinned(RuleBase):
    rule_id = "CHG-006"
    category = PolicyCategory.CHANGE_CONTROL
    title = "A re-plan may not remove or alter a mandatory node"
    severity = Severity.HIGH
    subjects = frozenset({_PLAN})

    def check_plan(self, subject: PlanSubject) -> list[Finding]:
        mandatory = set(subject.mandatory) | {
            i for i, spec in subject.graph_before.items() if spec.mandatory
        }
        findings: list[Finding] = []
        for node_id in sorted(mandatory):
            before, after = subject.graph_before.get(node_id), subject.graph_after.get(node_id)
            if before is None:
                problem = (
                    "declared mandatory but unknown to the plan (cannot verify it is preserved)"
                )
            elif after is None:
                problem = "mandatory node removed by the plan"
            elif before.spec_hash() != after.spec_hash():
                problem = "mandatory node altered by the plan"
            else:
                continue
            findings.append(self.finding(problem, location=node_id, evidence=node_id))
        for target in sorted(set(subject.expansion.rewire) & mandatory):
            findings.append(
                self.finding("plan rewires a mandatory node", location=target, evidence=target)
            )
        return findings


class PlanConformance(RuleBase):
    rule_id = "CHG-007"
    category = PolicyCategory.CHANGE_CONTROL
    title = "Plan expansion / integration conforms to its template"
    severity = Severity.HIGH
    subjects = frozenset({_PLAN, _CS})

    def check_plan(self, subject: PlanSubject) -> list[Finding]:
        exp, rules = subject.expansion, self.config.expansion_rules
        findings: list[Finding] = []
        if exp.owner not in subject.graph_before:
            findings.append(
                self.finding("expansion owner is not a node of the plan", location=exp.owner)
            )
        if len(exp.nodes) > rules.max_new_nodes:
            findings.append(
                self.finding(
                    f"expansion adds {len(exp.nodes)} nodes (limit {rules.max_new_nodes})",
                    location=exp.owner,
                )
            )
        for node in exp.nodes:
            findings.extend(self._node_problems(subject, node))
        for target, rewire in sorted(exp.rewire.items()):
            findings.extend(self._rewire_problems(subject, target, rewire))
        return findings

    def _node_problems(self, subject: PlanSubject, node: Any) -> list[Finding]:
        problems: list[str] = []
        agents = [node.agent, *([node.fallback_agent] if node.fallback_agent else [])]
        problems += [
            f"agent {a!r} is not allowed by the owner's expansion policy"
            for a in agents
            if a not in subject.allowed_agents
        ]
        if node.stage.value not in subject.allowed_stages:
            problems.append(
                f"stage {node.stage.value!r} is not allowed by the owner's expansion policy"
            )
        problems += [
            f"produces {k!r}, not allowed by the owner's expansion policy"
            for k in node.produces
            if k not in subject.allowed_produces
        ]
        if subject.expansion.owner not in node.depends_on:
            problems.append("node does not depend on the expansion owner")
        grabs = [
            name
            for name, value in (
                ("mandatory", node.mandatory),
                ("side_effecting", node.side_effecting),
                ("compensator", node.compensator),
                ("skip_if", node.skip_if),
                ("compensates", node.compensates),
            )
            if value
        ]
        problems += [f"proposed node sets governance attribute {g!r}" for g in grabs]
        return [self.finding(p, location=node.id, evidence=node.id) for p in problems]

    def _rewire_problems(self, subject: PlanSubject, target: str, rewire: Any) -> list[Finding]:
        problems: list[str] = []
        if target not in subject.graph_before:
            problems.append("rewire target does not exist")
        if target == subject.expansion.owner:
            problems.append("an expansion may not rewire its own owner")
        if not any(match_glob(target, p) for p in self.config.expansion_rules.rewire_targets):
            problems.append("rewire target is outside the allowed targets")
        problems += [
            f"rewire adds dependency on unknown node {d!r}"
            for d in rewire.add_depends_on
            if d not in subject.graph_after
        ]
        problems += [
            f"rewire overlays non-change-set kind {k!r}"
            for k in rewire.add_overlay_from
            if k not in CHANGESET_KINDS
        ]
        return [self.finding(p, location=target, evidence=target) for p in problems]

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        if not subject.sources:
            return []
        produced: dict[str, set[str | None]] = {}
        for source in subject.sources:
            for op in source.ops:
                produced.setdefault(casefold_key(op.path), set()).add(op.content)
        findings: list[Finding] = []
        for op in subject.changeset.ops:
            results = produced.get(casefold_key(op.path))
            if results is None:
                problem = "path is not touched by any source change set"
            elif op.content not in results:
                problem = (
                    "result differs from every source change set (not derivable by integration)"
                )
            else:
                continue
            findings.append(self.finding(problem, location=op.path, evidence=op.kind.value))
        return findings


# ---------------------------------------------------------------------------------------- CHG-008


class ProtectedPaths(RuleBase):
    rule_id = "CHG-008"
    category = PolicyCategory.CHANGE_CONTROL
    title = "Protected paths are not modifiable by agents"
    severity = Severity.HIGH
    subjects = frozenset({_CS})

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        raw = self.config.protected_paths
        folded = tuple(p.casefold() for p in raw)
        return [
            self.finding(
                "agents may not modify protected paths (CI, container, dependency manifests, env "
                "files, keys)",
                location=op.path,
                evidence=op.kind.value,
                remediation="Ask a human to make this change outside the agent workflow.",
            )
            for op in subject.changeset.ops
            if _any_match(op.path, raw) or _any_match(op.path.casefold(), folded)
        ]


def default_rules(config: PolicyConfig | None = None) -> list[PolicyRule]:
    cfg = config if config is not None else PolicyConfig()
    return [
        ImmutableMigrations(cfg),
        ChangeSizeBudget(cfg),
        ScopeWithinImpact(cfg),
        ApiCompatibility(cfg),
        ReleaseDocumentation(cfg),
        MandatoryNodesPinned(cfg),
        PlanConformance(cfg),
        ProtectedPaths(cfg),
    ]
