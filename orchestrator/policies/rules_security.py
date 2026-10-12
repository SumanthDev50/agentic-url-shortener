"""SEC-001 ... SEC-008: security guardrails (catalogue: docs/architecture.md section 1.5).

SEC-002/003/004/006 are *heuristic tripwires, not a security boundary*; the boundary is structural
(proposals only, autonomy checks, sandboxed execution, human approval).  Every rule here fails
closed: text or code it cannot analyse becomes a HIGH finding.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from orchestrator.core.errors import PathViolation
from orchestrator.core.paths import match_any, validate_rel_path
from orchestrator.core.serde import canonical_json, to_jsonable
from orchestrator.core.types import (
    ActionSubject,
    ArtifactSubject,
    ChangeSetSubject,
    FileOp,
    Finding,
    PlanSubject,
    PolicyCategory,
    PolicyRule,
    Severity,
    SubjectKind,
    ToolAction,
)
from orchestrator.policies import pyscan, scanners
from orchestrator.policies.base import RuleBase, python_ops
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.hits import Hit, cap_evidence
from orchestrator.policies.routes import Route, routes_of_text

_CS, _ART, _ACT, _PLAN = (
    SubjectKind.CHANGESET,
    SubjectKind.ARTIFACT,
    SubjectKind.ACTION,
    SubjectKind.PLAN,
)


def _changeset_texts(subject: ChangeSetSubject) -> Iterator[tuple[str, str]]:
    """Every piece of text a change set carries: description, author, op paths and op contents."""
    yield "<description>", subject.changeset.description
    yield "<author>", subject.author or subject.changeset.author
    for index, op in enumerate(subject.changeset.ops):
        yield f"op[{index}].path", op.path
        if op.content is not None:
            yield op.path, op.content


class HardcodedSecrets(RuleBase):
    rule_id = "SEC-001"
    category = PolicyCategory.SECURITY
    title = "Hard-coded secret / private key / credential"
    severity = Severity.HIGH
    subjects = frozenset({_CS, _ART, _ACT, _PLAN})

    _FIX = "Remove the credential, load it from the environment or a secret store, and rotate it."

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        hits = [
            h
            for label, text in _changeset_texts(subject)
            for h in scanners.scan_secrets(text, label)
        ]
        return self.from_hits(hits, self._FIX)

    def check_artifact(self, subject: ArtifactSubject) -> list[Finding]:
        label = subject.artifact.id
        if subject.text is None:
            return [
                self.finding(
                    "artifact text was not supplied, so it cannot be scanned for secrets",
                    severity=Severity.HIGH,
                    location=label,
                )
            ]
        hits = scanners.scan_secrets(subject.text, label)
        try:
            metadata = canonical_json(dict(subject.artifact.metadata)).decode("ascii")
        except (TypeError, ValueError):
            return [
                *self.from_hits(hits, self._FIX),
                self.finding(
                    "artifact metadata cannot be serialised for scanning",
                    severity=Severity.HIGH,
                    location=label,
                ),
            ]
        return self.from_hits(
            [*hits, *scanners.scan_secrets(metadata, f"{label}#metadata")], self._FIX
        )

    def check_action(self, subject: ActionSubject) -> list[Finding]:
        return self.from_hits(scanners.scan_secrets(subject.target, "<action target>"), self._FIX)

    def check_plan(self, subject: PlanSubject) -> list[Finding]:
        # One JSON element per line: a realistic expansion (3+ nodes) is one compact line far longer
        # than the scanner's line cap, which would make every plan an "unscannable" HIGH finding.
        text = json.dumps(
            to_jsonable(subject.expansion.to_dict()),
            sort_keys=True,
            indent=1,
            ensure_ascii=True,
            allow_nan=False,
        )
        return self.from_hits(
            scanners.scan_secrets(text, f"plan:{subject.expansion.owner}"), self._FIX
        )


class DangerousConstructs(RuleBase):
    rule_id = "SEC-002"
    category = PolicyCategory.SECURITY
    title = "Dangerous Python construct"
    severity = Severity.HIGH
    subjects = frozenset({_CS})

    _FIX = (
        "Avoid dynamic code execution and shell commands; pass argument lists, use safe loaders, "
        "keep TLS verification on."
    )

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        hits = [
            h
            for op in python_ops(subject)
            for h in pyscan.analyse(op.content or "", op.path, pyscan.dangerous_calls)
        ]
        return self.from_hits(hits, self._FIX)


class RouteAuthentication(RuleBase):
    rule_id = "SEC-003"
    category = PolicyCategory.SECURITY
    title = "HTTP route without an auth dependency"
    severity = Severity.HIGH
    subjects = frozenset({_CS})

    _FIX = (
        "Add an authentication dependency (Depends(require_api_key)) or declare the route public "
        "in the policy config."
    )

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        names = self.config.auth_dependency_names
        findings: list[Finding] = []
        for op in python_ops(subject):
            routes, problem = routes_of_text(op.content or "", op.path, names)
            if routes is None:
                findings.extend(self.from_hits([problem] if problem else [], self._FIX))
                continue
            before = self._baseline_routes(subject, op)
            findings.extend(
                self._route_finding(op, r) for r in routes if self._is_unprotected(r, before)
            )
        return findings

    def _is_unprotected(self, route: Route, before: set[tuple[str, str, bool]]) -> bool:
        if (route.key, route.signature, route.authenticated) in before:
            return False  # unchanged legacy route: not part of this change
        return not route.authenticated and not self.config.is_public_route(route.method, route.path)

    def _route_finding(self, op: FileOp, route: Route) -> Finding:
        return self.finding(
            f"route {route.key} ({route.handler}) has no authentication dependency",
            location=f"{op.path}:{route.line}",
            evidence=route.key,
            remediation=self._FIX,
        )

    def _baseline_routes(self, subject: ChangeSetSubject, op: FileOp) -> set[tuple[str, str, bool]]:
        try:
            old_text = subject.baseline.read_text(op.path)
        except FileNotFoundError:
            return set()
        routes, _ = routes_of_text(old_text, op.path, self.config.auth_dependency_names)
        return {(r.key, r.signature, r.authenticated) for r in routes or []}


class SqlAssembly(RuleBase):
    rule_id = "SEC-004"
    category = PolicyCategory.SECURITY
    title = "SQL assembled by string formatting or concatenation"
    severity = Severity.HIGH
    subjects = frozenset({_CS})

    _FIX = (
        "Use parameterised queries (placeholders plus a parameter tuple); allow-list any "
        "identifier you must interpolate."
    )

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        hits = [
            h
            for op in python_ops(subject)
            for h in pyscan.analyse(op.content or "", op.path, pyscan.sql_construction)
        ]
        return self.from_hits(hits, self._FIX)


class AutonomyBoundary(RuleBase):
    rule_id = "SEC-005"
    category = PolicyCategory.SECURITY
    title = "Autonomy-boundary violation"
    severity = Severity.HIGH
    subjects = frozenset({_CS, _ACT})

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        findings: list[Finding] = []
        ops = subject.changeset.ops
        valid: list[FileOp] = []
        for index, op in enumerate(ops):
            try:
                validate_rel_path(op.path)
            except PathViolation as exc:
                findings.append(
                    self._escape(f"op[{index}] path rejected: {exc}", op.path, f"op[{index}]")
                )
            else:
                valid.append(op)
        profile = subject.profile
        if profile is None:
            attributed = bool(subject.author or subject.node_id or subject.changeset.author)
            findings.append(
                self.finding(
                    "change set carries no autonomy profile, so the autonomy boundary cannot be "
                    "verified",
                    severity=Severity.CRITICAL if attributed else Severity.HIGH,
                    location=subject.node_id or None,
                    remediation="The engine must attach the agent's narrowed AutonomyProfile.",
                )
            )
            return findings
        for op in valid:
            if not self._in_scope(op.path, profile.write_globs):
                findings.append(
                    self.finding(
                        "path is outside the agent's write_globs",
                        location=op.path,
                        evidence=op.kind.value,
                    )
                )
        findings.extend(
            self._budget_findings(
                ops, profile.max_changed_files, profile.max_file_bytes, profile.max_total_bytes
            )
        )
        return findings

    def check_action(self, subject: ActionSubject) -> list[Finding]:
        profile, action, target = subject.profile, subject.action, subject.target
        label = f"{action.value}:{cap_evidence(ascii(target), 50)}"
        if action in (ToolAction.READ, ToolAction.WRITE, ToolAction.DELETE):
            try:
                path = validate_rel_path(target)
            except PathViolation as exc:
                return [self._escape(f"{action.value} target rejected: {exc}", target, label)]
            globs = profile.read_globs if action == ToolAction.READ else profile.write_globs
            kind = "read_globs" if action == ToolAction.READ else "write_globs"
            if not self._in_scope(path, globs):
                return [
                    self.finding(
                        f"{action.value} target is outside the agent's {kind}",
                        location=label,
                        evidence=path,
                    )
                ]
            return []
        findings = [
            self._escape(f"{action.value} target rejected: {exc}", target, label)
            for exc in self._bad_exec_tokens(target)
        ]
        if not profile.can_execute:
            findings.append(
                self.finding(
                    f"{action.value} requested but the agent may not execute", location=label
                )
            )
        return findings

    # -- helpers -------------------------------------------------------------------------
    def _escape(self, message: str, path: str, location: str) -> Finding:
        return self.finding(
            message,
            severity=Severity.CRITICAL,
            location=location,
            evidence=cap_evidence(ascii(path), 60),
            remediation="Use a plain relative POSIX path inside the workspace.",
        )

    @staticmethod
    def _in_scope(path: str, globs: tuple[str, ...]) -> bool:
        try:
            return match_any(path, globs)
        except PathViolation:
            return False  # an unusable glob grants nothing

    @staticmethod
    def _bad_exec_tokens(target: str) -> Iterator[PathViolation]:
        for token in target.split():
            if token.startswith("-"):
                continue
            candidate = token.split("::", 1)[0]  # pytest node ids: path::test
            if "/" in candidate or "." in candidate:
                try:
                    validate_rel_path(candidate)
                except PathViolation as exc:
                    yield exc

    def _budget_findings(
        self, ops: tuple[FileOp, ...], max_files: int, max_file: int, max_total: int
    ) -> list[Finding]:
        findings: list[Finding] = []
        if len(ops) > max_files:
            findings.append(
                self.finding(
                    f"change set touches {len(ops)} files; the agent may change at most {max_files}"
                )
            )
        total = 0
        for op in ops:
            size = len(op.content.encode("utf-8")) if op.content is not None else 0
            total += size
            if size > max_file:
                findings.append(
                    self.finding(
                        f"file is {size} bytes; the agent may write at most {max_file} per file",
                        location=op.path,
                    )
                )
        if total > max_total:
            findings.append(
                self.finding(
                    f"change set carries {total} bytes; the agent may write at most {max_total} in "
                    f"total"
                )
            )
        return findings


class PromptInjection(RuleBase):
    rule_id = "SEC-006"
    category = PolicyCategory.SECURITY
    title = "Prompt-injection indicators in human-supplied text"
    severity = Severity.MEDIUM
    subjects = frozenset({_ART})

    _FIX = (
        "Human confirmation required: rephrase the text without instructions aimed at the "
        "assistant, or waive."
    )

    def check_artifact(self, subject: ArtifactSubject) -> list[Finding]:
        if not subject.untrusted_input:
            return []
        label = subject.artifact.id
        if subject.text is None:
            return [
                self.finding(
                    "untrusted input was not supplied as text, so it cannot be screened",
                    severity=Severity.HIGH,
                    location=label,
                )
            ]
        return self.from_hits(scanners.scan_prompt_injection(subject.text, label), self._FIX)


class TrojanSource(RuleBase):
    rule_id = "SEC-007"
    category = PolicyCategory.SECURITY
    title = "Trojan Source / hidden Unicode / non-ASCII identifiers"
    severity = Severity.HIGH
    subjects = frozenset({_CS, _ART})

    _FIX = "Remove invisible/bidirectional characters and use ASCII identifiers."

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        hits: list[Hit] = []
        for label, text in _changeset_texts(subject):
            hits.extend(scanners.scan_trojan_source(text, label))
        return self.from_hits(hits, self._FIX)

    def check_artifact(self, subject: ArtifactSubject) -> list[Finding]:
        if subject.untrusted_input:
            return []  # human text is screened by SEC-006 (MEDIUM, human-confirmable)
        label = subject.artifact.id
        if subject.text is None:
            return [
                self.finding(
                    "artifact text was not supplied, so it cannot be checked for hidden Unicode",
                    location=label,
                )
            ]
        return self.from_hits(scanners.scan_trojan_source(subject.text, label), self._FIX)


class EgressLibraries(RuleBase):
    rule_id = "SEC-008"
    category = PolicyCategory.SECURITY
    title = "Egress/execution library outside allow-listed modules"
    severity = Severity.HIGH
    subjects = frozenset({_CS})

    _FIX = (
        "Network and process access belongs in an allow-listed module "
        "(PolicyConfig.egress_modules)."
    )

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        hits = [
            h
            for op in python_ops(subject)
            if not self.config.is_egress_module(op.path)
            for h in pyscan.analyse(op.content or "", op.path, pyscan.egress_uses)
        ]
        return self.from_hits(hits, self._FIX)


def default_rules(config: PolicyConfig | None = None) -> list[PolicyRule]:
    cfg = config if config is not None else PolicyConfig()
    return [
        HardcodedSecrets(cfg),
        DangerousConstructs(cfg),
        RouteAuthentication(cfg),
        SqlAssembly(cfg),
        AutonomyBoundary(cfg),
        PromptInjection(cfg),
        TrojanSource(cfg),
        EgressLibraries(cfg),
    ]
