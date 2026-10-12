"""``PolicyEngine``: dispatch, fail-closed post-processing, floor-protected enforcement, waivers.

Guarantees (each covered by ``tests/orchestrator/policies/test_policy_engine.py``):

* **Only matching rules run** - rules are indexed by ``SubjectKind``.
* **Fail closed** - a rule that raises (or returns something that is not a list of ``Finding``)
  becomes a HIGH ``POL-000`` finding; an unsupported subject does too.  Nothing a rule does can
  crash the scheduler.
* **Every finding is post-processed** - credentials masked, control characters stripped, lengths
  capped, severity raised to its floor/override, duplicates removed, order deterministic, findings
  per rule bounded (the suppressed remainder is summarised at the highest suppressed severity, never
  dropped silently).
* **Floor-protected enforcement** - ``PolicyReport.from_findings`` uses the fixed severity ->
  enforcement mapping (HIGH denies, CRITICAL safe-stops); the constructor refuses a mapping that
  would weaken it.
* **Waivers** apply to MEDIUM-or-lower findings only (never HIGH/CRITICAL, never ``POL-000``) and
  only when the waiver matches rule id, ``Finding.key()``, the node *and* the subject's evidence
  hash (``evidence_hash``).
* **Manifest** - at construction the rule ids must equal ``config.CATALOGUE`` (category/severity
  too) and every ``SubjectKind`` must have at least one rule; ``manifest_hash`` pins that into
  ``RUN_CREATED``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from orchestrator.core.errors import ConfigError
from orchestrator.core.paths import RULE_ID_RE
from orchestrator.core.sanitize import strip_controls
from orchestrator.core.serde import hash_jsonable, sha256_hex
from orchestrator.core.types import (
    DEFAULT_ENFORCEMENT,
    ActionSubject,
    ArtifactSubject,
    ChangeSetSubject,
    Enforcement,
    Finding,
    PlanSubject,
    PolicyCategory,
    PolicyEvaluator,
    PolicyReport,
    PolicyRule,
    PolicySubject,
    RiskAssessment,
    RiskLevel,
    RunSubject,
    Severity,
    SubjectKind,
    Waiver,
)
from orchestrator.policies import rules_change, rules_compliance, rules_quality, rules_security
from orchestrator.policies.config import CATALOGUE, PolicyConfig
from orchestrator.policies.hits import cap_evidence
from orchestrator.policies.risk import assess_changeset_risk
from orchestrator.policies.scanners import redact_text
from orchestrator.tools.changeset import digest

ENGINE_FAULT_RULE = "POL-000"
MAX_FINDINGS_PER_RULE = 50
EVIDENCE_LIMIT = 80
MESSAGE_LIMIT = 300
LOCATION_LIMIT = 160

_SUBJECT_TYPES = (ChangeSetSubject, ArtifactSubject, ActionSubject, PlanSubject, RunSubject)


def default_rules(config: PolicyConfig | None = None) -> list[PolicyRule]:
    """The full rule catalogue (SEC, CMP, CHG, QLT) bound to ``config``."""
    cfg = config if config is not None else PolicyConfig()
    return [
        *rules_security.default_rules(cfg),
        *rules_compliance.default_rules(cfg),
        *rules_change.default_rules(cfg),
        *rules_quality.default_rules(cfg),
    ]


# ------------------------------------------------------------------------------- subject identity


def subject_node_id(subject: PolicySubject) -> str:
    """The node a subject belongs to (waivers are bound to it)."""
    if isinstance(subject, ArtifactSubject):
        return subject.artifact.producer
    if isinstance(subject, PlanSubject):
        return subject.expansion.owner
    return subject.node_id


def evidence_hash(subject: PolicySubject) -> str:
    """Deterministic hash of the *content* a policy decision was made on.

    A waiver is valid only for the evidence it was granted for: any change to the change set ops,
    artifact bytes, action, plan or run facts changes this hash and so voids the waiver.
    """
    if isinstance(subject, ChangeSetSubject):
        return hash_jsonable(
            {"kind": "changeset", "node": subject.node_id, "ops": digest(subject.changeset)}
        )
    if isinstance(subject, ArtifactSubject):
        text_hash = sha256_hex(subject.text) if subject.text is not None else None
        return hash_jsonable(
            {
                "kind": "artifact",
                "id": subject.artifact.id,
                "content": subject.artifact.content_hash,
                "text": text_hash,
                "untrusted": subject.untrusted_input,
            }
        )
    if isinstance(subject, ActionSubject):
        return hash_jsonable(
            {
                "kind": "action",
                "agent": subject.agent_id,
                "node": subject.node_id,
                "action": subject.action.value,
                "target": subject.target,
            }
        )
    if isinstance(subject, PlanSubject):
        return hash_jsonable(
            {"kind": "plan", "run": subject.run_id, "expansion": subject.expansion.to_dict()}
        )
    view = subject.view
    return hash_jsonable(
        {
            "kind": "run",
            "run": subject.run_id,
            "node": subject.node_id,
            "artifacts": sorted((a.id, a.content_hash) for a in view.artifacts()),
            "approvals": sorted((a.id, a.status.value, len(a.votes)) for a in view.approvals()),
            "decisions": sorted(d.id for d in view.decisions()),
            "waivers": sorted(w.id for w in view.waivers()),
        }
    )


# ----------------------------------------------------------------------------------------- engine


class PolicyEngine(PolicyEvaluator):
    """Evaluates policy subjects; immutable and thread-safe."""

    def __init__(
        self,
        rules: Iterable[PolicyRule] | None = None,
        config: PolicyConfig | None = None,
        waivers: Sequence[Waiver] = (),
    ) -> None:
        self._config = config if config is not None else PolicyConfig()
        self._rules: tuple[PolicyRule, ...] = (
            tuple(rules) if rules is not None else tuple(default_rules(self._config))
        )
        self._waivers: tuple[Waiver, ...] = tuple(waivers)
        if not all(isinstance(w, Waiver) for w in self._waivers):
            raise ConfigError("waivers must be Waiver objects")
        self._enforcement: Mapping[Severity, Enforcement] = DEFAULT_ENFORCEMENT
        self._check_enforcement_floors()
        self._check_manifest()
        self._by_kind: dict[SubjectKind, tuple[PolicyRule, ...]] = {
            kind: tuple(r for r in self._rules if kind in r.subjects) for kind in SubjectKind
        }

    # -- identity ------------------------------------------------------------------
    @property
    def config(self) -> PolicyConfig:
        return self._config

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted(r.rule_id for r in self._rules))

    @property
    def waivers(self) -> tuple[Waiver, ...]:
        return self._waivers

    @property
    def manifest_hash(self) -> str:
        """Hash of the active rules (id, category, severity, title, subjects) and the enforcement
        mapping.
        """
        rules = sorted(
            (
                r.rule_id,
                r.category.value,
                r.severity.name,
                r.title,
                sorted(k.value for k in r.subjects),
            )
            for r in self._rules
        )
        return hash_jsonable(
            {"rules": rules, "enforcement": {s.name: e.value for s, e in self._enforcement.items()}}
        )

    @property
    def config_hash(self) -> str:
        return self._config.config_hash()

    # -- construction checks ------------------------------------------------------------
    def _check_enforcement_floors(self) -> None:
        mapping = self._enforcement
        if (
            mapping[Severity.CRITICAL] != Enforcement.SAFE_STOP
            or mapping[Severity.HIGH] not in (Enforcement.DENY, Enforcement.SAFE_STOP)
            or mapping[Severity.MEDIUM] in (Enforcement.ALLOW, Enforcement.WARN)
        ):
            raise ConfigError(
                "severity -> enforcement mapping weakens the floors (CRITICAL stops, HIGH denies, "
                "MEDIUM needs a human)"
            )

    def _check_manifest(self) -> None:
        ids = [r.rule_id for r in self._rules]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"duplicate rule ids: {sorted({i for i in ids if ids.count(i) > 1})}")
        missing, extra = sorted(set(CATALOGUE) - set(ids)), sorted(set(ids) - set(CATALOGUE))
        if missing or extra:
            raise ConfigError(
                f"rule set differs from the catalogue manifest (missing {missing}, unexpected "
                f"{extra})"
            )
        for rule in self._rules:
            entry = CATALOGUE[rule.rule_id]
            if (
                not RULE_ID_RE.match(rule.rule_id)
                or rule.category != entry.category
                or rule.severity != entry.severity
            ):
                raise ConfigError(
                    f"rule {rule.rule_id} category/severity differs from the catalogue manifest"
                )
            if not rule.subjects or not all(isinstance(k, SubjectKind) for k in rule.subjects):
                raise ConfigError(f"rule {rule.rule_id} declares no valid subject kinds")
        uncovered = [k.value for k in SubjectKind if not any(k in r.subjects for r in self._rules)]
        if uncovered:
            raise ConfigError(f"no rule evaluates subject kinds {uncovered}")

    # -- evaluation ---------------------------------------------------------------------
    def evaluate(self, subject: PolicySubject) -> PolicyReport:
        if not isinstance(subject, _SUBJECT_TYPES):
            return self._report(
                [self._fault("unsupported policy subject type", type(subject).__name__)], [], ()
            )
        rules = self._by_kind[subject.kind]
        raw: list[Finding] = []
        for rule in rules:
            raw.extend(self._run_rule(rule, subject))
        findings = self._postprocess(raw)
        remaining, waived = self._apply_waivers(findings, subject)
        return self._report(remaining, waived, tuple(sorted(r.rule_id for r in rules)))

    def _run_rule(self, rule: PolicyRule, subject: PolicySubject) -> list[Finding]:
        try:
            produced = rule.check(subject)
            if not isinstance(produced, list) or not all(isinstance(f, Finding) for f in produced):
                raise TypeError("rule did not return a list of Finding")
        except Exception as exc:
            return [self._fault(f"rule {rule.rule_id} crashed ({type(exc).__name__})", str(exc))]
        return produced

    @staticmethod
    def _fault(message: str, detail: str) -> Finding:
        return Finding(
            ENGINE_FAULT_RULE,
            PolicyCategory.SECURITY,
            Severity.HIGH,
            message,
            evidence=detail or None,
            remediation="Policy could not be evaluated; treated as a violation (fail closed).",
        )

    def _report(
        self, findings: list[Finding], waived: list[Finding], evaluated: tuple[str, ...]
    ) -> PolicyReport:
        return PolicyReport.from_findings(
            findings, waived=waived, mapping=self._enforcement, rules_evaluated=evaluated
        )

    # -- post-processing ----------------------------------------------------------------
    def _postprocess(self, raw: list[Finding]) -> list[Finding]:
        cleaned: dict[tuple[str, str | None, str | None, str], Finding] = {}
        for finding in raw:
            fixed = self._clean(finding)
            cleaned.setdefault(
                (fixed.rule_id, fixed.location, fixed.evidence, fixed.message), fixed
            )
        return sorted(self._cap_per_rule(sorted(cleaned.values(), key=_order_key)), key=_order_key)

    def _clean(self, finding: Finding) -> Finding:
        if finding.rule_id == ENGINE_FAULT_RULE:
            severity = Severity.HIGH
        else:
            severity = self._config.effective_severity(finding.rule_id, finding.severity)
        return Finding(
            finding.rule_id,
            finding.category,
            severity,
            _bounded(finding.message, MESSAGE_LIMIT) or "",
            location=_bounded(finding.location, LOCATION_LIMIT),
            evidence=_bounded(finding.evidence, EVIDENCE_LIMIT),
            remediation=_bounded(finding.remediation, MESSAGE_LIMIT),
        )

    @staticmethod
    def _cap_per_rule(ordered: list[Finding]) -> list[Finding]:
        kept: list[Finding] = []
        counts: dict[str, int] = {}
        worst_dropped: dict[str, Finding] = {}
        dropped: dict[str, int] = {}
        for finding in ordered:  # already sorted most severe first
            n = counts.get(finding.rule_id, 0)
            if n < MAX_FINDINGS_PER_RULE:
                counts[finding.rule_id] = n + 1
                kept.append(finding)
            else:
                dropped[finding.rule_id] = dropped.get(finding.rule_id, 0) + 1
                worst_dropped.setdefault(finding.rule_id, finding)
        for rule_id, n in dropped.items():
            sample = worst_dropped[rule_id]
            kept.append(
                Finding(
                    rule_id,
                    sample.category,
                    sample.severity,
                    f"{n} further findings of this rule were suppressed",
                    evidence=f"+{n} more",
                )
            )
        return kept

    # -- waivers ----------------------------------------------------------------------
    def _apply_waivers(
        self, findings: list[Finding], subject: PolicySubject
    ) -> tuple[list[Finding], list[Finding]]:
        if not self._waivers:
            return findings, []
        node, ehash = subject_node_id(subject), evidence_hash(subject)
        granted = {
            (w.rule_id, w.finding_key)
            for w in self._waivers
            if w.node_id == node and w.evidence_hash == ehash
        }
        remaining: list[Finding] = []
        waived: list[Finding] = []
        for finding in findings:
            waivable = finding.severity <= Severity.MEDIUM and finding.rule_id != ENGINE_FAULT_RULE
            (
                waived if waivable and (finding.rule_id, finding.key()) in granted else remaining
            ).append(finding)
        return remaining, waived

    def with_waivers(self, waivers: Sequence[Waiver]) -> PolicyEngine:
        """A new engine honouring the union of its current waivers and ``waivers`` (deduplicated by
        id).
        """
        merged = {w.id: w for w in (*self._waivers, *waivers)}
        return PolicyEngine(self._rules, self._config, tuple(merged.values()))

    # -- risk / evidence ---------------------------------------------------------------
    def assess_risk(self, subject: PolicySubject) -> RiskAssessment:
        if isinstance(subject, ChangeSetSubject):
            return assess_changeset_risk(subject, self._config)
        return RiskAssessment(RiskLevel.LOW, ("not a change set: no change-set risk signals",), {})

    def evidence_hash(self, subject: PolicySubject) -> str:
        """The hash a waiver must carry to apply to ``subject`` (see ``evidence_hash``)."""
        return evidence_hash(subject)


def _order_key(f: Finding) -> tuple[int, str, str, str, str]:
    return (-f.severity, f.rule_id, f.location or "", f.evidence or "", f.message)


def _bounded(text: Any, limit: int) -> str | None:
    """Mask credentials, strip control characters, cap the length; ``None`` stays ``None``."""
    if text is None:
        return None
    return cap_evidence(strip_controls(redact_text(str(text)), keep_newlines=False), limit)
