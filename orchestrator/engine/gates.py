"""Generic gates and gate composition (engine-semantics section 8).

``default_gate_registry()`` registers the four generic gates:

* ``artifacts_present`` - exit: every kind in ``node.produces`` was staged and is well-formed; entry:
  every kind in ``node.consumes`` is available. Payload shapes are checked with
  ``schemas.validate_payload``.
* ``json_shape`` (param ``kind``) - the staged (exit) / upstream (entry) artifacts of one kind are
  valid for their schema.
* ``policy`` (optional param ``profile``: an ``AutonomyProfile.to_dict()``) - the always-on policy
  evaluation over the staged artifacts. Change sets become ``ChangeSetSubject`` s and need the
  *narrowed* autonomy profile of the executing agent (agent profile narrowed by the node); the engine
  builds that implicit ``GateSpec`` per attempt. **Without a profile a change set fails closed**
  (GATE-003) - the gate never invents a permissive profile. ``plan_expansion`` artifacts are skipped
  here: the scheduler evaluates them as ``PlanSubject`` against the live graph. The result carries
  ``evidence["policy_evaluations"]`` (``subject``, ``subject_hash``, ``rules_evaluated``,
  ``findings``) so the scheduler can emit ``POLICY_EVALUATED`` events.
* ``no_blocking_questions`` (optional param ``kinds``) - blocking ``open_questions`` of the staged
  payloads that no ``clarification_answers`` entry answers yet become ``AWAIT_INPUT``.

``run_gates`` evaluates **all** gates; a gate that cannot be built or raises is a fail-closed
``GATE-000`` HIGH finding (never a scheduler crash). ``compose`` implements the precedence
``SAFE_STOP > FAIL > AWAIT_INPUT > REQUIRE_APPROVAL > PASS``.

Finding ids: GATE-000 gate fault, GATE-001 required artifact missing, GATE-002 payload invalid,
GATE-003 policy gate misconfigured, GATE-004 malformed clarification question.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator.core.errors import ConfigError
from orchestrator.core.paths import KIND_RE
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.schemas import SHAPES, validate_payload
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import (
    CHANGESET_KINDS,
    KEYED_KINDS,
    Artifact,
    ArtifactSubject,
    AutonomyProfile,
    ChangeSet,
    ChangeSetSubject,
    ClarificationQuestion,
    Enforcement,
    Finding,
    Gate,
    GateContext,
    GateDecision,
    GatePhase,
    GateResult,
    GateSpec,
    PolicyCategory,
    PolicySubject,
    RiskAssessment,
    RiskLevel,
    Severity,
)
from orchestrator.engine.registry import GateRegistry
from orchestrator.tools.changeset import digest

GATE_FAULT = "GATE-000"
GATE_MISSING = "GATE-001"
GATE_INVALID = "GATE-002"
GATE_POLICY_CONFIG = "GATE-003"
GATE_QUESTION = "GATE-004"

K_PLAN_EXPANSION = "plan_expansion"
K_INTEGRATED = "integrated_changeset"
K_CHANGESET = "changeset"
K_IMPACT = "impact_report"
K_ANSWERS = "clarification_answers"
TEXT_KINDS = frozenset({"request", "release_patch", "engineering_summary"})
MAX_PROBLEMS_PER_ARTIFACT = 5
MAX_QUESTIONS = 50
_PRECEDENCE = (
    GateDecision.SAFE_STOP,
    GateDecision.FAIL,
    GateDecision.AWAIT_INPUT,
    GateDecision.REQUIRE_APPROVAL,
    GateDecision.PASS,
)


def _finding(
    rule_id: str,
    message: str,
    *,
    location: str | None = None,
    remediation: str | None = None,
) -> Finding:
    return Finding(
        rule_id,
        PolicyCategory.QUALITY,
        Severity.HIGH,
        sanitize_for_terminal(message, 240),
        location=sanitize_for_terminal(location, 120) if location else None,
        remediation=remediation,
    )


def _result(gate: str, findings: Sequence[Finding], message: str = "") -> GateResult:
    decision = GateDecision.FAIL if findings else GateDecision.PASS
    return GateResult(gate, decision, message, tuple(findings))


# --------------------------------------------------------------------------- payload helpers


def _has_schema(kind: str) -> bool:
    return kind in SHAPES or kind in CHANGESET_KINDS or kind == K_PLAN_EXPANSION


def _shape_findings(ctx: GateContext, art: Artifact) -> list[Finding]:
    """Problems of one artifact's payload (shared by ``artifacts_present`` and ``json_shape`` so a
    problem reported by both gates is one finding after composition)."""
    if art.media_type != "application/json":
        if _has_schema(art.kind):
            return [
                _finding(
                    GATE_INVALID,
                    f"{art.kind} must be a JSON payload",
                    location=art.id,
                    remediation=f"Return the {art.kind} content as structured JSON.",
                )
            ]
        return []
    try:
        payload = ctx.reader.read_json(art)
    except ValueError:
        return [
            _finding(
                GATE_INVALID,
                f"{art.kind} is not valid JSON",
                location=art.id,
                remediation=f"Return well-formed JSON for {art.kind}.",
            )
        ]
    problems = validate_payload(art.kind, payload)
    return [
        _finding(
            GATE_INVALID,
            f"{art.kind} payload invalid: {problem}",
            location=art.id,
            remediation=f"Make the {art.kind} payload conform to its schema (docs/artifact-schemas.md).",
        )
        for problem in problems[:MAX_PROBLEMS_PER_ARTIFACT]
    ]


def _missing(kind: str, where: str) -> Finding:
    return _finding(
        GATE_MISSING,
        f"required artifact kind {kind!r} is missing ({where})",
        location=kind,
        remediation=f"Produce an artifact of kind {kind!r}.",
    )


def _produced(ctx: GateContext, kind: str) -> list[Artifact]:
    return sorted((a for a in ctx.produced if a.kind == kind), key=lambda a: a.id)


def _upstream(ctx: GateContext, kind: str) -> list[Artifact]:
    return ctx.artifacts.active(kind)


# --------------------------------------------------------------------------- gates


class ArtifactsPresentGate:
    name = "artifacts_present"

    def evaluate(self, ctx: GateContext) -> GateResult:
        findings: list[Finding] = []
        if ctx.phase == GatePhase.ENTRY:
            for kind in ctx.node.consumes:
                if not _upstream(ctx, kind):
                    findings.append(_missing(kind, "consumed input"))
            return _result(self.name, findings, "inputs present" if not findings else "inputs missing")
        for kind in ctx.node.produces:
            found = _produced(ctx, kind)
            if kind in KEYED_KINDS:
                found = [a for a in found if a.key == ctx.node.id]
            if not found:
                findings.append(_missing(kind, "not produced by the attempt"))
                continue
            for art in found:
                findings.extend(_shape_findings(ctx, art))
        return _result(self.name, findings, "outputs present" if not findings else "outputs invalid")


class JsonShapeGate:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.name = f"json_shape:{kind}"

    def evaluate(self, ctx: GateContext) -> GateResult:
        arts = _upstream(ctx, self.kind) if ctx.phase == GatePhase.ENTRY else _produced(ctx, self.kind)
        if not arts:
            return _result(self.name, [_missing(self.kind, "nothing to validate")])
        findings: list[Finding] = []
        for art in arts:
            findings.extend(_shape_findings(ctx, art))
        return _result(self.name, findings)


class NoBlockingQuestionsGate:
    name = "no_blocking_questions"

    def __init__(self, kinds: frozenset[str] | None = None) -> None:
        self.kinds = kinds

    def evaluate(self, ctx: GateContext) -> GateResult:
        answered = self._answered(ctx)
        questions: dict[str, ClarificationQuestion] = {}
        findings: list[Finding] = []
        for art in sorted(ctx.produced, key=lambda a: a.id):
            if art.media_type != "application/json" or (self.kinds and art.kind not in self.kinds):
                continue
            try:
                payload = ctx.reader.read_json(art)
            except ValueError:
                continue  # json_shape reports unreadable payloads
            raw = payload.get("open_questions") if isinstance(payload, Mapping) else None
            if not isinstance(raw, list):
                continue
            for index, entry in enumerate(raw):
                try:
                    question = ClarificationQuestion.from_dict(entry)
                except (KeyError, TypeError, ValueError, AttributeError):
                    findings.append(
                        _finding(
                            GATE_QUESTION,
                            f"open_questions[{index}] is malformed",
                            location=art.id,
                            remediation="Describe open questions with id and text.",
                        )
                    )
                    continue
                if question.blocking and question.id not in answered:
                    questions.setdefault(question.id, question)
        if findings:
            return GateResult(self.name, GateDecision.FAIL, "malformed questions", tuple(findings))
        if questions:
            ordered = tuple(questions[qid] for qid in sorted(questions))[:MAX_QUESTIONS]
            return GateResult(
                self.name,
                GateDecision.AWAIT_INPUT,
                f"{len(ordered)} blocking question(s) need a human answer",
                questions=ordered,
            )
        return GateResult(self.name, GateDecision.PASS, "no blocking questions")

    @staticmethod
    def _answered(ctx: GateContext) -> frozenset[str]:
        if K_ANSWERS not in ctx.node.input_kinds:
            return frozenset()
        art = ctx.artifacts.latest(K_ANSWERS)
        if art is None:
            return frozenset()
        try:
            payload = ctx.artifacts.read_json(art)
        except ValueError:
            return frozenset()
        answers = payload.get("answers") if isinstance(payload, Mapping) else None
        return frozenset(str(k) for k in answers) if isinstance(answers, Mapping) else frozenset()


class PolicyGate:
    name = "policy"

    def __init__(self, profile: AutonomyProfile | None = None) -> None:
        self.profile = profile

    def evaluate(self, ctx: GateContext) -> GateResult:
        findings: dict[Finding, None] = {}
        evaluations: list[dict[str, Any]] = []
        risks: list[RiskAssessment] = []
        enforcement = Enforcement.ALLOW
        waived = 0
        for art in sorted(ctx.produced, key=lambda a: a.id):
            if art.kind == K_PLAN_EXPANSION:
                continue
            subject, label, problems = self._subject(ctx, art)
            findings.update(dict.fromkeys(problems))
            if subject is None:
                continue
            report = ctx.policy.evaluate(subject)
            findings.update(dict.fromkeys(report.findings))
            enforcement = max(enforcement, report.enforcement, key=_ENFORCEMENT_ORDER.index)
            waived += len(report.waived)
            evaluations.append(
                {
                    "subject": label,
                    "subject_hash": self._subject_hash(subject),
                    "rules_evaluated": list(report.rules_evaluated),
                    "findings": [
                        {"rule_id": f.rule_id, "severity": f.severity.name, "location": f.location}
                        for f in report.findings[:50]
                    ],
                }
            )
            if isinstance(subject, ChangeSetSubject):
                risks.append(ctx.policy.assess_risk(subject))
        decision = _policy_decision(enforcement)
        if any(f.rule_id.startswith("GATE-") for f in findings):
            decision = _worse(decision, GateDecision.FAIL)
        return GateResult(
            self.name,
            decision,
            f"{len(evaluations)} subject(s) evaluated",
            tuple(findings),
            risk=_max_risk(risks),
            evidence={"policy_evaluations": evaluations, "waived": waived},
        )

    def _subject(
        self, ctx: GateContext, art: Artifact
    ) -> tuple[PolicySubject | None, str, list[Finding]]:
        label = f"{'changeset' if art.kind in CHANGESET_KINDS else 'artifact'}:{art.id}"
        if art.kind not in CHANGESET_KINDS:
            try:
                text: str | None = ctx.reader.read_text(art)
            except UnicodeDecodeError:
                text = None  # the rules fail closed on missing text
            return ArtifactSubject(art, text=text), label, []
        if self.profile is None:
            return (
                None,
                label,
                [
                    _finding(
                        GATE_POLICY_CONFIG,
                        "no autonomy profile was supplied to the policy gate; change sets cannot be "
                        "checked against it",
                        location=art.id,
                        remediation="Engine misconfiguration: the policy gate needs the agent's profile.",
                    )
                ],
            )
        try:
            changeset = ChangeSet.from_dict(ctx.reader.read_json(art))
        except (ValueError, KeyError, TypeError, AttributeError):
            return (
                None,
                label,
                [
                    _finding(
                        GATE_INVALID,
                        f"{art.kind} is not a valid change set",
                        location=art.id,
                        remediation="Return a change set with valid relative paths and file operations.",
                    )
                ],
            )
        subject = ChangeSetSubject(
            changeset=changeset,
            baseline=ctx.baseline,
            node_id=ctx.node.id,
            author=art.agent_id,
            impact=self._impact(ctx),
            profile=self.profile,
            sources=self._sources(ctx, art),
        )
        return subject, label, []

    @staticmethod
    def _impact(ctx: GateContext) -> Mapping[str, Any] | None:
        if K_IMPACT not in ctx.node.input_kinds:
            return None
        art = ctx.artifacts.latest(K_IMPACT)
        if art is None:
            return None
        payload = ctx.artifacts.read_json(art)
        return payload if isinstance(payload, Mapping) else None

    @staticmethod
    def _sources(ctx: GateContext, art: Artifact) -> tuple[ChangeSet, ...]:
        if art.kind != K_INTEGRATED or K_CHANGESET not in ctx.node.input_kinds:
            return ()
        return tuple(
            ChangeSet.from_dict(ctx.artifacts.read_json(source))
            for source in ctx.artifacts.active(K_CHANGESET)
        )

    @staticmethod
    def _subject_hash(subject: PolicySubject) -> str:
        if isinstance(subject, ChangeSetSubject):
            return hash_jsonable(
                {"kind": "changeset", "node": subject.node_id, "ops": digest(subject.changeset)}
            )
        assert isinstance(subject, ArtifactSubject)
        return hash_jsonable({"kind": "artifact", "content": subject.artifact.content_hash})


_ENFORCEMENT_ORDER = [
    Enforcement.ALLOW,
    Enforcement.WARN,
    Enforcement.REQUIRE_APPROVAL,
    Enforcement.DENY,
    Enforcement.SAFE_STOP,
]


def _policy_decision(enforcement: Enforcement) -> GateDecision:
    if enforcement == Enforcement.SAFE_STOP:
        return GateDecision.SAFE_STOP
    if enforcement == Enforcement.DENY:
        return GateDecision.FAIL
    if enforcement == Enforcement.REQUIRE_APPROVAL:
        return GateDecision.REQUIRE_APPROVAL
    return GateDecision.PASS


def _worse(a: GateDecision, b: GateDecision) -> GateDecision:
    return a if _PRECEDENCE.index(a) <= _PRECEDENCE.index(b) else b


def _max_risk(risks: Sequence[RiskAssessment]) -> RiskAssessment | None:
    if not risks:
        return None
    level = max(r.level for r in risks)
    top = [r for r in risks if r.level == level]
    reasons = tuple(dict.fromkeys(reason for r in top for reason in r.reasons))
    signals: dict[str, Any] = {}
    for r in top:
        for key, value in r.signals.items():
            signals.setdefault(key, value)
    return RiskAssessment(level, reasons, signals)


# --------------------------------------------------------------------------- registry


def _json_shape_factory(params: Mapping[str, Any]) -> Gate:
    kind = params.get("kind")
    if not isinstance(kind, str) or not KIND_RE.fullmatch(kind):
        raise ConfigError("json_shape needs a valid artifact kind in params['kind']")
    return JsonShapeGate(kind)


def _policy_factory(params: Mapping[str, Any]) -> Gate:
    raw = params.get("profile")
    if raw is None:
        return PolicyGate()
    if not isinstance(raw, Mapping):
        raise ConfigError("policy params['profile'] must be an AutonomyProfile dict")
    try:
        return PolicyGate(AutonomyProfile.from_dict(raw))
    except (ValueError, TypeError, KeyError) as exc:
        raise ConfigError(f"policy params['profile'] is invalid: {type(exc).__name__}") from exc


def _questions_factory(params: Mapping[str, Any]) -> Gate:
    raw = params.get("kinds")
    if raw is None:
        return NoBlockingQuestionsGate()
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(k, str) and KIND_RE.fullmatch(k) for k in raw)
    ):
        raise ConfigError("no_blocking_questions params['kinds'] must be a list of artifact kinds")
    return NoBlockingQuestionsGate(frozenset(raw))


def default_gate_registry() -> GateRegistry:
    registry = GateRegistry()
    registry.register("artifacts_present", lambda params: ArtifactsPresentGate())
    registry.register("json_shape", _json_shape_factory)
    registry.register("policy", _policy_factory)
    registry.register("no_blocking_questions", _questions_factory)
    return registry


# --------------------------------------------------------------------------- run + compose


def _fault(spec_type: str, ctx: GateContext, exc: BaseException) -> GateResult:
    detail = sanitize_for_terminal(str(exc), 160)
    finding = Finding(
        GATE_FAULT,
        PolicyCategory.QUALITY,
        Severity.HIGH,
        f"gate {sanitize_for_terminal(spec_type, 60)!r} could not be evaluated "
        f"({type(exc).__name__}: {detail})",
        location=ctx.node.id,
        remediation="Gate evaluation failed; treated as a rejection (fail closed).",
    )
    return GateResult(spec_type, GateDecision.FAIL, "gate fault", (finding,))


def run_gates(
    specs: Sequence[GateSpec], registry: GateRegistry, ctx: GateContext
) -> list[GateResult]:
    """Evaluate every gate (no short-circuit). A gate that cannot be built, raises or returns a
    malformed result yields a fail-closed ``GATE-000`` result instead of an exception."""
    results: list[GateResult] = []
    for spec in specs:
        gate_ctx = dataclasses.replace(ctx, params=spec.params)
        try:
            gate = registry.build(spec)
            result = gate.evaluate(gate_ctx)
            if not isinstance(result, GateResult) or not isinstance(result.decision, GateDecision):
                raise TypeError("gate returned a malformed result")
        except Exception as exc:
            result = _fault(spec.type, ctx, exc)
        results.append(result)
    return results


@dataclass(frozen=True)
class Composite:
    decision: GateDecision
    findings: tuple[Finding, ...]
    questions: tuple[ClarificationQuestion, ...]
    risk: RiskAssessment | None


def compose(results: Sequence[GateResult]) -> Composite:
    """Precedence ``SAFE_STOP > FAIL > AWAIT_INPUT > REQUIRE_APPROVAL > PASS``; findings are unioned
    (exact duplicates once), questions merged by id (first wins), risk is the maximum."""
    present = {r.decision for r in results}
    decision = next((d for d in _PRECEDENCE if d in present), GateDecision.PASS)
    findings = tuple(dict.fromkeys(f for r in results for f in r.findings))
    questions: dict[str, ClarificationQuestion] = {}
    for result in results:
        for question in result.questions:
            questions.setdefault(question.id, question)
    return Composite(
        decision,
        findings,
        tuple(questions.values()),
        _max_risk([r.risk for r in results if r.risk is not None]),
    )


__all__ = [
    "RiskLevel",
    "Composite",
    "compose",
    "default_gate_registry",
    "run_gates",
]
