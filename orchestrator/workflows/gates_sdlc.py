"""The SDLC gate catalogue (``docs/sdlc-contract.md`` section "Skip predicates and gate catalogue").

Every gate is a small class with ``name`` and ``evaluate(ctx: GateContext) -> GateResult`` that the
engine instantiates through the ``GateRegistry`` (``register_sdlc_gates``); the one skip predicate
(``greenfield``) is registered by ``register_sdlc_skips``.

Principles (shared by every gate)
---------------------------------
* **Fail closed.**  Missing, unreadable or malformed evidence is a FAIL with a finding, never a
  pass.  A gate that cannot reach positive evidence (a rule that did not run, a report that checked
  nothing) says so.  A gate never raises on a hostile payload.
* **Evidence is untrusted.**  Artifact payloads are agent output: every string that ends up in a
  finding is control-stripped and length-capped, every id is re-validated against the id grammar of
  ``core.schemas`` and every list is bounded.
* **Deterministic.**  Gate results contain no times, pids, random ids or absolute paths.
* **Findings use catalogue rule ids where one exists** (``QLT-001``, ``CHG-003``, ``CHG-004``,
  ``CMP-001``, ``CMP-003``, ``CHG-005``, ``SEC-*`` re-emitted from the security report).  Gates
  without a catalogue rule report ``GATE-0NN`` ids (``RULE_IDS``; ``GATE-000`` stays reserved for
  the engine's crash handling) so remediation feedback always has a stable rule id and location.

Gate semantics worth knowing
----------------------------
* ``acceptance_criteria_testable`` exempts an active requirement without criteria that the
  requirements artifact lists under ``unsupported``: a human decides about it at ``decompose``
  (``plan_covers_requirements`` answers ``AWAIT_INPUT`` with an ``unsupported`` question).
* ``ambiguity_dispositions_complete`` answers ``AWAIT_INPUT`` (never FAIL) for an ``open``
  ambiguity whose blocking question is listed in ``open_questions``: that is the clarification
  round.  Human answers are looked up in the decision log and the ``clarification_answers``
  artifact.
* ``design_risk`` is the *only* source of the HIGH risk of a design (data model change or breaking
  API change => HIGH; any other API change => MEDIUM; otherwise LOW).  It never lowers the risk the
  architect declared in the design document, and a malformed change list fails instead of
  defaulting to LOW.
* ``scope_within_plan`` mirrors CHG-003 but is plan-aware: the allowed scope is the union of the
  impact report (modules, ``expected_paths``, tests, recipe anchors), the gate params, the node
  params and the task plan entries of this node.  Plan-provided patterns are agent-authored, so a
  universal pattern (``**``) is ignored there; the always-on CHG-003 policy rule (impact report
  only) still applies.
* ``traceability`` / ``release_ready`` delegate to the policy engine (``RunSubject`` evaluated at
  the gate's node position) and only look at their own rules; a rule that is missing from
  ``PolicyReport.rules_evaluated`` is a FAIL.
* ``baseline_unchanged`` compares the target tree (run parameter ``apply_root``) with the baseline.
  The optional run parameters ``baseline_include`` / ``baseline_exclude`` must mirror the
  scenario's baseline selection, otherwise files outside the selection count as drift (strict,
  safe default).  Without ``apply_root`` the gate passes, except on a side-effecting node (a real
  apply without a known target fails closed).
* ``release_verified`` treats ``healthz_ok = None`` as "not applicable" (a standalone module has no
  service); anything but ``None``/``True`` fails.

Layering: L5 (workflows) - imports every lower layer; nothing imports it except the registry wiring.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from orchestrator.agents.base import question_id
from orchestrator.core.errors import ConfigError, OrchestratorError
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.paths import NODE_ID_RE, RULE_ID_RE, match_glob
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.schemas import AC_ID_RE, QUESTION_ID_RE, REQ_ID_RE, Kind
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import (
    CHANGESET_KINDS,
    EXTERNAL_PRODUCER,
    ActorType,
    Artifact,
    ChangeSet,
    ClarificationQuestion,
    DecisionKind,
    FileOpKind,
    FileView,
    Finding,
    GateContext,
    GateDecision,
    GateResult,
    NodeSpec,
    PolicyCategory,
    RiskAssessment,
    RiskLevel,
    RunSubject,
    RunView,
    Severity,
)
from orchestrator.engine.registry import GateRegistry, SkipRegistry
from orchestrator.policies.hits import cap_evidence
from orchestrator.policies.pyscan import compile_check, parse_python
from orchestrator.tools.workspace import DEFAULT_EXCLUDES, DirectoryView

# Gate-specific finding ids (GATE-000 is reserved for the engine's crash handling).
RULE_IDS: Mapping[str, str] = {
    "acceptance_criteria_testable": "GATE-001",
    "ambiguity_dispositions_complete": "GATE-002",
    "design_risk": "GATE-003",
    "design_covers_requirements": "GATE-004",
    "plan_covers_requirements": "GATE-005",
    "tests_cover_acceptance_criteria": "GATE-006",
    "integration_clean": "GATE-007",
    "tests_passed": "GATE-008",
    "fail_to_pass": "GATE-009",
    "security_clean": "GATE-010",
    "baseline_unchanged": "GATE-011",
    "release_verified": "GATE-012",
    "traceability": "GATE-013",
    "release_ready": "GATE-014",
}

MAX_FINDINGS = 20  # per gate result: feedback must stay readable
_MAX_ITEMS = 2000  # per payload list that a gate iterates
_MAX_TEXT = 2000  # characters of untrusted text a heuristic looks at
_MAX_LINE = 2000  # characters of a docstring line that is parsed
ENGINE_FAULT_RULE = "POL-000"

_PASS, _FAIL = GateDecision.PASS, GateDecision.FAIL
_HIGH, _MEDIUM = Severity.HIGH, Severity.MEDIUM
# What reading an artifact (or a run view) can raise (deeply nested hostile JSON: RecursionError).
_READ_ERRORS: tuple[type[BaseException], ...] = (
    OrchestratorError,
    KeyError,
    ValueError,
    OSError,
    RecursionError,
)
# ... and what turning a hostile payload into a ``ChangeSet`` can raise on top of that.
_CHANGESET_ERRORS: tuple[type[BaseException], ...] = (*_READ_ERRORS, TypeError, AttributeError)


class _Stop(Exception):
    """Unwinds a gate with a ready-made result (missing evidence)."""

    def __init__(self, result: GateResult) -> None:
        super().__init__(result.message)
        self.result = result


def _clip(value: object, limit: int = 120) -> str:
    return sanitize_for_terminal(value, limit)


def _items(value: object) -> list[Mapping[str, Any]]:
    """The mapping entries of a payload list (bounded; anything else is ignored)."""
    if not isinstance(value, list):
        return []
    return [v for v in value[:_MAX_ITEMS] if isinstance(v, Mapping)]


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value[:_MAX_ITEMS] if isinstance(v, str)]


class _GateBase:
    """Template: ``check`` raises ``_Stop`` (via ``load``) when evidence is unavailable."""

    name: str = ""
    category: PolicyCategory = PolicyCategory.QUALITY

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params: Mapping[str, Any] = dict(params or {})

    @property
    def rule_id(self) -> str:
        return RULE_IDS[self.name]

    def evaluate(self, ctx: GateContext) -> GateResult:
        try:
            return self.check(ctx)
        except _Stop as stop:
            return stop.result

    def check(self, ctx: GateContext) -> GateResult:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- results -------------------------------------------------------------------
    def finding(
        self,
        message: str,
        *,
        location: str | None = None,
        evidence: str | None = None,
        severity: Severity = _HIGH,
        remediation: str | None = None,
        rule_id: str | None = None,
        category: PolicyCategory | None = None,
    ) -> Finding:
        return Finding(
            rule_id or self.rule_id,
            category or self.category,
            severity,
            _clip(message, 200),
            location=_clip(location, 120) if location else None,
            evidence=cap_evidence(_clip(evidence, 120), 80) if evidence else None,
            remediation=_clip(remediation, 300) if remediation else None,
        )

    def result(
        self,
        decision: GateDecision,
        message: str,
        findings: Iterable[Finding] = (),
        *,
        risk: RiskAssessment | None = None,
        questions: Sequence[ClarificationQuestion] = (),
        **evidence: Any,
    ) -> GateResult:
        return GateResult(
            self.name,
            decision,
            message,
            tuple(findings)[:MAX_FINDINGS],
            risk,
            tuple(questions),
            evidence,
        )

    def passed(self, message: str = "", **evidence: Any) -> GateResult:
        return self.result(_PASS, message or "ok", **evidence)

    def failed(self, message: str, findings: Iterable[Finding] = (), **evidence: Any) -> GateResult:
        found = list(findings) or [self.finding(message)]
        return self.result(_FAIL, message, found, **evidence)

    def count(self, payload: Mapping[str, Any], key: str) -> int:
        value = payload.get(key)
        if not (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            raise self.unavailable(f"the {key} counter")
        return value

    def unavailable(self, what: str) -> _Stop:
        return _Stop(self.failed(f"{what} is unavailable or unreadable (fail closed)"))

    # -- evidence ------------------------------------------------------------------
    def _produced(self, ctx: GateContext, kind: str) -> Artifact | None:
        return next((a for a in ctx.produced if a.kind == kind and a.key == ""), None)

    def try_load(
        self, ctx: GateContext, kind: str, *, from_run: bool = False
    ) -> Mapping[str, Any] | None:
        """The JSON object of ``kind`` (the attempt's own output first), or ``None``.

        ``from_run=True`` reads through the ``RunView`` for kinds the node does not declare; only
        artifacts of strict ancestors (or external inputs) count then.
        """
        try:
            if from_run:
                art = ctx.run_view.latest(kind)
                if art is None or art.producer not in (
                    ctx.run_view.ancestors(ctx.node.id) | {EXTERNAL_PRODUCER}
                ):
                    return None
                payload = ctx.run_view.read_json(art)
            else:
                art = self._produced(ctx, kind) or ctx.reader.latest(kind)
                if art is None:
                    return None
                payload = ctx.reader.read_json(art)
        except _READ_ERRORS:
            return None
        return payload if isinstance(payload, Mapping) else None

    def load(self, ctx: GateContext, kind: str, *, from_run: bool = False) -> Mapping[str, Any]:
        payload = self.try_load(ctx, kind, from_run=from_run)
        if payload is None:
            raise self.unavailable(f"the {kind} artifact")
        return payload

    def load_requirements(self, ctx: GateContext) -> Mapping[str, Any]:
        """The requirements payload; a missing or empty requirement list is a FAIL (fail closed)."""
        requirements = self.load(ctx, Kind.REQUIREMENTS)
        if not _items(requirements.get("requirements")):
            raise _Stop(self.failed("the requirements artifact lists no requirements"))
        return requirements

    def human_answers(self, ctx: GateContext) -> tuple[frozenset[str], frozenset[str]]:
        """Question ids a human answered / decided on: ``(clarified, scoped)``.

        ``clarified`` = HUMAN CLARIFICATION decisions (``data.question_id``) plus the keys of the
        cumulative ``clarification_answers`` artifact (external input an agent cannot write);
        ``scoped`` = HUMAN SCOPE decisions that name a question.  Agent decisions never count.
        """
        clarified: set[str] = set()
        scoped: set[str] = set()
        for decision in ctx.run_view.decisions():
            if decision.actor.type != ActorType.HUMAN:
                continue
            qid = decision.data.get("question_id") if isinstance(decision.data, Mapping) else None
            if not isinstance(qid, str):
                continue
            if decision.kind == DecisionKind.CLARIFICATION:
                clarified.add(qid)
            elif decision.kind == DecisionKind.SCOPE:
                scoped.add(qid)
        answers = self.try_load(ctx, Kind.CLARIFICATION_ANSWERS, from_run=True)
        raw = answers.get("answers") if answers is not None else None
        if isinstance(raw, Mapping):
            clarified.update(k for k in raw if isinstance(k, str) and QUESTION_ID_RE.match(k))
        return frozenset(clarified), frozenset(scoped)

    def changesets(self, ctx: GateContext) -> list[tuple[Artifact, ChangeSet]]:
        """The attempt's staged change sets (every ``CHANGESET_KINDS`` artifact it produced)."""
        found: list[tuple[Artifact, ChangeSet]] = []
        for art in ctx.produced:
            if art.kind not in CHANGESET_KINDS:
                continue
            try:
                payload = ctx.reader.read_json(art)
                found.append((art, ChangeSet.from_dict(payload)))
            except _CHANGESET_ERRORS:
                raise _Stop(
                    self.failed(
                        "a staged change set cannot be read (fail closed)",
                        [self.finding("unreadable change set", location=art.id)],
                    )
                ) from None
        if not found:
            raise self.unavailable("a staged change set")
        return found


# ---------------------------------------------------------------------------- requirements helpers

_STATUSES = frozenset({"active", "deferred", "descoped"})
# Pseudo requirement id of the requirements agent for "the request as a whole".
_REQUEST_SCOPE = "request"


def _active_requirements(requirements: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in _items(requirements.get("requirements")) if r.get("status") == "active"]


def _criteria(requirement: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _items(requirement.get("acceptance_criteria"))


def _requirement_id(requirement: Mapping[str, Any]) -> str:
    rid = requirement.get("id")
    return rid if isinstance(rid, str) and REQ_ID_RE.match(rid) else "R-???"


def _unsupported_ids(requirements: Mapping[str, Any]) -> set[str]:
    """Requirement ids the requirements artifact reports as deliverable by no capability."""
    return {
        entry["requirement_id"]
        for entry in _items(requirements.get("unsupported"))
        if isinstance(entry.get("requirement_id"), str)
    }


# ---------------------------------------------------------------------------- 1. testable criteria

_VAGUE = re.compile(
    r"\b(fast|quick(?:ly)?|easy|easily|simple|simply|user[- ]friendly|intuitive|robust|scalable|"
    r"good|nice|better|best|well|properly|adequate(?:ly)?|sufficient(?:ly)?|decent|enough|"
    r"efficient(?:ly)?|performant|seamless(?:ly)?|reliable|appropriate(?:ly)?|reasonabl[ey]|"
    r"flexible|modern|clean|secure|safe(?:r|ly)?|smooth(?:ly)?|improved?|enhanced?|"
    r"optimi[sz]ed?|large|small|many|few|several|lots?|etc)\b",
    re.IGNORECASE,
)
_COMPARATOR = re.compile(
    r"\b(at least|at most|no more than|no less than|less than|more than|greater than|"
    r"fewer than|within|exactly|equals?|equal to|between)\b",
    re.IGNORECASE,
)
_OBSERVABLE = re.compile(
    r"\b(?:return|respond|answer|reject|redirect|creat|record|stor|contain|rais|fail|succeed|"
    r"increment|decrement|delet|expir|log|emit|includ|exclud|show|list|export|report|persist|"
    r"set|mark|stop|block|allow|den(?:y|ie)|accept|produc|writ|read|send|yield|classif|abandon|"
    r"retr(?:y|ie)|prefix|state|never|only|without|inject|work|run|follow|contact|verif|prevent|"
    r"refus|display|render|valid|detect|comput|calculat|match|convert|trigger|notif|hash|"
    r"encrypt|sanitis|sanitiz|escap|quot|limit|bound|expos|requir|authent|authori[sz]|forbid|"
    r"permit|reach|resolv|bind|serve|download|upload|insert|updat|query|enforc|appl(?:y|ie))\w*\b",
    re.IGNORECASE,
)
_LITERAL = re.compile(
    r"`[^`\n]{1,80}`|\"[^\"\n]{1,80}\"|'[^'\n]{1,80}'|\b[a-z]+_[a-z_]+\b|\b[45]\d\d\b"
)
_DIGIT = re.compile(r"\d")
_MIN_WORDS = 4


def _is_sentence(text: str) -> bool:
    return len(text.strip()[:_MAX_TEXT].split()) >= _MIN_WORDS


def is_measurable(text: str) -> bool:
    """Heuristic: an acceptance criterion names an observable outcome with a concrete anchor.

    Measurable = at least four words and either a concrete anchor (a digit, a comparator such as
    "at most", a quoted literal, an identifier, an HTTP status) or an observable verb ("returns",
    "rejects", ...) without vague wording ("fast", "user-friendly", ...).
    """
    sample = text.strip()[:_MAX_TEXT]
    if not _is_sentence(sample):
        return False
    if _DIGIT.search(sample) or _COMPARATOR.search(sample) or _LITERAL.search(sample):
        return True
    return bool(_OBSERVABLE.search(sample)) and not _VAGUE.search(sample)


class AcceptanceCriteriaTestable(_GateBase):
    """Every active requirement has at least one acceptance criterion with measurable text.

    Exempt: an active requirement that has no criterion *and* is listed under ``unsupported`` (no
    capability can deliver it): ``plan_covers_requirements`` stops the run with a human question for
    it.  A ``verification=inspection`` requirement is judged by a human, so its criterion only has
    to be a sentence.  Unknown statuses, invalid or duplicate ids fail (the artifact is malformed).
    """

    name = "acceptance_criteria_testable"

    def check(self, ctx: GateContext) -> GateResult:
        requirements = self.load_requirements(ctx)
        listed = _items(requirements.get("requirements"))
        unsupported = _unsupported_ids(requirements)
        findings: list[Finding] = []
        seen_requirements: set[str] = set()
        seen_criteria: set[str] = set()
        active = 0
        for req in listed:
            rid = _requirement_id(req)
            status = req.get("status")
            if not (isinstance(status, str) and status in _STATUSES):  # may be unhashable junk
                findings.append(
                    self.finding(
                        "requirement has an unknown status",
                        location=rid,
                        evidence=status if isinstance(status, str) else type(status).__name__,
                    )
                )
                continue
            if rid == "R-???" or rid in seen_requirements:
                findings.append(
                    self.finding(
                        "requirement id is invalid or duplicated",
                        location=rid,
                        evidence=_clip(req.get("id"), 40),
                    )
                )
            seen_requirements.add(rid)
            if status != "active":
                continue
            active += 1
            findings.extend(self._check_active(req, rid, unsupported, seen_criteria))
        if findings:
            return self.failed(
                f"{len(findings)} acceptance criteria problem(s)", findings, active=active
            )
        if active == 0:
            return self.failed("the requirements artifact has no active requirement to deliver")
        return self.passed(f"{active} active requirement(s) have testable criteria", active=active)

    def _check_active(
        self,
        req: Mapping[str, Any],
        rid: str,
        unsupported: set[str],
        seen_criteria: set[str],
    ) -> list[Finding]:
        criteria = _criteria(req)
        if not criteria:
            if rid in unsupported:
                return []  # a human decides at ``decompose`` (plan_covers_requirements)
            return [
                self.finding(
                    "active requirement has no acceptance criterion",
                    location=rid,
                    remediation="Add at least one measurable acceptance criterion.",
                )
            ]
        inspection = req.get("verification") == "inspection"
        findings: list[Finding] = []
        measurable = False
        for crit in criteria:
            cid = crit.get("id")
            if not (isinstance(cid, str) and AC_ID_RE.match(cid)):
                findings.append(
                    self.finding(
                        "acceptance criterion has an invalid id",
                        location=rid,
                        evidence=_clip(cid, 40),
                    )
                )
            elif cid in seen_criteria:
                findings.append(
                    self.finding("duplicate acceptance criterion id", location=f"{rid}/{cid}")
                )
            else:
                seen_criteria.add(cid)
            text = crit.get("text")
            if isinstance(text, str) and (
                is_measurable(text) or (inspection and _is_sentence(text))
            ):
                measurable = True
        if not measurable:
            first = next((c["text"] for c in criteria if isinstance(c.get("text"), str)), None)
            findings.append(
                self.finding(
                    "no acceptance criterion of the requirement is measurable",
                    location=rid,
                    evidence=first,
                    remediation=(
                        "State an observable outcome with a concrete value, status code or "
                        "threshold."
                    ),
                )
            )
        return findings


# ---------------------------------------------------------------------------- 2. ambiguities


def _clean_question(question: ClarificationQuestion) -> ClarificationQuestion:
    """A copy whose strings are control-free and bounded (questions are shown to a human)."""
    return dataclasses.replace(
        question,
        text=_clip(question.text, 500),
        rationale=_clip(question.rationale, 300),
        options=tuple(_clip(option, 120) for option in question.options[:20]),
        default_assumption=(
            None if question.default_assumption is None else _clip(question.default_assumption, 300)
        ),
        related_requirements=tuple(_clip(r, 20) for r in question.related_requirements[:20]),
        category=_clip(question.category, 20),
    )


def _pending_question(raw: Mapping[str, Any] | None) -> ClarificationQuestion | None:
    """The question an ``open`` ambiguity waits for (``None``: absent, malformed or invalid id)."""
    if raw is None:
        return None
    try:
        question = ClarificationQuestion.from_dict(raw)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    if not (isinstance(question.id, str) and QUESTION_ID_RE.match(question.id)):
        return None
    if not isinstance(question.text, str) or not question.text.strip():
        return None
    return _clean_question(question)


class AmbiguityDispositionsComplete(_GateBase):
    """Every ambiguity is answered, assumed (with an assumption entry) or deferred (human decision).

    * ``answered`` needs the answer text and a human answer on record (a HUMAN CLARIFICATION
      decision or the ``clarification_answers`` artifact); ``deferred`` needs a human decision for
      the question; ``assumed`` needs an ``assumptions`` entry for it.
    * ``open`` is the clarification round, not a defect: when ``open_questions`` carries the
      blocking question the result is ``AWAIT_INPUT`` (a FAIL would win over
      ``no_blocking_questions`` and the human would never be asked).  An open ambiguity without
      such a question is a FAIL.

    Returns a MEDIUM ``RiskAssessment`` when any ambiguity is ``assumed`` so the requirements
    approval (threshold MEDIUM) is raised.
    """

    name = "ambiguity_dispositions_complete"

    def check(self, ctx: GateContext) -> GateResult:
        requirements = self.load_requirements(ctx)
        if not isinstance(requirements.get("ambiguities"), list):
            raise self.unavailable("the ambiguity list")
        clarified, scoped = self.human_answers(ctx)
        assumed_q = {
            a["question_id"]
            for a in _items(requirements.get("assumptions"))
            if isinstance(a.get("question_id"), str)
        }
        raw_questions = {
            q["id"]: q
            for q in _items(requirements.get("open_questions"))
            if isinstance(q.get("id"), str)
        }
        findings: list[Finding] = []
        assumed: list[str] = []
        waiting: dict[str, ClarificationQuestion] = {}
        for amb in _items(requirements.get("ambiguities")):
            amb_id, qid = _clip(amb.get("id"), 20), amb.get("question_id")
            disposition = amb.get("disposition")
            problem: str | None = None
            if not (isinstance(qid, str) and QUESTION_ID_RE.match(qid)):
                problem = "ambiguity has no valid question id"
            elif disposition == "answered":
                answer = amb.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    problem = "answered ambiguity has no answer text"
                elif qid not in clarified:
                    problem = "answered ambiguity has no human answer on record"
            elif disposition == "assumed":
                assumed.append(amb_id)
                if qid not in assumed_q:
                    problem = "assumed ambiguity has no recorded assumption"
            elif disposition == "deferred":
                if qid not in clarified and qid not in scoped:
                    problem = "deferred ambiguity has no human decision on record"
            elif disposition == "open":
                question = _pending_question(raw_questions.get(qid))
                if qid in clarified:
                    problem = "ambiguity is still open although a human answered its question"
                elif question is None or not question.blocking:
                    problem = "open ambiguity has no blocking question to ask"
                else:
                    waiting[qid] = question
            else:
                problem = f"ambiguity has an unknown disposition {_clip(disposition, 20)}"
            if problem:
                findings.append(
                    self.finding(
                        problem,
                        location=amb_id,
                        remediation="Answer it, record an assumption, or defer it (human).",
                    )
                )
        if findings:
            return self.failed(f"{len(findings)} ambiguity problem(s)", findings)
        if waiting:
            return self.result(
                GateDecision.AWAIT_INPUT,
                f"{len(waiting)} ambiguity(ies) wait for a human answer",
                questions=[waiting[qid] for qid in sorted(waiting)],
                waiting=sorted(waiting),
            )
        risk = (
            RiskAssessment(
                RiskLevel.MEDIUM,
                (f"{len(assumed)} ambiguity(ies) resolved by assumption",),
                {"assumed": sorted(assumed)},
            )
            if assumed
            else None
        )
        return self.result(
            _PASS, "all ambiguities have a disposition", risk=risk, assumed=sorted(assumed)
        )


# ---------------------------------------------------------------------------- 3. design gates


class DesignRisk(_GateBase):
    """HIGH for a data model change or a breaking API change; MEDIUM for any other API change.

    The only source of the HIGH risk of a design; it never lowers the risk the architect declared.
    A malformed change list (or breaking flag) fails: the risk cannot be assessed, so it must not
    default to LOW.
    """

    name = "design_risk"

    def _changes(self, design: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
        value = design.get(key)
        if (
            not isinstance(value, list)
            or len(value) > _MAX_ITEMS
            or not all(isinstance(v, Mapping) for v in value)
        ):
            raise self.unavailable(f"the design's {key} list")
        return list(value)

    def check(self, ctx: GateContext) -> GateResult:
        design = self.load(ctx, Kind.DESIGN_DOC)
        contract = self.try_load(ctx, Kind.API_CONTRACT)
        data_changes = self._changes(design, "data_model_changes")
        api_changes = self._changes(design, "api_changes")
        breaking = 0
        for change in api_changes:
            flag = change.get("breaking")
            if not isinstance(flag, bool):
                raise self.unavailable("an api_changes breaking flag")
            if flag or change.get("change") == "remove":
                breaking += 1
        declared_breaking = False
        if contract is not None:
            flag = contract.get("breaking")
            if not isinstance(flag, bool):
                raise self.unavailable("the api_contract breaking flag")
            declared_breaking = flag
        level = RiskLevel.LOW
        reasons: list[str] = []
        if data_changes:
            level = RiskLevel.HIGH
            reasons.append(f"{len(data_changes)} data model change(s) (migration)")
        if breaking or declared_breaking:
            level = RiskLevel.HIGH
            reasons.append("breaking API change")
        if level < RiskLevel.MEDIUM and api_changes:
            level = RiskLevel.MEDIUM
            reasons.append("public API surface changes")
        declared = design.get("risk")
        declared_name = declared.get("level") if isinstance(declared, Mapping) else None
        if (
            isinstance(declared_name, str)
            and declared_name in RiskLevel.__members__
            and RiskLevel[declared_name] > level
        ):
            level = RiskLevel[declared_name]
            reasons.append(f"architect declared {declared_name} risk")
        risk = RiskAssessment(
            level,
            tuple(reasons) or ("no data model or API surface change",),
            {"data_model_changes": len(data_changes), "api_changes": len(api_changes)},
        )
        return self.result(_PASS, f"design risk {level.name}", risk=risk, level=level.name)


class DesignCoversRequirements(_GateBase):
    """Every active requirement maps to at least one design component.

    No exemption: the architect lists requirements no capability can deliver in an explicit
    "unsupported scope" component, so they stay visible here until ``plan_covers_requirements``
    asks a human what to do with them.
    """

    name = "design_covers_requirements"

    def check(self, ctx: GateContext) -> GateResult:
        requirements = self.load_requirements(ctx)
        design = self.load(ctx, Kind.DESIGN_DOC)
        covered = {
            rid
            for comp in _items(design.get("components"))
            for rid in _strings(comp.get("requirement_ids"))
        }
        active = [_requirement_id(r) for r in _active_requirements(requirements)]
        missing = [rid for rid in active if rid not in covered]
        if not active:
            return self.failed("the requirements artifact has no active requirement to cover")
        if missing:
            return self.failed(
                f"{len(missing)} requirement(s) are not covered by a design component",
                [
                    self.finding(
                        "requirement is not mapped to any design component",
                        location=rid,
                        remediation="Add or extend a component whose requirement_ids lists it.",
                    )
                    for rid in missing
                ],
                missing=missing[:MAX_FINDINGS],
            )
        return self.passed(f"{len(active)} requirement(s) covered", covered=len(active))


# ---------------------------------------------------------------------------- 4. plan coverage


def unsupported_question_id(requirement_id: str) -> str:
    """Id of the human question about an unsupported requirement (derived from the id alone)."""
    return question_id("unsupported", f"unsupported requirement {requirement_id}")


def unsupported_question(requirement_id: str, reason: str) -> ClarificationQuestion:
    """Blocking question for a requirement no capability can deliver (category ``unsupported``).

    The id is derived from the requirement id only, so an answer survives re-planning and
    amendments (``plan_covers_requirements`` treats an answered question as decided).
    """
    subject = (
        "Part of the request"
        if requirement_id == _REQUEST_SCOPE
        else f"Requirement {requirement_id}"
    )
    return ClarificationQuestion(
        id=unsupported_question_id(requirement_id),
        text=(
            f"{subject} cannot be delivered by any available capability ({_clip(reason, 100)}). "
            "Descope or defer it, or amend the request to describe a supported capability."
        ),
        rationale="The reference agents never guess: an unsupported need requires a human.",
        options=("descope", "defer"),
        default_assumption=None,
        blocking=True,
        related_requirements=(requirement_id,),
        category="unsupported",
    )


class PlanCoversRequirements(_GateBase):
    """Every active requirement has a task; implement tasks live in ``implement.*`` groups.

    ``test`` and ``doc`` tasks are the declared non-implementation work (``author-tests`` /
    ``docs``); any other kind fails.  When the ``plan_expansion`` artifact is available the group's
    node id must be one of its nodes.  Requirements the plan (or the requirements artifact) lists as
    ``unsupported`` need no task: they turn the result into ``AWAIT_INPUT`` (a human must descope or
    defer) once the rest of the plan is structurally sound, unless a human already answered that
    question (the decision is on record and the run moves on; ``traceability`` has the last word).
    """

    name = "plan_covers_requirements"

    def check(self, ctx: GateContext) -> GateResult:
        requirements = self.load_requirements(ctx)
        plan = self.load(ctx, Kind.TASK_PLAN)
        expansion = self.try_load(ctx, Kind.PLAN_EXPANSION)
        known = {_requirement_id(r) for r in _items(requirements.get("requirements"))}
        active = {_requirement_id(r): r for r in _active_requirements(requirements)}
        unsupported: dict[str, str] = {}
        for entry in (*_items(plan.get("unsupported")), *_items(requirements.get("unsupported"))):
            rid = entry.get("requirement_id")
            if isinstance(rid, str) and (rid in active or rid == _REQUEST_SCOPE):
                unsupported.setdefault(rid, _clip(entry.get("reason"), 100))
        clarified, _scoped = self.human_answers(ctx)
        undecided = {
            rid: reason
            for rid, reason in unsupported.items()
            if unsupported_question_id(rid) not in clarified
        }
        tasks, groups = _items(plan.get("tasks")), _items(plan.get("groups"))
        group_nodes = {str(g.get("id")): str(g.get("node_id")) for g in groups}
        expansion_ids = (
            {str(n.get("id")) for n in _items(expansion.get("nodes"))}
            if expansion is not None
            else None
        )
        findings: list[Finding] = []
        covered: set[str] = set()
        for task in tasks:
            tid = _clip(task.get("id"), 20)
            ids = _strings(task.get("requirement_ids"))
            covered.update(ids)
            findings.extend(
                self.finding("task references an unknown requirement", location=tid, evidence=ref)
                for ref in ids
                if ref not in known
            )
            kind = task.get("kind")
            if kind in ("test", "doc"):
                continue
            group = task.get("group")
            node_id = group_nodes.get(str(group))
            if kind != "implement":
                problem, evidence = "task has an unknown kind", kind
            elif node_id is None:
                problem, evidence = "task belongs to an unknown group", group
            elif not (node_id.startswith("implement.") and NODE_ID_RE.match(node_id)):
                problem, evidence = "task group has no implement.* node", group
            elif expansion_ids is not None and node_id not in expansion_ids:
                problem, evidence = "task group's node is not part of the plan expansion", group
            else:
                continue
            findings.append(self.finding(problem, location=tid, evidence=_clip(evidence, 40)))
        findings.extend(
            self.finding(
                "active requirement has no task",
                location=rid,
                remediation="Add a task for the requirement or report it as unsupported.",
            )
            for rid in active
            if rid not in covered and rid not in unsupported
        )
        if findings:
            return self.failed(f"{len(findings)} plan coverage problem(s)", findings)
        if undecided:
            ordered = sorted(undecided)
            return self.result(
                GateDecision.AWAIT_INPUT,
                f"{len(ordered)} requirement(s) are unsupported and need a human decision",
                [
                    self.finding(
                        "requirement is unsupported by every capability",
                        location=rid,
                        severity=_MEDIUM,
                        evidence=undecided[rid],
                    )
                    for rid in ordered
                ],
                questions=[unsupported_question(rid, undecided[rid]) for rid in ordered],
                unsupported=ordered,
            )
        return self.passed(
            f"{len(active)} requirement(s) planned",
            tasks=len(tasks),
            decided_unsupported=sorted(set(unsupported) - set(undecided)),
        )


# ---------------------------------------------------------------------------- 5. change set gates


class SyntaxValid(_GateBase):
    """QLT-001: every created/modified ``.py`` file of the staged change sets compiles."""

    name = "syntax_valid"
    category = PolicyCategory.QUALITY

    @property
    def rule_id(self) -> str:
        return "QLT-001"

    def check(self, ctx: GateContext) -> GateResult:
        findings: list[Finding] = []
        checked = 0
        for _art, cs in self.changesets(ctx):
            for op in cs.ops:
                if op.kind == FileOpKind.DELETE or not op.path.lower().endswith(".py"):
                    continue
                checked += 1
                hit = compile_check(op.content or "", op.path)
                if hit is not None:
                    findings.append(
                        self.finding(
                            hit.message,
                            location=hit.location,
                            remediation="Fix the syntax error before proposing the change.",
                        )
                    )
        if findings:
            return self.failed(
                f"{len(findings)} Python file(s) do not compile", findings, checked=checked
            )
        return self.passed(f"{checked} Python file(s) compile", checked=checked)


_UNIVERSAL_PATTERNS = frozenset({"**", "**/*"})


def _globs(value: object, *, plan_authored: bool) -> list[str]:
    patterns = [p for p in _strings(value) if p]
    if plan_authored:
        patterns = [p for p in patterns if p not in _UNIVERSAL_PATTERNS]
    return patterns


def _any_glob(path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        try:
            if match_glob(path, pattern):
                return True
        except OrchestratorError:  # PathViolation: an unusable pattern never widens the scope
            continue
    return False


class ScopeWithinPlan(_GateBase):
    """CHG-003: modified or deleted *existing* files must lie inside the planned scope.

    Allowed = impact report (modules, expected paths, tests, recipe anchors) + gate params
    (``expected_paths`` / ``write_scope``) + node params + this node's group/tasks in the task plan.
    Creating new files is never out of scope.  With existing files touched and no scope information
    at all the gate fails closed.
    """

    name = "scope_within_plan"
    category = PolicyCategory.CHANGE_CONTROL

    @property
    def rule_id(self) -> str:
        return "CHG-003"

    def scope(self, ctx: GateContext) -> list[str]:
        patterns: list[str] = []
        impact = self.try_load(ctx, Kind.IMPACT_REPORT)
        if impact is not None:
            patterns += [str(m.get("path")) for m in _items(impact.get("modules")) if m.get("path")]
            for key in ("expected_paths", "tests", "recipe_anchors"):
                patterns += _globs(impact.get(key), plan_authored=False)
        for key in ("expected_paths", "write_scope"):
            patterns += _globs(self.params.get(key), plan_authored=False)
            patterns += _globs(ctx.node.params.get(key), plan_authored=True)
        plan = self.try_load(ctx, Kind.TASK_PLAN)
        if plan is not None:
            group_ids = {
                str(g.get("id"))
                for g in _items(plan.get("groups"))
                if g.get("node_id") == ctx.node.id
            }
            for group in _items(plan.get("groups")):
                if str(group.get("id")) in group_ids:
                    patterns += _globs(group.get("write_scope"), plan_authored=True)
            for task in _items(plan.get("tasks")):
                if str(task.get("group")) in group_ids:
                    for key in ("expected_paths", "write_scope"):
                        patterns += _globs(task.get(key), plan_authored=True)
        return patterns

    def check(self, ctx: GateContext) -> GateResult:
        touched: list[tuple[str, str]] = []
        for _art, cs in self.changesets(ctx):
            touched.extend(
                (op.path, op.kind.value)
                for op in cs.ops
                if op.kind != FileOpKind.CREATE or ctx.baseline.exists(op.path)
            )
        if not touched:
            return self.passed("no existing file is modified", touched=0)
        allowed = self.scope(ctx)
        if not allowed:
            return self.failed(
                "scope unverifiable: existing files are modified, no impact report or plan scope",
                [
                    self.finding(
                        "scope unverifiable (no impact report, no plan scope)",
                        severity=_MEDIUM,
                        evidence=f"modified={len(touched)}",
                        remediation="Run codebase analysis or declare the group's expected paths.",
                    )
                ],
                touched=len(touched),
            )
        outside = [(p, k) for p, k in touched if not _any_glob(p, allowed)]
        if outside:
            return self.failed(
                f"{len(outside)} existing file(s) changed outside the planned scope",
                [
                    self.finding(
                        f"{kind} of an existing file outside the planned scope",
                        location=path,
                        severity=_MEDIUM,
                        evidence="outside planned scope",
                        remediation="Restrict the change to the planned scope or extend the plan.",
                    )
                    for path, kind in outside
                ],
                touched=len(touched),
                outside=len(outside),
            )
        return self.passed(
            f"{len(touched)} existing file(s) inside the planned scope", touched=len(touched)
        )


# ---------------------------------------------------------------------------- 6. tests cover ACs

_COVER_ID = re.compile(r"\b(?:AC-\d{3}-\d{1,2}|R-\d{3})\b")


def covered_ids(docstring: str | None) -> set[str]:
    """Ids named on the ``Covers:`` lines of a test docstring (linear time, bounded line length)."""
    found: set[str] = set()
    for line in (docstring or "").splitlines():
        text = line.strip()[:_MAX_LINE]
        if text.startswith("Covers:"):
            found.update(_COVER_ID.findall(text[len("Covers:") :]))
    return found


class AcceptanceCriteriaCovered(_GateBase):
    """Each active acceptance criterion appears on a ``Covers:`` line of a test docstring.

    Requirements verified by ``inspection`` are exempt (CMP-001 handles them).  Files that do not
    parse contribute no coverage (``syntax_valid`` reports them).
    """

    name = "tests_cover_acceptance_criteria"

    def check(self, ctx: GateContext) -> GateResult:
        requirements = self.load_requirements(ctx)
        covered: set[str] = set()
        tests = unparsed = 0
        for _art, cs in self.changesets(ctx):
            for op in cs.ops:
                if op.kind == FileOpKind.DELETE or not op.path.lower().endswith(".py"):
                    continue
                tree, problem = parse_python(op.content or "", op.path)
                if tree is None or problem is not None:
                    unparsed += 1
                    continue
                for node in ast.walk(tree):
                    if isinstance(
                        node, ast.FunctionDef | ast.AsyncFunctionDef
                    ) and node.name.startswith("test"):
                        tests += 1
                        covered |= covered_ids(ast.get_docstring(node))
        findings: list[Finding] = []
        needed = 0
        for req in _active_requirements(requirements):
            if req.get("verification") == "inspection":
                continue
            rid = _requirement_id(req)
            for crit in _criteria(req):
                cid = crit.get("id")
                if not (isinstance(cid, str) and AC_ID_RE.match(cid)):
                    continue
                needed += 1
                if cid not in covered:
                    findings.append(
                        self.finding(
                            "acceptance criterion is not named on any test's Covers: line",
                            location=f"{rid}/{cid}",
                            remediation="Add a test with a 'Covers: R-..., AC-...' docstring line.",
                        )
                    )
        if findings:
            return self.failed(
                f"{len(findings)} acceptance criteria have no covering test",
                findings,
                criteria=needed,
                tests=tests,
                unparsed_files=unparsed,
            )
        return self.passed(
            f"{needed} criteria covered by {tests} test(s)", criteria=needed, tests=tests
        )


# ---------------------------------------------------------------------------- 7. integration


class IntegrationClean(_GateBase):
    """No unresolved integration conflicts (an automatic serialising re-plan is a limitation)."""

    name = "integration_clean"

    def check(self, ctx: GateContext) -> GateResult:
        report = self.load(ctx, Kind.INTEGRATION_REPORT)
        conflicts = report.get("conflicts")
        if not isinstance(conflicts, list):
            raise self.unavailable("the integration conflict list")
        if not conflicts:
            return self.passed("no integration conflicts", conflicts=0)
        findings = [
            self.finding(
                _clip(c.get("reason") or "two change sets touch the path differently", 120),
                location=_clip(c.get("path"), 120),
                evidence=", ".join(_strings(c.get("authors"))[:4]),
                remediation="Make the parallel groups touch disjoint paths or merge them by hand.",
            )
            if isinstance(c, Mapping)
            else self.finding("malformed conflict entry")
            for c in conflicts[:MAX_FINDINGS]
        ]
        return self.failed(
            f"{len(conflicts)} unresolved integration conflict(s)",
            findings,
            conflicts=len(conflicts),
        )


# ---------------------------------------------------------------------------- 8. verify gates


class SuiteGreen(_GateBase):
    """The suite is green: tests ran, none failed or errored, the perf budget (if checked) held."""

    name = "tests_passed"

    def check(self, ctx: GateContext) -> GateResult:
        report = self.load(ctx, Kind.TEST_REPORT)
        findings: list[Finding] = []
        total, failed, errors = (
            self.count(report, "total"),
            self.count(report, "failed"),
            self.count(report, "errors"),
        )
        if total == 0:
            findings.append(self.finding("no tests ran", remediation="Author and run tests."))
        if report.get("passed") is not True:
            findings.append(self.finding("the test report is not marked passed"))
        listed = _items(report.get("tests"))
        if listed and all(test.get("outcome") == "skipped" for test in listed):
            findings.append(self.finding("every test was skipped", remediation="Run the tests."))
        for test in listed:
            if test.get("outcome") in ("failed", "error"):
                findings.append(
                    self.finding(
                        f"test {test.get('outcome')}",
                        location=_clip(test.get("nodeid"), 160),
                        remediation="Fix the implementation (or the test) so the test passes.",
                    )
                )
        if (failed or errors) and not any(f.location for f in findings):
            findings.append(self.finding(f"{failed} test(s) failed, {errors} error(s)"))
        perf = report.get("perf")
        if isinstance(perf, Mapping) and perf.get("checked") is True and perf.get("ok") is not True:
            findings.append(
                self.finding(
                    "performance budget exceeded",
                    remediation="Bring the hot path inside the budget.",
                )
            )
        if findings:
            return self.failed(
                f"suite is not green ({failed} failed, {errors} errors, {total} total)",
                findings,
                total=total,
                failed=failed,
                errors=errors,
            )
        return self.passed(f"{total} test(s) green", total=total)


class FailToPass(_GateBase):
    """Authored tests fail on the baseline and pass on the integrated view (check ran and held)."""

    name = "fail_to_pass"

    def check(self, ctx: GateContext) -> GateResult:
        report = self.load(ctx, Kind.TEST_REPORT)
        f2p = report.get("fail_to_pass")
        if not isinstance(f2p, Mapping) or f2p.get("checked") is not True:
            return self.failed("the fail-to-pass check did not run (fail closed)")
        if f2p.get("ok") is not True:
            # ``tests`` flipped as required; ``vacuous`` = authored tests passing without the change
            vacuous = _strings(f2p.get("vacuous"))
            findings = [
                self.finding(
                    "authored test already passes without the change",
                    location=_clip(name, 160),
                    remediation="Assert the new behaviour so the test fails without the change.",
                )
                for name in vacuous[:MAX_FINDINGS]
            ]
            return self.failed(
                "authored tests do not all fail on the baseline and pass on the integrated view",
                findings
                or [self.finding("not every authored test flipped from failing to passing")],
                flipped=len(_strings(f2p.get("tests"))),
                vacuous=len(vacuous),
            )
        return self.passed("authored tests fail on the baseline and pass on the integrated view")


class ApiCompat(_GateBase):
    """CHG-004: breaking API changes must have been declared in the ``api_contract``."""

    name = "api_compat"
    category = PolicyCategory.CHANGE_CONTROL

    @property
    def rule_id(self) -> str:
        return "CHG-004"

    def check(self, ctx: GateContext) -> GateResult:
        report = self.load(ctx, Kind.TEST_REPORT)
        contract = self.load(ctx, Kind.API_CONTRACT, from_run=True)
        compat = report.get("api_compat")
        if not isinstance(compat, Mapping) or not isinstance(contract.get("breaking"), bool):
            raise self.unavailable("the API compatibility evidence")
        if compat.get("checked") is not True:
            if ctx.baseline.list_files("**"):
                return self.failed(
                    "API compatibility was not checked against a non-empty baseline (fail closed)",
                    [
                        self.finding(
                            "API compatibility is unverifiable",
                            remediation="Export the OpenAPI schema before and after the change.",
                        )
                    ],
                )
            return self.passed("no baseline API to compare with", checked=False)
        raw = compat.get("breaking")
        if not isinstance(raw, list):
            raise self.unavailable("the list of breaking API changes")
        breaking = [entry if isinstance(entry, str) else "malformed entry" for entry in raw]
        if breaking and contract.get("breaking") is not True:
            return self.failed(
                f"{len(breaking)} undeclared breaking API change(s)",
                [
                    self.finding(
                        "breaking public API change that the design did not declare",
                        evidence=entry,
                        remediation="Make the change additive or declare it in the design (HIGH).",
                    )
                    for entry in breaking[:MAX_FINDINGS]
                ],
                breaking=len(breaking),
            )
        return self.passed(
            f"{len(breaking)} breaking change(s), all declared"
            if breaking
            else "no breaking API change",
            breaking=len(breaking),
        )


# ---------------------------------------------------------------------------- 9. security


def _severity(value: object) -> Severity:
    return (
        Severity[value]
        if isinstance(value, str) and value in Severity.__members__
        else Severity.HIGH
    )


class SecurityClean(_GateBase):
    """No HIGH or CRITICAL security finding; a CRITICAL finding requests a safe stop."""

    name = "security_clean"
    category = PolicyCategory.SECURITY

    def check(self, ctx: GateContext) -> GateResult:
        report = self.load(ctx, Kind.SECURITY_REPORT)
        if not _strings(report.get("rules_evaluated")):
            return self.failed("the security scan evaluated no rules (fail closed)")
        listed = report.get("findings")
        if not isinstance(listed, list):
            raise self.unavailable("the security findings list")
        findings: list[Finding] = []
        worst = Severity.INFO
        malformed = sum(1 for entry in listed[:_MAX_ITEMS] if not isinstance(entry, Mapping))
        if malformed or len(listed) > _MAX_ITEMS:
            worst = Severity.HIGH
            findings.append(self.finding(f"{malformed} malformed or excess security finding(s)"))
        for raw in _items(listed):
            severity = _severity(raw.get("severity"))
            if severity < Severity.HIGH:
                continue
            worst = max(worst, severity)
            rule = raw.get("rule_id")
            category = raw.get("category")
            findings.append(
                self.finding(
                    str(raw.get("message") or "security finding"),
                    location=str(raw["location"]) if raw.get("location") else None,
                    evidence=str(raw["evidence"]) if raw.get("evidence") else None,
                    severity=severity,
                    remediation=str(raw["remediation"]) if raw.get("remediation") else None,
                    rule_id=rule if isinstance(rule, str) and RULE_ID_RE.match(rule) else None,
                    category=(
                        PolicyCategory(category)
                        if isinstance(category, str)
                        and category in {c.value for c in PolicyCategory}
                        else None
                    ),
                )
            )
        if report.get("passed") is not True and not findings:
            findings.append(self.finding("the security report is not marked passed"))
        if findings:
            decision = GateDecision.SAFE_STOP if worst >= Severity.CRITICAL else _FAIL
            return self.result(decision, f"{len(findings)} HIGH+ security finding(s)", findings)
        return self.passed("no HIGH or CRITICAL security finding")


# ---------------------------------------------------------------------------- 10. run-level gates


class _PolicyGate(_GateBase):
    """Delegates to the policy engine: ``RunSubject`` evaluated at this node's position."""

    rules: tuple[str, ...] = ()
    category = PolicyCategory.COMPLIANCE

    def check(self, ctx: GateContext) -> GateResult:
        try:
            report = ctx.policy.evaluate(RunSubject(ctx.run_id, ctx.run_view, ctx.node.id))
        except Exception as exc:  # fail closed: an evaluator fault never passes
            return self.failed(f"the policy evaluation failed ({type(exc).__name__}, fail closed)")
        missing = [r for r in self.rules if r not in report.rules_evaluated]
        if missing:
            return self.failed(
                f"policy rule(s) {', '.join(missing)} were not evaluated (fail closed)",
                rules_evaluated=list(report.rules_evaluated),
            )
        relevant = {*self.rules, ENGINE_FAULT_RULE}
        mine = [f for f in report.findings if f.rule_id in relevant]
        if not mine:
            return self.passed("policy rules satisfied", rules=list(self.rules))
        if any(f.severity >= Severity.CRITICAL for f in mine):
            decision = GateDecision.SAFE_STOP
        elif any(f in report.blocking for f in mine):
            decision = _FAIL
        elif any(f in report.needs_approval for f in mine):
            decision = GateDecision.REQUIRE_APPROVAL
        else:
            decision = _PASS
        return self.result(decision, f"{len(mine)} policy finding(s)", mine, rules=list(self.rules))


class Traceability(_PolicyGate):
    """CMP-001: each active acceptance criterion has a passed test, inspection or deferral."""

    name = "traceability"
    rules = ("CMP-001",)


class ReleaseReady(_PolicyGate):
    """CMP-003 (approvals, waivers, decision records), CHG-005 (change reference, rollback plan)."""

    name = "release_ready"
    rules = ("CMP-003", "CHG-005")


# ---------------------------------------------------------------------------- 11. promote / verify


def _manifest(view: FileView) -> dict[str, str]:
    return {p: sha256_text(view.read_text(p)) for p in view.list_files("**")}


class BaselineUnchanged(_GateBase):
    """Entry gate of ``promote`` (apply mode): the target tree still equals the baseline snapshot.

    PASS when no ``apply_root`` run parameter is set (FAIL on a side-effecting node: a real apply
    needs a known target).  A drifted target yields ``AWAIT_INPUT`` with a question whose id embeds
    the manifest digests, so a later, different drift asks again.
    """

    name = "baseline_unchanged"
    category = PolicyCategory.CHANGE_CONTROL

    def check(self, ctx: GateContext) -> GateResult:
        params = ctx.run_view.params()
        root = params.get("apply_root")
        if root is None:
            if ctx.node.side_effecting:  # a real apply without a known target must not go on
                return self.failed("no apply target is configured for the side-effecting node")
            return self.passed("no apply target configured", checked=False)
        if not isinstance(root, str) or not root or "\x00" in root or not Path(root).is_dir():
            return self.failed("the apply target is not a readable directory (fail closed)")
        include = _strings(params.get("baseline_include")) or ["**"]
        exclude = _strings(params.get("baseline_exclude"))
        try:
            target = _manifest(
                DirectoryView(root, include=include, exclude=(*DEFAULT_EXCLUDES, *exclude))
            )
            baseline = _manifest(ctx.baseline)
        except (ConfigError, OrchestratorError, OSError, ValueError):
            return self.failed("the target tree cannot be fingerprinted (fail closed)")
        target_hash, baseline_hash = hash_jsonable(target), hash_jsonable(baseline)
        if target_hash == baseline_hash:
            return self.passed("target tree equals the baseline", checked=True, files=len(baseline))
        changed = sorted(
            p for p in target.keys() | baseline.keys() if target.get(p) != baseline.get(p)
        )
        question = ClarificationQuestion(
            id=question_id("baseline", f"baseline drift {baseline_hash[:12]} {target_hash[:12]}"),
            text=(
                f"The target tree changed since the baseline snapshot ({len(changed)} file(s) "
                f"differ; baseline {baseline_hash[:12]}, target {target_hash[:12]}). Restore the "
                "tree and answer 'retry', or answer 'abort' to stop the release."
            ),
            rationale="A patch reviewed against another tree could overwrite changes made since.",
            options=("retry", "abort"),
            default_assumption=None,
            blocking=True,
            category="baseline",
        )
        findings = [
            self.finding("file differs from the baseline snapshot", location=path, severity=_MEDIUM)
            for path in changed[:MAX_FINDINGS]
        ]
        return self.result(
            GateDecision.AWAIT_INPUT,
            f"target tree differs from the baseline in {len(changed)} file(s)",
            findings,
            questions=[question],
            differing=len(changed),
        )


class ReleaseVerified(_GateBase):
    """The bundle applies cleanly to a fresh copy, the suite is green, ``/healthz`` is not down."""

    name = "release_verified"

    def check(self, ctx: GateContext) -> GateResult:
        report = self.load(ctx, Kind.RELEASE_VERIFICATION)
        problems: list[str] = []
        if report.get("applied_cleanly") is not True:
            problems.append("the patch does not apply cleanly to a fresh baseline copy")
        if report.get("tests_passed") is not True:
            problems.append("the suite is not green on the released tree")
        healthz = report.get("healthz_ok")
        if healthz is not None and healthz is not True:  # None = no service to probe
            problems.append("the /healthz smoke check failed")
        files = report.get("checked_files")
        if not (isinstance(files, int) and not isinstance(files, bool) and files > 0):
            problems.append("no file was verified")
        if problems:
            return self.failed(
                f"{len(problems)} release verification problem(s)",
                [self.finding(p) for p in problems],
            )
        return self.passed("release verified", checked_files=files)


# ---------------------------------------------------------------------------- skip predicate


def greenfield_skip(view: RunView, node: NodeSpec) -> str | None:
    """Skip reason when the intake found an empty workspace; ``None`` (do not skip) when unsure."""
    try:
        art = view.latest(Kind.INTAKE_REPORT)
        payload = view.read_json(art) if art is not None else None
    except _READ_ERRORS:
        return None
    workspace = payload.get("workspace") if isinstance(payload, Mapping) else None
    if isinstance(workspace, Mapping) and workspace.get("is_empty") is True:
        return "greenfield: the workspace is empty, there is no code to analyse"
    return None


# ---------------------------------------------------------------------------- registration

GATE_CLASSES: tuple[type[_GateBase], ...] = (
    AcceptanceCriteriaTestable,
    AmbiguityDispositionsComplete,
    DesignRisk,
    DesignCoversRequirements,
    PlanCoversRequirements,
    SyntaxValid,
    ScopeWithinPlan,
    AcceptanceCriteriaCovered,
    IntegrationClean,
    SuiteGreen,
    FailToPass,
    ApiCompat,
    SecurityClean,
    Traceability,
    ReleaseReady,
    BaselineUnchanged,
    ReleaseVerified,
)
GATE_NAMES: tuple[str, ...] = tuple(cls.name for cls in GATE_CLASSES)


def register_sdlc_gates(registry: GateRegistry) -> None:
    """Register every SDLC gate type (``GateSpec.type`` == the gate's ``name``)."""
    for cls in GATE_CLASSES:
        registry.register(cls.name, cls)


def register_sdlc_skips(skips: SkipRegistry) -> None:
    """Register the SDLC skip predicates (``greenfield``)."""
    skips.register("greenfield", greenfield_skip)
