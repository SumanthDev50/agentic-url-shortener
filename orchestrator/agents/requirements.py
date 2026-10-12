"""Requirements agent: free text + clarification answers -> normalised, testable requirements.

This is a *deterministic, rule-based stand-in* for a model-backed analyst. It never guesses:
whatever the catalog of capabilities (``agents.catalog``) cannot cover is reported as
``unsupported`` and what the text leaves open becomes a clarification question or a recorded
assumption.

Heuristics (all patterns are linear-time, word-bounded, case-insensitive; the request is hostile
input and is only ever matched, never interpreted):

H1  Sentences. Markdown headings are removed (the first one becomes ``problem.goal``), the rest is
    split with ``base.split_sentences`` (sentences and bullets). Sentences that look like an
    instruction to the agent (``ignore previous instructions``, ``approve every ...``) are dropped
    from the analysis (they stay data; the engine's SEC-006 screening flags them for a human).
H2  Candidates. A sentence is a requirement candidate when it has a modal (must/shall/should/need/
    want/can/may/will/``would be nice``/...), starts with an imperative (Make/Build/Add/Allow/
    Provide/Support/Create/Export/...), is a normative statement without a modal (``Only http is
    allowed.``, ``Links keep working``), or is hedged (maybe/perhaps/``some kind of``/``?``).
    A candidate without any content word (``We need this soon.``) is not a requirement: with a
    time cue it only feeds the UNBOUNDED_TIME detector.
H3  Kind: ``constraint`` (must not / never / only / cannot ...), else ``non_functional``
    (performance, security, availability, concurrency, retry lexicon), else ``functional``.
H4  Priority: must (must/shall/need/required/imperative/normative), should (should/want/will),
    could (the rest).
H5  Capability matching. ``catalog.match_capabilities`` runs over the request plus the answers
    (request first, so an amendment overrides an older answer for the same parameter). A candidate
    is covered when an *aspect* trigger of a matched capability hits its effective text (sentence +
    the answers to its ambiguities); each hit aspect becomes an acceptance criterion
    ``{id, text, aspect}`` whose text is the aspect template formatted with the recipe-validated
    parameters (``extract_params`` + ``validate_params``; nothing else is ever interpolated).
    A candidate that only hits capability-level triggers is a *scope statement*
    (``problem.in_scope``) when another requirement of that capability is covered, else
    ``unsupported``. Intent statements (``we need``/``wants`` without a hard modal) are scope
    statements, never ``unsupported``.
H6  Ambiguity taxonomy (one ambiguity + one question per distinct topic; ids are content-derived):
    VAGUE_QUALITY   better/fast/safer/secure/scalable/robust/reliable/easy/improve/optimise/modern/
                    flexible/user-friendly/good enough, and ``report(s)``/``reporting`` without a
                    defined output (blocking)
    UNQUANTIFIED    latency/throughput/response time/performance/capacity/timeout/rate limit ...
                    without a number+unit in the sentence (blocking)
    UNCERTAIN       maybe/perhaps/possibly/``some kind of``/``nice to have``/``?`` (non-blocking)
    UNBOUNDED_TIME  soon/asap/later/next quarter/... without a date (non-blocking)
    UNDEFINED_TERM  brownfield only: a quoted term, a CamelCase name or an ``existing `x` ``
                    identifier that a literal ``toolbox.search`` cannot find (non-blocking)
    MULTI_CAPABILITY two or more capabilities matched: confirm the scope (non-blocking)
    A quantity with a unit exempts speed words; ``better``/``improve`` only count when no other
    vague topic is present in the sentence. A limit word that is one item of an enumeration in a
    classifying sentence (``classified as one of: healthy, timeout, or blocked``) is an outcome
    label, not a missing target.
H7  Answers (``clarification_answers``, cumulative) resolve ambiguities: ``answered``; the accepted
    default (or ``@default``) is ``assumed``; an answer containing out of scope / defer / not now /
    later marks the related requirements ``deferred`` with ``status_decision = 'SCOPE:<id>'`` and a
    SCOPE ``DecisionDraft`` citing the human decisions found through ``ctx.run_view``.
    Unanswered *non-blocking* questions become ``ASM-nnn`` assumptions with ASSUMPTION decisions;
    unanswered *blocking* ones are returned as ``AgentResult.clarifications`` (the provisional
    requirements artifact is still produced and lists them in ``open_questions``).

Limitations (documented, not hidden): keyword heuristics, English only; a requirement that no
capability covers is reported, not invented; answers to questions that no longer exist (the text
changed) are ignored; the MULTI_CAPABILITY answer is interpreted only through its options.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from orchestrator.agents import catalog
from orchestrator.agents.base import (
    BaseAgent,
    ac_id,
    ambiguity_id,
    asm_id,
    json_artifact,
    question_id,
    req_id,
    split_sentences,
    stable_hex,
)
from orchestrator.agents.recipes.base import (
    AspectSpec,
    Capability,
    RecipeParamError,
    validate_params,
)
from orchestrator.core.errors import AgentError
from orchestrator.core.sanitize import strip_controls, truncate
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    K_CLARIFICATION_ANSWERS,
    K_REQUEST,
    ActorType,
    AgentContext,
    AgentResult,
    AutonomyProfile,
    ClarificationQuestion,
    DecisionDraft,
    DecisionKind,
    Toolbox,
)

MAX_REQUIREMENTS = 60
MAX_ANSWER_CHARS = 2048
MAX_TERM_SEARCHES = 12
SPAN_CHARS = 160
TEXT_CHARS = 400

VAGUE_QUALITY = "vague_quality"
UNQUANTIFIED = "unquantified"
UNCERTAIN = "uncertain"
UNBOUNDED_TIME = "unbounded_time"
UNDEFINED_TERM = "undefined_term"
MULTI_CAPABILITY = "multi_capability"
BLOCKING_CATEGORIES = frozenset({VAGUE_QUALITY, UNQUANTIFIED})

_I = re.IGNORECASE
# --- H2/H4: candidates and priority
_STRONG = re.compile(
    r"\b(?:must|shall|required|requires?|need(?:s|ed)?|has to|have to|cannot)\b", _I
)
_MEDIUM = re.compile(r"\b(?:should|ought to|want(?:s|ed)?|would like|will)\b", _I)
_WEAK = re.compile(
    r"\b(?:can|could|may|might|optionally|would (?:also )?be nice|nice (?:to have|if)|maybe|"
    r"perhaps|possibly)\b",
    _I,
)
_IMPERATIVE = re.compile(
    r"^(?:please\s+)?(?:make|build|add|allow|provide|support|create|export|implement|enable|"
    r"ensure)\b",
    _I,
)
# Normative statements without a modal verb ("Only http and https destinations are allowed.").
_NORMATIVE = re.compile(
    r"\b(?:is|are) (?:not )?(?:allowed|permitted|forbidden|prohibited|disallowed|mandatory)\b"
    r"|^only\b|\b(?:keeps?|remains?|stays?) (?:working|unchanged|valid|available|compatible)\b",
    _I,
)
_INTENT = re.compile(r"\b(?:wants?|needs?|would like|asks? for)\b", _I)
_HARD = re.compile(
    r"\b(?:must|shall|should|required|requires?|has to|have to|cannot|can|may|could|might|will|"
    r"optionally)\b",
    _I,
)
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override)\b[^.]{0,40}\b(?:previous|prior|above|earlier|"
    r"all)\b[^.]{0,40}"
    r"\b(?:instructions?|rules?|prompts?|polic(?:y|ies))\b"
    r"|\bsystem prompt\b|\byou are now\b|\bapprove\b[^.]{0,30}\b(?:every|all)\b",
    _I,
)
# --- H3: kind
_CONSTRAINT = re.compile(
    r"\b(?:must not|shall not|should not|never|only|cannot|may not|not be affected)\b", _I
)
_NFR = re.compile(
    r"\b(?:latency|throughput|performance|response times?|p9\d|time ?outs?|retr(?:y|ies|ied)|"
    r"back-?off|"
    r"availab\w+|uptime|secur\w+|encrypt\w*|authenticat\w+|authoriz\w+|privacy|concurren\w+|"
    r"scalab\w+|"
    r"rate[- ]limit\w*|within\s+\d+\s*(?:ms|s|seconds?|minutes?))\b",
    _I,
)
# --- H6: ambiguity lexicons
# The number is a bounded, possessive run: ``\d[\d,.]*`` backtracked quadratically on ``1,1,1,...``.
_NUM_UNIT = re.compile(
    r"\b\d[\d,.]{0,15}+\s?(?:(?:ms|milliseconds?|s|secs?|seconds?|mins?|minutes?|hours?|hrs?|"
    r"days?|weeks?|"
    r"months?|years?|kb|mb|gb|tb|requests?|reqs?|rps|qps|times|attempts|retries|clicks|links|"
    r"per)\b|%)",
    _I,
)
_VAGUE_TOPICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("output", re.compile(r"\breport(?:s|ing)?\b", _I)),
    ("speed", re.compile(r"\b(?:fast(?:er|est)?|quick(?:er|ly)?|speedy|snappy|responsive)\b", _I)),
    ("safety", re.compile(r"\b(?:safe(?:r|ly)?|secure(?:ly)?|safety)\b", _I)),
    (
        "quality",
        re.compile(
            r"\b(?:scalab\w+|robust\w*|reliab\w+|flexib\w+|modern\w*|user[- ]friendl\w+|easy|"
            r"easier|"
            r"intuitive|good enough)\b",
            _I,
        ),
    ),
)
_IMPROVEMENT = re.compile(r"\b(?:better|improv\w+|optimi[sz]\w+|enhanc\w+)\b", _I)
_OUTPUT_DEFINED = re.compile(
    r"\b(?:with|containing|contains|including|includes|listing|showing|per|csv|json|pdf|xlsx|table|"
    r"columns?|fields?|counts?|number of|list of)\b",
    _I,
)
_UNQUANTIFIED = re.compile(
    r"\b(?:latency|throughput|response times?|performance|capacity|bandwidth|rate[- ]limit\w*|"
    r"time ?outs?|retention|p9\d(?:\.\d+)?)\b",
    _I,
)
# A limit word that is one item of an enumeration inside a classifying sentence ("... one of:
# healthy, timeout, or blocked") names an outcome label, not a quality that needs a target.
_LABEL_CUE = re.compile(
    r"\b(?:classif\w*|categor\w*|status(?:es)?|outcomes?|results?|labels?|one of)\b", _I
)
_LIST_BEFORE = re.compile(r"(?:[,:(/]|\b(?:or|and))\s*$", _I)
_LIST_AFTER = re.compile(r"\s*(?:[,;.)/]|\b(?:or|and)\b|$)", _I)
_LOOK = 12  # characters of context inspected on each side of a candidate label
_HEDGE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|might)\b|some (?:kind|sort) of|nice[- ]to[- ]have|not "
    r"sure",
    _I,
)
_UNBOUNDED = re.compile(
    r"\b(?:soon|asap|as soon as possible|later(?! than)|eventually|at some point|sometime|shortly|"
    r"in the future|next (?:quarter|month|year|sprint|release))\b",
    _I,
)
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\bq[1-4]\s?(?:fy)?\s?\d{2,4}\b|\b(?:within|in)\s+\d+\s+(?:days?|"
    r"weeks?|months?)\b|"
    r"\bby\s+(?:the\s+end\s+of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b",
    _I,
)
_QUOTED = re.compile(r"[\"“]([^\"”\n]{2,40})[\"”]")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_EXISTING = re.compile(r"\b(?:existing|current)\s+`([A-Za-z_][\w.]{1,40})`", _I)
_DEFER = re.compile(
    r"\b(?:out of scope|not in scope|descop\w+|defer\w*|not now|postpone\w*|later(?! than)|won'?t "
    r"do)\b",
    _I,
)
_WORD = re.compile(r"[a-z][a-z0-9_-]{3,}")
_STOP = frozenset(
    [
        "this",
        "that",
        "these",
        "those",
        "with",
        "without",
        "from",
        "have",
        "has",
        "had",
        "does",
        "did",
        "need",
        "needs",
        "needed",
        "want",
        "wants",
        "wanted",
        "would",
        "should",
        "could",
        "must",
        "shall",
        "will",
        "also",
        "just",
        "really",
        "very",
        "soon",
        "asap",
        "later",
        "eventually",
        "sometime",
        "shortly",
        "there",
        "here",
        "then",
        "they",
        "them",
        "their",
        "your",
        "yours",
        "ours",
        "please",
        "maybe",
        "perhaps",
        "possibly",
        "when",
        "what",
        "which",
        "while",
        "into",
        "onto",
        "over",
        "under",
        "than",
        "been",
        "being",
        "were",
    ]
)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _hit(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(p, text, _I) for p in patterns)


# ----------------------------------------------------------------------------- data model


@dataclass
class _Sent:
    text: str
    kind: str  # "requirement" | "scheduling" | "context"
    priority: str = "could"
    intent_only: bool = False


@dataclass
class _Amb:
    category: str
    topic: str
    term: str
    text: str  # the question text (content-derived id)
    span: str
    sentences: list[int] = field(default_factory=list)
    caps: tuple[str, ...] = ()  # MULTI_CAPABILITY: the matched capability ids
    answer: str | None = None
    question: ClarificationQuestion | None = None
    disposition: str = "open"
    deferred: bool = False
    out_of_scope_span: bool = False
    chosen_cap: str | None = None  # MULTI_CAPABILITY: ``Deliver only <title>``

    @property
    def qid(self) -> str:
        return question_id(self.category, self.text)


@dataclass
class _Req:
    sent: int
    text: str
    kind: str
    priority: str
    covered: dict[str, list[AspectSpec]]
    unsupported: str | None = None
    status: str = "active"
    rid: str = ""


# ----------------------------------------------------------------------------- sentence analysis


def _prepare(request: str) -> tuple[list[str], list[str], int]:
    """Headings, analysable sentences and the number of dropped instruction-like sentences (H1)."""
    headings: list[str] = []
    body: list[str] = []
    for line in request.split("\n"):
        if line.strip().startswith("#"):
            headings.append(line.strip().lstrip("#").strip())
        else:
            body.append(line)
    sentences: list[str] = []
    dropped = 0
    for sentence in split_sentences("\n".join(body)):
        if _INJECTION.search(sentence):
            dropped += 1
        else:
            sentences.append(sentence)
    return headings, sentences, dropped


def _classify(text: str) -> _Sent:
    hedged = bool(_HEDGE.search(text)) or "?" in text
    normative = _NORMATIVE.search(text) is not None
    candidate = bool(
        _STRONG.search(text)
        or _MEDIUM.search(text)
        or _WEAK.search(text)
        or _IMPERATIVE.search(text)
        or normative
        or hedged
    )
    if not candidate:
        return _Sent(text, "context")
    if not any(w not in _STOP for w in _WORD.findall(text.lower())):
        return _Sent(text, "scheduling" if _UNBOUNDED.search(text) else "context")
    if _STRONG.search(text) or _IMPERATIVE.search(text) or normative:
        priority = "must"
    elif _MEDIUM.search(text):
        priority = "should"
    else:
        priority = "could"
    intent_only = bool(_INTENT.search(text)) and not (
        _HARD.search(text) or _IMPERATIVE.search(text) or normative or hedged
    )
    return _Sent(text, "requirement", priority, intent_only)


def _requirement_kind(text: str) -> str:
    if _CONSTRAINT.search(text):
        return "constraint"
    if _NFR.search(text):
        return "non_functional"
    return "functional"


# ----------------------------------------------------------------------------- ambiguity detection

_VAGUE_QUESTIONS = {
    "output": (
        'What exactly should the "{term}" contain, and in which format '
        "(for example a CSV export of daily click counts per link)?"
    ),
    "speed": (
        'What does "{term}" mean here? Which operation should be quicker, and what latency target '
        "(for example 200 ms at p95) is acceptable?"
    ),
    "safety": (
        'What does "{term}" mean here? Which concrete risk should be reduced '
        "(for example abuse of the API or exposure of data), and how will we verify it?"
    ),
    "quality": 'What does "{term}" mean here? Which observable behaviour or measurable target '
    "shows it is achieved?",
    "improvement": 'What exactly should be "{term}", and what measurable target shows it?',
}
_RATIONALE = {
    VAGUE_QUALITY: "Vague quality words cannot be tested; a concrete, measurable meaning is "
    "needed.",
    UNQUANTIFIED: "A performance or limit statement without a number and unit cannot be verified.",
    UNCERTAIN: "A hedged statement is not a requirement until someone confirms it.",
    UNBOUNDED_TIME: "A deadline without a date cannot be planned.",
    UNDEFINED_TERM: "The term is unknown to the existing code; the intended meaning must be "
    "confirmed.",
    MULTI_CAPABILITY: "Several capabilities matched; the delivery scope must be confirmed.",
}
_IN_SCOPE = "Include in scope"
_OUT_OF_SCOPE = "Out of scope for this iteration"
_ALL_CAPS = "Deliver all matched capabilities"


def _is_enumerated(sentence: str, match: re.Match[str]) -> bool:
    """True when ``match`` sits between list separators (``,`` ``or`` ``and``) on both sides."""
    before = sentence[max(0, match.start() - _LOOK) : match.start()]
    return (
        _LIST_BEFORE.search(before) is not None
        and _LIST_AFTER.match(sentence[match.end() : match.end() + _LOOK]) is not None
    )


def _detect(sentence: str) -> list[tuple[str, str, str]]:
    """``(category, topic, term)`` findings of one sentence, in a fixed order (H6)."""
    found: list[tuple[str, str, str]] = []
    quantified = _NUM_UNIT.search(sentence) is not None
    vague_topics = 0
    for topic, pattern in _VAGUE_TOPICS:
        m = pattern.search(sentence)
        if m is None:
            continue
        if topic == "output" and _OUTPUT_DEFINED.search(sentence):
            continue
        if topic == "speed" and quantified:
            continue
        found.append((VAGUE_QUALITY, topic, m.group(0).lower()))
        vague_topics += 1
    if not vague_topics:
        m = _IMPROVEMENT.search(sentence)
        if m is not None:
            found.append((VAGUE_QUALITY, "improvement", m.group(0).lower()))
    if not quantified:
        classifying = _LABEL_CUE.search(sentence) is not None
        for m in _UNQUANTIFIED.finditer(sentence):
            if classifying and _is_enumerated(sentence, m):
                continue
            term = m.group(0).lower()
            found.append((UNQUANTIFIED, term.replace(" ", "_"), term))
            break
    m = _HEDGE.search(sentence)
    if m is not None or "?" in sentence:
        found.append((UNCERTAIN, f"hedge-{stable_hex(sentence)}", m.group(0).lower() if m else "?"))
    m = _UNBOUNDED.search(sentence)
    if m is not None and _DATE.search(sentence) is None:
        found.append((UNBOUNDED_TIME, m.group(0).lower().replace(" ", "_"), m.group(0).lower()))
    return found


def _question_text(category: str, topic: str, term: str, sentence: str) -> str:
    if category == VAGUE_QUALITY:
        return _VAGUE_QUESTIONS[topic].format(term=term)
    if category == UNQUANTIFIED:
        return (
            f'What is the measurable target for "{term}" '
            "(a number with a unit, for example 200 ms at p95)?"
        )
    if category == UNCERTAIN:
        return f'Is this in scope for this change: "{_clip(sentence, 100)}"?'
    if category == UNBOUNDED_TIME:
        return f'By when is this needed ("{term}")? Please give a date or a milestone.'
    return f'"{term}" does not appear in the existing code. What does it refer to?'


def _undefined_terms(sentence: str) -> list[str]:
    seen: dict[str, None] = {}
    for pattern in (_QUOTED, _CAMEL, _EXISTING):
        for m in pattern.finditer(sentence):
            seen.setdefault(m.group(1) if pattern is not _CAMEL else m.group(0), None)
    return list(seen)


def _term_known(toolbox: Toolbox, term: str) -> bool:
    variants = dict.fromkeys((term, term.lower(), term.lower().replace(" ", "_")))
    return any(toolbox.search(v, "**", 1) for v in variants)


def _detect_all(
    sents: Sequence[_Sent], brownfield: bool, toolbox: Toolbox
) -> dict[tuple[str, str], _Amb]:
    ambs: dict[tuple[str, str], _Amb] = {}

    def register(idx: int, category: str, topic: str, term: str) -> None:
        key = (category, topic)
        amb = ambs.get(key)
        if amb is None:
            text = _question_text(category, topic, term, sents[idx].text)
            amb = ambs[key] = _Amb(category, topic, term, text, _clip(sents[idx].text, SPAN_CHARS))
        if idx not in amb.sentences:
            amb.sentences.append(idx)

    searches = 0
    for idx, sent in enumerate(sents):
        if sent.kind == "context":
            continue
        for category, topic, term in _detect(sent.text):
            register(idx, category, topic, term)
        if brownfield and sent.kind == "requirement":
            for term in _undefined_terms(sent.text):
                if searches >= MAX_TERM_SEARCHES:
                    break
                searches += 1
                if not _term_known(toolbox, term):
                    register(idx, UNDEFINED_TERM, term.lower(), term)
    return ambs


# ----------------------------------------------------------------------------- questions


def _default_for(amb: _Amb, matched: Sequence[Capability]) -> str:
    if amb.category == UNCERTAIN:
        return _IN_SCOPE
    if amb.category == UNBOUNDED_TIME:
        return "No fixed deadline: the change is scheduled at normal priority"
    if amb.category == UNDEFINED_TERM:
        return f'Treat "{amb.term}" as a new concept introduced by this change'
    if amb.category == MULTI_CAPABILITY:
        return _ALL_CAPS
    if amb.topic == "output" and len(matched) == 1:
        return f'Deliver "{matched[0].title}" with its default parameters'
    return f'Treat "{amb.term}" as out of scope until a measurable target is given'


def _options_for(amb: _Amb, titles: Mapping[str, str]) -> tuple[str, ...]:
    if amb.category == UNCERTAIN:
        return (_IN_SCOPE, _OUT_OF_SCOPE)
    if amb.category == MULTI_CAPABILITY:
        return (_ALL_CAPS, *(f"Deliver only {titles[c]}" for c in amb.caps))
    return ()


def _build_question(
    amb: _Amb, related: tuple[str, ...], matched: Sequence[Capability], titles: Mapping[str, str]
) -> ClarificationQuestion:
    return ClarificationQuestion(
        id=amb.qid,
        text=amb.text,
        rationale=_RATIONALE[amb.category],
        options=_options_for(amb, titles),
        default_assumption=_default_for(amb, matched),
        blocking=amb.category in BLOCKING_CATEGORIES,
        related_requirements=related,
        category=amb.category,
    )


# ----------------------------------------------------------------------------- answers


def _read_answers(ctx: AgentContext) -> dict[str, str]:
    """Cleaned ``{question_id: answer}`` of the cumulative answers (hostile shapes tolerated)."""
    if not ctx.has(K_CLARIFICATION_ANSWERS):
        return {}
    payload = ctx.json(K_CLARIFICATION_ANSWERS)
    raw = payload.get("answers") if isinstance(payload, Mapping) else None
    out: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for qid, entry in raw.items():
            answer = entry.get("answer") if isinstance(entry, Mapping) else None
            if isinstance(answer, str):
                cleaned = " ".join(strip_controls(answer).split())[:MAX_ANSWER_CHARS]
                if cleaned:
                    out[str(qid)] = cleaned
    return out


def _human_decisions(ctx: AgentContext) -> dict[str, tuple[str, ...]]:
    found: dict[str, list[str]] = {}
    for d in ctx.run_view.decisions():
        qid = d.data.get("question_id") if isinstance(d.data, Mapping) else None
        if d.actor.type == ActorType.HUMAN and isinstance(qid, str):
            found.setdefault(qid, []).append(d.id)
    return {k: tuple(v) for k, v in found.items()}


# ----------------------------------------------------------------------------- capabilities


def _coverage(
    text: str, caps: Sequence[Capability]
) -> tuple[dict[str, list[AspectSpec]], set[str]]:
    covered: dict[str, list[AspectSpec]] = {}
    cap_only: set[str] = set()
    for cap in caps:
        aspects = [a for a in cap.aspects if _hit(a.triggers, text)]
        if aspects:
            covered[cap.id] = aspects
        elif _hit(cap.triggers, text):
            cap_only.add(cap.id)
    return covered, cap_only


def _params(cap: Capability, text: str) -> dict[str, Any]:
    try:
        return validate_params(cap.recipe.params, catalog.extract_params(cap, text))
    except ValueError as exc:
        raise RecipeParamError(
            f"{cap.id}: a parameter value in the request or answers is not a valid number"
        ) from exc


def _criterion(cap: Capability, aspect: AspectSpec, params: Mapping[str, Any]) -> str:
    try:
        return aspect.criterion.format_map(dict(params))
    except (KeyError, IndexError, ValueError, AttributeError) as exc:
        raise AgentError(
            f"capability {cap.id}: criterion of aspect {aspect.id} does not fit its parameters",
            kind="fatal",
            retryable=False,
        ) from exc


# ----------------------------------------------------------------------------- the agent


class RequirementsAgent(BaseAgent):
    agent_id = "requirements"
    role = "requirements"
    version = "2"  # 2: linear-time number pattern, label enumerations, modal-less rules
    autonomy = AutonomyProfile(
        read_globs=("**",), write_globs=(), can_execute=False, max_tool_calls=100
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        request = strip_controls(ctx.text(K_REQUEST), keep_newlines=True)
        intake = ctx.json(Kind.INTAKE_REPORT)
        classification = {
            "workspace_kind": intake["classification"]["workspace_kind"],
            "change_type": intake["classification"]["change_type"],
        }
        brownfield = classification["workspace_kind"] == "brownfield"

        headings, sentences, dropped = _prepare(request)
        sents = [_classify(s) for s in sentences]
        ambs = _detect_all(sents, brownfield, ctx.toolbox)
        answers = _read_answers(ctx)
        for amb in ambs.values():
            amb.answer = answers.get(amb.qid)

        clean_request = "\n".join(sentences)
        base_text = "\n".join([clean_request, *(a.answer for a in ambs.values() if a.answer)])
        matches = catalog.match_capabilities(base_text)
        usable = [m.capability for m in matches if brownfield or not m.capability.requires_baseline]
        blocked = [
            m.capability for m in matches if not brownfield and m.capability.requires_baseline
        ]
        titles = {c.id: c.title for c in (*usable, *blocked)}
        if len(usable) >= 2:
            multi = _Amb(
                MULTI_CAPABILITY,
                "scope",
                "+".join(c.id for c in usable),
                "This request matches several capabilities ({}). Should all of them be delivered "
                "in this change?".format("; ".join(c.title for c in usable)),
                _clip(", ".join(c.id for c in usable), SPAN_CHARS),
                caps=tuple(c.id for c in usable),
            )
            multi.answer = answers.get(multi.qid)
            ambs[(MULTI_CAPABILITY, "scope")] = multi

        reqs, scope_statements = self._classify_requirements(sents, ambs, usable, blocked, titles)
        ids_of = {
            r.sent: r.rid for r in reqs if r.rid
        }  # requirements beyond MAX_REQUIREMENTS have no id
        for amb in ambs.values():
            related = tuple(ids_of[i] for i in amb.sentences if i in ids_of)
            if amb.category == MULTI_CAPABILITY:
                related = tuple(r.rid for r in reqs if r.covered)
            amb.question = _build_question(amb, related, usable, titles)
            self._resolve(amb, related)

        humans = _human_decisions(ctx)
        self._apply_scope(reqs, ambs)
        used = self._used_capabilities(reqs, usable)
        params_by_cap = {c.id: _params(c, base_text) for c in used}
        for r in reqs:
            if r.status != "active" and r.unsupported is None:
                r.covered = {}
            for cid in list(r.covered):
                if cid not in params_by_cap:
                    del r.covered[cid]
            if r.status == "active" and not r.covered and r.unsupported is None:
                r.unsupported = "no capability remains to deliver this requirement"

        payload = self._payload(
            headings, sents, scope_statements, reqs, ambs, used, params_by_cap, classification
        )
        decisions = self._decisions(reqs, ambs, payload["assumptions"], humans)
        open_questions = [
            a.question for a in ambs.values() if a.disposition == "open" and a.question is not None
        ]
        notes = f"{len(reqs)} requirement(s), {len(open_questions)} open blocking question(s)."
        if dropped:
            notes += f" Ignored {dropped} sentence(s) that look like instructions to the agent."
        return AgentResult(
            artifacts=[json_artifact(Kind.REQUIREMENTS, payload)],
            decisions=decisions,
            clarifications=open_questions,
            rationale=notes,
            usage=dict(ctx.toolbox.usage()),
        )

    # -- stages ---------------------------------------------------------------------------------
    @staticmethod
    def _classify_requirements(
        sents: Sequence[_Sent],
        ambs: Mapping[tuple[str, str], _Amb],
        usable: Sequence[Capability],
        blocked: Sequence[Capability],
        titles: Mapping[str, str],
    ) -> tuple[list[_Req], list[str]]:
        """H5: coverage, scope statements, unsupported entries and stable ``R-nnn`` ids."""
        answers_of: dict[int, list[str]] = {}
        for amb in ambs.values():
            if amb.answer:
                for i in amb.sentences:
                    answers_of.setdefault(i, []).append(amb.answer)
        info: dict[int, tuple[dict[str, list[AspectSpec]], set[str], set[str]]] = {}
        for idx, sent in enumerate(sents):
            if sent.kind == "requirement":
                effective = " ".join([sent.text, *answers_of.get(idx, [])])
                covered, cap_only = _coverage(effective, usable)
                blocked_cov, blocked_only = _coverage(effective, blocked)
                info[idx] = (covered, cap_only, set(blocked_cov) | blocked_only)
        supported = {cid for covered, _, _ in info.values() for cid in covered}
        reqs: list[_Req] = []
        scope: list[str] = []
        for idx, (covered, cap_only, blocked_ids) in info.items():
            sent = sents[idx]
            req = _Req(
                idx,
                truncate(sent.text, TEXT_CHARS),
                _requirement_kind(sent.text),
                sent.priority,
                covered,
            )
            if covered:
                pass
            elif sent.intent_only or cap_only & supported:
                scope.append(truncate(sent.text, TEXT_CHARS))
                continue
            elif blocked_ids:
                req.unsupported = (
                    "needs an existing codebase (baseline), but the workspace is empty: "
                    + ", ".join(sorted(titles[c] for c in blocked_ids))
                )
            elif cap_only:
                req.unsupported = (
                    "a capability matches ("
                    + ", ".join(sorted(titles[c] for c in cap_only))
                    + ") but no acceptance aspect applies; state the expected behaviour in "
                    "testable terms"
                )
            else:
                req.unsupported = "no capability in the catalog covers this requirement"
            if any(r.text == req.text for r in reqs):
                continue
            reqs.append(req)
        for n, req in enumerate(reqs[:MAX_REQUIREMENTS], 1):
            req.rid = req_id(n)
        for req in reqs[MAX_REQUIREMENTS:]:
            req.rid = ""
        return reqs, scope

    @staticmethod
    def _resolve(amb: _Amb, related: tuple[str, ...]) -> None:
        question = amb.question
        assert question is not None
        answer = amb.answer
        if answer is None:
            amb.disposition = "open" if question.blocking else "assumed"
        elif amb.category == MULTI_CAPABILITY:
            chosen = next(
                (c for c in amb.caps if answer == f"Deliver only {_title_of(question, c)}"), None
            )
            amb.chosen_cap = chosen
            amb.disposition = "assumed" if answer == question.default_assumption else "answered"
        elif _DEFER.search(answer) and related:
            amb.disposition, amb.deferred = "deferred", True
        elif answer == question.default_assumption or answer.lower() == "@default":
            amb.disposition = "assumed"
        else:
            amb.disposition = "answered"
            amb.out_of_scope_span = bool(_DEFER.search(answer))

    @staticmethod
    def _apply_scope(reqs: Sequence[_Req], ambs: Mapping[tuple[str, str], _Amb]) -> None:
        """Deferral (H7) and MULTI_CAPABILITY descoping: statuses and ``SCOPE:<id>`` markers."""
        ids_deferred = {i for a in ambs.values() if a.deferred for i in a.sentences}
        chosen = {a.chosen_cap for a in ambs.values() if a.chosen_cap}
        for r in reqs:
            if r.sent in ids_deferred:
                r.status, r.covered, r.unsupported = "deferred", {}, None
            elif chosen and r.covered:
                r.covered = {c: v for c, v in r.covered.items() if c in chosen}
                if not r.covered:
                    r.status, r.unsupported = "descoped", None

    @staticmethod
    def _used_capabilities(reqs: Sequence[_Req], usable: Sequence[Capability]) -> list[Capability]:
        wanted = {cid for r in reqs if r.status == "active" for cid in r.covered}
        return sorted((c for c in usable if c.id in wanted), key=lambda c: c.id)

    @staticmethod
    def _payload(
        headings: Sequence[str],
        sents: Sequence[_Sent],
        scope_statements: Sequence[str],
        reqs: Sequence[_Req],
        ambs: Mapping[tuple[str, str], _Amb],
        used: Sequence[Capability],
        params_by_cap: Mapping[str, dict[str, Any]],
        classification: Mapping[str, str],
    ) -> dict[str, Any]:
        kept = [r for r in reqs if r.rid]
        entries: list[dict[str, Any]] = []
        for r in kept:
            criteria: list[dict[str, str]] = []
            aspects: list[AspectSpec] = []
            caps = sorted(r.covered) if r.status == "active" else []
            for cid in caps:
                cap = next(c for c in used if c.id == cid)
                for aspect in r.covered[cid]:
                    aspects.append(aspect)
                    criteria.append(
                        {
                            "id": ac_id(int(r.rid[2:]), len(criteria) + 1),
                            "text": _criterion(cap, aspect, params_by_cap[cid]),
                            "aspect": aspect.id,
                        }
                    )
            merged: dict[str, Any] = {}
            for cid in caps:
                merged.update(params_by_cap[cid])
            entries.append(
                {
                    "id": r.rid,
                    "text": r.text,
                    "kind": r.kind,
                    "priority": r.priority,
                    "status": r.status,
                    "status_decision": f"SCOPE:{r.rid}"
                    if r.status in ("deferred", "descoped")
                    else None,
                    "capabilities": caps,
                    "parameters": merged,
                    "verification": "inspection"
                    if aspects and all(a.verification == "inspection" for a in aspects)
                    else "test",
                    "acceptance_criteria": criteria,
                }
            )
        unsupported = [
            {"requirement_id": r.rid, "reason": r.unsupported}
            for r in kept
            if r.status == "active" and r.unsupported
        ]
        if len(reqs) > MAX_REQUIREMENTS:
            unsupported.append(
                {
                    "requirement_id": "request",
                    "reason": (
                        f"the request has more than {MAX_REQUIREMENTS} requirements; "
                        "the rest was not analysed"
                    ),
                }
            )
        if not reqs:
            unsupported.append(
                {
                    "requirement_id": "request",
                    "reason": "no testable requirement could be derived from the request",
                }
            )
        assumptions: list[dict[str, Any]] = []
        for amb in ambs.values():
            if amb.disposition == "assumed" and amb.question is not None:
                assumptions.append(
                    {
                        "id": asm_id(len(assumptions) + 1),
                        "text": f"{amb.question.default_assumption} (re: {_clip(amb.span, 80)})",
                        "question_id": amb.qid,
                    }
                )
        goal = headings[0] if headings else next((s.text for s in sents), "")
        in_scope = [*scope_statements, *(c.title for c in used)]
        out_of_scope = [r.text for r in kept if r.status in ("deferred", "descoped")]
        out_of_scope += [a.span for a in ambs.values() if a.out_of_scope_span]
        return {
            "problem": {
                "goal": _clip(goal, 200),
                "in_scope": list(dict.fromkeys(in_scope)),
                "out_of_scope": list(dict.fromkeys(out_of_scope)),
                "constraints": [
                    r.text for r in kept if r.kind == "constraint" and r.status == "active"
                ],
            },
            "classification": dict(classification),
            "requirements": entries,
            "assumptions": assumptions,
            "ambiguities": [
                {
                    "id": ambiguity_id(a.category, f"{a.topic}:{a.term}"),
                    "category": a.category,
                    "span": a.span,
                    "question_id": a.qid,
                    "disposition": a.disposition,
                    "answer": a.answer,
                }
                for a in ambs.values()
            ],
            "open_questions": [
                a.question.to_dict()
                for a in ambs.values()
                if a.disposition == "open" and a.question is not None
            ],
            "unsupported": unsupported,
            "capabilities": [
                {
                    "id": c.id,
                    "title": c.title,
                    "requirement_ids": [
                        r.rid for r in kept if r.status == "active" and c.id in r.covered
                    ],
                    "params": dict(params_by_cap[c.id]),
                }
                for c in used
            ],
        }

    @staticmethod
    def _decisions(
        reqs: Sequence[_Req],
        ambs: Mapping[tuple[str, str], _Amb],
        assumptions: Sequence[Mapping[str, Any]],
        humans: Mapping[str, tuple[str, ...]],
    ) -> list[DecisionDraft]:
        """SCOPE / ASSUMPTION drafts; human decisions of the run view become parents (H7)."""
        asm_by_question = {a["question_id"]: a["id"] for a in assumptions}
        by_sentence = {r.sent: r for r in reqs if r.rid}
        out: list[DecisionDraft] = []
        for amb in ambs.values():
            qid = amb.qid
            parents = humans.get(qid, ())
            data = {"question_id": qid, "category": amb.category}
            related = [by_sentence[i] for i in amb.sentences if i in by_sentence]
            if amb.deferred:
                for r in related:
                    out.append(
                        DecisionDraft(
                            DecisionKind.SCOPE,
                            f"Deferred {r.rid}: {_clip(r.text, 100)}",
                            rationale="A human answer put this requirement out of scope for this "
                            "iteration.",
                            outcome="deferred",
                            parents=parents,
                            refs=(r.rid,),
                            data={**data, "requirement_id": r.rid},
                        )
                    )
            elif amb.chosen_cap:
                for r in reqs:
                    if r.status == "descoped" and r.rid:
                        out.append(
                            DecisionDraft(
                                DecisionKind.SCOPE,
                                f"Descoped {r.rid}: {_clip(r.text, 100)}",
                                rationale="A human answer chose to deliver only one of the "
                                "matched capabilities.",
                                outcome="descoped",
                                parents=parents,
                                refs=(r.rid,),
                                data={**data, "requirement_id": r.rid},
                            )
                        )
            elif amb.disposition == "answered":
                out.append(
                    DecisionDraft(
                        DecisionKind.SCOPE,
                        f"Scope clarified ({amb.category}): {_clip(amb.span, 100)}",
                        rationale=f"Human answer: {_clip(amb.answer or '', 200)}",
                        outcome="answered",
                        parents=parents,
                        refs=tuple(r.rid for r in related),
                        data=data,
                    )
                )
            if qid in asm_by_question:
                default_text = (amb.question.default_assumption or "") if amb.question else ""
                out.append(
                    DecisionDraft(
                        DecisionKind.ASSUMPTION,
                        f"{asm_by_question[qid]}: " + _clip(default_text, 100),
                        rationale=_RATIONALE[amb.category],
                        outcome="assumed",
                        parents=parents,
                        refs=(asm_by_question[qid], *(r.rid for r in related)),
                        data=data,
                    )
                )
        return out


def _title_of(question: ClarificationQuestion, cap_id: str) -> str:
    """Title of ``cap_id`` as spelled in the question's ``Deliver only <title>`` options."""
    for option in question.options[1:]:
        title = option.removeprefix("Deliver only ")
        if (
            catalog.all_capabilities().get(cap_id) is not None
            and catalog.get_capability(cap_id).title == title
        ):
            return title
    return ""
