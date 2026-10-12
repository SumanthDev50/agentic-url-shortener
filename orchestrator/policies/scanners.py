"""Public scanner API of the policy layer (rule-neutral detectors returning :class:`Hit`).

``scan_secrets`` / ``scan_trojan_source`` / ``scan_prompt_injection`` live here (text detectors);
``scan_python`` fronts the AST detectors of :mod:`orchestrator.policies.pyscan`.  Every function

* refuses oversize input (``hits.MAX_TEXT_CHARS`` / ``MAX_LINE_CHARS``) with an ``unscannable`` hit
  instead of scanning it - the calling rule turns that into a HIGH finding, never a silent pass;
* uses only **linear-time regular expressions**: no nested quantifiers, bounded repeats, and every
  token pattern is anchored by a look-behind on its own character class so a match attempt can only
  start at the beginning of a run (a ``eyJeyJeyJ...`` run is therefore scanned once, not once per
  position).  ``tests/orchestrator/policies/test_scanners.py`` fuzzes this under a time bound;
* never returns a secret: evidence is masked (``AKIA****``) and capped.
"""

from __future__ import annotations

import bisect
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from itertools import islice

from orchestrator.core.types import Severity
from orchestrator.policies import pyscan
from orchestrator.policies.hits import (
    UNSCANNABLE,
    Hit,
    cap_evidence,
    format_location,
    limit_hit,
    mask_secret,
)

__all__ = [
    "MAX_HITS",
    "Hit",
    "redact_text",
    "scan_prompt_injection",
    "scan_python",
    "scan_secrets",
    "scan_trojan_source",
]

MAX_HITS = 200  # a text with this many hits is rejected anyway; bounds memory on hostile input


def _line_numbers(text: str) -> Callable[[int], int]:
    """Return ``offset -> 1-based line`` using one pass over the newlines (bisect, not repeated
    counts).
    """
    newlines = [m.start() for m in re.finditer("\n", text)]
    return lambda offset: bisect.bisect_left(newlines, offset) + 1


# -------------------------------------------------------------------------------- SEC-001 secrets

_RUN_START = r"(?<![A-Za-z0-9_-])"  # a token may only start at the beginning of its character run
# Each pattern carries lowercase literal ``needles``: a text containing none of them is skipped with
# a
# C-speed substring test, so the regexes only run on texts that could possibly match.
_TOKEN_PATTERNS: tuple[tuple[str, tuple[str, ...], re.Pattern[str]], ...] = (
    (
        "aws_access_key_id",
        ("akia", "asia", "agpa", "aida", "aroa", "aipa", "anpa", "anva"),
        re.compile(
            r"(?<![A-Za-z0-9])(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[A-Z0-9]{16}(?![A-Za-z0-9])"
        ),
    ),
    (
        "github_token",
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"),
        re.compile(
            r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})(?![A-Za-z0-9_])"
        ),
    ),
    ("gitlab_token", ("glpat-",), re.compile(_RUN_START + r"glpat-[A-Za-z0-9_-]{20,100}")),
    ("slack_token", ("xox",), re.compile(_RUN_START + r"xox[abposr]-[A-Za-z0-9-]{10,200}")),
    (
        "slack_webhook",
        ("hooks.slack.com",),
        re.compile(
            r"https://hooks\.slack\.com/services/T[A-Za-z0-9]{6,20}/B[A-Za-z0-9]{6,20}/[A-Za-z0-9]{16,40}"
        ),
    ),
    (
        "stripe_key",
        ("sk_", "rk_"),
        re.compile(r"(?<![A-Za-z0-9_])(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,99}(?![A-Za-z0-9])"),
    ),
    (
        "google_api_key",
        ("aiza",),
        re.compile(_RUN_START + r"AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"),
    ),
    (
        "google_oauth_token",
        ("ya29.",),
        re.compile(r"(?<![A-Za-z0-9._-])ya29\.[0-9A-Za-z_-]{20,300}"),
    ),
    (
        "sk_token",
        ("sk-",),
        re.compile(
            _RUN_START + r"sk-(?:ant-|proj-)?(?=[A-Za-z0-9_-]{0,200}[0-9])[A-Za-z0-9_-]{20,200}"
        ),
    ),
    (
        "jwt",
        ("eyj",),
        re.compile(
            _RUN_START
            + r"eyJ[A-Za-z0-9_-]{5,2000}\.eyJ[A-Za-z0-9_-]{5,2000}\.[A-Za-z0-9_-]{0,2000}"
        ),
    ),
    ("npm_token", ("npm_",), re.compile(r"(?<![A-Za-z0-9_])npm_[A-Za-z0-9]{36}(?![A-Za-z0-9])")),
    (
        "pypi_token",
        ("pypi-",),
        re.compile(_RUN_START + r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{50,300}"),
    ),
    (
        "sendgrid_key",
        ("sg.",),
        re.compile(r"(?<![A-Za-z0-9._-])SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}"),
    ),
    ("twilio_key", ("sk",), re.compile(r"(?<![A-Za-z0-9])SK[0-9a-f]{32}(?![A-Za-z0-9])")),
    (
        "azure_account_key",
        ("accountkey", "sharedaccesskey"),
        re.compile(r"(?i:accountkey|sharedaccesskey)=[A-Za-z0-9+/]{30,120}={0,2}"),
    ),
    ("azure_sas_signature", ("sig=",), re.compile(r"(?i:[?&;]sig)=[A-Za-z0-9%+/=]{30,300}")),
    (
        "bearer_token",
        ("bearer",),
        re.compile(
            r"(?i:\bbearer)[ \t]+(?=[A-Za-z0-9._~+/=-]{0,200}[0-9])[A-Za-z0-9._~+/=-]{24,200}"
        ),
    ),
)
# Truncated blocks ("-----BEGIN RSA PRIVATE" with no END, or cut mid-header) are still a leak.
_PRIVATE_KEY = re.compile(r"-----BEGIN[ A-Z0-9]{0,40}PRIVATE")
_USERINFO = re.compile(
    r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]{1,15}://([^\s/:@'\"<>]{0,128}):([^\s/@'\"<>]{3,256})@"
)
_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_.-])([A-Za-z_][A-Za-z0-9_.-]{0,60})[\"']?"
    r"(?:[ \t]*:[ \t]*[A-Za-z_][\w.\[\]|]{0,40})?"  # optional annotation
    r"[ \t]*[:=][ \t]*([\"']?)([^\s\"'<>,;()\[\]{}`\\]{8,200})"
)
_SECRET_NAME = re.compile(
    r"(?:password|passwd|pwd|passphrase|secret|token|api[_-]?key|apikey|private[_-]?key|access[_-]?key"
    r"|credentials?|auth[_-]?key|signing[_-]?key|client[_-]?secret)[0-9]*\Z",
    re.IGNORECASE,
)
_PLACEHOLDERS = frozenset(
    [
        "password",
        "passwd",
        "secret",
        "token",
        "changeme",
        "change-me",
        "example",
        "dummy",
        "username",
        "yourpassword",
        "your-password",
        "your_password",
        "your-token",
        "your_token",
        "your-api-key",
        "none",
        "null",
        "redacted",
        "undefined",
        "required",
        "string",
        "optional",
    ]
)
_IDENTIFIER_LIKE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered in _PLACEHOLDERS
        or value[0] in "$%*<{"
        or "****" in value
        or len(set(value)) <= 2
        or lowered.startswith(("os.environ", "getenv", "env."))
    )


def _entropy(value: str) -> float:
    counts = Counter(value)
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _looks_like_secret_value(value: str, quoted: bool) -> bool:
    if _is_placeholder(value):
        return False
    has_digit = any(ch.isdigit() for ch in value)
    if not has_digit and _IDENTIFIER_LIKE.match(value):
        return False  # ``password = get_password``: a reference, not a literal
    if quoted:
        return _entropy(value) >= 3.0
    return len(value) >= 16 and has_digit and _entropy(value) >= 3.5


def scan_secrets(text: str, location: str) -> list[Hit]:
    """SEC-001 detector over any text (all file types, paths, descriptions). Evidence is masked."""
    capped = limit_hit(text, location)
    if capped is not None:
        return [capped]
    line_of = _line_numbers(text)
    found: dict[tuple[int, str], Hit] = {}
    lowered = text.lower()

    def add(
        code: str, offset: int, token: str, severity: Severity | None = None, shown: str = ""
    ) -> None:
        line = line_of(offset)
        evidence = shown or mask_secret(token)
        found.setdefault(
            (line, evidence),
            Hit(
                code,
                format_location(location, line),
                f"hard-coded credential ({code})",
                evidence,
                severity,
            ),
        )

    for m in _PRIVATE_KEY.finditer(text) if "-----begin" in lowered else ():
        add("private_key", m.start(), m.group(0), Severity.CRITICAL, "-----BEGIN ****")
    for code, needles, pattern in _TOKEN_PATTERNS:
        if any(n in lowered for n in needles):
            for m in pattern.finditer(text):
                add(code, m.start(), m.group(0))
        if len(found) >= MAX_HITS:
            break
    for m in _USERINFO.finditer(text) if "://" in text else ():
        if not _is_placeholder(m.group(2)):
            add(
                "url_credentials",
                m.start(),
                m.group(2),
                shown=f"://{mask_secret(m.group(1))}:****@",
            )
    for m in _ASSIGNMENT.finditer(text):
        name, quote, value = m.group(1), m.group(2), m.group(3)
        if _SECRET_NAME.search(name) and _looks_like_secret_value(value, bool(quote)):
            add("high_entropy_secret", m.start(), value, shown=f"{name[:40]}={mask_secret(value)}")
    return sorted(found.values(), key=lambda h: h.location)[:MAX_HITS]


_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    _PRIVATE_KEY,
    *(p for _, _, p in _TOKEN_PATTERNS),
)
_REDACT_INPUT_CAP = 8192  # evidence/messages are short; never run the patterns over unbounded text


def redact_text(text: str) -> str:
    """Replace every recognisable credential in ``text`` with its masked form."""
    out = text[:_REDACT_INPUT_CAP]
    for pattern in _REDACTION_PATTERNS:
        out = pattern.sub(lambda m: mask_secret(m.group(0)), out)
    return _USERINFO.sub(lambda m: m.group(0).replace(m.group(2), "****"), out)


# -------------------------------------------------------------------------- SEC-007 Trojan Source

_ALLOWED_CONTROLS = frozenset("\n\t\f")


def _is_hidden(ch: str) -> bool:
    category = unicodedata.category(ch)
    if category == "Cc":
        return ch not in _ALLOWED_CONTROLS
    return category in ("Cf", "Cs", "Zl", "Zp")


def _script(ch: str) -> str:
    if ch.isascii():
        return "LATIN"
    return unicodedata.name(ch, "UNKNOWN").split(" ", 1)[0]


def _hidden_character_hits(text: str, label: str) -> list[Hit]:
    """Bidi controls, zero-width / other format characters, stray control characters."""
    bad = {ch for ch in set(text) if _is_hidden(ch)}
    if not bad:
        return []
    line_of = _line_numbers(text)
    pattern = re.compile("[" + "".join(re.escape(ch) for ch in sorted(bad)) + "]")
    hits: list[Hit] = []
    for m in pattern.finditer(text):
        char = m.group(0)
        hits.append(
            Hit(
                "hidden_character",
                format_location(label, line_of(m.start())),
                "invisible or bidirectional control character (Trojan Source)",
                f"U+{ord(char):04X} {unicodedata.category(char)}",
            )
        )
        if len(hits) >= MAX_HITS:
            break
    return hits


_WORD = re.compile(r"\w+")


def _mixed_script_hits(text: str, label: str) -> list[Hit]:
    """A word mixing writing systems (Latin letters plus a look-alike from another script) is a
    homoglyph attack in code and docs alike.
    """
    if text.isascii():
        return []
    line_of = _line_numbers(text)
    hits: list[Hit] = []
    for m in _WORD.finditer(text):
        word = m.group(0)
        if word.isascii():
            continue
        scripts = {_script(ch) for ch in word if ch.isalpha()}
        if len(scripts) > 1:
            hits.append(
                Hit(
                    "mixed_script_word",
                    format_location(label, line_of(m.start())),
                    "word mixes writing systems (possible homoglyph)",
                    cap_evidence(ascii(word), 60),
                )
            )
            if len(hits) >= MAX_HITS:
                break
    return hits


_SQL_QUOTED = re.compile(r"'(?:[^']|'')*'|--[^\n]*|/\*.*?\*/", re.DOTALL)


def _sql_identifier_hits(text: str, label: str) -> list[Hit]:
    """Non-ASCII characters outside string literals and comments of an SQL file are identifiers."""
    stripped = _SQL_QUOTED.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    if stripped.isascii():
        return []
    line_of = _line_numbers(stripped)
    return [
        Hit(
            "non_ascii_identifier",
            format_location(label, line_of(m.start())),
            "non-ASCII character in an SQL identifier",
            f"U+{ord(m.group(0)):04X}",
        )
        for m in islice(re.finditer(r"[^\x00-\x7f]", stripped), MAX_HITS)
    ]


def scan_trojan_source(text: str, path: str) -> list[Hit]:
    """SEC-007 detector; ``path`` selects extra identifier checks (``.py`` AST, ``.sql``)."""
    capped = limit_hit(text, path, check_lines=False)
    if capped is not None:
        return [capped]
    hits = _hidden_character_hits(text, path) + _mixed_script_hits(text, path)
    lowered = path.lower()
    if lowered.endswith(".py") and not text.isascii():
        tree, _ = pyscan.parse_python(text, path)
        if tree is not None:
            hits.extend(pyscan.non_ascii_identifiers(tree, path))
    elif lowered.endswith(".sql"):
        hits.extend(_sql_identifier_hits(text, path))
    return hits[:MAX_HITS]


# ----------------------------------------------------------------------- SEC-006 prompt injection

# Whitespace is consumed by exactly one ``\s+`` per word gap (optional words carry their own
# ``\s+``): adjacent ``\s*`` / ``\s+`` pairs would backtrack quadratically on a long run of blanks.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for name, pattern in (
        (
            "ignore_instructions",
            r"\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any|the|your|every)\s+)?"
            r"(?:(?:previous|prior|above|earlier|preceding|safety|security)\s+)?"
            r"(?:instructions?|prompts?|rules?|guidelines?|polic(?:y|ies)|context|restrictions"
            r"|guardrails)\b",
        ),
        ("system_prompt", r"\bsystem\s+(?:prompt|message)\b"),
        (
            "reveal_secrets",
            r"\b(?:reveal|leak|expose|exfiltrate|print|show|output)\s+(?:me\s+)?"
            r"(?:(?:the|your|all|any)\s+)?"
            r"(?:hidden|secret|passwords?|credentials?|api[ _-]?keys?|instructions?"
            r"|system\s+prompt)\b",
        ),
        (
            "role_override",
            r"\byou\s+are\s+now\b|\bfrom\s+now\s+on,?\s+you\b|\bact\s+as\b"
            r"|\bpretend\s+(?:to\s+be|you\s+are)\b|\broleplay\s+as\b",
        ),
        (
            "jailbreak",
            r"\bdeveloper\s+mode\b|\bjailbreak|\bdo\s+anything\s+now\b|\bnew\s+instructions?:",
        ),
        (
            "bypass_controls",
            r"\bbypass\s+(?:(?:the|all|your)\s+)?"
            r"(?:safety|security|polic(?:y|ies)|approvals?|guardrails|restrictions|checks?)\b"
            r"|\bwithout\s+(?:asking|requiring)\s+(?:for\s+)?(?:approval|confirmation|permission)\b",
        ),
        (
            "chat_markup",
            r"<\|(?:im_start|im_end|system|endoftext)\|>|\[/?INST\]|<<SYS>>"
            r"|</?(?:system|instructions?)>|^[ \t]*(?:system|assistant)[ \t]*:",
        ),
    )
)
_BASE64_BLOB = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{100,2000}={0,2}(?![A-Za-z0-9+/=])")


def _injection_match_hits(text: str, label: str) -> Iterable[Hit]:
    line_of = _line_numbers(text)
    for code, pattern in _INJECTION_PATTERNS:
        for m in pattern.finditer(text):
            yield Hit(
                code,
                format_location(label, line_of(m.start())),
                "prompt-injection indicator in human-supplied text",
                cap_evidence(m.group(0), 60),
            )
    for m in _BASE64_BLOB.finditer(text):
        yield Hit(
            "base64_blob",
            format_location(label, line_of(m.start())),
            "long base64-like blob in human-supplied text (may hide instructions)",
            mask_secret(m.group(0)),
        )


def scan_prompt_injection(text: str, location: str) -> list[Hit]:
    """SEC-006 detector for *untrusted human text* (heuristic tripwire, not a boundary)."""
    capped = limit_hit(text, location)
    if capped is not None:
        return [capped]
    hits = list(_injection_match_hits(text, location))
    hits.extend(
        Hit(
            "hidden_unicode",
            h.location,
            "hidden or bidirectional Unicode in human-supplied text",
            h.evidence,
        )
        for h in _hidden_character_hits(text, location)
    )
    return hits[:MAX_HITS]


# ---------------------------------------------------------------------------- Python (AST) façade


def scan_python(text: str, path: str) -> list[Hit]:
    """AST detectors in one pass: dangerous constructs, SQL assembly, egress libraries.

    A file that cannot be parsed (syntax error, nesting bomb, oversize) yields a single
    ``unparseable`` / ``unscannable`` hit - rules turn that into a finding, never a pass. Callers
    that need one family filter on ``Hit.code`` (see ``pyscan``).
    """
    tree, problem = pyscan.parse_python(text, path)
    if tree is None:
        return (
            [problem]
            if problem is not None
            else [Hit(UNSCANNABLE, path, "no syntax tree available")]
        )
    return [
        *pyscan.dangerous_calls(tree, path),
        *pyscan.sql_construction(tree, path),
        *pyscan.egress_uses(tree, path),
    ]
