"""Path, glob and identifier safety — the single source of truth for every
allow/deny decision made on a path or an identifier.

Why this module exists
----------------------
Paths and identifiers originate from untrusted places (agent output, LLM JSON,
requirement text, CLI arguments, resumed state files) and the primary
development platform is Windows (``:`` is an NTFS alternate-data-stream
separator, reserved device names exist, the file system is case-insensitive).
Therefore:

* **Never** use ``fnmatch``, ``PurePath.match``, ``glob`` or ``Path.glob`` for an
  allow/deny decision. ``fnmatch`` lets ``*`` cross ``/``; ``PurePath.match`` is
  right-anchored (``tests/*.py`` would match ``evil/tests/x.py``) and its ``**``
  semantics changed in 3.13. Use :func:`match_glob`.
* **Always** call :func:`validate_rel_path` on a path *before* matching it or
  touching the file system.

Glob grammar (:func:`match_glob`)
---------------------------------
Patterns and paths are split on ``/``. A segment equal to ``**`` matches zero or
more whole segments (and, as the final segment, one or more). ``*``, ``?`` and
``[...]`` never cross ``/``. Matching is anchored at both ends and
**case-sensitive on every platform** (casefold collisions are rejected earlier,
by :func:`casefold_key` uniqueness checks in change sets).
"""

from __future__ import annotations

import functools
import re
import unicodedata

from orchestrator.core.errors import PathViolation

MAX_PATH_CHARS = 240
MAX_PATH_SEGMENTS = 16

# Identifier grammars. Node ids may contain one '.' (implement.<group>) but NEVER ':'.
# ``\Z`` (not ``$``): ``$`` also matches before a trailing newline, so a ``$``-anchored
# pattern would accept an identifier such as "kind" followed by a line feed.
NODE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,40}(\.[a-z0-9][a-z0-9_-]{0,40})?\Z")
KIND_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}\Z")  # no '.', so artifact ids parse unambiguously
KEY_RE = re.compile(r"^\Z|^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\Z")
FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
RUN_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
RULE_ID_RE = re.compile(r"^[A-Z]{2,4}-\d{3}\Z")

_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
    # Windows also reserves the superscript spellings (COM, LPT + U+00B9 / U+00B2 / U+00B3)
    | {f"{dev}{sup}" for dev in ("com", "lpt") for sup in "¹²³"}
)
_DENIED_SEGMENTS = frozenset({".git", ".hg", ".svn"})


def _is_control_or_format(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat in ("Cc", "Cf", "Cs", "Co", "Cn") or ch in "  \u0085"


def validate_rel_path(path: str) -> str:
    """Return the NFC-normalised POSIX relative path or raise :class:`PathViolation`.

    Rejects: non-str, empty, NUL/control/format/bidi/zero-width characters,
    backslashes, ``:`` (drive letters, NTFS alternate data streams), absolute
    and UNC paths, empty/``.``/``..`` segments, segments ending in ``.`` or a
    space, Windows reserved device names (any extension), version-control
    directories (``.git``), and over-long paths.
    """
    if not isinstance(path, str):
        raise PathViolation("path must be a string")
    p = unicodedata.normalize("NFC", path)
    if not p:
        raise PathViolation("empty path")
    if len(p) > MAX_PATH_CHARS:
        raise PathViolation(f"path too long ({len(p)} > {MAX_PATH_CHARS})")
    for ch in p:
        if _is_control_or_format(ch):
            raise PathViolation(f"control/format character in path: U+{ord(ch):04X}")
    if "\\" in p:
        raise PathViolation("backslash in path")
    if ":" in p:
        raise PathViolation("colon in path (drive letter / alternate data stream)")
    if p.startswith("/"):
        raise PathViolation("absolute path")
    segments = p.split("/")
    if len(segments) > MAX_PATH_SEGMENTS:
        raise PathViolation("path too deep")
    for seg in segments:
        if seg in ("", ".", ".."):
            raise PathViolation(f"illegal path segment {seg!r}")
        if seg.endswith((".", " ")):
            raise PathViolation(f"segment ends with dot/space: {seg!r}")
        if seg.lower() in _DENIED_SEGMENTS:
            raise PathViolation(f"version-control directory not allowed: {seg!r}")
        stem = seg.split(".", 1)[0].rstrip(" ").lower()
        if stem in _WINDOWS_RESERVED:
            raise PathViolation(f"reserved device name: {seg!r}")
    return p


def casefold_key(path: str) -> str:
    """Key under which two paths collide on case-insensitive file systems."""
    return unicodedata.normalize("NFC", path).casefold()


def _class_regex(body: str, negate: bool) -> str:
    """One bracket expression as a regex that can never match ``/``.

    Ranges are parsed here instead of being passed through, so a hostile ``[z-a]`` is a
    :class:`PathViolation` (not a ``re.error``) and a range such as ``[.-0]`` (which spans
    ``/``) cannot smuggle a separator into a segment.
    """
    items: list[str] = []
    i = 0
    while i < len(body):
        if i + 2 < len(body) and body[i + 1] == "-":
            low, high = body[i], body[i + 2]
            if low > high:
                raise PathViolation(f"reversed range in glob character class: {body!r}")
            items.append(f"{re.escape(low)}-{re.escape(high)}")
            i += 3
        else:
            items.append(re.escape(body[i]))
            i += 1
    inner = "".join(items)
    return f"[^/{inner}]" if negate else f"(?!/)[{inner}]"


def _tokenize_segment(seg: str) -> list[str | None]:
    """Split a glob segment into single-character regex pieces; ``None`` stands for ``*``."""
    tokens: list[str | None] = []
    i, n = 0, len(seg)
    while i < n:
        ch = seg[i]
        if ch == "*":
            if not tokens or tokens[-1] is not None:  # consecutive stars collapse
                tokens.append(None)
        elif ch == "?":
            tokens.append("[^/]")
        elif ch == "[":
            j = seg.find("]", i + 2 if seg[i + 1 : i + 2] in ("!", "^") else i + 1)
            if j == -1:
                tokens.append(re.escape(ch))
            else:
                body = seg[i + 1 : j]
                negate = body[:1] in ("!", "^")
                if negate:
                    body = body[1:]
                if "/" in body or not body:
                    raise PathViolation(f"invalid character class in glob: {seg!r}")
                tokens.append(_class_regex(body, negate))
                i = j
        else:
            tokens.append(re.escape(ch))
        i += 1
    return tokens


def _translate_segment(seg: str) -> str:
    """Regex matching exactly one path segment, in time linear in the segment length.

    A naive translation (``*`` -> ``[^/]*``) backtracks polynomially in the number of ``*``
    *and* multiplicatively across segments (``*a*a*a*a*a*a*a*a*b`` against 60 ``a`` ran for
    minutes), which would turn an allow/deny glob into a denial-of-service lever. The text
    between two ``*`` is a fixed-length run of single-character matchers, so taking its
    leftmost occurrence is always optimal; atomic groups (Python >= 3.11) commit to it, and an
    outer atomic group per segment stops backtracking from re-entering earlier segments.
    """
    blocks: list[list[str]] = [[]]
    for token in _tokenize_segment(seg):
        if token is None:
            blocks.append([])
        else:
            blocks[-1].append(token)
    head, *rest = ("".join(b) for b in blocks)
    if not rest:
        return f"(?>{head})"
    *middle, tail = rest
    parts = [head, *(f"(?>[^/]*?{m})" for m in middle), f"[^/]*{tail}(?![^/])"]
    return "(?>" + "".join(parts) + ")"


def _build(segments: list[str]) -> str:
    if not segments:
        return ""
    seg, rest = segments[0], segments[1:]
    if seg == "**":
        return ".+" if not rest else "(?:[^/]+/)*" + _build(rest)
    head = _translate_segment(seg)
    return head if not rest else head + "/" + _build(rest)


def _collapse_double_stars(segments: list[str]) -> list[str]:
    """``**/**`` is ``**`` (zero-or-more twice == zero-or-more; as the tail, one-or-more)."""
    out: list[str] = []
    for seg in segments:
        if not (seg == "**" and out and out[-1] == "**"):
            out.append(seg)
    return out


def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a glob pattern (see module docstring). Raises :class:`PathViolation` if unsafe."""
    if not isinstance(pattern, str) or not pattern:  # also keeps unhashable junk away from the cache
        raise PathViolation("glob pattern must be a non-empty string")
    return _compile_glob(pattern)


@functools.lru_cache(maxsize=512)
def _compile_glob(pattern: str) -> re.Pattern[str]:
    if "\\" in pattern or pattern.startswith("/") or ".." in pattern.split("/"):
        raise PathViolation(f"unsafe glob pattern: {pattern!r}")
    if len(pattern) > MAX_PATH_CHARS:
        raise PathViolation("glob pattern too long")
    for ch in pattern:
        if _is_control_or_format(ch):
            raise PathViolation("control character in glob pattern")
    segments = pattern.split("/")
    if any(s == "" for s in segments):
        raise PathViolation(f"empty segment in glob pattern: {pattern!r}")
    try:
        return re.compile("^" + _build(_collapse_double_stars(segments)) + r"\Z")
    except (re.error, RecursionError, OverflowError) as exc:
        raise PathViolation(f"unsupported glob pattern: {pattern!r}") from exc


def match_glob(path: str, pattern: str) -> bool:
    """True iff the (already validated, POSIX, relative) ``path`` matches ``pattern``."""
    return compile_glob(pattern).match(path) is not None


def match_any(path: str, patterns: tuple[str, ...] | list[str] | frozenset[str]) -> bool:
    return any(match_glob(path, pat) for pat in patterns)


def validate_globs(patterns: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate (compile) every pattern; returns them as a tuple."""
    for pat in patterns:
        compile_glob(pat)
    return tuple(patterns)


def parse_artifact_id(artifact_id: str) -> tuple[str, str, int]:
    """Split ``<kind>[.<key>]@v<N>`` into ``(kind, key, version)``.

    ``kind`` never contains ``.``; ``key`` may. Only the canonical spelling is accepted (ASCII
    digits, no leading zeros): ``str.isdigit`` alone would let ``@v<ARABIC-INDIC ONE>`` or
    ``@v01`` alias ``@v1`` under a different id string.
    """
    if not isinstance(artifact_id, str) or "@v" not in artifact_id:
        raise ValueError(f"malformed artifact id: {artifact_id!r}")
    head, _, ver = artifact_id.rpartition("@v")
    if not (ver.isascii() and ver.isdigit()) or str(int(ver)) != ver:
        raise ValueError(f"malformed artifact id version: {artifact_id!r}")
    kind, _, key = head.partition(".")
    if not KIND_RE.match(kind) or not KEY_RE.match(key):
        raise ValueError(f"malformed artifact id: {artifact_id!r}")
    return kind, key, int(ver)


def make_artifact_id(kind: str, key: str, version: int) -> str:
    if not KIND_RE.match(kind):
        raise ValueError(f"invalid artifact kind: {kind!r}")
    if not KEY_RE.match(key):
        raise ValueError(f"invalid artifact key: {key!r}")
    return f"{kind}{'.' + key if key else ''}@v{version}"
