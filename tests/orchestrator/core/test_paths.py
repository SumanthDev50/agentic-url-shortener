"""paths: relative-path validation, glob semantics, identifier grammars, artifact ids."""

from __future__ import annotations

import random
import re
import time
import unicodedata
from typing import Any

import pytest

from orchestrator.core import paths as paths_module
from orchestrator.core.errors import AutonomyViolation, PathViolation
from orchestrator.core.paths import (
    FILENAME_RE,
    KEY_RE,
    KIND_RE,
    MAX_PATH_CHARS,
    MAX_PATH_SEGMENTS,
    NODE_ID_RE,
    RULE_ID_RE,
    RUN_ID_RE,
    casefold_key,
    compile_glob,
    make_artifact_id,
    match_any,
    match_glob,
    parse_artifact_id,
    validate_globs,
    validate_rel_path,
)

# =============================================================================
# validate_rel_path
# =============================================================================

SAFE_PATHS = [
    "a",
    "a.py",
    "src/a/b.py",
    "tests/test_x.py",
    "docs/readme.md",
    ".github/workflows/ci.yml",
    ".env",
    ".gitignore",
    ".gitattributes",
    "a-b_c/d.e",
    "a b/c d",
    "a.b.c/d.e.f",
    "..foo",
    "a..b/c",
    "...x",
    "com10.txt",  # only COM1..COM9 are reserved
    "console.py",
    "auxiliary/x",
    "nulled/x",
    "prn_file.txt",
    "lpt0",
    "x/y/z/w/v/u/t/s/r/q/p/o/n/m/l",  # 15 segments
    "\u00e9.py",
    "\u65e5\u672c\u8a9e/\u30d5\u30a1\u30a4\u30eb.py",
    "emoji_\U0001f600.txt",
    "a\u00a0b",  # NBSP is an ordinary (separator-space) character
    "UPPER/Case.PY",
    "a" * 100 + "/" + "b" * 100,
]

UNSAFE_PATHS = [
    # empty / absolute / UNC
    "",
    "/",
    "/abs",
    "/etc/passwd",
    "//unc/share/x",
    "\\\\unc\\share\\x",
    "\\abs",
    # drive letters and alternate data streams
    "C:/x",
    "c:\\x",
    "C:x",
    "a:b",
    "a.txt:stream",
    "a/b:$DATA",
    "a/b.txt::$DATA",
    ":",
    # backslashes
    "a\\b",
    "a/b\\c",
    "..\\x",
    # traversal and empty/dot segments
    "..",
    "../x",
    "a/..",
    "a/../b",
    "a/b/../../..",
    ".",
    "./a",
    "a/./b",
    "a//b",
    "a/",
    "a/b/",
    "//",
    # trailing dot / space (Windows strips them: aliases)
    "a.",
    "a ",
    "a/b.",
    "a/b ",
    "a. /b",
    "dir./x",
    "dir /x",
    "a/b. ",
    # Windows reserved device names (any case, any extension, trailing space)
    "con",
    "CON",
    "Con.txt",
    "a/con.txt",
    "NUL.tar.gz",
    "aux",
    "prn",
    "COM1",
    "com9.log",
    "LPT1",
    "lpt9.log",
    "a/Lpt5/x",
    "prn .txt",
    "CONIN$",
    "conout$.x",
    "COM\u00b9",
    "com\u00b2.txt",
    "LPT\u00b3",
    # version-control metadata
    ".git",
    ".git/config",
    ".GIT/config",
    "a/.git/hooks/pre-commit",
    ".hg/store",
    ".svn/entries",
    # control, format and bidi characters
    "a\x00b",
    "a\nb",
    "a\rb",
    "a\tb",
    "a\x1bb",
    "a\x7fb",
    "a\x85b",
    "a\u2028b",
    "a\u2029b",
    "a\u200bb",
    "a\u200eb",
    "a\u202eb",
    "a\u2066b",
    "a\u2069b",
    "a\ufeffb",
    "a\u00adb",  # soft hyphen (Cf)
    "a\ue000b",  # private use
    "a\u0378b",  # unassigned
    "a\ud800b",  # lone surrogate
    "a\U000e0001b",  # language tag (Cf)
    # bounds
    "a/" * 16 + "a",  # 17 segments
    "a" * (MAX_PATH_CHARS + 1),
]


class TestValidateRelPath:
    @pytest.mark.parametrize("path", SAFE_PATHS)
    def test_safe_paths_are_returned_normalised(self, path: str) -> None:
        assert validate_rel_path(path) == unicodedata.normalize("NFC", path)

    @pytest.mark.parametrize("path", UNSAFE_PATHS)
    def test_unsafe_paths_raise_path_violation(self, path: str) -> None:
        with pytest.raises(PathViolation):
            validate_rel_path(path)

    @pytest.mark.parametrize("bad", [None, 5, b"a/b", ["a"], ("a",), object()])
    def test_non_strings_are_rejected(self, bad: Any) -> None:
        with pytest.raises(PathViolation, match="string"):
            validate_rel_path(bad)

    def test_limits_are_exact(self) -> None:
        assert validate_rel_path("a" * MAX_PATH_CHARS) == "a" * MAX_PATH_CHARS
        assert validate_rel_path("/".join(["a"] * MAX_PATH_SEGMENTS)) == "/".join(
            ["a"] * MAX_PATH_SEGMENTS
        )
        with pytest.raises(PathViolation, match="too deep"):
            validate_rel_path("/".join(["a"] * (MAX_PATH_SEGMENTS + 1)))

    def test_paths_are_nfc_normalised_before_checking(self) -> None:
        decomposed = "e\u0301.py"
        assert validate_rel_path(decomposed) == "\u00e9.py"
        # a decomposed sequence that composes into a forbidden name is judged after composition
        assert validate_rel_path("\u212a/x") == "K/x"  # KELVIN SIGN normalises to a plain K

    def test_the_error_type_is_a_value_error_and_an_autonomy_violation(self) -> None:
        with pytest.raises(PathViolation) as err:
            validate_rel_path("../x")
        assert isinstance(err.value, ValueError) and isinstance(err.value, AutonomyViolation)
        assert err.value.kind == "autonomy_violation" and err.value.retryable is False

    def test_the_result_is_idempotent(self) -> None:
        for path in SAFE_PATHS:
            once = validate_rel_path(path)
            assert validate_rel_path(once) == once

    def test_randomised_inputs_never_produce_an_unsafe_result(self) -> None:
        rng = random.Random(0xC0FFEE)
        alphabet = [
            *(
                "a",
                "b",
                "C",
                ".",
                "..",
                "/",
                "//",
                "\\",
                ":",
                " ",
                "\t",
                "\n",
                "\u202e",
                "\u200b",
                "\x00",
            ),
            *(
                "con",
                "NUL",
                ".git",
                "e\u0301",
                "\u00e9",
                "~",
                "$",
                "*",
                "?",
                "[",
                "\ud800",
                "\u2028",
            ),
        ]
        accepted = 0
        for _ in range(4000):
            candidate = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))
            try:
                result = validate_rel_path(candidate)
            except PathViolation:
                continue
            accepted += 1
            assert result == unicodedata.normalize("NFC", result)
            assert not result.startswith("/") and "\\" not in result and ":" not in result
            segments = result.split("/")
            assert all(seg not in ("", ".", "..") for seg in segments)
            assert all(not seg.endswith((".", " ")) for seg in segments)
            assert all(seg.lower() not in (".git", ".hg", ".svn") for seg in segments)
            assert all(
                unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Co", "Cn") for c in result
            )
        assert accepted > 100  # the corpus exercised both outcomes

    @pytest.mark.xfail(
        strict=True,
        reason="known gap: NTFS 8.3 short names (GIT~1 aliases .git) are not rejected",
    )
    def test_windows_short_names_that_alias_dot_git_are_rejected(self) -> None:
        with pytest.raises(PathViolation):
            validate_rel_path("GIT~1/config")


class TestCasefoldKey:
    def test_case_and_normalisation_collisions_share_a_key(self) -> None:
        assert casefold_key("A/B.py") == casefold_key("a/b.PY")
        assert casefold_key("e\u0301") == casefold_key("\u00c9")  # NFD vs NFC, different case
        assert casefold_key("Stra\u00dfe") == casefold_key("STRASSE")  # full case folding

    def test_different_paths_have_different_keys(self) -> None:
        assert casefold_key("a/b") != casefold_key("a/c")
        assert casefold_key("a/b") != casefold_key("ab")


# =============================================================================
# globs
# =============================================================================

MATCH_CASES: list[tuple[str, str, bool]] = [
    # single segment wildcards never cross "/"
    ("tests/*.py", "tests/a.py", True),
    ("tests/*.py", "tests/sub/a.py", False),
    (
        "tests/*.py",
        "evil/tests/x.py",
        False,
    ),  # right-anchored matchers (PurePath.match) would accept this
    ("tests/*.py", "tests/a.pyc", False),
    ("tests/*.py", "tests/.py", True),  # * may be empty
    ("tests/*.py", "xtests/a.py", False),
    ("*", "a", True),
    ("*", "a/b", False),
    ("*.py", "a/b.py", False),
    ("a*", "abc", True),
    ("a*c", "ac", True),
    ("a*c", "abbc", True),
    ("a*c", "abcd", False),
    ("*a*", "xax", True),
    ("*a*", "xxx", False),
    ("a?c", "abc", True),
    ("a?c", "ac", False),
    ("a?c", "a/c", False),
    ("???", "abc", True),
    ("???", "ab", False),
    # "**" is a whole-segment wildcard
    ("**", "a", True),
    ("**", "a/b/c", True),
    ("src/**", "src/a", True),
    ("src/**", "src/a/b/c", True),
    ("src/**", "src", False),  # as the last segment ** needs at least one segment
    ("src/**", "srcx/a", False),
    ("src/**", "evil/src/a", False),
    ("**/x.py", "x.py", True),
    ("**/x.py", "a/x.py", True),
    ("**/x.py", "a/b/c/x.py", True),
    ("**/x.py", "ax.py", False),
    ("**/x.py", "a/x.pyc", False),
    ("src/**/x", "src/x", True),
    ("src/**/x", "src/a/x", True),
    ("src/**/x", "src/a/b/x", True),
    ("src/**/x", "src/xx", False),
    ("src/**/x", "src/a/xx", False),
    ("**/**", "a", True),
    ("**/**", "a/b", True),
    ("a/**/**/b", "a/b", True),
    ("a/**/**/b", "a/x/y/b", True),
    ("**/a/**/b/**", "a/b/c", True),
    ("**/a/**/b/**", "x/a/y/b/z/w", True),
    ("**/a/**/b/**", "a/b", False),
    ("a**", "abc", True),  # "**" inside a segment is just "*"
    ("a**", "a/b", False),
    ("**a", "xa", True),
    ("**a", "x/a", False),
    # anchoring and case
    ("a", "a", True),
    ("a", "ba", False),
    ("a", "ab", False),
    ("a", "a/b", False),
    ("Tests/*.py", "tests/a.py", False),
    ("tests/*.py", "Tests/a.py", False),
    ("*.PY", "a.py", False),
    ("a/b", "a/b/", False),
    # trailing newline must not slip through a "$"-anchored translation
    ("tests/*.py", "tests/a.py\n", False),
    ("a", "a\n", False),
    # character classes
    ("x[abc].py", "xa.py", True),
    ("x[abc].py", "xd.py", False),
    ("x[a-c].py", "xb.py", True),
    ("x[a-c].py", "xd.py", False),
    ("x[!a-c].py", "xd.py", True),
    ("x[!a-c].py", "xb.py", False),
    ("x[^a-c].py", "xd.py", True),
    ("x[^a-c].py", "xa.py", False),
    ("x[^a-c].py", "x/.py", False),  # a negated class never matches "/"
    ("x[a-]y", "x-y", True),
    ("x[-a]y", "x-y", True),
    ("x[a-]y", "xby", False),
    ("x[.-0]y", "x/y", False),  # this range spans "/" but classes never cross a separator
    ("x[.-0]y", "x.y", True),
    ("x[*]y", "x*y", True),
    ("x[*]y", "xay", False),
    ("x[?]y", "x?y", True),
    ("x[[]y", "x[y", True),
    ("x[a&&b]y", "x&y", True),
    ("x[a|b]y", "x|y", True),
    ("x[~]y", "x~y", True),
    # unclosed brackets are literals; "/" splits segments before brackets are parsed
    ("[abc", "[abc", True),
    ("a[", "a[", True),
    ("[!", "[!", True),
    ("[", "[", True),
    ("[/]", "[/]", True),
    ("x[a/b]y", "x[a/b]y", True),
    ("x[a/b]y", "xa", False),
    # regex metacharacters are literals
    ("a.b", "a.b", True),
    ("a.b", "axb", False),
    ("a+b", "a+b", True),
    ("a+b", "aab", False),
    ("(x)", "(x)", True),
    ("a{2}", "a{2}", True),
    ("a{2}", "aa", False),
    ("a$", "a$", True),
    ("^a", "^a", True),
    ("a|b", "a|b", True),
    ("a|b", "a", False),
    ("a%b", "a%b", True),
    # patterns that name dot-directories (used by the default excludes)
    (".git/**", ".git/config", True),
    (".git/**", ".git/a/b", True),
    (".git/**", ".gitignore", False),
    (".git/**", "x/.git/config", False),
    ("**/__pycache__/**", "a/__pycache__/x.pyc", True),
    ("**/__pycache__/**", "__pycache__/x.pyc", True),
    ("**/__pycache__/**", "a/b/__pycache__/c/d.pyc", True),
    ("**/__pycache__/**", "a/__pycache__", False),
    (".env.*", ".env.local", True),
    (".env.*", ".env", False),
    ("*.egg-info/**", "x.egg-info/PKG-INFO", True),
    ("*.egg-info/**", "a/x.egg-info/PKG-INFO", False),
    ("**/*.pem", "keys/server.pem", True),
    ("**/*.pem", "server.pem", True),
    ("**/*.pem", "server.pem.txt", False),
]


class TestMatchGlob:
    @pytest.mark.parametrize(("pattern", "path", "expected"), MATCH_CASES)
    def test_match_table(self, pattern: str, path: str, expected: bool) -> None:
        assert match_glob(path, pattern) is expected

    def test_unicode_is_matched_literally_and_case_sensitively(self) -> None:
        assert match_glob("\u00e9/x", "\u00e9/*")
        assert not match_glob("\u00c9/x", "\u00e9/*")
        assert match_glob("\u65e5\u672c/x.py", "**/*.py")

    def test_match_any(self) -> None:
        assert match_any("a/b.py", ("*.txt", "a/*.py"))
        assert not match_any("a/b.py", ("*.txt", "b/*.py"))
        assert not match_any("a/b.py", ())
        assert match_any("a/b.py", ["a/**"])
        assert match_any("a/b.py", frozenset({"a/**"}))

    def test_compiled_patterns_are_cached(self) -> None:
        assert compile_glob("src/**/x") is compile_glob("src/**/x")


INVALID_PATTERNS = [
    "",
    "/",
    "/abs",
    "a//b",
    "a/",
    "//",
    "a\\b",
    "\\",
    "..",
    "../x",
    "a/..",
    "a/../b",
    "[]",
    "[!]",
    "[^]",
    "x[]y",
    "[z-a]",
    "x[9-0]y",
    "a\x00b",
    "a\nb",
    "a\tb",
    "a\u202eb",
    "a\u200bb",
    "a" * (MAX_PATH_CHARS + 1),
]


class TestGlobValidation:
    @pytest.mark.parametrize("pattern", INVALID_PATTERNS)
    def test_unsafe_or_malformed_patterns_are_rejected_with_path_violation(
        self, pattern: str
    ) -> None:
        with pytest.raises(PathViolation):
            compile_glob(pattern)
        with pytest.raises(PathViolation):
            match_glob("a", pattern)

    @pytest.mark.parametrize("bad", [None, 5, b"*", ["*"]])
    def test_non_string_patterns_are_rejected(self, bad: Any) -> None:
        with pytest.raises(PathViolation):
            compile_glob(bad)

    @pytest.mark.parametrize(
        "pattern",
        [
            "*",
            "**",
            "...",
            "..a",
            "a..",
            "*.py",
            ".git/**",
            "[",
            "[abc",
            "a[",
            "[!",
            "[a-c]",
            "x[!a-c]*",
            "a/**/b/**",
        ],
    )
    def test_odd_but_legal_patterns_compile(self, pattern: str) -> None:
        compile_glob(pattern)
        assert match_glob("a/b", pattern) in (True, False)  # matching never raises

    def test_validate_globs_returns_a_tuple_and_rejects_the_first_bad_pattern(self) -> None:
        assert validate_globs(["a/**", "*.py"]) == ("a/**", "*.py")
        assert validate_globs(()) == ()
        with pytest.raises(PathViolation):
            validate_globs(("a/**", "../x", "*.py"))

    def test_class_translation_emits_no_regex_warnings(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for pattern in ("x[[]y", "x[a&&b]y", "x[a||b]y", "x[~~]y", "x[a-c&&d]y", "x[[a]]y"):
                paths_module._compile_glob.__wrapped__(
                    pattern
                )  # bypass the cache: compile for real


class _Reference:
    """An independent, obviously-correct glob matcher (dynamic programming, no regexes)."""

    @staticmethod
    def tokens(seg: str) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        i = 0
        while i < len(seg):
            ch = seg[i]
            if ch == "*":
                if not out or out[-1][0] != "star":
                    out.append(("star", None))
            elif ch == "?":
                out.append(("one", None))
            elif ch == "[":
                j = seg.index("]", i + 1 + (1 if seg[i + 1] in "!^" else 0))
                body = seg[i + 1 : j]
                negate = body[0] in "!^"
                body = body[1:] if negate else body
                chars: set[str] = set()
                k = 0
                while k < len(body):
                    if k + 2 < len(body) and body[k + 1] == "-":
                        chars.update(chr(c) for c in range(ord(body[k]), ord(body[k + 2]) + 1))
                        k += 3
                    else:
                        chars.add(body[k])
                        k += 1
                out.append(("set", (chars, negate)))
                i = j
            else:
                out.append(("lit", ch))
            i += 1
        return out

    @classmethod
    def segment(cls, pattern: str, text: str) -> bool:
        toks = cls.tokens(pattern)
        # dp[j] = tokens[:i] can match text[:j]
        dp = [True] + [False] * len(text)
        for kind, arg in toks:
            nxt = [False] * (len(text) + 1)
            if kind == "star":
                seen = False
                for j in range(len(text) + 1):
                    seen = seen or dp[j]
                    nxt[j] = seen
            else:
                for j in range(1, len(text) + 1):
                    ch = text[j - 1]
                    ok = (
                        True
                        if kind == "one"
                        else ch == arg
                        if kind == "lit"
                        else ((ch in arg[0]) != arg[1])
                    )
                    nxt[j] = dp[j - 1] and ok
            dp = nxt
        return dp[len(text)]

    @classmethod
    def match(cls, pattern: str, path: str) -> bool:
        ps, xs = pattern.split("/"), path.split("/")
        memo: dict[tuple[int, int], bool] = {}

        def go(i: int, j: int) -> bool:
            key = (i, j)
            if key in memo:
                return memo[key]
            if i == len(ps):
                result = j == len(xs)
            elif ps[i] == "**":
                if i == len(ps) - 1:
                    result = j < len(xs)
                else:
                    result = any(go(i + 1, k) for k in range(j, len(xs) + 1))
            else:
                result = j < len(xs) and cls.segment(ps[i], xs[j]) and go(i + 1, j + 1)
            memo[key] = result
            return result

        return go(0, 0)


class TestAgainstReferenceMatcher:
    def test_randomised_patterns_agree_with_the_reference(self) -> None:
        rng = random.Random(20260101)
        atoms = ["a", "b", "c", ".", "-", "*", "?", "[ab]", "[a-b]", "[!a]", "[^b]", "[c.]"]

        def pattern_segment() -> str:
            if rng.random() < 0.25:
                return "**"
            return "".join(rng.choice(atoms) for _ in range(rng.randint(1, 4)))

        def path_segment() -> str:
            return "".join(rng.choice("abc.-") for _ in range(rng.randint(1, 3)))

        agreed = matched = 0
        for _ in range(6000):
            segments = [pattern_segment() for _ in range(rng.randint(1, 4))]
            if ".." in segments:
                continue  # rejected by design (traversal), tested elsewhere
            pattern = "/".join(segments)
            path = "/".join(path_segment() for _ in range(rng.randint(1, 5)))
            expected = _Reference.match(pattern, path)
            assert match_glob(path, pattern) is expected, (pattern, path)
            agreed += 1
            matched += expected
        assert agreed > 5000 and 200 < matched < agreed - 200  # both outcomes occurred often

    def test_exhaustive_agreement_on_a_tiny_alphabet(self) -> None:
        patterns = [
            "*",
            "a*",
            "*a",
            "*a*",
            "a*b",
            "?*",
            "*?",
            "**",
            "a/**",
            "**/a",
            "**/a/**",
            "*/*",
            "a/*/b",
            "[ab]*",
        ]
        words = ["a", "b", "ab", "ba", "aa", "bb", "aba"]
        paths = list(words)
        paths += [f"{x}/{y}" for x in words[:4] for y in words[:4]]
        paths += [f"{x}/{y}/{z}" for x in ("a", "b") for y in ("a", "b") for z in ("a", "b", "ab")]
        for pattern in patterns:
            for path in paths:
                assert match_glob(path, pattern) is _Reference.match(pattern, path), (pattern, path)


class TestGlobsAreLinearTime:
    """A glob is an allow/deny decision function: it must not be a denial-of-service lever."""

    BUDGET_S = 1.0

    def _timed(self, pattern: str, path: str) -> float:
        start = time.perf_counter()
        assert match_glob(path, pattern) is False
        return time.perf_counter() - start

    def test_many_stars_in_one_segment(self) -> None:
        assert self._timed("*a" * 10 + "*b", "a" * 60) < self.BUDGET_S

    def test_many_stars_in_many_segments(self) -> None:
        pattern = "/".join(["*a*a*a*a"] * 12) + "/b"
        path = "/".join(["a" * 15] * 12) + "/c"
        assert self._timed(pattern, path) < self.BUDGET_S

    def test_many_double_stars_separated_by_literals(self) -> None:
        pattern = "/".join(["**", "a"] * 8) + "/**/b"
        path = "/".join(["a"] * 15) + "/c"
        assert self._timed(pattern, path) < self.BUDGET_S

    def test_consecutive_double_stars_collapse(self) -> None:
        pattern = "/".join(["**"] * 60) + "/x"
        assert compile_glob(pattern).pattern == compile_glob("**/x").pattern
        assert self._timed(pattern, "/".join(["a"] * 15) + "/y") < self.BUDGET_S

    def test_longest_patterns_against_longest_paths(self) -> None:
        pattern = ("*?" * 120)[: MAX_PATH_CHARS - 1] + "x"
        assert len(pattern) == MAX_PATH_CHARS
        assert self._timed(pattern, "a" * (MAX_PATH_CHARS - 1) + "b") < self.BUDGET_S
        classes = "[ab]*" * 40
        assert self._timed(classes + "c", "ab" * 100) < self.BUDGET_S


# =============================================================================
# identifier grammars
# =============================================================================


class TestIdentifierGrammars:
    @pytest.mark.parametrize(
        ("regex", "valid", "invalid"),
        [
            (
                NODE_ID_RE,
                [
                    "intake",
                    "a",
                    "implement.core",
                    "implement.a1",
                    "author-tests",
                    "a_b-c",
                    "x" * 41,
                ],
                [
                    "",
                    "Intake",
                    "1a",
                    "a:b",
                    "a.b.c",
                    "implement.",
                    ".x",
                    "a/b",
                    "a b",
                    "x" * 42,
                    "implement.-x",
                ],
            ),
            (
                KIND_RE,
                ["request", "a", "design_doc", "x" * 41],
                ["", "Request", "1a", "a.b", "a-b", "a b", "x" * 42, "../x"],
            ),
            (
                KEY_RE,
                ["", "a", "implement.core", "Core", "a_b-c.d", "9", "x" * 81],
                ["-a", ".a", "_a", "a/b", "a b", "a:b", "x" * 82, "a\\b"],
            ),
            (
                FILENAME_RE,
                ["a", "a.py", "evil-name.py", "A_b.c", "x" * 100],
                ["", ".hidden", "-x", "a/b", "../x", "a b", "x" * 101],
            ),
            (
                RUN_ID_RE,
                ["run1", "R", "run-1", "run_1", "r.1", "x" * 64],
                ["", "1run", "-run", ".run", "a/b", "a b", "x" * 65, "..", "a\\b"],
            ),
            (
                RULE_ID_RE,
                ["SEC-001", "CMP-003", "CHG-008", "QLT-001", "AB-123", "ABCD-999"],
                ["", "SEC-01", "SEC-0001", "sec-001", "SEC001", "A-001", "ABCDE-001", "SEC-00a"],
            ),
        ],
    )
    def test_grammar(self, regex: re.Pattern[str], valid: list[str], invalid: list[str]) -> None:
        for value in valid:
            assert regex.match(value), value
        for value in invalid:
            assert not regex.match(value), value

    @pytest.mark.parametrize(
        "regex", [NODE_ID_RE, KIND_RE, KEY_RE, FILENAME_RE, RUN_ID_RE, RULE_ID_RE]
    )
    def test_a_trailing_line_feed_is_never_accepted(self, regex: re.Pattern[str]) -> None:
        """``$`` matches before a final newline; the grammars are anchored with ``\\Z`` instead."""
        samples = {
            NODE_ID_RE: "intake",
            KIND_RE: "request",
            KEY_RE: "implement.core",
            FILENAME_RE: "a.py",
            RUN_ID_RE: "run1",
            RULE_ID_RE: "SEC-001",
        }
        sample = samples[regex]
        assert regex.match(sample)
        for suffix in ("\n", "\r\n", "\r", " ", "\x00", "\n\n"):
            assert not regex.match(sample + suffix), repr(suffix)
        assert not regex.match("\n" + sample)

    def test_the_empty_key_is_valid_but_a_lone_newline_is_not(self) -> None:
        assert KEY_RE.match("")
        assert not KEY_RE.match("\n")


# =============================================================================
# artifact ids
# =============================================================================


class TestArtifactIds:
    @pytest.mark.parametrize(
        ("kind", "key", "version", "expected"),
        [
            ("request", "", 1, "request@v1"),
            ("design_doc", "", 12, "design_doc@v12"),
            ("changeset", "implement.core", 3, "changeset.implement.core@v3"),
            ("changeset", "author-tests", 1, "changeset.author-tests@v1"),
            ("changeset", "a.b.c", 7, "changeset.a.b.c@v7"),
            ("x", "9", 0, "x.9@v0"),
        ],
    )
    def test_make_and_parse_round_trip(
        self, kind: str, key: str, version: int, expected: str
    ) -> None:
        artifact_id = make_artifact_id(kind, key, version)
        assert artifact_id == expected
        assert parse_artifact_id(artifact_id) == (kind, key, version)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "request",
            "request@",
            "request@v",
            "request@vx",
            "request@v-1",
            "request@v1.5",
            "request@v1\n",
            "request@v 1",
            "request@v\u0661",  # an Arabic-Indic digit is str.isdigit() but not an ASCII version
            "@v1",
            ".key@v1",
            "Request@v1",
            "re quest@v1",
            "request.key/x@v1",
            "request.k\ney@v1",
            "request.key\n@v1",
            "request.-key@v1",
            "../x@v1",
            "a@v1@v2x",
            "request@v01",
            "request@v00",
            "request@v1_000",
        ],
    )
    def test_malformed_ids_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_artifact_id(bad)

    @pytest.mark.parametrize("bad", [None, 5, b"request@v1", ["request@v1"]])
    def test_non_strings_are_rejected(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="malformed"):
            parse_artifact_id(bad)

    @pytest.mark.parametrize(
        ("kind", "key"),
        [
            ("Bad", ""),
            ("a.b", ""),
            ("", ""),
            ("a b", ""),
            ("k", "bad key"),
            ("k", "a/b"),
            ("k", "k\n"),
            ("k\n", ""),
            ("k", "-x"),
        ],
    )
    def test_make_artifact_id_validates_its_parts(self, kind: str, key: str) -> None:
        with pytest.raises(ValueError):
            make_artifact_id(kind, key, 1)

    def test_a_dotted_key_is_unambiguous_because_kinds_have_no_dots(self) -> None:
        assert parse_artifact_id("changeset.implement.core@v1") == (
            "changeset",
            "implement.core",
            1,
        )
