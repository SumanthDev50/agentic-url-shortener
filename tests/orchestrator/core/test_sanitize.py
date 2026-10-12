"""sanitize: untrusted text must be inert in terminals, Markdown and diagram sources."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pytest

from orchestrator.core.sanitize import (
    code_fence,
    escape_markdown,
    mermaid_id,
    mermaid_label,
    sanitize_block,
    sanitize_for_terminal,
    strip_controls,
    table_cell,
    truncate,
)

ESC = "\x1b"


class TestStripControls:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            (f"{ESC}[31mred{ESC}[0m", "red"),
            (f"{ESC}[1;38;5;196mx{ESC}[m", "x"),
            (f"{ESC}[2Jclear", "clear"),
            (f"{ESC}[?25lhidden{ESC}[?25h", "hidden"),
            (f"{ESC}]0;window title\x07text", "text"),
            (f"{ESC}]8;;http://evil.example\x07link{ESC}]8;;\x07", "link"),
            (f"{ESC}]0;title{ESC}\\text", "text"),
            (f"{ESC}Mtext", "text"),
            (f"{ESC}c reset", "c reset"),  # ESC is dropped; the stray letter is harmless text
            ("a\rb", "ab"),
            ("a\x00b", "ab"),
            ("a\x07b", "ab"),
            ("a\x08b", "ab"),
            ("a\x7fb", "ab"),
            ("a\x85b", "ab"),
            ("a\x9b31mb", "a31mb"),  # an 8-bit CSI is dropped; the harmless remainder stays as text
            ("a\u2028b", "ab"),
            ("a\u2029b", "ab"),
            ("a\u200bb\u200cc\u200dd", "abcd"),
            ("a\u202eb", "ab"),
            ("a\u202ab\u202bc\u202cd\u202de", "abcde"),
            ("a\u2066b\u2067c\u2068d\u2069e", "abcde"),
            ("a\ufeffb", "ab"),
            ("a\u00adb", "ab"),
            ("a\ue000b", "ab"),
            ("a\u0378b", "ab"),
            ("a\ud800b", "ab"),
            ("a\U000e0041b", "ab"),
            ("a\tb", "a b"),
        ],
    )
    def test_dangerous_characters_are_removed(self, raw: str, clean: str) -> None:
        assert strip_controls(raw) == clean

    def test_no_escape_character_survives_any_construction(self) -> None:
        nasty = [
            f"{ESC}",
            f"{ESC}{ESC}[31m",
            f"{ESC}[",
            f"{ESC}]0;never terminated",
            f"{ESC}[31{ESC}[0m",
            f"{ESC}P1$r\x1b\\",
            f"x{ESC}",
        ]
        for raw in nasty:
            assert ESC not in strip_controls(raw)
            assert ESC not in strip_controls(raw, keep_newlines=True)

    def test_newlines_are_kept_only_on_request_and_carriage_returns_never(self) -> None:
        assert strip_controls("a\nb\r\nc") == "abc"
        assert strip_controls("a\nb\r\nc", keep_newlines=True) == "a\nb\nc"
        assert "\r" not in strip_controls("a\rb", keep_newlines=True)

    def test_ordinary_text_is_untouched(self) -> None:
        # accents, CJK, emoji, a combining mark, right-to-left letters (not bidi controls)
        text = "café 日本語 😀 é مرحبا a-b_c.d/e:f"
        assert strip_controls(text) == text

    def test_whitespace_that_is_not_a_control_is_kept(self) -> None:
        assert strip_controls("a b\u00a0c") == "a b\u00a0c"


class TestTruncate:
    @pytest.mark.parametrize("length", [1, 49, 50, 51, 100, 101, 500, 5000])
    @pytest.mark.parametrize("cap", [0, 1, 10, 49, 50, 100, 160, 500])
    def test_the_cap_is_a_hard_limit_marker_included(self, length: int, cap: int) -> None:
        result = truncate("x" * length, cap)
        if length <= cap:
            assert result == "x" * length
        else:
            assert len(result) <= max(cap, 0)

    def test_short_text_is_returned_unchanged(self) -> None:
        assert truncate("abc", 3) == "abc"
        assert truncate("", 0) == ""
        assert truncate("abc", 10) == "abc"

    def test_the_marker_records_the_original_length_and_a_digest(self) -> None:
        text = "y" * 1000
        result = truncate(text, 200)
        digest = hashlib.sha256(text.encode()).hexdigest()[:8]
        assert len(result) == 200
        assert result.endswith(f"...[truncated 1000 chars sha256={digest}]")
        assert result.startswith("y" * 100)

    def test_a_cap_too_small_for_the_marker_hard_cuts(self) -> None:
        assert truncate("abcdefghij" * 10, 8) == "abcdefgh"
        assert truncate("abc", 0) == ""
        assert truncate("abc", -3) == ""

    def test_the_digest_is_deterministic_and_text_dependent(self) -> None:
        a, b = "a" * 300, "b" * 300
        assert truncate(a, 100) == truncate(a, 100)
        assert truncate(a, 100) != truncate(b, 100)

    def test_lone_surrogates_do_not_crash(self) -> None:
        assert len(truncate("\ud800" * 300, 100)) == 100


class TestTerminalSanitisers:
    def test_sanitize_for_terminal_is_single_line_control_free_and_capped(self) -> None:
        out = sanitize_for_terminal(
            f"{ESC}[31mred{ESC}[0m\nsecond\r\nthird\u202e" + "z" * 1000, 120
        )
        assert "\n" not in out and "\r" not in out and ESC not in out and "\u202e" not in out
        assert len(out) == 120
        assert out.startswith("redsecondthird")

    def test_non_string_input_is_stringified(self) -> None:
        assert sanitize_for_terminal(12345) == "12345"
        assert sanitize_for_terminal(None) == "None"
        assert sanitize_for_terminal(ValueError(f"bad{ESC}[0m")) == "bad"

    def test_hostile_str_methods_are_not_trusted_to_be_safe_text(self) -> None:
        class Evil:
            def __str__(self) -> str:
                return f"{ESC}]0;pwned\x07boom\u202e"

        assert sanitize_for_terminal(Evil()) == "boom"

    def test_default_cap_is_500(self) -> None:
        assert len(sanitize_for_terminal("a" * 5000)) == 500

    def test_sanitize_block_keeps_newlines_but_nothing_else_dangerous(self) -> None:
        out = sanitize_block(f"line1\n{ESC}[31mline2{ESC}[0m\r\nline3\u202e", 100)
        assert out == "line1\nline2\nline3"
        assert len(sanitize_block("a\n" * 5000, 300)) == 300

    def test_sanitize_block_default_cap_is_2000(self) -> None:
        assert len(sanitize_block("a" * 9000)) == 2000


class TestEscapeMarkdown:
    SPECIAL = "\\`*_{}[]()<>#+!|~"

    def test_every_special_character_is_backslash_escaped(self) -> None:
        out = escape_markdown(self.SPECIAL)
        assert out == "".join("\\" + ch for ch in self.SPECIAL)

    def test_links_images_html_and_headings_become_literal_text(self) -> None:
        out = escape_markdown(
            "[click](http://evil.example) ![x](http://evil/x.png) <script>alert(1)</script>\n# H1"
        )
        assert "[click]" not in out and "![" not in out and "<script>" not in out
        assert out.startswith("\\[click\\]\\(http://evil.example\\)")
        assert "\\<script\\>" in out
        assert "\n\\# H1" in out

    def test_ampersands_are_html_escaped_first_so_entities_cannot_be_forged(self) -> None:
        assert escape_markdown("&lt;b&gt; & &amp;") == "&amp;lt;b&amp;gt; &amp; &amp;amp;"
        assert escape_markdown("&lt;") == "&amp;lt;"
        assert escape_markdown("a & b") == "a &amp; b"

    def test_no_unescaped_special_character_remains(self) -> None:
        hostile = "[a](b) *c* _d_ `e` <f> # g | h ~i~ !j {k} +l \\m"
        out = escape_markdown(hostile)
        for ch in "[]()*_`<>#|~!{}+":
            for match in re.finditer(re.escape(ch), out):
                preceding = out[: match.start()]
                backslashes = len(preceding) - len(preceding.rstrip("\\"))
                assert backslashes % 2 == 1, (ch, out)

    def test_controls_are_stripped_and_length_is_capped(self) -> None:
        out = escape_markdown(f"{ESC}[31m**bold**{ESC}[0m" + "x" * 5000, 100)
        assert ESC not in out
        assert len(out) <= 100

    def test_plain_text_is_unchanged(self) -> None:
        assert escape_markdown("hello world 123 .,;:") == "hello world 123 .,;:"


class TestCodeFence:
    def _lines(self, fenced: str) -> list[str]:
        return fenced.split("\n")

    def test_simple_text_is_wrapped_in_a_triple_fence(self) -> None:
        assert code_fence("hello") == "```\nhello\n```"
        assert code_fence("hello", "python") == "```python\nhello\n```"

    @pytest.mark.parametrize("run", [1, 2, 3, 4, 10, 40])
    def test_the_fence_is_longer_than_any_backtick_run_inside(self, run: int) -> None:
        body = "before\n" + "`" * run + "\n# injected heading\n" + "`" * run + "\nafter"
        lines = self._lines(code_fence(body))
        fence = lines[0]
        assert set(fence) == {"`"} and len(fence) == max(3, run + 1)
        assert lines[-1] == fence
        assert all(fence not in line for line in lines[1:-1])  # nothing inside can close the fence

    def test_control_characters_inside_are_removed_and_text_is_capped(self) -> None:
        out = code_fence(f"{ESC}[31mred{ESC}[0m\r\nline", max_len=50)
        assert out == "```\nred\nline\n```"
        long = code_fence("x" * 10_000, max_len=100)
        assert len(long) < 200

    def test_a_non_string_is_stringified(self) -> None:
        assert code_fence(5) == "```\n5\n```"


class TestTableCell:
    def test_cells_are_single_line_escaped_and_capped(self) -> None:
        out = table_cell("a|b\nc `d` <e>", 160)
        assert "\n" not in out
        assert "|" not in out.replace("\\|", "")
        assert out == "a\\|bc \\`d\\` \\<e\\>"

    def test_cap(self) -> None:
        assert len(table_cell("x" * 1000)) <= 160
        assert len(table_cell("|" * 1000, 50)) <= 50


class TestMermaid:
    def test_ids_are_generated_never_derived_from_text(self) -> None:
        assert [mermaid_id(i) for i in (0, 1, 25)] == ["n0", "n1", "n25"]

    @pytest.mark.parametrize(
        "hostile",
        [
            'A"]; click A "javascript:alert(1)',
            "A]\nclick A callback",
            "x --> y; style y fill:red",
            "<img src=x onerror=alert(1)>",
            "`code` {braces} (parens) [brackets] | pipe | ; semi",
            "%%{init: {'securityLevel':'loose'}}%%",
            f"{ESC}[31mred",
            "\u202e",
        ],
    )
    def test_labels_cannot_break_out_of_a_node_definition(self, hostile: str) -> None:
        label = mermaid_label(hostile)
        assert label  # never empty
        assert not set(label) & set("\"[](){}<>|;`%'=\n\r\\&#"), label
        assert ESC not in label and "\u202e" not in label

    def test_labels_keep_readable_characters_and_are_capped(self) -> None:
        assert mermaid_label("implement.core: step 2/3 - ok_1") == "implement.core: step 2/3 - ok_1"
        assert len(mermaid_label("x" * 500)) <= 60
        assert len(mermaid_label("x" * 500, 20)) <= 20

    @pytest.mark.parametrize("empty", ["", "\u202e", "[]", '"""', "\n"])
    def test_a_label_that_sanitises_to_nothing_gets_a_placeholder(self, empty: str) -> None:
        assert mermaid_label(empty) == "node"

    def test_non_strings_are_stringified(self) -> None:
        assert mermaid_label(42) == "42"
        assert mermaid_label(None) == "None"


def test_every_public_helper_accepts_arbitrary_objects() -> None:
    odd: list[Any] = [None, 0, 1.5, b"bytes", ["l"], {"k": "v"}, object, ValueError("x")]
    for value in odd:
        for fn in (
            sanitize_for_terminal,
            sanitize_block,
            escape_markdown,
            code_fence,
            table_cell,
            mermaid_label,
        ):
            assert isinstance(fn(value), str)
