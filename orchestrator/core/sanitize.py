"""Output sanitisation for untrusted text shown to humans.

Requirement text, agent prose, finding messages and approval comments are all
attacker-influenced. A human reviewer is the final control, so everything that
reaches a terminal, a Markdown report or a diagram passes through here:

* terminals: strip ESC/CSI/OSC sequences, CR, other control and bidi/format
  characters (no line-overwrite or prompt spoofing);
* Markdown: no raw HTML, no active links, untrusted blocks are code-fenced;
* Mermaid/DOT: identifiers are generated, labels are escaped, no ``click``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


def strip_controls(text: str, *, keep_newlines: bool = False) -> str:
    """Remove ANSI escapes and control / format / bidi / zero-width characters."""
    text = _ANSI_RE.sub("", text)
    out: list[str] = []
    for ch in text:
        if ch == "\n" and keep_newlines:
            out.append(ch)
        elif ch == "\t":
            out.append(" ")
        elif unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn") or ch in "  \u0085":
            continue
        else:
            out.append(ch)
    return "".join(out)


def truncate(text: str, max_len: int) -> str:
    """At most ``max_len`` characters, *including* the marker that records what was cut."""
    if len(text) <= max_len:
        return text
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]
    marker = f"...[truncated {len(text)} chars sha256={digest}]"
    if max_len <= len(marker):
        return text[: max(0, max_len)]  # no room for a marker: hard cut
    return text[: max_len - len(marker)] + marker


def sanitize_for_terminal(text: object, max_len: int = 500) -> str:
    """Single-line, control-free, length-capped text safe to print."""
    s = strip_controls(str(text), keep_newlines=False)
    return truncate(s, max_len)


def sanitize_block(text: object, max_len: int = 2000) -> str:
    """Multi-line variant (newlines kept) for untrusted narrative blocks."""
    return truncate(strip_controls(str(text), keep_newlines=True), max_len)


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+!|~])")


def escape_markdown(text: object, max_len: int = 2000) -> str:
    """Escape Markdown/HTML metacharacters; the result renders as literal text."""
    s = strip_controls(str(text), keep_newlines=True)
    s = s.replace("&", "&amp;")
    s = _MD_SPECIAL.sub(r"\\\1", s)
    return truncate(s, max_len)


def code_fence(text: object, lang: str = "", max_len: int = 4000) -> str:
    """Wrap untrusted text in a fence that cannot be closed from the inside."""
    s = sanitize_block(text, max_len)
    longest = max((len(m) for m in re.findall(r"`+", s)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{s}\n{fence}"


def table_cell(text: object, max_len: int = 160) -> str:
    """Safe content for one Markdown table cell."""
    return escape_markdown(sanitize_for_terminal(text, max_len), max_len).replace("\n", " ")


def mermaid_id(index: int) -> str:
    return f"n{index}"


def mermaid_label(text: object, max_len: int = 60) -> str:
    s = sanitize_for_terminal(text, max_len)
    return re.sub(r"[^A-Za-z0-9 _.:/-]", "", s) or "node"
