"""Scanner primitives shared by every detector: ``Hit``, hard input caps, masking.

Scanners are *rule-neutral*: they return :class:`Hit` objects and the rule that called them decides
the rule id, category and severity.  Two properties are enforced here so no detector can forget
them:

* **Hard input caps.**  A text above ``MAX_TEXT_CHARS`` (or, for regex-driven detectors, with a line
  above ``MAX_LINE_CHARS``) is *never* scanned and never silently passed: the caller gets an
  ``unscannable`` hit that rules turn into a HIGH finding.  Bounded work per input is what makes the
  (linear-time) detectors safe to run on hostile agent output.
* **Masked evidence.**  A hit never carries a secret: ``mask_secret`` keeps at most four leading
  characters, and ``cap_evidence`` bounds the length.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.core.types import Severity

MAX_TEXT_CHARS = 1 * 1024 * 1024
MAX_LINE_CHARS = 2000
MAX_EVIDENCE_CHARS = 80

UNSCANNABLE = "unscannable"
UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class Hit:
    """One detector observation.

    ``location`` is already formatted (``path:line`` or just the label).  ``severity`` is ``None``
    when the calling rule's default applies; detectors set it only to escalate (e.g. a private key
    is CRITICAL while other secrets are HIGH).
    """

    code: str
    location: str
    message: str
    evidence: str = ""
    severity: Severity | None = None


def format_location(label: str, line: int = 0) -> str:
    """``label:line`` for a positive line number, else just ``label``."""
    return f"{label}:{line}" if line > 0 else label


def mask_secret(value: str) -> str:
    """Keep at most the first four characters (``AKIA****``): recognisable, never usable."""
    keep = min(4, max(0, len(value) // 4))
    return f"{value[:keep]}****"


def cap_evidence(text: str, limit: int = MAX_EVIDENCE_CHARS) -> str:
    """Single-line, length-bounded evidence."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def limit_hit(text: str, label: str, *, check_lines: bool = True) -> Hit | None:
    """Return an ``unscannable`` hit if ``text`` exceeds the hard caps, else ``None``.

    ``check_lines`` is ``False`` for AST based detectors: the parser's cost does not depend on line
    length, only the regex-driven detectors need the per-line bound.
    """
    if not isinstance(text, str):
        return Hit(UNSCANNABLE, label, "content is not text and cannot be scanned")
    if len(text) > MAX_TEXT_CHARS:
        return Hit(UNSCANNABLE, label, f"content exceeds the {MAX_TEXT_CHARS}-character scan limit")
    if check_lines and _longest_line(text) > MAX_LINE_CHARS:
        return Hit(UNSCANNABLE, label, f"a line exceeds the {MAX_LINE_CHARS}-character scan limit")
    return None


def _longest_line(text: str) -> int:
    longest = 0
    start = 0
    while True:
        end = text.find("\n", start)
        if end == -1:
            return max(longest, len(text) - start)
        longest = max(longest, end - start)
        start = end + 1
