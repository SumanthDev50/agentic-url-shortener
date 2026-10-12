"""Coarse user-agent classification (browser family + device class).

Only these fixed labels are ever stored, so an attacker-chosen ``User-Agent`` can neither
bloat the database nor smuggle personal data into it (NFR-08).  The raw header is truncated
before matching so hostile input cannot trigger expensive scanning.
"""

from __future__ import annotations

MAX_USER_AGENT_LENGTH = 512

UNKNOWN = "Unknown"
OTHER = "Other"

_BOT_MARKERS = (
    "bot",
    "spider",
    "crawler",
    "slurp",
    "curl/",
    "wget/",
    "python-requests",
    "python-urllib",
    "httpx",
    "go-http-client",
    "okhttp",
    "facebookexternalhit",
    "headlesschrome",
)
#: Order matters: the first marker found wins (Edge and Opera also say "Chrome").
_BROWSER_MARKERS = (
    ("Edge", ("edg/", "edga/", "edgios/", "edge/")),
    ("Opera", ("opr/", "opera")),
    ("Samsung Internet", ("samsungbrowser",)),
    ("Firefox", ("firefox/", "fxios/")),
    ("Chrome", ("chrome/", "crios/", "chromium/")),
    ("Safari", ("safari/",)),
    ("IE", ("msie ", "trident/")),
)
_TABLET_MARKERS = ("ipad", "tablet", "kindle", "silk/")
_MOBILE_MARKERS = ("mobile", "iphone", "ipod", "android", "windows phone")


def classify_user_agent(user_agent: str | None) -> tuple[str, str]:
    """Return ``(browser_family, device_class)`` for a raw ``User-Agent`` header."""
    if not user_agent or not user_agent.strip():
        return UNKNOWN, "unknown"
    text = user_agent[:MAX_USER_AGENT_LENGTH].lower()
    if any(marker in text for marker in _BOT_MARKERS):
        return "Bot", "bot"
    return _browser_family(text), _device_class(text)


def _browser_family(text: str) -> str:
    for family, markers in _BROWSER_MARKERS:
        if any(marker in text for marker in markers):
            return family
    return OTHER


def _device_class(text: str) -> str:
    if any(marker in text for marker in _TABLET_MARKERS):
        return "tablet"
    if "android" in text and "mobile" not in text:
        return "tablet"
    if any(marker in text for marker in _MOBILE_MARKERS):
        return "mobile"
    return "desktop"
