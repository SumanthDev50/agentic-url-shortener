"""Input validation: destination URLs (SSRF hardening), aliases, expiry, idempotency keys.

Everything here is a pure function (no I/O, in particular no DNS resolution).  Consequently a
host *name* that merely resolves to a private address cannot be detected; the redirector never
fetches destinations itself, and the limitation is documented in ``docs/product-notes.md``.

The central design rule for URLs: parse the stripped input once, validate the *parsed*
components, and return a canonical re-serialisation built from those components.  The raw input
is never stored or redirected to, so parser differentials between this module and a browser
cannot smuggle a different target through (see ``docs/product-spec.md`` section 6).
"""

from __future__ import annotations

import ipaddress
import re
import string
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import SplitResult, urlsplit

from shortener.domain.errors import (
    InvalidAliasError,
    InvalidExpiryError,
    InvalidIdempotencyKeyError,
    InvalidUrlError,
)

MAX_URL_LENGTH = 2048
MAX_HOST_LENGTH = 253
MIN_TTL_SECONDS = 60
MAX_TTL_SECONDS = 157_680_000  # five years of 365 days
MAX_EXPIRY_SPAN = timedelta(seconds=MAX_TTL_SECONDS)

ALIAS_PATTERN = re.compile(r"[A-Za-z0-9_-]{3,32}")
IDEMPOTENCY_KEY_PATTERN = re.compile(r"[\x21-\x7e]{1,128}")
REDIRECT_TARGET_PATTERN = re.compile(r"[\x21-\x7e]{1,2048}")

#: Case-insensitively rejected aliases (they collide with routes or well-known files).
RESERVED_ALIASES = frozenset(
    {
        "api",
        "admin",
        "docs",
        "redoc",
        "openapi.json",
        "healthz",
        "readyz",
        "metrics",
        "static",
        "favicon.ico",
        "robots.txt",
    }
)

#: Internal-only DNS suffixes that are never acceptable destinations.
BLOCKED_HOST_SUFFIXES = (
    "localhost",
    "local",
    "internal",
    "localdomain",
    "home.arpa",
    "lan",
    "intranet",
)

_DEFAULT_PORTS = {"http": 80, "https": 443}
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Zl", "Zp", "Zs"})
_LABEL_PATTERN = re.compile(r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?")
_NUMERIC_LABEL_PATTERN = re.compile(r"0[xX][0-9a-fA-F]*|[0-9]+")
_IPV4_PART_PATTERNS = (
    (re.compile(r"0[xX][0-9a-fA-F]+"), 16),
    (re.compile(r"0[0-7]+"), 8),
    (re.compile(r"(?:0|[1-9][0-9]*)"), 10),
)
_CJK_SCRIPTS = frozenset({"CJK", "HIRAGANA", "KATAKANA", "HANGUL", "BOPOMOFO"})

_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_PATH_SAFE = _UNRESERVED | frozenset("!$&'()*+,;=:@/")
_QUERY_SAFE = _PATH_SAFE | frozenset("?")
_HEX_DIGITS = frozenset(string.hexdigits)

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True, slots=True)
class UrlPolicy:
    """Deployment-specific destination rules (all hosts already canonical).

    ``allow_hosts`` (when non-empty) restricts destinations to those domains and their
    subdomains; ``deny_hosts`` adds blocked domains.  Neither can relax the built-in blocking
    of non-public addresses and internal names.
    """

    own_hosts: frozenset[str] = frozenset()
    allow_hosts: tuple[str, ...] = ()
    deny_hosts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedHost:
    text: str  # canonical authority text (IPv6 in brackets)
    ip: IPAddress | None


# --------------------------------------------------------------------------- URLs


def validate_destination_url(raw: object, policy: UrlPolicy | None = None) -> str:
    """Validate a destination and return its canonical, ASCII-only serialisation."""
    policy = policy or UrlPolicy()
    text = _screen(raw)
    parts = _split(text)
    host = parse_host(_hostname_of(parts))
    _ensure_public(host)
    _enforce_policy(host, policy)
    canonical = _serialise(parts, host, _port_of(parts))
    if len(canonical) > MAX_URL_LENGTH:
        raise InvalidUrlError("The destination URL is too long once normalised.")
    return canonical


def canonical_host(raw: str) -> str:
    """Canonical host text for configuration values (no public-address enforcement)."""
    return parse_host(raw).text


def parse_host(raw: str) -> ParsedHost:
    """Parse a hostname or IP literal (any IPv4 notation) into its canonical form."""
    if ":" in raw:
        return _ipv6_host(raw)
    name = _to_ascii_hostname(raw.removesuffix("."))
    ip = _numeric_host_as_ip(name)
    if ip is not None:
        return ParsedHost(text=str(ip), ip=ip)
    return ParsedHost(text=name, ip=None)


def _screen(raw: object) -> str:
    """Reject non-strings, empty/oversized input and any control/whitespace/format character."""
    if not isinstance(raw, str):
        raise InvalidUrlError("The destination URL must be a string.")
    text = raw.strip(" ")
    if not text:
        raise InvalidUrlError("The destination URL is required.")
    if len(text) > MAX_URL_LENGTH:
        raise InvalidUrlError("The destination URL is too long.")
    if any(_is_forbidden_char(ch) for ch in text):
        raise InvalidUrlError(
            "The destination URL contains whitespace, control, backslash or format characters."
        )
    return text


def _is_forbidden_char(ch: str) -> bool:
    return ch == "\\" or ch.isspace() or unicodedata.category(ch) in _FORBIDDEN_CATEGORIES


def _split(text: str) -> SplitResult:
    try:
        parts = urlsplit(text)
    except ValueError:
        raise InvalidUrlError("The destination URL is malformed.") from None
    if parts.scheme not in _DEFAULT_PORTS:
        raise InvalidUrlError("Only http and https destination URLs are allowed.")
    if not parts.netloc:
        raise InvalidUrlError("The destination URL must contain a host.")
    if "@" in parts.netloc:
        raise InvalidUrlError("The destination URL must not contain credentials.")
    if ("[" in parts.netloc or "]" in parts.netloc) and not parts.netloc.startswith("["):
        raise InvalidUrlError("The destination URL host is malformed.")
    return parts


def _hostname_of(parts: SplitResult) -> str:
    hostname = parts.hostname
    if not hostname:
        raise InvalidUrlError("The destination URL must contain a host.")
    return hostname


def _port_of(parts: SplitResult) -> int | None:
    try:
        port = parts.port
    except ValueError:
        raise InvalidUrlError("The destination URL port is invalid.") from None
    if port == 0:
        raise InvalidUrlError("The destination URL port is invalid.")
    return port


def _ipv6_host(raw: str) -> ParsedHost:
    if "%" in raw:
        raise InvalidUrlError("IPv6 zone identifiers are not allowed.")
    try:
        ip = ipaddress.IPv6Address(raw)
    except ValueError:
        raise InvalidUrlError("The destination URL host is malformed.") from None
    mapped = ip.ipv4_mapped
    if mapped is not None:
        return ParsedHost(text=str(mapped), ip=mapped)
    return ParsedHost(text=f"[{ip}]", ip=ip)


def _to_ascii_hostname(name: str) -> str:
    """IDNA/punycode conversion plus label, length and mixed-script validation."""
    try:
        ascii_name = name.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise InvalidUrlError("The destination URL host is not a valid domain name.") from None
    if not ascii_name or len(ascii_name) > MAX_HOST_LENGTH:
        raise InvalidUrlError("The destination URL host is not a valid domain name.")
    if not all(_LABEL_PATTERN.fullmatch(label) for label in ascii_name.split(".")):
        raise InvalidUrlError("The destination URL host is not a valid domain name.")
    _ensure_single_script(ascii_name)
    return ascii_name


def _ensure_single_script(ascii_name: str) -> None:
    """Reject labels mixing scripts (homograph spoofing); CJK scripts form one group."""
    if "xn--" not in ascii_name:
        return
    try:
        unicode_name = ascii_name.encode("ascii").decode("idna")
    except UnicodeError:
        raise InvalidUrlError("The destination URL host is not a valid domain name.") from None
    for label in unicode_name.split("."):
        scripts = {script for ch in label if (script := _script_of(ch)) is not None}
        if len(scripts) > 1:
            raise InvalidUrlError("The destination URL host mixes scripts.")


def _script_of(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    word = unicodedata.name(ch, "UNKNOWN").split(" ", 1)[0]
    return "CJK" if word in _CJK_SCRIPTS else word


def _numeric_host_as_ip(name: str) -> ipaddress.IPv4Address | None:
    """Interpret legacy IPv4 notations (``127.1``, ``2130706433``, ``0x7f.1``, octal).

    Following the WHATWG rule, a host whose last label is numeric *must* be a valid IPv4
    address; otherwise ``foo.123`` style names would be ambiguous between parsers.
    """
    last_label = name.rsplit(".", 1)[-1]
    if _NUMERIC_LABEL_PATTERN.fullmatch(last_label) is None:
        return None
    labels = name.split(".")
    if len(labels) > 4:
        raise InvalidUrlError("The destination URL host is not a valid IPv4 address.")
    numbers = [_parse_ipv4_part(label) for label in labels]
    *leading, last = numbers
    if any(value > 255 for value in leading) or last >= 256 ** (5 - len(numbers)):
        raise InvalidUrlError("The destination URL host is not a valid IPv4 address.")
    value = last
    for position, part in enumerate(leading):
        value += part << (8 * (3 - position))
    return ipaddress.IPv4Address(value)


def _parse_ipv4_part(label: str) -> int:
    for pattern, base in _IPV4_PART_PATTERNS:
        if pattern.fullmatch(label):
            digits = label[2:] if base == 16 else label
            return int(digits, base)
    raise InvalidUrlError("The destination URL host is not a valid IPv4 address.")


def _ensure_public(host: ParsedHost) -> None:
    if host.ip is not None:
        if _is_blocked_ip(host.ip):
            raise InvalidUrlError("The destination host is not allowed.")
        return
    if "." not in host.text or _has_blocked_suffix(host.text):
        raise InvalidUrlError("The destination host is not allowed.")


def _is_blocked_ip(ip: IPAddress) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    )


def _has_blocked_suffix(name: str) -> bool:
    return any(_domain_matches(name, suffix) for suffix in BLOCKED_HOST_SUFFIXES)


def _domain_matches(name: str, domain: str) -> bool:
    return name == domain or name.endswith("." + domain)


def _enforce_policy(host: ParsedHost, policy: UrlPolicy) -> None:
    if host.text in policy.own_hosts:
        raise InvalidUrlError("The destination points back at this service.")
    matches = _domain_matches if host.ip is None else _same_host
    if any(matches(host.text, entry) for entry in policy.deny_hosts):
        raise InvalidUrlError("The destination host is not allowed.")
    if policy.allow_hosts and not any(matches(host.text, entry) for entry in policy.allow_hosts):
        raise InvalidUrlError("The destination host is not allowed.")


def _same_host(name: str, entry: str) -> bool:
    return name == entry


def _serialise(parts: SplitResult, host: ParsedHost, port: int | None) -> str:
    authority = host.text
    if port is not None and port != _DEFAULT_PORTS[parts.scheme]:
        authority = f"{authority}:{port}"
    path = _percent_encode(parts.path, _PATH_SAFE) or "/"
    query = f"?{_percent_encode(parts.query, _QUERY_SAFE)}" if parts.query else ""
    fragment = f"#{_percent_encode(parts.fragment, _QUERY_SAFE)}" if parts.fragment else ""
    return f"{parts.scheme}://{authority}{path}{query}{fragment}"


def _percent_encode(value: str, safe: frozenset[str]) -> str:
    """Percent-encode everything outside ``safe``; keep valid ``%XX`` escapes, fix stray ``%``."""
    encoded: list[str] = []
    for index, ch in enumerate(value):
        if ch in safe or (ch == "%" and _is_hex_pair(value[index + 1 : index + 3])):
            encoded.append(ch)
        else:
            encoded.extend(f"%{byte:02X}" for byte in ch.encode("utf-8"))
    return "".join(encoded)


def _is_hex_pair(pair: str) -> bool:
    return len(pair) == 2 and all(ch in _HEX_DIGITS for ch in pair)


def is_safe_redirect_target(url: str) -> bool:
    """Defence in depth for stored data: printable ASCII only, so a tampered row can never
    inject header lines (response splitting) through the ``Location`` header."""
    return REDIRECT_TARGET_PATTERN.fullmatch(url) is not None


# --------------------------------------------------------------------------- aliases / codes


def validate_alias(alias: object) -> str:
    """Return the alias unchanged if it satisfies the alias rules (case-sensitive)."""
    if not isinstance(alias, str) or ALIAS_PATTERN.fullmatch(alias) is None:
        raise InvalidAliasError(
            "The alias must be 3-32 characters from A-Z, a-z, 0-9, underscore and hyphen."
        )
    if alias.lower() in RESERVED_ALIASES:
        raise InvalidAliasError("The alias is reserved.")
    return alias


def is_valid_code_format(code: str) -> bool:
    """Cheap pre-filter so junk paths never reach the cache or the database."""
    return ALIAS_PATTERN.fullmatch(code) is not None


# --------------------------------------------------------------------------- expiry


def resolve_expiry(
    *, now: datetime, expires_at: datetime | None, ttl_seconds: int | None
) -> datetime | None:
    """Turn the mutually exclusive ``expires_at`` / ``ttl_seconds`` inputs into an instant."""
    if expires_at is not None and ttl_seconds is not None:
        raise InvalidExpiryError("Provide either expires_at or ttl_seconds, not both.")
    if ttl_seconds is not None:
        return now + timedelta(seconds=_checked_ttl(ttl_seconds))
    if expires_at is None:
        return None
    return _checked_expires_at(now, expires_at)


def _checked_ttl(ttl_seconds: int) -> int:
    if isinstance(ttl_seconds, bool) or not MIN_TTL_SECONDS <= ttl_seconds <= MAX_TTL_SECONDS:
        raise InvalidExpiryError(
            f"ttl_seconds must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}."
        )
    return ttl_seconds


def _checked_expires_at(now: datetime, expires_at: datetime) -> datetime:
    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        raise InvalidExpiryError("expires_at must include a UTC offset.")
    try:
        expires_utc = expires_at.astimezone(UTC)
    except (OverflowError, ValueError):
        raise InvalidExpiryError("expires_at is out of range.") from None
    if expires_utc <= now:
        raise InvalidExpiryError("expires_at must be in the future.")
    if expires_utc - now > MAX_EXPIRY_SPAN:
        raise InvalidExpiryError("expires_at must be within five years.")
    return expires_utc


# --------------------------------------------------------------------------- idempotency


def validate_idempotency_key(key: object) -> str:
    """Printable ASCII, 1-128 characters (header-safe, log-safe)."""
    if not isinstance(key, str) or IDEMPOTENCY_KEY_PATTERN.fullmatch(key) is None:
        raise InvalidIdempotencyKeyError(
            "The Idempotency-Key must be 1-128 printable ASCII characters without spaces."
        )
    return key
