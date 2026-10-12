"""Privacy-preserving reductions for analytics (NFR-08).

* IPs are truncated (IPv4 /24, IPv6 /48) *before* hashing and are never stored.
* ``visitor_id = HMAC-SHA256(daily_salt, truncated_ip || "|" || ua_family)[:16]`` where
  ``daily_salt = HMAC(server_secret, utc_date)`` is derived on the fly and never stored, so ids
  are unlinkable across days and cannot be reversed without the server secret.
* Only the referrer scheme + host survive; paths and queries (which may carry tokens) do not.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
from datetime import date
from urllib.parse import urlsplit

UNKNOWN_IP = "unknown"
VISITOR_ID_LENGTH = 16
MAX_REFERER_INPUT_LENGTH = 2048
MAX_REFERRER_HOST_LENGTH = 255

_REFERRER_HOST_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?")
_IPV4_PREFIX = 24
_IPV6_PREFIX = 48


def truncate_ip(ip_text: str | None) -> str:
    """Zero the host part of an IP (IPv4-mapped IPv6 is treated as IPv4)."""
    if not ip_text:
        return UNKNOWN_IP
    try:
        ip = ipaddress.ip_address(ip_text.strip())
    except ValueError:
        return UNKNOWN_IP
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    prefix = _IPV4_PREFIX if ip.version == 4 else _IPV6_PREFIX
    network = ipaddress.ip_network((ip, prefix), strict=False)
    return str(network.network_address)


def daily_salt(server_secret: bytes, day: date) -> bytes:
    """Per-day salt derived from the long-lived server secret."""
    return hmac.new(server_secret, day.isoformat().encode("ascii"), hashlib.sha256).digest()


def visitor_id(server_secret: bytes, day: date, client_ip: str | None, ua_family: str) -> str:
    """Stable-within-a-day, unlinkable-across-days visitor identifier (hex, 16 chars)."""
    material = f"{truncate_ip(client_ip)}|{ua_family}".encode()
    digest = hmac.new(daily_salt(server_secret, day), material, hashlib.sha256).hexdigest()
    return digest[:VISITOR_ID_LENGTH]


def referrer_origin(referer: str | None) -> str | None:
    """``scheme://host`` of a Referer header, or ``None`` if absent/unusable."""
    if not referer or len(referer) > MAX_REFERER_INPUT_LENGTH:
        return None
    try:
        parts = urlsplit(referer.strip())
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not host:
        return None
    if len(host) > MAX_REFERRER_HOST_LENGTH or _REFERRER_HOST_PATTERN.fullmatch(host) is None:
        return None
    return f"{parts.scheme}://{host}"
