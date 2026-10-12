"""Client address resolution with a trust boundary for ``X-Forwarded-For``.

The peer address of the TCP connection is authoritative.  ``X-Forwarded-For`` is consulted only
when the peer is one of the configured trusted proxies, and then it is read from the right:
the first hop that is *not* a trusted proxy is the client.  Anything to its left is
client-controlled and ignored, so a spoofed header cannot choose the rate-limit identity.
The result is used only as a throttling key and as input to the hashed visitor id; it is never
stored or logged.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence

UNKNOWN_CLIENT = "unknown"
MAX_FORWARDED_HOPS = 20

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


class ClientIpResolver:
    def __init__(self, trusted_proxies: Sequence[IPNetwork] = ()) -> None:
        self._trusted = tuple(trusted_proxies)

    def resolve(self, peer: str | None, forwarded_for: str | None = None) -> str:
        """Best-effort client address (normalised text) or ``"unknown"``."""
        current = _parse_ip(peer)
        if current is None:
            return UNKNOWN_CLIENT
        if forwarded_for and self._is_trusted(current):
            hops = forwarded_for.split(",")[-MAX_FORWARDED_HOPS:]
            for hop in reversed(hops):
                candidate = _parse_ip(hop)
                if candidate is None:
                    break  # malformed hop: stop trusting the chain here
                current = candidate
                if not self._is_trusted(candidate):
                    break
        return str(current)

    def _is_trusted(self, ip: IPAddress) -> bool:
        return any(ip in network for network in self._trusted if network.version == ip.version)


def _parse_ip(text: str | None) -> IPAddress | None:
    if not text:
        return None
    try:
        ip = ipaddress.ip_address(text.strip())
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip
