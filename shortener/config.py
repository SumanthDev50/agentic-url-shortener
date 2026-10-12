"""Configuration: a frozen ``Settings`` dataclass and ``load_settings(env)`` (NFR-07, NFR-10).

* Every setting comes from a ``SHORTENER_*`` environment variable (documented in
  ``.env.example``) and is validated at startup.  All problems are reported together.
* Error messages name the offending variable but **never echo its value**, so a mistyped
  secret cannot leak into logs.
* Unsafe combinations are refused (``Settings.__post_init__`` enforces them for every
  construction path, not just the environment loader): no API keys without an explicit
  ``SHORTENER_AUTH_DISABLED=true``; auth disabled only on a loopback bind; docs only with auth.
* API key secrets are reduced to SHA-256 digests immediately; ``Settings`` never holds them.
* Obviously fake placeholders (as shown in ``.env.example``) are rejected, so copying the
  example file verbatim cannot produce a running service with a known key.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any
from urllib.parse import urlsplit

from shortener.domain.errors import InvalidUrlError
from shortener.domain.validation import UrlPolicy, canonical_host
from shortener.services.auth import digest_api_key

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

ENV_PREFIX = "SHORTENER_"
ENV_VARIABLES: tuple[str, ...] = (
    "SHORTENER_HOST",
    "SHORTENER_PORT",
    "SHORTENER_BASE_URL",
    "SHORTENER_DATABASE_PATH",
    "SHORTENER_API_KEYS",
    "SHORTENER_AUTH_DISABLED",
    "SHORTENER_ENABLE_DOCS",
    "SHORTENER_ANALYTICS_SECRET",
    "SHORTENER_CLICK_RETENTION_DAYS",
    "SHORTENER_TRUSTED_PROXIES",
    "SHORTENER_OWN_HOSTS",
    "SHORTENER_ALLOW_HOSTS",
    "SHORTENER_DENY_HOSTS",
    "SHORTENER_API_RATE_BURST",
    "SHORTENER_API_RATE_PER_MINUTE",
    "SHORTENER_REDIRECT_RATE_BURST",
    "SHORTENER_REDIRECT_RATE_PER_MINUTE",
    "SHORTENER_AUTH_FAILURE_RATE_BURST",
    "SHORTENER_AUTH_FAILURE_RATE_PER_MINUTE",
    "SHORTENER_AUTH_FAILURE_DELAY_MS",
    "SHORTENER_RATE_LIMIT_MAX_KEYS",
    "SHORTENER_CACHE_MAX_ENTRIES",
    "SHORTENER_CACHE_TTL_SECONDS",
    "SHORTENER_NEGATIVE_CACHE_TTL_SECONDS",
    "SHORTENER_CLICK_QUEUE_SIZE",
    "SHORTENER_CLICK_BATCH_SIZE",
    "SHORTENER_LOG_LEVEL",
)

MIN_SECRET_LENGTH = 32
MAX_SECRET_LENGTH = 256
MIN_DISTINCT_SECRET_CHARS = 12
MAX_BODY_BYTES = 16 * 1024
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_KEY_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,32}")
_SECRET_PATTERN = re.compile(rf"[A-Za-z0-9_-]{{{MIN_SECRET_LENGTH},{MAX_SECRET_LENGTH}}}")
_PLACEHOLDER_MARKERS = (
    "replace",
    "changeme",
    "change-me",
    "change_me",
    "placeholder",
    "example",
    "password",
    "dummy",
)
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """Invalid configuration.  The message never contains secret values."""


@dataclass(frozen=True, slots=True)
class ApiKeyDigest:
    key_id: str
    digest: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class RateLimit:
    """Token bucket parameters: ``burst`` tokens, refilled at ``per_minute`` per minute."""

    burst: int
    per_minute: int

    @property
    def refill_per_second(self) -> float:
        return self.per_minute / 60.0


def is_loopback_host(host: str) -> bool:
    """``True`` only for ``localhost`` and loopback IP literals (never for wildcard binds)."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    base_url: str = "http://127.0.0.1:8000"
    database_path: str = "data/shortener.db"
    api_keys: tuple[ApiKeyDigest, ...] = field(default=(), repr=False)
    auth_disabled: bool = False
    enable_docs: bool = False
    analytics_secret: bytes | None = field(default=None, repr=False)
    click_retention_days: int = 90
    trusted_proxies: tuple[IPNetwork, ...] = ()
    own_hosts: tuple[str, ...] = ()
    allow_hosts: tuple[str, ...] = ()
    deny_hosts: tuple[str, ...] = ()
    api_rate: RateLimit = RateLimit(burst=30, per_minute=120)
    redirect_rate: RateLimit = RateLimit(burst=120, per_minute=600)
    auth_failure_rate: RateLimit = RateLimit(burst=5, per_minute=10)
    auth_failure_delay_ms: int = 200
    rate_limit_max_keys: int = 100_000
    cache_max_entries: int = 10_000
    cache_ttl_seconds: int = 60
    negative_cache_ttl_seconds: int = 5
    click_queue_size: int = 10_000
    click_batch_size: int = 200
    log_level: str = "INFO"
    max_body_bytes: int = MAX_BODY_BYTES

    def __post_init__(self) -> None:
        """Refuse unsafe combinations regardless of how the object was constructed."""
        if not self.api_keys and not self.auth_disabled:
            raise ConfigError(
                "SHORTENER_API_KEYS is required unless SHORTENER_AUTH_DISABLED=true "
                "(the service refuses to start without authentication configured)."
            )
        if self.auth_disabled and self.api_keys:
            raise ConfigError(
                "SHORTENER_AUTH_DISABLED=true must not be combined with SHORTENER_API_KEYS."
            )
        if self.auth_disabled and not is_loopback_host(self.host):
            raise ConfigError(
                "SHORTENER_AUTH_DISABLED=true is only allowed when SHORTENER_HOST is a "
                "loopback address."
            )
        if self.auth_disabled and self.enable_docs:
            raise ConfigError("SHORTENER_ENABLE_DOCS=true requires authentication to be enabled.")
        if self.analytics_secret is not None and len(self.analytics_secret) < 16:
            raise ConfigError("SHORTENER_ANALYTICS_SECRET is too short.")

    @property
    def url_policy(self) -> UrlPolicy:
        return UrlPolicy(
            own_hosts=frozenset(self.own_hosts),
            allow_hosts=self.allow_hosts,
            deny_hosts=self.deny_hosts,
        )


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build validated ``Settings`` from ``env`` (default: ``os.environ``)."""
    reader = _EnvReader(os.environ if env is None else env)
    reader.reject_unknown_variables()
    base_url, base_host = _parse_base_url(reader)
    host = _parse_bind_host(reader)
    port = reader.integer("SHORTENER_PORT", _default("port"), 1, 65535)
    database_path = _parse_database_path(reader)
    api_keys = _parse_api_keys(reader)
    auth_disabled = reader.boolean("SHORTENER_AUTH_DISABLED", default=False)
    enable_docs = reader.boolean("SHORTENER_ENABLE_DOCS", default=False)
    analytics_secret = _parse_analytics_secret(reader)
    retention_days = reader.integer(
        "SHORTENER_CLICK_RETENTION_DAYS", _default("click_retention_days"), 1, 3650
    )
    trusted_proxies = _parse_trusted_proxies(reader)
    own_hosts = _parse_own_hosts(reader, base_host)
    allow_hosts = _parse_host_list(reader, "SHORTENER_ALLOW_HOSTS")
    deny_hosts = _parse_host_list(reader, "SHORTENER_DENY_HOSTS")
    api_rate = _parse_rate(reader, "SHORTENER_API_RATE", _default("api_rate"))
    redirect_rate = _parse_rate(reader, "SHORTENER_REDIRECT_RATE", _default("redirect_rate"))
    auth_failure_rate = _parse_rate(
        reader, "SHORTENER_AUTH_FAILURE_RATE", _default("auth_failure_rate")
    )
    auth_failure_delay_ms = reader.integer(
        "SHORTENER_AUTH_FAILURE_DELAY_MS", _default("auth_failure_delay_ms"), 0, 5000
    )
    rate_limit_max_keys = reader.integer(
        "SHORTENER_RATE_LIMIT_MAX_KEYS", _default("rate_limit_max_keys"), 100, 10_000_000
    )
    cache_max_entries = reader.integer(
        "SHORTENER_CACHE_MAX_ENTRIES", _default("cache_max_entries"), 1, 10_000_000
    )
    cache_ttl_seconds = reader.integer(
        "SHORTENER_CACHE_TTL_SECONDS", _default("cache_ttl_seconds"), 1, 86_400
    )
    negative_cache_ttl_seconds = reader.integer(
        "SHORTENER_NEGATIVE_CACHE_TTL_SECONDS", _default("negative_cache_ttl_seconds"), 1, 3600
    )
    click_queue_size = reader.integer(
        "SHORTENER_CLICK_QUEUE_SIZE", _default("click_queue_size"), 1, 10_000_000
    )
    click_batch_size = reader.integer(
        "SHORTENER_CLICK_BATCH_SIZE", _default("click_batch_size"), 1, 10_000
    )
    log_level = _parse_log_level(reader)
    reader.raise_if_errors()
    return Settings(
        host=host,
        port=port,
        base_url=base_url,
        database_path=database_path,
        api_keys=api_keys,
        auth_disabled=auth_disabled,
        enable_docs=enable_docs,
        analytics_secret=analytics_secret,
        click_retention_days=retention_days,
        trusted_proxies=trusted_proxies,
        own_hosts=own_hosts,
        allow_hosts=allow_hosts,
        deny_hosts=deny_hosts,
        api_rate=api_rate,
        redirect_rate=redirect_rate,
        auth_failure_rate=auth_failure_rate,
        auth_failure_delay_ms=auth_failure_delay_ms,
        rate_limit_max_keys=rate_limit_max_keys,
        cache_max_entries=cache_max_entries,
        cache_ttl_seconds=cache_ttl_seconds,
        negative_cache_ttl_seconds=negative_cache_ttl_seconds,
        click_queue_size=click_queue_size,
        click_batch_size=click_batch_size,
        log_level=log_level,
    )


_FIELD_DEFAULTS: dict[str, Any] = {item.name: item.default for item in fields(Settings)}


def _default(name: str) -> Any:
    """The dataclass default of a ``Settings`` field (single source of truth for defaults)."""
    return _FIELD_DEFAULTS[name]


class _EnvReader:
    """Typed access to environment variables that accumulates every problem found."""

    def __init__(self, env: Mapping[str, str]) -> None:
        self._env = env
        self._errors: list[str] = []

    def error(self, message: str) -> None:
        self._errors.append(message)

    def reject_unknown_variables(self) -> None:
        """A typo such as ``SHORTENER_AUTH_DISABLE`` must not be silently ignored."""
        for name in sorted(self._env):
            if name.upper().startswith(ENV_PREFIX) and name not in ENV_VARIABLES:
                self.error(f"{name}: unknown configuration variable.")

    def raise_if_errors(self) -> None:
        if self._errors:
            raise ConfigError("Invalid configuration:\n- " + "\n- ".join(self._errors))

    def text(self, name: str, default: str = "") -> str:
        value = self._env.get(name)
        return value.strip() if value is not None and value.strip() else default

    def integer(self, name: str, default: int, minimum: int, maximum: int) -> int:
        raw = self.text(name)
        if not raw:
            return default
        if raw.isascii() and raw.isdigit() and minimum <= int(raw) <= maximum:
            return int(raw)
        self.error(f"{name}: must be an integer between {minimum} and {maximum}.")
        return default

    def boolean(self, name: str, *, default: bool) -> bool:
        raw = self.text(name).lower()
        if not raw:
            return default
        if raw in _TRUTHY:
            return True
        if raw in _FALSY:
            return False
        self.error(f"{name}: must be true or false.")
        return default

    def items(self, name: str) -> list[str]:
        return [item.strip() for item in self.text(name).split(",") if item.strip()]


def _parse_bind_host(reader: _EnvReader) -> str:
    host = reader.text("SHORTENER_HOST", "127.0.0.1")
    if host.lower() == "localhost":
        return "localhost"
    try:
        ipaddress.ip_address(host)
    except ValueError:
        reader.error("SHORTENER_HOST: must be an IP address or 'localhost'.")
        return "127.0.0.1"
    return host


def _parse_base_url(reader: _EnvReader) -> tuple[str, str]:
    """Returns ``(normalised_base_url, canonical_host)`` (defaults on error)."""
    fallback = ("http://127.0.0.1:8000", "127.0.0.1")
    raw = reader.text("SHORTENER_BASE_URL", fallback[0])
    try:
        parts = urlsplit(raw)
        port = parts.port
        hostname = parts.hostname
    except ValueError:
        reader.error("SHORTENER_BASE_URL: must be an absolute http(s) URL.")
        return fallback
    valid = (
        parts.scheme in ("http", "https")
        and bool(hostname)
        and "@" not in parts.netloc
        and parts.path in ("", "/")
        and not parts.query
        and not parts.fragment
    )
    if not valid or hostname is None:
        reader.error(
            "SHORTENER_BASE_URL: must be an absolute http(s) URL without credentials, "
            "path, query or fragment."
        )
        return fallback
    try:
        host = canonical_host(hostname)
    except InvalidUrlError:
        reader.error("SHORTENER_BASE_URL: the host is not a valid host name or address.")
        return fallback
    authority = host if port is None else f"{host}:{port}"
    return f"{parts.scheme}://{authority}", host


def _parse_database_path(reader: _EnvReader) -> str:
    path = reader.text("SHORTENER_DATABASE_PATH", "data/shortener.db")
    if path.startswith(":memory:") or "\x00" in path:
        reader.error("SHORTENER_DATABASE_PATH: must be a file path.")
        return "data/shortener.db"
    return path


def looks_like_placeholder(value: str) -> bool:
    """Heuristic for obviously fake secrets such as the ones in ``.env.example``."""
    lowered = value.lower()
    return (
        any(marker in lowered for marker in _PLACEHOLDER_MARKERS)
        or len(set(value)) < MIN_DISTINCT_SECRET_CHARS
    )


def _secret_problem(secret: str) -> str | None:
    if _SECRET_PATTERN.fullmatch(secret) is None:
        return (
            f"must be {MIN_SECRET_LENGTH}-{MAX_SECRET_LENGTH} characters from A-Z, a-z, 0-9, "
            "underscore and hyphen (generate one with secrets.token_urlsafe(32))"
        )
    if looks_like_placeholder(secret):
        return "looks like a placeholder; generate a real one with secrets.token_urlsafe(32)"
    return None


def _parse_api_keys(reader: _EnvReader) -> tuple[ApiKeyDigest, ...]:
    keys: list[ApiKeyDigest] = []
    for position, entry in enumerate(reader.items("SHORTENER_API_KEYS"), start=1):
        key_id, separator, secret = entry.partition(":")
        label = f"SHORTENER_API_KEYS entry {position}"
        if not separator or _KEY_ID_PATTERN.fullmatch(key_id) is None:
            reader.error(f"{label}: expected 'id:secret' with an id of 1-32 [A-Za-z0-9_.-].")
            continue
        problem = _secret_problem(secret)
        if problem is not None:
            reader.error(f"{label}: the secret {problem}.")
            continue
        keys.append(ApiKeyDigest(key_id=key_id, digest=digest_api_key(secret)))
    _reject_duplicate_keys(reader, keys)
    return tuple(keys)


def _reject_duplicate_keys(reader: _EnvReader, keys: list[ApiKeyDigest]) -> None:
    if len({key.key_id for key in keys}) != len(keys):
        reader.error("SHORTENER_API_KEYS: key ids must be unique.")
    if len({key.digest for key in keys}) != len(keys):
        reader.error("SHORTENER_API_KEYS: secrets must be unique.")


def _parse_analytics_secret(reader: _EnvReader) -> bytes | None:
    secret = reader.text("SHORTENER_ANALYTICS_SECRET")
    if not secret:
        return None
    problem = _secret_problem(secret)
    if problem is not None:
        reader.error(f"SHORTENER_ANALYTICS_SECRET: the value {problem}.")
        return None
    return secret.encode("ascii")


def _parse_trusted_proxies(reader: _EnvReader) -> tuple[IPNetwork, ...]:
    networks: list[IPNetwork] = []
    for position, entry in enumerate(reader.items("SHORTENER_TRUSTED_PROXIES"), start=1):
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            reader.error(f"SHORTENER_TRUSTED_PROXIES entry {position}: not an IP or CIDR network.")
    return tuple(networks)


def _parse_host_list(reader: _EnvReader, name: str) -> tuple[str, ...]:
    hosts: list[str] = []
    for position, entry in enumerate(reader.items(name), start=1):
        try:
            hosts.append(canonical_host(entry))
        except InvalidUrlError:
            reader.error(f"{name} entry {position}: not a valid host name or address.")
    return tuple(dict.fromkeys(hosts))


def _parse_own_hosts(reader: _EnvReader, base_host: str) -> tuple[str, ...]:
    extra = _parse_host_list(reader, "SHORTENER_OWN_HOSTS")
    return tuple(dict.fromkeys((base_host, *extra)))


def _parse_rate(reader: _EnvReader, prefix: str, default: RateLimit) -> RateLimit:
    """Reads ``<prefix>_BURST`` and ``<prefix>_PER_MINUTE``."""
    burst = reader.integer(f"{prefix}_BURST", default.burst, 1, 100_000)
    per_minute = reader.integer(f"{prefix}_PER_MINUTE", default.per_minute, 1, 6_000_000)
    return RateLimit(burst=burst, per_minute=per_minute)


def _parse_log_level(reader: _EnvReader) -> str:
    level = reader.text("SHORTENER_LOG_LEVEL", "INFO").upper()
    if level not in LOG_LEVELS:
        reader.error(f"SHORTENER_LOG_LEVEL: must be one of {', '.join(LOG_LEVELS)}.")
        return "INFO"
    return level
