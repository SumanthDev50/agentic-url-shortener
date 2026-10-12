"""Configuration loading: validation, secure defaults, placeholders, refusal to start."""

from __future__ import annotations

import ipaddress
import re
import os
import runpy
import sys
from pathlib import Path

import pytest

from shortener import __main__ as entrypoint
from shortener.config import (
    ENV_VARIABLES,
    ConfigError,
    RateLimit,
    Settings,
    is_loopback_host,
    load_settings,
    looks_like_placeholder,
)
from shortener.services.auth import digest_api_key
from tests.shortener.support import new_secret

ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = ROOT / ".env.example"
NOTES = ROOT / "docs" / "product-notes.md"


def env(**values: str) -> dict[str, str]:
    """Environment with a valid key unless overridden (keys: ``SHORTENER_`` prefix omitted)."""
    base = {"SHORTENER_API_KEYS": f"ci:{new_secret()}"}
    base.update({f"SHORTENER_{name}": value for name, value in values.items()})
    return base


def error_text(variables: dict[str, str]) -> str:
    with pytest.raises(ConfigError) as excinfo:
        load_settings(variables)
    return str(excinfo.value)


# ----------------------------------------------------------------------------- defaults


def test_defaults_are_secure_and_loopback() -> None:
    settings = load_settings(env())
    assert settings.host == "127.0.0.1"
    assert settings.auth_disabled is False
    assert settings.enable_docs is False
    assert settings.base_url == "http://127.0.0.1:8000"
    assert settings.own_hosts == ("127.0.0.1",)
    assert settings.click_retention_days == 90
    assert settings.max_body_bytes == 16 * 1024
    assert settings.api_rate == RateLimit(30, 120)
    assert settings.analytics_secret is None
    assert len(settings.api_keys) == 1


def test_all_values_can_be_overridden() -> None:
    secret = new_secret()
    analytics = new_secret()
    settings = load_settings(
        {
            "SHORTENER_API_KEYS": f"alice:{secret}, bob.2:{new_secret()}",
            "SHORTENER_HOST": "0.0.0.0",  # noqa: S104 - bind address value under test
            "SHORTENER_PORT": "9000",
            "SHORTENER_BASE_URL": "https://Sho.Example.com/",
            "SHORTENER_DATABASE_PATH": "/var/lib/shortener/db.sqlite",
            "SHORTENER_ENABLE_DOCS": "yes",
            "SHORTENER_ANALYTICS_SECRET": analytics,
            "SHORTENER_CLICK_RETENTION_DAYS": "30",
            "SHORTENER_TRUSTED_PROXIES": "10.0.0.0/8, 192.0.2.1, fd00::/8",
            "SHORTENER_OWN_HOSTS": "alt.example.com, Short.Example.NET",
            "SHORTENER_ALLOW_HOSTS": "partner.example",
            "SHORTENER_DENY_HOSTS": "evil.example, 203.0.113.9",
            "SHORTENER_API_RATE_BURST": "7",
            "SHORTENER_API_RATE_PER_MINUTE": "60",
            "SHORTENER_REDIRECT_RATE_BURST": "9",
            "SHORTENER_REDIRECT_RATE_PER_MINUTE": "90",
            "SHORTENER_AUTH_FAILURE_RATE_BURST": "2",
            "SHORTENER_AUTH_FAILURE_RATE_PER_MINUTE": "4",
            "SHORTENER_AUTH_FAILURE_DELAY_MS": "0",
            "SHORTENER_RATE_LIMIT_MAX_KEYS": "500",
            "SHORTENER_CACHE_MAX_ENTRIES": "50",
            "SHORTENER_CACHE_TTL_SECONDS": "30",
            "SHORTENER_NEGATIVE_CACHE_TTL_SECONDS": "2",
            "SHORTENER_CLICK_QUEUE_SIZE": "100",
            "SHORTENER_CLICK_BATCH_SIZE": "10",
            "SHORTENER_LOG_LEVEL": "debug",
        }
    )
    assert (settings.host, settings.port) == ("0.0.0.0", 9000)  # noqa: S104
    assert settings.base_url == "https://sho.example.com"
    assert settings.database_path == "/var/lib/shortener/db.sqlite"
    assert settings.enable_docs is True
    assert settings.analytics_secret == analytics.encode()
    assert settings.click_retention_days == 30
    assert ipaddress.ip_network("10.0.0.0/8") in settings.trusted_proxies
    assert len(settings.trusted_proxies) == 3
    assert settings.own_hosts == ("sho.example.com", "alt.example.com", "short.example.net")
    assert settings.allow_hosts == ("partner.example",)
    assert settings.deny_hosts == ("evil.example", "203.0.113.9")
    assert settings.api_rate == RateLimit(7, 60)
    assert settings.api_rate.refill_per_second == 1.0
    assert settings.redirect_rate == RateLimit(9, 90)
    assert settings.auth_failure_rate == RateLimit(2, 4)
    assert settings.auth_failure_delay_ms == 0
    assert settings.rate_limit_max_keys == 500
    assert (settings.cache_max_entries, settings.cache_ttl_seconds) == (50, 30)
    assert settings.negative_cache_ttl_seconds == 2
    assert (settings.click_queue_size, settings.click_batch_size) == (100, 10)
    assert settings.log_level == "DEBUG"
    assert [key.key_id for key in settings.api_keys] == ["alice", "bob.2"]


def test_only_digests_of_keys_are_kept() -> None:
    secret = new_secret()
    settings = load_settings({"SHORTENER_API_KEYS": f"alice:{secret}"})
    assert settings.api_keys[0].digest == digest_api_key(secret)
    assert secret not in repr(settings)
    assert secret not in repr(settings.api_keys)


def test_analytics_secret_is_not_in_repr() -> None:
    analytics = new_secret()
    settings = load_settings(env(ANALYTICS_SECRET=analytics))
    assert analytics not in repr(settings)


# ----------------------------------------------------------------------------- refusals


def test_refuses_to_start_without_keys_unless_auth_is_disabled() -> None:
    assert "SHORTENER_API_KEYS is required" in error_text({})
    assert "SHORTENER_API_KEYS is required" in error_text({"SHORTENER_API_KEYS": "  "})
    settings = load_settings({"SHORTENER_AUTH_DISABLED": "true"})
    assert settings.auth_disabled is True
    assert settings.api_keys == ()


def test_auth_disabled_requires_a_loopback_bind() -> None:
    for host in ("0.0.0.0", "192.0.2.10", "::"):  # noqa: S104
        message = error_text({"SHORTENER_AUTH_DISABLED": "true", "SHORTENER_HOST": host})
        assert "loopback" in message
    for host in ("127.0.0.1", "::1", "localhost", "LOCALHOST", "127.5.5.5"):
        assert load_settings({"SHORTENER_AUTH_DISABLED": "true", "SHORTENER_HOST": host})


def test_auth_disabled_cannot_be_combined_with_keys_or_docs() -> None:
    both = env(AUTH_DISABLED="true")
    assert "must not be combined" in error_text(both)
    docs = {"SHORTENER_AUTH_DISABLED": "true", "SHORTENER_ENABLE_DOCS": "true"}
    assert "requires authentication" in error_text(docs)


def test_settings_enforce_invariants_for_direct_construction_too() -> None:
    with pytest.raises(ConfigError):
        Settings()  # no keys, auth not disabled
    with pytest.raises(ConfigError):
        Settings(auth_disabled=True, host="0.0.0.0")  # noqa: S104
    with pytest.raises(ConfigError):
        Settings(auth_disabled=True, analytics_secret=b"short")


def test_unknown_shortener_variables_are_rejected_by_name_only() -> None:
    message = error_text(env(AUTH_DISABLE="true"))
    assert "SHORTENER_AUTH_DISABLE" in message
    assert "unknown configuration variable" in message
    load_settings({**env(), "OTHER_TOOL_SETTING": "x", "PATH": "/bin"})  # other prefixes are fine


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PORT", "0"),
        ("PORT", "65536"),
        ("PORT", "eighty"),
        ("PORT", "-1"),
        ("PORT", "8_000"),
        ("HOST", "example.com"),
        ("HOST", "999.1.1.1"),
        ("BASE_URL", "ftp://example.com"),
        ("BASE_URL", "example.com"),
        ("BASE_URL", "https://user:pw@example.com"),
        ("BASE_URL", "https://example.com/path"),
        ("BASE_URL", "https://example.com/?q=1"),
        ("BASE_URL", "https://example.com/#frag"),
        ("BASE_URL", "https://exa mple.com"),
        ("BASE_URL", "https://example.com:notaport"),
        ("DATABASE_PATH", ":memory:"),
        ("DATABASE_PATH", "a\x00b"),
        ("AUTH_DISABLED", "maybe"),
        ("ENABLE_DOCS", "2"),
        ("CLICK_RETENTION_DAYS", "0"),
        ("CLICK_RETENTION_DAYS", "100000"),
        ("TRUSTED_PROXIES", "not-a-network"),
        ("TRUSTED_PROXIES", "10.0.0.0/33"),
        ("OWN_HOSTS", "exa mple.com"),
        ("ALLOW_HOSTS", "-bad-.example"),
        ("DENY_HOSTS", "a..b"),
        ("API_RATE_BURST", "0"),
        ("API_RATE_PER_MINUTE", "x"),
        ("AUTH_FAILURE_DELAY_MS", "999999"),
        ("RATE_LIMIT_MAX_KEYS", "5"),
        ("CACHE_TTL_SECONDS", "0"),
        ("LOG_LEVEL", "LOUD"),
    ],
)
def test_invalid_values_are_reported_by_variable_name_without_the_value(
    name: str, value: str
) -> None:
    variable = f"SHORTENER_{name}"
    message = error_text(env(**{name: value}))
    assert variable in message or "BASE_URL" in message
    if len(value) > 4 and value.isalnum():
        assert value not in message


def test_all_problems_are_reported_together() -> None:
    message = error_text(env(PORT="0", LOG_LEVEL="LOUD", CACHE_TTL_SECONDS="x"))
    for variable in ("SHORTENER_PORT", "SHORTENER_LOG_LEVEL", "SHORTENER_CACHE_TTL_SECONDS"):
        assert variable in message


# ----------------------------------------------------------------------------- secrets


@pytest.mark.parametrize(
    "entry",
    [
        "alice",  # no separator
        ":" + "a1B2" * 10,  # empty id
        "bad id:" + "a1B2" * 10,
        "x" * 33 + ":" + "a1B2" * 10,
        "alice:",  # empty secret
        "alice:short",
        "alice:" + "a1B2" * 80,  # too long
        "alice:" + "a1B2c" * 8 + "!",  # character outside the allowed alphabet
        "alice:" + "a1B2" * 9 + "==",
    ],
)
def test_malformed_api_key_entries_are_rejected_without_echoing_them(entry: str) -> None:
    message = error_text({"SHORTENER_API_KEYS": entry})
    assert "SHORTENER_API_KEYS entry 1" in message
    secret = entry.partition(":")[2]
    if len(secret) >= 8:
        assert secret not in message


@pytest.mark.parametrize(
    "secret",
    [
        "REPLACE_ME_with_output_of_secrets_token_urlsafe_32",
        "replace-me-" + "a1B2c3D4" * 4,
        "changeme" + "a1B2c3D4" * 4,
        "Password" + "a1B2c3D4" * 4,
        "example_example_example_example_",
        "your_placeholder_value_goes_here_xx",
        "a" * 40,
        "ab" * 20,
        "0123456789" * 4,
    ],
)
def test_placeholder_and_low_entropy_secrets_are_rejected(secret: str) -> None:
    assert looks_like_placeholder(secret)
    message = error_text({"SHORTENER_API_KEYS": f"ci:{secret}"})
    assert "SHORTENER_API_KEYS entry 1" in message
    assert secret not in message


def test_generated_secrets_are_never_mistaken_for_placeholders() -> None:
    assert not any(looks_like_placeholder(new_secret()) for _ in range(500))


def test_duplicate_ids_and_duplicate_secrets_are_rejected() -> None:
    secret = new_secret()
    assert "ids must be unique" in error_text(
        {"SHORTENER_API_KEYS": f"same:{secret},same:{new_secret()}"}
    )
    assert "secrets must be unique" in error_text({"SHORTENER_API_KEYS": f"a:{secret},b:{secret}"})


def test_bad_analytics_secret_is_rejected_without_echo() -> None:
    message = error_text(env(ANALYTICS_SECRET="REPLACE_ME_with_output_of_secrets_token_urlsafe_32"))
    assert "SHORTENER_ANALYTICS_SECRET" in message
    assert "REPLACE_ME" not in message
    assert "SHORTENER_ANALYTICS_SECRET" in error_text(env(ANALYTICS_SECRET="short"))


# ----------------------------------------------------------------------------- example file


def example_text() -> str:
    """The documented example environment (a fenced ``env`` block in docs/product-notes.md)."""
    notes = NOTES.read_text(encoding="utf-8")
    match = re.search(r"```env\n(.*?)```", notes, flags=re.DOTALL)
    assert match is not None, "docs/product-notes.md must contain the ```env example block"
    return match.group(1)


def parse_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in example_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, _, value = line.partition("=")
            values[name.strip()] = value.strip()
    return values


def test_example_documents_every_variable_exactly_once() -> None:
    text = example_text()
    assert set(parse_example()) == set(ENV_VARIABLES)
    for name in ENV_VARIABLES:
        assert len(re.findall(rf"^{name}=", text, flags=re.MULTILINE)) == 1


def test_example_is_ascii() -> None:
    assert example_text().isascii()


def test_example_placeholders_are_rejected_by_the_loader() -> None:
    message = error_text(parse_example())
    assert "SHORTENER_API_KEYS entry 1" in message
    assert "SHORTENER_ANALYTICS_SECRET" in message
    assert "REPLACE_ME" not in message


def test_example_is_valid_once_placeholders_are_replaced() -> None:
    values = parse_example()
    values["SHORTENER_API_KEYS"] = f"local-dev:{new_secret()}"
    values["SHORTENER_ANALYTICS_SECRET"] = new_secret()
    settings = load_settings(values)
    assert settings.auth_disabled is False and settings.host == "127.0.0.1"


def test_example_contains_no_real_looking_secrets() -> None:
    """Values are defaults or obvious placeholders; nothing that resembles a real secret."""
    for name, value in parse_example().items():
        if "REPLACE_ME" not in value:
            assert not re.search(r"[A-Za-z0-9_-]{32,}", value), name


def test_env_example_file_matches_the_documented_block_when_present() -> None:
    if not ENV_EXAMPLE.exists():
        pytest.skip(".env.example has not been created (see docs/product-notes.md)")
    on_disk = ENV_EXAMPLE.read_text(encoding="utf-8")  # universal newlines: CRLF -> LF
    assert on_disk == example_text()


# ----------------------------------------------------------------------------- helpers / entry point


def test_loopback_detection() -> None:
    assert all(is_loopback_host(h) for h in ("127.0.0.1", "::1", "localhost", "127.1.2.3"))
    assert not any(is_loopback_host(h) for h in ("0.0.0.0", "::", "10.0.0.1", "example.com", ""))  # noqa: S104


def test_load_settings_reads_the_process_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(os.environ):
        if name.startswith("SHORTENER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SHORTENER_AUTH_DISABLED", "true")
    assert load_settings().auth_disabled is True


def test_entrypoint_refuses_unsafe_configuration_with_exit_code_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith("SHORTENER_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(entrypoint.uvicorn, "run", lambda *a, **k: pytest.fail("must not serve"))
    assert entrypoint.main() == entrypoint.EXIT_CONFIG_ERROR
    assert "SHORTENER_API_KEYS is required" in capsys.readouterr().err


def test_entrypoint_serves_with_hardened_uvicorn_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in list(os.environ):
        if name.startswith("SHORTENER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SHORTENER_API_KEYS", f"ci:{new_secret()}")
    monkeypatch.setenv("SHORTENER_DATABASE_PATH", str(tmp_path / "x.db"))
    captured: dict[str, object] = {}
    monkeypatch.setattr(entrypoint.uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))
    assert entrypoint.main() == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    assert captured["server_header"] is False
    assert captured["proxy_headers"] is False
    assert captured["access_log"] is False


def test_running_the_module_exits_with_the_entrypoint_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith("SHORTENER_"):
            monkeypatch.delenv(name)
    monkeypatch.delitem(sys.modules, "shortener.__main__", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("shortener.__main__", run_name="__main__")
    assert excinfo.value.code == 2
    assert "SHORTENER_API_KEYS" in capsys.readouterr().err
