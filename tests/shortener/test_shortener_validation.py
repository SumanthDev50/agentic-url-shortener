"""Domain validation: destination URLs (SSRF vectors), aliases, expiry, idempotency keys."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from shortener.domain.errors import (
    InvalidAliasError,
    InvalidExpiryError,
    InvalidIdempotencyKeyError,
    InvalidUrlError,
)
from shortener.domain.validation import (
    BLOCKED_HOST_SUFFIXES,
    MAX_TTL_SECONDS,
    RESERVED_ALIASES,
    UrlPolicy,
    canonical_host,
    is_safe_redirect_target,
    is_valid_code_format,
    resolve_expiry,
    validate_alias,
    validate_destination_url,
    validate_idempotency_key,
)

NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
POLICY = UrlPolicy(own_hosts=frozenset({"sho.example.com"}))


def check(url: object) -> str:
    return validate_destination_url(url, POLICY)


# ----------------------------------------------------------------------------- accepted URLs


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("https://example.com", "https://example.com/"),
        ("http://example.com/path?q=1#frag", "http://example.com/path?q=1#frag"),
        ("  https://example.com/a  ", "https://example.com/a"),
        ("HTTPS://EXAMPLE.COM/Path", "https://example.com/Path"),
        ("https://example.com:443/x", "https://example.com/x"),
        ("http://example.com:80/x", "http://example.com/x"),
        ("https://example.com:8443/x", "https://example.com:8443/x"),
        ("https://example.com./x", "https://example.com/x"),
        ("https://example.com/a b".replace(" ", "%20"), "https://example.com/a%20b"),
        ("https://example.com/%zz", "https://example.com/%25zz"),
        ("https://example.com/%2f%41", "https://example.com/%2f%41"),
        ("https://example.com/caf\u00e9", "https://example.com/caf%C3%A9"),
        ("https://example.com/?q=\u00e9&r=a|b", "https://example.com/?q=%C3%A9&r=a%7Cb"),
        ("https://example.com/#sec\u00e9", "https://example.com/#sec%C3%A9"),
        ("https://example.com/?", "https://example.com/"),
        ("https://sub.example.co.uk/a/b/", "https://sub.example.co.uk/a/b/"),
        ("https://8.8.8.8/dns", "https://8.8.8.8/dns"),
        ("https://16843009/x", "https://1.1.1.1/x"),  # decimal notation of a public address
        ("https://[2606:4700:4700::1111]/x", "https://[2606:4700:4700::1111]/x"),
        ("https://[::ffff:8.8.8.8]/x", "https://8.8.8.8/x"),
        ("https://xn--bcher-kva.example/", "https://xn--bcher-kva.example/"),
        ("https://b\u00fccher.example/", "https://xn--bcher-kva.example/"),
        ("https://\u4f8b\u3048.example/", "https://xn--r8jz45g.example/"),
        ("https://example.com/\u00fc\u2603", "https://example.com/%C3%BC%E2%98%83"),
    ],
)
def test_valid_urls_are_canonicalised(raw: str, canonical: str) -> None:
    assert check(raw) == canonical


def test_canonical_form_is_ascii_and_idempotent() -> None:
    first = check("https://B\u00fccher.example:8443/p\u00e4th?x=\u00e9#\u00e8")
    assert first.isascii()
    assert check(first) == first


def test_max_length_boundary() -> None:
    prefix = "https://example.com/"
    assert check(prefix + "a" * (2048 - len(prefix))) is not None
    with pytest.raises(InvalidUrlError):
        check(prefix + "a" * (2049 - len(prefix)))


def test_url_that_expands_beyond_limit_when_normalised_is_rejected() -> None:
    with pytest.raises(InvalidUrlError):
        check("https://example.com/" + "\u00e9" * 1000)


# ----------------------------------------------------------------------------- scheme / shape


@pytest.mark.parametrize(
    "raw",
    [
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
        "ftp://example.com/file",
        "gopher://example.com",
        "mailto:a@example.com",
        "java\tscript:alert(1)",
        "java\nscript:alert(1)",
        "\tjavascript:alert(1)",
        "https:\\evil.com",
        "https:\\\\evil.com",
        "https://example.com\\@evil.com",
        "https://example.com/\\evil",
        "//example.com/path",
        "example.com/path",
        "/relative/path",
        "https:example.com",
        "https:/example.com",
        "https:///example.com",
        "https://",
        "https://:443/",
        "",
        "   ",
        "http://",
    ],
)
def test_non_http_schemes_and_malformed_shapes_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidUrlError):
        check(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "https://example.com/\r\nSet-Cookie: session=1",
        "https://example.com/\nX-Injected: 1",
        "https://example.com/\rfoo",
        "https://exam ple.com/",
        "https://example.com/a b",
        "https://example.com/\x00",
        "https://example.com/\x7f",
        "https://example.com/\x1b[31m",
        "https://example.com/\u0085",  # next line (C1 control)
        "https://example.com/\u2028",  # line separator
        "https://example.com/\u00a0",  # no-break space
        "https://example.com/\u200b",  # zero width space (format)
        "https://example.com/\u202e",  # right-to-left override (bidi)
        "https://example.com/\u2066",  # isolate (bidi)
        "https://exa\u200dmple.com/",  # zero width joiner
        "https://example.com/\ufeff",  # BOM
        "https://example.com/\ue000",  # private use
        "https://example.com/\ud800",  # lone surrogate
    ],
)
def test_control_whitespace_and_format_characters_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidUrlError):
        check(raw)


def test_percent_encoded_crlf_is_kept_encoded_not_decoded() -> None:
    assert (
        check("https://example.com/%0d%0aSet-Cookie:x") == "https://example.com/%0d%0aSet-Cookie:x"
    )


def test_non_string_input_is_rejected() -> None:
    for value in (None, 123, b"https://example.com", ["https://example.com"]):
        with pytest.raises(InvalidUrlError):
            check(value)


@pytest.mark.parametrize(
    "raw",
    [
        "https://user@example.com/",
        "https://user:pass@example.com/",
        "https://:pass@example.com/",
        "https://@example.com/",
        "http://a@b",
        "https://example.com@evil.com/",
        "https://example.com:80@evil.com/",
    ],
)
def test_embedded_credentials_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidUrlError):
        check(raw)


def test_fragment_may_contain_at_sign_and_does_not_confuse_the_host() -> None:
    assert check("https://example.com/#@evil.com") == "https://example.com/#@evil.com"
    assert check("https://example.com?next=@evil.com") == "https://example.com/?next=@evil.com"


@pytest.mark.parametrize(
    "raw",
    [
        "https://example.com:0/",
        "https://example.com:65536/",
        "https://example.com:99999999999/",
        "https://example.com:abc/",
        "https://example.com:-1/",
        "https://example.com:80:80/",
        "https://example.com:\u0663/",  # non-ASCII digit
        "https://[::1/",
        "https://::1]/",
        "https://evil[2606:4700:4700::1111]/",
    ],
)
def test_bad_ports_and_malformed_brackets_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidUrlError):
        check(raw)


# ----------------------------------------------------------------------------- SSRF vectors


@pytest.mark.parametrize(
    "host",
    [
        # IPv4 loopback in every notation
        "127.0.0.1",
        "127.1",
        "127.0.1",
        "2130706433",
        "0x7f000001",
        "0x7f.1",
        "0x7f.0x0.0x0.0x1",
        "0177.0.0.1",
        "017700000001",
        "127.000.000.001",
        "127.0.0.1.",
        "[::ffff:127.0.0.1]",
        "[::ffff:7f00:1]",
        "[0:0:0:0:0:ffff:127.0.0.1]",
        # full-width / circled digits and ideographic dots normalised by IDNA
        "\uff11\uff12\uff17.\uff10.\uff10.\uff11",
        "127\u30020\u30020\u30021",
        "\u2460\u2461\u2466.0.0.1",
        # other special-purpose IPv4
        "0.0.0.0",
        "0",
        "10.0.0.1",
        "10.255.255.255",
        "172.16.0.1",
        "172.31.255.254",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "192.0.0.1",
        "192.0.2.1",
        "198.18.0.1",
        "198.51.100.1",
        "203.0.113.5",
        "224.0.0.1",
        "239.255.255.250",
        "240.0.0.1",
        "255.255.255.255",
        # IPv6
        "[::1]",
        "[::]",
        "[fe80::1]",
        "[fe80::1%25eth0]",
        "[fc00::1]",
        "[fd12:3456:789a::1]",
        "[ff02::1]",
        "[2001:db8::1]",
        "[64:ff9b::7f00:1]",
        "[::7f00:1]",
        "[2002:7f00:1::1]",
        "[::ffff:10.0.0.1]",
        "[::ffff:169.254.169.254]",
        # names
        "localhost",
        "LOCALHOST",
        "localhost.",
        "foo.localhost",
        "a.b.localhost",
        "printer.local",
        "db.internal",
        "svc.cluster.internal",
        "host.localdomain",
        "router.home.arpa",
        "nas.lan",
        "wiki.intranet",
        # single-label names resolve via search domains
        "intranet",
        "metadata",
        "example",
    ],
)
def test_non_public_destinations_are_rejected(host: str) -> None:
    with pytest.raises(InvalidUrlError):
        check(f"http://{host}/")


@pytest.mark.parametrize(
    "host",
    ["foo.123", "1.2.3.4.5", "1.2.3.256", "256.1.1.1", "1.2.3.0x", "0x.1", "08.1.1.1", "1_0.5"],
)
def test_numeric_last_label_must_be_a_valid_ipv4_address(host: str) -> None:
    """``foo.123`` would be parsed differently by different clients, so it is refused."""
    with pytest.raises(InvalidUrlError):
        check(f"http://{host}/")


@pytest.mark.parametrize("suffix", BLOCKED_HOST_SUFFIXES)
def test_every_blocked_suffix_matches_itself_and_subdomains(suffix: str) -> None:
    for host in (suffix, f"x.{suffix}", f"a.b.{suffix}"):
        with pytest.raises(InvalidUrlError):
            check(f"https://{host}/")


def test_lookalike_of_blocked_suffix_is_allowed() -> None:
    assert check("https://notlocalhost.com/") == "https://notlocalhost.com/"
    assert check("https://example.localhost.com/") == "https://example.localhost.com/"


# ----------------------------------------------------------------------------- hosts / IDN


@pytest.mark.parametrize(
    "host",
    [
        "-bad.example",
        "bad-.example",
        "a..example",
        ".example.com",
        "exa mple.com",
        "ex%61mple.com",
        "ex*mple.com",
        "ex!mple.com",
        "a" * 64 + ".example",
        ".".join(["a" * 60] * 5) + ".com",
        "xn--zzzz-.example",  # invalid punycode
        "xn--.example",
    ],
)
def test_invalid_host_names_are_rejected(host: str) -> None:
    with pytest.raises(InvalidUrlError):
        check(f"https://{host}/")


def test_mixed_script_labels_are_rejected() -> None:
    cyrillic_a = "\u0430"
    with pytest.raises(InvalidUrlError):
        check(f"https://{cyrillic_a}pple.example/")  # Cyrillic a + Latin pple
    with pytest.raises(InvalidUrlError):
        check("https://xn--pple-43d.example/")  # same host, already punycode


def test_single_script_idn_and_cjk_mixture_are_allowed() -> None:
    assert check("https://\u043f\u0440\u0438\u043c\u0435\u0440.example/").startswith("https://xn--")
    assert check("https://\u65e5\u672c\u8a9e\u306e\u30c6\u30b9\u30c8.example/").startswith(
        "https://xn--"
    )


def test_each_label_is_checked_independently() -> None:
    assert check("https://\u043f\u0440\u0438\u043c\u0435\u0440.example.com/") is not None


def test_host_length_boundary() -> None:
    ok = ".".join(["a" * 63, "a" * 63, "a" * 61, "a" * 59, "com"])  # 253 characters
    assert len(ok) == 253
    assert check(f"https://{ok}/") == f"https://{ok}/"
    with pytest.raises(InvalidUrlError):
        check(f"https://{ok}x/")


def test_canonical_host_is_used_for_configuration_values() -> None:
    assert canonical_host("Example.COM.") == "example.com"
    assert canonical_host("B\u00fccher.example") == "xn--bcher-kva.example"
    assert canonical_host("0x7f.1") == "127.0.0.1"
    assert canonical_host("::1") == "[::1]"
    assert canonical_host("::ffff:8.8.8.8") == "8.8.8.8"
    with pytest.raises(InvalidUrlError):
        canonical_host("fe80::1%eth0")
    with pytest.raises(InvalidUrlError):
        canonical_host("not a host")


# ----------------------------------------------------------------------------- policy


def test_own_host_loop_is_rejected_in_every_spelling() -> None:
    for raw in (
        "https://sho.example.com/abc",
        "http://SHO.EXAMPLE.COM:8080/abc",
        "https://sho.example.com./abc",
    ):
        with pytest.raises(InvalidUrlError):
            check(raw)
    assert check("https://other.example.com/abc") == "https://other.example.com/abc"


def test_own_host_ip_literal_loop_is_rejected_even_when_public() -> None:
    policy = UrlPolicy(own_hosts=frozenset({"8.8.8.8"}))
    with pytest.raises(InvalidUrlError):
        validate_destination_url("http://8.8.8.8/x", policy)
    with pytest.raises(InvalidUrlError):
        validate_destination_url("http://134744072/x", policy)  # same address, decimal


def test_deny_list_blocks_domain_and_subdomains() -> None:
    policy = UrlPolicy(deny_hosts=("evil.example",))
    for host in ("evil.example", "www.evil.example"):
        with pytest.raises(InvalidUrlError):
            validate_destination_url(f"https://{host}/", policy)
    assert validate_destination_url("https://notevil.example/", policy)


def test_allow_list_restricts_destinations() -> None:
    policy = UrlPolicy(allow_hosts=("partner.example", "8.8.8.8"))
    assert validate_destination_url("https://partner.example/a", policy)
    assert validate_destination_url("https://shop.partner.example/a", policy)
    assert validate_destination_url("https://8.8.8.8/a", policy)
    with pytest.raises(InvalidUrlError):
        validate_destination_url("https://other.example/a", policy)
    with pytest.raises(InvalidUrlError):
        validate_destination_url("https://1.8.8.8/a", policy)  # IPs match exactly, not by suffix


def test_deny_list_ip_entries_match_exactly() -> None:
    policy = UrlPolicy(deny_hosts=("8.8.8.8",))
    with pytest.raises(InvalidUrlError):
        validate_destination_url("https://8.8.8.8/", policy)
    assert validate_destination_url("https://1.8.8.8/", policy)


def test_allow_list_cannot_unblock_private_addresses() -> None:
    policy = UrlPolicy(allow_hosts=("localhost", "127.0.0.1"))
    for host in ("localhost", "127.0.0.1"):
        with pytest.raises(InvalidUrlError):
            validate_destination_url(f"http://{host}/", policy)


def test_error_messages_never_echo_the_input() -> None:
    marker = "ReflectedMarker12345"
    for raw in (f"javascript:{marker}", f"http://127.0.0.1/{marker}\r\n", f"ftp://{marker}.com"):
        with pytest.raises(InvalidUrlError) as excinfo:
            check(raw)
        assert marker not in excinfo.value.message


# ----------------------------------------------------------------------------- aliases


@pytest.mark.parametrize("alias", ["abc", "A-b_9", "a" * 32, "0" * 3, "Spring-Sale_2026", "-_-"])
def test_valid_aliases(alias: str) -> None:
    assert validate_alias(alias) == alias


@pytest.mark.parametrize(
    "alias",
    [
        "",
        "ab",
        "a" * 33,
        "has space",
        "dot.dot",
        "slash/slash",
        "caf\u00e9",
        "abc\n",
        "abc\x00",
        "\u0430bc",
        "<script>",
        "a%20b",
        "..",
        None,
        123,
    ],
)
def test_invalid_aliases(alias: object) -> None:
    with pytest.raises(InvalidAliasError):
        validate_alias(alias)


@pytest.mark.parametrize("word", sorted(RESERVED_ALIASES))
def test_reserved_aliases_are_rejected_case_insensitively(word: str) -> None:
    for variant in (word, word.upper(), word.title()):
        with pytest.raises(InvalidAliasError):
            validate_alias(variant)


def test_aliases_are_case_sensitive() -> None:
    assert validate_alias("Promo") == "Promo"
    assert validate_alias("promo") == "promo"


def test_code_format_prefilter() -> None:
    assert is_valid_code_format("abc")
    assert is_valid_code_format("a" * 32)
    assert not is_valid_code_format("ab")
    assert not is_valid_code_format("a" * 33)
    assert not is_valid_code_format("favicon.ico")
    assert not is_valid_code_format("abc\n")


# ----------------------------------------------------------------------------- expiry


def test_no_expiry() -> None:
    assert resolve_expiry(now=NOW, expires_at=None, ttl_seconds=None) is None


def test_ttl_boundaries() -> None:
    assert resolve_expiry(now=NOW, expires_at=None, ttl_seconds=60) == NOW + timedelta(seconds=60)
    assert resolve_expiry(now=NOW, expires_at=None, ttl_seconds=MAX_TTL_SECONDS) == NOW + timedelta(
        seconds=MAX_TTL_SECONDS
    )
    for bad in (59, 0, -1, MAX_TTL_SECONDS + 1):
        with pytest.raises(InvalidExpiryError):
            resolve_expiry(now=NOW, expires_at=None, ttl_seconds=bad)


def test_boolean_ttl_is_rejected() -> None:
    with pytest.raises(InvalidExpiryError):
        resolve_expiry(now=NOW, expires_at=None, ttl_seconds=True)


def test_expires_at_and_ttl_are_mutually_exclusive() -> None:
    with pytest.raises(InvalidExpiryError):
        resolve_expiry(now=NOW, expires_at=NOW + timedelta(days=1), ttl_seconds=3600)


def test_expires_at_must_be_in_the_future_and_within_five_years() -> None:
    assert resolve_expiry(now=NOW, expires_at=NOW + timedelta(seconds=1), ttl_seconds=None)
    limit = NOW + timedelta(seconds=MAX_TTL_SECONDS)
    assert resolve_expiry(now=NOW, expires_at=limit, ttl_seconds=None) == limit
    for bad in (NOW, NOW - timedelta(days=1), limit + timedelta(seconds=1)):
        with pytest.raises(InvalidExpiryError):
            resolve_expiry(now=NOW, expires_at=bad, ttl_seconds=None)


def test_expires_at_is_normalised_to_utc_and_requires_an_offset() -> None:
    plus_two = timezone(timedelta(hours=2))
    value = datetime(2026, 2, 1, 12, 0, tzinfo=plus_two)
    assert resolve_expiry(now=NOW, expires_at=value, ttl_seconds=None) == datetime(
        2026, 2, 1, 10, 0, tzinfo=UTC
    )
    with pytest.raises(InvalidExpiryError):
        resolve_expiry(now=NOW, expires_at=datetime(2026, 2, 1, 12, 0), ttl_seconds=None)


def test_extreme_expires_at_does_not_overflow() -> None:
    extreme = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone(timedelta(hours=-23)))
    with pytest.raises(InvalidExpiryError):
        resolve_expiry(now=NOW, expires_at=extreme, ttl_seconds=None)


# ----------------------------------------------------------------------------- misc


@pytest.mark.parametrize("key", ["a", "k" * 128, "abc-123_XYZ.~", "!#$%&'()*+,-./:;<=>?@[]^_`{|}~"])
def test_valid_idempotency_keys(key: str) -> None:
    assert validate_idempotency_key(key) == key


@pytest.mark.parametrize("key", ["", "k" * 129, "has space", "tab\t", "nl\n", "caf\u00e9", None, 5])
def test_invalid_idempotency_keys(key: object) -> None:
    with pytest.raises(InvalidIdempotencyKeyError):
        validate_idempotency_key(key)


def test_redirect_target_safety_check() -> None:
    assert is_safe_redirect_target("https://example.com/a?b=c%20d")
    assert not is_safe_redirect_target("https://example.com/\r\nSet-Cookie: x")
    assert not is_safe_redirect_target("https://example.com/\u00e9")
    assert not is_safe_redirect_target("")
    assert not is_safe_redirect_target("https://example.com/" + "a" * 3000)
