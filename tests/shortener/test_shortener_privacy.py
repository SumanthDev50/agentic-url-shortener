"""Privacy reductions (IP truncation, visitor id, referrer) and user-agent classification."""

from __future__ import annotations

from datetime import date

import pytest

from shortener.domain.privacy import (
    VISITOR_ID_LENGTH,
    daily_salt,
    referrer_origin,
    truncate_ip,
    visitor_id,
)
from shortener.domain.useragent import (
    MAX_USER_AGENT_LENGTH,
    OTHER,
    UNKNOWN,
    classify_user_agent,
)

SECRET = b"k" * 32
DAY = date(2026, 1, 15)


@pytest.mark.parametrize(
    ("ip", "expected"),
    [
        ("203.0.113.77", "203.0.113.0"),
        ("10.1.2.3", "10.1.2.0"),
        ("2001:db8:1234:5678:9abc:def0:1234:5678", "2001:db8:1234::"),
        ("::ffff:203.0.113.77", "203.0.113.0"),
        ("  203.0.113.77  ", "203.0.113.0"),
        ("::1", "::"),
        (None, "unknown"),
        ("", "unknown"),
        ("not-an-ip", "unknown"),
        ("203.0.113.256", "unknown"),
        ("203.0.113.77\r\nX: y", "unknown"),
    ],
)
def test_truncate_ip(ip: str | None, expected: str) -> None:
    assert truncate_ip(ip) == expected


def test_visitor_id_is_stable_within_a_day_and_for_one_network() -> None:
    first = visitor_id(SECRET, DAY, "203.0.113.5", "Chrome")
    assert first == visitor_id(SECRET, DAY, "203.0.113.5", "Chrome")
    assert first == visitor_id(SECRET, DAY, "203.0.113.200", "Chrome")  # same /24
    assert len(first) == VISITOR_ID_LENGTH
    assert int(first, 16) >= 0


def test_visitor_id_differs_by_network_browser_day_and_secret() -> None:
    base = visitor_id(SECRET, DAY, "203.0.113.5", "Chrome")
    assert base != visitor_id(SECRET, DAY, "203.0.114.5", "Chrome")
    assert base != visitor_id(SECRET, DAY, "203.0.113.5", "Firefox")
    assert base != visitor_id(SECRET, date(2026, 1, 16), "203.0.113.5", "Chrome")  # unlinkable
    assert base != visitor_id(b"x" * 32, DAY, "203.0.113.5", "Chrome")


def test_visitor_id_does_not_contain_the_address() -> None:
    value = visitor_id(SECRET, DAY, "203.0.113.5", "Chrome")
    assert "203" not in value and "113" not in value


def test_daily_salt_is_derived_not_stored_and_day_specific() -> None:
    assert daily_salt(SECRET, DAY) == daily_salt(SECRET, DAY)
    assert daily_salt(SECRET, DAY) != daily_salt(SECRET, date(2026, 1, 16))
    assert len(daily_salt(SECRET, DAY)) == 32


@pytest.mark.parametrize(
    ("referer", "expected"),
    [
        ("https://news.example.org/story?id=1&token=abc#x", "https://news.example.org"),
        ("http://Example.COM:8080/path", "http://example.com"),
        ("https://sub.example.org", "https://sub.example.org"),
        ("https://user:pw@example.org/x", "https://example.org"),
        (None, None),
        ("", None),
        ("   ", None),
        ("ftp://example.org/x", None),
        ("javascript:alert(1)", None),
        ("example.org/path", None),
        ("https:///path", None),
        ("https://[::1]/x", None),
        ("https://exa mple.org/", None),
        ("https://" + "a" * 300 + ".org/", None),
        ("https://example.org/" + "a" * 3000, None),
        ("https://[bad/", None),
    ],
)
def test_referrer_origin_keeps_only_scheme_and_host(
    referer: str | None, expected: str | None
) -> None:
    assert referrer_origin(referer) == expected


# ----------------------------------------------------------------------------- user agents


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36",
            ("Chrome", "desktop"),
        ),
        (
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/120.0 Safari/537.36 Edg/120.0",
            ("Edge", "desktop"),
        ),
        (
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/120.0 Safari/537.36 OPR/105.0",
            ("Opera", "desktop"),
        ),
        (
            "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
            ("Firefox", "desktop"),
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            ("Safari", "desktop"),
        ),
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
            "Version/17.0 Mobile/15E148 Safari/604.1",
            ("Safari", "mobile"),
        ),
        (
            "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 "
            "Mobile/15E148 Safari/604.1",
            ("Safari", "tablet"),
        ),
        (
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/120.0 Mobile "
            "Safari/537.36",
            ("Chrome", "mobile"),
        ),
        (
            "Mozilla/5.0 (Linux; Android 13; SM-X700) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
            ("Chrome", "tablet"),
        ),
        (
            "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 SamsungBrowser/23.0 Chrome/115.0 "
            "Mobile Safari/537.36",
            ("Samsung Internet", "mobile"),
        ),
        ("Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; Trident/4.0)", ("IE", "desktop")),
        ("curl/8.4.0", ("Bot", "bot")),
        ("python-requests/2.31.0", ("Bot", "bot")),
        (
            "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            ("Bot", "bot"),
        ),
        ("SomethingUnknown/1.0", (OTHER, "desktop")),
        (None, (UNKNOWN, "unknown")),
        ("", (UNKNOWN, "unknown")),
        ("   ", (UNKNOWN, "unknown")),
    ],
)
def test_classification(user_agent: str | None, expected: tuple[str, str]) -> None:
    assert classify_user_agent(user_agent) == expected


def test_only_a_fixed_label_set_can_ever_be_stored() -> None:
    labels = {
        classify_user_agent(f"Mozilla/{index} custom-{index} Evil<script>{index}")
        for index in range(50)
    }
    assert labels == {(OTHER, "desktop")}


def test_user_agent_is_truncated_before_matching() -> None:
    padded = "x" * MAX_USER_AGENT_LENGTH + " Firefox/120.0"
    assert classify_user_agent(padded) == (OTHER, "desktop")
