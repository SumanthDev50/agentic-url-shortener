"""API integration tests (TestClient): the product behaviour end to end.

Tests that assert totals, counters, timing or rate limits use the per-test ``client`` fixture;
tests that only check how single requests are validated or answered share one application per
module (``shared``) to keep the suite fast.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shortener.container import Services
from shortener.services.idempotency import IDEMPOTENCY_TTL
from tests.shortener.conftest import SharedApp
from tests.shortener.support import FakeClock, api_client

Headers = dict[str, str]


def create(
    client: TestClient, headers: Headers, url: str = "https://example.com/page", **extra: Any
) -> Any:
    return client.post("/api/v1/links", json={"url": url, **extra}, headers=headers)


def flush(services: Services) -> None:
    assert services.recorder.flush(timeout=5)


# ----------------------------------------------------------------------------- the main flow


def test_create_redirect_analytics_disable_flow(
    client: TestClient, services: Services, alice: Headers
) -> None:
    created = create(client, alice, "https://Example.com/landing?utm=1")
    assert created.status_code == 201
    link = created.json()
    assert set(link) == {
        "code",
        "short_url",
        "original_url",
        "created_at",
        "expires_at",
        "status",
        "click_count",
    }
    assert re.fullmatch(r"[0-9A-Za-z]{7}", link["code"])
    assert link["short_url"] == f"https://sho.example.com/{link['code']}"
    assert link["original_url"] == "https://example.com/landing?utm=1"
    assert (link["expires_at"], link["status"], link["click_count"]) == (None, "active", 0)
    assert created.headers["location"] == f"/api/v1/links/{link['code']}"

    redirect = client.get(
        f"/{link['code']}",
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux) Firefox/120.0",
            "Referer": "https://news.example.org/a?b=c",
        },
    )
    assert redirect.status_code == 302
    assert redirect.headers["location"] == "https://example.com/landing?utm=1"
    assert redirect.headers["cache-control"] == "no-store"
    assert redirect.headers["referrer-policy"] == "no-referrer"
    assert redirect.headers["x-content-type-options"] == "nosniff"
    assert redirect.text == "Redirecting"

    flush(services)
    analytics = client.get(f"/api/v1/links/{link['code']}/analytics?days=7", headers=alice).json()
    assert analytics["code"] == link["code"]
    assert (analytics["total_clicks"], analytics["unique_visitors"], analytics["days"]) == (1, 1, 7)
    assert len(analytics["clicks_by_day"]) == 7
    assert analytics["clicks_by_day"][-1] == {"date": "2026-01-15", "clicks": 1}
    assert analytics["top_referrers"] == [{"name": "https://news.example.org", "clicks": 1}]
    assert analytics["browsers"] == [{"name": "Firefox", "clicks": 1}]
    assert analytics["devices"] == [{"name": "desktop", "clicks": 1}]
    assert analytics["first_click_at"] == analytics["last_click_at"] is not None

    details = client.get(f"/api/v1/links/{link['code']}", headers=alice).json()
    assert (details["click_count"], details["status"]) == (1, "active")

    assert client.delete(f"/api/v1/links/{link['code']}", headers=alice).status_code == 204
    assert client.delete(f"/api/v1/links/{link['code']}", headers=alice).status_code == 204
    gone = client.get(f"/{link['code']}")
    assert gone.status_code == 410
    assert gone.json()["error"]["code"] == "link_gone"
    assert client.get(f"/api/v1/links/{link['code']}", headers=alice).json()["status"] == "disabled"


def test_unique_visitors_and_zero_filled_days(
    client: TestClient, services: Services, alice: Headers
) -> None:
    code = create(client, alice).json()["code"]
    for agent in ("Firefox/120", "Firefox/120", "Chrome/120"):
        client.get(f"/{code}", headers={"User-Agent": agent})
    flush(services)
    data = client.get(f"/api/v1/links/{code}/analytics", headers=alice).json()
    assert data["days"] == 30 and len(data["clicks_by_day"]) == 30
    assert data["total_clicks"] == 3
    assert data["unique_visitors"] == 2  # same network, two browser families
    assert sum(day["clicks"] for day in data["clicks_by_day"]) == 3
    assert [day["clicks"] for day in data["clicks_by_day"][:-1]] == [0] * 29


def test_analytics_window_follows_the_clock(
    client: TestClient, services: Services, alice: Headers, clock: FakeClock
) -> None:
    code = create(client, alice).json()["code"]
    client.get(f"/{code}")
    flush(services)
    clock.advance(3 * 86400)
    data = client.get(f"/api/v1/links/{code}/analytics?days=2", headers=alice).json()
    assert data["total_clicks"] == 0  # the click is outside the two-day window
    assert [d["date"] for d in data["clicks_by_day"]] == ["2026-01-17", "2026-01-18"]
    wide = client.get(f"/api/v1/links/{code}/analytics?days=5", headers=alice).json()
    assert wide["total_clicks"] == 1


def test_ttl_expiry_makes_the_link_gone(
    client: TestClient, alice: Headers, clock: FakeClock
) -> None:
    link = create(client, alice, ttl_seconds=60).json()
    assert link["expires_at"] == "2026-01-15T12:01:00Z"
    assert client.get(f"/{link['code']}").status_code == 302
    clock.advance(60)
    gone = client.get(f"/{link['code']}")
    assert gone.status_code == 410
    assert gone.json()["error"]["message"] == "The link has expired."
    status = client.get(f"/api/v1/links/{link['code']}", headers=alice).json()["status"]
    assert status == "expired"


def test_rejected_creations_store_nothing(client: TestClient, alice: Headers) -> None:
    for payload in (
        {"url": "http://127.0.0.1/"},
        {"url": "https://example.com", "alias": "api"},
        {"url": "https://example.com", "ttl_seconds": 1},
        {"url": "https://example.com", "bogus": True},
    ):
        assert client.post("/api/v1/links", json=payload, headers=alice).status_code == 422
    assert client.get("/api/v1/links", headers=alice).json()["total"] == 0


# ----------------------------------------------------------------------------- creation rules


def test_custom_alias_conflicts_return_409_and_keep_the_original(
    client: TestClient, alice: Headers, bob: Headers
) -> None:
    first = create(client, alice, "https://example.com/one", alias="promo")
    assert first.status_code == 201 and first.json()["code"] == "promo"
    clash = create(client, bob, "https://example.com/two", alias="promo")
    assert clash.status_code == 409
    assert clash.json()["error"]["code"] == "alias_taken"
    assert client.get("/promo").headers["location"] == "https://example.com/one"
    assert create(client, alice, alias="Promo").status_code == 201  # aliases are case-sensitive


@pytest.mark.parametrize(
    ("alias", "code"),
    [
        ("api", "invalid_alias"),
        ("ADMIN", "invalid_alias"),
        ("metrics", "invalid_alias"),
        ("ab", "validation_error"),
        ("a" * 33, "validation_error"),
        ("has space", "validation_error"),
        ("dot.dot", "validation_error"),
        ("slash/slash", "validation_error"),
        ("abc\n", "validation_error"),
    ],
)
def test_invalid_aliases_are_rejected(shared: SharedApp, alias: str, code: str) -> None:
    response = create(shared.client, shared.alice, alias=alias)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


def test_expires_at_is_normalised_to_utc(shared: SharedApp) -> None:
    link = create(shared.client, shared.alice, expires_at="2026-02-01T12:00:00+02:00").json()
    assert link["expires_at"] == "2026-02-01T10:00:00Z"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"ttl_seconds": 59}, "validation_error"),
        ({"ttl_seconds": 157_680_001}, "validation_error"),
        ({"ttl_seconds": "3600"}, "validation_error"),
        ({"ttl_seconds": True}, "validation_error"),
        ({"ttl_seconds": 60.5}, "validation_error"),
        ({"expires_at": "2026-01-15T11:59:59Z"}, "invalid_expiry"),
        ({"expires_at": "2026-01-15T12:00:00Z"}, "invalid_expiry"),
        ({"expires_at": "2031-01-16T00:00:00Z"}, "invalid_expiry"),
        ({"expires_at": "2026-06-01T00:00:00"}, "invalid_expiry"),
        ({"expires_at": "not a date"}, "validation_error"),
        ({"expires_at": 1893456000}, "validation_error"),
        ({"expires_at": "2026-06-01T00:00:00Z", "ttl_seconds": 3600}, "invalid_expiry"),
    ],
)
def test_invalid_expiry_inputs(shared: SharedApp, payload: dict[str, Any], code: str) -> None:
    response = create(shared.client, shared.alice, **payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html;base64,AAAA",
        "file:///etc/passwd",
        "http://127.0.0.1/admin",
        "http://127.1/",
        "http://2130706433/",
        "http://[::ffff:127.0.0.1]/",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost:8080/",
        "http://service.internal/",
        "http://a@b/",
        "https://user:pw@example.com/",
        "https://example.com/\r\nSet-Cookie: x=1",
        "https:\\\\evil.com",
        "https://sho.example.com/self",
        "http://" + "a" * 64 + ".example.com/",
        "https://example.com/" + "a" * 2100,
        "",
    ],
)
def test_hostile_destination_urls_are_rejected(shared: SharedApp, url: str) -> None:
    response = create(shared.client, shared.alice, url)
    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"invalid_url", "validation_error"}


def test_idn_destinations_are_stored_as_punycode(shared: SharedApp) -> None:
    link = create(shared.client, shared.alice, "https://bücher.example/café?q=é").json()
    assert link["original_url"] == "https://xn--bcher-kva.example/caf%C3%A9?q=%C3%A9"
    assert shared.client.get(f"/{link['code']}").headers["location"] == link["original_url"]


def test_unknown_and_mistyped_fields_are_rejected(shared: SharedApp) -> None:
    for payload in (
        {"url": "https://example.com", "owner": "bob"},
        {"url": "https://example.com", "ttl": 5},
        {},
    ):
        response = shared.client.post("/api/v1/links", json=payload, headers=shared.alice)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
    for bad in (
        {"url": 5},
        {"url": None},
        {"url": ["https://example.com"]},
        {"url": "https://example.com", "alias": 5},
    ):
        assert (
            shared.client.post("/api/v1/links", json=bad, headers=shared.alice).status_code == 422
        )


def test_validation_details_name_the_field_without_echoing_values(shared: SharedApp) -> None:
    echoed = "SensitiveValue123"
    response = shared.client.post(
        "/api/v1/links",
        json={"url": "https://example.com", "ttl_seconds": echoed},
        headers=shared.alice,
    )
    assert response.json()["error"]["details"][0]["location"] == "body.ttl_seconds"
    assert echoed not in response.text


def test_malformed_json_bodies_are_rejected_with_the_error_model(shared: SharedApp) -> None:
    for content in (b"{not json", b"", b"[]", b'"str"', b"\xff\xfe"):
        response = shared.client.post(
            "/api/v1/links",
            content=content,
            headers={**shared.alice, "Content-Type": "application/json"},
        )
        assert response.status_code == 422, content
        assert response.json()["error"]["code"] == "validation_error"


# ----------------------------------------------------------------------------- redirects


def test_redirect_to_unknown_code_is_404_with_error_model(shared: SharedApp) -> None:
    response = shared.client.get("/doesnotexist")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize(
    "path", ["/ab", "/" + "x" * 100, "/favicon.ico", "/robots.txt", "/api", "/a%20b", "/%00"]
)
def test_malformed_or_reserved_paths_are_404(shared: SharedApp, path: str) -> None:
    assert shared.client.get(path).status_code == 404


def test_404_bodies_never_reflect_the_requested_path(shared: SharedApp) -> None:
    marker = "ReflectedMarker987"
    for path in (f"/{marker}", f"/api/v1/{marker}", f"/{marker}/deeper"):
        assert marker not in shared.client.get(path).text


def test_redirect_only_accepts_get(shared: SharedApp) -> None:
    code = create(shared.client, shared.alice).json()["code"]
    assert shared.client.head(f"/{code}").status_code == 405
    assert shared.client.post(f"/{code}").status_code == 405
    assert shared.client.delete(f"/{code}").status_code == 405


# ----------------------------------------------------------------------------- list / details


def test_listing_is_paginated_newest_first_and_owner_scoped(
    client: TestClient, alice: Headers, bob: Headers
) -> None:
    codes = [create(client, alice, f"https://example.com/{i}").json()["code"] for i in range(5)]
    create(client, bob, "https://example.com/bob")
    page = client.get("/api/v1/links?limit=2&offset=0", headers=alice).json()
    assert [item["code"] for item in page["items"]] == [codes[4], codes[3]]
    assert (page["total"], page["limit"], page["offset"]) == (5, 2, 0)
    page = client.get("/api/v1/links?limit=2&offset=4", headers=alice).json()
    assert [item["code"] for item in page["items"]] == [codes[0]]
    default = client.get("/api/v1/links", headers=alice).json()
    assert (default["limit"], default["offset"], len(default["items"])) == (20, 0, 5)
    assert client.get("/api/v1/links", headers=bob).json()["total"] == 1


@pytest.mark.parametrize(
    "query", ["limit=0", "limit=101", "limit=-1", "offset=-1", "limit=abc", "offset=1000001"]
)
def test_list_parameters_are_bounded(shared: SharedApp, query: str) -> None:
    response = shared.client.get(f"/api/v1/links?{query}", headers=shared.alice)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize("days", [0, 366, -3, "x"])
def test_analytics_days_are_bounded(shared: SharedApp, days: Any) -> None:
    code = create(shared.client, shared.alice).json()["code"]
    url = f"/api/v1/links/{code}/analytics?days={days}"
    assert shared.client.get(url, headers=shared.alice).status_code == 422


def test_analytics_days_boundaries_are_accepted(shared: SharedApp) -> None:
    code = create(shared.client, shared.alice).json()["code"]
    for days in (1, 365):
        data = shared.client.get(
            f"/api/v1/links/{code}/analytics?days={days}", headers=shared.alice
        ).json()
        assert len(data["clicks_by_day"]) == days


# ----------------------------------------------------------------------------- ownership


def test_other_owners_links_are_invisible_and_untouchable(
    client: TestClient, alice: Headers, bob: Headers
) -> None:
    code = create(client, alice).json()["code"]
    for response in (
        client.get(f"/api/v1/links/{code}", headers=bob),
        client.get(f"/api/v1/links/{code}/analytics", headers=bob),
        client.delete(f"/api/v1/links/{code}", headers=bob),
    ):
        assert response.status_code == 404  # never 403: existence is not revealed
        assert response.json()["error"]["code"] == "not_found"
    assert client.get(f"/{code}").status_code == 302  # bob's DELETE had no effect
    assert client.get(f"/api/v1/links/{code}", headers=alice).json()["status"] == "active"


def test_not_found_for_foreign_and_missing_links_is_indistinguishable(
    shared: SharedApp,
) -> None:
    code = create(shared.client, shared.alice).json()["code"]
    foreign = shared.client.get(f"/api/v1/links/{code}", headers=shared.bob).json()["error"]
    missing = shared.client.get("/api/v1/links/zzzzzzz", headers=shared.bob).json()["error"]
    assert {k: foreign[k] for k in ("code", "message")} == {
        k: missing[k] for k in ("code", "message")
    }


def test_created_by_is_the_authenticated_key_id(shared: SharedApp) -> None:
    mine = create(shared.client, shared.alice).json()["code"]
    theirs = create(shared.client, shared.bob).json()["code"]
    assert shared.services.links.get(mine, "alice").created_by == "alice"
    assert shared.services.links.get(theirs, "bob").created_by == "bob"


# ----------------------------------------------------------------------------- idempotency


def test_idempotent_replay_returns_the_same_response(client: TestClient, alice: Headers) -> None:
    headers = {**alice, "Idempotency-Key": "order-2026-0001"}
    first = create(client, headers, alias="idem-one", ttl_seconds=3600)
    second = create(client, headers, alias="idem-one", ttl_seconds=3600)
    assert (first.status_code, second.status_code) == (201, 201)
    assert second.content == first.content
    assert "idempotent-replay" not in first.headers
    assert second.headers["idempotent-replay"] == "true"
    assert second.headers["location"] == first.headers["location"]
    assert client.get("/api/v1/links", headers=alice).json()["total"] == 1


def test_idempotent_replay_does_not_regenerate_the_code(client: TestClient, alice: Headers) -> None:
    headers = {**alice, "Idempotency-Key": "generated-code"}
    first = create(client, headers).json()["code"]
    assert create(client, headers).json()["code"] == first
    assert create(client, alice).json()["code"] != first  # no key: a new link


def test_same_key_with_a_different_payload_is_rejected(client: TestClient, alice: Headers) -> None:
    headers = {**alice, "Idempotency-Key": "reused-key"}
    assert create(client, headers, "https://example.com/a").status_code == 201
    conflict = create(client, headers, "https://example.com/b")
    assert conflict.status_code == 422
    assert conflict.json()["error"]["code"] == "idempotency_key_reuse"
    assert client.get("/api/v1/links", headers=alice).json()["total"] == 1


def test_idempotency_keys_are_scoped_per_api_key(
    client: TestClient, alice: Headers, bob: Headers
) -> None:
    shared_key = "shared-key-1"
    first = create(client, {**alice, "Idempotency-Key": shared_key})
    second = create(client, {**bob, "Idempotency-Key": shared_key})
    assert (first.status_code, second.status_code) == (201, 201)
    assert first.json()["code"] != second.json()["code"]
    assert "idempotent-replay" not in second.headers


def test_idempotency_keys_expire_after_24_hours(
    client: TestClient, alice: Headers, clock: FakeClock
) -> None:
    headers = {**alice, "Idempotency-Key": "daily-key"}
    first = create(client, headers).json()["code"]
    clock.advance(IDEMPOTENCY_TTL.total_seconds() - 1)
    assert create(client, headers).headers["idempotent-replay"] == "true"
    clock.advance(1)
    renewed = create(client, headers)
    assert "idempotent-replay" not in renewed.headers
    assert renewed.json()["code"] != first


def test_a_failed_creation_does_not_consume_the_key(client: TestClient, alice: Headers) -> None:
    headers = {**alice, "Idempotency-Key": "retry-me"}
    assert create(client, headers, "http://127.0.0.1/").status_code == 422
    assert create(client, headers, "https://example.com/ok").status_code == 201


def test_invalid_idempotency_keys_are_rejected(shared: SharedApp) -> None:
    for key in (b"has space", b"tab\tkey", "café".encode()):
        response = shared.client.post(
            "/api/v1/links",
            json={"url": "https://example.com"},
            headers={**shared.alice, "Idempotency-Key": key},  # type: ignore[dict-item]
        )
        assert response.status_code == 422


def test_idempotency_key_length_limit(shared: SharedApp) -> None:
    ok = create(shared.client, {**shared.alice, "Idempotency-Key": "k" * 128})
    too_long = create(shared.client, {**shared.alice, "Idempotency-Key": "k" * 129})
    assert (ok.status_code, too_long.status_code) == (201, 422)
    blank = shared.client.post(
        "/api/v1/links",
        json={"url": "https://example.com"},
        headers={**shared.alice, "Idempotency-Key": ""},
    )
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "invalid_idempotency_key"


# ----------------------------------------------------------------------------- concurrency


def run_concurrently(count: int, work: Callable[[int], Any]) -> list[Any]:
    barrier = threading.Barrier(count)
    results: list[Any] = [None] * count

    def worker(index: int) -> None:
        barrier.wait()
        results[index] = work(index)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_concurrent_creation_of_the_same_alias_yields_exactly_one_201(
    client: TestClient, app: FastAPI, alice: Headers
) -> None:
    def attempt(index: int) -> int:
        worker = api_client(app)
        return create(worker, alice, f"https://example.com/{index}", alias="race-alias").status_code

    statuses = run_concurrently(8, attempt)
    assert sorted(statuses) == [201] + [409] * 7
    assert client.get("/api/v1/links", headers=alice).json()["total"] == 1


def test_concurrent_requests_with_the_same_idempotency_key_create_one_link(
    client: TestClient, app: FastAPI, alice: Headers
) -> None:
    headers = {**alice, "Idempotency-Key": "concurrent-key"}

    def attempt(_index: int) -> tuple[int, str]:
        response = create(api_client(app), headers, "https://example.com/same")
        return response.status_code, response.json()["code"]

    results = run_concurrently(6, attempt)
    assert {status for status, _ in results} == {201}
    assert len({code for _, code in results}) == 1
    assert client.get("/api/v1/links", headers=alice).json()["total"] == 1


def test_concurrent_creation_of_generated_codes_never_collides(
    client: TestClient, app: FastAPI, alice: Headers
) -> None:
    def attempt(index: int) -> str:
        response = create(api_client(app), alice, f"https://example.com/{index}")
        assert response.status_code == 201
        return str(response.json()["code"])

    assert len(set(run_concurrently(8, attempt))) == 8
    assert client.get("/api/v1/links", headers=alice).json()["total"] == 8
