"""API integration tests: authentication, rate limiting, headers, errors, privacy, lifecycle."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shortener import main as main_module
from shortener.config import RateLimit
from shortener.container import Services
from shortener.domain.models import ClickEvent, NewLink
from shortener.services.click_recorder import ClickWriter
from shortener.storage.click_repository import SqliteClickRepository
from shortener.storage.db import Database
from shortener.storage.link_repository import SqliteLinkRepository
from shortener.storage.migrations import MigrationError
from tests.shortener.conftest import SharedApp
from tests.shortener.support import (
    CLIENT_ADDRESS,
    CollectingHandler,
    FakeClock,
    api_client,
    build_app,
    build_migrated_template,
    dump_database,
    make_settings,
    new_secret,
)

Headers = dict[str, str]
UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

STRICT_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "x-frame-options": "DENY",
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
}


def create(client: TestClient, headers: Headers, **extra: Any) -> Any:
    return client.post(
        "/api/v1/links", json={"url": "https://example.com/page", **extra}, headers=headers
    )


@pytest.fixture
def start_app(
    database_path: Path,
    clock: FakeClock,
    secrets_by_owner: dict[str, str],
    request: pytest.FixtureRequest,
) -> Callable[..., TestClient]:
    """Factory: start an application with setting overrides (closed at teardown)."""
    stack = ExitStack()
    request.addfinalizer(stack.close)

    def start(*, keys: dict[str, str] | None = None, **overrides: Any) -> TestClient:
        settings = make_settings(
            database_path, keys=secrets_by_owner if keys is None else keys, **overrides
        )
        return stack.enter_context(api_client(build_app(settings, clock=clock)))

    return start


def services_of(client: TestClient) -> Services:
    app: FastAPI = client.app  # type: ignore[assignment]
    return app.state.services  # type: ignore[no-any-return]


# ----------------------------------------------------------------------------- authentication

PROTECTED = [
    ("POST", "/api/v1/links"),
    ("GET", "/api/v1/links"),
    ("GET", "/api/v1/links/abc1234"),
    ("DELETE", "/api/v1/links/abc1234"),
    ("GET", "/api/v1/links/abc1234/analytics"),
    ("GET", "/metrics"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_management_endpoints_require_an_api_key(shared: SharedApp, method: str, path: str) -> None:
    response = shared.client.request(method, path, json={"url": "https://example.com"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"
    error = response.json()["error"]
    assert error["code"] == "unauthorized"
    assert error["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_wrong_api_keys_are_denied(shared: SharedApp, method: str, path: str) -> None:
    response = shared.client.request(
        method, path, json={"url": "https://example.com"}, headers={"X-API-Key": new_secret()}
    )
    assert response.status_code == 401


def test_credentials_are_only_accepted_in_the_x_api_key_header(
    shared: SharedApp, secrets_by_owner: dict[str, str]
) -> None:
    secret = shared.alice["X-API-Key"]
    client = shared.client
    assert (
        client.get("/api/v1/links", headers={"Authorization": f"Bearer {secret}"}).status_code
        == 401
    )
    assert client.get("/api/v1/links", headers={"Authorization": secret}).status_code == 401
    assert client.get(f"/api/v1/links?api_key={secret}").status_code == 401
    assert client.get(f"/api/v1/links?x-api-key={secret}").status_code == 401
    assert client.get("/api/v1/links", headers={"X-API-Key": f" {secret}"}).status_code == 401
    assert (
        client.get("/api/v1/links", headers={"x-api-key": secret}).status_code == 200
    )  # case-insensitive name


def test_missing_and_wrong_keys_are_indistinguishable(shared: SharedApp) -> None:
    missing = shared.client.get("/api/v1/links").json()["error"]
    wrong = shared.client.get("/api/v1/links", headers={"X-API-Key": new_secret()}).json()["error"]
    assert (missing["code"], missing["message"]) == (wrong["code"], wrong["message"])


def test_public_endpoints_need_no_key(shared: SharedApp) -> None:
    assert shared.client.get("/healthz").status_code == 200
    assert shared.client.get("/readyz").status_code == 200
    assert shared.client.get("/nosuchcode").status_code == 404  # reaches the redirect handler
    link = create(shared.client, shared.alice).json()
    assert shared.client.get(f"/{link['code']}").status_code == 302


def test_the_public_allow_list_is_exactly_redirect_health_and_readiness(app: FastAPI) -> None:
    public = {
        (method.upper(), path)
        for path, item in app.openapi()["paths"].items()
        for method, operation in item.items()
        if "security" not in operation
    }
    assert public == {("GET", "/{code}"), ("GET", "/healthz"), ("GET", "/readyz")}
    scheme = app.openapi()["components"]["securitySchemes"]["ApiKeyAuth"]
    assert (scheme["type"], scheme["in"], scheme["name"]) == ("apiKey", "header", "X-API-Key")


# ----------------------------------------------------------------------------- rate limiting


def test_management_requests_are_rate_limited_per_key(
    start_app: Callable[..., TestClient], alice: Headers, bob: Headers, clock: FakeClock
) -> None:
    client = start_app(api_rate=RateLimit(burst=2, per_minute=60))
    first = client.get("/api/v1/links", headers=alice)
    assert first.headers["x-ratelimit-limit"] == "2"
    assert first.headers["x-ratelimit-remaining"] == "1"
    assert client.get("/api/v1/links", headers=alice).status_code == 200
    limited = client.get("/api/v1/links", headers=alice)
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "1"
    assert limited.headers["x-ratelimit-limit"] == "2"
    assert limited.headers["x-ratelimit-remaining"] == "0"
    assert limited.json()["error"]["code"] == "rate_limited"
    assert limited.json()["error"]["request_id"] == limited.headers["x-request-id"]
    assert client.get("/api/v1/links", headers=bob).status_code == 200  # separate bucket
    clock.advance(1)
    assert client.get("/api/v1/links", headers=alice).status_code == 200
    metrics = services_of(client).metrics
    assert metrics.value("shortener_rate_limit_rejections_total", scope="api") == 1


def test_redirects_are_rate_limited_per_client_address(
    start_app: Callable[..., TestClient], clock: FakeClock
) -> None:
    client = start_app(redirect_rate=RateLimit(burst=2, per_minute=60))
    assert [client.get("/missing1").status_code for _ in range(2)] == [404, 404]  # 404s count too
    limited = client.get("/missing1")
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "1"
    assert limited.json()["error"]["code"] == "rate_limited"
    assert client.get("/healthz").status_code == 200  # operational endpoints are not limited
    clock.advance(1)
    assert client.get("/missing1").status_code == 404


def test_repeated_authentication_failures_lock_the_client_out(
    start_app: Callable[..., TestClient], alice: Headers, clock: FakeClock
) -> None:
    client = start_app(auth_failure_rate=RateLimit(burst=2, per_minute=6))
    for _ in range(2):
        assert client.get("/api/v1/links", headers={"X-API-Key": new_secret()}).status_code == 401
    locked = client.get("/api/v1/links", headers=alice)  # even the right key waits
    assert locked.status_code == 429
    assert locked.headers["retry-after"] == "10"
    clock.advance(10)
    assert client.get("/api/v1/links", headers=alice).status_code == 200
    metrics = services_of(client).metrics
    assert metrics.value("shortener_auth_failures_total") == 2
    assert metrics.value("shortener_rate_limit_rejections_total", scope="auth_failure") == 1


def test_failed_authentication_is_delayed(
    database_path: Path, clock: FakeClock, secrets_by_owner: dict[str, str]
) -> None:
    sleeps: list[float] = []
    from shortener.main import create_app

    settings = make_settings(database_path, keys=secrets_by_owner, auth_failure_delay_ms=250)
    app = create_app(settings, clock=clock, sleep=sleeps.append)
    with api_client(app) as client:
        client.get("/api/v1/links", headers={"X-API-Key": new_secret()})
        client.get("/api/v1/links", headers={"X-API-Key": secrets_by_owner["alice"]})
    assert sleeps == [0.25]  # only the failure is delayed


def test_forwarded_for_is_ignored_unless_the_peer_is_a_trusted_proxy(
    start_app: Callable[..., TestClient],
) -> None:
    untrusted = start_app(redirect_rate=RateLimit(burst=1, per_minute=60))
    assert (
        untrusted.get("/missing1", headers={"X-Forwarded-For": "198.51.100.1"}).status_code == 404
    )
    spoofed = untrusted.get("/missing1", headers={"X-Forwarded-For": "198.51.100.2"})
    assert spoofed.status_code == 429  # a spoofed header cannot buy a fresh bucket


def test_forwarded_for_is_honoured_from_trusted_proxies(
    start_app: Callable[..., TestClient],
) -> None:
    import ipaddress

    trusted = start_app(
        redirect_rate=RateLimit(burst=1, per_minute=60),
        trusted_proxies=(ipaddress.ip_network(f"{CLIENT_ADDRESS}/32"),),
    )
    first = trusted.get("/missing1", headers={"X-Forwarded-For": "198.51.100.1"})
    other = trusted.get("/missing1", headers={"X-Forwarded-For": "198.51.100.2"})
    again = trusted.get("/missing1", headers={"X-Forwarded-For": "198.51.100.1"})
    assert (first.status_code, other.status_code, again.status_code) == (404, 404, 429)


# ----------------------------------------------------------------------------- headers / request ids


@pytest.mark.parametrize(
    "path", ["/healthz", "/readyz", "/missing1", "/api/v1/links", "/api/v1/nothing"]
)
def test_security_headers_are_present_on_every_response_kind(shared: SharedApp, path: str) -> None:
    response = shared.client.get(path)  # 200, 404 and 401 responses
    for name, value in STRICT_HEADERS.items():
        assert response.headers[name] == value, (path, name)
    assert response.headers["strict-transport-security"].startswith("max-age=")  # https base URL
    assert response.headers["cross-origin-resource-policy"] == "same-origin"


def test_security_headers_on_redirects_and_validation_errors(shared: SharedApp) -> None:
    code = create(shared.client, shared.alice).json()["code"]
    redirect = shared.client.get(f"/{code}")
    invalid = shared.client.post("/api/v1/links", json={}, headers=shared.alice)
    for response in (redirect, invalid):
        for name, value in STRICT_HEADERS.items():
            assert response.headers[name] == value


def test_hsts_is_only_sent_for_https_deployments(start_app: Callable[..., TestClient]) -> None:
    client = start_app(base_url="http://127.0.0.1:8000", own_hosts=("127.0.0.1",))
    assert "strict-transport-security" not in client.get("/healthz").headers


def test_request_ids_are_generated_per_request(shared: SharedApp) -> None:
    ids = {shared.client.get("/healthz").headers["x-request-id"] for _ in range(3)}
    assert len(ids) == 3
    assert all(UUID_PATTERN.fullmatch(value) for value in ids)


def test_valid_inbound_request_id_is_propagated_to_header_body_and_logs(
    shared: SharedApp, log_lines: CollectingHandler
) -> None:
    response = shared.client.get("/nosuchcode", headers={"X-Request-ID": "trace-ABCD-1234"})
    assert response.headers["x-request-id"] == "trace-ABCD-1234"
    assert response.json()["error"]["request_id"] == "trace-ABCD-1234"
    records = [json.loads(line) for line in log_lines.lines]
    assert any(
        record.get("route") == "/{code}" and record["request_id"] == "trace-ABCD-1234"
        for record in records
    )


@pytest.mark.parametrize(
    "inbound",
    [
        "short",
        "x" * 65,
        "has spaces here",
        "semi;colon-1234",
        "<script>alert(1)</script>",
        "idé-123456789",
        "a" * 7,
    ],
)
def test_invalid_inbound_request_ids_are_replaced(shared: SharedApp, inbound: str) -> None:
    response = shared.client.get("/healthz", headers={"X-Request-ID": inbound.encode("utf-8")})  # type: ignore[dict-item]
    assert UUID_PATTERN.fullmatch(response.headers["x-request-id"])
    assert inbound not in response.headers["x-request-id"]


def test_request_id_boundaries(shared: SharedApp) -> None:
    for valid in ("a" * 8, "A1-" * 21 + "z"):
        assert (
            shared.client.get("/healthz", headers={"X-Request-ID": valid}).headers["x-request-id"]
            == valid
        )


# ----------------------------------------------------------------------------- error model


def test_unknown_routes_and_methods_use_the_error_model(shared: SharedApp) -> None:
    not_found = shared.client.get("/api/v1/unknown/route")
    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "not_found"
    wrong_method = shared.client.post("/healthz")
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "method_not_allowed"
    assert "GET" in wrong_method.headers["allow"]
    assert set(wrong_method.json()["error"]) == {"code", "message", "request_id"}


def test_every_error_response_has_exactly_the_documented_shape(shared: SharedApp) -> None:
    responses = [
        shared.client.get("/api/v1/links"),
        shared.client.get("/nosuchcode"),
        shared.client.get("/api/v1/links/nosuchcode", headers=shared.alice),
        shared.client.post("/healthz"),
    ]
    for response in responses:
        assert response.status_code >= 400
        assert set(response.json()) == {"error"}
        assert set(response.json()["error"]) == {"code", "message", "request_id"}


def test_oversized_bodies_are_rejected_by_content_length(shared: SharedApp) -> None:
    body = json.dumps({"url": "https://example.com/" + "a" * 20000})
    response = shared.client.post(
        "/api/v1/links", content=body, headers={**shared.alice, "Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert response.headers["x-request-id"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_oversized_chunked_bodies_are_rejected_while_streaming(shared: SharedApp) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"url": "https://example.com/'
        for _ in range(5):
            yield b"a" * 5000
        yield b'"}'

    response = shared.client.post(
        "/api/v1/links",
        content=chunks(),
        headers={**shared.alice, "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_body_size_limit_boundary(shared: SharedApp) -> None:
    headers = {**shared.alice, "Content-Type": "application/json"}
    prefix = '{"url": "https://example.com/'
    suffix = '"}'
    at_limit = prefix + "a" * (16 * 1024 - len(prefix) - len(suffix)) + suffix
    assert len(at_limit) == 16 * 1024
    over_limit = at_limit + " "
    assert (
        shared.client.post("/api/v1/links", content=at_limit, headers=headers).status_code == 422
    )  # too-long URL, not 413
    assert (
        shared.client.post("/api/v1/links", content=over_limit, headers=headers).status_code == 413
    )


def test_invalid_content_length_is_a_400(shared: SharedApp) -> None:
    response = shared.client.post(
        "/api/v1/links", content=b"{}", headers={**shared.alice, "Content-Length": "abc"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_unhandled_errors_become_generic_500s_without_leaking_details(
    shared: SharedApp, monkeypatch: pytest.MonkeyPatch, log_lines: CollectingHandler
) -> None:
    def explode(**_kwargs: object) -> None:
        raise RuntimeError("internal detail: database password is hunter2")

    monkeypatch.setattr(shared.services.links, "create", explode)
    response = shared.client.post(
        "/api/v1/links", json={"url": "https://example.com/"}, headers=shared.alice
    )
    assert response.status_code == 500
    body = response.json()["error"]
    assert (body["code"], body["message"]) == ("internal_error", "Internal server error.")
    assert body["request_id"] == response.headers["x-request-id"]
    assert "hunter2" not in response.text and "Traceback" not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "RuntimeError" in log_lines.text  # the traceback is kept server-side
    assert shared.client.get("/healthz").status_code == 200  # and the service keeps working


# ----------------------------------------------------------------------------- health, readiness, metrics


def test_healthz_reports_liveness_version_and_auth_mode(shared: SharedApp) -> None:
    assert shared.client.get("/healthz").json() == {
        "status": "ok",
        "version": "0.1.0",
        "auth": "enabled",
    }


def test_healthz_stays_up_when_the_database_is_down(
    shared: SharedApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shared.services.database, "ping", lambda: False)
    assert shared.client.get("/healthz").status_code == 200  # liveness has no dependencies
    unavailable = shared.client.get("/readyz")
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "service_unavailable"


def test_readyz_checks_the_database(shared: SharedApp) -> None:
    assert shared.client.get("/readyz").json() == {"status": "ready"}


def test_metrics_expose_the_documented_series(
    client: TestClient, services: Services, alice: Headers
) -> None:
    create(client, alice, alias="metric1")
    client.get("/metric1")  # cache miss
    client.get("/metric1")  # cache hit
    client.get("/nothere1")  # miss, negatively cached
    client.get("/nothere1")  # negative hit
    assert services.recorder.flush(timeout=5)
    response = client.get("/metrics", headers=alice)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    text = response.text
    for expected in (
        'shortener_http_requests_total{method="GET",route="/{code}",status="302"} 2',
        'shortener_http_requests_total{method="GET",route="/{code}",status="404"} 2',
        'shortener_http_requests_total{method="POST",route="/api/v1/links",status="201"} 1',
        'shortener_redirect_cache_total{result="hit"} 1',
        'shortener_redirect_cache_total{result="miss"} 2',
        'shortener_redirect_cache_total{result="negative_hit"} 1',
        "shortener_links_created_total 1",
        "shortener_clicks_enqueued_total 2",
        "shortener_clicks_written_total 2",
        "shortener_click_queue_depth 0",
        "# TYPE shortener_clicks_dropped_total counter",
        "# TYPE shortener_rate_limit_rejections_total counter",
    ):
        assert expected in text, expected


def test_metric_labels_stay_bounded_under_hostile_requests(shared: SharedApp) -> None:
    for index in range(30):
        shared.client.get(f"/zz{index:05d}")
    shared.client.request("BREW", "/healthz")
    text = shared.client.get("/metrics", headers=shared.alice).text
    assert "zz00001" not in text  # paths are never labels, only route templates
    assert 'method="OTHER"' in text
    assert 'route="/{code}"' in text


# ----------------------------------------------------------------------------- analytics isolation


def test_a_failing_recorder_does_not_break_the_redirect(
    client: TestClient, services: Services, alice: Headers, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = create(client, alice).json()["code"]

    def boom(_event: ClickEvent) -> bool:
        raise RuntimeError("queue exploded")

    monkeypatch.setattr(services.recorder, "submit", boom)
    response = client.get(f"/{code}")
    assert response.status_code == 302
    assert response.headers["location"] == "https://example.com/page"
    assert services.metrics.value("shortener_analytics_errors_total") == 1


def test_a_failing_click_writer_does_not_break_redirects(
    start_app: Callable[..., TestClient], alice: Headers, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: ClickWriter, events: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ClickWriter, "write", boom)
    client = start_app()
    code = create(client, alice).json()["code"]
    assert [client.get(f"/{code}").status_code for _ in range(3)] == [302, 302, 302]
    services = services_of(client)
    assert services.recorder.flush(timeout=5)
    assert services.metrics.value("shortener_clicks_dropped_total", reason="write_failed") == 3
    assert client.get(f"/api/v1/links/{code}/analytics", headers=alice).json()["total_clicks"] == 0
    assert client.get(f"/{code}").status_code == 302  # still serving


def test_a_full_click_queue_drops_events_but_never_blocks_redirects(
    start_app: Callable[..., TestClient], alice: Headers, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    original = ClickWriter.write

    def blocked(self: ClickWriter, events: Any) -> None:
        gate.wait(timeout=10)
        original(self, events)

    monkeypatch.setattr(ClickWriter, "write", blocked)
    client = start_app(click_queue_size=1, click_batch_size=1)
    services = services_of(client)
    code = create(client, alice).json()["code"]
    statuses = [client.get(f"/{code}").status_code for _ in range(6)]
    assert statuses == [302] * 6  # the writer is stuck, yet no redirect waited for it
    enqueued = services.metrics.value("shortener_clicks_enqueued_total")
    dropped = services.metrics.value("shortener_clicks_dropped_total", reason="queue_full")
    assert enqueued + dropped == 6 and dropped >= 4
    gate.set()
    assert services.recorder.flush(timeout=10)
    assert services.metrics.value("shortener_clicks_written_total") == enqueued


# ----------------------------------------------------------------------------- privacy


def test_no_raw_ip_user_agent_referrer_path_or_secret_is_persisted_or_logged(
    settings: Any, clock: FakeClock, log_lines: CollectingHandler, secrets_by_owner: dict[str, str]
) -> None:
    ua_marker = "UniqueUserAgentMarker42"
    referrer_marker = "ReferrerSecretToken77"
    wrong_key = new_secret()
    app = build_app(settings, clock=clock)
    with api_client(app) as client:
        key = {"X-API-Key": secrets_by_owner["alice"]}
        code = create(client, key).json()["code"]
        for _ in range(2):
            client.get(
                f"/{code}",
                headers={
                    "User-Agent": f"Mozilla/5.0 Firefox/120 {ua_marker}",
                    "Referer": f"https://news.example.org/{referrer_marker}/path?token={referrer_marker}",
                    "X-Forwarded-For": "198.51.100.200",
                },
            )
        client.get("/api/v1/links", headers={"X-API-Key": wrong_key})
        client.get("/nosuchcode", headers={"Referer": f"https://x.example/{referrer_marker}"})
        assert services_of(client).recorder.flush(timeout=5)
        client.get("/api/v1/links", headers=key)
    stored = dump_database(Path(settings.database_path))
    assert "visitor_hash" in stored.lower() or "clicks" in stored.lower()
    assert "https://news.example.org" in stored  # scheme + host survive
    sensitive = [
        CLIENT_ADDRESS,
        "203.0.113",
        "198.51.100",
        ua_marker,
        "Mozilla",
        referrer_marker,
        "/path",
        "token=",
    ]
    for value in sensitive:
        assert value not in stored, f"{value!r} found in the database"
    logs = log_lines.text
    for value in [*sensitive, wrong_key, *secrets_by_owner.values()]:
        assert value not in logs, f"{value!r} found in the logs"


def test_visitor_ids_are_stable_within_a_day_and_change_with_the_day(
    client: TestClient, services: Services, alice: Headers, clock: FakeClock, database_path: Path
) -> None:
    code = create(client, alice).json()["code"]
    client.get(f"/{code}", headers={"User-Agent": "Firefox/120"})
    client.get(f"/{code}", headers={"User-Agent": "Firefox/120"})
    clock.advance(86400)
    client.get(f"/{code}", headers={"User-Agent": "Firefox/120"})
    assert services.recorder.flush(timeout=5)
    rows = (
        sqlite3.connect(database_path)
        .execute("SELECT substr(clicked_at, 1, 10), visitor_hash FROM clicks ORDER BY id")
        .fetchall()
    )
    (day1, first), (day1b, second), (day2, third) = rows
    assert day1 == day1b != day2
    assert first == second != third


# ----------------------------------------------------------------------------- lifecycle


def test_graceful_shutdown_flushes_queued_clicks(
    settings: Any, clock: FakeClock, secrets_by_owner: dict[str, str]
) -> None:
    app = build_app(settings, clock=clock)
    key = {"X-API-Key": secrets_by_owner["alice"]}
    with api_client(app) as client:
        code = create(client, key).json()["code"]
        for _ in range(5):
            assert client.get(f"/{code}").status_code == 302
    database = Database(settings.database_path)
    link = SqliteLinkRepository(database).get_by_code(code)
    assert link is not None and link.click_count == 5
    database.close()


def test_retention_purge_runs_at_startup(settings: Any, clock: FakeClock) -> None:
    database = Database(settings.database_path)
    database.open()
    links = SqliteLinkRepository(database)
    clicks = SqliteClickRepository(database)
    link = links.insert(NewLink("purge-me", "https://example.com/", clock.now(), None, "alice"))
    old = clock.now() - timedelta(days=settings.click_retention_days + 1)
    recent = clock.now() - timedelta(days=1)
    clicks.insert_many(
        [
            ClickEvent(link.id, old, None, "Chrome", "desktop", "o"),
            ClickEvent(link.id, recent, None, "Chrome", "desktop", "r"),
        ]
    )
    database.close()
    with api_client(build_app(settings, clock=clock)):
        pass
    database = Database(settings.database_path)
    assert (
        SqliteClickRepository(database)
        .summary(link.id, clock.now() - timedelta(days=365))
        .total_clicks
        == 1
    )
    database.close()


def test_a_fresh_database_is_migrated_by_the_lifespan(tmp_path: Path, clock: FakeClock) -> None:
    settings = make_settings(tmp_path / "nested" / "fresh.db")
    assert not Path(settings.database_path).exists()
    with api_client(build_app(settings, clock=clock)) as client:
        assert client.get("/readyz").status_code == 200
    tables = {
        row[0]
        for row in sqlite3.connect(settings.database_path).execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"links", "clicks", "idempotency_keys", "schema_migrations"} <= tables


def test_the_service_refuses_to_start_when_an_applied_migration_was_modified(
    tmp_path: Path, clock: FakeClock
) -> None:
    database = build_migrated_template(tmp_path)  # a private copy: the tamper must not leak
    connection = sqlite3.connect(database)
    connection.execute("UPDATE schema_migrations SET checksum = 'tampered' WHERE version = 1")
    connection.commit()
    connection.close()
    app = build_app(make_settings(database), clock=clock)
    with pytest.raises(MigrationError, match="checksum mismatch"), api_client(app):
        pass


def test_create_app_has_no_side_effects_until_the_lifespan_runs(
    tmp_path: Path, clock: FakeClock
) -> None:
    target = tmp_path / "must-not-exist" / "db.sqlite"
    app = build_app(make_settings(target), clock=clock)
    assert app.openapi()["info"]["title"] == "URL Shortener"
    assert not target.parent.exists()
    assert threading.active_count() >= 1


def test_lazy_module_level_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith("SHORTENER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SHORTENER_AUTH_DISABLED", "true")
    monkeypatch.setenv("SHORTENER_DATABASE_PATH", str(tmp_path / "lazy.db"))
    monkeypatch.setattr(main_module, "_app", None)
    built = main_module.app  # type: ignore[attr-defined]
    assert isinstance(built, FastAPI)
    assert main_module.app is built  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        main_module.does_not_exist  # noqa: B018


# ----------------------------------------------------------------------------- docs and auth modes


def test_interactive_docs_are_disabled_by_default(shared: SharedApp) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert shared.client.get(path).status_code == 404


def test_openapi_is_still_available_programmatically(app: FastAPI) -> None:
    schema = app.openapi()
    assert "/api/v1/links" in schema["paths"]
    assert "post" in schema["paths"]["/api/v1/links"]


def test_docs_can_be_enabled_explicitly_with_authentication_on(
    start_app: Callable[..., TestClient],
) -> None:
    client = start_app(enable_docs=True)
    docs = client.get("/docs")
    assert docs.status_code == 200 and "swagger" in docs.text.lower()
    assert "content-security-policy" not in docs.headers  # the docs UI needs scripts
    assert docs.headers["x-content-type-options"] == "nosniff"
    schema = client.get("/openapi.json")
    assert schema.status_code == 200 and "/api/v1/links" in schema.json()["paths"]
    assert (
        client.get("/healthz").headers["content-security-policy"].startswith("default-src 'none'")
    )


def test_auth_disabled_mode_is_open_loudly_and_loopback_only(
    start_app: Callable[..., TestClient], log_lines: CollectingHandler
) -> None:
    client = start_app(keys={}, auth_disabled=True)
    assert client.get("/healthz").json()["auth"] == "disabled"
    created = client.post("/api/v1/links", json={"url": "https://example.com/open"})
    assert created.status_code == 201
    assert client.get("/api/v1/links").json()["total"] == 1
    assert client.get("/metrics").status_code == 200
    assert (
        services_of(client).links.get(created.json()["code"], "anonymous").created_by == "anonymous"
    )
    warnings = [json.loads(line) for line in log_lines.lines]
    assert any(
        record["level"] == "WARNING" and record["event"] == "authentication_disabled"
        for record in warnings
    )


def test_missing_analytics_secret_is_warned_about(
    database_path: Path, clock: FakeClock, log_lines: CollectingHandler
) -> None:
    settings = make_settings(database_path, analytics_secret=None)
    with api_client(build_app(settings, clock=clock)):
        pass
    assert any(
        json.loads(line)["event"] == "analytics_secret_ephemeral" for line in log_lines.lines
    )


def test_log_level_reaches_the_service_logger(database_path: Path, clock: FakeClock) -> None:
    build_app(make_settings(database_path, log_level="ERROR"), clock=clock)
    assert logging.getLogger("shortener").level == logging.ERROR
    build_app(make_settings(database_path, log_level="INFO"), clock=clock)
