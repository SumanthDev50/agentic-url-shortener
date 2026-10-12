"""Analytics, idempotency, authentication, client address and maintenance services."""

from __future__ import annotations

import hmac
import ipaddress
import logging
import threading
from datetime import date, timedelta
from pathlib import Path

import pytest

from shortener.domain.errors import (
    AuthenticationError,
    IdempotencyKeyReuseError,
    InvalidInputError,
    RateLimitedError,
)
from shortener.domain.models import ClickEvent, NewLink
from shortener.observability.metrics import Metrics
from shortener.services.analytics_service import AnalyticsService
from shortener.services.auth import (
    ANONYMOUS,
    ApiKeyAuthenticator,
    AuthService,
    digest_api_key,
)
from shortener.services.client_ip import UNKNOWN_CLIENT, ClientIpResolver
from shortener.services.idempotency import (
    IDEMPOTENCY_TTL,
    IdempotencyService,
    request_fingerprint,
)
from shortener.services.maintenance import MaintenanceJob, MaintenanceRunner
from shortener.services.rate_limiter import TokenBucketLimiter
from shortener.storage.click_repository import SqliteClickRepository
from shortener.storage.db import Database
from shortener.storage.idempotency_repository import SqliteIdempotencyRepository
from shortener.storage.link_repository import SqliteLinkRepository
from tests.shortener.support import FakeClock, new_secret

PUBLIC_IP = "203.0.113.77"


@pytest.fixture
def database(migrated_database: Database) -> Database:
    return migrated_database


# ----------------------------------------------------------------------------- analytics


class StubRecorder:
    def __init__(self) -> None:
        self.events: list[ClickEvent] = []
        self.fail = False

    def submit(self, event: ClickEvent) -> bool:
        if self.fail:
            raise RuntimeError("queue exploded")
        self.events.append(event)
        return True


def make_analytics(
    database: Database,
    clock: FakeClock,
    recorder: StubRecorder,
    metrics: Metrics,
    retention: int = 90,
) -> AnalyticsService:
    return AnalyticsService(
        clicks=SqliteClickRepository(database),
        recorder=recorder,  # type: ignore[arg-type]
        clock=clock,
        server_secret=b"s" * 32,
        metrics=metrics,
        retention_days=retention,
    )


def test_record_click_reduces_the_request_to_privacy_safe_fields(
    database: Database, clock: FakeClock
) -> None:
    recorder = StubRecorder()
    service = make_analytics(database, clock, recorder, Metrics())
    service.record_click(
        link_id=7,
        client_ip=PUBLIC_IP,
        user_agent="Mozilla/5.0 (X11; Linux) Firefox/120.0 SECRET-TRACKING-TOKEN",
        referer="https://news.example.org/story?id=42&token=abc#frag",
    )
    (event,) = recorder.events
    assert (event.link_id, event.clicked_at) == (7, clock.now())
    assert event.referrer_host == "https://news.example.org"
    assert (event.browser, event.device) == ("Firefox", "desktop")
    assert len(event.visitor_hash) == 16
    serialised = repr(event)
    for sensitive in (PUBLIC_IP, "203.0.113", "SECRET-TRACKING-TOKEN", "token=abc", "story"):
        assert sensitive not in serialised


def test_record_click_never_raises_even_if_the_recorder_does(
    database: Database, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = StubRecorder()
    recorder.fail = True
    metrics = Metrics()
    service = make_analytics(database, clock, recorder, metrics)
    with caplog.at_level(logging.ERROR):
        service.record_click(link_id=1, client_ip=PUBLIC_IP, user_agent="x", referer=None)
    assert metrics.value("shortener_analytics_errors_total") == 1
    assert PUBLIC_IP not in caplog.text


def test_record_click_survives_garbage_headers(database: Database, clock: FakeClock) -> None:
    recorder = StubRecorder()
    service = make_analytics(database, clock, recorder, Metrics())
    service.record_click(
        link_id=1, client_ip="not-an-ip", user_agent="\x00" * 5000, referer="\ud800"
    )
    service.record_click(link_id=1, client_ip=None, user_agent=None, referer=None)
    assert len(recorder.events) == 2
    assert all(e.browser and e.device for e in recorder.events)


def test_report_is_windowed_zero_filled_and_aggregated(
    database: Database, clock: FakeClock
) -> None:
    links = SqliteLinkRepository(database)
    clicks = SqliteClickRepository(database)
    link = links.insert(NewLink("abc1234", "https://example.com/", clock.now(), None, "alice"))
    today = clock.now()
    clicks.insert_many(
        [
            ClickEvent(link.id, today, "https://a.example", "Chrome", "desktop", "v1"),
            ClickEvent(link.id, today, "https://a.example", "Chrome", "desktop", "v1"),
            ClickEvent(link.id, today - timedelta(days=2), None, "Firefox", "mobile", "v2"),
            ClickEvent(link.id, today - timedelta(days=10), None, "Safari", "mobile", "v3"),
        ]
    )
    service = make_analytics(database, clock, StubRecorder(), Metrics())
    report = service.report(link, days=3)
    assert report.days == 3
    assert [(d.day, d.clicks) for d in report.clicks_by_day] == [
        (date(2026, 1, 13), 1),
        (date(2026, 1, 14), 0),
        (date(2026, 1, 15), 2),
    ]
    assert (report.summary.total_clicks, report.summary.unique_visitors) == (3, 2)
    assert [(r.name, r.clicks) for r in report.top_referrers] == [("https://a.example", 2)]
    assert [(b.name, b.clicks) for b in report.browsers] == [("Chrome", 2), ("Firefox", 1)]
    assert [(d.name, d.clicks) for d in report.devices] == [("desktop", 2), ("mobile", 1)]
    wide = service.report(link, days=30)
    assert wide.summary.total_clicks == 4
    assert len(wide.clicks_by_day) == 30
    assert wide.clicks_by_day[-1].day == date(2026, 1, 15)


def test_report_for_a_link_without_clicks_is_all_zero(database: Database, clock: FakeClock) -> None:
    link = SqliteLinkRepository(database).insert(
        NewLink("empty12", "https://example.com/", clock.now(), None, "alice")
    )
    report = make_analytics(database, clock, StubRecorder(), Metrics()).report(link, days=2)
    assert [d.clicks for d in report.clicks_by_day] == [0, 0]
    assert report.summary.first_click_at is None and report.top_referrers == ()


@pytest.mark.parametrize("days", [0, -1, 366, 10**9])
def test_report_rejects_out_of_range_windows(
    database: Database, clock: FakeClock, days: int
) -> None:
    link = SqliteLinkRepository(database).insert(
        NewLink("range12", "https://example.com/", clock.now(), None, "alice")
    )
    with pytest.raises(InvalidInputError):
        make_analytics(database, clock, StubRecorder(), Metrics()).report(link, days=days)


def test_retention_purge_removes_only_expired_clicks(database: Database, clock: FakeClock) -> None:
    links = SqliteLinkRepository(database)
    clicks = SqliteClickRepository(database)
    link = links.insert(NewLink("purge12", "https://example.com/", clock.now(), None, "alice"))
    clicks.insert_many(
        [
            ClickEvent(link.id, clock.now() - timedelta(days=91), None, "Chrome", "desktop", "old"),
            ClickEvent(link.id, clock.now() - timedelta(days=89), None, "Chrome", "desktop", "new"),
        ]
    )
    service = make_analytics(database, clock, StubRecorder(), Metrics(), retention=90)
    assert service.purge_expired() == 1
    assert service.purge_expired() == 0


# ----------------------------------------------------------------------------- idempotency


def make_idempotency(
    database: Database, clock: FakeClock, metrics: Metrics | None = None
) -> IdempotencyService:
    return IdempotencyService(
        repository=SqliteIdempotencyRepository(database),
        transactions=database,
        clock=clock,
        metrics=metrics or Metrics(),
    )


def test_request_fingerprint_is_stable_and_order_independent() -> None:
    assert request_fingerprint({"a": 1, "b": [1, 2]}) == request_fingerprint({"b": [1, 2], "a": 1})
    assert request_fingerprint({"a": 1}) != request_fingerprint({"a": 2})
    assert len(request_fingerprint({})) == 64


def test_without_a_key_the_operation_always_runs(database: Database, clock: FakeClock) -> None:
    service = make_idempotency(database, clock)
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        return "{}"

    for _ in range(2):
        result = service.execute(key=None, owner="alice", request_hash="h", operation=operation)
        assert result.replayed is False
    assert len(calls) == 2


def test_same_key_and_payload_replays_the_stored_response(
    database: Database, clock: FakeClock
) -> None:
    metrics = Metrics()
    service = make_idempotency(database, clock, metrics)
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        return f'{{"call": {len(calls)}}}'

    first = service.execute(key="k1", owner="alice", request_hash="h", operation=operation)
    second = service.execute(key="k1", owner="alice", request_hash="h", operation=operation)
    assert (first.replayed, second.replayed) == (False, True)
    assert first.body == second.body == '{"call": 1}'
    assert len(calls) == 1
    assert metrics.value("shortener_idempotent_replays_total") == 1


def test_same_key_with_a_different_payload_is_rejected(
    database: Database, clock: FakeClock
) -> None:
    service = make_idempotency(database, clock)
    service.execute(key="k1", owner="alice", request_hash="h1", operation=lambda: "{}")
    with pytest.raises(IdempotencyKeyReuseError):
        service.execute(key="k1", owner="alice", request_hash="h2", operation=lambda: "{}")


def test_keys_are_scoped_per_owner(database: Database, clock: FakeClock) -> None:
    service = make_idempotency(database, clock)
    alice = service.execute(key="shared", owner="alice", request_hash="h", operation=lambda: '"a"')
    bob = service.execute(key="shared", owner="bob", request_hash="h", operation=lambda: '"b"')
    assert (alice.body, bob.body) == ('"a"', '"b"')
    assert bob.replayed is False


def test_keys_expire_after_24_hours(database: Database, clock: FakeClock) -> None:
    service = make_idempotency(database, clock)
    service.execute(key="k", owner="alice", request_hash="h1", operation=lambda: '"first"')
    clock.advance(IDEMPOTENCY_TTL.total_seconds() - 1)
    assert service.execute(
        key="k", owner="alice", request_hash="h1", operation=lambda: "x"
    ).replayed
    clock.advance(1)
    renewed = service.execute(
        key="k", owner="alice", request_hash="h2", operation=lambda: '"second"'
    )
    assert (renewed.replayed, renewed.body) == (False, '"second"')  # even with a new payload


def test_failed_operations_store_nothing_and_can_be_retried(
    database: Database, clock: FakeClock
) -> None:
    service = make_idempotency(database, clock)

    def failing() -> str:
        raise ValueError("business rule violated")

    with pytest.raises(ValueError, match="business rule"):
        service.execute(key="k", owner="alice", request_hash="h", operation=failing)
    retry = service.execute(key="k", owner="alice", request_hash="h", operation=lambda: '"ok"')
    assert (retry.replayed, retry.body) == (False, '"ok"')


def test_side_effects_roll_back_with_the_idempotency_record(
    database: Database, clock: FakeClock
) -> None:
    """The stored response and the operation's writes commit (or fail) together."""
    links = SqliteLinkRepository(database)
    service = make_idempotency(database, clock)

    def create_then_fail() -> str:
        links.insert(NewLink("ghost12", "https://example.com/", clock.now(), None, "alice"))
        raise RuntimeError("crash after insert")

    with pytest.raises(RuntimeError):
        service.execute(key="k", owner="alice", request_hash="h", operation=create_then_fail)
    assert links.get_by_code("ghost12") is None


def test_concurrent_requests_with_the_same_key_execute_the_operation_once(
    database_path: Path, clock: FakeClock
) -> None:
    path = database_path
    executions: list[int] = []
    bodies: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def worker() -> None:
        database = Database(path)
        try:
            service = make_idempotency(database, clock)
            barrier.wait()

            def operation() -> str:
                executions.append(1)
                return '{"created": true}'

            bodies.append(
                service.execute(key="k", owner="alice", request_hash="h", operation=operation).body
            )
        except BaseException as error:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(error)
        finally:
            database.close()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(executions) == 1
    assert bodies == ['{"created": true}'] * 6


def test_expired_keys_are_purged(database: Database, clock: FakeClock) -> None:
    service = make_idempotency(database, clock)
    service.execute(key="old", owner="alice", request_hash="h", operation=lambda: "{}")
    clock.advance(IDEMPOTENCY_TTL.total_seconds() + 1)
    service.execute(key="new", owner="alice", request_hash="h", operation=lambda: "{}")
    assert service.purge_expired() == 1


# ----------------------------------------------------------------------------- authentication


class Harness:
    def __init__(
        self,
        clock: FakeClock,
        *,
        disabled: bool = False,
        api_burst: int = 100,
        failure_burst: int = 3,
    ) -> None:
        self.secrets = {"alice": new_secret(), "bob": new_secret(), "carol": new_secret()}
        self.metrics = Metrics()
        self.sleeps: list[float] = []
        self.service = AuthService(
            authenticator=ApiKeyAuthenticator(
                [(owner, digest_api_key(secret)) for owner, secret in self.secrets.items()]
            ),
            disabled=disabled,
            api_limiter=TokenBucketLimiter(capacity=api_burst, refill_per_second=1.0, clock=clock),
            failure_limiter=TokenBucketLimiter(
                capacity=failure_burst, refill_per_second=1 / 6, clock=clock
            ),
            metrics=self.metrics,
            failure_delay_seconds=0.25,
            sleep=self.sleeps.append,
        )


def test_valid_keys_resolve_to_their_owner(clock: FakeClock) -> None:
    harness = Harness(clock)
    for owner, secret in harness.secrets.items():
        assert harness.service.authenticate(secret, PUBLIC_IP).principal.key_id == owner
    assert harness.sleeps == []


@pytest.mark.parametrize("presented", [None, "", "wrong", "x" * 300, "café", " "])
def test_invalid_keys_are_rejected_and_delayed(clock: FakeClock, presented: str | None) -> None:
    harness = Harness(clock)
    with pytest.raises(AuthenticationError):
        harness.service.authenticate(presented, PUBLIC_IP)
    assert harness.sleeps == [0.25]
    assert harness.metrics.value("shortener_auth_failures_total") == 1


def test_key_prefixes_and_suffixes_do_not_authenticate(clock: FakeClock) -> None:
    harness = Harness(clock, failure_burst=10)
    secret = harness.secrets["alice"]
    for presented in (secret[:-1], secret + "x", secret.upper(), f" {secret}"):
        with pytest.raises(AuthenticationError):
            harness.service.authenticate(presented, PUBLIC_IP)


def test_every_configured_key_is_compared_without_early_exit(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(clock)
    calls: list[int] = []
    real = hmac.compare_digest

    def counting(a: bytes, b: bytes) -> bool:
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr("shortener.services.auth.hmac.compare_digest", counting)
    harness.service.authenticate(harness.secrets["alice"], PUBLIC_IP)  # first configured key
    assert len(calls) == 3


def test_repeated_failures_lock_the_client_out_before_the_key_is_evaluated(
    clock: FakeClock,
) -> None:
    harness = Harness(clock, failure_burst=3)
    for _ in range(3):
        with pytest.raises(AuthenticationError):
            harness.service.authenticate("wrong", PUBLIC_IP)
    # Even the *correct* key is not evaluated while the failure bucket is empty.
    with pytest.raises(RateLimitedError) as excinfo:
        harness.service.authenticate(harness.secrets["alice"], PUBLIC_IP)
    assert excinfo.value.retry_after == 6
    assert harness.metrics.value("shortener_rate_limit_rejections_total", scope="auth_failure") == 1
    # Another client is unaffected, and the lockout ends as tokens refill.
    assert (
        harness.service.authenticate(harness.secrets["alice"], "198.51.100.9").principal.key_id
        == "alice"
    )
    clock.advance(6)
    assert (
        harness.service.authenticate(harness.secrets["alice"], PUBLIC_IP).principal.key_id
        == "alice"
    )


def test_successful_requests_are_rate_limited_per_key(clock: FakeClock) -> None:
    harness = Harness(clock, api_burst=2)
    secret = harness.secrets["alice"]
    harness.service.authenticate(secret, PUBLIC_IP)
    harness.service.authenticate(secret, "198.51.100.1")  # another IP, same key: same bucket
    with pytest.raises(RateLimitedError):
        harness.service.authenticate(secret, PUBLIC_IP)
    assert harness.service.authenticate(harness.secrets["bob"], PUBLIC_IP).principal.key_id == "bob"
    assert harness.metrics.value("shortener_rate_limit_rejections_total", scope="api") == 1


def test_disabled_authentication_yields_the_anonymous_principal(clock: FakeClock) -> None:
    harness = Harness(clock, disabled=True)
    assert harness.service.enabled is False
    assert harness.service.authenticate(None, PUBLIC_IP).principal == ANONYMOUS
    assert harness.service.authenticate("anything", PUBLIC_IP).principal == ANONYMOUS


def test_digest_is_sha256_and_secrets_are_not_retained() -> None:
    authenticator = ApiKeyAuthenticator([("k", digest_api_key("s" * 40))])
    assert len(digest_api_key("s")) == 32
    assert "s" * 40 not in repr(vars(authenticator))


# ----------------------------------------------------------------------------- client address


def resolver(*networks: str) -> ClientIpResolver:
    return ClientIpResolver([ipaddress.ip_network(n) for n in networks])


def test_peer_address_is_authoritative_without_trusted_proxies() -> None:
    assert resolver().resolve(PUBLIC_IP, "6.6.6.6") == PUBLIC_IP


def test_forwarded_for_is_ignored_from_untrusted_peers() -> None:
    assert resolver("10.0.0.0/8").resolve(PUBLIC_IP, "6.6.6.6") == PUBLIC_IP


def test_trusted_proxy_hands_over_the_client_address() -> None:
    assert resolver("10.0.0.0/8").resolve("10.0.0.5", "198.51.100.9") == "198.51.100.9"


def test_rightmost_untrusted_hop_wins_and_spoofed_prefixes_are_ignored() -> None:
    chain = "6.6.6.6, 198.51.100.9, 10.0.0.7"  # the client lied about the leftmost entry
    assert resolver("10.0.0.0/8").resolve("10.0.0.5", chain) == "198.51.100.9"


def test_all_trusted_chain_falls_back_to_the_last_trusted_hop() -> None:
    assert resolver("10.0.0.0/8").resolve("10.0.0.5", "10.0.0.7, 10.0.0.8") == "10.0.0.7"


def test_malformed_hop_stops_trusting_the_chain() -> None:
    assert resolver("10.0.0.0/8").resolve("10.0.0.5", "6.6.6.6, garbage, 10.0.0.7") == "10.0.0.7"
    assert resolver("10.0.0.0/8").resolve("10.0.0.5", "garbage") == "10.0.0.5"


def test_unknown_or_missing_peer_and_ipv6() -> None:
    assert resolver().resolve(None) == UNKNOWN_CLIENT
    assert resolver().resolve("testclient") == UNKNOWN_CLIENT
    assert resolver().resolve("::ffff:203.0.113.5") == "203.0.113.5"
    assert resolver("fd00::/8").resolve("fd00::1", "2001:db8::5") == "2001:db8::5"
    assert resolver("10.0.0.0/8").resolve("fd00::1", "6.6.6.6") == "fd00::1"  # families differ


def test_forwarded_chain_length_is_bounded() -> None:
    chain = ", ".join(["10.0.0.9"] * 1000 + ["6.6.6.6"])
    assert resolver("10.0.0.0/8").resolve("10.0.0.5", chain) == "6.6.6.6"


# ----------------------------------------------------------------------------- maintenance


def test_run_once_reports_results_and_isolates_failures(caplog: pytest.LogCaptureFixture) -> None:
    def boom() -> int:
        raise RuntimeError("db gone")

    runner = MaintenanceRunner([MaintenanceJob("ok", lambda: 3), MaintenanceJob("bad", boom)])
    with caplog.at_level(logging.ERROR):
        assert runner.run_once() == {"ok": 3, "bad": -1}
    assert "maintenance_job_failed" in caplog.text


def test_background_loop_runs_jobs_repeatedly_until_stopped() -> None:
    runs = threading.Semaphore(0)
    counter: list[int] = []

    def job() -> int:
        counter.append(1)
        runs.release()
        return 0

    runner = MaintenanceRunner([MaintenanceJob("tick", job)], interval_seconds=0.001)
    runner.start()
    runner.start()  # idempotent
    try:
        for _ in range(3):  # the startup run plus at least two periodic runs
            assert runs.acquire(timeout=5)
    finally:
        runner.stop()
    stopped_at = len(counter)
    assert stopped_at >= 3
    runner.stop()  # safe twice


def test_start_runs_the_jobs_synchronously_first() -> None:
    seen: list[str] = []
    runner = MaintenanceRunner(
        [MaintenanceJob("job", lambda: seen.append("ran") or 0)], interval_seconds=3600
    )
    runner.start()
    assert seen == ["ran"]
    runner.stop()
