"""LinkService: create/get/list/disable/resolve with cache, expiry, ownership, collisions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import timedelta

import pytest

from shortener.domain.codegen import BASE62, CodeGenerator
from shortener.domain.errors import (
    AliasTakenError,
    CodeGenerationError,
    InvalidAliasError,
    InvalidExpiryError,
    InvalidUrlError,
    LinkGoneError,
    LinkNotFoundError,
)
from shortener.domain.models import Link, LinkStatus, NewLink
from shortener.domain.validation import UrlPolicy
from shortener.observability.metrics import Metrics
from shortener.services.cache import LruTtlCache
from shortener.services.link_service import NOT_FOUND, LinkService, RedirectCache
from shortener.storage.db import Database
from shortener.storage.link_repository import SqliteLinkRepository
from tests.shortener.support import FakeClock

POLICY = UrlPolicy(own_hosts=frozenset({"sho.example.com"}))


class CountingRepository(SqliteLinkRepository):
    """Counts point lookups so tests can prove the cache is (not) consulted."""

    def __init__(self, database: Database) -> None:
        super().__init__(database)
        self.lookups = 0

    def get_by_code(self, code: str) -> Link | None:
        self.lookups += 1
        return super().get_by_code(code)


class Harness:
    def __init__(self, database: Database, clock: FakeClock, codes: CodeGenerator | None) -> None:
        self.repository = CountingRepository(database)
        self.metrics = Metrics()
        self.cache: RedirectCache = LruTtlCache(max_entries=100, default_ttl=60, clock=clock)
        self.service = LinkService(
            links=self.repository,
            codes=codes or CodeGenerator(),
            cache=self.cache,
            clock=clock,
            url_policy=POLICY,
            metrics=self.metrics,
            negative_ttl=5,
        )


@pytest.fixture
def database(migrated_database: Database) -> Database:
    return migrated_database


@pytest.fixture
def make_harness(database: Database, clock: FakeClock) -> Callable[..., Harness]:
    def build(codes: CodeGenerator | None = None) -> Harness:
        return Harness(database, clock, codes)

    return build


@pytest.fixture
def harness(make_harness: Callable[..., Harness]) -> Harness:
    return make_harness()


def create(
    harness: Harness, url: str = "https://example.com/a", owner: str = "alice", **kw: object
) -> Link:
    return harness.service.create(url=url, owner=owner, **kw)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- create


def test_create_generates_a_base62_code_and_canonical_url(
    harness: Harness, clock: FakeClock
) -> None:
    link = create(harness, "HTTPS://Example.COM:443/a b".replace(" ", "%20"))
    assert len(link.code) == 7
    assert set(link.code) <= set(BASE62)
    assert link.original_url == "https://example.com/a%20b"
    assert link.created_by == "alice"
    assert link.created_at == clock.now()
    assert (link.expires_at, link.disabled_at, link.click_count) == (None, None, 0)
    assert harness.repository.get_by_code(link.code) == link


def test_create_with_alias_and_ttl(harness: Harness, clock: FakeClock) -> None:
    link = create(harness, alias="spring-sale", ttl_seconds=3600)
    assert link.code == "spring-sale"
    assert link.expires_at == clock.now() + timedelta(hours=1)


def test_create_with_absolute_expiry(harness: Harness, clock: FakeClock) -> None:
    when = clock.now() + timedelta(days=2)
    assert create(harness, expires_at=when).expires_at == when


def test_duplicate_alias_is_a_conflict_and_does_not_modify_the_original(harness: Harness) -> None:
    create(harness, "https://example.com/first", alias="promo")
    with pytest.raises(AliasTakenError):
        create(harness, "https://example.com/second", alias="promo", owner="bob")
    assert harness.service.get("promo", "alice").original_url == "https://example.com/first"


def test_aliases_are_case_sensitive(harness: Harness) -> None:
    create(harness, alias="Promo")
    assert create(harness, alias="promo").code == "promo"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"url": "http://127.0.0.1/"}, InvalidUrlError),
        ({"url": "https://sho.example.com/loop"}, InvalidUrlError),
        ({"alias": "api"}, InvalidAliasError),
        ({"alias": "no spaces"}, InvalidAliasError),
        ({"ttl_seconds": 5}, InvalidExpiryError),
    ],
)
def test_invalid_input_creates_nothing(
    harness: Harness, kwargs: dict[str, object], error: type[Exception]
) -> None:
    arguments: dict[str, object] = {"url": "https://example.com/ok", "owner": "alice", **kwargs}
    with pytest.raises(error):
        harness.service.create(**arguments)  # type: ignore[arg-type]
    assert harness.repository.count_by_owner("alice") == 0


def test_mutually_exclusive_expiry_inputs(harness: Harness, clock: FakeClock) -> None:
    with pytest.raises(InvalidExpiryError):
        create(harness, ttl_seconds=3600, expires_at=clock.now() + timedelta(days=1))


def test_generated_code_collision_is_retried_with_a_new_candidate(
    make_harness: Callable[..., Harness],
) -> None:
    sources = iter("a" * 7 + "b" * 7)
    harness = make_harness(CodeGenerator(lambda _alphabet: next(sources)))
    create(harness, alias="aaaaaaa")
    assert create(harness).code == "bbbbbbb"


def test_length_is_bumped_after_three_collisions(make_harness: Callable[..., Harness]) -> None:
    harness = make_harness(CodeGenerator(lambda _alphabet: "a"))
    create(harness, alias="aaaaaaa")
    assert create(harness).code == "aaaaaaaa"  # 3 collisions at length 7, then length 8 succeeds


def test_code_space_exhaustion_fails_with_a_bounded_error(
    make_harness: Callable[..., Harness],
) -> None:
    calls = 0

    def always_a(_alphabet: str) -> str:
        nonlocal calls
        calls += 1
        return "a"

    harness = make_harness(CodeGenerator(always_a))
    create(harness, alias="aaaaaaa")
    create(harness, alias="aaaaaaaa")
    calls = 0
    with pytest.raises(CodeGenerationError) as excinfo:
        create(harness)
    assert excinfo.value.retry_after_seconds == 1
    assert calls == 3 * 7 + 2 * 8  # exactly five bounded attempts, no infinite loop


def test_creating_a_code_clears_a_stale_negative_cache_entry(harness: Harness) -> None:
    with pytest.raises(LinkNotFoundError):
        harness.service.resolve("promo")
    assert harness.cache.get("promo") is NOT_FOUND
    create(harness, alias="promo")
    assert harness.service.resolve("promo").url == "https://example.com/a"


# ----------------------------------------------------------------------------- get / list


def test_get_is_scoped_to_the_owner(harness: Harness) -> None:
    link = create(harness, owner="alice")
    assert harness.service.get(link.code, "alice") == link
    with pytest.raises(LinkNotFoundError):
        harness.service.get(link.code, "bob")  # indistinguishable from "does not exist"
    with pytest.raises(LinkNotFoundError):
        harness.service.get("missing", "alice")


def test_get_rejects_malformed_codes_without_touching_the_database(harness: Harness) -> None:
    for code in ("", "ab", "x" * 33, "a b", "../etc", "abc\n"):
        with pytest.raises(LinkNotFoundError):
            harness.service.get(code, "alice")
    assert harness.repository.lookups == 0


def test_list_returns_only_the_owners_links_newest_first(harness: Harness) -> None:
    mine = [create(harness, owner="alice") for _ in range(3)]
    create(harness, owner="bob")
    page = harness.service.list("alice", limit=2, offset=0)
    assert [link.code for link in page.items] == [mine[2].code, mine[1].code]
    assert (page.total, page.limit, page.offset) == (3, 2, 0)
    rest = harness.service.list("alice", limit=2, offset=2)
    assert [link.code for link in rest.items] == [mine[0].code]


# ----------------------------------------------------------------------------- resolve


def test_resolve_returns_target_and_caches_it(harness: Harness) -> None:
    link = create(harness)
    first = harness.service.resolve(link.code)
    second = harness.service.resolve(link.code)
    assert first == second
    assert (first.link_id, first.url) == (link.id, link.original_url)
    assert harness.repository.lookups == 1
    assert harness.metrics.value("shortener_redirect_cache_total", result="miss") == 1
    assert harness.metrics.value("shortener_redirect_cache_total", result="hit") == 1


def test_positive_cache_entries_expire_after_the_ttl(harness: Harness, clock: FakeClock) -> None:
    link = create(harness)
    harness.service.resolve(link.code)
    clock.advance(59)
    harness.service.resolve(link.code)
    assert harness.repository.lookups == 1
    clock.advance(1)
    harness.service.resolve(link.code)
    assert harness.repository.lookups == 2


def test_unknown_codes_are_negatively_cached_for_a_short_time(
    harness: Harness, clock: FakeClock
) -> None:
    for _ in range(3):
        with pytest.raises(LinkNotFoundError):
            harness.service.resolve("unknown1")
    assert harness.repository.lookups == 1
    assert harness.metrics.value("shortener_redirect_cache_total", result="negative_hit") == 2
    clock.advance(5)
    with pytest.raises(LinkNotFoundError):
        harness.service.resolve("unknown1")
    assert harness.repository.lookups == 2


def test_malformed_codes_never_reach_cache_or_database(harness: Harness) -> None:
    for code in ("x", "x" * 100, "favicon.ico", "a/b", "a b"):
        with pytest.raises(LinkNotFoundError):
            harness.service.resolve(code)
    assert harness.repository.lookups == 0
    assert len(harness.cache) == 0  # attacker-chosen junk cannot churn the cache


def test_expiry_is_enforced_even_for_cached_entries(harness: Harness, clock: FakeClock) -> None:
    link = create(harness, ttl_seconds=60)
    assert harness.service.resolve(link.code)
    clock.advance(59.9)
    assert harness.service.resolve(link.code)
    clock.advance(0.1)
    with pytest.raises(LinkGoneError) as excinfo:
        harness.service.resolve(link.code)
    assert excinfo.value.reason == LinkStatus.EXPIRED


def test_disable_revokes_immediately_even_when_cached(harness: Harness) -> None:
    link = create(harness)
    harness.service.resolve(link.code)  # populate the cache
    harness.service.disable(link.code, "alice")
    with pytest.raises(LinkGoneError) as excinfo:
        harness.service.resolve(link.code)
    assert excinfo.value.reason == LinkStatus.DISABLED


def test_disable_is_idempotent_and_owner_scoped(harness: Harness) -> None:
    link = create(harness)
    harness.service.disable(link.code, "alice")
    harness.service.disable(link.code, "alice")  # second call is fine
    with pytest.raises(LinkNotFoundError):
        harness.service.disable(link.code, "bob")
    with pytest.raises(LinkNotFoundError):
        harness.service.disable("missing", "alice")
    assert harness.service.status_of(harness.service.get(link.code, "alice")) == LinkStatus.DISABLED


def test_status_reflects_expiry_and_disable(harness: Harness, clock: FakeClock) -> None:
    link = create(harness, ttl_seconds=60)
    assert harness.service.status_of(link) == LinkStatus.ACTIVE
    clock.advance(60)
    assert harness.service.status_of(link) == LinkStatus.EXPIRED


def test_stored_urls_with_header_injection_are_never_served(
    harness: Harness, database: Database, clock: FakeClock
) -> None:
    """Defence in depth: even a tampered row cannot produce a split Location header."""
    SqliteLinkRepository(database).insert(
        NewLink("evil123", "https://example.com/\r\nSet-Cookie: x=1", clock.now(), None, "alice")
    )
    with pytest.raises(LinkNotFoundError):
        harness.service.resolve("evil123")


def test_sequence_return_types_are_stable(harness: Harness) -> None:
    create(harness)
    page = harness.service.list("alice", 10, 0)
    assert isinstance(page.items, Sequence)
