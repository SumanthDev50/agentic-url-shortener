"""LRU+TTL cache driven by a fake clock."""

from __future__ import annotations

import threading

import pytest

from shortener.services.cache import LruTtlCache
from tests.shortener.support import FakeClock


def make_cache(clock: FakeClock, *, max_entries: int = 3, ttl: float = 10) -> LruTtlCache[str, int]:
    return LruTtlCache(max_entries=max_entries, default_ttl=ttl, clock=clock)


def test_get_returns_stored_value_until_ttl_expires(clock: FakeClock) -> None:
    cache = make_cache(clock)
    cache.put("a", 1)
    clock.advance(9.999)
    assert cache.get("a") == 1
    clock.advance(0.001)
    assert cache.get("a") is None
    assert len(cache) == 0  # expired entries are dropped on access


def test_per_entry_ttl_overrides_the_default(clock: FakeClock) -> None:
    cache = make_cache(clock)
    cache.put("negative", 0, ttl=2)
    cache.put("positive", 1)
    clock.advance(2)
    assert cache.get("negative") is None
    assert cache.get("positive") == 1


def test_put_refreshes_ttl_and_value(clock: FakeClock) -> None:
    cache = make_cache(clock)
    cache.put("a", 1)
    clock.advance(8)
    cache.put("a", 2)
    clock.advance(8)
    assert cache.get("a") == 2


def test_least_recently_used_entry_is_evicted(clock: FakeClock) -> None:
    cache = make_cache(clock, max_entries=3)
    for index, key in enumerate("abc"):
        cache.put(key, index)
    assert cache.get("a") == 0  # touching "a" makes "b" the least recently used
    cache.put("d", 3)
    assert cache.get("b") is None
    assert [cache.get(key) for key in "acd"] == [0, 2, 3]
    assert len(cache) == 3


def test_size_never_exceeds_the_bound(clock: FakeClock) -> None:
    cache = make_cache(clock, max_entries=5)
    for index in range(100):
        cache.put(f"key-{index}", index)
        assert len(cache) <= 5


def test_invalidate_and_clear(clock: FakeClock) -> None:
    cache = make_cache(clock)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.invalidate("a")
    cache.invalidate("missing")  # no error
    assert cache.get("a") is None
    assert cache.get("b") == 2
    cache.clear()
    assert len(cache) == 0


def test_zero_is_a_valid_cached_value(clock: FakeClock) -> None:
    cache = make_cache(clock)
    cache.put("zero", 0)
    assert cache.get("zero") == 0


@pytest.mark.parametrize("kwargs", [{"max_entries": 0}, {"default_ttl": 0}, {"default_ttl": -1}])
def test_invalid_configuration(kwargs: dict[str, float], clock: FakeClock) -> None:
    arguments: dict[str, float] = {"max_entries": 1, "default_ttl": 1, **kwargs}
    with pytest.raises(ValueError, match="must be positive"):
        LruTtlCache(clock=clock, **arguments)  # type: ignore[arg-type]


def test_concurrent_use_keeps_the_bound(clock: FakeClock) -> None:
    cache = make_cache(clock, max_entries=50)

    def worker(offset: int) -> None:
        for index in range(500):
            cache.put(f"{offset}-{index % 80}", index)
            cache.get(f"{offset}-{(index * 7) % 80}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(cache) <= 50
