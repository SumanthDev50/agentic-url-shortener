"""Token bucket: refill, bounded state, peek, headers."""

from __future__ import annotations

import pytest

from shortener.domain.errors import RateLimitedError
from shortener.services.rate_limiter import TokenBucketLimiter
from tests.shortener.support import FakeClock


def limiter(
    clock: FakeClock, *, capacity: int = 3, rate: float = 1.0, max_entries: int = 100
) -> TokenBucketLimiter:
    return TokenBucketLimiter(
        capacity=capacity, refill_per_second=rate, clock=clock, max_entries=max_entries
    )


def test_burst_is_allowed_then_rejected_with_retry_after(clock: FakeClock) -> None:
    bucket = limiter(clock, capacity=3, rate=1.0)
    decisions = [bucket.acquire("client") for _ in range(4)]
    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert [d.remaining for d in decisions] == [2, 1, 0, 0]
    assert decisions[3].retry_after == 1
    assert decisions[3].limit == 3


def test_tokens_refill_over_time_up_to_capacity(clock: FakeClock) -> None:
    bucket = limiter(clock, capacity=3, rate=1.0)
    for _ in range(3):
        bucket.acquire("client")
    assert not bucket.acquire("client").allowed
    clock.advance(1)
    assert bucket.acquire("client").allowed
    assert not bucket.acquire("client").allowed
    clock.advance(1000)  # never accumulates beyond the capacity
    assert [bucket.acquire("client").allowed for _ in range(4)] == [True, True, True, False]


def test_fractional_refill_rate(clock: FakeClock) -> None:
    bucket = limiter(clock, capacity=1, rate=0.5)  # one token every two seconds
    assert bucket.acquire("client").allowed
    rejected = bucket.acquire("client")
    assert not rejected.allowed
    assert rejected.retry_after == 2
    clock.advance(1.9)
    assert not bucket.acquire("client").allowed
    clock.advance(0.1)
    assert bucket.acquire("client").allowed


def test_keys_are_independent(clock: FakeClock) -> None:
    bucket = limiter(clock, capacity=1)
    assert bucket.acquire("a").allowed
    assert not bucket.acquire("a").allowed
    assert bucket.acquire("b").allowed


def test_peek_does_not_consume_or_create_state(clock: FakeClock) -> None:
    bucket = limiter(clock, capacity=1)
    for _ in range(5):
        assert bucket.peek("client").allowed
    assert len(bucket) == 0
    assert bucket.acquire("client").allowed
    assert not bucket.peek("client").allowed
    assert not bucket.peek("client").allowed
    clock.advance(1)
    assert bucket.peek("client").allowed


def test_state_is_bounded_under_a_key_flood(clock: FakeClock) -> None:
    bucket = limiter(clock, max_entries=50)
    for index in range(10_000):
        bucket.acquire(f"attacker-{index}")
        assert len(bucket) <= 50
    assert len(bucket) == 50


def test_least_recently_used_bucket_is_evicted_first(clock: FakeClock) -> None:
    bucket = limiter(clock, capacity=1, max_entries=2)
    bucket.acquire("a")
    bucket.acquire("b")
    bucket.acquire("a")  # "a" is now more recent than "b"
    bucket.acquire("c")  # evicts "b"
    assert not bucket.peek("a").allowed  # "a" kept its (empty) bucket
    assert bucket.peek("b").allowed  # "b" was evicted -> fresh


def test_decision_headers_and_error(clock: FakeClock) -> None:
    bucket = limiter(clock, capacity=2, rate=1.0)
    allowed = bucket.acquire("c")
    assert allowed.headers() == {
        "X-RateLimit-Limit": "2",
        "X-RateLimit-Remaining": "1",
        "X-RateLimit-Reset": "1",
    }
    bucket.acquire("c")
    rejected = bucket.acquire("c")
    assert rejected.headers()["Retry-After"] == "1"
    assert rejected.headers()["X-RateLimit-Remaining"] == "0"
    error = rejected.to_error()
    assert isinstance(error, RateLimitedError)
    assert (error.limit, error.remaining, error.retry_after) == (2, 0, 1)


@pytest.mark.parametrize("kwargs", [{"capacity": 0}, {"refill_per_second": 0}, {"max_entries": 0}])
def test_invalid_configuration(kwargs: dict[str, float], clock: FakeClock) -> None:
    arguments: dict[str, float] = {"capacity": 1, "refill_per_second": 1.0, "max_entries": 1}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match="must be positive"):
        TokenBucketLimiter(
            capacity=int(arguments["capacity"]),
            refill_per_second=arguments["refill_per_second"],
            clock=clock,
            max_entries=int(arguments["max_entries"]),
        )
