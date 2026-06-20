"""RateLimiter — IP window + per-session cap (6.4).

Pure-logic tests with an injected clock: window expiry is exercised by advancing a fake
clock, never by sleeping. The FastAPI 429 mapping is covered at the route level in
test_api.py; here we pin the budget arithmetic and the "reject without consuming" rule.
"""

import pytest

from ahx.api.limits import RateLimited, RateLimiter, SessionStatus


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_session_cap_decrements_then_exhausts() -> None:
    limiter = RateLimiter(per_window=0, window_seconds=60, session_cap=2)

    assert await limiter.consume(ip="1.1.1.1", session_id="s") == SessionStatus(
        limit=2, remaining=1
    )
    assert await limiter.consume(ip="1.1.1.1", session_id="s") == SessionStatus(
        limit=2, remaining=0
    )
    with pytest.raises(RateLimited) as exc:
        await limiter.consume(ip="1.1.1.1", session_id="s")
    assert exc.value.reason == "session_cap_reached"
    assert exc.value.limit == 2


async def test_ip_window_blocks_then_recovers() -> None:
    clock = FakeClock()
    limiter = RateLimiter(per_window=2, window_seconds=60, session_cap=0, clock=clock)

    await limiter.consume(ip="1.1.1.1", session_id="a")
    await limiter.consume(ip="1.1.1.1", session_id="b")
    with pytest.raises(RateLimited) as exc:
        await limiter.consume(ip="1.1.1.1", session_id="c")
    assert exc.value.reason == "rate_limited"
    assert exc.value.retry_after is not None and exc.value.retry_after >= 1

    clock.now = 61.0  # both timestamps fall out of the 60s window
    assert await limiter.consume(ip="1.1.1.1", session_id="d") == SessionStatus(
        limit=0, remaining=0
    )


async def test_ip_window_is_per_ip() -> None:
    limiter = RateLimiter(per_window=1, window_seconds=60, session_cap=0)
    await limiter.consume(ip="1.1.1.1", session_id="a")
    # a different IP has its own budget
    await limiter.consume(ip="2.2.2.2", session_id="a")
    with pytest.raises(RateLimited):
        await limiter.consume(ip="1.1.1.1", session_id="a")


async def test_rejected_ip_request_does_not_consume_session() -> None:
    # cap=5, IP window=1: the second call is IP-blocked and must NOT burn a session query.
    clock = FakeClock()
    limiter = RateLimiter(per_window=1, window_seconds=60, session_cap=5, clock=clock)
    first = await limiter.consume(ip="1.1.1.1", session_id="s")
    assert first.remaining == 4
    with pytest.raises(RateLimited):
        await limiter.consume(ip="1.1.1.1", session_id="s")
    clock.now = 61.0  # window passes; the only successful consume so far was the first
    third = await limiter.consume(ip="1.1.1.1", session_id="s")
    assert third.remaining == 3  # 5 - 2 successful consumes, not 5 - 3


async def test_both_limits_off_never_rejects() -> None:
    limiter = RateLimiter(per_window=0, window_seconds=60, session_cap=0)
    for _ in range(100):
        assert await limiter.consume(ip="1.1.1.1", session_id="s") == SessionStatus(
            limit=0, remaining=0
        )


async def test_daily_cap_is_global_across_ips_and_sessions() -> None:
    # The daily ceiling is a single global counter, so rotating IP *and* session id
    # cannot dodge it (unlike the per-IP window / per-session cap).
    limiter = RateLimiter(per_window=0, window_seconds=60, session_cap=0, daily_cap=3)
    for i in range(3):
        await limiter.consume(ip=f"10.0.0.{i}", session_id=f"s{i}")
    with pytest.raises(RateLimited) as exc:
        await limiter.consume(ip="10.0.0.99", session_id="s99")
    assert exc.value.reason == "daily_cap_reached"
    assert exc.value.retry_after is not None and exc.value.retry_after >= 1


async def test_daily_cap_recovers_after_24h() -> None:
    clock = FakeClock()
    limiter = RateLimiter(per_window=0, window_seconds=60, session_cap=0, daily_cap=2, clock=clock)
    await limiter.consume(ip="1.1.1.1", session_id="a")
    await limiter.consume(ip="1.1.1.1", session_id="b")
    with pytest.raises(RateLimited):
        await limiter.consume(ip="1.1.1.1", session_id="c")
    clock.now = 86_401.0  # both requests fall out of the rolling 24h window
    assert await limiter.consume(ip="1.1.1.1", session_id="d") == SessionStatus(
        limit=0, remaining=0
    )


async def test_daily_cap_rejection_does_not_consume_other_budgets() -> None:
    # A daily-capped request must not burn an IP slot or a session query — the same
    # "reject without consuming" contract the other limits honour.
    clock = FakeClock()
    limiter = RateLimiter(per_window=5, window_seconds=60, session_cap=5, daily_cap=1, clock=clock)
    first = await limiter.consume(ip="1.1.1.1", session_id="s")
    assert first.remaining == 4  # one session query consumed
    with pytest.raises(RateLimited) as exc:
        await limiter.consume(ip="1.1.1.1", session_id="s")  # blocked by the daily cap
    assert exc.value.reason == "daily_cap_reached"
    clock.now = 86_401.0  # daily window clears; the IP/session budgets were untouched
    third = await limiter.consume(ip="1.1.1.1", session_id="s")
    assert third.remaining == 3  # 5 - 2 successful consumes (not 5 - 3)


async def test_session_store_is_lru_bounded() -> None:
    # An attacker rotating X-Session-Id can't grow the store without limit: past
    # max_tracked, the least-recently-seen entry is evicted (memory-exhaustion guard).
    limiter = RateLimiter(per_window=0, window_seconds=60, session_cap=5, max_tracked=3)
    for i in range(10):
        await limiter.consume(ip="1.1.1.1", session_id=f"s{i}")
    assert len(limiter._session_used) == 3  # pyright: ignore[reportPrivateUsage]
    # Only the 3 most recent session ids survive.
    assert set(limiter._session_used) == {"s7", "s8", "s9"}  # pyright: ignore[reportPrivateUsage]


async def test_ip_store_is_lru_bounded() -> None:
    limiter = RateLimiter(per_window=2, window_seconds=60, session_cap=0, max_tracked=3)
    for i in range(10):
        await limiter.consume(ip=f"10.0.0.{i}", session_id="s")
    assert len(limiter._hits) == 3  # pyright: ignore[reportPrivateUsage]
    assert set(limiter._hits) == {"10.0.0.7", "10.0.0.8", "10.0.0.9"}  # pyright: ignore[reportPrivateUsage]
