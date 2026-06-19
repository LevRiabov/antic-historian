"""Rate limiting + per-session caps (6.4) — abuse + free-tier protection.

A public, abusable demo where every query is hosted spend (phase-6-plan §cost-of-a-query)
needs two distinct limits:
- **IP sliding window** — abuse protection. Max N requests per IP per window; over it →
  a structured 429 + Retry-After.
- **Per-session lifetime cap** — free-tier protection, keyed on a client-sent X-Session-Id
  header. Exhausted → a structured 429; while allowed, every answer carries "N of M left"
  (the route emits it as a `meta` SSE event).

In-memory + single-instance by design (fine for a free-tier deploy); the store sits behind
`RateLimiter` so a Redis-backed variant (with TTLs, the natural fix for unbounded growth) is
a drop-in. The limiter is pure (injectable clock) so windows are testable without sleeping;
the FastAPI glue (`enforce_limits`) translates a rejection into an HTTP 429.

Session identity is the client's header, not security: an attacker rotates it freely — the IP
window is the backstop for gross abuse, the session cap just protects the honest free tier.
"""

import asyncio
import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import NoReturn, TypeVar

from fastapi import HTTPException, Request
from pydantic import BaseModel

from ahx.config import Settings, get_settings

_V = TypeVar("_V")

# Cap on distinct IPs / session ids tracked in memory. The store is keyed by client
# IP and a freely-rotatable X-Session-Id, so without a bound an attacker rotating
# either grows it without limit (a slow memory-exhaustion DoS). Past the cap the
# least-recently-seen entry is evicted (LRU): an evicted session simply gets its
# free-tier count reset — acceptable, since the IP window is the real abuse backstop
# and the session id was never a security boundary (see module docstring). Generous
# enough that legitimate traffic is never evicted in practice. Redis with TTLs is the
# documented scale path that removes the cap entirely.
_MAX_TRACKED = 100_000


class SessionStatus(BaseModel):
    """Surfaced to the client as the `meta` SSE event. `limit == 0` means uncapped
    (don't render a badge); otherwise `remaining` is the queries left in this session."""

    limit: int
    remaining: int


class RateLimited(Exception):
    """A request to reject. Framework-free (the dependency maps it to a 429) so the
    limiter stays unit-testable without FastAPI."""

    def __init__(
        self,
        *,
        reason: str,
        message: str,
        retry_after: int | None = None,
        limit: int = 0,
        remaining: int = 0,
    ) -> None:
        super().__init__(message)
        self.reason = reason  # "rate_limited" | "session_cap_reached"
        self.message = message
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


class RateLimiter:
    """In-memory IP window + per-session cap. One instance per process, shared across
    requests; an asyncio.Lock serializes the read-modify-write so concurrent /ask calls
    can't both slip past a limit. Both limits disabled when their setting is 0."""

    def __init__(
        self,
        *,
        per_window: int,
        window_seconds: int,
        session_cap: int,
        clock: Callable[[], float] = time.monotonic,
        max_tracked: int = _MAX_TRACKED,
    ) -> None:
        self._per_window = per_window
        self._window = window_seconds
        self._cap = session_cap
        self._clock = clock
        self._max_tracked = max_tracked
        # OrderedDicts (not plain dicts) so we can evict the least-recently-touched
        # entry once either store exceeds max_tracked — bounding memory under abuse.
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()  # ip -> recent timestamps
        self._session_used: OrderedDict[str, int] = OrderedDict()  # session id -> lifetime count
        self._lock = asyncio.Lock()

    def _bound(self, store: OrderedDict[str, _V]) -> None:
        """Evict least-recently-touched entries until the store fits max_tracked."""
        while len(store) > self._max_tracked:
            store.popitem(last=False)

    def _ip_retry_after(self, ip: str, now: float) -> int | None:
        """None if the IP is under its window; else whole seconds until a slot frees.
        Prunes timestamps older than the window as a side effect (bounds the deque)."""
        if self._per_window <= 0:
            return None
        q = self._hits.get(ip)
        if q is None:
            q = deque[float]()
            self._hits[ip] = q
            self._bound(self._hits)  # a fresh IP may push the store over the cap
        else:
            self._hits.move_to_end(ip)  # touched -> most-recently-seen (LRU)
        cutoff = now - self._window
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= self._per_window:
            return max(1, math.ceil(self._window - (now - q[0])))
        return None

    async def consume(self, *, ip: str, session_id: str) -> SessionStatus:
        """Check both limits, then consume one unit of each. Raises RateLimited (without
        consuming anything) when either limit is hit, so a rejected request never burns a
        slot or a session query."""
        async with self._lock:
            now = self._clock()

            retry = self._ip_retry_after(ip, now)
            if retry is not None:
                raise RateLimited(
                    reason="rate_limited",
                    message=f"Too many requests; retry in {retry}s.",
                    retry_after=retry,
                )

            used = self._session_used.get(session_id, 0)
            if self._cap > 0 and used >= self._cap:
                raise RateLimited(
                    reason="session_cap_reached",
                    message=f"Session query cap reached ({self._cap}).",
                    limit=self._cap,
                    remaining=0,
                )

            if self._per_window > 0:
                self._hits[ip].append(now)  # entry created + LRU-touched in _ip_retry_after
            if self._cap > 0:
                self._session_used[session_id] = used + 1
                self._session_used.move_to_end(session_id)  # most-recently-seen (LRU)
                self._bound(self._session_used)
                return SessionStatus(limit=self._cap, remaining=self._cap - used - 1)
            return SessionStatus(limit=0, remaining=0)


def limiter_from_settings(settings: Settings) -> RateLimiter:
    return RateLimiter(
        per_window=settings.rate_limit_per_window,
        window_seconds=settings.rate_limit_window_seconds,
        session_cap=settings.session_query_cap,
    )


def client_ip(request: Request, trust_forwarded_for: bool) -> str:
    """The real client IP. Behind a proxy (prod) it's the first X-Forwarded-For hop;
    direct, it's request.client. request.client is None under ASGITransport tests."""
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def session_key(request: Request, ip: str) -> str:
    """The X-Session-Id header, or the IP when absent — so a header-less client is still
    capped (it can't bypass the cap by simply omitting the header)."""
    return request.headers.get("x-session-id") or ip


def _raise_429(err: RateLimited) -> NoReturn:
    headers = {"Retry-After": str(err.retry_after)} if err.retry_after is not None else None
    raise HTTPException(
        status_code=429,
        detail={
            "error": err.reason,
            "message": err.message,
            "limit": err.limit,
            "remaining": err.remaining,
        },
        headers=headers,
    )


async def enforce_limits(request: Request) -> SessionStatus:
    """Route dependency: consume the IP + session budgets or 429. Returns the session
    status so the route can emit "N of M left". A no-op (uncapped status) when the limiter
    isn't on app.state — e.g. ASGITransport tests that skip lifespan."""
    limiter: RateLimiter | None = getattr(request.app.state, "limiter", None)
    if limiter is None:
        return SessionStatus(limit=0, remaining=0)
    settings = get_settings()
    ip = client_ip(request, settings.trust_forwarded_for)
    try:
        return await limiter.consume(ip=ip, session_id=session_key(request, ip))
    except RateLimited as err:
        _raise_429(err)
