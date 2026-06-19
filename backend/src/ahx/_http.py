"""Shared async-HTTP plumbing: a per-instance, loop-aware AsyncClient cache.

THE embedding module and the LLM layer each hold ONE long-lived httpx.AsyncClient
so the serving process reuses connections (keep-alive) instead of paying a fresh
TLS handshake on every call — a single /ask is one query-embed + >=1 chat (deep
mode: many), each previously opening and tearing down its own client (and, inside
the retry loop, a brand-new client per attempt).

The client is built lazily on first use and rebound if the running event loop
changes: the offline CLI / eval harness runs one `asyncio.run` per process, so a
client cached on a since-closed loop must be replaced rather than reused (httpx
binds its pool to the loop of first use). On the serving path the loop is the
process's single uvicorn loop, so the same client serves every request.
"""

import asyncio

import httpx


class AsyncClientCache:
    """Owns one httpx.AsyncClient, lazily created and rebound across event loops."""

    def __init__(
        self,
        timeout: httpx.Timeout | float,
        transport: httpx.MockTransport | None = None,
    ) -> None:
        self._timeout = timeout
        self._transport = transport  # tests inject a fake server here
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> httpx.AsyncClient:
        """The shared client for the running loop. Must be called from within a
        running loop (every caller is an async method, so it always is)."""
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            # First use, or a new loop (CLI: one asyncio.run per process). A client
            # cached on a since-closed loop can't be reused; its sockets died with
            # that loop, so just drop the reference and build a fresh one here.
            self._client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport)
            self._loop = loop
        return self._client

    async def aclose(self) -> None:
        """Drain and close the pooled client (lifespan shutdown). Safe to call when
        no client was ever created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._loop = None
