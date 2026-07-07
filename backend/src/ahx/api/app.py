"""FastAPI application: streaming cited answers over SSE.

Async end-to-end (rule #7): async embed -> async pgvector retrieve -> async
LLM stream. Event order is sources -> delta* -> done, so a client can render
the source panel before the first answer token arrives (module-10 UX).

Resources live on app.state via lifespan (engine = connection pool, created
once per process); routes receive them through Depends so tests can swap in
fakes with app.dependency_overrides — no server, DB, or LLM needed.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal

if TYPE_CHECKING:
    from langfuse import Langfuse

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import ahx
from ahx.agent.prompts import AGENT_PROMPT_VERSION
from ahx.agent.runner import AgentStreamer, make_agent_streamer
from ahx.api.chunks import ChunkOut, ChunksProvider, get_chunks_async
from ahx.api.evals import (
    load_latest_agent_run,
    load_latest_rag_run,
    load_latest_security_baseline,
    load_latest_security_defended,
)
from ahx.api.limits import SessionStatus, enforce_limits, limiter_from_settings
from ahx.api.sources import SourceOut, SourcesProvider, list_sources_async
from ahx.config import get_settings, validate_serving_config
from ahx.db import create_async_db_engine
from ahx.evals.generation import GenerationRun
from ahx.evals.retrieval import RetrievalRun
from ahx.evals.security import SecurityRun
from ahx.generation.pipeline import (
    DeltaEvent,
    DoneEvent,
    ReasoningEvent,
    Retriever,
    SourcesEvent,
    StepEvent,
)
from ahx.guard import DefenseConfig, guard_config_from_settings, guard_stream, guarded_events
from ahx.llm import ChatModel, aclose_chat_model, chat_model_from_settings
from ahx.obs import init_langfuse, trace_request, traced_chat, traced_retriever
from ahx.retrieval.dense import dense_retrieve_async
from ahx.retrieval.embedding import EmbeddingClient

logger = logging.getLogger("ahx.api")


class SecurityHeadersMiddleware:
    """Add baseline hardening headers to every response.

    Pure-ASGI on purpose: Starlette's BaseHTTPMiddleware buffers the response body,
    which would break the /ask SSE stream. This only rewrites the response-start
    headers and leaves the (streamed) body untouched. nginx may set these too in
    prod (ADR-004); duplicating them here keeps the API safe if hit directly."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "no-referrer"
            await send(message)

        await self.app(scope, receive, send_with_headers)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    # Fail loud BEFORE serving if a hosted endpoint is missing its key (would 401 on
    # the first real query while booting green); log soft deployment warnings.
    for warning in validate_serving_config(settings):
        logger.warning("serving config: %s", warning)
    engine = create_async_db_engine(settings.database_url)
    embedder = EmbeddingClient(settings)  # closed on shutdown to drain its keep-alive pool
    retriever: Retriever = partial(dense_retrieve_async, engine, embedder)
    chat = chat_model_from_settings(settings)

    # Tracing seam (6.1): wrap chat + retriever only when Langfuse is configured;
    # otherwise the raw objects flow through and the API behaves identically.
    langfuse = init_langfuse(settings)
    if langfuse is not None:
        chat = traced_chat(chat, langfuse)
        retriever = traced_retriever(retriever, langfuse)

    app.state.engine = engine  # exposed for the /ready DB liveness probe
    app.state.retriever = retriever
    app.state.sources = partial(list_sources_async, engine)  # /sources corpus listing (Phase 7)
    # /chunks passage lookup (Phase 7): turns a cited chunk id back into a readable
    # passage for the citation drawer (the eval records carry only ids).
    app.state.chunks = partial(get_chunks_async, engine)
    # Published eval runs (Phase 7): the latest -rag / -agent records, loaded once
    # (frozen artifacts). None when the runs dir has no such record -> route 503s.
    app.state.eval_rag = load_latest_rag_run(settings.eval_runs_dir)
    app.state.eval_agent = load_latest_agent_run(settings.eval_runs_dir)
    # Security audit (Phase 7): the latest baseline + defended runs, paired by attack id
    # on the page. None when no such record exists -> route 503s.
    app.state.security_baseline = load_latest_security_baseline(settings.security_runs_dir)
    app.state.security_defended = load_latest_security_defended(settings.security_runs_dir)
    app.state.chat = chat  # CompositeChatModel when AHX_CHAT_FALLBACKS is set (6.4)
    app.state.langfuse = langfuse
    app.state.guard = guard_config_from_settings(settings)  # 6.3 security stack
    app.state.canary = settings.prompt_canary
    app.state.limiter = limiter_from_settings(settings)  # 6.4 rate limit + session cap
    # Deep mode (6.7): build the agent ONCE over the shared engine + the traced/composite
    # chat, so its loop steps are traced and fall over across providers like single-shot.
    app.state.agent_streamer = make_agent_streamer(
        settings, engine, chat, settings.agent_retriever, settings.agent_max_steps
    )
    yield
    if langfuse is not None:
        langfuse.flush()  # drain the export buffer before the process exits
    # Close the keep-alive HTTP pools (chat + query-embed) before the loop shuts down,
    # then the DB pool. aclose_chat_model handles the traced/composite wrapping; the
    # agent streamer's own embedder is process-lifetime and released at exit.
    await aclose_chat_model(chat)
    await embedder.aclose()
    await engine.dispose()


app = FastAPI(title="Antique Historian API", version=ahx.__version__, lifespan=lifespan)
app.add_middleware(SecurityHeadersMiddleware)


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_engine(request: Request) -> AsyncEngine | None:
    # None when lifespan hasn't run (ASGITransport tests) — /ready then reports 503.
    return getattr(request.app.state, "engine", None)


def get_sources(request: Request) -> SourcesProvider | None:
    # None when lifespan hasn't run (ASGITransport tests override this) — the route
    # then 503s rather than touching a DB that was never wired up.
    return getattr(request.app.state, "sources", None)


def get_chunks(request: Request) -> ChunksProvider | None:
    # None when lifespan hasn't run (ASGITransport tests override this) — the route 503s.
    return getattr(request.app.state, "chunks", None)


def get_eval_rag(request: Request) -> RetrievalRun | None:
    return getattr(request.app.state, "eval_rag", None)


def get_eval_agent(request: Request) -> GenerationRun | None:
    return getattr(request.app.state, "eval_agent", None)


def get_security_baseline(request: Request) -> SecurityRun | None:
    return getattr(request.app.state, "security_baseline", None)


def get_security_defended(request: Request) -> SecurityRun | None:
    return getattr(request.app.state, "security_defended", None)


def get_chat(request: Request) -> ChatModel:
    return request.app.state.chat


def get_langfuse(request: Request) -> "Langfuse | None":
    # getattr default: when lifespan hasn't run (e.g. ASGITransport tests), the
    # attribute is absent — that just means tracing is off, not an error.
    return getattr(request.app.state, "langfuse", None)


def get_guard(request: Request) -> DefenseConfig:
    # Default to an all-off config when lifespan hasn't run (ASGITransport tests):
    # the guard is then a no-op and the route behaves like the raw pipeline.
    return getattr(request.app.state, "guard", DefenseConfig())


def get_canary(request: Request) -> str:
    return getattr(request.app.state, "canary", "")


def get_agent_streamer(request: Request) -> AgentStreamer | None:
    # None when lifespan hasn't run (ASGITransport tests override this) — deep mode is
    # then unavailable and the route returns a 503 rather than pretending to serve it.
    return getattr(request.app.state, "agent_streamer", None)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up. Cheap, no dependencies — keep it that way so a
    dead DB doesn't also fail liveness and trigger a pointless restart loop."""
    return {"status": "ok", "version": ahx.__version__}


@app.get("/ready")
async def ready(engine: Annotated[AsyncEngine | None, Depends(get_engine)]) -> dict[str, str]:
    """Readiness: the instance can actually serve a query (DB reachable). An
    orchestrator routes traffic on this, not /health — otherwise an instance with a
    dead pool stays 'healthy' while every /ask 500s."""
    if engine is None:
        raise HTTPException(status_code=503, detail="not ready: no database engine")
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception as exc:  # pool exhausted / DB down / network — not ready to serve
        logger.warning("readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="not ready: database unreachable") from exc
    return {"status": "ready", "version": ahx.__version__}


@app.get("/sources")
async def sources_route(
    sources: Annotated[SourcesProvider | None, Depends(get_sources)],
) -> list[SourceOut]:
    """The full corpus the demo answers over — one entry per loaded work, with its
    public-domain basis, publisher, and passage count. Read-only; no spend (Phase 7)."""
    if sources is None:
        raise HTTPException(status_code=503, detail="sources are not available")
    return await sources()


@app.get("/chunks")
async def chunks_route(
    chunks: Annotated[ChunksProvider | None, Depends(get_chunks)],
    ids: Annotated[list[int], Query(min_length=1, max_length=50)],
) -> list[ChunkOut]:
    """Fetch corpus passages by chunk id — the readable, verifiable text behind a
    cited marker (the eval records store only ids). Read-only; no spend (Phase 7)."""
    if chunks is None:
        raise HTTPException(status_code=503, detail="chunks are not available")
    return await chunks(ids)


@app.get("/evals/rag")
async def evals_rag_route(
    run: Annotated[RetrievalRun | None, Depends(get_eval_rag)],
) -> RetrievalRun:
    """The latest retrieval-tier eval (recall@k, MRR per question) — the published
    `-rag` run. Pairs with /evals/agent by question_id for the golden page (Phase 7)."""
    if run is None:
        raise HTTPException(status_code=503, detail="no retrieval eval run available")
    return run


@app.get("/evals/agent")
async def evals_agent_route(
    run: Annotated[GenerationRun | None, Depends(get_eval_agent)],
) -> GenerationRun:
    """The latest generation-tier eval (answer + faithfulness/completeness/attribution/
    refusal per question) — the published `-agent` run (Phase 7)."""
    if run is None:
        raise HTTPException(status_code=503, detail="no generation eval run available")
    return run


@app.get("/evals/security/baseline")
async def security_baseline_route(
    run: Annotated[SecurityRun | None, Depends(get_security_baseline)],
) -> SecurityRun:
    """The latest UNDEFENDED security audit (attack success rate per attack, no defense) —
    the published `-baseline` run. Pairs with /evals/security/defended by attack id (Phase 7)."""
    if run is None:
        raise HTTPException(status_code=503, detail="no baseline security run available")
    return run


@app.get("/evals/security/defended")
async def security_defended_route(
    run: Annotated[SecurityRun | None, Depends(get_security_defended)],
) -> SecurityRun:
    """The latest DEFENDED security audit (same attacks, defense stack on) — the published
    `-defended` run. The before/after of the production defense stack (Phase 7)."""
    if run is None:
        raise HTTPException(status_code=503, detail="no defended security run available")
    return run


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    # "fast" = single-shot default (cheap, public-safe); "deep" = the streamed agent
    # ("watch it search", 6.7). top_k is ignored in deep mode (the agent sizes its own
    # searches). Default fast so a public demo never lands on the expensive path by accident.
    mode: Literal["fast", "deep"] = "fast"


_EVENT_NAMES = {
    SourcesEvent: "sources",
    DeltaEvent: "delta",
    ReasoningEvent: "reasoning",
    DoneEvent: "done",
    StepEvent: "step",
}


@app.post("/ask")
async def ask_route(
    body: AskRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    chat: Annotated[ChatModel, Depends(get_chat)],
    langfuse: Annotated["Langfuse | None", Depends(get_langfuse)],
    guard: Annotated[DefenseConfig, Depends(get_guard)],
    canary: Annotated[str, Depends(get_canary)],
    session: Annotated[SessionStatus, Depends(enforce_limits)],
    agent_streamer: Annotated[AgentStreamer | None, Depends(get_agent_streamer)],
) -> EventSourceResponse:
    # enforce_limits runs as a dependency: a rate-limit / session-cap rejection raises a
    # structured 429 BEFORE the SSE stream opens (no model spend on a rejected request).
    if body.mode == "deep" and agent_streamer is None:
        # Fail loudly rather than silently downgrade to fast — the client asked for deep.
        raise HTTPException(status_code=503, detail="deep mode is not available")

    # Select the event source: deep wraps the agent stream, fast wraps single-shot — both
    # through the SAME guard (input blocklist + output validation), labelled by their prompt.
    if body.mode == "deep":
        assert agent_streamer is not None  # narrowed by the guard above
        source = guard_stream(
            body.question, agent_streamer(body.question), guard, canary, AGENT_PROMPT_VERSION
        )
    else:
        source = guarded_events(body.question, retriever, chat, guard, canary, top_k=body.top_k)

    timeout_s = get_settings().request_timeout_seconds

    async def events() -> AsyncIterator[dict[str, str]]:
        # `meta` first: the client renders "N of M left" before the answer streams (6.4).
        yield {"event": "meta", "data": session.model_dump_json()}
        # The whole stream runs inside the root span so retrieve/chat nest under it; the
        # trace is finished with the answer on the terminal DoneEvent. A blocked request
        # still yields a well-formed envelope without touching retrieval or the model (6.3).
        async with trace_request(langfuse, question=body.question, top_k=body.top_k) as trace:
            try:
                # Overall wall-clock cap: a stalled upstream otherwise pins this
                # connection (and its pool slot) for the LLM read timeout x N steps.
                async with asyncio.timeout(timeout_s):
                    async for event in source:
                        if isinstance(event, DoneEvent):
                            trace.finish(
                                answer=event.answer,
                                refused=event.refused,
                                usage=event.usage,
                                cost=event.cost,
                                blocked=event.blocked,
                                served_by=event.served_by,
                            )
                        name = _EVENT_NAMES.get(type(event))
                        if name is None:  # unknown event type: skip, never crash mid-stream
                            logger.warning("ask: skipping unknown event %s", type(event).__name__)
                            continue
                        yield {"event": name, "data": event.model_dump_json()}
            except (
                Exception
            ):  # NOT CancelledError (BaseException) — client disconnect still cancels
                # The 200 + SSE headers already shipped, so this can't become an HTTP
                # error. Emit a terminal `error` frame (timeout lands here too) so the
                # client stops waiting on a truncated stream instead of hanging.
                logger.exception("ask: stream failed (mode=%s)", body.mode)
                yield {"event": "error", "data": json.dumps({"detail": "the answer stream failed"})}

    return EventSourceResponse(events())
