"""FastAPI application: streaming cited answers over SSE.

Async end-to-end (rule #7): async embed -> async pgvector retrieve -> async
LLM stream. Event order is sources -> delta* -> done, so a client can render
the source panel before the first answer token arrives (module-10 UX).

Resources live on app.state via lifespan (engine = connection pool, created
once per process); routes receive them through Depends so tests can swap in
fakes with app.dependency_overrides — no server, DB, or LLM needed.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, Annotated, Literal

if TYPE_CHECKING:
    from langfuse import Langfuse

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import ahx
from ahx.agent.prompts import AGENT_PROMPT_VERSION
from ahx.agent.runner import AgentStreamer, make_agent_streamer
from ahx.api.evals import (
    load_latest_agent_run,
    load_latest_rag_run,
    load_latest_security_baseline,
    load_latest_security_defended,
)
from ahx.api.limits import SessionStatus, enforce_limits, limiter_from_settings
from ahx.api.sources import SourceOut, SourcesProvider, list_sources_async
from ahx.config import get_settings
from ahx.db import create_async_db_engine
from ahx.evals.generation import GenerationRun
from ahx.evals.retrieval import RetrievalRun
from ahx.evals.security import SecurityRun
from ahx.generation.pipeline import DeltaEvent, DoneEvent, Retriever, SourcesEvent, StepEvent
from ahx.guard import DefenseConfig, guard_config_from_settings, guard_stream, guarded_events
from ahx.llm import ChatModel, chat_model_from_settings
from ahx.obs import init_langfuse, trace_request, traced_chat, traced_retriever
from ahx.retrieval.dense import dense_retrieve_async
from ahx.retrieval.embedding import EmbeddingClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    engine = create_async_db_engine(settings.database_url)
    retriever: Retriever = partial(dense_retrieve_async, engine, EmbeddingClient(settings))
    chat = chat_model_from_settings(settings)

    # Tracing seam (6.1): wrap chat + retriever only when Langfuse is configured;
    # otherwise the raw objects flow through and the API behaves identically.
    langfuse = init_langfuse(settings)
    if langfuse is not None:
        chat = traced_chat(chat, langfuse)
        retriever = traced_retriever(retriever, langfuse)

    app.state.retriever = retriever
    app.state.sources = partial(list_sources_async, engine)  # /sources corpus listing (Phase 7)
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
    await engine.dispose()


app = FastAPI(title="Antic Historian API", version=ahx.__version__, lifespan=lifespan)


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_sources(request: Request) -> SourcesProvider | None:
    # None when lifespan hasn't run (ASGITransport tests override this) — the route
    # then 503s rather than touching a DB that was never wired up.
    return getattr(request.app.state, "sources", None)


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
    return {"status": "ok", "version": ahx.__version__}


@app.get("/sources")
async def sources_route(
    sources: Annotated[SourcesProvider | None, Depends(get_sources)],
) -> list[SourceOut]:
    """The full corpus the demo answers over — one entry per loaded work, with its
    public-domain basis, publisher, and passage count. Read-only; no spend (Phase 7)."""
    if sources is None:
        raise HTTPException(status_code=503, detail="sources are not available")
    return await sources()


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


_EVENT_NAMES = {SourcesEvent: "sources", DeltaEvent: "delta", DoneEvent: "done", StepEvent: "step"}


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

    async def events() -> AsyncIterator[dict[str, str]]:
        # `meta` first: the client renders "N of M left" before the answer streams (6.4).
        yield {"event": "meta", "data": session.model_dump_json()}
        # The whole stream runs inside the root span so retrieve/chat nest under it; the
        # trace is finished with the answer on the terminal DoneEvent. A blocked request
        # still yields a well-formed envelope without touching retrieval or the model (6.3).
        async with trace_request(langfuse, question=body.question, top_k=body.top_k) as trace:
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
                yield {"event": _EVENT_NAMES[type(event)], "data": event.model_dump_json()}

    return EventSourceResponse(events())
