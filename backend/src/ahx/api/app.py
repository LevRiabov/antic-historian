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
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from langfuse import Langfuse

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import ahx
from ahx.config import get_settings
from ahx.db import create_async_db_engine
from ahx.generation.pipeline import DeltaEvent, DoneEvent, Retriever, SourcesEvent, ask
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
    app.state.chat = chat
    app.state.langfuse = langfuse
    yield
    if langfuse is not None:
        langfuse.flush()  # drain the export buffer before the process exits
    await engine.dispose()


app = FastAPI(title="Antic Historian API", version=ahx.__version__, lifespan=lifespan)


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_chat(request: Request) -> ChatModel:
    return request.app.state.chat


def get_langfuse(request: Request) -> "Langfuse | None":
    # getattr default: when lifespan hasn't run (e.g. ASGITransport tests), the
    # attribute is absent — that just means tracing is off, not an error.
    return getattr(request.app.state, "langfuse", None)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": ahx.__version__}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


_EVENT_NAMES = {SourcesEvent: "sources", DeltaEvent: "delta", DoneEvent: "done"}


@app.post("/ask")
async def ask_route(
    body: AskRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
    chat: Annotated[ChatModel, Depends(get_chat)],
    langfuse: Annotated["Langfuse | None", Depends(get_langfuse)],
) -> EventSourceResponse:
    async def events() -> AsyncIterator[dict[str, str]]:
        # The whole stream runs inside the root span so retrieve/chat nest under
        # it; the trace is finished with the answer on the terminal DoneEvent.
        async with trace_request(langfuse, question=body.question, top_k=body.top_k) as trace:
            async for event in ask(body.question, retriever, chat, top_k=body.top_k):
                if isinstance(event, DoneEvent):
                    trace.finish(
                        answer=event.answer,
                        refused=event.refused,
                        usage=event.usage,
                        cost=event.cost,
                    )
                yield {"event": _EVENT_NAMES[type(event)], "data": event.model_dump_json()}

    return EventSourceResponse(events())
