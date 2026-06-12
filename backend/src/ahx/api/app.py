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
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import ahx
from ahx.config import get_settings
from ahx.db import create_async_db_engine
from ahx.generation.pipeline import DeltaEvent, DoneEvent, Retriever, SourcesEvent, ask
from ahx.llm import ChatModel, chat_model_from_settings
from ahx.retrieval.dense import dense_retrieve_async
from ahx.retrieval.embedding import EmbeddingClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    engine = create_async_db_engine(settings.database_url)
    app.state.retriever = partial(dense_retrieve_async, engine, EmbeddingClient(settings))
    app.state.chat = chat_model_from_settings(settings)
    yield
    await engine.dispose()


app = FastAPI(title="Antic Historian API", version=ahx.__version__, lifespan=lifespan)


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_chat(request: Request) -> ChatModel:
    return request.app.state.chat


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
) -> EventSourceResponse:
    async def events() -> AsyncIterator[dict[str, str]]:
        async for event in ask(body.question, retriever, chat, top_k=body.top_k):
            yield {"event": _EVENT_NAMES[type(event)], "data": event.model_dump_json()}

    return EventSourceResponse(events())
