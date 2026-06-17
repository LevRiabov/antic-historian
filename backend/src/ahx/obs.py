# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Observability seam (Phase 6.1) — Langfuse tracing, isolated to this module.

ALL Langfuse specifics live here. Business logic (pipeline, llm, retrieval)
never imports langfuse: instead we wrap the two thin seams it already exposes —
the `ChatModel` Protocol and the `Retriever` callable — in tracing decorators
that are shape-identical to what they wrap (ADR-001 thin-waist). The API
composes the wrappers in its lifespan; everything else is unaffected, so the
eval harness and CLI run untraced exactly as before.

Tracing is OPT-IN: `init_langfuse` returns None unless host + both keys are
set, and every wrapper degrades to a transparent pass-through when given None.
A missing/misconfigured Langfuse must never change what the API returns.

Async safety (rule #7): the Langfuse calls here only build spans in memory;
export happens on a background OTEL thread, so nothing blocks the event loop.
`flush()` is blocking but runs once at lifespan shutdown, not per request.
"""

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any

from langfuse import Langfuse

from ahx.config import Settings
from ahx.generation.pipeline import Retriever
from ahx.llm import ChatMessage, ChatModel, StreamEnd, StreamEvent, Usage
from ahx.pricing import Cost
from ahx.retrieval.dense import RetrievedChunk


def init_langfuse(settings: Settings) -> Langfuse | None:
    """The configured client, or None when tracing is not set up. None keeps
    the whole feature dormant — the wrappers below pass straight through."""
    if not (
        settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key
    ):
        return None
    return Langfuse(
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )


def _usage_details(usage: Usage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {"input": usage.prompt_tokens, "output": usage.completion_tokens}


class TracedChatModel:
    """Wraps a ChatModel; emits one Langfuse `generation` per call with the
    model name, the messages as input, and the answer + token usage as output.
    Structurally a ChatModel itself, so it drops into the same injection slot."""

    def __init__(self, inner: ChatModel, langfuse: Langfuse) -> None:
        self._inner = inner
        self._lf = langfuse

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def _input(self, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [m.model_dump() for m in messages]

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        # A sync `with` is correct around an async generator: it sets the OTEL
        # context on enter and resets on exit, and contextvars survive `await`,
        # so the generation stays the current span across the whole stream.
        with self._lf.start_as_current_observation(
            as_type="generation",
            name="chat.stream",
            model=self._inner.model_name,
            input=self._input(messages),
        ) as gen:
            pieces: list[str] = []
            usage: Usage | None = None
            async for event in self._inner.stream(messages):
                if isinstance(event, StreamEnd):
                    usage = event.usage
                else:
                    pieces.append(event.text)
                yield event
            gen.update(output="".join(pieces), usage_details=_usage_details(usage))

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        with self._lf.start_as_current_observation(
            as_type="generation",
            name="chat.complete",
            model=self._inner.model_name,
            input=self._input(messages),
        ) as gen:
            result = await self._inner.complete(messages, response_format)
            gen.update(output=result.text, usage_details=_usage_details(result.usage))
            return result


def traced_chat(chat: ChatModel, langfuse: Langfuse) -> ChatModel:
    """A ChatModel that traces every call. Separate constructor (vs using the
    class directly) keeps the wiring in app.py symmetric with traced_retriever."""
    return TracedChatModel(chat, langfuse)


def traced_retriever(retriever: Retriever, langfuse: Langfuse) -> Retriever:
    """Wrap a Retriever so each call is a `retrieve` span recording the query,
    hit count, and the ranked chunk ids + top score (the forensic minimum)."""

    async def _traced(query: str, top_k: int) -> list[RetrievedChunk]:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieve",
            input={"query": query, "top_k": top_k},
        ) as span:
            chunks = await retriever(query, top_k)
            span.update(
                output={
                    "count": len(chunks),
                    "top_score": chunks[0].score if chunks else None,
                    "chunk_ids": [c.chunk_id for c in chunks],
                }
            )
            return chunks

    return _traced


class RequestTrace:
    """Handle yielded by `trace_request`: lets the route stamp the final answer
    onto the root span (which Langfuse surfaces as the trace's output)."""

    def __init__(self, span: Any | None) -> None:
        self._span = span

    @property
    def trace_id(self) -> str | None:
        """The Langfuse trace id for this request — None when tracing is off. Lets an eval
        record carry the id (6.1) so a failed question links straight to its trace. SDK v4
        exposes `trace_id` on the span object; getattr keeps it robust across SDK versions."""
        return getattr(self._span, "trace_id", None) if self._span is not None else None

    def finish(
        self,
        *,
        answer: str,
        refused: bool,
        usage: Usage | None,
        cost: Cost | None = None,
        blocked: bool = False,
        served_by: str | None = None,
    ) -> None:
        if self._span is None:
            return
        self._span.update(
            output=answer,
            metadata={
                "refused": refused,
                "blocked": blocked,  # 6.3: a security block is visible in the trace
                "served_by": served_by,  # 6.4: which model answered (fallback-aware)
                "completion_tokens": usage.completion_tokens if usage else None,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "cost_usd": cost.usd if cost else None,
                "cost_priced": cost.priced if cost else None,
            },
        )


@contextlib.asynccontextmanager
async def trace_request(
    langfuse: Langfuse | None,
    *,
    question: str,
    top_k: int,
    name: str = "ask",
    metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[RequestTrace]:
    """One root span per request. Child spans (retrieve, chat) created inside the
    `with` block nest under it via OTEL context. The root span's name/input/output
    stand in for trace-level fields (version-robust — no reliance on trace-mutation
    helpers that move between SDK releases). `name`/`metadata` let the eval label a
    trace by question id + category (the API uses the defaults).

    No-op when tracing is off, so the caller reads identically either way."""
    if langfuse is None:
        yield RequestTrace(None)
        return
    with langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        input={"question": question, "top_k": top_k},
        metadata=metadata,
    ) as span:
        yield RequestTrace(span)
