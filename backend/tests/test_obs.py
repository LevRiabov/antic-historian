"""Tracing-seam tests (Phase 6.1). The load-bearing invariant: wrapping a
ChatModel / Retriever in tracing must NOT change what they return — a trace
backend outage can never alter an answer. We assert pass-through, plus that the
generation span is fed the accumulated text + token usage.

A fake Langfuse stands in for the SDK: start_as_current_observation is a context
manager recording (kind, name) and returning a span that captures update() args.
No network, no real client."""

from collections.abc import AsyncIterator, Generator, Sequence
from contextlib import contextmanager
from typing import Any

from pydantic import SecretStr

from ahx.config import Settings
from ahx.llm import ChatMessage, ChatResult, StreamEnd, StreamEvent, TextDelta, Usage
from ahx.obs import init_langfuse, trace_request, traced_chat, traced_retriever
from ahx.retrieval.dense import RetrievedChunk


class FakeSpan:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.trace_id = "trace-fake"  # SDK v4 exposes this; RequestTrace.trace_id reads it

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeLangfuse:
    def __init__(self) -> None:
        self.observations: list[tuple[str, str]] = []
        self.spans: list[FakeSpan] = []

    @contextmanager
    def start_as_current_observation(
        self, *, as_type: str, name: str, **_: Any
    ) -> Generator[FakeSpan]:
        self.observations.append((as_type, name))
        span = FakeSpan()
        self.spans.append(span)
        yield span


class FakeChat:
    model_name = "fake-model"

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text="Hello ")
        yield TextDelta(text="world.")
        yield StreamEnd(usage=Usage(prompt_tokens=11, completion_tokens=3))

    async def complete(
        self, messages: Sequence[ChatMessage], response_format: dict[str, Any] | None = None
    ) -> ChatResult:
        return ChatResult(text="done", usage=Usage(prompt_tokens=5, completion_tokens=2))


def _chunk(rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=100 + rank,
        pg_id=1,
        author="Sallust",
        work_title="Catiline",
        locator=["1"],
        text="text",
        score=0.9 - rank * 0.1,
        char_start=0,
        char_end=10,
        rank=rank,
    )


async def test_init_langfuse_opt_in() -> None:
    # Missing any of host/keys -> dormant (None); all three -> a client. Pass the
    # fields explicitly (incl. None) so the result doesn't depend on ambient .env,
    # which on this machine carries real langfuse keys.
    dormant = Settings(
        langfuse_host="http://localhost:3000",
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )
    assert init_langfuse(dormant) is None
    full = Settings(
        langfuse_host="http://localhost:3000",
        langfuse_public_key="pk-lf-x",
        langfuse_secret_key=SecretStr("sk-lf-x"),
    )
    assert init_langfuse(full) is not None


async def test_traced_chat_stream_passthrough_and_records_usage() -> None:
    lf = FakeLangfuse()
    chat = traced_chat(FakeChat(), lf)  # type: ignore[arg-type]  # FakeLangfuse duck-types the SDK

    events = [e async for e in chat.stream([ChatMessage(role="user", content="hi")])]

    # Pass-through: identical event sequence to the inner model.
    assert [e for e in events if isinstance(e, TextDelta)] == [
        TextDelta(text="Hello "),
        TextDelta(text="world."),
    ]
    assert lf.observations == [("generation", "chat.stream")]
    # The span saw the joined answer and mapped token usage.
    assert lf.spans[0].updates == [
        {"output": "Hello world.", "usage_details": {"input": 11, "output": 3}}
    ]


async def test_traced_chat_complete_passthrough() -> None:
    lf = FakeLangfuse()
    chat = traced_chat(FakeChat(), lf)  # type: ignore[arg-type]
    result = await chat.complete([ChatMessage(role="user", content="hi")])
    assert result.text == "done"
    assert lf.observations == [("generation", "chat.complete")]
    assert lf.spans[0].updates == [{"output": "done", "usage_details": {"input": 5, "output": 2}}]


async def test_trace_request_exposes_trace_id_and_none_when_off() -> None:
    # The eval reads RequestTrace.trace_id to land a clickable trace link in each record.
    lf = FakeLangfuse()
    async with trace_request(lf, question="q", top_k=5, name="eval:q-001") as trace:  # type: ignore[arg-type]
        pass
    assert trace.trace_id == "trace-fake"
    assert lf.observations == [("span", "eval:q-001")]  # the eval label flows through

    async with trace_request(None, question="q", top_k=5) as off:
        pass
    assert off.trace_id is None  # tracing off -> no id, no error


async def test_traced_retriever_passthrough_and_records_hits() -> None:
    lf = FakeLangfuse()

    async def inner(query: str, top_k: int) -> list[RetrievedChunk]:
        return [_chunk(1), _chunk(2)]

    retriever = traced_retriever(inner, lf)  # type: ignore[arg-type]
    chunks = await retriever("why did Catiline fail?", 2)

    assert [c.chunk_id for c in chunks] == [101, 102]
    assert lf.observations == [("span", "retrieve")]
    out = lf.spans[0].updates[0]["output"]
    assert out["count"] == 2
    assert out["top_score"] == chunks[0].score
    assert out["chunk_ids"] == [101, 102]
