"""Generation layer tests: citation contract, prompt rendering, ask pipeline.

All pure / in-process — the ChatModel and Retriever Protocols are satisfied
by tiny fakes, no mocking framework needed (structural typing at work).
"""

from collections.abc import AsyncIterator, Sequence

from ahx.generation.citations import citations_from_chunks, extract_markers
from ahx.generation.pipeline import (
    AskEvent,
    DeltaEvent,
    DoneEvent,
    Retriever,
    SourcesEvent,
    ask,
)
from ahx.generation.prompt import PROMPT_VERSION, REFUSAL_TEXT, build_messages
from ahx.llm import ChatMessage, ChatResult, StreamEnd, StreamEvent, TextDelta, Usage
from ahx.retrieval.dense import RetrievedChunk


def chunk(rank: int, text: str = "Some passage text.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=100 + rank,
        pg_id=1,
        author="Suetonius",
        work_title="Lives of the Twelve Caesars",
        locator=["1", str(rank)],
        text=text,
        score=0.9 - rank * 0.1,
        char_start=rank * 1000,
        char_end=rank * 1000 + 500,
        rank=rank,
    )


class FakeChat:
    """Satisfies the ChatModel Protocol structurally."""

    model_name = "fake-model"

    def __init__(self, deltas: list[str], usage: Usage | None = None) -> None:
        self._deltas = deltas
        self._usage = usage
        self.received: list[ChatMessage] | None = None

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        self.received = list(messages)
        for delta in self._deltas:
            yield TextDelta(text=delta)
        yield StreamEnd(usage=self._usage)

    async def complete(self, messages: Sequence[ChatMessage]) -> ChatResult:
        self.received = list(messages)
        return ChatResult(text="".join(self._deltas), usage=self._usage)


def make_retriever(chunks: list[RetrievedChunk]) -> Retriever:
    async def retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]

    return retrieve


async def run_ask(chunks: list[RetrievedChunk], chat: FakeChat, top_k: int = 5) -> list[AskEvent]:
    return [e async for e in ask("How did Caesar die?", make_retriever(chunks), chat, top_k)]


# --- citations ---


def test_citations_marker_equals_rank() -> None:
    citations = citations_from_chunks([chunk(1), chunk(2)])
    assert [(c.marker, c.chunk_id) for c in citations] == [(1, 101), (2, 102)]


def test_extract_markers_dedup_order_dangling() -> None:
    audit = extract_markers("Stabbed [2], groaned [1], again [2], ghost [9].", valid={1, 2, 3})
    assert audit.used == [2, 1]  # order of first appearance, no duplicates
    assert audit.dangling == [9]


def test_extract_markers_handles_comma_grouped_form() -> None:
    # Observed live: gemma writes [1, 2] despite the prompt showing [1][3].
    audit = extract_markers("He fell at the pedestal [1, 2], struck 23 times [1][9].", valid={1, 2})
    assert audit.used == [1, 2]
    assert audit.dangling == [9]


def test_extract_markers_ignores_non_numeric_brackets() -> None:
    audit = extract_markers("He said [sic] something [1].", valid={1})
    assert audit.used == [1]
    assert audit.dangling == []


# --- prompt ---


def test_build_messages_numbers_sources_and_carries_question() -> None:
    messages = build_messages("How did Caesar die?", [chunk(1), chunk(2)])
    assert messages[0].role == "system"
    user = messages[1].content
    assert "[1] Suetonius, Lives of the Twelve Caesars (1.1)" in user
    assert "[2] Suetonius, Lives of the Twelve Caesars (1.2)" in user
    assert user.endswith("Question: How did Caesar die?")


def test_system_prompt_contains_refusal_contract() -> None:
    messages = build_messages("q", [chunk(1)])
    assert REFUSAL_TEXT in messages[0].content


# --- pipeline ---


async def test_ask_event_sequence_and_audit() -> None:
    chat = FakeChat(
        ["Stabbed ", "23 times [1]."], usage=Usage(prompt_tokens=50, completion_tokens=8)
    )
    events = await run_ask([chunk(1), chunk(2)], chat)

    sources = events[0]
    assert isinstance(sources, SourcesEvent)
    assert sources.prompt_version == PROMPT_VERSION
    assert [c.marker for c in sources.citations] == [1, 2]

    assert [e.text for e in events[1:-1] if isinstance(e, DeltaEvent)] == [
        "Stabbed ",
        "23 times [1].",
    ]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer == "Stabbed 23 times [1]."
    assert done.refused is False
    assert done.markers.used == [1]
    assert done.markers.dangling == []
    assert done.usage == Usage(prompt_tokens=50, completion_tokens=8)

    # The chat model saw the retrieved sources in its prompt.
    assert chat.received is not None
    assert "[2] Suetonius" in chat.received[1].content


async def test_ask_respects_top_k() -> None:
    events = await run_ask([chunk(1), chunk(2), chunk(3)], FakeChat(["x"]), top_k=2)
    sources = events[0]
    assert isinstance(sources, SourcesEvent)
    assert len(sources.citations) == 2


async def test_refusal_detected_exact_and_quote_wrapped() -> None:
    for raw in (REFUSAL_TEXT, f'"{REFUSAL_TEXT}"', f"  {REFUSAL_TEXT}\n"):
        events = await run_ask([chunk(1)], FakeChat([raw]))
        done = events[-1]
        assert isinstance(done, DoneEvent)
        assert done.refused is True, f"not detected as refusal: {raw!r}"


async def test_sloppy_refusal_is_not_absorbed() -> None:
    events = await run_ask([chunk(1)], FakeChat(["Sorry, the sources do not mention this."]))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.refused is False
