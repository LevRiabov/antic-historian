"""Agent tests — pure / in-process, no services.

Two surfaces, both fully testable without a DB or a model:
* the citation adapter (build_agent_events): [c<id>] renumbering, dedup, refusal
  detection, unknown-id handling — driven by hand-built AgentStates;
* the graph loop (build_agent_graph + invoke_agent): a FakeChat returns scripted
  Decisions and a fake retriever returns scripted chunks, so we drive a full
  think->act->finalize run and the forced-finalize budget bound in-process.

The grammar itself (llama.cpp enforcing the Decision schema) is validated live,
not here — these tests assume a valid Decision and check the wiring around it.
"""

from collections.abc import AsyncIterator, Sequence
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from ahx.agent.actions import Decision, Finalize, Search
from ahx.agent.graph import build_agent_graph, invoke_agent
from ahx.agent.runner import build_agent_events
from ahx.agent.state import AgentResult, AgentState
from ahx.agent.tools import Toolbox, execute
from ahx.config import Settings
from ahx.generation.pipeline import Retriever
from ahx.generation.prompt import REFUSAL_TEXT
from ahx.llm import ChatMessage, ChatResult, StreamEnd, StreamEvent
from ahx.retrieval.dense import RetrievedChunk
from ahx.retrieval.embedding import EmbeddingClient


def chunk(chunk_id: int, text: str = "passage", pg_id: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        pg_id=pg_id,
        author="Suetonius",
        work_title="Lives of the Twelve Caesars",
        locator=["Julius", "82"],
        text=text,
        score=0.9,
        char_start=chunk_id * 10,
        char_end=chunk_id * 10 + 100,
        rank=1,
    )


def make_state(collected: list[RetrievedChunk], final: AgentResult) -> AgentState:
    return AgentState(
        question="q", history=[], collected=collected, step=1, final=final, pending=None
    )


class FakeChat:
    """Returns scripted Decisions as JSON (what the model would emit under the
    grammar). Satisfies the ChatModel Protocol structurally."""

    model_name = "fake-model"

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = decisions
        self._i = 0

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        yield StreamEnd(usage=None)

    async def complete(
        self, messages: Sequence[ChatMessage], response_format: dict[str, object] | None = None
    ) -> ChatResult:
        decision = self._decisions[min(self._i, len(self._decisions) - 1)]
        self._i += 1
        return ChatResult(text=decision.model_dump_json(), usage=None)


def make_retriever(chunks: list[RetrievedChunk]) -> Retriever:
    async def retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]

    return retrieve


def fake_toolbox(retriever: Retriever) -> Toolbox:
    """settings/engine/embedder are unused on the search(pg_id=None) path the
    tests exercise — cast None rather than stand up a DB."""
    return Toolbox(
        settings=cast(Settings, None),
        engine=cast(AsyncEngine, None),
        embedder=cast(EmbeddingClient, None),
        retriever=retriever,
    )


# --- adapter: build_agent_events ---


def test_adapter_numbers_all_collected_not_just_cited() -> None:
    # Two collected; answer cites only one. The judge must still see BOTH.
    state = make_state(
        [chunk(101), chunk(102)],
        AgentResult(answer="Stabbed 23 times [c101].", refused=False, cited_chunk_ids=[101]),
    )
    sources, done = build_agent_events(state)
    assert [(c.marker, c.chunk_id) for c in sources.citations] == [(1, 101), (2, 102)]
    assert done.markers.used == [1]  # only the cited one is "used"


def test_adapter_renumbers_in_first_appearance_order() -> None:
    state = make_state(
        [chunk(101), chunk(102)],
        AgentResult(answer="Foo [c102] bar [c101] baz [c102].", refused=False, cited_chunk_ids=[]),
    )
    _, done = build_agent_events(state)
    # markers assigned by collected order (101->1, 102->2); prose rewritten to them
    assert done.answer == "Foo [2] bar [1] baz [2]."
    assert done.markers.used == [2, 1]  # order of first appearance in the answer


def test_adapter_handles_comma_grouped_citations() -> None:
    # gemma groups citations in one bracket: [c101, c102] -> [1][2].
    state = make_state(
        [chunk(101), chunk(102)],
        AgentResult(answer="Both sources agree [c101, c102].", refused=False, cited_chunk_ids=[]),
    )
    _, done = build_agent_events(state)
    assert done.answer == "Both sources agree [1][2]."
    assert done.markers.used == [1, 2]


def test_adapter_dedups_repeated_chunks() -> None:
    state = make_state(
        [chunk(101), chunk(102), chunk(101)],  # 101 resurfaced across searches
        AgentResult(answer="x [c101].", refused=False, cited_chunk_ids=[]),
    )
    sources, _ = build_agent_events(state)
    assert [c.chunk_id for c in sources.citations] == [101, 102]


def test_adapter_leaves_uncollected_cite_untouched() -> None:
    state = make_state(
        [chunk(101)],
        AgentResult(answer="Known [c101], phantom [c777].", refused=False, cited_chunk_ids=[]),
    )
    _, done = build_agent_events(state)
    assert done.answer == "Known [1], phantom [c777]."  # phantom left as-is
    assert done.markers.used == [1]
    assert done.markers.dangling == []  # [c777] doesn't match the [n] marker form


def test_adapter_detects_refusal_even_when_model_flag_is_false() -> None:
    # The model wrote the contract sentence but set refused=False; the mechanical
    # flag is the exact-sentence test, matching single-shot.
    state = make_state(
        [chunk(101)],
        AgentResult(answer=REFUSAL_TEXT, refused=False, cited_chunk_ids=[]),
    )
    _, done = build_agent_events(state)
    assert done.refused is True


# --- graph loop ---


async def test_graph_runs_search_then_finalize() -> None:
    chunks = [chunk(101, "Caesar was stabbed three and twenty times"), chunk(102)]
    chat = FakeChat(
        [
            Decision(thought="look it up", action=Search(tool="search", query="caesar wounds")),
            Decision(
                thought="answer it",
                action=Finalize(
                    tool="finalize", answer="Stabbed 23 times [c101].", cited_chunk_ids=[101]
                ),
            ),
        ]
    )
    graph = build_agent_graph(chat, fake_toolbox(make_retriever(chunks)), max_steps=8)
    state = await invoke_agent(graph, "how many wounds?", max_steps=8)

    assert [s.action for s in state["history"]] == ["search", "finalize"]
    assert [c.chunk_id for c in state["collected"]] == [101, 102]
    _sources, done = build_agent_events(state)
    assert done.answer == "Stabbed 23 times [1]."
    assert done.markers.used == [1]
    assert done.refused is False


async def test_graph_forced_finalize_refuses_on_budget() -> None:
    # The model never finalizes; the loop must stop itself and refuse.
    chat = FakeChat([Decision(thought="keep going", action=Search(tool="search", query="x"))])
    graph = build_agent_graph(chat, fake_toolbox(make_retriever([chunk(101)])), max_steps=3)
    state = await invoke_agent(graph, "q", max_steps=3)

    final = state["final"]
    assert final is not None
    assert final.refused is True
    assert final.answer == REFUSAL_TEXT
    assert sum(1 for s in state["history"] if s.action == "search") == 3  # 3 turns, then stop
    _, done = build_agent_events(state)
    assert done.refused is True


# --- tools dispatch ---


async def test_execute_search_renders_chunk_ids_and_collects() -> None:
    tb = fake_toolbox(make_retriever([chunk(101, "the war"), chunk(102, "the senate")]))
    result = await execute(Search(tool="search", query="x"), tb)
    assert "[c101]" in result.observation  # label == citation token (model copies it)
    assert "[c102]" in result.observation
    assert [c.chunk_id for c in result.chunks] == [101, 102]


async def test_execute_rejects_finalize() -> None:
    tb = fake_toolbox(make_retriever([]))
    with pytest.raises(ValueError):
        await execute(Finalize(tool="finalize", answer="x"), tb)
