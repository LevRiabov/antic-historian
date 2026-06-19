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

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from ahx.agent.actions import Decision, Finalize, Search
from ahx.agent.graph import astream_agent, build_agent_graph, invoke_agent
from ahx.agent.runner import build_agent_events, stream_agent_events
from ahx.agent.state import AgentResult, AgentState, Step
from ahx.agent.tools import Toolbox, execute
from ahx.config import Settings
from ahx.generation.pipeline import DeltaEvent, DoneEvent, Retriever, SourcesEvent, StepEvent
from ahx.generation.prompt import REFUSAL_TEXT
from ahx.llm import ChatMessage, ChatResult, StreamEnd, StreamEvent, Usage
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


def make_state(
    collected: list[RetrievedChunk],
    final: AgentResult,
    kept: list[int] | None = None,
) -> AgentState:
    # kept defaults to "all collected ids" so the legacy adapter tests express
    # "everything is relevant"; the filtering tests pass an explicit kept set.
    return AgentState(
        question="q",
        history=[],
        collected=collected,
        step=1,
        final=final,
        pending=None,
        kept=kept if kept is not None else [c.chunk_id for c in collected],
        prompt_tokens=0,
        completion_tokens=0,
    )


class FakeChat:
    """Returns scripted moves as JSON (what the model would emit under the
    grammar). A `str` item is sent verbatim — for testing malformed/truncated
    output. Satisfies the ChatModel Protocol structurally."""

    model_name = "fake-model"

    def __init__(self, script: list[Decision | str]) -> None:
        self._script = script
        self._i = 0

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        yield StreamEnd(usage=None)

    async def complete(
        self, messages: Sequence[ChatMessage], response_format: dict[str, object] | None = None
    ) -> ChatResult:
        item = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        text = item if isinstance(item, str) else item.model_dump_json()
        return ChatResult(text=text, usage=Usage(prompt_tokens=10, completion_tokens=5))


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


def test_adapter_numbers_kept_even_when_not_cited() -> None:
    # Two kept; answer cites only one. The judge must still see BOTH kept passages
    # (agent-v5: the "see uncited grounding" property holds WITHIN the kept set).
    state = make_state(
        [chunk(101), chunk(102)],
        AgentResult(answer="Stabbed 23 times [c101].", refused=False),
        kept=[101, 102],
    )
    sources, done = build_agent_events(state)
    assert [(c.marker, c.chunk_id) for c in sources.citations] == [(1, 101), (2, 102)]
    assert done.markers.used == [1]  # only the cited one is "used"


def test_adapter_judge_sees_only_kept_plus_cited() -> None:
    # agent-v5 relevance filter: 3 collected, agent kept only 101 and cites 102.
    # The judge set is kept-union-cited = {101, 102}; 103 (neither) is dropped.
    state = make_state(
        [chunk(101), chunk(102), chunk(103)],
        AgentResult(answer="Point [c102].", refused=False),
        kept=[101],
    )
    sources, done = build_agent_events(state)
    assert [c.chunk_id for c in sources.citations] == [101, 102]  # 103 filtered out
    assert done.answer == "Point [2]."  # 101->1, 102->2 by first-appearance order
    assert done.markers.used == [2]


def test_adapter_renumbers_in_first_appearance_order() -> None:
    state = make_state(
        [chunk(101), chunk(102)],
        AgentResult(answer="Foo [c102] bar [c101] baz [c102].", refused=False),
    )
    _, done = build_agent_events(state)
    # markers assigned by collected order (101->1, 102->2); prose rewritten to them
    assert done.answer == "Foo [2] bar [1] baz [2]."
    assert done.markers.used == [2, 1]  # order of first appearance in the answer


def test_adapter_handles_comma_grouped_citations() -> None:
    # gemma groups citations in one bracket: [c101, c102] -> [1][2].
    state = make_state(
        [chunk(101), chunk(102)],
        AgentResult(answer="Both sources agree [c101, c102].", refused=False),
    )
    _, done = build_agent_events(state)
    assert done.answer == "Both sources agree [1][2]."
    assert done.markers.used == [1, 2]


def test_adapter_dedups_repeated_chunks() -> None:
    state = make_state(
        [chunk(101), chunk(102), chunk(101)],  # 101 resurfaced across searches
        AgentResult(answer="x [c101].", refused=False),
    )
    sources, _ = build_agent_events(state)
    assert [c.chunk_id for c in sources.citations] == [101, 102]


def test_adapter_leaves_uncollected_cite_untouched() -> None:
    state = make_state(
        [chunk(101)],
        AgentResult(answer="Known [c101], phantom [c777].", refused=False),
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
        AgentResult(answer=REFUSAL_TEXT, refused=False),
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
                action=Finalize(tool="finalize", answer="Stabbed 23 times [c101]."),
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


async def test_graph_keep_ids_flow_filters_judge_set() -> None:
    # keep_ids on the decision accumulates into state['kept'] and drives the
    # judge/citation set: 102 was retrieved but neither kept nor cited -> dropped.
    chunks = [chunk(101, "kept passage"), chunk(102, "dropped passage")]
    chat = FakeChat(
        [
            Decision(thought="look", action=Search(tool="search", query="x")),
            Decision(
                thought="found it",
                keep_ids=[101],
                action=Finalize(tool="finalize", answer="Answer [c101]."),
            ),
        ]
    )
    graph = build_agent_graph(chat, fake_toolbox(make_retriever(chunks)), max_steps=8)
    state = await invoke_agent(graph, "q", max_steps=8)
    assert state["kept"] == [101]
    assert [c.chunk_id for c in state["collected"]] == [101, 102]  # both retrieved
    sources, done = build_agent_events(state)
    assert [c.chunk_id for c in sources.citations] == [101]  # only the kept/cited one
    assert done.answer == "Answer [1]."


async def test_graph_accumulates_token_usage_across_think_calls() -> None:
    chat = FakeChat(
        [
            Decision(thought="s", action=Search(tool="search", query="x")),
            Decision(thought="f", action=Finalize(tool="finalize", answer="done")),
        ]
    )
    graph = build_agent_graph(chat, fake_toolbox(make_retriever([chunk(101)])), max_steps=8)
    state = await invoke_agent(graph, "q", max_steps=8)
    assert state["prompt_tokens"] == 20  # 2 think calls x 10
    assert state["completion_tokens"] == 10  # 2 think calls x 5
    _, done = build_agent_events(state)
    assert done.usage == Usage(prompt_tokens=20, completion_tokens=10)


async def test_graph_survives_unparseable_generation() -> None:
    # A truncated/degenerate generation -> invalid JSON. The think node must end
    # the question with a refusal, not raise (the crash that killed the 161-run).
    chat = FakeChat(['{"thought": "oops", "action": {"tool": "finalize", "answer": '])
    graph = build_agent_graph(chat, fake_toolbox(make_retriever([chunk(101)])), max_steps=8)
    state = await invoke_agent(graph, "q", max_steps=8)
    final = state["final"]
    assert final is not None
    assert final.refused is True


async def test_graph_rerolls_a_malformed_reply_instead_of_refusing() -> None:
    # agent-v5.1: hosted providers enforce response_format only best-effort, so a
    # single call can return unparseable JSON. That must NOT become an instant
    # zero-retrieval refusal (eval-log 2026-06-16) — the think node re-rolls and
    # the next (valid) generation wins. Token usage sums across both attempts.
    chat = FakeChat(
        [
            "not valid json — a hosted-provider hiccup",  # attempt 1: unparseable
            Decision(thought="recovered", action=Finalize(tool="finalize", answer="Ans [c101].")),
        ]
    )
    graph = build_agent_graph(chat, fake_toolbox(make_retriever([chunk(101)])), max_steps=8)
    state = await invoke_agent(graph, "q", max_steps=8)

    final = state["final"]
    assert final is not None
    assert final.refused is False
    assert final.answer == "Ans [c101]."
    assert state["prompt_tokens"] == 20  # 10 per attempt, both paid for


class SlowChat:
    """A chat whose complete() blocks past the per-step bound — used to prove the
    timeout fires instead of the call hanging to the overall request budget."""

    model_name = "slow-model"

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        yield StreamEnd(usage=None)

    async def complete(
        self, messages: Sequence[ChatMessage], response_format: dict[str, object] | None = None
    ) -> ChatResult:
        await asyncio.sleep(1.0)
        return ChatResult(text="{}", usage=None)


async def test_graph_step_timeout_aborts_a_stalled_call() -> None:
    # A single hung model call must raise (-> the terminal error frame), not eat the
    # whole request budget. With the per-step bound tiny and complete() sleeping past
    # it, the run raises TimeoutError. The eval path leaves step_timeout None (untimed).
    graph = build_agent_graph(
        SlowChat(),
        fake_toolbox(make_retriever([chunk(101)])),
        max_steps=8,
        step_timeout_seconds=0.01,
    )
    with pytest.raises(TimeoutError):
        await invoke_agent(graph, "q", max_steps=8)


async def test_graph_forced_synthesis_answers_from_evidence() -> None:
    # Budget spent WITH evidence in hand: the loop must run one synthesis turn and
    # write the answer from collected passages, not blind-refuse (eval-log con-012).
    chat = FakeChat(
        [
            Decision(thought="s", action=Search(tool="search", query="x")),
            Decision(thought="s", action=Search(tool="search", query="y")),
            Decision(thought="s", action=Search(tool="search", query="z")),
            # the synthesis turn runs under the finalize-only grammar, so the model
            # emits a BARE Finalize (not a Decision) — sent verbatim as a str.
            Finalize(tool="finalize", answer="Ans [c101].").model_dump_json(),
        ]
    )
    graph = build_agent_graph(chat, fake_toolbox(make_retriever([chunk(101)])), max_steps=3)
    state = await invoke_agent(graph, "q", max_steps=3)

    final = state["final"]
    assert final is not None
    assert final.refused is False
    assert final.answer == "Ans [c101]."
    assert sum(1 for s in state["history"] if s.action == "search") == 3  # 3 searches, then synth
    _, done = build_agent_events(state)
    assert done.answer == "Ans [1]."  # renumbered through the adapter


async def test_graph_forced_synthesis_falls_back_to_refusal_when_unparseable() -> None:
    # Budget spent and the synthesis turn can't produce a valid Finalize (model
    # never finalizes) -> honest refusal, the safety net behind synthesis.
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


# --- deep-mode streaming (6.7) ---


async def test_astream_agent_emits_steps_then_final_state() -> None:
    # The streaming twin of invoke_agent: each completed Step lands as it happens, the
    # final AgentState comes last (so the runner can build the same events as single-shot).
    chat = FakeChat(
        [
            Decision(thought="search", action=Search(tool="search", query="caesar")),
            Decision(thought="answer", action=Finalize(tool="finalize", answer="Stabbed [c101].")),
        ]
    )
    graph = build_agent_graph(chat, fake_toolbox(make_retriever([chunk(101)])), max_steps=8)
    items = [item async for item in astream_agent(graph, "q", 8)]

    steps = [i for i in items if isinstance(i, Step)]
    assert [s.action for s in steps] == ["search", "finalize"]
    final = items[-1]
    assert not isinstance(final, Step)  # final AgentState last
    assert final["final"] is not None and final["final"].answer == "Stabbed [c101]."


async def test_stream_agent_events_steps_then_eval_identical_answer() -> None:
    # Deep mode emits a StepEvent per live step, then the SAME sources/deltas/done as
    # single-shot. The finalize step is suppressed; the streamed deltas re-join to the
    # exact (renumbered) answer build_agent_events produced — eval==served.
    chat = FakeChat(
        [
            Decision(
                thought="search the corpus",
                keep_ids=[101],
                action=Search(tool="search", query="caesar"),
            ),
            Decision(
                thought="enough",
                action=Finalize(tool="finalize", answer="Stabbed 23 times [c101]."),
            ),
        ]
    )
    graph = build_agent_graph(chat, fake_toolbox(make_retriever([chunk(101)])), max_steps=8)
    events = [e async for e in stream_agent_events(graph, "q", "fake-model", 8)]

    names = [type(e).__name__ for e in events]
    assert names[0] == "StepEvent"  # live step first
    assert names[1] == "SourcesEvent"
    assert names[-1] == "DoneEvent"

    steps = [e for e in events if isinstance(e, StepEvent)]
    assert len(steps) == 1  # the search; the finalize step is suppressed
    assert steps[0].tool == "search" and steps[0].index == 1 and steps[0].searches_left == 7
    assert steps[0].chunk_ids == [101]

    sources = next(e for e in events if isinstance(e, SourcesEvent))
    assert [c.chunk_id for c in sources.citations] == [101]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer == "Stabbed 23 times [1]."  # renumbered, eval-identical
    assert done.served_by == "fake-model"

    deltas = "".join(e.text for e in events if isinstance(e, DeltaEvent))
    assert deltas == done.answer  # cosmetic stream re-joins to the exact answer
