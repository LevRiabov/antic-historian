"""Run the agent and adapt its result to the single-shot event contract.

Two jobs:

1. `build_agent_events` — the citation adapter. The agent accumulates evidence
   across many searches and cites chunks by id as `[c<id>]`. This numbers EVERY
   collected (deduped) passage [1]..[N] in first-appearance order — the full
   source table the model saw — rewrites the answer's [c<id>] tokens to those
   markers, and builds the standard Citation table. Output is shape-identical to
   the single-shot pipeline's SourcesEvent/DoneEvent, so evals/generation.py
   (score_generation, judge_question, aggregates) consume the agent unchanged —
   agent-vs-single-shot becomes a like-for-like comparison.

2. `run_agent` — build the Toolbox + graph for one question, invoke the loop,
   and return (SourcesEvent, DoneEvent, final AgentState). The state is returned
   too so the smoke test / a trace viewer can inspect the ReAct history.

The mechanical `refused` flag reuses the pipeline's exact-sentence test (not the
model's self-reported refused bool) so it matches single-shot; the judge layer
handles semantic refusals downstream, exactly as for single-shot.
"""

import re
from collections.abc import AsyncIterator, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncEngine

from ahx.agent.graph import (
    DEFAULT_MAX_STEPS,
    AgentGraph,
    astream_agent,
    build_agent_graph,
    invoke_agent,
)
from ahx.agent.prompts import AGENT_PROMPT_VERSION
from ahx.agent.state import AgentState, Step
from ahx.agent.tools import Toolbox
from ahx.config import Settings
from ahx.db import create_async_db_engine
from ahx.generation.citations import Citation, extract_markers
from ahx.generation.pipeline import (
    AskEvent,
    DeltaEvent,
    DoneEvent,
    SourcesEvent,
    StepEvent,
    _is_refusal,  # pyright: ignore[reportPrivateUsage]
)
from ahx.llm import ChatModel, Usage, chat_model_from_settings
from ahx.pricing import cost_for, load_price_table
from ahx.retrieval.dense import RetrievedChunk
from ahx.retrieval.embedding import EmbeddingClient
from ahx.retrieval.factory import build_async_retriever

# The model cites with the search label token [c<id>], and gemma groups them in
# one bracket — [c41, c88] or [c41][c88] (same footgun citations.py documents). Match
# a whole bracket group of c-prefixed ids; _CID_RE pulls the numbers out of it.
_CITE_GROUP_RE = re.compile(r"\[\s*c\d+(?:\s*,\s*c?\d+)*\s*\]")
_CID_RE = re.compile(r"\d+")


def _dedup(chunks: list[RetrievedChunk]) -> dict[int, RetrievedChunk]:
    """First occurrence wins — the same chunk can resurface across searches."""
    by_id: dict[int, RetrievedChunk] = {}
    for chunk in chunks:
        by_id.setdefault(chunk.chunk_id, chunk)
    return by_id


def build_agent_events(
    state: AgentState, model_name: str | None = None
) -> tuple[SourcesEvent, DoneEvent]:
    final = state["final"]
    assert final is not None  # the graph always terminates with a final result

    # agent-v5: the judge scores the RELEVANT set — the passages the agent kept
    # (keep_ids) plus any it actually cited — not every passage it ever retrieved.
    # The agent's own relevance filter shrinks the judge's input (its lost-in-the-
    # middle failure mode) while still showing it everything the answer rests on,
    # cited-or-not (the judge-v2 "see uncited grounding" property holds WITHIN the
    # kept set). Markers are 1..N over that set in first-appearance order.
    by_id = _dedup(state["collected"])  # dict preserves first-occurrence order
    cited = {int(n) for grp in _CITE_GROUP_RE.findall(final.answer) for n in _CID_RE.findall(grp)}
    relevant_ids = set(state["kept"]) | cited
    relevant = {cid: chunk for cid, chunk in by_id.items() if cid in relevant_ids}
    marker_of = {cid: marker for marker, cid in enumerate(relevant, start=1)}

    def _rewrite(match: re.Match[str]) -> str:
        # Map each c-id in the (possibly grouped) bracket to its 1..N marker;
        # emit single-bracket form [1][3]. Drop unknown ids; if none resolve,
        # leave the original token untouched.
        markers = [
            marker_of[cid] for cid in map(int, _CID_RE.findall(match.group(0))) if cid in marker_of
        ]
        return "".join(f"[{m}]" for m in markers) if markers else match.group(0)

    rewritten = _CITE_GROUP_RE.sub(_rewrite, final.answer)

    citations = [
        Citation(
            marker=marker,
            chunk_id=cid,
            pg_id=chunk.pg_id,
            author=chunk.author,
            work_title=chunk.work_title,
            locator=chunk.locator,
            text=chunk.text,
            score=chunk.score,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for (cid, chunk), marker in zip(relevant.items(), marker_of.values(), strict=True)
    ]
    sources = SourcesEvent(citations=citations, prompt_version=AGENT_PROMPT_VERSION)
    # Summed across every think call — the agent's total generation tokens.
    usage = Usage(
        prompt_tokens=state["prompt_tokens"], completion_tokens=state["completion_tokens"]
    )
    # `used` = the prose [c<id>] markers — the comparable, single-shot-style signal.
    done = DoneEvent(
        answer=rewritten,
        refused=_is_refusal(rewritten),
        markers=extract_markers(rewritten, set(marker_of.values())),
        usage=usage,
        # Model name only known at the call site (the graph hides it); None -> no cost.
        cost=cost_for(model_name, usage, load_price_table()) if model_name else None,
        # served_by (6.4): the nominal chat model. A mid-run provider fallover prices
        # the whole run at this id (the graph doesn't track per-call served_by) — a
        # documented approximation, acceptable since fallover is the rare path.
        served_by=model_name,
    )
    return sources, done


# A per-question deep-mode event source (6.7). Returns the lazy async iterator so a
# caller (the guard) can decide NOT to start it — e.g. on a blocked input.
AgentStreamer = Callable[[str], AsyncIterator[AskEvent | StepEvent]]


def _answer_deltas(answer: str) -> list[str]:
    """Split the already-decided answer into word-sized pieces for a typing effect.
    Cosmetic ONLY — the text is byte-identical to what the eval/judge scored
    (eval==served); deep mode just displays it progressively (trailing space kept
    with each word so the pieces re-join exactly)."""
    return re.findall(r"\S+\s*", answer)


async def stream_agent_events(
    graph: AgentGraph,
    question: str,
    model_name: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AsyncIterator[AskEvent | StepEvent]:
    """Deep-mode event stream (6.7): a `StepEvent` per live ReAct step, then the SAME
    SourcesEvent / DeltaEvent* / DoneEvent the single-shot path emits (built by
    build_agent_events) — so eval==served holds: the answer text is exactly what the
    judge scored, the deltas are a cosmetic display stream of it. The finalize step is
    suppressed (it's conveyed by the sources + streamed answer that follow)."""
    index = 0
    final_state: AgentState | None = None
    async for item in astream_agent(graph, question, max_steps):
        if isinstance(item, Step):
            if item.action == "finalize":
                continue
            index += 1
            yield StepEvent(
                index=index,
                thought=item.thought,
                tool=item.action,
                args=item.args,
                observation=item.observation,
                chunk_ids=item.chunk_ids,
                searches_left=max(0, max_steps - index),
            )
        else:
            final_state = item
    assert final_state is not None
    sources, done = build_agent_events(final_state, model_name)
    yield sources
    for piece in _answer_deltas(done.answer):
        yield DeltaEvent(text=piece)
    yield done


def make_agent_streamer(
    settings: Settings,
    engine: AsyncEngine,
    chat: ChatModel,
    retriever_name: str,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentStreamer:
    """Build the deep-mode agent ONCE over an externally-managed engine (the API
    lifespan seam) and return a per-question streaming callable. Mirrors
    make_agent_engine but yields the live step stream; the compiled graph stays hidden
    in the closure (thin-waist rule). `chat` is the API's traced+composite model, so
    deep-mode steps are traced and fall over across providers for free."""
    embedder = EmbeddingClient(settings)
    retriever = build_async_retriever(settings, engine, embedder, retriever_name)
    toolbox = Toolbox(settings=settings, engine=engine, embedder=embedder, retriever=retriever)
    graph = build_agent_graph(chat, toolbox, max_steps)

    def run_one(question: str) -> AsyncIterator[AskEvent | StepEvent]:
        return stream_agent_events(graph, question, chat.model_name, max_steps)

    return run_one


def make_agent_engine(
    settings: Settings,
    engine: AsyncEngine,
    chat: ChatModel,
    retriever_name: str,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Callable[[str], Awaitable[tuple[SourcesEvent, DoneEvent]]]:
    """Build the agent ONCE over an externally-managed DB engine and return a
    per-question callable producing the same (SourcesEvent, DoneEvent) as
    single-shot. This is the eval harness's engine seam: it never sees a LangGraph
    type — the compiled graph is hidden in the closure (thin-waist rule)."""
    embedder = EmbeddingClient(settings)
    retriever = build_async_retriever(settings, engine, embedder, retriever_name)
    toolbox = Toolbox(settings=settings, engine=engine, embedder=embedder, retriever=retriever)
    graph = build_agent_graph(chat, toolbox, max_steps)

    async def run_one(question: str) -> tuple[SourcesEvent, DoneEvent]:
        state = await invoke_agent(graph, question, max_steps)
        return build_agent_events(state, chat.model_name)

    return run_one


async def run_agent(
    question: str,
    settings: Settings,
    chat: ChatModel | None = None,
    retriever_name: str = "rerank-cohere-pro-v1",
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[SourcesEvent, DoneEvent, AgentState]:
    """Run the agent on one question. Builds its own engine and disposes it —
    fine for a single question / smoke test; the full eval will share one engine
    across questions."""
    chat = chat or chat_model_from_settings(settings)
    engine = create_async_db_engine(settings.database_url)
    try:
        embedder = EmbeddingClient(settings)
        retriever = build_async_retriever(settings, engine, embedder, retriever_name)
        toolbox = Toolbox(settings=settings, engine=engine, embedder=embedder, retriever=retriever)
        graph = build_agent_graph(chat, toolbox, max_steps)
        final_state = await invoke_agent(graph, question, max_steps)
        sources, done = build_agent_events(final_state, chat.model_name)
        return sources, done, final_state
    finally:
        await engine.dispose()
