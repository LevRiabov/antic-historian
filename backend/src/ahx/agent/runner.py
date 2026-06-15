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
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncEngine

from ahx.agent.graph import DEFAULT_MAX_STEPS, build_agent_graph, invoke_agent
from ahx.agent.prompts import AGENT_PROMPT_VERSION
from ahx.agent.state import AgentState
from ahx.agent.tools import Toolbox
from ahx.config import Settings
from ahx.db import create_async_db_engine
from ahx.generation.citations import Citation, extract_markers
from ahx.generation.pipeline import (
    DoneEvent,
    SourcesEvent,
    _is_refusal,  # pyright: ignore[reportPrivateUsage]
)
from ahx.llm import ChatModel, Usage, chat_model_from_settings
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


def build_agent_events(state: AgentState) -> tuple[SourcesEvent, DoneEvent]:
    final = state["final"]
    assert final is not None  # the graph always terminates with a final result

    # EVERY passage the model saw becomes a numbered source (judge-v2 principle,
    # generation.py: the judge sees ALL retrieved passages, cited ones flagged —
    # not just the cited subset). Markers are 1..N over the deduped collected set
    # in first-appearance order; the answer's [c<id>] tokens are rewritten to them.
    by_id = _dedup(state["collected"])  # dict preserves first-occurrence order
    marker_of = {cid: marker for marker, cid in enumerate(by_id, start=1)}

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
        for (cid, chunk), marker in zip(by_id.items(), marker_of.values(), strict=True)
    ]
    sources = SourcesEvent(citations=citations, prompt_version=AGENT_PROMPT_VERSION)
    # `used` = the prose [c<id>] markers (the comparable, single-shot-style signal);
    # final.cited_chunk_ids is an advisory deliberate-citing nudge, not scored here.
    done = DoneEvent(
        answer=rewritten,
        refused=_is_refusal(rewritten),
        markers=extract_markers(rewritten, set(marker_of.values())),
        # Summed across every think call — the agent's total generation cost.
        usage=Usage(
            prompt_tokens=state["prompt_tokens"], completion_tokens=state["completion_tokens"]
        ),
    )
    return sources, done


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
        return build_agent_events(state)

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
        sources, done = build_agent_events(final_state)
        return sources, done, final_state
    finally:
        await engine.dispose()
