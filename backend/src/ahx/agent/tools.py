"""Agent tools — the executors behind each grammar-constrained action.

The think node (graph.py) decides WHICH action to take; this module RUNS it
against the corpus and renders the outcome as an `observation` string fed back
to the model on its next turn. Design rules in force:

* Async throughout (rule #7): the agent runs inside FastAPI, so no sync IO on
  the event loop. The disk-reading tool (read, on its `pad` path) is pushed to
  a worker thread with asyncio.to_thread.
* Shared retrieval CORE, not the MCP layer (rule #4, alignment): whole-corpus
  `search` reuses the shipped rerank-pro retriever; source-isolated `search`
  uses filtered dense (within one work the candidate pool is small and
  homogeneous, so rerank's cross-source value is low — a v1 choice, revisit by
  ablation). Both hit the exact vectors Phase 4 measured.
* Dependencies built once and passed in (no global Settings — ADR-001).

`Finalize` is NOT executed here: it ends the loop and is handled by the graph.
"""

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ahx.agent.actions import ListSources, Read, Search, ToolCall
from ahx.config import Settings
from ahx.db import ChunkRow, SourceRow
from ahx.evals.golden import (
    _canonical_for,  # pyright: ignore[reportPrivateUsage]
)
from ahx.generation.pipeline import Retriever
from ahx.retrieval.dense import RetrievedChunk, dense_retrieve_async
from ahx.retrieval.embedding import EmbeddingClient


@dataclass(frozen=True)
class Toolbox:
    """Everything the tools need, built once (graph.py) and threaded into each
    node call. `retriever` is the whole-corpus rerank-pro callable; `engine` +
    `embedder` drive the source-isolated (filtered dense) path."""

    settings: Settings
    engine: AsyncEngine
    embedder: EmbeddingClient
    retriever: Retriever


class ToolResult(BaseModel):
    """Uniform tool output: the `observation` the model reads next turn, plus any
    citable `chunks` to merge into state['collected'] (search only — read /
    list_sources inform the model but add no new citable units)."""

    observation: str
    chunks: list[RetrievedChunk] = Field(default_factory=list[RetrievedChunk])


def _render_hits(chunks: list[RetrievedChunk]) -> str:
    """Show each hit's FULL text (as single-shot feeds the generator). The label
    is the EXACT citation token `[c<id>]` so the model copies it verbatim into the
    answer (the adapter parses that token) — not a divergent display form."""
    if not chunks:
        return "No passages found."
    lines: list[str] = []
    for c in chunks:
        loc = " > ".join(c.locator) if c.locator else ""
        head = f"[c{c.chunk_id}] {c.author}, {c.work_title}" + (f" > {loc}" if loc else "")
        lines.append(f"{head}\n  {' '.join(c.text.split())}")
    return "\n".join(lines)


async def _search(action: Search, tb: Toolbox) -> ToolResult:
    if action.pg_id is None:
        chunks = await tb.retriever(action.query, action.top_k)  # whole corpus, rerank-pro
    else:
        chunks = await dense_retrieve_async(  # source-isolation: filtered dense
            tb.engine, tb.embedder, action.query, action.top_k, action.pg_id
        )
    return ToolResult(observation=_render_hits(chunks), chunks=chunks)


async def _read(action: Read, tb: Toolbox) -> ToolResult:
    """Read one chunk's full verbatim text by id; with `pad`, widen to surrounding
    canonical context (for an answer that straddles the chunk boundary)."""
    async with AsyncSession(tb.engine) as session:
        chunk = (
            await session.execute(select(ChunkRow).where(ChunkRow.id == action.chunk_id))
        ).scalar_one_or_none()
    if chunk is None:
        return ToolResult(observation=f"read: chunk {action.chunk_id} not found.")
    if action.pad <= 0:
        return ToolResult(observation=f"chunk {action.chunk_id} (pg{chunk.pg_id}):\n{chunk.text}")
    canonical = await asyncio.to_thread(
        _canonical_for, str(tb.settings.corpus_normalized_dir), chunk.pg_id
    )
    if canonical is None:
        return ToolResult(observation=f"chunk {action.chunk_id} (pg{chunk.pg_id}):\n{chunk.text}")
    start = max(0, chunk.char_start - action.pad)
    end = min(len(canonical), chunk.char_end + action.pad)
    return ToolResult(
        observation=f"pg{chunk.pg_id} chars {start}-{end} (chunk {action.chunk_id} + context):\n"
        f"{canonical[start:end]}"
    )


async def _list_sources(tb: Toolbox) -> ToolResult:
    statement = (
        select(
            SourceRow.pg_id,
            SourceRow.author,
            SourceRow.title,
            SourceRow.category,
            func.count(ChunkRow.id),
        )
        .join(ChunkRow, ChunkRow.pg_id == SourceRow.pg_id)
        .group_by(SourceRow.pg_id, SourceRow.author, SourceRow.title, SourceRow.category)
        .order_by(SourceRow.pg_id)
    )
    async with AsyncSession(tb.engine) as session:
        rows = (await session.execute(statement)).all()
    lines = [
        f"pg{pg_id}  {author} — {title}  ({category}, {n} chunks)"
        for pg_id, author, title, category, n in rows
    ]
    return ToolResult(observation="\n".join(lines) if lines else "Corpus is empty.")


async def execute(action: ToolCall, tb: Toolbox) -> ToolResult:
    """Run a non-finalize action. The graph never routes Finalize here (it ends
    the loop); the final raise is a guard, not an expected path."""
    if isinstance(action, Search):
        return await _search(action, tb)
    if isinstance(action, Read):
        return await _read(action, tb)
    if isinstance(action, ListSources):
        return await _list_sources(tb)
    raise ValueError(f"execute() called on non-tool action {type(action).__name__}")
