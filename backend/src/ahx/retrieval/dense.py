"""Dense retrieval — the ONE implementation shared by evals, CLI, and API.

Phase 3 design decision (phase-3-plan.md): the eval harness and the serving
path call the same function, so the numbers we measure are the numbers we
ship. Sync (CLI / eval harness) and async (API routes, rule #7) variants
share a single SQL statement builder — they cannot drift apart.

Scoring: pgvector cosine distance over the HNSW index; `score` on the result
is cosine similarity (1 - distance), higher = better.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Row, Select, select
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import Session

from ahx.db import ChunkRow, SourceRow
from ahx.retrieval.embedding import EmbeddingClient


class RetrievedChunk(BaseModel):
    """A ranked retrieval hit with everything downstream consumers need:
    span coverage for evals (pg_id + char offsets), citation rendering for
    the API (author/work/locator/text), ablation forensics (score)."""

    chunk_id: int
    pg_id: int
    author: str
    work_title: str
    locator: list[str]
    text: str  # verbatim passage — what generation + citations show
    score: float  # cosine similarity, 1.0 = identical direction
    char_start: int
    char_end: int
    rank: int  # 1-based (reassigned post-rerank in the 4.2 arm)
    # The contextualized text that was EMBEDDED (context_note + heading_path +
    # chunk_text). The reranker scores THIS, not `text` — alignment law (rule
    # #4). None for the ~11 unenriched chunks; rerank falls back to `text`.
    retrieval_text: str | None = None
    rerank_score: float | None = None  # set only when a reranker reordered the list


def _statement(vector: list[float], top_k: int) -> Select[tuple[ChunkRow, str, str, Any]]:
    distance = ChunkRow.embedding.cosine_distance(vector)  # pyright: ignore[reportAttributeAccessIssue]
    return (
        select(ChunkRow, SourceRow.author, SourceRow.title, distance.label("distance"))
        .join(SourceRow, SourceRow.pg_id == ChunkRow.pg_id)
        .order_by(distance)
        .limit(top_k)
    )


def _to_chunks(rows: Sequence[Row[tuple[ChunkRow, str, str, Any]]]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            pg_id=chunk.pg_id,
            author=author,
            work_title=title,
            locator=chunk.locator,
            text=chunk.text,
            score=1.0 - float(distance),
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            rank=rank,
            retrieval_text=chunk.retrieval_text,
        )
        for rank, (chunk, author, title, distance) in enumerate(rows, start=1)
    ]


def dense_retrieve(
    engine: Engine, embedder: EmbeddingClient, query: str, top_k: int
) -> list[RetrievedChunk]:
    """Sync path: CLI tools and the eval harness."""
    vector = embedder.embed_query_sync(query)
    with Session(engine) as session:
        rows = session.execute(_statement(vector, top_k)).all()
    return _to_chunks(rows)


async def dense_retrieve_async(
    engine: AsyncEngine, embedder: EmbeddingClient, query: str, top_k: int
) -> list[RetrievedChunk]:
    """Async path: FastAPI routes (rule #7 — never block the event loop)."""
    vector = await embedder.embed_query(query)
    async with AsyncSession(engine) as session:
        rows = (await session.execute(_statement(vector, top_k))).all()
    return _to_chunks(rows)
