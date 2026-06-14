"""Sparse (BM25-class) retrieval over Postgres full-text search (Phase 4.3).

The lexical half of hybrid: a tsvector + GIN inverted index that matches the
query's EXACT tokens — the complement to dense's semantic match. Where dense
blurs a rare proper noun (Vercingetorix) or a Victorian spelling into nearby
concepts, this nails the chunk that literally contains the word.

Scored with `ts_rank_cd` (cover-density: rewards query terms appearing, and
appearing close together), which approximates BM25's term-frequency/rarity
weighting natively — no extension, one store joinable with the vectors (D3).
Returns the SAME RetrievedChunk shape as dense so RRF fuses the two uniformly;
the `score` here is the ts_rank, not comparable to cosine (RRF ignores it).

Match is on the verbatim `text` (the words a reader sees), not retrieval_text —
keyword search should hit the real passage, not the LLM context note.
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Row, Select, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import Session

from ahx.db import ChunkRow, SourceRow
from ahx.retrieval.dense import RetrievedChunk

_TS_CONFIG = "english"


def _statement(query: str, top_k: int) -> Select[tuple[ChunkRow, str, str, Any]]:
    # websearch_to_tsquery tolerates arbitrary natural-language input (no tsquery
    # syntax errors) — robust for golden-set questions.
    tsquery = func.websearch_to_tsquery(_TS_CONFIG, query)
    rank = func.ts_rank_cd(ChunkRow.text_tsv, tsquery)
    return (
        select(ChunkRow, SourceRow.author, SourceRow.title, rank.label("rank_score"))
        .join(SourceRow, SourceRow.pg_id == ChunkRow.pg_id)
        .where(ChunkRow.text_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
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
            score=float(rank_score),  # ts_rank_cd — lexical, not cosine
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            rank=rank,
            retrieval_text=chunk.retrieval_text,
        )
        for rank, (chunk, author, title, rank_score) in enumerate(rows, start=1)
    ]


def sparse_retrieve(engine: Engine, query: str, top_k: int) -> list[RetrievedChunk]:
    """Sync path: CLI / eval harness."""
    with Session(engine) as session:
        rows = session.execute(_statement(query, top_k)).all()
    return _to_chunks(rows)


async def sparse_retrieve_async(
    engine: AsyncEngine, query: str, top_k: int
) -> list[RetrievedChunk]:
    """Async path: FastAPI routes (rule #7)."""
    async with AsyncSession(engine) as session:
        rows = (await session.execute(_statement(query, top_k))).all()
    return _to_chunks(rows)
