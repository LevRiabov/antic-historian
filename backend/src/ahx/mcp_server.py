"""MCP server exposing the corpus to MCP clients (Claude Code / Desktop).

Purpose: the golden-set authoring workflow (docs/golden-set.md) — a Claude
instance with these tools can hunt for question material in the real corpus,
verify quotes resolve, and read surrounding context.

Run: `uv run ahx mcp serve` (stdio). Repo-level .mcp.json wires it up for
Claude Code automatically. Requires: docker DB up; llama-swap up (for search).
"""

from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP

from ahx.config import Settings, get_settings
from ahx.evals.golden import GoldSpan, ResolutionError, resolve_span
from ahx.retrieval.embedding import EmbeddingClient

server = FastMCP("ahx-corpus")


@lru_cache
def _settings() -> Settings:
    return get_settings()


@lru_cache
def _embedder() -> EmbeddingClient:
    return EmbeddingClient(_settings())


def _engine() -> Any:
    from ahx.db import create_sync_engine

    return create_sync_engine(_settings().database_url)


@server.tool()
def list_sources() -> list[dict[str, Any]]:
    """List all works in the corpus: pg_id, author, title, category, chunk count."""
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    from ahx.db import ChunkRow, SourceRow

    with Session(_engine()) as session:
        rows = session.execute(
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
        ).all()
    return [
        {"pg_id": pg_id, "author": author, "title": title, "category": category, "chunks": count}
        for pg_id, author, title, category, count in rows
    ]


@server.tool()
def search_corpus(query: str, top_k: int = 8, pg_id: int | None = None) -> list[dict[str, Any]]:
    """Dense similarity search over all chunks. Optionally restrict to one work
    (pg_id). Returns chunk text with author/title/locator and char offsets into
    that work's canonical text."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from ahx.db import ChunkRow, SourceRow

    vector = _embedder().embed_query_sync(query)
    with Session(_engine()) as session:
        distance = ChunkRow.embedding.cosine_distance(vector)  # pyright: ignore[reportAttributeAccessIssue]
        statement = (
            select(ChunkRow, SourceRow.author, SourceRow.title, distance.label("d"))
            .join(SourceRow, SourceRow.pg_id == ChunkRow.pg_id)
            .order_by(distance)
            .limit(top_k)
        )
        if pg_id is not None:
            statement = statement.where(ChunkRow.pg_id == pg_id)
        rows = session.execute(statement).all()
    return [
        {
            "pg_id": chunk.pg_id,
            "author": author,
            "title": title,
            "locator": ".".join(chunk.locator),
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "similarity": round(1 - d, 4),
            "text": chunk.text,
        }
        for chunk, author, title, d in rows
    ]


@server.tool()
def read_passage(pg_id: int, char_start: int, char_end: int, pad: int = 0) -> dict[str, Any]:
    """Read an exact slice of a work's canonical text by char offsets, with
    optional `pad` characters of surrounding context on each side."""
    from ahx.evals.golden import _canonical_for  # pyright: ignore[reportPrivateUsage]

    canonical = _canonical_for(str(_settings().corpus_normalized_dir), pg_id)
    if canonical is None:
        return {"error": f"work pg{pg_id} not found in corpus/normalized/"}
    start = max(0, char_start - pad)
    end = min(len(canonical), char_end + pad)
    return {
        "pg_id": pg_id,
        "char_start": char_start,
        "char_end": char_end,
        "text": canonical[start:end],
        "padded": pad > 0,
    }


@server.tool()
def find_quote(pg_id: int, quote: str) -> dict[str, Any]:
    """Verify a gold-span quote: does it occur EXACTLY ONCE in the work's
    canonical text? Returns offsets if yes; 'not-found'/'ambiguous' otherwise.
    Use this before putting a quote into the golden set."""
    result = resolve_span(
        GoldSpan(pg_id=pg_id, quote=quote), _settings().corpus_normalized_dir, "mcp"
    )
    if isinstance(result, ResolutionError):
        return {"ok": False, "problem": result.problem, "occurrences": result.occurrences}
    return {"ok": True, "char_start": result.char_start, "char_end": result.char_end}


def run() -> None:
    server.run()
