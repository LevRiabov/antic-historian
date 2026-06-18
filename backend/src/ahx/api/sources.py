"""Read model for the corpus-listing endpoint (Phase 7 /sources page).

One row per *work as the DB holds it* — i.e. per manifest line, so multi-volume
sets (Grote 12 vols, Cassius Dio 6, Strabo 3 ...) appear as the separate volumes
they actually are, not the curated groupings of the design mock. Honest 1:1 with
the `sources` table; the frontend decides how to present it.

Async-only (rule #7): the API never blocks the event loop on a DB call. The sync
corpus listing already lives in mcp_server.list_sources for the offline tools.
"""

from collections.abc import Awaitable, Callable
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ahx.db import ChunkRow, SourceRow

# Host -> human label for the "Source" column. Derived (not stored) so the landing
# URL stays the single source of truth; an unknown host falls back to the bare host.
_SOURCE_LABELS: dict[str, str] = {
    "gutenberg.org": "Project Gutenberg",
    "archive.org": "Internet Archive",
    "classics.mit.edu": "Internet Classics Archive",
}


def source_label(landing_url: str | None) -> str:
    """Map a landing URL to its publisher label (e.g. 'Project Gutenberg')."""
    if not landing_url:
        return "Unknown"
    host = urlparse(landing_url).netloc.lower().removeprefix("www.")
    for suffix, label in _SOURCE_LABELS.items():
        if host == suffix or host.endswith("." + suffix):
            return label
    return host or "Unknown"


class SourceOut(BaseModel):
    """One corpus work as the /sources endpoint returns it (API boundary -> pydantic)."""

    pg_id: int
    author: str
    title: str
    translator: str
    category: Literal["primary", "scholarship"]
    pd_basis: str  # EU public-domain justification (e.g. "Macaulay d.1915 (+70=1985)")
    source: str  # derived publisher label
    landing_url: str  # canonical source page for the "↗" link
    chunks: int  # retrievable passages in the DB — auditable corpus size per work


# Bound to the engine in the API lifespan; overridable in tests (see get_retriever).
SourcesProvider = Callable[[], Awaitable[list[SourceOut]]]


async def list_sources_async(engine: AsyncEngine) -> list[SourceOut]:
    """Every loaded work + its passage count, ordered for display.

    INNER join on chunks: a source only appears once it has retrievable passages
    (a half-loaded row isn't part of the answerable corpus). GROUP BY the PK lets
    Postgres carry the other source columns by functional dependency.
    """
    statement = (
        select(
            SourceRow.pg_id,
            SourceRow.author,
            SourceRow.title,
            SourceRow.translator,
            SourceRow.category,
            SourceRow.pd_basis,
            SourceRow.landing_url,
            func.count(ChunkRow.id).label("chunks"),
        )
        .join(ChunkRow, ChunkRow.pg_id == SourceRow.pg_id)
        .group_by(SourceRow.pg_id)
        .order_by(SourceRow.author, SourceRow.title)
    )
    async with AsyncSession(engine) as session:
        rows = (await session.execute(statement)).all()
    return [
        SourceOut(
            pg_id=pg_id,
            author=author,
            title=title,
            translator=translator,
            category=category,  # pyright: ignore[reportArgumentType]  # validated by the manifest
            pd_basis=pd_basis or "",
            source=source_label(landing_url),
            landing_url=landing_url or "",
            chunks=chunks,
        )
        for pg_id, author, title, translator, category, pd_basis, landing_url, chunks in rows
    ]
