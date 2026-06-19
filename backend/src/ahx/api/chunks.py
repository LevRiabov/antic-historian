"""Read model for fetching passages by chunk id (Phase 7 citation drawer).

The eval records store only chunk *ids* (a marker `n` maps to the answer's
`retrieved_chunk_ids[n-1]`), not the passage text — so the golden-set page needs
a way to turn an id back into a readable, verifiable passage when a user clicks a
citation. In the live /ask flow the full Citation already arrives on the SSE
`sources` event; this endpoint serves the same shape for the offline records, so
one drawer component renders both.

Async-only (rule #7): the API never blocks the event loop on a DB call.
"""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ahx.db import ChunkRow, SourceRow


class ChunkOut(BaseModel):
    """One corpus passage as the /chunks endpoint returns it (API boundary -> pydantic).

    Mirrors the live Citation shape (author/work/locator/text) minus the per-answer
    marker + score, so the citation drawer is fed identically from either source."""

    chunk_id: int
    pg_id: int
    author: str
    work_title: str
    locator: list[str]  # canonical path, e.g. ["Book 1", "§31"]
    heading: str | None
    text: str
    char_start: int
    char_end: int
    pd_basis: str  # EU public-domain justification, for the drawer's trust badge


# Bound to the engine in the API lifespan; overridable in tests (see get_chunks).
ChunksProvider = Callable[[list[int]], Awaitable[list[ChunkOut]]]


async def get_chunks_async(engine: AsyncEngine, ids: list[int]) -> list[ChunkOut]:
    """Fetch the passages for `ids`, joined to their source for author/work/PD basis.

    Returns only the ids that exist (a chunk dropped by a re-chunk simply won't
    appear); the caller decides how to render a missing one. Order is unspecified —
    the client looks rows up by id."""
    if not ids:
        return []
    statement = (
        select(ChunkRow, SourceRow.author, SourceRow.title, SourceRow.pd_basis)
        .join(SourceRow, SourceRow.pg_id == ChunkRow.pg_id)
        .where(ChunkRow.id.in_(ids))
    )
    async with AsyncSession(engine) as session:
        rows = (await session.execute(statement)).all()
    return [
        ChunkOut(
            chunk_id=row.id,
            pg_id=row.pg_id,
            author=author,
            work_title=title,
            locator=row.locator,
            heading=row.heading,
            text=row.text,
            char_start=row.char_start,
            char_end=row.char_end,
            pd_basis=pd_basis or "",
        )
        for row, author, title, pd_basis in rows
    ]
