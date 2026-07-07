"""Citation contract: source table derived from retrieval, marker audit.

Marker number == chunk rank in the prompt's source list, so the mapping
[n] -> source is fixed before the LLM produces a single token.
"""

import re
from collections.abc import Container, Iterable

from pydantic import BaseModel

from ahx.retrieval.dense import RetrievedChunk

# Matches [2] and grouped forms like [1, 3] — small local models group
# markers despite the prompt showing [1][3] (caught by live smoke test).
_MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")


class Citation(BaseModel):
    """One row of the source table sent to the client and the eval harness."""

    marker: int  # the [n] the answer text refers to
    chunk_id: int
    pg_id: int
    author: str
    work_title: str
    locator: list[str]
    text: str
    score: float
    char_start: int
    char_end: int


class MarkerAudit(BaseModel):
    used: list[int]  # distinct valid markers, in order of first appearance
    dangling: list[int]  # markers that point at no source — a quality signal


def citations_from_chunks(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            marker=chunk.rank,
            chunk_id=chunk.chunk_id,
            pg_id=chunk.pg_id,
            author=chunk.author,
            work_title=chunk.work_title,
            locator=chunk.locator,
            text=chunk.text,
            score=chunk.score,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for chunk in chunks
    ]


def extract_markers(answer: str, valid: set[int]) -> MarkerAudit:
    """Audit which [n] markers the answer used; never raises on bad ones."""
    used: list[int] = []
    dangling: list[int] = []
    for match in _MARKER_RE.finditer(answer):
        for part in match.group(1).split(","):
            marker = int(part.strip())
            bucket = used if marker in valid else dangling
            if marker not in bucket:
                bucket.append(marker)
    return MarkerAudit(used=used, dangling=dangling)


def ungrounded_citations(cited_ids: Iterable[int], retrieved_ids: Container[int]) -> list[int]:
    """Cited chunk ids that point at no retrieved passage — forged attributions.

    The agent cites by chunk id (`[c<id>]`); an id it never actually retrieved is a
    fabricated citation — a faithfulness signal AND a security one (a prompt-injection
    that makes the model invent a source shows up here). Returned sorted + distinct.

    The agent adapter feeds this into MarkerAudit.dangling: auditing its rewritten
    answer can't catch forgeries (`_rewrite` drops forged ids from mixed groups, and an
    all-phantom token stays in `[c<id>]` form the `[n]` audit ignores). Single-shot
    already flags dangling `[n]` markers; this gives the multi-search agent path the
    same eye instead of swallowing the forgery silently.
    """
    return sorted({cid for cid in cited_ids if cid not in retrieved_ids})
