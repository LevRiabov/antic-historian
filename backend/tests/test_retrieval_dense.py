"""Unit tests for the shared dense-retrieval module (no DB required).

The SQL itself is exercised against a live Postgres by `ahx eval run` /
`ahx search`; here we pin the pure-Python parts: row -> RetrievedChunk
mapping, similarity conversion, rank assignment.
"""

from typing import Any, cast

from ahx.db import ChunkRow
from ahx.retrieval.dense import _to_chunks  # pyright: ignore[reportPrivateUsage]


def _row(distance: float, chunk_id: int) -> tuple[ChunkRow, str, str, float]:
    chunk = ChunkRow(
        id=chunk_id,
        pg_id=1,
        chunk_index=0,
        chunking_version="structural-v1",
        division_index=0,
        locator=["1", "2"],
        heading=None,
        text="Gallia est omnis divisa in partes tres.",
        char_start=100,
        char_end=139,
        token_count=12,
        embedding=None,
    )
    return (chunk, "Caesar", "The Gallic Wars", distance)


def test_to_chunks_maps_rows_in_rank_order() -> None:
    rows = cast(Any, [_row(0.25, chunk_id=7), _row(0.40, chunk_id=9)])
    chunks = _to_chunks(rows)

    assert [c.rank for c in chunks] == [1, 2]
    assert [c.chunk_id for c in chunks] == [7, 9]
    # score is cosine similarity = 1 - distance
    assert chunks[0].score == 0.75
    assert chunks[1].score == 0.60


def test_to_chunks_carries_citation_fields() -> None:
    (chunk,) = _to_chunks(cast(Any, [_row(0.1, chunk_id=1)]))

    assert chunk.author == "Caesar"
    assert chunk.work_title == "The Gallic Wars"
    assert chunk.locator == ["1", "2"]
    assert (chunk.char_start, chunk.char_end) == (100, 139)
