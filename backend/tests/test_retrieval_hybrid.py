"""Unit tests for the hybrid pieces: RRF fusion (pure) and the sparse row->chunk
mapping. The FTS SQL itself is exercised live by `ahx eval run --retriever
hybrid-v1`; here we pin the fusion math and the RetrievedChunk construction."""

from typing import Any, cast

from ahx.db import ChunkRow
from ahx.retrieval.dense import RetrievedChunk
from ahx.retrieval.rrf import DEFAULT_RRF_K, rrf_fuse
from ahx.retrieval.sparse import _to_chunks  # pyright: ignore[reportPrivateUsage]


def _chunk(chunk_id: int, rank: int, *, retrieval_text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        pg_id=1,
        author="A",
        work_title="W",
        locator=["1"],
        text=f"text-{chunk_id}",
        score=0.5,
        char_start=0,
        char_end=100,
        rank=rank,
        retrieval_text=retrieval_text,
    )


def test_rrf_rewards_appearing_in_both_lists() -> None:
    # Chunk 1: dense #1 only. Chunk 2: sparse #1 only. Chunk 3: #2 in BOTH.
    dense = [_chunk(1, 1), _chunk(3, 2)]
    sparse = [_chunk(2, 1), _chunk(3, 2)]
    out = rrf_fuse([dense, sparse], top_k=3)

    k = DEFAULT_RRF_K
    expected_3 = 1 / (k + 2) + 1 / (k + 2)  # in both at rank 2
    expected_1 = 1 / (k + 1)  # dense #1 only
    # chunk 3 (both) outscores the rank-1-in-one-list singletons
    assert [c.chunk_id for c in out] == [3, 1, 2]
    assert out[0].score == expected_3
    assert out[1].score == expected_1
    assert [c.rank for c in out] == [1, 2, 3], "rank reassigned 1-based"


def test_rrf_dedups_by_chunk_id() -> None:
    dense = [_chunk(7, 1), _chunk(8, 2)]
    sparse = [_chunk(7, 1)]
    out = rrf_fuse([dense, sparse], top_k=10)
    assert sorted(c.chunk_id for c in out) == [7, 8], "each chunk appears once"


def test_rrf_top_k_cut() -> None:
    dense = [_chunk(i, i) for i in range(1, 6)]
    out = rrf_fuse([dense], top_k=2)
    assert len(out) == 2
    assert [c.chunk_id for c in out] == [1, 2]


def test_rrf_representative_prefers_retrieval_text() -> None:
    # Same chunk id in both lists; the rep must carry retrieval_text for a later
    # alignment-correct rerank, regardless of which list saw it first.
    sparse_first = [_chunk(5, 1, retrieval_text=None)]
    dense_has_ctx = [_chunk(5, 2, retrieval_text="ctx + text-5")]
    out = rrf_fuse([sparse_first, dense_has_ctx], top_k=1)
    assert out[0].retrieval_text == "ctx + text-5"


def _row(chunk_id: int, rank_score: float) -> tuple[ChunkRow, str, str, float]:
    chunk = ChunkRow(
        id=chunk_id,
        pg_id=2,
        chunk_index=0,
        chunking_version="structural-v1",
        division_index=0,
        locator=["3"],
        heading=None,
        text="Vercingetorix surrendered at Alesia.",
        char_start=10,
        char_end=46,
        token_count=6,
        embedding=None,
        retrieval_text="ctx. Vercingetorix surrendered at Alesia.",
    )
    return (chunk, "Caesar", "The Gallic Wars", rank_score)


def test_sparse_to_chunks_maps_rank_and_score() -> None:
    rows = cast(Any, [_row(1, 0.9), _row(2, 0.4)])
    chunks = _to_chunks(rows)
    assert [c.rank for c in chunks] == [1, 2]
    assert chunks[0].score == 0.9  # ts_rank_cd carried as score
    assert chunks[0].retrieval_text == "ctx. Vercingetorix surrendered at Alesia."
    assert chunks[0].author == "Caesar"
