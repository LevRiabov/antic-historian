"""Reciprocal Rank Fusion — merge ranked lists by RANK, not score (Phase 4.3).

Dense (cosine) and sparse (ts_rank_cd) scores live on incompatible scales, so
fusing their raw scores needs fragile normalization. RRF sidesteps that: a
chunk's fused score is the sum over lists of 1/(k + rank_in_that_list). Chunks
ranking well in EITHER list surface; chunks ranking well in BOTH win. k (~60)
damps the long tail so deep ranks barely contribute. Parameter-light and the
field standard for dense+sparse fusion.
"""

from collections.abc import Sequence

from ahx.retrieval.dense import RetrievedChunk

DEFAULT_RRF_K = 60


def rrf_fuse(
    lists: Sequence[Sequence[RetrievedChunk]], top_k: int, k: int = DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one top-k list. Each input list must be in its own
    rank order (chunk.rank 1-based). The output's `score` is the RRF score and
    `rank` is reassigned 1-based; the representative chunk for a fused id prefers
    one carrying retrieval_text (so a later rerank stays alignment-correct)."""
    scores: dict[int, float] = {}
    rep: dict[int, RetrievedChunk] = {}
    for ranked in lists:
        for chunk in ranked:
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + chunk.rank)
            current = rep.get(chunk.chunk_id)
            if current is None or (
                current.retrieval_text is None and chunk.retrieval_text is not None
            ):
                rep[chunk.chunk_id] = chunk
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        rep[chunk_id].model_copy(update={"rank": new_rank, "score": fused})
        for new_rank, (chunk_id, fused) in enumerate(ordered[:top_k], start=1)
    ]
