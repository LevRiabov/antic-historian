"""Retriever dispatch: map a run label to a retrieval implementation.

The eval harness (and later the API) select a retriever by label so that a
"measured config change" is always a label change (the Phase 4 measurement
protocol). No framework types here (D1 interface rule) — just our own callables
over RetrievedChunk.

Convention: a label starting with `rerank` builds the dense->rerank pipeline
(reranker model + pool depth come from Settings, so the 4-model sweep is one env
var per arm); ANY other label is plain dense retrieval — the label only names the
corpus/config in the record (dense-v1, dense-ctx-v1, dense-8b-1024-nebius-v1, …),
the dense query path is identical across them. This keeps every historical dense
label working unchanged.
"""

from collections.abc import Callable

from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from ahx.config import Settings
from ahx.generation.pipeline import Retriever
from ahx.retrieval.dense import (
    RetrievedChunk,
    dense_retrieve,
    dense_retrieve_async,
)
from ahx.retrieval.embedding import EmbeddingClient
from ahx.retrieval.rerank import RerankClient, rerank_retrieve, rerank_retrieve_async
from ahx.retrieval.rrf import rrf_fuse
from ahx.retrieval.sparse import sparse_retrieve, sparse_retrieve_async

# (query, top_k) -> ranked hits. top_k is the number of FINAL results; for a
# rerank/hybrid retriever the candidate pool (settings.*_pool_n) is internal.
SyncRetriever = Callable[[str, int], list[RetrievedChunk]]


def is_rerank_label(label: str) -> bool:
    return label.startswith("rerank")


def is_hybrid_label(label: str) -> bool:
    return label.startswith("hybrid")


def build_sync_retriever(
    settings: Settings, engine: Engine, embedder: EmbeddingClient, label: str
) -> SyncRetriever:
    if is_rerank_label(label):
        reranker = RerankClient(settings)
        pool_n = settings.rerank_pool_n
        return lambda query, top_k: rerank_retrieve(
            engine, embedder, reranker, query, top_k, pool_n
        )
    if is_hybrid_label(label):
        pool_n = settings.hybrid_pool_n
        k = settings.rrf_k
        return lambda query, top_k: rrf_fuse(
            [
                dense_retrieve(engine, embedder, query, pool_n),
                sparse_retrieve(engine, query, pool_n),
            ],
            top_k,
            k,
        )
    return lambda query, top_k: dense_retrieve(engine, embedder, query, top_k)


def build_async_retriever(
    settings: Settings, engine: AsyncEngine, embedder: EmbeddingClient, label: str
) -> Retriever:
    """Async twin of build_sync_retriever for the generation eval / API routes
    (rule #7). Same label convention; matches the generation Retriever protocol."""
    if is_rerank_label(label):
        reranker = RerankClient(settings)
        pool_n = settings.rerank_pool_n

        async def _rerank(query: str, top_k: int) -> list[RetrievedChunk]:
            return await rerank_retrieve_async(engine, embedder, reranker, query, top_k, pool_n)

        return _rerank

    if is_hybrid_label(label):
        pool_n = settings.hybrid_pool_n
        k = settings.rrf_k

        async def _hybrid(query: str, top_k: int) -> list[RetrievedChunk]:
            dense = await dense_retrieve_async(engine, embedder, query, pool_n)
            sparse = await sparse_retrieve_async(engine, query, pool_n)
            return rrf_fuse([dense, sparse], top_k, k)

        return _hybrid

    async def _dense(query: str, top_k: int) -> list[RetrievedChunk]:
        return await dense_retrieve_async(engine, embedder, query, top_k)

    return _dense
