"""Retriever-dispatch tests (factory.py).

The factory maps a run LABEL to a retrieval implementation, so "a measured config
change is always a label change" (the Phase 4 ablation protocol). A typo routing
`hybrid-*` to plain dense would silently confound every ablation, yet the dispatch
itself needs no DB — only the returned callables hit one. We pin the label->branch
selection here (the returned closures carry distinct __name__s) and the pure label
predicates that gate it. No network, no DB: the engine is a stand-in never invoked.
"""

from typing import Any, cast

from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from ahx.config import Settings
from ahx.retrieval.embedding import EmbeddingClient
from ahx.retrieval.factory import (
    build_async_retriever,
    build_sync_retriever,
    is_hybrid_label,
    is_rerank_label,
)


def _settings() -> Settings:
    return Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def _embedder() -> EmbeddingClient:
    # Default embed model has a prefix policy, so this constructs without network.
    return EmbeddingClient(_settings())


# A stand-in engine: build_*_retriever only closes over it (the DB is touched when the
# returned callable runs, which these tests never do), so its concrete type is irrelevant.
_FAKE_ASYNC_ENGINE = cast(AsyncEngine, object())
_FAKE_SYNC_ENGINE = cast(Engine, object())


def _branch_name(retriever: object) -> str:
    """The dispatched closure's name (_dense / _hybrid / _rerank) — Retriever is a
    bare Callable alias, so reach __name__ off the concrete function object."""
    return getattr(retriever, "__name__", "")


def test_label_predicates() -> None:
    assert is_rerank_label("rerank-cohere-pro-v1")
    assert is_rerank_label("rerank")
    assert not is_rerank_label("dense-ctx-v1")
    assert not is_rerank_label("hybrid-v1")

    assert is_hybrid_label("hybrid-v1")
    assert is_hybrid_label("hybrid")
    assert not is_hybrid_label("dense-v1")
    assert not is_hybrid_label("rerank-v1")


def test_async_dispatch_dense_for_plain_labels() -> None:
    # Any non-rerank, non-hybrid label is plain dense — every historical dense label
    # (dense-v1, dense-ctx-v1, dense-8b-1024-nebius-v1, …) routes here unchanged.
    for label in ["dense-v1", "dense-ctx-v1", "dense-8b-1024-nebius-v1", "anything-else"]:
        retriever = build_async_retriever(_settings(), _FAKE_ASYNC_ENGINE, _embedder(), label)
        assert _branch_name(retriever) == "_dense", label


def test_async_dispatch_rerank_and_hybrid() -> None:
    rerank = build_async_retriever(_settings(), _FAKE_ASYNC_ENGINE, _embedder(), "rerank-v1")
    assert _branch_name(rerank) == "_rerank"

    hybrid = build_async_retriever(_settings(), _FAKE_ASYNC_ENGINE, _embedder(), "hybrid-v1")
    assert _branch_name(hybrid) == "_hybrid"


def test_sync_builders_construct_each_branch() -> None:
    # Sync builders return lambdas (indistinguishable by name), so assert each branch
    # CONSTRUCTS without error — the rerank branch builds a RerankClient and reads
    # rerank_pool_n; the hybrid branch reads hybrid_pool_n + rrf_k.
    settings, embedder = _settings(), _embedder()
    for label in ["dense-v1", "hybrid-v1", "rerank-v1"]:
        retriever: Any = build_sync_retriever(settings, _FAKE_SYNC_ENGINE, embedder, label)
        assert callable(retriever), label
