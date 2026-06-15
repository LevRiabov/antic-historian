"""Unit tests for THE rerank module — httpx.MockTransport plays the /rerank
server (no llama-swap / OpenRouter needed in CI). The live endpoint is exercised
by `ahx eval run --retriever rerank-v1`; here we pin the pure-Python parts:
the Cohere-shape client (instruction policy, sort, provider/auth), dedup, and
the candidate-reorder mapping."""

import json
from typing import Any

import httpx
import pytest

from ahx.config import Settings
from ahx.retrieval.dense import RetrievedChunk
from ahx.retrieval.rerank import (
    RerankClient,
    RerankError,
    RerankResult,
    _aligned_text,  # pyright: ignore[reportPrivateUsage]
    _reorder,  # pyright: ignore[reportPrivateUsage]
    dedup_overlapping,
    rerank_query_instruction_for,
)


def make_client(
    recorded: list[dict[str, Any]],
    model: str = "qwen3-reranker-0.6b",
    **settings_overrides: Any,
) -> RerankClient:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        recorded.append({"json": payload, "headers": dict(request.headers)})
        docs = payload["documents"]
        # Score = index, returned in reversed order on purpose: the client must
        # sort by score (descending), not trust response order.
        results = [{"index": i, "relevance_score": float(i)} for i in reversed(range(len(docs)))]
        return httpx.Response(200, json={"results": results})

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        rerank_model=model,
        **settings_overrides,
    )
    return RerankClient(settings, transport=httpx.MockTransport(handler))


def sent(recorded: list[dict[str, Any]]) -> dict[str, Any]:
    return recorded[0]["json"]


def _chunk(
    chunk_id: int,
    *,
    text: str = "x",
    retrieval_text: str | None = None,
    pg_id: int = 1,
    char_start: int = 0,
    char_end: int = 100,
    rank: int = 1,
    score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        pg_id=pg_id,
        author="A",
        work_title="W",
        locator=["1"],
        text=text,
        score=score,
        char_start=char_start,
        char_end=char_end,
        rank=rank,
        retrieval_text=retrieval_text,
    )


# --- the client: instruction policy, sort, headers ---


def test_qwen3_reranker_sends_query_bare() -> None:
    # llama.cpp templates the qwen3-reranker query itself (verified live);
    # we send it bare and let the server format it.
    recorded: list[dict[str, Any]] = []
    make_client(recorded).rerank_sync("How did Caesar die?", ["a", "b"])
    assert sent(recorded)["query"] == "How did Caesar die?"
    assert sent(recorded)["documents"] == ["a", "b"], "documents are scored bare"


def test_cohere_reranker_sends_query_unmodified() -> None:
    recorded: list[dict[str, Any]] = []
    make_client(recorded, model="cohere/rerank-v3.5").rerank_sync("q", ["a"])
    assert sent(recorded)["query"] == "q"


def test_results_sorted_by_score_descending() -> None:
    recorded: list[dict[str, Any]] = []
    out = make_client(recorded).rerank_sync("q", ["a", "b", "c"])
    # handler scores doc i with score i -> doc 2 best, then 1, then 0
    assert [r.index for r in out] == [2, 1, 0]
    assert [r.score for r in out] == [2.0, 1.0, 0.0]


async def test_async_rerank_matches_sync_shape() -> None:
    recorded: list[dict[str, Any]] = []
    out = await make_client(recorded).rerank("q", ["a", "b"])
    assert [r.index for r in out] == [1, 0]
    assert sent(recorded)["query"] == "q"


def test_auth_header_and_provider_only_when_set() -> None:
    with_key: list[dict[str, Any]] = []
    make_client(
        with_key, model="cohere/rerank-4-pro", rerank_api_key="sk-or-x", rerank_provider="cohere"
    ).rerank_sync("q", ["a"])
    assert with_key[0]["headers"]["authorization"] == "Bearer sk-or-x"
    assert sent(with_key)["provider"] == {"order": ["cohere"], "allow_fallbacks": False}

    bare: list[dict[str, Any]] = []
    make_client(bare).rerank_sync("q", ["a"])
    assert "authorization" not in bare[0]["headers"]
    assert "provider" not in sent(bare)


def test_score_key_variant_tolerated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "score": 0.9}]})

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    client = RerankClient(settings, transport=httpx.MockTransport(handler))
    (result,) = client.rerank_sync("q", ["a"])
    assert result.score == 0.9


def _no_sleep(_seconds: float) -> None:
    return None


def test_rerank_retries_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    # First call returns an upstream error body (no 'results'); the client retries.
    monkeypatch.setattr("ahx.retrieval.rerank.time.sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"error": "rate limited"})  # no "results"
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    client = RerankClient(
        Settings(_env_file=None),  # pyright: ignore[reportCallIssue]
        transport=httpx.MockTransport(handler),
    )
    (result,) = client.rerank_sync("q", ["a"])
    assert result.index == 0
    assert calls["n"] == 2, "retried exactly once after the transient failure"


def test_rerank_raises_after_exhausting_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ahx.retrieval.rerank.time.sleep", _no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "always bad"})  # never has "results"

    client = RerankClient(
        Settings(_env_file=None),  # pyright: ignore[reportCallIssue]
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RerankError, match="no 'results'"):
        client.rerank_sync("q", ["a"])


def test_out_of_range_index_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 5, "relevance_score": 1.0}]})

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    client = RerankClient(settings, transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="out of range"):
        client.rerank_sync("q", ["a"])


# --- the instruction registry (footgun: unknown model must fail loudly) ---


def test_instruction_registry_covers_families_and_rejects_unknown() -> None:
    # All known families are bare today (the server templates the query); the
    # registry's job is the loud-fail on an unknown reranker.
    assert rerank_query_instruction_for("qwen3-reranker-0.6b") == ""
    assert rerank_query_instruction_for("bge-reranker-v2-m3") == ""
    assert rerank_query_instruction_for("cohere/rerank-4-pro") == ""
    with pytest.raises(ValueError, match="no query-instruction policy"):
        rerank_query_instruction_for("mystery-reranker-v9")


def test_unknown_reranker_fails_at_construction() -> None:
    with pytest.raises(ValueError):
        make_client([], model="mystery-reranker-v9")


# --- alignment + dedup + reorder (pure functions) ---


def test_aligned_text_prefers_retrieval_text_falls_back_to_text() -> None:
    assert _aligned_text(_chunk(1, text="bare", retrieval_text="ctx + bare")) == "ctx + bare"
    assert _aligned_text(_chunk(2, text="bare", retrieval_text=None)) == "bare"


def test_dedup_drops_overlapping_same_work_keeps_distinct() -> None:
    chunks = [
        _chunk(1, pg_id=1, char_start=0, char_end=100),
        _chunk(2, pg_id=1, char_start=50, char_end=150),  # overlaps #1 -> dropped
        _chunk(3, pg_id=1, char_start=200, char_end=300),  # disjoint -> kept
        _chunk(4, pg_id=2, char_start=50, char_end=150),  # other work -> kept
    ]
    assert [c.chunk_id for c in dedup_overlapping(chunks)] == [1, 3, 4]


def test_dedup_keeps_the_higher_dense_ranked_of_a_pair() -> None:
    # Input is in dense-rank order; the first (better) survives.
    chunks = [_chunk(7, char_start=0, char_end=100), _chunk(8, char_start=10, char_end=90)]
    assert [c.chunk_id for c in dedup_overlapping(chunks)] == [7]


def test_reorder_applies_rerank_order_and_stamps_scores() -> None:
    candidates = [_chunk(10, rank=1), _chunk(11, rank=2), _chunk(12, rank=3)]
    ranked = [
        RerankResult(index=2, score=9.0),
        RerankResult(index=0, score=4.0),
        RerankResult(index=1, score=1.0),
    ]
    out = _reorder(candidates, ranked, top_k=2)

    assert [c.chunk_id for c in out] == [12, 10], "reranker order applied"
    assert [c.rank for c in out] == [1, 2], "rank reassigned 1-based"
    assert [c.rerank_score for c in out] == [9.0, 4.0]
    assert out[0].score == 0.5, "dense cosine score preserved for forensics"
