"""THE reranking module (Phase 4.2) — the post-retrieval precision engine.

Mirror of the one-embedding-module rule (CLAUDE.md #3): every cross-encoder
rerank call goes through RerankClient. One wire dialect — the Cohere-compatible
POST /rerank shape — which BOTH our local llama.cpp rerank endpoint AND
OpenRouter's /api/v1/rerank speak. Swapping the local qwen3/bge reranker for a
hosted Cohere ceiling reference is three Settings values, zero code (the same
serving swap we proved at gate D2 for embeddings).

Alignment law (CLAUDE.md #4): the reranker scores the SAME contextualized text
that was embedded (`retrieval_text`), never bare `text` — bare-text rerank undid
contextual gains in rag-historian (47.9% vs 51.6%). `rerank_retrieve` enforces
this so the caller cannot get it wrong.

This module owns the per-family query-instruction policy: qwen3-reranker is
instruction-aware like its embedder sibling; bge / cohere are not. Unknown model
= hard error, never a silent default (same footgun class as the query prefix).

Verified live 2026-06-14: POST /v1/rerank works on both the bge-reranker-v2-m3 and
qwen3-reranker-0.6b llama-swap profiles. qwen3-reranker is scored with a BARE query
— llama.cpp applies the reranker's own (yes/no-token) template internally, and a
prepended instruction slightly LOWERED scores on a smoke test (0.975 -> 0.963), so
it belongs in no slot we control. QWEN3_RERANK_INSTRUCTION is kept as a documented
constant only — the fallback if a future server/model does NOT template the query.
"""

from typing import Any, cast

import httpx
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from ahx.config import Settings
from ahx.retrieval.dense import (
    RetrievedChunk,
    dense_retrieve,
    dense_retrieve_async,
)
from ahx.retrieval.embedding import EmbeddingClient

# Documented fallback only — NOT applied to qwen3-reranker by default: llama.cpp
# templates the query itself (verified 2026-06-14, see module docstring). Kept for
# a future server/model that does not template the reranker query.
QWEN3_RERANK_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages "
    "from ancient history texts that answer the query\nQuery: "
)

# Documents are always scored bare; this is the QUERY-side policy. Empty = the
# server formats the query for the model. Unknown model = hard error.
_RERANK_QUERY_INSTRUCTIONS: dict[str, str] = {
    "qwen3-reranker": "",  # llama.cpp applies the reranker's own template
    "bge-reranker": "",  # classic cross-encoder: no instruction
    "cohere": "",  # hosted cohere/rerank-*: server handles formatting
}


def rerank_query_instruction_for(model: str) -> str:
    for family, instruction in _RERANK_QUERY_INSTRUCTIONS.items():
        if family in model:
            return instruction
    raise ValueError(
        f"no query-instruction policy for reranker {model!r} — "
        "add the family to _RERANK_QUERY_INSTRUCTIONS (rerank.py)"
    )


class RerankResult(BaseModel):
    index: int  # position in the documents list sent to rerank()
    score: float  # relevance score, higher = better (range is model-dependent)


class RerankClient:
    def __init__(self, settings: Settings, transport: httpx.MockTransport | None = None) -> None:
        self._base_url = settings.rerank_base_url
        self._model = settings.rerank_model
        self._api_key = settings.rerank_api_key
        self._provider = settings.rerank_provider
        self._query_instruction = rerank_query_instruction_for(settings.rerank_model)
        self._transport = transport  # tests inject a fake server here
        # Generous timeout: a local llama-swap reranker may cold-load on first call.
        self._timeout = httpx.Timeout(300.0, connect=10.0)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _body(self, query: str, documents: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "query": self._query_instruction + query,
            "documents": documents,
        }
        if self._provider:
            body["provider"] = {"order": [self._provider], "allow_fallbacks": False}
        return body

    def _parse(self, payload: object, expected: int) -> list[RerankResult]:
        assert isinstance(payload, dict)
        results = cast(list[dict[str, Any]], payload["results"])
        if not results:
            raise ValueError("empty rerank response")
        parsed: list[RerankResult] = []
        for item in results:
            index = int(item["index"])
            if not 0 <= index < expected:
                raise ValueError(f"rerank index {index} out of range [0,{expected})")
            raw_score = item.get("relevance_score")
            if raw_score is None:
                raw_score = item["score"]  # tolerate the `score` key variant
            parsed.append(RerankResult(index=index, score=float(raw_score)))
        # The API contract sorts by relevance, but don't rely on it — sort here.
        parsed.sort(key=lambda r: r.score, reverse=True)
        return parsed

    def rerank_sync(self, query: str, documents: list[str]) -> list[RerankResult]:
        """Sync path: CLI / retrieval eval harness."""
        with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
            response = client.post(
                f"{self._base_url}/rerank",
                json=self._body(query, documents),
                headers=self._headers(),
            )
            response.raise_for_status()
            return self._parse(response.json(), expected=len(documents))

    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        """Async path: FastAPI routes (rule #7) / generation eval."""
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            response = await client.post(
                f"{self._base_url}/rerank",
                json=self._body(query, documents),
                headers=self._headers(),
            )
            response.raise_for_status()
            return self._parse(response.json(), expected=len(documents))


def _aligned_text(chunk: RetrievedChunk) -> str:
    """Alignment law: rerank the embedded (contextualized) text. Bare `text` is
    the documented fallback for the ~11 chunks left unenriched at enrich-v1."""
    return chunk.retrieval_text or chunk.text


def dedup_overlapping(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop near-duplicate passages from the candidate pool before rerank.

    The 500/50 chunk overlap means adjacent chunks from one work can both cover a
    single span — wasting rerank slots and risking two near-dups landing in the
    final top-k. A chunk is dropped if it overlaps (same work, overlapping char
    range) a higher-dense-ranked chunk already kept. Input must be in dense-rank
    order; output preserves it. Toggleable (measured with/without)."""
    kept: list[RetrievedChunk] = []
    for candidate in chunks:
        overlaps = any(
            kept_chunk.pg_id == candidate.pg_id
            and max(kept_chunk.char_start, candidate.char_start)
            < min(kept_chunk.char_end, candidate.char_end)
            for kept_chunk in kept
        )
        if not overlaps:
            kept.append(candidate)
    return kept


def _reorder(
    candidates: list[RetrievedChunk], ranked: list[RerankResult], top_k: int
) -> list[RetrievedChunk]:
    """Apply the reranker's order to the candidates, reassigning 1-based rank and
    stamping rerank_score; keep the dense cosine `score` for forensics."""
    reordered: list[RetrievedChunk] = []
    for new_rank, result in enumerate(ranked[:top_k], start=1):
        chunk = candidates[result.index]
        reordered.append(chunk.model_copy(update={"rank": new_rank, "rerank_score": result.score}))
    return reordered


def rerank_retrieve(
    engine: Engine,
    embedder: EmbeddingClient,
    reranker: RerankClient,
    query: str,
    top_k: int,
    pool_n: int,
    dedup: bool = True,
) -> list[RetrievedChunk]:
    """Sync: dense top-`pool_n` -> dedup -> rerank on retrieval_text -> top-`top_k`."""
    candidates = dense_retrieve(engine, embedder, query, pool_n)
    if dedup:
        candidates = dedup_overlapping(candidates)
    if not candidates:
        return []
    ranked = reranker.rerank_sync(query, [_aligned_text(c) for c in candidates])
    return _reorder(candidates, ranked, top_k)


async def rerank_retrieve_async(
    engine: AsyncEngine,
    embedder: EmbeddingClient,
    reranker: RerankClient,
    query: str,
    top_k: int,
    pool_n: int,
    dedup: bool = True,
) -> list[RetrievedChunk]:
    """Async: same pipeline as rerank_retrieve, for API routes (rule #7)."""
    candidates = await dense_retrieve_async(engine, embedder, query, pool_n)
    if dedup:
        candidates = dedup_overlapping(candidates)
    if not candidates:
        return []
    ranked = await reranker.rerank(query, [_aligned_text(c) for c in candidates])
    return _reorder(candidates, ranked, top_k)
