"""THE embedding module (CLAUDE.md hard rule #3).

Every embed call in this codebase goes through EmbeddingClient — nothing else
may talk to an embedding endpoint. This module owns:

1. **The prefix policy.** Qwen3-Embedding is instruction-aware: queries are
   prefixed with a task instruction, documents are embedded bare. Corpus
   vectors and query vectors MUST come from the same policy or retrieval
   silently degrades (docs/embeddings.md §6, footgun 1).
2. **Output format.** `encoding_format="float"` is pinned (footgun 6: some
   OpenAI-compatible stacks default to base64 and parse to garbage).
3. **Parity.** `ahx ingest parity` compares live vectors against a committed
   fixture (cosine >= 0.999) — run it after ANY runtime/model change.

Provisional model (Gate D2 pending): qwen3-embedding-0.6b on local llama-swap.
The D2 ablation swaps models by swapping this module's config, nowhere else.
"""

import asyncio
import json
import math
from collections.abc import Iterator
from typing import Any, cast

import httpx

from ahx._http import AsyncClientCache
from ahx.config import Settings

# Transient statuses worth retrying on the async query path (mirrors llm.py): with the
# embedding endpoint hosted on OpenRouter a concurrent run WILL hit 429s, and an
# unretried embed fails the whole /ask where the adjacent LLM call would have recovered.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5

QWEN3_QUERY_PREFIX = (
    "Instruct: Given a search query, retrieve relevant passages "
    "from ancient history texts that answer the query\nQuery:"
)

# Per-model-family query prefixes (documents are always embedded bare in this
# corpus). Unknown model = hard error, never a silent default — a wrong or
# missing prefix is a silent 1-5%+ quality loss (docs/embeddings.md footgun #1).
_QUERY_PREFIXES: dict[str, str] = {
    "qwen3-embedding": QWEN3_QUERY_PREFIX,  # local 0.6b AND hosted qwen/qwen3-embedding-8b
    "gte-modernbert": "",  # ModernBERT GTE: explicitly prompt-free
    # voyage-4-nano: add from the model card at D2 integration time.
}


def query_prefix_for(model: str) -> str:
    for family, prefix in _QUERY_PREFIXES.items():
        if family in model:
            return prefix
    raise ValueError(
        f"no query-prefix policy for embedding model {model!r} — "
        "add the family to _QUERY_PREFIXES (docs/embeddings.md footgun #1)"
    )


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    # A zero/empty vector has no direction: define similarity as 0 rather than
    # dividing by zero. This is the function behind the parity guard (rule #3),
    # so it must degrade to a meaningful value, not crash on a bad embedding.
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class EmbeddingClient:
    def __init__(self, settings: Settings, transport: httpx.MockTransport | None = None) -> None:
        self._base_url = settings.embed_base_url
        self._model = settings.embed_model
        self._dim = settings.embed_dim
        self._batch_size = settings.embed_batch_size
        self._api_key = (
            settings.embed_api_key.get_secret_value() if settings.embed_api_key else None
        )
        self._mrl_truncate = settings.embed_mrl_truncate
        self._provider = settings.embed_provider
        self._query_prefix = query_prefix_for(settings.embed_model)
        self._transport = transport  # tests inject a fake server here
        # ONE keep-alive client for the async query path, reused across requests
        # (the sync ingest/eval paths open their own short-lived clients below).
        self._http = AsyncClientCache(httpx.Timeout(120.0), transport)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _body(self, texts: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }
        if self._provider:
            body["provider"] = {"order": [self._provider], "allow_fallbacks": False}
        return body

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _parse_response(self, payload: object, expected: int) -> list[list[float]]:
        assert isinstance(payload, dict)
        data = cast(list[dict[str, Any]], payload["data"])
        if len(data) != expected:
            raise ValueError(f"embedding count mismatch: sent {expected}, got {len(data)}")
        # Re-order by index — API contract does not guarantee input order.
        vectors: list[list[float]] = [[] for _ in range(expected)]
        for item in data:
            index = cast(int, item["index"])
            vector = cast(list[float], item["embedding"])
            if len(vector) != self._dim:
                if self._mrl_truncate and len(vector) > self._dim:
                    # MRL truncation: keep the leading dims, renormalize
                    # (footgun #4 — only valid for MRL-trained models).
                    vector = _l2_normalize(vector[: self._dim])
                else:
                    raise ValueError(f"dim mismatch: expected {self._dim}, got {len(vector)}")
            vectors[index] = [float(v) for v in vector]
        return vectors

    def _embed_sync(self, client: httpx.Client, texts: list[str]) -> list[list[float]]:
        response = client.post(
            f"{self._base_url}/embeddings",
            json=self._body(texts),
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._parse_response(response.json(), expected=len(texts))

    def _client(self, timeout: int) -> httpx.Client:
        return httpx.Client(timeout=timeout, transport=self._transport)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Ingest-side: NO prefix. Sync + batched (CLI batch jobs)."""
        vectors: list[list[float]] = []
        with self._client(timeout=300) as client:
            for batch in _batched(texts, self._batch_size):
                vectors.extend(self._embed_sync(client, batch))
        return vectors

    def embed_query_sync(self, text: str) -> list[float]:
        """Query-side, sync (CLI / eval harness): model's query prefix applied."""
        with self._client(timeout=120) as client:
            return self._embed_sync(client, [self._query_prefix + text])[0]

    async def embed_query(self, text: str) -> list[float]:
        """Query-side, async (API routes — rule #7): model's query prefix applied.

        Retries transient 429/5xx with backoff (mirrors llm.py): every /ask and every
        agent search embeds a query, and an unretried 429 under concurrency becomes a
        false failure where the adjacent LLM call self-heals (eval-log 2026-06-15
        cohere incident). Uses ONE keep-alive client rather than a fresh handshake."""
        client = self._http.get()
        body = self._body([self._query_prefix + text])
        headers = self._headers()
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = await client.post(
                    f"{self._base_url}/embeddings", json=body, headers=headers
                )
                response.raise_for_status()
                return self._parse_response(response.json(), expected=1)[0]
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS - 1:
                    raise
            except (httpx.TransportError, json.JSONDecodeError):
                # connection reset / read timeout or a malformed body — transient.
                # (A dim/count ValueError is a real problem and is NOT retried.)
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
            await asyncio.sleep(0.5 * 2**attempt)
        raise RuntimeError("unreachable: embed retry loop exhausted without return or raise")

    async def aclose(self) -> None:
        """Release the pooled async client (API lifespan shutdown). The sync ingest/
        eval paths open and close their own per-call clients, so nothing else leaks."""
        await self._http.aclose()
