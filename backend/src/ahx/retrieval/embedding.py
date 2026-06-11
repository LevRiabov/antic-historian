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

import math
from collections.abc import Iterator
from typing import Any, cast

import httpx

from ahx.config import Settings

QWEN3_QUERY_PREFIX = (
    "Instruct: Given a search query, retrieve relevant passages "
    "from ancient history texts that answer the query\nQuery:"
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class EmbeddingClient:
    def __init__(self, settings: Settings, transport: httpx.MockTransport | None = None) -> None:
        self._base_url = settings.embed_base_url
        self._model = settings.embed_model
        self._dim = settings.embed_dim
        self._batch_size = settings.embed_batch_size
        self._transport = transport  # tests inject a fake server here

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
                raise ValueError(f"dim mismatch: expected {self._dim}, got {len(vector)}")
            vectors[index] = [float(v) for v in vector]
        return vectors

    def _embed_sync(self, client: httpx.Client, texts: list[str]) -> list[list[float]]:
        response = client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": texts, "encoding_format": "float"},
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
        """Query-side, sync (CLI / eval harness): instruction prefix applied."""
        with self._client(timeout=120) as client:
            return self._embed_sync(client, [QWEN3_QUERY_PREFIX + text])[0]

    async def embed_query(self, text: str) -> list[float]:
        """Query-side, async (API routes — rule #7): instruction prefix applied."""
        async with httpx.AsyncClient(timeout=120, transport=self._transport) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                json={
                    "model": self._model,
                    "input": [QWEN3_QUERY_PREFIX + text],
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            return self._parse_response(response.json(), expected=1)[0]
