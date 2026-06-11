"""Unit tests for THE embedding module — httpx.MockTransport plays the server
(≈ MSW in the JS world), so no llama-swap is needed in CI."""

import json
from typing import Any

import httpx

from ahx.config import Settings
from ahx.retrieval.embedding import QWEN3_QUERY_PREFIX, EmbeddingClient, cosine


def make_client(recorded: list[dict[str, Any]], dim: int = 4) -> EmbeddingClient:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        recorded.append(payload)
        data = [
            # Reversed order on purpose — the client must re-sort by index.
            {"index": i, "embedding": [float(i + 1)] * dim}
            for i in reversed(range(len(payload["input"])))
        ]
        return httpx.Response(200, json={"data": data})

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        embed_dim=dim,
        embed_batch_size=2,
    )
    return EmbeddingClient(settings, transport=httpx.MockTransport(handler))


def test_documents_get_no_prefix_and_are_batched() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    vectors = client.embed_documents(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert len(recorded) == 2, "batch_size=2 -> 3 texts need 2 requests"
    sent_texts = [text for payload in recorded for text in payload["input"]]
    assert sent_texts == ["alpha", "beta", "gamma"], "documents must be sent verbatim"
    assert all(payload["encoding_format"] == "float" for payload in recorded)


def test_query_gets_instruction_prefix() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    client.embed_query_sync("How did Caesar die?")
    assert recorded[0]["input"] == [QWEN3_QUERY_PREFIX + "How did Caesar die?"]


async def test_async_query_gets_instruction_prefix() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    await client.embed_query("Why did the expedition fail?")
    assert recorded[0]["input"] == [QWEN3_QUERY_PREFIX + "Why did the expedition fail?"]


def test_vectors_reordered_by_index() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    vectors = client.embed_documents(["first", "second"])
    # Handler returns indices reversed; client must restore input order.
    assert vectors[0] == [1.0, 1.0, 1.0, 1.0]
    assert vectors[1] == [2.0, 2.0, 2.0, 2.0]


def test_cosine_basics() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9
