"""Unit tests for THE embedding module — httpx.MockTransport plays the server
(≈ MSW in the JS world), so no llama-swap is needed in CI."""

import json
import math
from typing import Any

import httpx
import pytest

from ahx.config import Settings
from ahx.retrieval import embedding
from ahx.retrieval.embedding import (
    QWEN3_QUERY_PREFIX,
    EmbeddingClient,
    cosine,
    query_prefix_for,
)


def make_client(
    recorded: list[dict[str, Any]],
    dim: int = 4,
    served_dim: int | None = None,
    **settings_overrides: Any,
) -> EmbeddingClient:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        recorded.append({"json": payload, "headers": dict(request.headers)})
        data = [
            # Reversed order on purpose — the client must re-sort by index.
            {"index": i, "embedding": [float(i + 1)] * (served_dim or dim)}
            for i in reversed(range(len(payload["input"])))
        ]
        return httpx.Response(200, json={"data": data})

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        embed_dim=dim,
        embed_batch_size=2,
        **settings_overrides,
    )
    return EmbeddingClient(settings, transport=httpx.MockTransport(handler))


def sent_payloads(recorded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry["json"] for entry in recorded]


def test_documents_get_no_prefix_and_are_batched() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    vectors = client.embed_documents(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert len(recorded) == 2, "batch_size=2 -> 3 texts need 2 requests"
    sent_texts = [text for payload in sent_payloads(recorded) for text in payload["input"]]
    assert sent_texts == ["alpha", "beta", "gamma"], "documents must be sent verbatim"
    assert all(payload["encoding_format"] == "float" for payload in sent_payloads(recorded))


def test_query_gets_instruction_prefix() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    client.embed_query_sync("How did Caesar die?")
    assert sent_payloads(recorded)[0]["input"] == [QWEN3_QUERY_PREFIX + "How did Caesar die?"]


async def test_async_query_gets_instruction_prefix() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    await client.embed_query("Why did the expedition fail?")
    expected = [QWEN3_QUERY_PREFIX + "Why did the expedition fail?"]
    assert sent_payloads(recorded)[0]["input"] == expected


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instant retry sleeps so the retry tests don't burn wall clock."""

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(embedding.asyncio, "sleep", instant)


async def test_embed_query_retries_transient_5xx(no_backoff: None) -> None:
    # The async query path runs on every /ask; a transient 429/5xx must self-heal
    # rather than fail the request (H2), matching the LLM layer's retry.
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)  # transient hiccup
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 1.0, 1.0, 1.0]}]})

    settings = Settings(_env_file=None, embed_dim=4)  # pyright: ignore[reportCallIssue]
    client = EmbeddingClient(settings, transport=httpx.MockTransport(handler))
    vector = await client.embed_query("How did Caesar die?")

    assert attempts == 2  # retried once, then served
    assert vector == [1.0, 1.0, 1.0, 1.0]


async def test_embed_query_raises_after_exhausting_retries(no_backoff: None) -> None:
    settings = Settings(_env_file=None, embed_dim=4)  # pyright: ignore[reportCallIssue]
    client = EmbeddingClient(
        settings, transport=httpx.MockTransport(lambda _r: httpx.Response(503))
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.embed_query("persistently down")


async def test_aclose_is_idempotent_and_client_rebuilds() -> None:
    # aclose() (lifespan shutdown) is safe to call repeatedly, and the client lazily
    # rebuilds on the next use — so a closed client never wedges a later request.
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    await client.embed_query("first")
    await client.aclose()
    await client.aclose()  # no error on a second close
    await client.embed_query("second")  # rebuilds the pooled client
    assert len(recorded) == 2


def test_vectors_reordered_by_index() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded)
    vectors = client.embed_documents(["first", "second"])
    # Handler returns indices reversed; client must restore input order.
    assert vectors[0] == [1.0, 1.0, 1.0, 1.0]
    assert vectors[1] == [2.0, 2.0, 2.0, 2.0]


# --- D2 ablation infrastructure ---


def test_prefix_registry_covers_families_and_rejects_unknown() -> None:
    assert query_prefix_for("qwen3-embedding-0.6b") == QWEN3_QUERY_PREFIX
    assert query_prefix_for("qwen/qwen3-embedding-8b") == QWEN3_QUERY_PREFIX
    assert query_prefix_for("gte-modernbert-base") == ""
    with pytest.raises(ValueError, match="no query-prefix policy"):
        query_prefix_for("mystery-embedder-v9")


def test_unknown_model_fails_at_client_construction() -> None:
    # Footgun #1 defense: a model without a prefix policy must fail loudly
    # before any vector is produced.
    with pytest.raises(ValueError):
        make_client([], embed_model="mystery-embedder-v9")


def test_auth_header_sent_only_when_api_key_set() -> None:
    with_key: list[dict[str, Any]] = []
    make_client(with_key, embed_api_key="sk-or-test").embed_documents(["x"])
    assert with_key[0]["headers"]["authorization"] == "Bearer sk-or-test"

    without_key: list[dict[str, Any]] = []
    make_client(without_key).embed_documents(["x"])
    assert "authorization" not in without_key[0]["headers"]


def test_provider_pinning_in_body_only_when_set() -> None:
    pinned: list[dict[str, Any]] = []
    make_client(pinned, embed_provider="nebius").embed_documents(["x"])
    assert sent_payloads(pinned)[0]["provider"] == {"order": ["nebius"], "allow_fallbacks": False}

    unpinned: list[dict[str, Any]] = []
    make_client(unpinned, embed_provider=None).embed_documents(["x"])
    assert "provider" not in sent_payloads(unpinned)[0]


def test_mrl_truncation_renormalizes() -> None:
    # Server returns 8 dims, client configured for 4 with truncation enabled.
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded, dim=4, served_dim=8, embed_mrl_truncate=True)
    (vector,) = client.embed_documents(["x"])
    assert len(vector) == 4
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_dim_mismatch_without_truncation_flag_raises() -> None:
    recorded: list[dict[str, Any]] = []
    client = make_client(recorded, dim=4, served_dim=8, embed_mrl_truncate=False)
    with pytest.raises(ValueError, match="dim mismatch"):
        client.embed_documents(["x"])


def test_cosine_basics() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_zero_vector_is_zero_not_crash() -> None:
    # A zero/empty vector has no direction: define similarity as 0 rather than raising
    # ZeroDivisionError. cosine() backs the parity guard, so it must degrade gracefully.
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0
    assert cosine([], []) == 0.0
