"""LLM access layer tests — wire-format parsing against a mock server.

No real LLM anywhere: httpx.MockTransport plays the OpenAI-compatible
endpoint, so these run in CI and pin the request payload we send as well as
the response parsing.
"""

import json
from typing import Any

import httpx

from ahx.llm import (
    ChatMessage,
    ChatModel,
    OpenAICompatChat,
    StreamEnd,
    TextDelta,
    Usage,
)

MESSAGES = [
    ChatMessage(role="system", content="Answer only from the sources."),
    ChatMessage(role="user", content="How did Caesar die?"),
]

COMPLETION_BODY: dict[str, Any] = {
    "choices": [{"message": {"role": "assistant", "content": "Stabbed 23 times."}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
}

SSE_BODY = (
    'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Et tu, "}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Brute?"}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16}}\n\n'
    "data: [DONE]\n\n"
)


class FakeServer:
    """Captures the request and returns a canned response."""

    def __init__(self, body: str | dict[str, Any], content_type: str = "application/json"):
        self.request: httpx.Request | None = None
        self._body = body
        self._content_type = content_type

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        if isinstance(self._body, dict):
            return httpx.Response(200, json=self._body)
        return httpx.Response(200, text=self._body, headers={"content-type": self._content_type})

    @property
    def sent_payload(self) -> dict[str, Any]:
        assert self.request is not None
        return json.loads(self.request.content)


def make_model(server: FakeServer, api_key: str | None = None) -> OpenAICompatChat:
    return OpenAICompatChat(
        base_url="http://test/v1",
        model="test-model",
        api_key=api_key,
        transport=httpx.MockTransport(server.handler),
    )


async def test_complete_parses_text_and_usage() -> None:
    server = FakeServer(COMPLETION_BODY)
    result = await make_model(server).complete(MESSAGES)

    assert result.text == "Stabbed 23 times."
    assert result.usage == Usage(prompt_tokens=12, completion_tokens=5)

    payload = server.sent_payload
    assert payload["model"] == "test-model"
    assert payload["messages"][0] == {"role": "system", "content": "Answer only from the sources."}
    assert "stream" not in payload


async def test_stream_yields_deltas_then_end_with_usage() -> None:
    server = FakeServer(SSE_BODY, content_type="text/event-stream")
    events = [event async for event in make_model(server).stream(MESSAGES)]

    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == ["Et tu, ", "Brute?"]
    assert isinstance(events[-1], StreamEnd)
    assert events[-1].usage == Usage(prompt_tokens=12, completion_tokens=4)

    payload = server.sent_payload
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


async def test_stream_without_usage_chunk_ends_with_none() -> None:
    body = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    server = FakeServer(body, content_type="text/event-stream")
    events = [event async for event in make_model(server).stream(MESSAGES)]

    assert events == [TextDelta(text="hi"), StreamEnd(usage=None, served_by="test-model")]


async def test_auth_header_only_when_api_key_set() -> None:
    with_key = FakeServer(COMPLETION_BODY)
    await make_model(with_key, api_key="sk-secret").complete(MESSAGES)
    assert with_key.request is not None
    assert with_key.request.headers["authorization"] == "Bearer sk-secret"

    without_key = FakeServer(COMPLETION_BODY)
    await make_model(without_key).complete(MESSAGES)
    assert without_key.request is not None
    assert "authorization" not in without_key.request.headers


def test_openai_compat_satisfies_protocol() -> None:
    # Structural typing check — pyright verifies this assignment statically.
    model: ChatModel = make_model(FakeServer(COMPLETION_BODY))
    assert model.model_name == "test-model"


async def test_complete_passes_response_format_when_given() -> None:
    schema: dict[str, Any] = {"type": "json_object"}
    server = FakeServer(COMPLETION_BODY)
    await make_model(server).complete(MESSAGES, response_format=schema)
    assert server.sent_payload["response_format"] == schema


async def test_complete_omits_response_format_by_default() -> None:
    server = FakeServer(COMPLETION_BODY)
    await make_model(server).complete(MESSAGES)
    assert "response_format" not in server.sent_payload
