"""LLM access layer — the thin provider-agnostic seam (gate D5).

Business logic depends on the ChatModel Protocol, never on a concrete
provider or framework type (ADR-001). One wire dialect is implemented:
OpenAI `/chat/completions`, which covers local llama-swap, vLLM, and most
hosted providers — switching providers is two Settings values, zero code.

Async-only by design: API routes must be async (rule #7), and the eval
harness wraps calls in `asyncio.run`. Streaming yields TextDelta events
followed by exactly one StreamEnd carrying usage (an async generator cannot
smuggle a return value out of an `async for`, so the terminal event IS the
return value). These map 1:1 onto the API's SSE `delta`/`done` events.
"""

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol, cast

import httpx
from pydantic import BaseModel

from ahx.config import Settings

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class TextDelta(BaseModel):
    text: str


class StreamEnd(BaseModel):
    usage: Usage | None


StreamEvent = TextDelta | StreamEnd


class ChatResult(BaseModel):
    text: str
    usage: Usage | None


class ChatModel(Protocol):
    """Structural interface: anything with these methods is a ChatModel."""

    @property
    def model_name(self) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]: ...

    async def complete(self, messages: Sequence[ChatMessage]) -> ChatResult: ...


class OpenAICompatChat:
    """OpenAI chat-completions dialect over any base_url.

    Generation params (temperature, max_tokens) live here, passed explicitly
    by the caller — the generation module owns and versions them (3.3).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        transport: httpx.MockTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._transport = transport  # tests inject a fake server here
        # Generous read timeout: llama-swap may cold-load a model on first call.
        self._timeout = httpx.Timeout(600.0, connect=10.0)

    @property
    def model_name(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _payload(self, messages: Sequence[ChatMessage], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if stream:
            payload["stream"] = True
            # Ask for a final usage chunk; servers without support ignore this.
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _parse_usage(raw: object) -> Usage | None:
        if not isinstance(raw, dict):
            return None
        data = cast(dict[str, Any], raw)
        return Usage(
            prompt_tokens=int(data["prompt_tokens"]),
            completion_tokens=int(data["completion_tokens"]),
        )

    async def complete(self, messages: Sequence[ChatMessage]) -> ChatResult:
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                json=self._payload(messages, stream=False),
                headers=self._headers(),
            )
            response.raise_for_status()
            data = cast(dict[str, Any], response.json())
        text = data["choices"][0]["message"]["content"] or ""
        return ChatResult(text=text, usage=self._parse_usage(data.get("usage")))

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        usage: Usage | None = None
        async with (
            httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client,
            client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=self._payload(messages, stream=True),
                headers=self._headers(),
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue  # SSE comments / blank keep-alives
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                chunk = cast(dict[str, Any], json.loads(data))
                if chunk.get("usage"):
                    usage = self._parse_usage(chunk["usage"])
                choices = cast(list[dict[str, Any]], chunk.get("choices") or [])
                if choices:
                    delta = cast(dict[str, Any], choices[0].get("delta") or {})
                    content = delta.get("content")
                    if content:
                        yield TextDelta(text=str(content))
        yield StreamEnd(usage=usage)


def chat_model_from_settings(settings: Settings) -> ChatModel:
    return OpenAICompatChat(
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        api_key=settings.chat_api_key,
    )


def judge_model_from_settings(settings: Settings) -> ChatModel | None:
    """None unless judge settings are configured (judge runs are opt-in)."""
    if not (settings.judge_base_url and settings.judge_model):
        return None
    return OpenAICompatChat(
        base_url=settings.judge_base_url,
        model=settings.judge_model,
        api_key=settings.judge_api_key,
    )
