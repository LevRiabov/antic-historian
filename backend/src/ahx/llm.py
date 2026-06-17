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

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, Protocol, cast

import httpx
from pydantic import BaseModel

from ahx.config import Settings

# Transient statuses worth retrying: 429 (rate limit) + the 5xx family. With every
# stage hosted (agent/embed/judge all on OpenRouter), a concurrent run WILL hit
# 429s; an unretried one surfaces as a false refusal (eval-log 2026-06-15 cohere
# incident). Retry makes higher --concurrency safe.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5

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
    # The model that actually produced this stream — set by the concrete model to
    # its own id (and passed through unchanged by CompositeChatModel, so a fallback
    # reports the alternate that served). The pipeline reads this for cost pricing +
    # the `served_by` indicator (6.4); None falls back to the chat's nominal name.
    served_by: str | None = None


StreamEvent = TextDelta | StreamEnd


class ChatResult(BaseModel):
    text: str
    usage: Usage | None
    served_by: str | None = None  # see StreamEnd.served_by (6.4)


class ChatModel(Protocol):
    """Structural interface: anything with these methods is a ChatModel."""

    @property
    def model_name(self) -> str: ...

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]: ...

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult: ...


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

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        stream: bool,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if response_format is not None:
            # llama.cpp converts a JSON schema to a GBNF grammar -> the model
            # CANNOT emit malformed/extra tokens. Reliability for batch jobs.
            payload["response_format"] = response_format
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

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload = self._payload(messages, stream=False, response_format=response_format)
        for attempt in range(_MAX_ATTEMPTS):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, transport=self._transport
                ) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        json=payload,
                        headers=self._headers(),
                    )
                    response.raise_for_status()
                    data = cast(dict[str, Any], response.json())
                text = data["choices"][0]["message"]["content"] or ""
                return ChatResult(
                    text=text,
                    usage=self._parse_usage(data.get("usage")),
                    served_by=self._model,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS - 1:
                    raise
            except (httpx.TransportError, json.JSONDecodeError, KeyError, IndexError):
                # connection reset / read timeout, OR a 200 with an empty or malformed
                # body / missing choices (an OpenRouter hiccup under concurrency) — all
                # transient, all retryable rather than a hard crash mid-run.
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
            # exponential backoff: 0.5, 1, 2, 4s — deterministic (resume-safe, no RNG)
            await asyncio.sleep(0.5 * 2**attempt)
        raise RuntimeError("unreachable: retry loop exhausted without return or raise")

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
        yield StreamEnd(usage=usage, served_by=self._model)


class CompositeChatModel:
    """Cross-provider fallback over an ordered lineup (D5: primary -> alternates on
    distinct providers, so one outage != total outage). The layer ABOVE each model's
    own retry/backoff (OpenAICompatChat): when a model exhausts its retries and raises,
    the composite advances to the next.

    Fallover happens ONLY before the first delta (6.4): a model's stream fails at the
    first `__anext__()` (that is where `raise_for_status` runs), which is before any
    token shipped, so switching is safe. A failure mid-stream — after deltas are out —
    can't be undone, so it propagates. `served_by` is not tracked as instance state
    (the composite is shared across concurrent requests): each inner model stamps its
    own id onto StreamEnd/ChatResult and the composite passes that through unchanged.
    """

    def __init__(self, models: Sequence[ChatModel]) -> None:
        if not models:
            raise ValueError("CompositeChatModel needs at least one model")
        self._models = list(models)

    @property
    def model_name(self) -> str:
        # The lineup's nominal (primary) id; the model that actually served rides on
        # served_by. Used only where the served id isn't yet known (e.g. trace setup).
        return self._models[0].model_name

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        last_exc: Exception | None = None
        for model in self._models:
            gen = model.stream(messages)
            try:
                first = await gen.__anext__()  # HTTP/transport failure surfaces HERE
            except StopAsyncIteration:
                return  # empty stream, but it opened cleanly: this model "served"
            except httpx.HTTPError as exc:
                # __anext__ raising (not StopAsyncIteration) already finalizes the
                # generator, so no explicit aclose is needed before moving on.
                last_exc = exc
                continue  # nothing shipped yet -> fall over to the next provider
            # Committed: the first event arrived without error. From here a failure is
            # mid-stream and must propagate (tokens already shipped, can't switch).
            yield first
            async for event in gen:
                yield event
            return
        assert last_exc is not None  # the loop only exits here via `continue`
        raise last_exc

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        last_exc: Exception | None = None
        for model in self._models:
            try:
                return await model.complete(messages, response_format)
            except httpx.HTTPError as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc


def chat_model_from_settings(settings: Settings) -> ChatModel:
    """The served chat model. With AHX_CHAT_FALLBACKS set, the primary is wrapped in a
    CompositeChatModel over [primary, *fallbacks]; otherwise the bare primary (whose own
    served_by stamping makes the wrap unnecessary when there is nothing to fall over to)."""
    primary = OpenAICompatChat(
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        api_key=settings.chat_api_key,
        temperature=settings.chat_temperature,
    )
    if not settings.chat_fallbacks:
        return primary
    alternates = [
        OpenAICompatChat(
            base_url=ep.base_url,
            model=ep.model,
            api_key=ep.api_key,
            temperature=settings.chat_temperature,
        )
        for ep in settings.chat_fallbacks
    ]
    return CompositeChatModel([primary, *alternates])


def judge_model_from_settings(settings: Settings) -> ChatModel | None:
    """None unless judge settings are configured (judge runs are opt-in)."""
    if not (settings.judge_base_url and settings.judge_model):
        return None
    return OpenAICompatChat(
        base_url=settings.judge_base_url,
        model=settings.judge_model,
        api_key=settings.judge_api_key,
    )


def attribution_judge_from_settings(settings: Settings) -> ChatModel | None:
    """A separate, stronger judge for the attribution rubric only (the rubric a
    flash judge can't score stably). None unless AHX_ATTRIB_JUDGE_* is set — the
    caller then falls back to the main judge for attribution too."""
    if not (settings.attrib_judge_base_url and settings.attrib_judge_model):
        return None
    return OpenAICompatChat(
        base_url=settings.attrib_judge_base_url,
        model=settings.attrib_judge_model,
        api_key=settings.attrib_judge_api_key,
    )
