"""CompositeChatModel — cross-provider fallover (6.4).

Two layers of coverage: the switching LOGIC against scripted fake models (fast, exact
control over where a model fails), and one real httpx.MockTransport fallover proving the
HTTP path behaves the same. The invariant under test: fall over only BEFORE the first
delta; a mid-stream failure propagates; `served_by` reports the model that actually served.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import pytest

from ahx.config import ChatEndpoint, Settings
from ahx.llm import (
    ChatMessage,
    ChatModel,
    ChatResult,
    CompositeChatModel,
    OpenAICompatChat,
    StreamEnd,
    TextDelta,
    chat_model_from_settings,
)

MESSAGES = [ChatMessage(role="user", content="How did Caesar die?")]


class ScriptedModel:
    """A ChatModel whose failure point is scriptable: `fail_at_start` raises before any
    delta (the fall-over-able case); `fail_after` raises mid-stream after N deltas."""

    def __init__(
        self,
        name: str,
        *,
        fail_at_start: bool = False,
        fail_after: int | None = None,
        deltas: Sequence[str] = ("a", "b"),
    ) -> None:
        self._name = name
        self._fail_at_start = fail_at_start
        self._fail_after = fail_after
        self._deltas = list(deltas)

    @property
    def model_name(self) -> str:
        return self._name

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[Any]:
        if self._fail_at_start:
            raise httpx.ConnectError("provider down")
        for i, delta in enumerate(self._deltas):
            if i == self._fail_after:
                raise httpx.ReadError("connection dropped mid-stream")
            yield TextDelta(text=delta)
        yield StreamEnd(usage=None, served_by=self._name)

    async def complete(
        self, messages: Sequence[ChatMessage], response_format: dict[str, Any] | None = None
    ) -> ChatResult:
        if self._fail_at_start:
            raise httpx.ConnectError("provider down")
        return ChatResult(text=f"ok-{self._name}", usage=None, served_by=self._name)


def test_composite_satisfies_chatmodel_protocol() -> None:
    model: ChatModel = CompositeChatModel([ScriptedModel("primary")])
    assert model.model_name == "primary"  # nominal id = the primary's


async def test_stream_falls_over_before_first_delta() -> None:
    composite = CompositeChatModel(
        [ScriptedModel("primary", fail_at_start=True), ScriptedModel("alt")]
    )
    events = [e async for e in composite.stream(MESSAGES)]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["a", "b"]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.served_by == "alt"  # the alternate that actually served


async def test_stream_midstream_failure_propagates_no_switch() -> None:
    # primary ships one delta then dies; the alternate must NOT be tried (tokens shipped).
    composite = CompositeChatModel(
        [ScriptedModel("primary", fail_after=1, deltas=["x", "y", "z"]), ScriptedModel("alt")]
    )
    seen: list[str] = []
    with pytest.raises(httpx.ReadError):
        async for event in composite.stream(MESSAGES):
            if isinstance(event, TextDelta):
                seen.append(event.text)
    assert seen == ["x"]  # the one delta before the drop; no "a"/"b" from the alternate


async def test_stream_all_providers_down_raises_last_error() -> None:
    composite = CompositeChatModel(
        [ScriptedModel("a", fail_at_start=True), ScriptedModel("b", fail_at_start=True)]
    )
    with pytest.raises(httpx.HTTPError):
        _ = [e async for e in composite.stream(MESSAGES)]


async def test_complete_falls_over_and_reports_served_by() -> None:
    composite = CompositeChatModel(
        [ScriptedModel("primary", fail_at_start=True), ScriptedModel("alt")]
    )
    result = await composite.complete(MESSAGES)
    assert result.text == "ok-alt"
    assert result.served_by == "alt"


async def test_complete_all_down_raises() -> None:
    composite = CompositeChatModel(
        [ScriptedModel("a", fail_at_start=True), ScriptedModel("b", fail_at_start=True)]
    )
    with pytest.raises(httpx.HTTPError):
        await composite.complete(MESSAGES)


def test_empty_lineup_rejected() -> None:
    with pytest.raises(ValueError):
        CompositeChatModel([])


# --- the real HTTP path (MockTransport), to prove OpenAICompatChat falls over identically ---

_SSE_BODY = (
    'data: {"choices":[{"delta":{"content":"Et tu, "}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Brute?"}}]}\n\n'
    "data: [DONE]\n\n"
)


async def test_real_transport_stream_fallover_reports_alternate() -> None:
    # stream() does NOT retry (only complete() does), so a 503 raises immediately on the
    # first __anext__ — exactly the before-first-delta point the composite switches at.
    down = OpenAICompatChat(
        base_url="http://primary/v1",
        model="primary-model",
        transport=httpx.MockTransport(lambda _req: httpx.Response(503)),
    )
    up = OpenAICompatChat(
        base_url="http://alt/v1",
        model="alt-model",
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(
                200, text=_SSE_BODY, headers={"content-type": "text/event-stream"}
            )
        ),
    )
    events = [e async for e in CompositeChatModel([down, up]).stream(MESSAGES)]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Et tu, ", "Brute?"]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.served_by == "alt-model"


# --- factory wiring: bare model vs composite, driven by AHX_CHAT_FALLBACKS ---


def test_factory_returns_bare_model_without_fallbacks() -> None:
    model = chat_model_from_settings(Settings(chat_fallbacks=[]))
    assert isinstance(model, OpenAICompatChat)


def test_factory_wraps_composite_with_fallbacks() -> None:
    settings = Settings(
        chat_model="primary-model",
        chat_fallbacks=[ChatEndpoint(base_url="http://alt/v1", model="alt-model")],
    )
    model = chat_model_from_settings(settings)
    assert isinstance(model, CompositeChatModel)
    assert model.model_name == "primary-model"  # primary stays the nominal id
