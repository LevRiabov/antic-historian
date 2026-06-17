"""The ask pipeline: retrieve -> prompt -> stream -> audit.

One engine, two consumers: the API maps these events onto SSE (3.4), the
generation eval drives it directly per golden-set question (3.5) — same
eval-equals-served guarantee as retrieval/dense.py.

Retrieval is injected as a callable Protocol, not imported: the API binds
dense_retrieve_async with functools.partial, tests pass a fake, and every
Phase 4 ablation is "pass a different retriever" with no pipeline change.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from pydantic import BaseModel

from ahx.generation.citations import (
    Citation,
    MarkerAudit,
    citations_from_chunks,
    extract_markers,
)
from ahx.generation.prompt import PROMPT_VERSION, REFUSAL_TEXT, build_messages
from ahx.llm import ChatModel, TextDelta, Usage
from ahx.pricing import Cost, cost_for, load_price_table
from ahx.retrieval.dense import RetrievedChunk


class Retriever(Protocol):
    async def __call__(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


class SourcesEvent(BaseModel):
    citations: list[Citation]
    prompt_version: str


class DeltaEvent(BaseModel):
    text: str


class DoneEvent(BaseModel):
    answer: str
    refused: bool
    markers: MarkerAudit
    usage: Usage | None
    # Generation cost (6.2). Optional/defaulted so non-API callers and the agent
    # path stay valid; the single-shot pipeline always fills it.
    cost: Cost | None = None
    # The model that actually served this answer (6.4). With a CompositeChatModel this
    # is the alternate when the primary fell over — the SSE indicator + trace key on it.
    served_by: str | None = None
    # Set by the security guard (6.3) when a request was blocked/redacted — distinct
    # from a content `refused` so the client + traces can flag a security event.
    blocked: bool = False


AskEvent = SourcesEvent | DeltaEvent | DoneEvent


async def ask(
    question: str,
    retriever: Retriever,
    chat: ChatModel,
    top_k: int = 5,
) -> AsyncIterator[AskEvent]:
    chunks = await retriever(question, top_k)
    citations = citations_from_chunks(chunks)
    yield SourcesEvent(citations=citations, prompt_version=PROMPT_VERSION)

    pieces: list[str] = []
    usage: Usage | None = None
    served_by: str | None = None
    async for event in chat.stream(build_messages(question, chunks)):
        if isinstance(event, TextDelta):
            pieces.append(event.text)
            yield DeltaEvent(text=event.text)
        else:
            usage = event.usage
            served_by = event.served_by

    # Price by the model that ACTUALLY served (a fallback alternate, not the nominal
    # primary chat.model_name) — wrong pricing otherwise once a composite falls over.
    answer_model = served_by or chat.model_name
    answer = "".join(pieces).strip()
    yield DoneEvent(
        answer=answer,
        refused=_is_refusal(answer),
        markers=extract_markers(answer, {c.marker for c in citations}),
        usage=usage,
        cost=cost_for(answer_model, usage, load_price_table()),
        served_by=answer_model,
    )


def _is_refusal(answer: str) -> bool:
    """Tolerate quote wrapping and trailing whitespace, nothing fancier —
    sloppier refusals should COST the model in the eval, not be absorbed."""
    return answer.strip().strip('"').strip() == REFUSAL_TEXT


def collect(events: Sequence[AskEvent]) -> tuple[SourcesEvent, DoneEvent]:
    """Eval-harness helper: pull the two bookend events out of a finished run."""
    sources = next(e for e in events if isinstance(e, SourcesEvent))
    done = next(e for e in events if isinstance(e, DoneEvent))
    return sources, done
