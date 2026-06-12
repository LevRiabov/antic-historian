"""Generation-tier evaluation: the full ask pipeline against the golden set.

Two layers (docs/golden-set.md cost policy):

1. **Mechanical** — free, runs on every change: citation span recall (gold
   spans covered by chunks the model actually CITED, midpoint rule),
   citation precision (used markers that point at a gold-covering chunk),
   refusal accuracy (out-of-scope questions are only measurable here),
   latency and token usage.
2. **Judge** — LLM-scored faithfulness (claims supported by cited sources)
   and completeness (vs ideal_answer), 1-5 rubrics. Optional; runs at phase
   boundaries with a strong judge (weak judges miscalibrate — known footgun).

Run records mirror the retrieval-tier layout: aggregates first, then
per-question results with ideal_answer next to the model's actual answer.
"""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from pydantic import BaseModel, Field

from ahx.config import Settings
from ahx.db import create_async_db_engine
from ahx.evals.golden import (
    CATEGORIES,
    Category,
    GoldenQuestion,
    ResolutionError,
    ResolvedSpan,
    resolve_span,
)
from ahx.generation.citations import Citation
from ahx.generation.pipeline import DoneEvent, SourcesEvent, ask, collect
from ahx.generation.prompt import PROMPT_VERSION
from ahx.llm import ChatMessage, ChatModel, chat_model_from_settings
from ahx.retrieval.dense import dense_retrieve_async
from ahx.retrieval.embedding import EmbeddingClient


class GenQuestionResult(BaseModel):
    question_id: str
    category: Category
    question: str
    ideal_answer: str
    answer: str
    refused: bool
    refusal_expected: bool  # True only for out-of-scope questions
    refusal_correct: bool
    markers_used: list[int]
    markers_dangling: list[int]
    retrieved_chunk_ids: list[int]
    cited_chunk_ids: list[int]
    citation_span_recall: float | None  # None for out-of-scope (no gold spans)
    citation_precision: float | None  # None when no markers were used
    faithfulness: int | None = None  # 1-5, judge layer
    completeness: int | None = None  # 1-5, judge layer
    judge_notes: str = ""
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None


class GenCategoryAggregate(BaseModel):
    count: int
    refused: int
    refusal_correct: float  # fraction
    citation_span_recall: float | None
    citation_precision: float | None
    faithfulness: float | None
    completeness: float | None
    mean_latency_ms: int


class GenAggregates(BaseModel):
    questions: int
    refusal_accuracy_oos: float | None  # refused / out-of-scope count
    false_refusal_rate: float  # refused in-scope / in-scope count
    citation_span_recall: float | None  # mean over in-scope questions
    citation_precision: float | None
    faithfulness: float | None
    completeness: float | None
    mean_latency_ms: int
    mean_completion_tokens: float | None
    by_category: dict[str, GenCategoryAggregate]


class GenerationRun(BaseModel):
    created_at: str
    label: str
    chat_model: str
    embed_model: str
    chunking_version: str
    prompt_version: str
    top_k: int
    judge_model: str | None
    judge_rubric: str | None = None  # None on pre-judge-v2 records
    aggregates: GenAggregates
    results: list[GenQuestionResult]


def _cited_covers_span(sources: SourcesEvent, markers: list[int], span: ResolvedSpan) -> bool:
    midpoint = (span.char_start + span.char_end) // 2
    for citation in sources.citations:
        if citation.marker not in markers:
            continue
        if citation.pg_id == span.pg_id and citation.char_start <= midpoint < citation.char_end:
            return True
    return False


def score_generation(
    question: GoldenQuestion,
    spans: list[ResolvedSpan],
    sources: SourcesEvent,
    done: DoneEvent,
    latency_ms: int,
) -> GenQuestionResult:
    used = done.markers.used
    by_marker = {c.marker: c for c in sources.citations}

    citation_span_recall: float | None = None
    if spans:
        covered = sum(1 for s in spans if _cited_covers_span(sources, used, s))
        citation_span_recall = covered / len(spans)

    citation_precision: float | None = None
    if used and spans:
        precise = sum(
            1 for marker in used if any(_cited_covers_span(sources, [marker], s) for s in spans)
        )
        citation_precision = precise / len(used)

    refusal_expected = question.category == "out-of-scope"
    return GenQuestionResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        ideal_answer=question.ideal_answer,
        answer=done.answer,
        refused=done.refused,
        refusal_expected=refusal_expected,
        refusal_correct=done.refused == refusal_expected,
        markers_used=used,
        markers_dangling=done.markers.dangling,
        retrieved_chunk_ids=[c.chunk_id for c in sources.citations],
        cited_chunk_ids=[by_marker[m].chunk_id for m in used],
        citation_span_recall=citation_span_recall,
        citation_precision=citation_precision,
        latency_ms=latency_ms,
        prompt_tokens=done.usage.prompt_tokens if done.usage else None,
        completion_tokens=done.usage.completion_tokens if done.usage else None,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_gen_aggregates(results: list[GenQuestionResult]) -> GenAggregates:
    def block(subset: list[GenQuestionResult]) -> GenCategoryAggregate:
        return GenCategoryAggregate(
            count=len(subset),
            refused=sum(1 for r in subset if r.refused),
            refusal_correct=sum(1 for r in subset if r.refusal_correct) / len(subset),
            citation_span_recall=_mean(
                [r.citation_span_recall for r in subset if r.citation_span_recall is not None]
            ),
            citation_precision=_mean(
                [r.citation_precision for r in subset if r.citation_precision is not None]
            ),
            faithfulness=_mean([float(r.faithfulness) for r in subset if r.faithfulness]),
            completeness=_mean([float(r.completeness) for r in subset if r.completeness]),
            mean_latency_ms=round(sum(r.latency_ms for r in subset) / len(subset)),
        )

    by_category = {
        category: block(subset)
        for category in CATEGORIES
        if (subset := [r for r in results if r.category == category])
    }

    oos = [r for r in results if r.refusal_expected]
    in_scope = [r for r in results if not r.refusal_expected]
    overall = block(results)
    return GenAggregates(
        questions=len(results),
        refusal_accuracy_oos=(sum(1 for r in oos if r.refused) / len(oos)) if oos else None,
        false_refusal_rate=(sum(1 for r in in_scope if r.refused) / len(in_scope))
        if in_scope
        else 0.0,
        citation_span_recall=overall.citation_span_recall,
        citation_precision=overall.citation_precision,
        faithfulness=overall.faithfulness,
        completeness=overall.completeness,
        mean_latency_ms=overall.mean_latency_ms,
        mean_completion_tokens=_mean(
            [float(r.completion_tokens) for r in results if r.completion_tokens is not None]
        ),
        by_category=by_category,
    )


# --- judge layer ---


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = ""


# Rubric history (rule #5 — judge changes are measured, see eval-log):
# judge-v1: judge saw only CITED chunks -> correct-but-miscited answers scored
#   like fabrications, double-counting what citation_precision already measures.
# judge-v2: judge sees ALL retrieved passages exactly as the answer model did;
#   misattribution of grounded content caps at 4, invention is the real failure.
JUDGE_RUBRIC_VERSION = "judge-v2"

FAITHFULNESS_RUBRIC = """You are grading a RAG system's answer for FAITHFULNESS: did the
model invent content, or is everything grounded in the source passages it was shown?

Below are ALL passages the model saw, numbered exactly as shown to it; the ones it
actually cited are flagged "(cited)". Score 1-5:
5 = every claim is supported by SOME passage below, even one it failed to cite;
4 = grounded, but a claim is attributed to the wrong author/source or wrong marker;
3 = mostly grounded, some unsupported embellishment;
1 = substantial invented content found in no passage below.
Wrong citation markers alone must NOT take the score below 4 — citation accuracy is
measured separately. Invented content is the failure this score exists to catch.

Question: {question}

Source passages:
{sources}

Answer to grade:
{answer}

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

COMPLETENESS_RUBRIC = """You are grading a RAG system's answer for COMPLETENESS
against a reference answer.
Score 1-5: 5 = covers all key facts of the reference; 3 = covers the core but misses
secondary facts; 1 = misses the point. Extra correct detail must not lower the score.

Question: {question}

Reference answer:
{ideal}

Answer to grade:
{answer}

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def parse_verdict(raw: str) -> JudgeVerdict | None:
    """Tolerant of code fences / prose around the JSON object."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return JudgeVerdict.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValueError):
        return None


async def judge_question(
    judge: ChatModel, result: GenQuestionResult, citations: list[Citation]
) -> None:
    """Mutates result with faithfulness/completeness scores (in-scope, answered only).

    The judge sees ALL retrieved passages (what the answer model saw), with
    cited ones flagged — judge-v2; see rubric history above.
    """
    if result.refused or result.refusal_expected:
        return
    sources_text = "\n\n".join(
        f"[{c.marker}]{' (cited)' if c.marker in result.markers_used else ''} "
        f"{c.author}, {c.work_title}:\n{c.text}"
        for c in citations
    )
    notes: list[str] = []
    for field, prompt in (
        (
            "faithfulness",
            FAITHFULNESS_RUBRIC.format(
                question=result.question,
                sources=sources_text or "(none cited)",
                answer=result.answer,
            ),
        ),
        (
            "completeness",
            COMPLETENESS_RUBRIC.format(
                question=result.question, ideal=result.ideal_answer, answer=result.answer
            ),
        ),
    ):
        response = await judge.complete([ChatMessage(role="user", content=prompt)])
        verdict = parse_verdict(response.text)
        if verdict is None:
            notes.append(f"{field}: unparseable judge reply")
            continue
        setattr(result, field, verdict.score)
        notes.append(f"{field}: {verdict.reason}")
    result.judge_notes = " | ".join(notes)


# --- runner ---


async def run_generation_eval(
    settings: Settings,
    questions: list[GoldenQuestion],
    label: str = "gen-baseline-v1",
    top_k: int = 5,
    judge: ChatModel | None = None,
    on_result: Callable[[GenQuestionResult], None] | None = None,
) -> GenerationRun:
    from ahx.ingest.chunker import CHUNKING_VERSION

    engine = create_async_db_engine(settings.database_url)
    retriever = partial(dense_retrieve_async, engine, EmbeddingClient(settings))
    chat = chat_model_from_settings(settings)

    results: list[GenQuestionResult] = []
    try:
        for question in questions:
            spans: list[ResolvedSpan] = []
            for span in question.gold_spans:
                resolved = resolve_span(span, settings.corpus_normalized_dir, question.id)
                if isinstance(resolved, ResolutionError):
                    raise ValueError(
                        f"{question.id}: unresolved gold span ({resolved.problem}) — "
                        "run `ahx eval validate` first"
                    )
                spans.append(resolved)

            started = time.perf_counter()
            events = [event async for event in ask(question.question, retriever, chat, top_k)]
            latency_ms = round((time.perf_counter() - started) * 1000)
            sources, done = collect(events)

            result = score_generation(question, spans, sources, done, latency_ms)
            if judge is not None:
                await judge_question(judge, result, sources.citations)
            results.append(result)
            if on_result is not None:
                on_result(result)
    finally:
        await engine.dispose()

    return GenerationRun(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        label=label,
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        chunking_version=CHUNKING_VERSION,
        prompt_version=PROMPT_VERSION,
        top_k=top_k,
        judge_model=judge.model_name if judge else None,
        judge_rubric=JUDGE_RUBRIC_VERSION if judge else None,
        aggregates=compute_gen_aggregates(results),
        results=results,
    )


async def rejudge_run(
    settings: Settings,
    record_path: Path,
    judge: ChatModel,
    label: str,
    on_result: Callable[[GenQuestionResult], None] | None = None,
) -> GenerationRun:
    """Re-score a saved run's FROZEN answers with the current judge/rubric.

    Isolates judge changes from generation nondeterminism (rule #5): answers
    stay byte-identical to the source record; only judge fields move. Chunk
    texts are refetched by id; markers are reconstructed from rank order
    (marker n == retrieved_chunk_ids[n-1], the prompt's numbering invariant).
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from ahx.db import ChunkRow, SourceRow, create_sync_engine

    run = GenerationRun.model_validate_json(record_path.read_text(encoding="utf-8"))

    needed = {cid for r in run.results for cid in r.retrieved_chunk_ids}
    engine = create_sync_engine(settings.database_url)
    with Session(engine) as session:
        rows = session.execute(
            select(ChunkRow, SourceRow.author, SourceRow.title)
            .join(SourceRow, SourceRow.pg_id == ChunkRow.pg_id)
            .where(ChunkRow.id.in_(needed))
        ).all()
    by_id = {
        chunk.id: Citation(
            marker=0,  # placeholder; set per-question from rank order below
            chunk_id=chunk.id,
            pg_id=chunk.pg_id,
            author=author,
            work_title=title,
            locator=chunk.locator,
            text=chunk.text,
            score=0.0,  # not stored in generation records; unused by the judge
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for chunk, author, title in rows
    }
    missing = needed - by_id.keys()
    if missing:
        raise ValueError(f"chunks no longer in DB (chunking changed?): {sorted(missing)[:5]}")

    for result in run.results:
        citations = [
            by_id[cid].model_copy(update={"marker": rank})
            for rank, cid in enumerate(result.retrieved_chunk_ids, start=1)
        ]
        result.faithfulness = None
        result.completeness = None
        result.judge_notes = ""
        await judge_question(judge, result, citations)
        if on_result is not None:
            on_result(result)

    run.created_at = datetime.now(UTC).isoformat(timespec="seconds")
    run.label = label
    run.judge_model = judge.model_name
    run.judge_rubric = JUDGE_RUBRIC_VERSION
    run.aggregates = compute_gen_aggregates(run.results)
    return run


def save_generation_run(run: GenerationRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = run.created_at.replace(":", "-").replace("+00-00", "Z")
    path = runs_dir / f"{stamp}-{run.label}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path
