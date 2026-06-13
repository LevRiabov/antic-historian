"""Retrieval-tier evaluation: recall@k and MRR against the golden set.

The cheap tier (no LLM, no cost beyond local query embeddings) — run it on
every retrieval change. The judge tier (faithfulness/completeness/refusal)
lives elsewhere and runs at phase boundaries.

Hit rule (carried from rag-historian, chunking-invariant): a retrieved chunk
COVERS a gold span iff it is from the same work and contains the span's
midpoint.

Recall is per *requirement group*, not per span (see docs/golden-set.md §4a).
Each span declares the answer requirement(s) it satisfies via `groups`; spans
sharing a label are alternatives (any one covers that requirement), so a fact
attested in five works counts once, not five times. A span with no label is its
own singleton requirement (the back-compatible default). Question recall@k =
requirement groups with ≥1 covering span in top-k / total requirement groups;
category and overall numbers are means over questions. MRR uses the rank of the
first chunk covering ANY of the question's spans.

Every run is persisted as a typed, versioned record (rule #5) under
backend/evals/runs/. Record layout mirrors rag-historian's results format
(the proven, familiar shape): aggregates first (overall + by-category),
then per-question results with gold_chunk_ids vs retrieved_chunk_ids as
compact, directly comparable id arrays. gold_chunk_ids is informational
(chunks overlap, so a span midpoint may live in 1-2 chunks); scoring is
always span-midpoint based, never id based.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ahx.config import Settings
from ahx.db import ChunkRow
from ahx.evals.golden import (
    CATEGORIES,
    Category,
    GoldenQuestion,
    ResolutionError,
    ResolvedSpan,
    resolve_span,
)
from ahx.retrieval.dense import RetrievedChunk, dense_retrieve
from ahx.retrieval.embedding import EmbeddingClient

K_VALUES = (1, 5, 10, 20)


class GoldSpanRef(BaseModel):
    pg_id: int
    char_start: int
    char_end: int
    groups: list[str] = []


class QuestionResult(BaseModel):
    question_id: str
    category: Category
    question: str
    ideal_answer: str
    notes: str = ""
    gold_spans: list[GoldSpanRef]
    gold_chunk_ids: list[int]  # chunks covering any gold span (1-2 per span: overlap)
    retrieved_chunk_ids: list[int]
    similarities: list[float]
    latency_ms: int
    recall: dict[int, float]  # k -> recall@k for this question
    first_hit_rank: int | None
    mrr: float  # reciprocal rank of the first hit


class CategoryAggregate(BaseModel):
    count: int
    recall: dict[int, float]
    mrr: float


class Aggregates(BaseModel):
    recall: dict[int, float]  # mean over all questions
    mrr: float
    by_category: dict[str, CategoryAggregate]


class RetrievalRun(BaseModel):
    created_at: str
    retriever: str
    embed_model: str
    chunking_version: str
    top_k: int
    aggregates: Aggregates  # declared before results -> serialized on top
    results: list[QuestionResult]


def _covers(chunk: RetrievedChunk, span: ResolvedSpan) -> bool:
    if chunk.pg_id != span.pg_id:
        return False
    midpoint = (span.char_start + span.char_end) // 2
    return chunk.char_start <= midpoint < chunk.char_end


def _requirement_groups(spans: list[ResolvedSpan]) -> dict[str, list[int]]:
    """Map each requirement label to the span indices that satisfy it. A span
    with no `groups` is its own singleton requirement (back-compat: every gold
    span independently required); spans sharing a label are alternatives; a span
    may belong to several requirements at once."""
    groups: dict[str, list[int]] = {}
    for i, span in enumerate(spans):
        labels = span.groups or [f"__singleton_{i}"]
        for label in labels:
            groups.setdefault(label, []).append(i)
    return groups


def score_question(
    question: GoldenQuestion,
    spans: list[ResolvedSpan],
    retrieved: list[RetrievedChunk],
    gold_chunk_ids: list[int],
    latency_ms: int = 0,
) -> QuestionResult:
    span_hits = [next((c.rank for c in retrieved if _covers(c, span)), None) for span in spans]
    hit_ranks = [rank for rank in span_hits if rank is not None]
    first_hit = min(hit_ranks) if hit_ranks else None
    groups = _requirement_groups(spans)
    total = len(groups)

    def covered_groups(k: int) -> int:
        count = 0
        for members in groups.values():
            for i in members:
                rank = span_hits[i]
                if rank is not None and rank <= k:
                    count += 1
                    break
        return count

    recall = {k: (covered_groups(k) / total) if total else 0.0 for k in K_VALUES}
    return QuestionResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        ideal_answer=question.ideal_answer,
        notes=question.notes,
        gold_spans=[
            GoldSpanRef(
                pg_id=s.pg_id, char_start=s.char_start, char_end=s.char_end, groups=s.groups
            )
            for s in spans
        ],
        gold_chunk_ids=gold_chunk_ids,
        retrieved_chunk_ids=[c.chunk_id for c in retrieved],
        similarities=[round(c.score, 4) for c in retrieved],
        latency_ms=latency_ms,
        recall=recall,
        first_hit_rank=first_hit,
        mrr=1.0 / first_hit if first_hit else 0.0,
    )


def compute_aggregates(results: list[QuestionResult]) -> Aggregates:
    def mean_block(subset: list[QuestionResult]) -> tuple[dict[int, float], float]:
        recall = {k: sum(q.recall[k] for q in subset) / len(subset) for k in K_VALUES}
        mrr = sum(q.mrr for q in subset) / len(subset)
        return recall, mrr

    by_category: dict[str, CategoryAggregate] = {}
    for category in CATEGORIES:
        subset = [q for q in results if q.category == category]
        if not subset:
            continue
        recall, mrr = mean_block(subset)
        by_category[category] = CategoryAggregate(count=len(subset), recall=recall, mrr=mrr)

    empty_recall: dict[int, float] = dict.fromkeys(K_VALUES, 0.0)
    overall_recall, overall_mrr = mean_block(results) if results else (empty_recall, 0.0)
    return Aggregates(recall=overall_recall, mrr=overall_mrr, by_category=by_category)


def _gold_chunk_ids(engine: Engine, spans: list[ResolvedSpan], chunking_version: str) -> list[int]:
    ids: set[int] = set()
    with Session(engine) as session:
        for span in spans:
            midpoint = (span.char_start + span.char_end) // 2
            rows = session.scalars(
                select(ChunkRow.id).where(
                    ChunkRow.pg_id == span.pg_id,
                    ChunkRow.chunking_version == chunking_version,
                    ChunkRow.char_start <= midpoint,
                    ChunkRow.char_end > midpoint,
                )
            ).all()
            ids.update(rows)
    return sorted(ids)


def run_retrieval_eval(
    settings: Settings,
    questions: list[GoldenQuestion],
    retriever_name: str = "dense-v1",
    top_k: int = 20,
) -> RetrievalRun:
    from ahx.db import create_sync_engine
    from ahx.ingest.chunker import CHUNKING_VERSION

    engine = create_sync_engine(settings.database_url)
    embedder = EmbeddingClient(settings)
    results: list[QuestionResult] = []
    for question in questions:
        if question.category == "out-of-scope":
            continue
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
        retrieved = dense_retrieve(engine, embedder, question.question, top_k)
        latency_ms = round((time.perf_counter() - started) * 1000)
        gold_ids = _gold_chunk_ids(engine, spans, CHUNKING_VERSION)
        results.append(score_question(question, spans, retrieved, gold_ids, latency_ms))

    return RetrievalRun(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        retriever=retriever_name,
        embed_model=settings.embed_model,
        chunking_version=CHUNKING_VERSION,
        top_k=top_k,
        aggregates=compute_aggregates(results),
        results=results,
    )


def save_run(run: RetrievalRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = run.created_at.replace(":", "-").replace("+00-00", "Z")
    path = runs_dir / f"{stamp}-{run.retriever}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path
