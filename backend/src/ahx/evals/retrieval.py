"""Retrieval-tier evaluation: recall@k and MRR against the golden set.

The cheap tier (no LLM, no cost beyond local query embeddings) — run it on
every retrieval change. The judge tier (faithfulness/completeness/refusal)
lives elsewhere and runs at phase boundaries.

Hit rule (carried from rag-historian, chunking-invariant): a retrieved chunk
COVERS a gold span iff it is from the same work and contains the span's
midpoint. Question recall@k = covered spans / total spans within top-k;
category and overall numbers are means over questions. MRR uses the rank of
the first chunk covering ANY of the question's spans.

Every run is persisted as a typed, versioned record (rule #5) under
backend/evals/runs/ for forensics and case-study receipts.
"""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ahx.config import Settings
from ahx.evals.golden import (
    Category,
    GoldenQuestion,
    ResolutionError,
    ResolvedSpan,
    resolve_span,
)
from ahx.retrieval.embedding import EmbeddingClient

K_VALUES = (1, 5, 10, 20)


class RetrievedRef(BaseModel):
    """Minimal slice of a retrieved chunk needed for span coverage."""

    pg_id: int
    char_start: int
    char_end: int
    rank: int  # 1-based


class QuestionResult(BaseModel):
    question_id: str
    category: Category
    spans_total: int
    covered_at: dict[int, int]  # k -> spans covered within top-k
    first_hit_rank: int | None  # for MRR

    def recall_at(self, k: int) -> float:
        return self.covered_at.get(k, 0) / self.spans_total if self.spans_total else 0.0

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_hit_rank if self.first_hit_rank else 0.0


class CategorySummary(BaseModel):
    category: Category
    questions: int
    recall: dict[int, float]  # k -> mean recall
    mrr: float


class RetrievalRun(BaseModel):
    created_at: str
    retriever: str
    embed_model: str
    chunking_version: str
    top_k: int
    questions: list[QuestionResult]

    def summarize(self, category: Category | None = None) -> CategorySummary | None:
        subset = [q for q in self.questions if category is None or q.category == category]
        if not subset:
            return None
        return CategorySummary(
            category=category or "literal",  # overall rows pass category=None; field unused
            questions=len(subset),
            recall={k: sum(q.recall_at(k) for q in subset) / len(subset) for k in K_VALUES},
            mrr=sum(q.reciprocal_rank for q in subset) / len(subset),
        )


def _covers(ref: RetrievedRef, span: ResolvedSpan) -> bool:
    if ref.pg_id != span.pg_id:
        return False
    midpoint = (span.char_start + span.char_end) // 2
    return ref.char_start <= midpoint < ref.char_end


def score_question(
    question_id: str,
    category: Category,
    spans: list[ResolvedSpan],
    retrieved: list[RetrievedRef],
) -> QuestionResult:
    first_hit: int | None = None
    span_first_hit: list[int | None] = []
    for span in spans:
        hit_rank = next((r.rank for r in retrieved if _covers(r, span)), None)
        span_first_hit.append(hit_rank)
        if hit_rank is not None and (first_hit is None or hit_rank < first_hit):
            first_hit = hit_rank
    covered_at = {
        k: sum(1 for rank in span_first_hit if rank is not None and rank <= k) for k in K_VALUES
    }
    return QuestionResult(
        question_id=question_id,
        category=category,
        spans_total=len(spans),
        covered_at=covered_at,
        first_hit_rank=first_hit,
    )


def dense_retrieve(
    settings: Settings, embedder: EmbeddingClient, query: str, top_k: int
) -> list[RetrievedRef]:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from ahx.db import ChunkRow, create_sync_engine

    vector = embedder.embed_query_sync(query)
    engine = create_sync_engine(settings.database_url)
    with Session(engine) as session:
        distance = ChunkRow.embedding.cosine_distance(vector)  # pyright: ignore[reportAttributeAccessIssue]
        rows = session.execute(
            select(ChunkRow.pg_id, ChunkRow.char_start, ChunkRow.char_end)
            .order_by(distance)
            .limit(top_k)
        ).all()
    return [
        RetrievedRef(pg_id=pg_id, char_start=start, char_end=end, rank=rank)
        for rank, (pg_id, start, end) in enumerate(rows, start=1)
    ]


def run_retrieval_eval(
    settings: Settings,
    questions: list[GoldenQuestion],
    retriever_name: str = "dense-v1",
    top_k: int = 20,
) -> RetrievalRun:
    from ahx.ingest.chunker import CHUNKING_VERSION

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
        retrieved = dense_retrieve(settings, embedder, question.question, top_k)
        results.append(score_question(question.id, question.category, spans, retrieved))

    return RetrievalRun(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        retriever=retriever_name,
        embed_model=settings.embed_model,
        chunking_version=CHUNKING_VERSION,
        top_k=top_k,
        questions=results,
    )


def save_run(run: RetrievalRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = run.created_at.replace(":", "-").replace("+00-00", "Z")
    path = runs_dir / f"{stamp}-{run.retriever}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path
