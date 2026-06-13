"""Generation-tier evaluation: the full ask pipeline against the golden set.

Two layers (docs/golden-set.md cost policy):

1. **Mechanical** — free, runs on every change: citation span recall (gold
   spans covered by chunks the model actually CITED, midpoint rule),
   citation precision (used markers that point at a gold-covering chunk),
   refusal accuracy (out-of-scope questions are only measurable here),
   latency and token usage.
2. **Judge** — LLM-scored faithfulness (claims supported by cited sources),
   completeness (vs ideal_answer), and attribution (surfacing disagreement +
   naming sources where the policy requires it), 1-5 rubrics. Optional; runs at
   phase boundaries with a strong judge (weak judges miscalibrate — known footgun).

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
    refused: bool  # mechanical: answer == the exact contract sentence
    refused_semantic: bool | None = None  # judge yes/no (judge-v3.1): accepts a
    # paraphrased abstention as a refusal; None until/unless the judge layer runs
    refusal_expected: bool  # True only for out-of-scope questions
    refusal_correct: bool  # uses refused_semantic when judged, else refused
    markers_used: list[int]
    markers_dangling: list[int]
    retrieved_chunk_ids: list[int]
    cited_chunk_ids: list[int]
    citation_span_recall: float | None  # None for out-of-scope (no gold spans)
    citation_precision: float | None  # None when no markers were used
    faithfulness: int | None = None  # 1-5, judge layer
    completeness: int | None = None  # 1-5, judge layer
    attribution: int | None = None  # 1-5, judge layer (judge-v3): surfacing
    # disagreement + naming sources where the policy requires it
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
    attribution: float | None = None  # judge-v3; absent on older run records
    mean_latency_ms: int


class GenAggregates(BaseModel):
    questions: int
    refusal_accuracy_oos: float | None  # refused / out-of-scope count
    false_refusal_rate: float  # refused in-scope / in-scope count
    citation_span_recall: float | None  # mean over in-scope questions
    citation_precision: float | None
    faithfulness: float | None
    completeness: float | None
    attribution: float | None = None  # judge-v3; absent on older run records
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


def _eff_refused(r: GenQuestionResult) -> bool:
    """Effective refusal: the semantic verdict once judged, else the mechanical flag."""
    return r.refused_semantic if r.refused_semantic is not None else r.refused


def compute_gen_aggregates(results: list[GenQuestionResult]) -> GenAggregates:
    def block(subset: list[GenQuestionResult]) -> GenCategoryAggregate:
        return GenCategoryAggregate(
            count=len(subset),
            refused=sum(1 for r in subset if _eff_refused(r)),
            refusal_correct=sum(1 for r in subset if r.refusal_correct) / len(subset),
            citation_span_recall=_mean(
                [r.citation_span_recall for r in subset if r.citation_span_recall is not None]
            ),
            citation_precision=_mean(
                [r.citation_precision for r in subset if r.citation_precision is not None]
            ),
            faithfulness=_mean([float(r.faithfulness) for r in subset if r.faithfulness]),
            completeness=_mean([float(r.completeness) for r in subset if r.completeness]),
            attribution=_mean([float(r.attribution) for r in subset if r.attribution]),
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
        refusal_accuracy_oos=(sum(1 for r in oos if _eff_refused(r)) / len(oos)) if oos else None,
        false_refusal_rate=(sum(1 for r in in_scope if _eff_refused(r)) / len(in_scope))
        if in_scope
        else 0.0,
        citation_span_recall=overall.citation_span_recall,
        citation_precision=overall.citation_precision,
        faithfulness=overall.faithfulness,
        completeness=overall.completeness,
        attribution=overall.attribution,
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
# judge-v3: adds the ATTRIBUTION dimension — faithfulness rewards a grounded answer
#   even if it silently picks one side of a contradiction, so that behavior (and
#   un-attributed multi-source synthesis) needs its own axis. Pairs with the
#   baseline-v2 answer prompt, which instructs the model to surface disagreement.
# judge-v3.1: three calibration fixes after the first judge-v3 run (over-severity on
#   concise correct answers, brittle refusal match):
#   (a) semantic refusal — the mechanical refused flag is an exact match on the contract
#       sentence, so a correct-but-paraphrased abstention (esp. out-of-scope) scored as a
#       non-refusal. A yes/no judge call now accepts paraphrased refusals; refusal_correct
#       and the refusal aggregates use it. The 1-5 dimensions stay None on out-of-scope
#       (no ideal answer; binary refuse/answer only).
#   (b) completeness graded against the QUESTION's scope, not every detail of the rich
#       reference — a concise answer that fully answers what was asked scores 5 (lit-001:
#       "23 wounds" answers "how many wounds", was wrongly dinged to 3 for omitting
#       reference context the question never asked for).
#   (c) attribution scored in explicit agree/disagree steps so the all-agree case (bare
#       markers, nothing misattributed) reliably scores 5 instead of being penalised for
#       lacking prose attribution it doesn't need. Disagreement = incompatible claims (not
#       mere omission/emphasis); misattribution = citing a source that doesn't support the
#       claim (not an extra on-topic corroborating marker).
JUDGE_RUBRIC_VERSION = "judge-v3.1"

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

COMPLETENESS_RUBRIC = """You are grading a RAG system's answer for COMPLETENESS: does it
cover everything the QUESTION asks for?

Judge against the QUESTION. Use the reference answer only as the gold standard for which
facts the question requires — the reference is a rich 5/5 example and often includes
context BEYOND what was asked. Do NOT penalize the answer for omitting such extra detail:
a concise answer that fully and correctly answers what the question asked scores 5.

Score 1-5:
5 = answers everything the question asks (being more concise than the reference is fine);
3 = answers the core but omits a part the question explicitly asks for;
1 = misses the point of the question.
Extra correct detail must not lower the score.

Question: {question}

Reference answer (gold standard — may exceed the question's scope):
{ideal}

Answer to grade:
{answer}

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

ATTRIBUTION_RUBRIC = """You are grading a RAG system's answer for ATTRIBUTION.

Policy: when the source passages DISAGREE, or the answer draws on SEVERAL different
sources, the answer must make clear IN PROSE which source each version or contribution
comes from (e.g. "Suetonius reports X, but Dio says Y", or "Plutarch describes... while
Arrian adds..."). When the sources simply agree, naming each one is optional and its
absence is NOT a fault. Bare citation markers like [1][2] are NOT prose attribution: they
cannot tell a reader that two sources DISAGREE.

Step 1 — do the passages relevant to the question AGREE or DISAGREE on the point at issue?
Treat them as DISAGREEING only when they make INCOMPATIBLE claims about the SAME point (X
cannot be true if Y is). One source merely OMITTING a detail another includes, or differing
in emphasis, wording, or which aspects it covers, is NOT a disagreement — that is agreement
plus extra detail, and needs no disagreement-surfacing.
Step 2 — score 1-5:
- If they AGREE (or only one source is used): score 5 as long as nothing is misattributed.
  Bare markers are correct here — do NOT penalize the absence of prose attribution, and do
  NOT invent a disagreement that isn't there.
- If they DISAGREE: 5 = the answer surfaces the disagreement AND names each version's
  source in prose (or, for multi-source synthesis, attributes the distinct contributions);
  3 = surfaces the disagreement but leaves it unattributed, or attributes some parts while
  blurring others; 1 = presents the contested point as settled (silently picks one side).
- Misattribution scores 1 in either case: a claim is credited to a source that does not
  actually support it. This INCLUDES naming the wrong author in prose — e.g. writing
  "Tacitus says X" when X comes from a different author's passage (say Gibbon) or from an
  editorial footnote, not from that author's own text. Check that the author NAMED in the
  prose matches the author of a passage that genuinely supports the claim.
  NOT misattribution: a claim correctly credited to a supporting source that ALSO carries
  extra on-topic corroborating markers from other authors (stray extra markers are a
  citation-precision matter; a question scoped to one author does not make a correct
  corroborating citation a misattribution).

Below are ALL passages the model saw, numbered as shown to it; the ones it cited are
flagged "(cited)".

Question: {question}

Source passages:
{sources}

Answer to grade:
{answer}

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


REFUSAL_JUDGE = """Decide whether the following answer is a REFUSAL: does it decline to
answer, stating in any wording that the provided sources do not contain the information
needed — as opposed to actually attempting to answer the question?

Question: {question}

Answer:
{answer}

Reply with ONLY one word: "yes" if it is a refusal/abstention, "no" if it attempts a
substantive answer."""


def _parse_yes_no(raw: str) -> bool | None:
    """First clear yes/no token wins; None if the reply commits to neither."""
    text = raw.strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    has_yes, has_no = "yes" in text, "no" in text
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


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
    """Mutates result with the judge-layer fields (judge-v3.1).

    First a yes/no semantic refusal check accepts a paraphrased abstention the
    exact-match `refused` flag missed, and refusal_correct is recomputed on it.
    Then — for answered, in-scope questions only — faithfulness, completeness,
    and attribution (1-5). The judge sees ALL retrieved passages (what the answer
    model saw), with cited ones flagged; see rubric history above.
    """
    if result.refused:
        result.refused_semantic = True  # the exact contract sentence is unambiguous
    else:
        reply = await judge.complete(
            [
                ChatMessage(
                    role="user",
                    content=REFUSAL_JUDGE.format(question=result.question, answer=result.answer),
                )
            ]
        )
        verdict = _parse_yes_no(reply.text)
        result.refused_semantic = verdict if verdict is not None else result.refused
    result.refusal_correct = result.refused_semantic == result.refusal_expected

    # 1-5 dimensions: answered, in-scope only (out-of-scope has no ideal answer —
    # it stays binary refuse/answer, scored above; the user's call, to keep the
    # faithfulness aggregate clean).
    if result.refused_semantic or result.refusal_expected:
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
        (
            "attribution",
            ATTRIBUTION_RUBRIC.format(
                question=result.question,
                sources=sources_text or "(none cited)",
                answer=result.answer,
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
        result.attribution = None
        result.refused_semantic = None
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
