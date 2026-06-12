"""Generation-eval tests: mechanical scoring, verdict parsing, judge layer.

All pure / in-process — the runner's DB/LLM legs are exercised by
`ahx eval generate` against live services.
"""

from collections.abc import AsyncIterator, Sequence

from ahx.evals.generation import (
    compute_gen_aggregates,
    judge_question,
    parse_verdict,
    score_generation,
)
from ahx.evals.golden import GoldenQuestion, GoldSpan, ResolvedSpan
from ahx.generation.citations import Citation, MarkerAudit
from ahx.generation.pipeline import DoneEvent, SourcesEvent
from ahx.llm import ChatMessage, ChatResult, StreamEnd, StreamEvent, TextDelta, Usage


def citation(marker: int, pg_id: int = 1, start: int = 0, end: int = 1000) -> Citation:
    return Citation(
        marker=marker,
        chunk_id=100 + marker,
        pg_id=pg_id,
        author="Suetonius",
        work_title="Lives",
        locator=["1"],
        text="passage text",
        score=0.8,
        char_start=start,
        char_end=end,
    )


def sources(*citations: Citation) -> SourcesEvent:
    return SourcesEvent(citations=list(citations), prompt_version="baseline-v1")


def done(
    answer: str = "An answer [1].",
    refused: bool = False,
    used: list[int] | None = None,
    dangling: list[int] | None = None,
) -> DoneEvent:
    return DoneEvent(
        answer=answer,
        refused=refused,
        markers=MarkerAudit(used=used or [], dangling=dangling or []),
        usage=Usage(prompt_tokens=100, completion_tokens=20),
    )


def question(category: str = "literal", n_spans: int = 1) -> GoldenQuestion:
    return GoldenQuestion.model_validate(
        {
            "id": "q-001",
            "category": category,
            "question": "What happened?",
            "ideal_answer": "Something." if category != "out-of-scope" else "",
            "gold_spans": [GoldSpan(pg_id=1, quote=f"q{i}") for i in range(n_spans)],
        }
    )


def span(pg_id: int = 1, start: int = 100, end: int = 200) -> ResolvedSpan:
    return ResolvedSpan(pg_id=pg_id, char_start=start, char_end=end)


# --- mechanical scoring ---


def test_cited_chunk_covering_span_counts() -> None:
    result = score_generation(
        question(),
        [span(start=100, end=200)],  # midpoint 150, inside citation [0, 1000)
        sources(citation(1)),
        done(used=[1]),
        latency_ms=1200,
    )
    assert result.citation_span_recall == 1.0
    assert result.citation_precision == 1.0
    assert result.refusal_correct is True  # in-scope, answered
    assert result.cited_chunk_ids == [101]
    assert result.latency_ms == 1200


def test_retrieved_but_uncited_does_not_count() -> None:
    result = score_generation(
        question(), [span()], sources(citation(1)), done(used=[]), latency_ms=1
    )
    assert result.citation_span_recall == 0.0
    assert result.citation_precision is None  # no markers used
    assert result.retrieved_chunk_ids == [101]
    assert result.cited_chunk_ids == []


def test_precision_counts_only_gold_covering_markers() -> None:
    # [1] covers the span; [2] points at another work entirely.
    result = score_generation(
        question(),
        [span(pg_id=1)],
        sources(citation(1, pg_id=1), citation(2, pg_id=99)),
        done(used=[1, 2]),
        latency_ms=1,
    )
    assert result.citation_span_recall == 1.0
    assert result.citation_precision == 0.5


def test_oos_refusal_is_correct_and_unscored() -> None:
    result = score_generation(
        question("out-of-scope", n_spans=0),
        [],
        sources(citation(1)),
        done(answer="The provided sources...", refused=True),
        latency_ms=1,
    )
    assert result.refusal_expected is True
    assert result.refusal_correct is True
    assert result.citation_span_recall is None
    assert result.citation_precision is None


def test_in_scope_refusal_is_false_refusal() -> None:
    result = score_generation(
        question(), [span()], sources(citation(1)), done(refused=True), latency_ms=1
    )
    assert result.refusal_correct is False


# --- aggregates ---


def test_aggregates_split_oos_from_in_scope() -> None:
    answered = score_generation(
        question(), [span()], sources(citation(1)), done(used=[1]), latency_ms=100
    )
    refused_oos = score_generation(
        question("out-of-scope", n_spans=0), [], sources(citation(1)), done(refused=True), 200
    )
    aggregates = compute_gen_aggregates([answered, refused_oos])
    assert aggregates.questions == 2
    assert aggregates.refusal_accuracy_oos == 1.0
    assert aggregates.false_refusal_rate == 0.0
    assert aggregates.citation_span_recall == 1.0
    assert aggregates.mean_latency_ms == 150
    assert aggregates.by_category["literal"].count == 1


# --- judge layer ---


def test_parse_verdict_tolerates_fences_and_rejects_garbage() -> None:
    assert parse_verdict('{"score": 4, "reason": "solid"}').score == 4  # type: ignore[union-attr]
    assert parse_verdict('```json\n{"score": 2, "reason": "weak"}\n```').score == 2  # type: ignore[union-attr]
    assert parse_verdict("no json here") is None
    assert parse_verdict('{"score": 9, "reason": "out of range"}') is None


class FakeJudge:
    model_name = "fake-judge"

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0
        self.prompts: list[str] = []

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        yield TextDelta(text=self._reply)
        yield StreamEnd(usage=None)

    async def complete(self, messages: Sequence[ChatMessage]) -> ChatResult:
        self.calls += 1
        self.prompts.append(messages[-1].content)
        return ChatResult(text=self._reply, usage=None)


async def test_judge_scores_answered_in_scope_question() -> None:
    result = score_generation(
        question(), [span()], sources(citation(1)), done(used=[1]), latency_ms=1
    )
    judge = FakeJudge('{"score": 4, "reason": "well supported"}')
    await judge_question(judge, result, [citation(1)])
    assert judge.calls == 2  # faithfulness + completeness
    assert result.faithfulness == 4
    assert result.completeness == 4
    assert "well supported" in result.judge_notes


async def test_judge_sees_all_retrieved_sources_with_cited_flags() -> None:
    # judge-v2: uncited [2] must reach the judge so miscited-but-grounded
    # content is not scored as fabrication.
    result = score_generation(
        question(), [span()], sources(citation(1), citation(2)), done(used=[1]), latency_ms=1
    )
    judge = FakeJudge('{"score": 4, "reason": "ok"}')
    await judge_question(judge, result, [citation(1), citation(2)])
    faithfulness_prompt = judge.prompts[0]
    assert "[1] (cited)" in faithfulness_prompt
    assert "[2] Suetonius" in faithfulness_prompt  # present, not flagged as cited
    assert "[2] (cited)" not in faithfulness_prompt


async def test_judge_skips_refusals() -> None:
    result = score_generation(
        question("out-of-scope", n_spans=0), [], sources(), done(refused=True), latency_ms=1
    )
    judge = FakeJudge('{"score": 5, "reason": "n/a"}')
    await judge_question(judge, result, [])
    assert judge.calls == 0
    assert result.faithfulness is None
