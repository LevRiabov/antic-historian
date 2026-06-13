from ahx.evals.golden import GoldenQuestion, GoldSpan, ResolvedSpan
from ahx.evals.retrieval import compute_aggregates, score_question
from ahx.retrieval.dense import RetrievedChunk


def ref(pg_id: int, start: int, end: int, rank: int) -> RetrievedChunk:
    """Span-coverage tests only care about pg_id/offsets/rank; the citation
    fields (author, title, text, score) are irrelevant here — dummies."""
    return RetrievedChunk(
        chunk_id=1000 + rank,
        pg_id=pg_id,
        author="author",
        work_title="title",
        locator=["1"],
        text="chunk text",
        score=0.5,
        char_start=start,
        char_end=end,
        rank=rank,
    )


def span(pg_id: int, start: int, end: int, groups: list[str] | None = None) -> ResolvedSpan:
    return ResolvedSpan(pg_id=pg_id, char_start=start, char_end=end, groups=groups or [])


def question(spans: list[ResolvedSpan], category: str = "literal") -> GoldenQuestion:
    return GoldenQuestion.model_validate(
        {
            "id": "q-001",
            "category": category,
            "question": "What happened?",
            "ideal_answer": "Something happened.",
            "notes": "test note",
            "gold_spans": [
                GoldSpan(pg_id=s.pg_id, quote=f"gold quote {i}") for i, s in enumerate(spans)
            ],
        }
    )


def score(
    spans: list[ResolvedSpan],
    retrieved: list[RetrievedChunk],
    category: str = "literal",
):
    return score_question(question(spans, category), spans, retrieved, gold_chunk_ids=[])


def test_midpoint_coverage_rule() -> None:
    # Span midpoint = 150; chunk [100, 200) covers it, chunk [0, 149) does not.
    result = score([span(1, 100, 200)], [ref(1, 0, 149, 1), ref(1, 100, 200, 2)])
    assert result.first_hit_rank == 2
    assert result.recall[1] == 0.0
    assert result.recall[5] == 1.0


def test_wrong_work_never_covers() -> None:
    result = score([span(1, 0, 100)], [ref(2, 0, 100, 1)])
    assert result.first_hit_rank is None
    assert result.recall[20] == 0.0
    assert result.mrr == 0.0


def test_multi_span_partial_recall() -> None:
    spans = [span(1, 0, 100), span(1, 1000, 1100), span(2, 0, 100)]
    retrieved = [
        ref(1, 0, 120, 1),  # covers span 0 at rank 1
        ref(2, 0, 500, 7),  # covers span 2 at rank 7
    ]
    result = score(spans, retrieved, "synthesis")
    assert result.recall[1] == 1 / 3
    assert result.recall[5] == 1 / 3
    assert result.recall[10] == 2 / 3
    assert result.first_hit_rank == 1
    assert result.mrr == 1.0


def test_grouped_spans_are_alternatives_any_one_covers() -> None:
    # Same fact attested in two works, one requirement group: retrieving either
    # alternative fully covers it (recall 1.0, not 0.5). This is the literal /
    # synonym case that per-span recall used to understate.
    spans = [
        span(1, 0, 100, ["fact"]),
        span(2, 0, 100, ["fact"]),
    ]
    retrieved = [ref(2, 0, 100, 3)]  # only the second alternative, at rank 3
    result = score(spans, retrieved, "literal")
    assert result.recall[1] == 0.0  # not within top-1
    assert result.recall[5] == 1.0  # the one requirement is covered
    assert result.first_hit_rank == 3


def test_multi_membership_span_covers_several_requirements() -> None:
    # A combined passage satisfies both hops at once (groups=[hop1, hop2]);
    # retrieving it alone covers both required requirements.
    spans = [
        span(1, 0, 100, ["hop1"]),
        span(1, 1000, 1100, ["hop2"]),
        span(2, 0, 100, ["hop1", "hop2"]),  # one chunk answering both hops
    ]
    retrieved = [ref(2, 0, 100, 4)]  # only the combined span
    result = score(spans, retrieved, "multi-hop")
    assert result.recall[1] == 0.0
    assert result.recall[5] == 1.0  # both requirements covered by one chunk


def test_distinct_groups_are_conjunctive() -> None:
    # Two distinct requirements (e.g. the two versions of a contradiction):
    # covering only one yields half recall.
    spans = [
        span(1, 0, 100, ["v1"]),
        span(2, 0, 100, ["v2"]),
    ]
    retrieved = [ref(1, 0, 100, 1)]  # only v1
    result = score(spans, retrieved, "contradiction")
    assert result.recall[5] == 0.5


def test_mrr_uses_first_hit_across_spans() -> None:
    spans = [span(1, 0, 100), span(1, 1000, 1100)]
    retrieved = [ref(1, 950, 1200, 3)]  # only second span, rank 3
    result = score(spans, retrieved, "multi-hop")
    assert result.first_hit_rank == 3
    assert result.mrr == 1 / 3


def test_record_carries_readable_context_and_id_arrays() -> None:
    spans = [span(1, 100, 200)]
    retrieved = [ref(1, 100, 200, 1), ref(2, 0, 100, 2)]
    result = score_question(
        question(spans), spans, retrieved, gold_chunk_ids=[42, 43], latency_ms=17
    )

    assert result.question == "What happened?"
    assert result.ideal_answer == "Something happened."
    assert result.notes == "test note"
    assert result.gold_chunk_ids == [42, 43]
    assert result.retrieved_chunk_ids == [1001, 1002]
    assert result.similarities == [0.5, 0.5]
    assert result.latency_ms == 17
    assert [(s.pg_id, s.char_start, s.char_end) for s in result.gold_spans] == [(1, 100, 200)]


def test_aggregates_overall_and_by_category() -> None:
    lit = score([span(1, 100, 200)], [ref(1, 100, 200, 1)])  # recall@1 = 1, mrr = 1
    hop = score([span(1, 0, 100)], [ref(2, 0, 100, 1)], "multi-hop")  # all zero

    aggregates = compute_aggregates([lit, hop])
    assert aggregates.recall[1] == 0.5
    assert aggregates.mrr == 0.5
    assert aggregates.by_category["literal"].count == 1
    assert aggregates.by_category["literal"].recall[1] == 1.0
    assert aggregates.by_category["multi-hop"].mrr == 0.0
    assert "synthesis" not in aggregates.by_category
