from ahx.evals.golden import ResolvedSpan
from ahx.evals.retrieval import RetrievedRef, score_question


def ref(pg_id: int, start: int, end: int, rank: int) -> RetrievedRef:
    return RetrievedRef(pg_id=pg_id, char_start=start, char_end=end, rank=rank)


def span(pg_id: int, start: int, end: int) -> ResolvedSpan:
    return ResolvedSpan(pg_id=pg_id, char_start=start, char_end=end)


def test_midpoint_coverage_rule() -> None:
    # Span midpoint = 150; chunk [100, 200) covers it, chunk [0, 149) does not.
    result = score_question(
        "q", "literal", [span(1, 100, 200)], [ref(1, 0, 149, 1), ref(1, 100, 200, 2)]
    )
    assert result.first_hit_rank == 2
    assert result.covered_at[1] == 0
    assert result.covered_at[5] == 1


def test_wrong_work_never_covers() -> None:
    result = score_question("q", "literal", [span(1, 0, 100)], [ref(2, 0, 100, 1)])
    assert result.first_hit_rank is None
    assert result.recall_at(20) == 0.0


def test_multi_span_partial_recall() -> None:
    spans = [span(1, 0, 100), span(1, 1000, 1100), span(2, 0, 100)]
    retrieved = [
        ref(1, 0, 120, 1),  # covers span 1 at rank 1
        ref(2, 0, 500, 7),  # covers span 3 at rank 7
    ]
    result = score_question("q", "synthesis", spans, retrieved)
    assert result.recall_at(1) == 1 / 3
    assert result.recall_at(5) == 1 / 3
    assert result.recall_at(10) == 2 / 3
    assert result.first_hit_rank == 1
    assert result.reciprocal_rank == 1.0


def test_mrr_uses_first_hit_across_spans() -> None:
    spans = [span(1, 0, 100), span(1, 1000, 1100)]
    retrieved = [ref(1, 950, 1200, 3)]  # only second span, rank 3
    result = score_question("q", "multi-hop", spans, retrieved)
    assert result.first_hit_rank == 3
    assert result.reciprocal_rank == 1 / 3
