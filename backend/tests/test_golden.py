from pathlib import Path

import pytest

from ahx.evals.golden import (
    GoldenQuestion,
    GoldSpan,
    ResolutionError,
    ResolvedSpan,
    load_golden_set,
    resolve_span,
)
from ahx.ingest.model import Division, NormalizedWork

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "evals" / "golden"


@pytest.fixture
def normalized_dir(tmp_path: Path) -> Path:
    work = NormalizedWork(
        pg_id=42,
        author="Test",
        title="T",
        category="primary",
        translator="T",
        parser="classical",
        divisions=[
            Division(
                locator=["I"],
                heading=None,
                paragraphs=[
                    "He was stabbed with three and twenty",
                    "wounds, uttering a groan only. The end.",
                    "A repeated sentence. A repeated sentence.",
                ],
            )
        ],
    )
    (tmp_path / "pg42.json").write_text(work.model_dump_json(), encoding="utf-8")
    return tmp_path


def test_resolves_across_paragraph_breaks(normalized_dir: Path) -> None:
    span = GoldSpan(pg_id=42, quote="three and twenty wounds, uttering a groan")
    result = resolve_span(span, normalized_dir, "q1")
    assert isinstance(result, ResolvedSpan)


def test_not_found(normalized_dir: Path) -> None:
    result = resolve_span(GoldSpan(pg_id=42, quote="not in the text"), normalized_dir, "q1")
    assert isinstance(result, ResolutionError)
    assert result.problem == "not-found"


def test_ambiguous(normalized_dir: Path) -> None:
    result = resolve_span(GoldSpan(pg_id=42, quote="A repeated sentence."), normalized_dir, "q1")
    assert isinstance(result, ResolutionError)
    assert result.problem == "ambiguous"
    assert result.occurrences == 2


def test_missing_work(normalized_dir: Path) -> None:
    result = resolve_span(GoldSpan(pg_id=999, quote="anything"), normalized_dir, "q1")
    assert isinstance(result, ResolutionError)
    assert result.problem == "work-not-found"


def test_repo_golden_set_loads_and_validates() -> None:
    questions = load_golden_set(GOLDEN_DIR)
    assert any(q.id == "lit-001" for q in questions)
    for question in questions:
        assert isinstance(question, GoldenQuestion)
        if question.category == "out-of-scope":
            assert question.gold_spans == []
        else:
            assert question.gold_spans, f"{question.id}: non-OOS question needs gold spans"


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    content = (
        "- {id: x-001, category: literal, question: q1, gold_spans: [{pg_id: 1, quote: a}]}\n"
        "- {id: x-001, category: literal, question: q2, gold_spans: [{pg_id: 1, quote: b}]}\n"
    )
    (tmp_path / "dup.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden_set(tmp_path)
