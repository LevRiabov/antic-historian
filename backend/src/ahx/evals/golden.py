"""Golden set v2 — schema, loading, and gold-span resolution.

Design decisions (docs/golden-set.md has the authoring guide):

- **Gold spans are exact quotes, not offsets.** Authors paste `(pg_id, quote)`;
  offsets into the canonical text are RESOLVED at load/eval time. Quotes
  survive re-normalization (parser fixes shift offsets, not wording), and a
  quote that stops resolving fails loudly instead of silently pointing at the
  wrong text.
- **YAML, one file per category** (backend/evals/golden/<category>.yaml) —
  human-editable, git-reviewable, validated by pydantic on load.
- The golden set is production code (CLAUDE.md rule #5): schema'd, validated,
  versioned.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel

from ahx.ingest.chunker import canonical_text
from ahx.ingest.model import NormalizedWork

Category = Literal[
    "literal",
    "synonym",
    "multi-hop",
    "synthesis",
    "cross-book",
    "contradiction",
    "out-of-scope",
]

CATEGORIES: tuple[Category, ...] = (
    "literal",
    "synonym",
    "multi-hop",
    "synthesis",
    "cross-book",
    "contradiction",
    "out-of-scope",
)

# Target sizes (see docs/golden-set.md): v2.0 unblocks the harness, v2.1 is
# the size where ±0.3-effects per category become measurable.
TARGET_V20_PER_CATEGORY = 10
TARGET_V21_PER_CATEGORY = 20


class GoldSpan(BaseModel):
    """Where the answer lives: an exact quote from one work's canonical text."""

    pg_id: int
    quote: str
    note: str = ""


class ResolvedSpan(BaseModel):
    pg_id: int
    char_start: int
    char_end: int


class GoldenQuestion(BaseModel):
    id: str  # e.g. "lit-001" — category prefix + running number
    category: Category
    question: str
    ideal_answer: str = ""  # empty for out-of-scope
    gold_spans: list[GoldSpan] = []  # empty for out-of-scope (pydantic copies defaults)
    notes: str = ""
    status: Literal["draft", "reviewed"] = "draft"


class ResolutionError(BaseModel):
    question_id: str
    pg_id: int
    quote_preview: str
    problem: Literal["work-not-found", "not-found", "ambiguous"]
    occurrences: int = 0


def load_golden_set(golden_dir: Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    for path in sorted(golden_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise ValueError(f"{path.name}: expected a top-level YAML list of questions")
        for item in cast(list[dict[str, Any]], raw):
            questions.append(GoldenQuestion.model_validate(item))
    ids = [q.id for q in questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate question ids: {sorted(duplicates)}")
    return questions


@lru_cache(maxsize=64)
def _canonical_for(normalized_dir: str, pg_id: int) -> str | None:
    path = Path(normalized_dir) / f"pg{pg_id}.json"
    if not path.exists():
        return None
    work = NormalizedWork.model_validate_json(path.read_text(encoding="utf-8"))
    return canonical_text(work)


def _quote_pattern(quote: str) -> re.Pattern[str]:
    """Whitespace-tolerant exact match: quote words must appear verbatim and in
    order, but any whitespace (spaces, paragraph breaks in the canonical text,
    line breaks in a pasted quote) matches any other whitespace."""
    words = quote.split()
    return re.compile(r"\s+".join(re.escape(word) for word in words))


def resolve_span(
    span: GoldSpan, normalized_dir: Path, question_id: str
) -> ResolvedSpan | ResolutionError:
    canonical = _canonical_for(str(normalized_dir), span.pg_id)
    preview = span.quote[:60]
    if canonical is None:
        return ResolutionError(
            question_id=question_id,
            pg_id=span.pg_id,
            quote_preview=preview,
            problem="work-not-found",
        )
    matches = list(_quote_pattern(span.quote).finditer(canonical))
    if not matches:
        return ResolutionError(
            question_id=question_id,
            pg_id=span.pg_id,
            quote_preview=preview,
            problem="not-found",
        )
    if len(matches) > 1:
        return ResolutionError(
            question_id=question_id,
            pg_id=span.pg_id,
            quote_preview=preview,
            problem="ambiguous",
            occurrences=len(matches),
        )
    return ResolvedSpan(pg_id=span.pg_id, char_start=matches[0].start(), char_end=matches[0].end())
