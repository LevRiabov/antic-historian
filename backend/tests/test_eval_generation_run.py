"""run_generation_eval orchestration: concurrency, ordering, and error aggregation.

The scoring/judge math is pinned in test_eval_generation.py. This file covers the
run LOOP itself — the riskiest untested code per rule #5 ("measurement bugs moved
numbers more than real changes"): that a single question's failure becomes a recorded
error_result instead of aborting the whole concurrent run, and that gather preserves
question order regardless of completion order.

No DB / network: the DB engine is faked and the `ask` pipeline is replaced with a
stub that yields canned events (or raises) — so the loop runs hermetically in CI.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest

import ahx.evals.generation as gen
from ahx.config import Settings
from ahx.evals.generation import run_generation_eval
from ahx.evals.golden import GoldenQuestion
from ahx.generation.citations import MarkerAudit
from ahx.generation.pipeline import AskEvent, DoneEvent, SourcesEvent
from ahx.llm import Usage


class _FakeEngine:
    """Stands in for the async DB engine: the loop only awaits dispose() on it."""

    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


def _questions(n: int) -> list[GoldenQuestion]:
    # No gold_spans -> resolve_span (which reads corpus files) is skipped.
    return [
        GoldenQuestion.model_validate(
            {
                "id": f"q-{i:03d}",
                "category": "literal",
                "question": f"question {i}" if i != 1 else "question 1 BOOM",
                "ideal_answer": "Something.",
                "gold_spans": [],
            }
        )
        for i in range(n)
    ]


def _fake_ask_factory() -> Any:
    async def fake_ask(
        question: str, retriever: Any, chat: Any, top_k: int = 5
    ) -> AsyncIterator[AskEvent]:
        if "BOOM" in question:
            raise RuntimeError("provider down")  # one question fails mid-run
        yield SourcesEvent(citations=[], prompt_version="test")
        yield DoneEvent(
            answer="An answer.",
            refused=False,
            markers=MarkerAudit(used=[], dangling=[]),
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    return fake_ask


@pytest.fixture
def hermetic_run(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_engine = _FakeEngine()

    def fake_create(_url: str) -> _FakeEngine:
        return fake_engine

    monkeypatch.setattr(gen, "create_async_db_engine", fake_create)
    monkeypatch.setattr(gen, "ask", _fake_ask_factory())


async def test_failed_question_recorded_not_fatal(hermetic_run: None) -> None:
    questions = _questions(3)
    run = await run_generation_eval(
        Settings(_env_file=None),  # pyright: ignore[reportCallIssue]
        questions,
        concurrency=3,  # all three in flight at once
    )

    # The whole run completes despite q-001 raising — order preserved by gather.
    assert [r.question_id for r in run.results] == ["q-000", "q-001", "q-002"]
    assert [r.errored for r in run.results] == [False, True, False]

    failed = run.results[1]
    assert failed.refused is True  # no valid answer/refuse decision was made
    assert "provider down" in (failed.judge_notes or "")
    assert run.aggregates is not None  # aggregation ran over the mixed results


async def test_on_result_fires_once_per_question(hermetic_run: None) -> None:
    seen: list[str] = []
    await run_generation_eval(
        Settings(_env_file=None),  # pyright: ignore[reportCallIssue]
        _questions(3),
        concurrency=2,
        on_result=lambda r: seen.append(r.question_id),
    )
    assert sorted(seen) == ["q-000", "q-001", "q-002"]  # progress callback fired for each
