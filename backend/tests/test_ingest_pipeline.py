"""Ingestion QA-flag heuristics (NormalizeReport.flags).

The orchestration legs (normalize_work / chunk_one / load_one) read files and hit
Postgres and are exercised by `ahx ingest …`. The per-book QA flags, though, are pure
threshold logic that decides whether a normalized book is silently malformed (coarse
divisions, giant paragraphs, fallback parser) — worth pinning so a threshold typo
doesn't quietly stop flagging bad ingests.
"""

from ahx.ingest.pipeline import NormalizeReport


def _report(**overrides: object) -> NormalizeReport:
    base: dict[str, object] = {
        "pg_id": 1,
        "title": "Test Work",
        "parser": "structural",
        "divisions": 10,
        "paragraphs": 100,
        "chars": 100_000,
        "max_paragraph_chars": 2_000,
    }
    base.update(overrides)
    return NormalizeReport.model_validate(base)


def test_clean_report_has_no_flags() -> None:
    assert _report().flags == []


def test_error_short_circuits_all_other_flags() -> None:
    # An errored row reports ONLY the error, even if other fields would also flag.
    report = _report(error="raw file missing", parser="flat", max_paragraph_chars=99_999)
    assert report.flags == ["ERROR: raw file missing"]


def test_flat_parser_flagged() -> None:
    assert "fallback parser — no structure extracted" in _report(parser="flat").flags


def test_coarse_divisions_flagged_above_30k_avg() -> None:
    # 100k chars / 3 divisions ≈ 33k avg -> coarse; 100k / 4 = 25k -> fine.
    assert any("too coarse" in f for f in _report(divisions=3, chars=100_000).flags)
    assert not any("too coarse" in f for f in _report(divisions=4, chars=100_000).flags)


def test_giant_paragraph_flagged_above_8k() -> None:
    assert any("giant paragraph" in f for f in _report(max_paragraph_chars=8_001).flags)
    assert not any("giant paragraph" in f for f in _report(max_paragraph_chars=8_000).flags)


def test_zero_divisions_does_not_divide_by_zero() -> None:
    # A degenerate book with 0 divisions must not crash the coarse-division check.
    assert _report(divisions=0, chars=100_000, parser="flat").flags == [
        "fallback parser — no structure extracted"
    ]


def test_multiple_flags_accumulate() -> None:
    report = _report(parser="flat", divisions=2, chars=100_000, max_paragraph_chars=9_000)
    assert len(report.flags) == 3  # flat + coarse + giant
