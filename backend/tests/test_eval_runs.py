"""Run-record filename convention: terminal tag + latest-by-tag selection."""

from pathlib import Path

import pytest

from ahx.evals.runs import latest_run_path, tagged_stem


def test_tagged_stem_appends_when_absent() -> None:
    assert tagged_stem("2026-06-17T14-09-09Z-gen-agent-v6", "agent") == (
        "2026-06-17T14-09-09Z-gen-agent-v6-agent"
    )
    assert tagged_stem("2026-06-14T20-24-19Z-dense-ctx-v1", "rag") == (
        "2026-06-14T20-24-19Z-dense-ctx-v1-rag"
    )


def test_tagged_stem_is_idempotent() -> None:
    # Re-saving an already-tagged record (e.g. a rejudge) must not double the suffix,
    # regardless of which tag the stem already carries.
    assert tagged_stem("run-agent", "agent") == "run-agent"
    assert tagged_stem("run-smoke", "agent") == "run-smoke"
    assert tagged_stem("run-rag", "smoke") == "run-rag"
    # Security tags share the same machinery: an audit-*-defended label keeps one suffix.
    assert tagged_stem("audit-deepseek-defended", "defended") == "audit-deepseek-defended"
    assert tagged_stem("audit-gemma-baseline", "baseline") == "audit-gemma-baseline"
    assert tagged_stem("security-deepseek", "baseline") == "security-deepseek-baseline"


def test_tagged_stem_rejects_unknown_tag() -> None:
    with pytest.raises(ValueError, match="unknown run tag"):
        tagged_stem("run", "bogus")


def test_latest_run_path_picks_newest_matching_tag(tmp_path: Path) -> None:
    for name in (
        "2026-06-14T20-24-19Z-dense-ctx-v1-rag.json",
        "2026-06-13T18-31-18Z-dense-v1-rag.json",
        "2026-06-17T14-09-09Z-gen-agent-v6-agent.json",
        "2026-06-17T13-11-50Z-agent-v6-smoke.json",  # newer than the rag runs, wrong tag
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    rag = latest_run_path(tmp_path, "rag")
    assert rag is not None and rag.name == "2026-06-14T20-24-19Z-dense-ctx-v1-rag.json"
    agent = latest_run_path(tmp_path, "agent")
    assert agent is not None and agent.name == "2026-06-17T14-09-09Z-gen-agent-v6-agent.json"
    assert latest_run_path(tmp_path, "rag") != latest_run_path(tmp_path, "smoke")


def test_latest_run_path_none_when_empty(tmp_path: Path) -> None:
    assert latest_run_path(tmp_path, "agent") is None
