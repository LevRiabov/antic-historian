"""Read layer for the published eval runs (Phase 7 golden-set page).

Serves the LATEST run of each tier straight from the typed record on disk:

  * retrieval (``-rag``)   -> RetrievalRun  (recall@k, MRR per question)
  * generation (``-agent``) -> GenerationRun (answer + faithfulness/completeness/
    attribution/refusal per question)

The golden page merges rag+agent by ``question_id``; the security page merges the
baseline+defended runs by attack ``id``. All are loaded ONCE at API startup (the
published record is a frozen artifact); a new eval becomes visible on the next
restart. Smoke/probe runs (``-smoke``) are excluded by construction — only the
published tags are eligible.
"""

from pathlib import Path

from ahx.evals.generation import GenerationRun
from ahx.evals.retrieval import RetrievalRun
from ahx.evals.runs import AGENT_TAG, BASELINE_TAG, DEFENDED_TAG, RAG_TAG, latest_run_path
from ahx.evals.security import SecurityRun


def load_latest_rag_run(runs_dir: Path) -> RetrievalRun | None:
    """Parse the newest ``-rag`` record, or None if the dir has none."""
    path = latest_run_path(runs_dir, RAG_TAG)
    if path is None:
        return None
    return RetrievalRun.model_validate_json(path.read_text(encoding="utf-8"))


def load_latest_agent_run(runs_dir: Path) -> GenerationRun | None:
    """Parse the newest ``-agent`` record, or None if the dir has none."""
    path = latest_run_path(runs_dir, AGENT_TAG)
    if path is None:
        return None
    return GenerationRun.model_validate_json(path.read_text(encoding="utf-8"))


def load_latest_security_run(runs_dir: Path, tag: str) -> SecurityRun | None:
    """Parse the newest security record carrying ``tag`` (baseline/defended), or None."""
    path = latest_run_path(runs_dir, tag)
    if path is None:
        return None
    return SecurityRun.model_validate_json(path.read_text(encoding="utf-8"))


def load_latest_security_baseline(runs_dir: Path) -> SecurityRun | None:
    return load_latest_security_run(runs_dir, BASELINE_TAG)


def load_latest_security_defended(runs_dir: Path) -> SecurityRun | None:
    return load_latest_security_run(runs_dir, DEFENDED_TAG)
