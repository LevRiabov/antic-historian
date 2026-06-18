"""Run-record filename convention + selection.

Every saved eval run filename ends in exactly one terminal tag identifying its
tier and status, so the API can publish the latest run of each tier and never a
quick smoke run:

  ``…-rag.json``       a retrieval-tier run (RetrievalRun: recall@k, MRR)
  ``…-agent.json``     a generation/agent-tier run (GenerationRun: answers + judge)
  ``…-smoke.json``     a quick / partial / probe run — NEVER served as published
  ``…-baseline.json``  a security run with NO defense (the unprotected floor)
  ``…-defended.json``  a security run with the defense stack on

The save functions append the tag; dry runs (``--limit`` / ``--ids`` / ``--smoke``)
get ``smoke`` so a 5-question debug pass can't become "the latest agent run", and
security runs are tagged ``baseline`` / ``defended`` from their ``defense`` field so
the page can pair the latest of each. This module owns only the string/path
convention — no model imports, so the eval-tier modules depend on it without a cycle.
"""

import re
from pathlib import Path

RAG_TAG = "rag"
AGENT_TAG = "agent"
SMOKE_TAG = "smoke"
BASELINE_TAG = "baseline"  # security: no defense
DEFENDED_TAG = "defended"  # security: defense stack on
RUN_TAGS = (RAG_TAG, AGENT_TAG, SMOKE_TAG, BASELINE_TAG, DEFENDED_TAG)

_TAG_RE = re.compile(r"-(?:rag|agent|smoke|baseline|defended)$")


def tagged_stem(stem: str, tag: str) -> str:
    """Append ``-tag`` to a run filename stem, unless it already carries a tag.

    Idempotent: re-saving an already-tagged record (e.g. a rejudge writing back a
    ``…-agent`` name) won't double the suffix.
    """
    if tag not in RUN_TAGS:
        raise ValueError(f"unknown run tag {tag!r}; expected one of {RUN_TAGS}")
    return stem if _TAG_RE.search(stem) else f"{stem}-{tag}"


def latest_run_path(runs_dir: Path, tag: str) -> Path | None:
    """Newest run file carrying ``tag``, or None if none exist.

    Filenames begin with an ISO-8601 UTC stamp (``2026-06-17T14-09-09Z-…``), so a
    plain lexical sort is chronological — the last match is the most recent run.
    """
    matches = sorted(runs_dir.glob(f"*-{tag}.json"))
    return matches[-1] if matches else None
