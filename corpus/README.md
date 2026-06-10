# Corpus

Everything here except this README and the manifest is **gitignored** — texts are
downloaded, never committed.

- `ai_historian_corpus_eu_pd.txt` — the source manifest (pipe-delimited: id | category |
  author | title | translator | pd_basis | txt_url | landing_url). EU-PD-cleared
  Project Gutenberg texts. Committed.
- `raw/` — downloaded original files, as fetched (created by `uv run ahx ingest`).
- `normalized/` — parsed document trees (Layer 1 output, see docs/chunking.md).

Measured 2026-06-10: the 16 manifest texts total **35.7 MB raw** (~8M tokens).
