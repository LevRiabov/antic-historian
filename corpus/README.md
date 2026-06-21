# Corpus

Everything here except this README and the manifest is **gitignored** — texts are
downloaded, never committed.

- `ai_historian_corpus_eu_pd.txt` — the source manifest (pipe-delimited: id | category |
  author | title | translator | pd_basis | txt_url | landing_url). EU-PD-cleared
  Project Gutenberg texts. Committed.
- `raw/` — downloaded original files, as fetched (created by `uv run ahx ingest`).
- `normalized/` — parsed document trees (Layer 1 output, see docs/chunking.md).

The corpus grew over Phase 1: **62 normalized works** (~46k embedded chunks) of
EU-public-domain ancient history (Project Gutenberg), Herodotus to Gibbon. The
`pd_basis` column documents the public-domain basis per work — preserve that diligence
for any source added.
