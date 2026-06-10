# Antic Historian

A production-grade RAG system over ancient-history primary sources — Herodotus to Gibbon —
built to demonstrate **reliable, measurable, cost-controlled LLM engineering**: grounded
answers with verifiable citations, honest refusals, published evals, and per-request costs
shown live.

> **Status: Phase 0 — scaffolding.** Planning docs complete; framework spike (Gate D1) next.
> Roadmap: [project-plan.md](project-plan.md).

## What it will do

- **Chat** — ask a question about antiquity, get an answer grounded in primary sources with
  inline citations down to the canonical locator (*Caesar, BG 4.25*), or an honest
  "I don't have a source for that."
- **Evals in the open** — a live tab with faithfulness / completeness / refusal scores on a
  curated golden set, and the ablation data behind every technique choice.
- **Cost transparency** — every answer shows what it cost to produce.

## Architecture (planned)

```
TS frontend (Phase 7) ──▶ FastAPI backend (Python 3.13, uv)
                            ├─ retrieval: hybrid (dense+BM25) → cross-encoder rerank,
                            │  contextual-retrieval embeddings  [docs/rag-techniques.md]
                            ├─ agent tool-loop for hard questions
                            ├─ Postgres + pgvector (Docker local / Neon prod)  [docs/vector-stores.md]
                            ├─ swappable LLMs, local → frontier  (gate D5)
                            └─ eval harness + golden set  (the methodology IS the product)
```

Corpus: ~16+ EU-public-domain works (~36 MB / ~8M tokens measured so far) from Project
Gutenberg — manifest in [corpus/](corpus/).

## Repo layout

| Path | What |
|---|---|
| [backend/](backend/) | Python service — ingest, retrieval, agent, evals, API ([commands](backend/README.md)) |
| [frontend/](frontend/) | TS app — Phase 7, not started |
| [docs/](docs/) | Decision docs: [stack](docs/python-stack.md) · [embeddings](docs/embeddings.md) · [vector stores](docs/vector-stores.md) · [chunking](docs/chunking.md) · [RAG techniques](docs/rag-techniques.md) |
| [project-plan.md](project-plan.md) | Phases, decision gates D1–D5, exit criteria |
| [corpus/](corpus/) | Source manifest (texts downloaded, not committed) |

## Quickstart (backend)

```sh
cd backend
uv sync                # deps + venv (install uv: https://docs.astral.sh/uv/)
uv run pytest          # tests
uv run ahx serve       # dev API → http://127.0.0.1:8000/docs
docker compose up -d   # local Postgres+pgvector (from repo root)
```

## Provenance

Successor to a hand-rolled TypeScript RAG lab (50-question golden set, measured ablations:
contextual retrieval +16 recall@5, HyDE −9.7, agent loop +0.64 completeness). This project
carries the methodology onto a 10× corpus with the production Python stack. Every technique
here enters through a measured ablation — the receipts will be published in the case study.
