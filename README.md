# Antic Historian

A production-grade RAG system over ancient-history primary sources — Herodotus to Gibbon —
built to demonstrate **reliable, measurable, cost-controlled LLM engineering**: grounded
answers with verifiable citations, honest refusals, published evals, and per-request costs
shown live.

> **Status: live.** Deployed demo → **https://historian.loroplanner.com**
> (chat · in-app evals tab · how-it-works). Roadmap & decisions: [project-plan.md](project-plan.md).
> The methodology *is* the product — every technique here entered through a measured
> ablation, kept or rejected with receipts in [docs/eval-log.md](docs/eval-log.md).
>
> **📄 [Case study](CASE-STUDY.md)** — the trust results, the techniques I rejected and why,
> and what the next dollar would buy.

## What it does

- **Chat** — ask a question about antiquity, get an answer grounded in primary sources with
  inline citations to the canonical locator (*Caesar, BG 4.25*), or an honest
  "I don't have a source for that." Two modes: a single-shot **fast path** (default) and a
  multi-step **deep mode** (a LangGraph agent that searches, reads, and assembles distributed
  answers) for hard cross-source questions.
- **Evals in the open** — a live tab with faithfulness / completeness / attribution / refusal
  scores on a curated 161-question golden set, served straight from the frozen run record.
- **Cost transparency** — every answer shows what it cost to produce.

## Results (latest published run — `agent-v8` / `judge-v3.6`)

Measured on the 161-question golden set (135 in-scope across literal / synonym / multi-hop /
synthesis / cross-book / contradiction, + 26 out-of-scope). Full narrative: [docs/eval-log.md](docs/eval-log.md).

| | |
|---|---|
| Faithfulness | **4.41 / 5** |
| Completeness | **4.91 / 5** |
| Attribution | **4.63 / 5** |
| In-scope false-refusal | **0.0%** (135/135 answered) |
| Out-of-scope honest refusal | **96.2%** |
| Citation span recall | **58.2%** |
| Prompt-injection ASR (defense stack) | **0%** (15–18% undefended) |

The retrieval headline: a strong embedder (qwen3-8b) lifted naive recall@5 **35% → 53%** and
**subsumed both BM25 and the cross-encoder reranker** — so production ships `dense-ctx-v1`
(contextual embeddings, no hybrid, no rerank), each rejection backed by a measured run.

## Architecture

```
Vite + React SPA  ──nginx /api proxy──▶  FastAPI backend (Python 3.13, uv, async/SSE)
(one origin, no CORS)                     ├─ retrieval: dense contextual embeddings
                                          │   (qwen3-embedding-8b, 1024d, Nebius)
                                          │   — hybrid BM25 & rerank measured + REJECTED
                                          ├─ generation: deepseek-v4-pro
                                          │   fast path (single-shot) + agent deep mode
                                          │   (LangGraph grammar-ReAct: search/read/finalize)
                                          ├─ Postgres + pgvector (self-hosted in-compose; Docker local)
                                          ├─ split LLM-judge: kimi-k2.6 + qwen3.7-max
                                          ├─ guardrails: input blocklist + output validation
                                          │   + grounding gate (0% ASR) · per-IP + per-session caps
                                          ├─ fallback chain + live per-request cost + Langfuse tracing
                                          └─ eval harness + golden set  (the methodology IS the product)
```

**Deploy:** the full stack (`db` + `backend` + `frontend`) runs as a Docker Compose project on
a shared **AWS EC2** box, behind an nginx edge. See [docker-compose.yml](docker-compose.yml).

Corpus: **62 EU-public-domain works** (~46k chunks) from Project Gutenberg — manifest in
[corpus/](corpus/).

## Repo layout

| Path | What |
|---|---|
| [backend/](backend/) | Python service — ingest, retrieval, agent, evals, API ([commands](backend/README.md)) |
| [frontend/](frontend/) | Vite + React + TS SPA — chat, evals tab, how-it-works (nginx + `/api` proxy, Dockerized) |
| [docs/](docs/) | Decision docs + ADRs: [stack](docs/python-stack.md) · [embeddings](docs/embeddings.md) · [vector stores](docs/vector-stores.md) · [chunking](docs/chunking.md) · [RAG techniques](docs/rag-techniques.md) · **[eval log](docs/eval-log.md)** |
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
carries the *methodology* — not the code — onto a 10×+ corpus with the production Python stack.
Every technique here entered through a measured ablation; the receipts are published in
[docs/eval-log.md](docs/eval-log.md).
