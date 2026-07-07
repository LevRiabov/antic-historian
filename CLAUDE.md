# CLAUDE.md

## What this project is

Production-grade RAG system over ancient-history sources (Greco-Roman antiquity, dozens of
public-domain books). Dual purpose: **freelance portfolio asset** (deployed demo + case study
that convinces clients) and **learning vehicle** for the production Python AI stack.
Successor to `../rag-historian` (hand-rolled TS learning repo) — we carry its *methodology
and findings*, not its code.

The user is a TypeScript fullstack developer learning Python — when introducing a Python
idiom, a one-line TS analogy helps (pydantic ≈ zod, uv ≈ pnpm, pytest fixtures ≈ DI).

## Where decisions live

| Doc | Contents |
|---|---|
| [project-plan.md](project-plan.md) | Goals, settled stack, **decision gates D1–D5**, phases with exit criteria |
| [docs/python-stack.md](docs/python-stack.md) | Tooling choices + TS translation table |
| [docs/embeddings.md](docs/embeddings.md) | Embedding landscape; D2 shortlist (front-runner: voyage-4-nano) |
| [docs/vector-stores.md](docs/vector-stores.md) | DB landscape; D3 default: Postgres+pgvector (prod: self-hosted in-compose) |
| [docs/chunking.md](docs/chunking.md) | Parse-then-chunk architecture, canonical locators |
| [docs/rag-techniques.md](docs/rag-techniques.md) | Full technique menu, interactions, Phase 4 build order |
| [docs/golden-set.md](docs/golden-set.md) | Golden set format + authoring workflow (MCP tools) |
| [docs/eval-log.md](docs/eval-log.md) | **Append-only measured-results log** — every significant eval run gets an entry |
| [module-10-build-plan.md](module-10-build-plan.md) | Buyer-facing feature thinking (UX North Star) |
| docs/adr/ | One ADR per gate decision when taken |

Gate status: **D1 decided** ([ADR-001](docs/adr/001-d1-framework.md)): LlamaIndex = RAG layer,
LangGraph = agent orchestration, thin project-owned interface between frameworks and our code
(no framework types in eval harness / API / ablation modules; models passed explicitly — never
LlamaIndex global `Settings`). **D2 decided** ([ADR-002](docs/adr/002-d2-embeddings.md)):
qwen3-embedding-8b hosted (OpenRouter pinned to Nebius, 1024d MRL); local qwen3-0.6b =
documented fallback. **D3 decided** (2026-06-14): Postgres + pgvector — hybrid
ablation confirmed the default (prod: self-hosted in-compose on EC2; Neon superseded at
deploy, DB > free tier). **D5 lineup decided** ([ADR-003](docs/adr/003-d5-llm-lineup.md)):
agent = deepseek-v4-pro, judge = split kimi-k2.6 + qwen3.7-max (attribution); retrieval ships
**rerank-free** (`dense-ctx-v1` — the cohere-pro reranker was dropped). Fast-path/cheap-tier/
fallback still open → Phase 6. **D4 decided** ([ADR-004](docs/adr/004-d4-frontend.md)): Vite+React
SPA, nginx-in-Docker, `/api` proxy (no CORS) — Phase 7 scaffold landed, pages next.

## Repo layout

- `backend/` — Python service (`ahx` package, src layout): `ingest/`, `retrieval/`, `agent/`,
  `evals/`, `api/`, `cli.py`. All offline work (ingest, evals) goes through the typer CLI.
- `frontend/` — TS app (Phase 7): Vite+React+TS SPA, Tailwind v4, typed SSE client in `src/lib/`,
  Dockerized (nginx + `/api` proxy). Scaffold landed; design mockups in `frontend/design/`.
- `corpus/` — manifest committed; downloaded texts gitignored.
- `docs/` — decision docs (above) + ADRs.

## Commands (run in `backend/`)

```sh
uv sync                # install everything (creates .venv, editable-installs ahx)
uv run pytest          # tests
uv run ruff format . && uv run ruff check .   # format + lint
uv run pyright         # strict type check
uv run ahx serve       # dev API → http://127.0.0.1:8000/docs
docker compose up -d   # local Postgres+pgvector (run at repo root)
```

CI mirrors exactly these (`.github/workflows/ci.yml`). All four must pass before a commit
is "done".

## Hard rules (project methodology — these are the point of the project)

1. **Evals first.** No retrieval/generation technique ships without golden-set evidence.
   Techniques enter through the Phase 4 ablation door: implement → measure → keep/reject →
   decision note. Rejections with receipts are case-study content, not failures.
2. **Gates, not vibes.** Framework/embedder/DB/frontend choices happen at gates D1–D5 with
   written criteria. Don't adopt a framework, model, or store ahead of its gate.
3. **One embedding module.** Every embed call goes through a single module owning the
   query/document prefix policy. Parity test (cosine ≥ 0.999 on fixtures) after ANY
   runtime/model change. Footguns: docs/embeddings.md §6.
4. **Representation alignment.** The reranker scores the same (contextualized) text that
   was embedded — proven law from rag-historian (bare-text rerank undid contextual gains).
5. **The eval harness is production code** — typed pydantic records, tests, versioned runs.
   Measurement bugs moved numbers more than real changes last time (3 documented cases).
6. **Verified claims only in docs.** Pricing/free-tier/benchmark numbers carry a date and,
   if unverified, a ⚠ marker. This repo's credibility is the product.
7. **Async end-to-end in the API** — no sync HTTP/DB calls inside FastAPI routes
   (blocks the event loop; Python won't warn you).

## Conventions

- pyright strict; pydantic models wherever data crosses a boundary (API, LLM output, config,
  eval records); plain type hints elsewhere.
- Costs: ingest-time spend (one-time, local GPU) is cheap; query-time spend (latency + $ per
  request, forever) is expensive — design decisions accordingly.
- LLM access is provider-agnostic (D5): never hardcode a provider/model in business logic.
- Corpus texts are EU-public-domain only; the manifest documents the PD basis per work —
  preserve that diligence for any source added.
