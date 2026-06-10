# Python Stack — Tooling Choices for a TS Developer

> **Status:** settled choices for Phase 0 scaffolding (framework choice itself is Gate D1, spiked separately).
> **Written:** 2026-06-10. The Python tool ecosystem consolidated hard in 2024–2026 around Astral's Rust-based tools (uv, ruff) — most older tutorials (pip+virtualenv+black+flake8+poetry) describe the *previous* era. This doc is the current mainstream.

---

## 1. The translation table (TS → Python)

Your TS instincts map almost 1:1 — these are the tools that fill the same slot:

| You know (TS) | Python equivalent | Notes |
|---|---|---|
| pnpm / npm | **uv** | Installer + venv + lockfile + Python-version manager in one Rust binary; 10–100× faster than pip |
| package.json | **pyproject.toml** | Deps, scripts-ish, tool config in one file |
| pnpm-lock.yaml | **uv.lock** | Committed, reproducible |
| Biome (lint+format) | **ruff** | One Rust binary replacing black (format) + flake8 (lint) + isort (imports); same "fast, zero-config" philosophy as Biome |
| tsc (type checking) | **pyright** | Type *checker* only — Python types are erased at runtime, like TS. (Astral's `ty` is the coming thing — beta, ~53% spec conformance; watch, don't adopt) |
| zod | **pydantic v2** | Runtime validation + parsing, Rust-core fast. THE load-bearing library of the Python AI ecosystem — FastAPI, LangChain, LlamaIndex all build on it |
| dotenv + config | **pydantic-settings** | Typed settings class, reads env/.env, validates at startup |
| vitest / jest | **pytest** (+ pytest-asyncio, pytest-cov) | Fixtures are DI-by-argument-name — different flavor, more powerful than beforeEach |
| express / fastify | **FastAPI** | Async, typed, OpenAPI docs auto-generated from pydantic models; uvicorn = the server (like node's runtime serving express) |
| fetch / axios | **httpx** | Sync + async, HTTP/2; (requests = the legacy axios) |
| commander / yargs | **typer** | CLI from type hints (by the FastAPI author); pairs with **rich** for output |
| ESM imports | Python modules | No build step ever; but watch the "src layout" gotcha (§4) |
| tsx (run TS directly) | just `uv run python file.py` | uv handles venv activation implicitly |

## 2. Choices by category (option → popularity → our pick)

### Package & project management — **uv** ✅
- **The field:** pip+venv (legacy default), poetry (the 2020–2023 standard, now losing ground), conda (data-science niche, heavyweight), uv (2024+, by Astral).
- **State of play:** uv is the de-facto standard for new projects in 2026 — handles Python installation itself (`uv python install 3.13`), venvs, lockfiles, tool isolation (`uv tool run`), and publishing. Poetry knowledge is still common in older repos; conda only matters for CUDA-pinned research stacks.
- **Us:** uv, Python **3.13** (3.14 is out but library wheels lag ~6–12 months; 3.13 is the safe-modern point).

### Lint & format — **ruff** ✅
- **The field:** black+flake8+isort+pylint (the old four-tool pile), ruff (replaces all four, ~100× faster).
- **Us:** ruff with format + lint, a curated rule set (start `E,F,I,UP,B,SIM`), in pre-commit and CI. No debates — like Biome, you adopt its opinions.

### Type checking — **pyright** ✅ (strict mode)
- **The field:** mypy (the original, slower, plugin ecosystem), pyright (Microsoft, powers VS Code's Pylance, 98% spec conformance, fast), ty & pyrefly (Rust newcomers, beta).
- **Us:** pyright strict. You'll feel at home; Python type syntax ≈ TS with different spelling (`list[str]`, `X | None`, `Protocol` ≈ structural interfaces). Re-evaluate ty at its 1.0.

### Validation & config — **pydantic v2 + pydantic-settings** ✅
- Everything crossing a boundary (API request, LLM structured output, config, eval records) gets a pydantic model — exactly where you used zod. `model_validate_json()` ≈ `schema.parse(JSON.parse(...))`.

### Testing — **pytest** ✅
- Plus **pytest-asyncio** (async tests), **pytest-cov** (coverage). The eval harness is *not* pytest — it's our own CLI producing scorecards — but harness unit tests are.

### API — **FastAPI + uvicorn** ✅
- **The field:** Django (batteries-included monolith — wrong shape for an API service), Flask (sync-era veteran), FastAPI (async, pydantic-native, OpenAPI for free, the AI-service default), Litestar (FastAPI competitor, technically excellent, smaller mindshare).
- **Us:** FastAPI. Streaming via **sse-starlette** (SSE for token/citation streaming). Rate limiting via **slowapi** or hand-rolled middleware (Phase 6 decision).

### Database access — **SQLAlchemy 2.0 + psycopg3 + Alembic** ✅
- **The field:** SQLAlchemy 2.0 (the standard ORM/query-builder; 2.0 style is typed and explicit), raw drivers (psycopg3/asyncpg) for SQL purists, SQLModel (pydantic×SQLAlchemy hybrid, FastAPI author — nice but thin), Django ORM (no).
- **Us:** SQLAlchemy 2.0 (Core + light ORM) with psycopg3 async driver; **Alembic** for migrations (≈ Prisma migrate); **pgvector-python** registers the vector type with both. Rationale: SQLAlchemy+Alembic is the combo you'll meet in client codebases — résumé-relevant; raw-SQL escape hatch stays open for the hybrid-search RRF queries.

### LLM / RAG framework — Gate D1 spike, candidates: ✅ to be decided
- **LlamaIndex** — RAG-first: ingestion pipelines, indices (incl. RAPTOR pack, PropertyGraphIndex), retrievers, rerankers as first-class objects. Working hypothesis for the RAG layer.
- **LangChain + LangGraph** — broadest ecosystem; LangGraph (graph-state agent orchestration) is genuinely good and widely demanded; LangChain-core less loved in 2026 but unavoidable vocabulary.
- **Haystack 2.x** — cleanest pipeline abstraction, production-lean, smaller mindshare.
- Supporting cast regardless of winner: **LiteLLM** (provider-agnostic LLM calls + fallback chains + cost tracking — implements our D5 swappability requirement directly; the framework's own LLM classes can sit on top of or beside it), **instructor** (pydantic-validated structured outputs) if the framework's native version disappoints.
- Spike plan stands (§Phase 0): same toy pipeline in 2–3 candidates, ADR decides.

### Embedding/reranker serving (local inference) — **sentence-transformers** first ✅
- **sentence-transformers** — the reference way to run embedders/cross-encoders (handles prompts/pooling correctly per model card — avoids the parity footguns).
- **ONNX Runtime via optimum / fastembed** — the CPU-deployment optimization (int8) once a model is *chosen*; verify parity vs sentence-transformers (cosine ≥ 0.999 on fixtures).
- **llama.cpp / TEI / infinity** — alternative servers; only if containerizing the model separately from the API.
- **tokenizers / tiktoken** — token counting for chunking budgets (≈ js-tiktoken).

### Eval — **own harness** (pydantic records + typer CLI), frameworks as garnish ✅
- **The field:** ragas (RAG-metric library — convenient, but opaque prompts and noisy versions), deepeval (pytest-style LLM evals), promptfoo (you know it; TS, works fine alongside), braintrust/langsmith (hosted, vendor-tied).
- **Us:** port the rag-historian methodology as our own code — the judge prompts, rubrics, and category breakdowns ARE the portfolio asset; black-box metrics would gut the case study. Optionally run ragas once as a sanity cross-check row.

### Observability — **langfuse** SDK ✅ (+ **structlog** for app logs)

### CLI / pipeline entrypoints — **typer + rich** ✅
- `uv run ahx ingest --source herodotus-1`, `uv run ahx eval --suite golden-v1` … one typer app, subcommands per pipeline stage, rich progress bars for the 35k-chunk passes.

### Exploration — **jupyter** via `uv run --with jupyter`, or **marimo** (modern reactive notebooks, git-friendly `.py` files). Notebooks for retrieval forensics; anything load-bearing graduates to the package.

## 3. What we deliberately skip (and why)

| Skipped | Why |
|---|---|
| poetry, pipenv, conda | uv replaced this entire generation |
| black, flake8, isort, pylint | ruff does all of it |
| Django, Flask | wrong shape for an async AI API service |
| celery + redis (task queue) | our only "background job" is offline local ingest — a CLI, not a queue |
| LangChain *without* the D1 spike | adopting by mindshare instead of measurement is exactly what this project argues against |
| asyncpg directly | psycopg3 async is the maintained mainstream path with SQLAlchemy 2.0 |

## 4. Project skeleton (Phase 0 target)

```
antic-historian/
├── backend/
│   ├── pyproject.toml      # deps, ruff/pyright/pytest config — the package.json
│   ├── uv.lock
│   ├── .python-version     # 3.13 — like .nvmrc
│   ├── src/
│   │   └── ahx/            # the package ("antic historian", short import name)
│   │       ├── config.py   # pydantic-settings
│   │       ├── ingest/     # parsers → normalize → chunk → embed (Layer 1/2 from chunking.md)
│   │       ├── retrieval/  # dense/hybrid/rerank — each technique one module, ablation-friendly
│   │       ├── agent/      # tool loop (framework-native)
│   │       ├── evals/      # golden set, judges, scorecards
│   │       ├── api/        # FastAPI app, SSE streaming, middleware
│   │       └── cli.py      # typer entrypoint
│   └── tests/
├── frontend/               # TS app (Phase 7, gate D4)
├── docs/                   # these decision docs + ADRs
├── corpus/                 # manifest committed; raw/normalized texts gitignored
└── docker-compose.yml      # local Postgres+pgvector
```

Gotchas a TS dev hits in week one, pre-answered:
- **src layout** requires the package be installed to import (`uv sync` does it via editable install) — prevents the "works from repo root only" trap.
- **Async is opt-in per stack, not ambient**: one sync call (`requests`, sync DB driver) inside an async route blocks the event loop — Python's version of "don't block the loop", but the compiler won't warn you. Stay `httpx.AsyncClient` + psycopg3-async throughout the API.
- **Mutable default arguments** (`def f(x=[])`) are shared across calls — the classic; ruff rule B006 catches it.
- Type hints don't validate anything at runtime — pydantic models where data crosses boundaries, plain hints elsewhere.

## 5. CI (Phase 0 exit criterion)

GitHub Actions: `uv sync` → `ruff check` + `ruff format --check` → `pyright` → `pytest` — each a separate fast job; pre-commit mirrors the first three locally. Eval runs are *not* CI (cost money, run at phase boundaries via the CLI).
