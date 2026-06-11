# Backend (`ahx`)

Python service: ingestion pipeline, retrieval stack, agent, eval harness, FastAPI API.
Stack rationale: [docs/python-stack.md](../docs/python-stack.md).

## Commands

```sh
uv sync                 # install deps + editable package into .venv
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check (strict)
uv run ahx --help       # pipeline CLI (ingest / eval / db / mcp / serve)
uv run ahx serve        # dev API server → http://127.0.0.1:8000/docs

# Ingestion (idempotent; requires docker DB + local llama-swap):
uv run ahx ingest download    # manifest → corpus/raw/
uv run ahx ingest normalize   # clean + structure-parse + QA report
uv run ahx ingest chunk       # → corpus/chunks/*.jsonl + stats
uv run ahx ingest load        # embed + load into pgvector
uv run ahx ingest parity      # embedding drift check (rule #3)

# Evaluation (golden set: evals/golden/*.yaml, results log: docs/eval-log.md):
uv run ahx eval validate      # schema + quote resolution + category counts
uv run ahx eval run           # recall@k / MRR scorecard → evals/runs/<ts>.json

uv run ahx search "How did Caesar die?"   # debug similarity search
```

Local Postgres (pgvector): `docker compose up -d` from the repo root.
Configuration: copy `.env.example` → `.env` (see `src/ahx/config.py`).
