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
uv run ahx --help       # pipeline CLI (ingest / eval / serve)
uv run ahx serve        # dev API server → http://127.0.0.1:8000/docs
```

Local Postgres (pgvector): `docker compose up -d` from the repo root.
Configuration: copy `.env.example` → `.env` (see `src/ahx/config.py`).
