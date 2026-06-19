# Serving image for the FastAPI backend. Offline work (ingest, evals) runs via the
# CLI, not here — this image only needs the runtime deps + the published eval-run
# JSON the read-only API serves at startup (api/evals.py).
#
# Base = uv's own python-3.13 image (uv + interpreter preinstalled, pinned digest by
# tag). Build deps with the lockfile so the container resolves identically to local.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependency layer first: cached unless the lock/manifest changes. --no-install-project
# so editing app code doesn't bust this (the slow) layer. README is referenced by
# pyproject (readme=...), so it must be present for the later project build.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# App code + the committed eval-run records (served read-only from <backend>/evals).
# config._BACKEND_DIR resolves to /app here, so the API reads /app/evals/runs.
COPY src ./src
COPY evals ./evals
RUN uv sync --frozen --no-dev

# Drop root: the serving process never needs to write the image.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# Linux uses uvloop/selector — the Windows proactor footgun handled in cli.serve does
# NOT apply here, so run uvicorn directly. 0.0.0.0 so nginx (other container) can reach
# it; only nginx is published to the host (see docker-compose.yml).
CMD ["uvicorn", "ahx.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
