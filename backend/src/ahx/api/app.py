"""FastAPI application entrypoint."""

from fastapi import FastAPI

import ahx

app = FastAPI(title="Antic Historian API", version=ahx.__version__)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": ahx.__version__}
