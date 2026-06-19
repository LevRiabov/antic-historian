"""Live integration tests — the legs the unit suite stubs out (real Postgres / hosted
LLM endpoints). Excluded from the default run and CI (the suite stays hermetic + fast);
run them against a live stack with:

    uv run pytest -m integration

Each test SKIPS itself unless its required env is present, so an accidental
`-m integration` on a bare machine is a clean skip, never a hard failure. This makes
the "exercised live by the CLI" claims runnable under pytest rather than only by hand.
"""

import os

import pytest
from sqlalchemy import text

from ahx.config import Settings
from ahx.db import create_async_db_engine
from ahx.retrieval.embedding import EmbeddingClient

pytestmark = pytest.mark.integration

# Opt-in signal: an explicitly-set AHX_DATABASE_URL means the caller pointed us at a
# real DB (the in-code default is not enough — we don't want to hit a stray localhost).
_HAS_DB = "AHX_DATABASE_URL" in os.environ
_HAS_EMBED_KEY = bool(os.environ.get("AHX_EMBED_API_KEY"))


@pytest.mark.skipif(not _HAS_DB, reason="set AHX_DATABASE_URL to run DB integration tests")
async def test_database_reachable() -> None:
    engine = create_async_db_engine(Settings().database_url)
    try:
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        await engine.dispose()


@pytest.mark.skipif(not _HAS_EMBED_KEY, reason="set AHX_EMBED_API_KEY to run live embed test")
async def test_embed_query_returns_configured_dim() -> None:
    settings = Settings()
    client = EmbeddingClient(settings)
    try:
        vector = await client.embed_query("How did Caesar die?")
        assert len(vector) == settings.embed_dim
    finally:
        await client.aclose()
