"""Database layer: SQLAlchemy 2.0 models + engine helpers.

One Postgres for everything (ADR-pending gate D3, default per
docs/vector-stores.md): relational backbone (sources, later golden set & eval
runs) AND vectors (pgvector) live side by side, joinable with SQL.

Schema management is `Base.metadata.create_all` for now; Alembic migrations
enter once the schema stabilizes (pre-Phase-3).
"""

from typing import Any

from pgvector.sqlalchemy import Vector  # pyright: ignore[reportMissingTypeStubs]
from sqlalchemy import Computed, ForeignKey, Index, Text, UniqueConstraint, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ahx.config import get_settings

# Config-driven for the D2 ablation (candidate dims: 768/1024/2048). Read at
# import time — the column type is fixed per process, so every process in a
# given ablation arm must run with the same AHX_EMBED_DIM.
EMBED_DIM = get_settings().embed_dim


class Base(DeclarativeBase):
    pass


class SourceRow(Base):
    __tablename__ = "sources"

    pg_id: Mapped[int] = mapped_column(primary_key=True)
    author: Mapped[str]
    title: Mapped[str]
    category: Mapped[str]
    translator: Mapped[str]
    parser: Mapped[str]


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pg_id: Mapped[int] = mapped_column(ForeignKey("sources.pg_id", ondelete="CASCADE"))
    chunk_index: Mapped[int]
    chunking_version: Mapped[str]
    division_index: Mapped[int]
    locator: Mapped[list[str]] = mapped_column(JSONB)
    heading: Mapped[str | None]
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int]
    char_end: Mapped[int]
    token_count: Mapped[int]
    # Phase 4.1 contextual enrichment (nullable: a pre-enrichment load leaves
    # them empty and behaves exactly like dense-v1). `retrieval_text` is what
    # gets embedded AND, in 4.2, reranked — alignment law (rule #4). `text`
    # above stays the verbatim passage for generation + citations.
    context_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    enrichment_version: Mapped[str | None] = mapped_column(nullable=True)
    entities: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    dates: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    embedding: Mapped[Any] = mapped_column(Vector(EMBED_DIM))
    # Phase 4.3 hybrid: BM25-class lexical index over the verbatim `text` (NOT
    # retrieval_text — keyword match is on the real passage words). A STORED
    # generated column so it auto-maintains and backfills on ADD (no re-embed).
    text_tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True), nullable=True
    )

    __table_args__ = (
        UniqueConstraint("pg_id", "chunking_version", "chunk_index"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_text_tsv_gin", "text_tsv", postgresql_using="gin"),
    )


def create_sync_engine(database_url: str) -> Engine:
    return create_engine(database_url)


def create_async_db_engine(database_url: str) -> AsyncEngine:
    """For API routes (rule #7). Same psycopg driver/URL as the sync engine.

    Windows footgun: psycopg async needs a selector event loop, but the
    default (and uvicorn's no-reload loop factory) is the proactor loop.
    Every entrypoint that awaits this engine must force one, e.g.
    `asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)` — see
    cli.serve for the uvicorn wiring.
    """
    return create_async_engine(database_url)


def init_db(engine: Engine) -> None:
    """Create the vector extension and all tables (idempotent)."""
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def ensure_fts(engine: Engine) -> None:
    """Add the BM25/FTS tsvector column + GIN index IN PLACE (Phase 4.3 hybrid).

    Idempotent. The column is STORED-generated from `text`, so Postgres backfills
    all existing rows on ADD — no reload, no re-embed. `create_all` (fresh DB)
    already produces the same column/index from the model; this path upgrades a
    DB that was loaded before the column existed. Same names both ways.
    """
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_tsv tsvector "
                "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
            )
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_chunks_text_tsv_gin ON chunks USING gin (text_tsv)")
        )


def reset_chunks(engine: Engine) -> None:
    """Drop + recreate the chunks table (D2 ablation: new embed model/dim).

    Sources stay; chunks reload from corpus/chunks/*.jsonl via `ahx ingest
    load`. Needed because the loader's idempotency key (pg_id +
    chunking_version) doesn't know about the embedding model, and a dim
    change alters the column type anyway.
    """
    chunks = Base.metadata.tables["chunks"]
    chunks.drop(engine, checkfirst=True)
    chunks.create(engine)
