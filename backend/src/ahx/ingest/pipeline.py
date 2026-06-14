"""Ingestion pipeline orchestration: raw files -> normalized works + QA report.

Each step is idempotent and re-runnable; a book failing loudly here is the
design (docs/chunking.md §6), not an error to paper over.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ahx.ingest.chunker import chunk_work
from ahx.ingest.clean import clean_raw
from ahx.ingest.gutenberg import split_paragraphs
from ahx.ingest.manifest import ManifestEntry
from ahx.ingest.model import NormalizedWork
from ahx.ingest.parsers import parse_structure

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from ahx.retrieval.embedding import EmbeddingClient


class NormalizeReport(BaseModel):
    """QA row for one work — what the per-book report table is built from."""

    pg_id: int
    title: str
    parser: str
    divisions: int
    paragraphs: int
    chars: int
    max_paragraph_chars: int
    error: str | None = None

    @property
    def flags(self) -> list[str]:
        flags: list[str] = []
        if self.error:
            flags.append(f"ERROR: {self.error}")
            return flags
        if self.parser == "flat":
            flags.append("fallback parser — no structure extracted")
        if self.divisions and self.chars // max(self.divisions, 1) > 30_000:
            flags.append("divisions look too coarse (>30k chars avg)")
        if self.max_paragraph_chars > 8_000:
            flags.append("giant paragraph (>8k chars) — check cleaning")
        return flags


def normalize_work(entry: ManifestEntry, raw_dir: Path, out_dir: Path) -> NormalizeReport:
    raw_path = raw_dir / entry.raw_filename
    if not raw_path.exists():
        return NormalizeReport(
            pg_id=entry.pg_id,
            title=entry.title,
            parser="-",
            divisions=0,
            paragraphs=0,
            chars=0,
            max_paragraph_chars=0,
            error="raw file missing — run `ahx ingest download` first",
        )
    try:
        cleaned = clean_raw(raw_path.read_text(encoding="utf-8"), entry.raw_file)
    except ValueError as exc:
        return NormalizeReport(
            pg_id=entry.pg_id,
            title=entry.title,
            parser="-",
            divisions=0,
            paragraphs=0,
            chars=0,
            max_paragraph_chars=0,
            error=str(exc),
        )

    paragraphs = split_paragraphs(cleaned)
    parser_name, divisions = parse_structure(paragraphs)
    work = NormalizedWork(
        pg_id=entry.pg_id,
        author=entry.author,
        title=entry.title,
        category=entry.category,
        translator=entry.translator,
        parser=parser_name,
        divisions=divisions,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / entry.normalized_filename).write_text(
        work.model_dump_json(indent=2), encoding="utf-8"
    )
    return NormalizeReport(
        pg_id=entry.pg_id,
        title=entry.title,
        parser=parser_name,
        divisions=len(divisions),
        paragraphs=work.paragraph_count,
        chars=work.char_count,
        max_paragraph_chars=max((len(p) for d in divisions for p in d.paragraphs), default=0),
    )


class LoadReport(BaseModel):
    pg_id: int
    title: str
    chunks: int
    status: str  # "loaded" | "cached" | "error"
    detail: str = ""


def load_one(
    entry: ManifestEntry,
    chunks_dir: Path,
    engine: "Engine",
    embedder: "EmbeddingClient",
) -> LoadReport:
    """Embed one work's chunks and upsert into Postgres. Idempotent: if the
    chunk count for (pg_id, chunking_version) already matches, it's a no-op."""
    from sqlalchemy import delete, func, select
    from sqlalchemy.orm import Session

    from ahx.db import ChunkRow, SourceRow
    from ahx.ingest.chunker import CHUNKING_VERSION, Chunk
    from ahx.ingest.enrich import (
        ENRICHMENT_VERSION,
        EnrichedChunk,
        enriched_path,
        heading_path,
        retrieval_representation,
    )

    jsonl_path = chunks_dir / f"pg{entry.pg_id}.jsonl"
    if not jsonl_path.exists():
        return LoadReport(
            pg_id=entry.pg_id,
            title=entry.title,
            chunks=0,
            status="error",
            detail="chunks file missing — run `ahx ingest chunk` first",
        )
    chunks = [
        Chunk.model_validate_json(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Phase 4.1: if this work has been enriched at the current version, embed
    # the contextual representation; otherwise fall back to bare text (dense-v1).
    enriched: dict[int, EnrichedChunk] = {}
    enr_path = enriched_path(chunks_dir.parent / "enriched", entry.pg_id)
    if enr_path.exists():
        for line in enr_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = EnrichedChunk.model_validate_json(line)
            if record.enrichment_version == ENRICHMENT_VERSION:
                enriched[record.chunk_index] = record

    def representation(chunk: Chunk) -> tuple[str, EnrichedChunk | None]:
        record = enriched.get(chunk.chunk_index)
        if record is None:
            return chunk.text, None
        head = heading_path(entry.author, entry.title, chunk.locator, chunk.heading)
        return retrieval_representation(record.context_note, head, chunk.text), record

    normalized_path = chunks_dir.parent / "normalized" / entry.normalized_filename
    work = NormalizedWork.model_validate_json(normalized_path.read_text(encoding="utf-8"))

    with Session(engine) as session:
        session.merge(
            SourceRow(
                pg_id=entry.pg_id,
                author=entry.author,
                title=entry.title,
                category=entry.category,
                translator=entry.translator,
                parser=work.parser,
            )
        )
        existing = session.scalar(
            select(func.count())
            .select_from(ChunkRow)
            .where(ChunkRow.pg_id == entry.pg_id)
            .where(ChunkRow.chunking_version == CHUNKING_VERSION)
        )
        if existing == len(chunks):
            session.commit()
            return LoadReport(
                pg_id=entry.pg_id, title=entry.title, chunks=len(chunks), status="cached"
            )

        session.execute(
            delete(ChunkRow)
            .where(ChunkRow.pg_id == entry.pg_id)
            .where(ChunkRow.chunking_version == CHUNKING_VERSION)
        )
        for start in range(0, len(chunks), embedder.batch_size):
            batch = chunks[start : start + embedder.batch_size]
            reps = [representation(c) for c in batch]
            vectors = embedder.embed_documents([rep for rep, _ in reps])
            session.add_all(
                ChunkRow(
                    pg_id=c.pg_id,
                    chunk_index=c.chunk_index,
                    chunking_version=c.chunking_version,
                    division_index=c.division_index,
                    locator=c.locator,
                    heading=c.heading,
                    text=c.text,
                    char_start=c.char_start,
                    char_end=c.char_end,
                    token_count=c.token_count,
                    context_note=record.context_note if record else None,
                    retrieval_text=rep if record else None,
                    enrichment_version=record.enrichment_version if record else None,
                    entities=record.entities if record else None,
                    dates=record.dates if record else None,
                    embedding=vector,
                )
                for c, (rep, record), vector in zip(batch, reps, vectors, strict=True)
            )
        session.commit()
    return LoadReport(pg_id=entry.pg_id, title=entry.title, chunks=len(chunks), status="loaded")


class ChunkReport(BaseModel):
    pg_id: int
    title: str
    chunks: int
    mean_tokens: int
    max_tokens: int
    oversize: int  # chunks above budget (single giant sentence fallthrough)
    error: str | None = None


def chunk_one(entry: ManifestEntry, normalized_dir: Path, chunks_dir: Path) -> ChunkReport:
    normalized_path = normalized_dir / entry.normalized_filename
    if not normalized_path.exists():
        return ChunkReport(
            pg_id=entry.pg_id,
            title=entry.title,
            chunks=0,
            mean_tokens=0,
            max_tokens=0,
            oversize=0,
            error="normalized file missing — run `ahx ingest normalize` first",
        )
    work = NormalizedWork.model_validate_json(normalized_path.read_text(encoding="utf-8"))
    chunks = chunk_work(work)

    chunks_dir.mkdir(parents=True, exist_ok=True)
    out_path = chunks_dir / f"pg{entry.pg_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")

    token_counts = [c.token_count for c in chunks]
    return ChunkReport(
        pg_id=entry.pg_id,
        title=entry.title,
        chunks=len(chunks),
        mean_tokens=sum(token_counts) // max(len(token_counts), 1),
        max_tokens=max(token_counts, default=0),
        oversize=sum(1 for t in token_counts if t > 600),
    )
