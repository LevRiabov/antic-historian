"""Phase 4.1 — contextual enrichment pass (docs/rag-techniques.md §2).

One offline LLM call per chunk produces a short **context note** (situates the
passage: work, section, who "he" is, which war) plus **entities/dates** metadata,
in a single grammar-constrained JSON reply. The note + heading path become the
chunk's *retrieval representation* (embedded and, in 4.2, reranked — alignment
law, CLAUDE.md rule #4); the original text + locator still drive generation.

Design for an unattended overnight run over ~46k chunks:

- **Cached to disk** (`corpus/enriched/pgNNNN.jsonl`), keyed by
  `enrichment_version`. The expensive LLM pass happens ONCE per version — every
  later re-embed (D2 follow-ups, dim changes, the rerank arm) reads this cache,
  never the LLM. This is the whole point: 46k chunks is too much to repeat.
- **Resumable + crash-safe.** Results are appended line-by-line and flushed; a
  re-run skips chunks already present for the current version. A dead process,
  a power cut, or an OOM costs only the in-flight calls.
- **Concurrent.** A global semaphore bounds in-flight calls to the llama-swap
  profile's parallel-slot count (continuous batching on the GPU).

Windowed context (not whole-document): local gemma's 16k ctx can't hold a
290k-token book, and the heading path carries cross-book disambiguation for
free — so each call sees only work/section headers + the chunk + its immediate
neighbors. Small prompts = high throughput.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from ahx.ingest.chunker import Chunk
from ahx.ingest.manifest import ManifestEntry
from ahx.llm import ChatMessage, ChatModel

ENRICHMENT_VERSION = "enrich-v1"

# Neighbor context is for disambiguation only — a few hundred chars of the
# adjacent passages is enough to resolve "he"/"the city"; more just inflates
# the prompt and slows the batch.
_NEIGHBOR_CHARS = 300

_SYSTEM_PROMPT = (
    "You annotate passages from public-domain ancient-history books to improve "
    "search retrieval. For the PASSAGE TO SITUATE, write a short context note "
    "(1-2 sentences, under 45 words) that situates it: name the work and author, the book or "
    "section, the people or peoples involved (resolve pronouns like 'he'/'they' "
    "to names), and the war, campaign, or event described. The note is prepended "
    "to the passage and embedded for search, so pack it with the proper nouns a "
    "searcher would type. Do NOT enumerate long lists in the note — summarize; "
    "the specific names belong in the entities field. Then list up to 12 key "
    "entities (people, peoples, places) and up to 8 dates or time periods "
    "mentioned. Use ONLY information "
    "present in the passages shown; do not invent facts. Respond with JSON only."
)

# llama.cpp compiles this JSON schema to a GBNF grammar, so the reply is always
# parseable — no malformed-JSON failure mode across 46k unattended calls.
ENRICH_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "chunk_enrichment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                # maxItems is enforced by the grammar -> the model CANNOT run the
                # array long enough to overflow max_tokens (the truncation that
                # produced unparseable JSON + slow full-length decodes in the spike).
                "context_note": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                "dates": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            },
            "required": ["context_note", "entities", "dates"],
            "additionalProperties": False,
        },
    },
}


class EnrichmentFields(BaseModel):
    """The model's JSON reply (what the LLM produces)."""

    context_note: str
    entities: list[str]
    dates: list[str]


class EnrichedChunk(BaseModel):
    """One cached enrichment record (one line of corpus/enriched/pgNNNN.jsonl).

    Keyed by (pg_id, chunk_index, chunking_version, enrichment_version) — the
    loader joins this to ChunkRow to build the retrieval representation.
    """

    pg_id: int
    chunk_index: int
    chunking_version: str
    enrichment_version: str
    context_note: str
    entities: list[str]
    dates: list[str]


def heading_path(author: str, title: str, locator: list[str], heading: str | None) -> str:
    """`Author, Title > BOOK I > IV (Heading)` — the free, structural part of
    the retrieval representation. Shared by the prompt and the loader so the
    string the note was written against is the string we embed."""
    path = f"{author}, {title}"
    if locator:
        path += " > " + " > ".join(locator)
    if heading and heading not in locator:
        path += f" ({heading})"
    return path


def retrieval_representation(context_note: str, heading: str, chunk_text: str) -> str:
    """What gets embedded (and, in 4.2, reranked). Order: note, then heading
    path, then the verbatim passage."""
    return f"{context_note}\n{heading}\n\n{chunk_text}"


def build_messages(
    entry: ManifestEntry,
    chunk: Chunk,
    prev_text: str,
    next_text: str,
) -> list[ChatMessage]:
    prev_ctx = prev_text[-_NEIGHBOR_CHARS:].strip() if prev_text else "(start of work)"
    next_ctx = next_text[:_NEIGHBOR_CHARS].strip() if next_text else "(end of work)"
    user = (
        f"Work: {entry.author}, {entry.title}\n"
        f"Section: {heading_path(entry.author, entry.title, chunk.locator, chunk.heading)}\n\n"
        f"Preceding passage (context, do not summarize):\n{prev_ctx}\n\n"
        f">>> PASSAGE TO SITUATE <<<\n{chunk.text}\n\n"
        f"Following passage (context, do not summarize):\n{next_ctx}"
    )
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]


def parse_enrichment(raw: str) -> EnrichmentFields | None:
    """Tolerant parse. With the grammar this is just json.loads; the substring
    fallback covers a model/runtime swap that doesn't honor response_format."""
    try:
        return EnrichmentFields.model_validate_json(raw)
    except ValidationError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return EnrichmentFields.model_validate_json(raw[start : end + 1])
    except ValidationError:
        return None


def enriched_path(enriched_dir: Path, pg_id: int) -> Path:
    return enriched_dir / f"pg{pg_id}.jsonl"


def load_done(path: Path, enrichment_version: str) -> set[int]:
    """chunk_index values already enriched at the current version (resume key).
    Records from a stale version are ignored, not counted as done."""
    if not path.exists():
        return set()
    done: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = EnrichedChunk.model_validate_json(line)
        except ValidationError:
            continue
        if rec.enrichment_version == enrichment_version:
            done.add(rec.chunk_index)
    return done


_MAX_ATTEMPTS = 6
_BACKOFF_SECONDS = 4.0


async def _enrich_one(
    model: ChatModel,
    entry: ManifestEntry,
    chunks: list[Chunk],
    i: int,
    max_tokens: int,
) -> EnrichedChunk | None:
    chunk = chunks[i]
    prev_text = chunks[i - 1].text if i > 0 else ""
    next_text = chunks[i + 1].text if i + 1 < len(chunks) else ""
    messages = build_messages(entry, chunk, prev_text, next_text)

    # Retry transient errors so an unattended multi-hour run never permanently
    # drops a chunk: HTTP 429 (llama-swap backpressure when the burst briefly
    # exceeds the slot count), 503 (cold-load / ttl-eviction reload mid-run), a
    # dropped connection. Backoff is linear and generous — backpressure clears
    # as in-flight calls drain, a model reload is ~tens of seconds. Capped so a
    # genuinely wedged server still fails the chunk eventually (caller logs it).
    result = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            result = await model.complete(messages, response_format=ENRICH_RESPONSE_FORMAT)
            break
        except Exception:
            if attempt + 1 == _MAX_ATTEMPTS:
                return None
            await asyncio.sleep(min(_BACKOFF_SECONDS * (attempt + 1), 20.0))
    if result is None:
        return None

    fields = parse_enrichment(result.text)
    if fields is None:
        return None
    return EnrichedChunk(
        pg_id=chunk.pg_id,
        chunk_index=chunk.chunk_index,
        chunking_version=chunk.chunking_version,
        enrichment_version=ENRICHMENT_VERSION,
        context_note=fields.context_note.strip(),
        entities=fields.entities,
        dates=fields.dates,
    )


class EnrichProgress(BaseModel):
    """Live counters for the CLI / log line."""

    done: int = 0
    failed: int = 0
    skipped: int = 0


async def enrich_work(
    entry: ManifestEntry,
    chunks: list[Chunk],
    model: ChatModel,
    enriched_dir: Path,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    progress: EnrichProgress,
    on_tick: Callable[[EnrichProgress], None] | None,
    limit: int | None,
) -> None:
    """Enrich one work's outstanding chunks, appending each result as it lands.

    `limit` (sample mode) caps total NEW records written across the whole run —
    the corpus driver passes the remaining budget so the spike stops early.
    """
    enriched_dir.mkdir(parents=True, exist_ok=True)
    out_path = enriched_path(enriched_dir, entry.pg_id)
    done = load_done(out_path, ENRICHMENT_VERSION)
    progress.skipped += len(done)

    todo = [i for i, c in enumerate(chunks) if c.chunk_index not in done]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return

    write_lock = asyncio.Lock()
    handle = out_path.open("a", encoding="utf-8")
    try:

        async def worker(i: int) -> None:
            async with semaphore:
                try:
                    record = await _enrich_one(model, entry, chunks, i, max_tokens)
                except Exception:  # one bad call must not kill the overnight batch
                    record = None
            if record is None:
                progress.failed += 1
            else:
                async with write_lock:
                    handle.write(record.model_dump_json() + "\n")
                    handle.flush()
                progress.done += 1
            if on_tick is not None:
                on_tick(progress)

        await asyncio.gather(*(worker(i) for i in todo))
    finally:
        handle.close()


def _read_chunks(path: Path) -> list[Chunk]:
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def enrich_corpus(
    entries: list[ManifestEntry],
    chunks_dir: Path,
    enriched_dir: Path,
    model: ChatModel,
    concurrency: int,
    max_tokens: int,
    sample: int | None = None,
    on_tick: Callable[[EnrichProgress], None] | None = None,
) -> EnrichProgress:
    """Drive the whole corpus: one global semaphore, works in manifest order,
    chunks within a work concurrent. `sample` caps total NEW attempts (spike
    mode). Returns final counters; per-work output is the durable artifact."""
    semaphore = asyncio.Semaphore(concurrency)
    progress = EnrichProgress()
    remaining = sample
    for entry in entries:
        if remaining is not None and remaining <= 0:
            break
        path = chunks_dir / f"pg{entry.pg_id}.jsonl"
        if not path.exists():
            continue
        attempted_before = progress.done + progress.failed
        await enrich_work(
            entry,
            _read_chunks(path),
            model,
            enriched_dir,
            semaphore,
            max_tokens,
            progress,
            on_tick,
            remaining,
        )
        if remaining is not None:
            remaining -= (progress.done + progress.failed) - attempted_before
    return progress
