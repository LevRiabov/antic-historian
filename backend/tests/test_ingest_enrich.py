"""Enrichment pass tests — no real LLM: a fake ChatModel returns canned JSON,
so prompt assembly, tolerant parsing, version-filtered resume, and the
crash-safe append loop all run in CI.
"""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from ahx.ingest.chunker import CHUNKING_VERSION, Chunk
from ahx.ingest.enrich import (
    ENRICH_RESPONSE_FORMAT,
    ENRICHMENT_VERSION,
    EnrichedChunk,
    build_messages,
    enrich_corpus,
    enriched_path,
    heading_path,
    load_done,
    parse_enrichment,
    retrieval_representation,
)
from ahx.ingest.manifest import ManifestEntry
from ahx.llm import ChatMessage, ChatResult, StreamEvent

VALID_REPLY = (
    '{"context_note":"In Caesar\'s Gallic War, Caesar besieges Vercingetorix '
    'at Alesia.","entities":["Caesar","Vercingetorix","Alesia"],"dates":["52 BC"]}'
)


class FakeEnrichModel:
    """Records calls; returns a fixed reply (or a per-call sequence)."""

    def __init__(self, reply: str = VALID_REPLY) -> None:
        self.reply = reply
        self.calls = 0
        self.last_response_format: dict[str, object] | None = None

    @property
    def model_name(self) -> str:
        return "fake-enrich"

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError  # enrichment uses complete(), never stream()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        response_format: dict[str, object] | None = None,
    ) -> ChatResult:
        self.calls += 1
        self.last_response_format = response_format
        return ChatResult(text=self.reply, usage=None)


def _entry() -> ManifestEntry:
    return ManifestEntry(
        pg_id=1,
        category="primary",
        author="Caesar",
        title="The Gallic War",
        translator="McDevitte",
        pd_basis="EU-PD: author d. >70y",
        txt_url="https://example.org/pg1.txt",
        landing_url="https://example.org/pg1",
    )


def _chunk(index: int, text: str = "He marched on the city.") -> Chunk:
    return Chunk(
        pg_id=1,
        chunk_index=index,
        chunking_version=CHUNKING_VERSION,
        division_index=0,
        locator=["BOOK VII", "LXVIII"],
        heading=None,
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=6,
    )


# --- pure helpers ---


def test_heading_path_assembles_author_title_locator() -> None:
    path = heading_path("Caesar", "The Gallic War", ["BOOK VII", "LXVIII"], None)
    assert path == "Caesar, The Gallic War > BOOK VII > LXVIII"


def test_heading_path_appends_distinct_heading() -> None:
    path = heading_path("Livy", "History", ["BOOK I"], "The founding of Rome")
    assert path.endswith("(The founding of Rome)")


def test_retrieval_representation_orders_note_heading_text() -> None:
    rep = retrieval_representation("note here", "Author, Title > BOOK I", "the passage")
    assert rep == "note here\nAuthor, Title > BOOK I\n\nthe passage"


def test_build_messages_includes_chunk_and_neighbors() -> None:
    messages = build_messages(_entry(), _chunk(1), "previous text", "following text")
    assert messages[0].role == "system"
    user = messages[1].content
    assert ">>> PASSAGE TO SITUATE <<<" in user
    assert "He marched on the city." in user
    assert "previous text" in user
    assert "following text" in user
    assert "Caesar, The Gallic War" in user


def test_build_messages_marks_work_edges() -> None:
    user = build_messages(_entry(), _chunk(0), "", "")[1].content
    assert "(start of work)" in user
    assert "(end of work)" in user


# --- parsing ---


def test_parse_enrichment_valid_json() -> None:
    fields = parse_enrichment(VALID_REPLY)
    assert fields is not None
    assert fields.context_note.startswith("In Caesar")
    assert "Vercingetorix" in fields.entities
    assert fields.dates == ["52 BC"]


def test_parse_enrichment_substring_fallback() -> None:
    messy = "Here is the JSON:\n" + VALID_REPLY + "\nHope that helps!"
    fields = parse_enrichment(messy)
    assert fields is not None
    assert "Alesia" in fields.entities


def test_parse_enrichment_garbage_returns_none() -> None:
    assert parse_enrichment("no json at all") is None
    assert parse_enrichment('{"context_note": "missing other fields"}') is None


# --- resume bookkeeping ---


def test_load_done_filters_by_version(tmp_path: Path) -> None:
    path = tmp_path / "pg1.jsonl"
    fresh = EnrichedChunk(
        pg_id=1,
        chunk_index=0,
        chunking_version=CHUNKING_VERSION,
        enrichment_version=ENRICHMENT_VERSION,
        context_note="x",
        entities=[],
        dates=[],
    )
    stale = fresh.model_copy(update={"chunk_index": 1, "enrichment_version": "enrich-v0"})
    path.write_text(fresh.model_dump_json() + "\n" + stale.model_dump_json() + "\n", "utf-8")
    done = load_done(path, ENRICHMENT_VERSION)
    assert done == {0}  # the v0 record is not counted as done


# --- end-to-end driver ---


def _write_chunks(chunks_dir: Path, pg_id: int, chunks: list[Chunk]) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / f"pg{pg_id}.jsonl").write_text(
        "\n".join(c.model_dump_json() for c in chunks) + "\n", "utf-8"
    )


async def test_enrich_corpus_writes_and_is_resumable(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    enriched_dir = tmp_path / "enriched"
    _write_chunks(chunks_dir, 1, [_chunk(0), _chunk(1), _chunk(2)])
    model = FakeEnrichModel()

    first = await enrich_corpus(
        [_entry()], chunks_dir, enriched_dir, model, concurrency=2, max_tokens=256
    )
    assert first.done == 3
    assert model.calls == 3
    assert model.last_response_format == ENRICH_RESPONSE_FORMAT  # grammar-constrained

    records = enriched_path(enriched_dir, 1).read_text("utf-8").strip().splitlines()
    assert len(records) == 3
    assert {EnrichedChunk.model_validate_json(r).chunk_index for r in records} == {0, 1, 2}

    # Second run: everything already cached -> zero new LLM calls.
    second = await enrich_corpus(
        [_entry()], chunks_dir, enriched_dir, model, concurrency=2, max_tokens=256
    )
    assert second.done == 0
    assert second.skipped == 3
    assert model.calls == 3  # unchanged


async def test_enrich_corpus_sample_caps_attempts(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    enriched_dir = tmp_path / "enriched"
    _write_chunks(chunks_dir, 1, [_chunk(i) for i in range(10)])
    model = FakeEnrichModel()

    progress = await enrich_corpus(
        [_entry()], chunks_dir, enriched_dir, model, concurrency=4, max_tokens=256, sample=4
    )
    assert progress.done == 4
    assert model.calls == 4


async def test_enrich_corpus_records_failures(tmp_path: Path) -> None:
    chunks_dir = tmp_path / "chunks"
    enriched_dir = tmp_path / "enriched"
    _write_chunks(chunks_dir, 1, [_chunk(0), _chunk(1)])
    model = FakeEnrichModel(reply="not json")

    progress = await enrich_corpus(
        [_entry()], chunks_dir, enriched_dir, model, concurrency=2, max_tokens=256
    )
    assert progress.done == 0
    assert progress.failed == 2
    assert not enriched_path(enriched_dir, 1).read_text("utf-8").strip()  # nothing written
