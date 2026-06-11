from itertools import pairwise

from ahx.ingest.chunker import canonical_text, chunk_work, token_count
from ahx.ingest.model import Division, NormalizedWork

SENTENCE = "The consul marched his legions across the river and made camp before nightfall."


def make_work(divisions: list[Division]) -> NormalizedWork:
    return NormalizedWork(
        pg_id=999,
        author="Test",
        title="Test Work",
        category="primary",
        translator="T",
        parser="classical",
        divisions=divisions,
    )


def repeated_paragraphs(count: int) -> list[str]:
    return [SENTENCE for _ in range(count)]


def test_offsets_reproduce_text_exactly() -> None:
    work = make_work(
        [
            Division(locator=["BOOK I", "I"], heading="BOOK I", paragraphs=repeated_paragraphs(40)),
            Division(locator=["BOOK I", "II"], heading="BOOK I", paragraphs=repeated_paragraphs(7)),
        ]
    )
    canonical = canonical_text(work)
    chunks = chunk_work(work)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.text == canonical[chunk.char_start : chunk.char_end]


def test_chunks_never_cross_division_walls() -> None:
    work = make_work(
        [
            Division(locator=["BOOK I", "I"], heading=None, paragraphs=repeated_paragraphs(40)),
            Division(locator=["BOOK I", "II"], heading=None, paragraphs=repeated_paragraphs(40)),
        ]
    )
    chunks = chunk_work(work)
    assert {tuple(c.locator) for c in chunks} == {("BOOK I", "I"), ("BOOK I", "II")}
    for chunk in chunks:
        # A chunk from division II must start after every division-I chunk ends.
        if chunk.locator == ["BOOK I", "II"]:
            division_one_end = max(c.char_end for c in chunks if c.locator == ["BOOK I", "I"])
            assert chunk.char_start >= division_one_end


def test_packing_and_overlap() -> None:
    work = make_work([Division(locator=["I"], heading=None, paragraphs=repeated_paragraphs(60))])
    chunks = chunk_work(work, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 500
    for previous, current in pairwise(chunks):
        # Overlap: each chunk starts before the previous one ends...
        assert current.char_start < previous.char_end
        # ...but the loop always advances.
        assert current.char_start > previous.char_start


def test_small_division_becomes_single_chunk() -> None:
    work = make_work(
        [Division(locator=["I"], heading=None, paragraphs=["Short paragraph.", "Another."])]
    )
    chunks = chunk_work(work)
    assert len(chunks) == 1
    assert chunks[0].text == "Short paragraph.\n\nAnother."


def test_giant_paragraph_falls_through_to_sentences() -> None:
    giant = " ".join(SENTENCE for _ in range(80))  # far above 500 tokens as ONE paragraph
    assert token_count(giant) > 500
    work = make_work([Division(locator=["I"], heading=None, paragraphs=[giant])])
    chunks = chunk_work(work)
    assert len(chunks) > 1, "giant paragraph should split into multiple chunks"
    for chunk in chunks:
        assert chunk.token_count <= 500


def test_empty_division_produces_no_chunks() -> None:
    work = make_work([Division(locator=["I"], heading=None, paragraphs=[])])
    assert chunk_work(work) == []
