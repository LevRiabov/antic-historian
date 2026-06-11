"""Layer 2 — the uniform chunker (docs/chunking.md §2).

Packs paragraphs greedily into ~CHUNK_SIZE-token chunks WITHIN a division,
never across (hard walls at the boundaries Layer 1 extracted). Paragraphs
larger than the budget fall through to sentence-level packing. Overlap is
unit-aligned: the next chunk re-includes the previous chunk's trailing units
up to the overlap budget.

Offsets invariant (tested): `chunk.text == canonical_text(work)[char_start:char_end]`.
The canonical text is all paragraphs in document order joined by "\\n\\n" —
golden-set gold spans will refer to exactly these offsets.

Token counts use tiktoken cl100k_base as the budget ruler. It is NOT the
embedder's tokenizer — budgets are approximate by design; the embedder's
real limit (32k) is far above any chunk this module emits.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from pydantic import BaseModel

from ahx.ingest.model import NormalizedWork

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHUNKING_VERSION = "structural-v1"

PARAGRAPH_SEPARATOR = "\n\n"

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z])")


class Chunk(BaseModel):
    pg_id: int
    chunk_index: int
    chunking_version: str
    division_index: int
    locator: list[str]
    heading: str | None
    text: str
    char_start: int
    char_end: int
    token_count: int


@lru_cache
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_encoder().encode(text))


def canonical_text(work: NormalizedWork) -> str:
    return PARAGRAPH_SEPARATOR.join(
        paragraph for division in work.divisions for paragraph in division.paragraphs
    )


@dataclass
class _Unit:
    """A packable piece (paragraph, or sentence run inside an oversized
    paragraph) addressed by canonical-text offsets."""

    start: int
    end: int
    tokens: int


def _sentence_units(paragraph: str, base: int) -> list[_Unit]:
    spans: list[tuple[int, int]] = []
    previous = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(paragraph):
        spans.append((previous, match.start()))
        previous = match.end()
    spans.append((previous, len(paragraph)))
    return [
        _Unit(start=base + s, end=base + e, tokens=token_count(paragraph[s:e]))
        for s, e in spans
        if paragraph[s:e].strip()
    ]


def _pack_units(units: list[_Unit], chunk_size: int, overlap: int) -> list[tuple[int, int, int]]:
    """Greedy packing -> list of (first_unit, last_unit, token_total)."""
    packed: list[tuple[int, int, int]] = []
    i = 0
    while i < len(units):
        j = i
        total = units[i].tokens
        while j + 1 < len(units) and total + units[j + 1].tokens <= chunk_size:
            j += 1
            total += units[j].tokens
        packed.append((i, j, total))
        if j + 1 >= len(units):
            break
        # Overlap: start the next chunk early enough to re-include up to
        # `overlap` tokens of this chunk's tail (always advancing past i).
        k = j + 1
        tail = 0
        while k - 1 > i and tail + units[k - 1].tokens <= overlap:
            k -= 1
            tail += units[k].tokens
        i = k
    return packed


def chunk_work(
    work: NormalizedWork,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
    version: str = CHUNKING_VERSION,
) -> list[Chunk]:
    canonical = canonical_text(work)
    chunks: list[Chunk] = []
    offset = 0
    for division_index, division in enumerate(work.divisions):
        units: list[_Unit] = []
        for paragraph in division.paragraphs:
            start = offset
            offset += len(paragraph) + len(PARAGRAPH_SEPARATOR)
            tokens = token_count(paragraph)
            if tokens > chunk_size:
                units.extend(_sentence_units(paragraph, base=start))
            else:
                units.append(_Unit(start=start, end=start + len(paragraph), tokens=tokens))

        for first, last, total in _pack_units(units, chunk_size, overlap):
            char_start = units[first].start
            char_end = units[last].end
            chunks.append(
                Chunk(
                    pg_id=work.pg_id,
                    chunk_index=len(chunks),
                    chunking_version=version,
                    division_index=division_index,
                    locator=division.locator,
                    heading=division.heading,
                    text=canonical[char_start:char_end],
                    char_start=char_start,
                    char_end=char_end,
                    token_count=total,
                )
            )
    return chunks
