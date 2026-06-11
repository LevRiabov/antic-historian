"""Normalized document model — the hard interface between Layer 1 (parsing)
and Layer 2 (chunking). See docs/chunking.md §1.

A work is a flat list of divisions in reading order; each division carries a
locator path (the canonical citation address, e.g. ["BOOK I", "IV"] for
Anabasis 1.4) and its paragraphs. The chunker packs paragraphs WITHIN a
division and never across divisions.
"""

from typing import Literal

from pydantic import BaseModel


class Division(BaseModel):
    locator: list[str]
    heading: str | None = None
    paragraphs: list[str]

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.paragraphs)


class NormalizedWork(BaseModel):
    pg_id: int
    author: str
    title: str
    category: Literal["primary", "scholarship"]
    translator: str
    parser: str
    divisions: list[Division]

    @property
    def paragraph_count(self) -> int:
        return sum(len(d.paragraphs) for d in self.divisions)

    @property
    def char_count(self) -> int:
        return sum(d.char_count for d in self.divisions)
