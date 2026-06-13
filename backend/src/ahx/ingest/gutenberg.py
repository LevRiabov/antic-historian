"""Project Gutenberg text cleaning: boilerplate stripping + paragraph extraction.

This is the entry half of Layer 1 (docs/chunking.md §1): everything downstream
(structural parsing, chunking, embedding) sees only what this module lets through.
"""

import re

_START_RE = re.compile(
    r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE,
)


def has_pg_markers(text: str) -> bool:
    """True if both Project Gutenberg START and END markers are present."""
    return _START_RE.search(text) is not None and _END_RE.search(text) is not None


def strip_boilerplate(raw: str) -> str:
    """Cut everything outside the `*** START/END OF THE PROJECT GUTENBERG EBOOK` markers."""
    text = raw.replace("\r\n", "\n")
    start = _START_RE.search(text)
    end = _END_RE.search(text)
    if start is None or end is None:
        raise ValueError("Gutenberg START/END markers not found")
    return text[start.end() : end.start()].strip()


def split_paragraphs(text: str) -> list[str]:
    """Blank-line-separated blocks -> unwrapped single-line paragraphs.

    Gutenberg hard-wraps prose at ~70 columns; the line breaks inside a block
    are layout, not meaning, so we unwrap them. The unwrapped paragraphs ARE
    the canonical text — all downstream char offsets refer to it.
    """
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        unwrapped = " ".join(line.strip() for line in block.splitlines())
        unwrapped = re.sub(r"\s{2,}", " ", unwrapped).strip()
        if unwrapped:
            paragraphs.append(unwrapped)
    return paragraphs
