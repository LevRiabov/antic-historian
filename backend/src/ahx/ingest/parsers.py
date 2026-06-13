"""Structural parsers: cleaned paragraphs -> divisions with canonical locators.

v0 archetypes (docs/chunking.md §1 expects ~3-5 total, refined iteratively):

- `classical`: the numbered-classics shape — ALL-CAPS headings open a part
  ("BOOK I", "C. JULIUS CAESAR"), Roman-numeral markers open a chapter
  ("IV. ...", "LXXXII. ..."). Covers Anabasis/Suetonius-like texts.
- `flat`: fallback — one division holding the whole work. Books landing here
  show up flagged in the QA report and get a real parser later.
"""

import re

from ahx.ingest.model import Division

_ROMAN_CHAPTER_RE = re.compile(r"^([IVXLCDM]{1,8})\.(?:\s+|$)")
_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 .,:;'\"()\[\]—-]{2,79}$")
# Back-of-book index headings ("INDEX", "GENERAL INDEX", "GREEK INDEX", ...).
_INDEX_HEADING_RE = re.compile(r"^(?:GENERAL|GREEK|LATIN|NAME|SUBJECT)?\s*INDEX\b", re.I)

FRONT_MATTER = "front-matter"


def parse_classical(paragraphs: list[str]) -> list[Division]:
    divisions: list[Division] = []
    heading: str | None = None
    chapter: str | None = None
    pending: list[str] = []

    def flush() -> None:
        nonlocal pending
        if pending:
            locator = [part for part in (heading, chapter) if part is not None]
            divisions.append(
                Division(locator=locator or [FRONT_MATTER], heading=heading, paragraphs=pending)
            )
            pending = []

    for paragraph in paragraphs:
        roman = _ROMAN_CHAPTER_RE.match(paragraph)
        if roman is not None:
            flush()
            chapter = roman.group(1)
            rest = paragraph[roman.end() :].strip()
            if rest:
                pending.append(rest)
            continue
        if _CAPS_HEADING_RE.match(paragraph) is not None:
            flush()
            heading = paragraph.rstrip(".").strip()
            chapter = None
            continue
        pending.append(paragraph)
    flush()
    return divisions


def parse_flat(paragraphs: list[str]) -> list[Division]:
    return [Division(locator=["full-text"], heading=None, paragraphs=paragraphs)]


def _drop_index_divisions(divisions: list[Division]) -> list[Division]:
    """Drop back-of-book index divisions. Their alphabetized page-reference
    entries ("Africa, circumnavigation of, iii. 283; ...") are dense keyword +
    page-number noise that outranks narrative on lexical queries — pure
    retrieval distractors (docs/eval-log.md 2026-06-12)."""
    return [d for d in divisions if not (d.heading and _INDEX_HEADING_RE.match(d.heading))]


def parse_structure(paragraphs: list[str]) -> tuple[str, list[Division]]:
    """Pick an archetype by evidence: enough Roman-numeral chapter markers ->
    classical, otherwise flat fallback."""
    chapter_markers = sum(1 for p in paragraphs if _ROMAN_CHAPTER_RE.match(p))
    if chapter_markers >= 10:
        return "classical", _drop_index_divisions(parse_classical(paragraphs))
    return "flat", parse_flat(paragraphs)
