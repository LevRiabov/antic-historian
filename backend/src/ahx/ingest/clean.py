"""Boilerplate stripping dispatch: raw bytes -> cleaned text, by source format.

The corpus mixes Project Gutenberg texts with non-Gutenberg sources scraped to
`corpus/raw/`: the Internet Classics Archive (hand-transcribed) and archive.org /
Google-Books `_djvu.txt` OCR scans. Each carries different front/back matter, so
`normalize_work` delegates to `clean_raw`, which routes by signature/filename and
applies the right cut.

Unknown formats raise (fail-loud, docs/chunking.md §6) — `normalize_work` catches
the ValueError and surfaces it as a per-book error in the QA report, rather than
letting unrecognized cruft leak into chunks.
"""

import re
from collections.abc import Iterator

from ahx.ingest.gutenberg import has_pg_markers, strip_boilerplate

# --- Internet Classics Archive (hand-transcribed; Tacitus) ---
_ICA_SIGNATURE = "Internet Classics Archive"
_RULE_RE = re.compile(r"^-{20,}$", re.MULTILINE)

# --- archive.org / Google-Books _djvu.txt OCR scans ---
# These have no structural markers, so they're routed by raw filename. Each was
# individually QA'd against the "clean full text or skip" bar before listing here
# (the footnote-heavy scholarship — Rostovtzeff, Bury's Later Roman Empire — was
# rejected: interleaved citation columns can't be separated from flat OCR text).
_DJVU_SOURCES = frozenset(
    {
        "appian-1899-roman-history-v1.txt",
        "appian-1899-roman-history-v2.txt",
        "diodorus-booth-1814-v2-books-11-20.txt",
        "bury-1900-history-of-greece.txt",
    }
)

# Google Books scan preamble (legalese about scanning / public domain). It's a
# contiguous run of blocks at the very top; drop them before prose filtering.
_GOOGLE_PREAMBLE_RE = re.compile(
    r"Google|digiti[sz]|watermark|automated quer|Usage guidelines|digital copy"
    r"|books\.google|public domain|copyright (?:term|expire)|legal copyright",
    re.I,
)
# Bohn publisher-catalogue advertisements bound into the back of the volumes.
_AD_RE = re.compile(r"\b\d+s\.|\b\d+d\.|\bvols?\.|\bnet\.|\bcloth\b|Translated by|Edited by")
# A block opening with a page reference ("P. 823.—", "Pp. 331, 332", "p. 45",
# or with a stray leading OCR colon ": Pp. 331") is a back-matter note/appendix
# entry (e.g. Bury's per-page "Sources"), not narrative.
_PAGE_REF_RE = re.compile(r"^\W*[Pp]p?\.\s*\d")
# Soft hyphen (­) / ¬ (¬) / plain hyphen at a wrapped line end.
_SOFT_HYPHEN_RE = re.compile(r"(\w)[­¬-]\s*\n\s*(\w)")
_MULTISPACE_RE = re.compile(r"\s{2,}")

_MIN_PROSE_CHARS = 180


def clean_raw(raw: str, raw_file: str | None = None) -> str:
    """Detect the source format and return its body text, boilerplate removed."""
    text = raw.replace("\r\n", "\n")
    if has_pg_markers(text):
        return strip_boilerplate(text)
    if _ICA_SIGNATURE in text[:500]:
        return strip_ica(text)
    if raw_file in _DJVU_SOURCES:
        return strip_djvu(text)
    raise ValueError("unrecognized source format (no Gutenberg/ICA markers, not a known djvu scan)")


def strip_ica(text: str) -> str:
    """Internet Classics Archive: keep the region between the first and last
    dash rule (drops header + copyright footer), then drop the internal rule
    lines (book delimiters) so they don't survive as dash-only paragraphs."""
    text = text.replace("\r\n", "\n")
    rules = list(_RULE_RE.finditer(text))
    if len(rules) < 2:
        raise ValueError("ICA format: expected at least two '----' rule lines")
    body = text[rules[0].end() : rules[-1].start()]
    body = _RULE_RE.sub("", body)
    return body.strip()


def _blocks(text: str) -> Iterator[str]:
    """Blank-line-separated blocks, line-unwrapped and whitespace-collapsed."""
    for raw_block in re.split(r"\n\s*\n", text):
        joined = " ".join(line.strip() for line in raw_block.splitlines())
        joined = _MULTISPACE_RE.sub(" ", joined).strip()
        if joined:
            yield joined


def _digit_ratio(block: str) -> float:
    non_space = [c for c in block if not c.isspace()]
    return sum(c.isdigit() for c in non_space) / len(non_space) if non_space else 0.0


def _is_prose(block: str) -> bool:
    """A narrative paragraph, not scan furniture. Furniture (title page, table
    of contents, index, running heads, page numbers, catalogue ads) is short,
    capital-heavy, digit-heavy, or advertising; narrative is long and
    lowercase-dense with sentence punctuation."""
    if len(block) < _MIN_PROSE_CHARS:
        return False
    if _PAGE_REF_RE.match(block):  # back-matter note/appendix entry
        return False
    letters = sum(c.isalpha() for c in block)
    lower = sum(c.islower() for c in block)
    if letters < 0.6 * len(block) or lower < 0.7 * letters:
        return False
    if _digit_ratio(block) > 0.05:  # page-number runs: index, table of contents
        return False
    if len(_AD_RE.findall(block)) >= 2:  # publisher-catalogue advertisements
        return False
    return block.count(".") + block.count(",") >= 3


def strip_djvu(text: str) -> str:
    """archive.org / Google _djvu.txt OCR: rejoin soft hyphens, drop the Google
    preamble, then keep only prose paragraphs. The narrative body is long
    lowercase-dense prose; everything else the scan contributes (front/back
    matter, running heads, page numbers, ads) is filtered out by `_is_prose`."""
    text = _SOFT_HYPHEN_RE.sub(r"\1\2", text.replace("\r\n", "\n"))
    blocks = list(_blocks(text))

    # The Google preamble is a run of legalese blocks at the very top, but a few
    # of them carry no Google keyword (e.g. "Marks, notations ... marginalia"),
    # so a contiguous match stops short. Instead drop everything up to the LAST
    # Google-signal block within the head window — the preamble is contiguous, so
    # this can't reach into the narrative (which begins well after block 18).
    last_google = -1
    for index, block in enumerate(blocks[:18]):
        if _GOOGLE_PREAMBLE_RE.search(block):
            last_google = index

    prose = [b for b in blocks[last_google + 1 :] if _is_prose(b)]
    if not prose:
        raise ValueError("djvu cleaner produced no prose blocks — check the source")
    return "\n\n".join(prose)
