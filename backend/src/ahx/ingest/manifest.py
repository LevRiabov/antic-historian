"""Parser for the corpus manifest (corpus/ai_historian_corpus_eu_pd.txt).

Format: pipe-delimited lines, `#` lines are comments.
Columns: id | category | author | title | translator | pd_basis | txt_url | landing_url
         [ | raw_file ]

The optional 9th column is an explicit raw filename, for non-Gutenberg sources
(synthetic ids >= 900000) whose scraped files don't follow the `pg<id>.txt`
convention. Gutenberg lines keep 8 columns and derive the filename from the id.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

_FIELDS_WITHOUT_RAW = 8
_FIELDS_WITH_RAW = 9


class ManifestEntry(BaseModel):
    """One source work. pydantic validates & coerces field types (≈ zod schema)."""

    pg_id: int
    category: Literal["primary", "scholarship"]
    author: str
    title: str
    translator: str
    pd_basis: str
    txt_url: str
    landing_url: str
    raw_file: str | None = None  # explicit override for non-Gutenberg sources

    @property
    def raw_filename(self) -> str:
        return self.raw_file or f"pg{self.pg_id}.txt"

    @property
    def normalized_filename(self) -> str:
        return f"pg{self.pg_id}.json"


def parse_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) not in (_FIELDS_WITHOUT_RAW, _FIELDS_WITH_RAW):
            raise ValueError(
                f"{path.name}:{line_no}: expected {_FIELDS_WITHOUT_RAW} or "
                f"{_FIELDS_WITH_RAW} fields, got {len(parts)}: {line[:80]!r}"
            )
        entries.append(
            ManifestEntry.model_validate(
                {
                    "pg_id": parts[0],
                    "category": parts[1],
                    "author": parts[2],
                    "title": parts[3],
                    "translator": parts[4],
                    "pd_basis": parts[5],
                    "txt_url": parts[6],
                    "landing_url": parts[7],
                    "raw_file": parts[8] if len(parts) == _FIELDS_WITH_RAW else None,
                }
            )
        )
    return entries
