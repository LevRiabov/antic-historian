"""Parser for the corpus manifest (corpus/ai_historian_corpus_eu_pd.txt).

Format: pipe-delimited lines, `#` lines are comments.
Columns: id | category | author | title | translator | pd_basis | txt_url | landing_url
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

_EXPECTED_FIELDS = 8


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

    @property
    def raw_filename(self) -> str:
        return f"pg{self.pg_id}.txt"

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
        if len(parts) != _EXPECTED_FIELDS:
            raise ValueError(
                f"{path.name}:{line_no}: expected {_EXPECTED_FIELDS} fields, "
                f"got {len(parts)}: {line[:80]!r}"
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
                }
            )
        )
    return entries
