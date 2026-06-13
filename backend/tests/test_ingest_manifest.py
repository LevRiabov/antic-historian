from pathlib import Path

import pytest

from ahx.ingest.manifest import parse_manifest

REPO_MANIFEST = Path(__file__).resolve().parents[2] / "corpus" / "ai_historian_corpus_eu_pd.txt"


def test_parses_real_manifest() -> None:
    """Structural invariants only — the manifest grows; tests shouldn't pin its size."""
    entries = parse_manifest(REPO_MANIFEST)
    assert len(entries) >= 16
    ids = [e.pg_id for e in entries]
    assert len(ids) == len(set(ids)), "duplicate pg_ids in manifest"
    # Gutenberg works (id < 900000) carry gutenberg.org URLs and derive their
    # raw filename from the id; synthetic non-Gutenberg ids may not.
    assert all(
        e.txt_url.startswith("https://www.gutenberg.org/") for e in entries if e.pg_id < 900000
    )
    assert {e.category for e in entries} == {"primary", "scholarship"}

    herodotus = next(e for e in entries if e.pg_id == 2707)
    assert herodotus.author == "Herodotus"
    assert herodotus.raw_filename == "pg2707.txt"


def test_rejects_malformed_line(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.txt"
    bad.write_text("123|primary|only|five|fields\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 8 or 9 fields"):
        parse_manifest(bad)


def test_skips_comments_and_blanks(tmp_path: Path) -> None:
    content = "# comment line\n\n1|primary|A|T|Tr|basis|http://x/t.txt|http://x\n"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(content, encoding="utf-8")
    entries = parse_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].pg_id == 1


def test_explicit_raw_file_override(tmp_path: Path) -> None:
    """A 9th field overrides the derived pg<id>.txt filename (non-Gutenberg sources)."""
    content = (
        "8|primary|A|T|Tr|basis|http://x/t.txt|http://x\n"
        "900001|primary|Tacitus|The Annals|Church|basis|http://x/t.txt|http://x|tacitus.txt\n"
    )
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(content, encoding="utf-8")
    derived, override = parse_manifest(manifest)
    assert derived.raw_file is None
    assert derived.raw_filename == "pg8.txt"
    assert override.raw_file == "tacitus.txt"
    assert override.raw_filename == "tacitus.txt"
    assert override.normalized_filename == "pg900001.json"  # still keyed on the id


def test_rejects_too_many_fields(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.txt"
    bad.write_text(
        "1|primary|A|T|Tr|basis|http://x/t.txt|http://x|raw.txt|extra\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected 8 or 9 fields"):
        parse_manifest(bad)
