from ahx.ingest.parsers import FRONT_MATTER, parse_classical, parse_structure

CLASSICAL_PARAGRAPHS = [
    "ANABASIS",
    "Preface text before any chapter.",
    "BOOK I",
    "I. Darius and Parysatis had two sons.",
    "More of chapter one.",
    "II. Now Cyrus gathered an army.",
    "BOOK II",
    "I. The second book begins.",
]


def test_parse_classical_builds_locators() -> None:
    divisions = parse_classical(CLASSICAL_PARAGRAPHS)
    # Front matter under the ANABASIS heading, then chapters under books.
    locators = [d.locator for d in divisions]
    assert ["ANABASIS"] in locators
    assert ["BOOK I", "I"] in locators
    assert ["BOOK I", "II"] in locators
    assert ["BOOK II", "I"] in locators

    book1_ch1 = next(d for d in divisions if d.locator == ["BOOK I", "I"])
    assert book1_ch1.paragraphs == [
        "Darius and Parysatis had two sons.",
        "More of chapter one.",
    ]


def test_parse_classical_front_matter_without_heading() -> None:
    divisions = parse_classical(["intro before everything", "I. chapter text"])
    assert divisions[0].locator == [FRONT_MATTER]
    assert divisions[0].paragraphs == ["intro before everything"]


def test_parse_structure_falls_back_to_flat() -> None:
    parser, divisions = parse_structure(["just prose"] * 50)
    assert parser == "flat"
    assert len(divisions) == 1
    assert divisions[0].locator == ["full-text"]
    assert len(divisions[0].paragraphs) == 50


def test_parse_structure_detects_classical() -> None:
    paragraphs = [
        f"{numeral}. chapter text here"
        for numeral in [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
        ]
    ]
    parser, divisions = parse_structure(paragraphs)
    assert parser == "classical"
    assert len(divisions) == 11
