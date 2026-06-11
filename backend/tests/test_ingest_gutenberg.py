import pytest

from ahx.ingest.gutenberg import split_paragraphs, strip_boilerplate

RAW = """The Project Gutenberg eBook of Test Book
Some legal preamble here.

*** START OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***

BOOK I

I. This is the first sentence,
wrapped across lines by Gutenberg.

A second paragraph.

*** END OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***

More legal text.
"""


def test_strip_boilerplate_cuts_header_and_footer() -> None:
    body = strip_boilerplate(RAW)
    assert body.startswith("BOOK I")
    assert "Project Gutenberg eBook of" not in body
    assert "More legal text" not in body


def test_strip_boilerplate_handles_crlf() -> None:
    body = strip_boilerplate(RAW.replace("\n", "\r\n"))
    assert body.startswith("BOOK I")


def test_strip_boilerplate_raises_without_markers() -> None:
    with pytest.raises(ValueError, match="markers not found"):
        strip_boilerplate("just some text without markers")


def test_split_paragraphs_unwraps_hard_wrapping() -> None:
    paragraphs = split_paragraphs(strip_boilerplate(RAW))
    assert paragraphs == [
        "BOOK I",
        "I. This is the first sentence, wrapped across lines by Gutenberg.",
        "A second paragraph.",
    ]
