import pytest

from ahx.ingest.clean import clean_raw, strip_djvu, strip_ica

# A miniature Internet Classics Archive text: provenance/title header, a rule,
# two books each fenced by rules, a trailing rule, then the copyright footer.
ICA = """Provided by The Internet Classics Archive.
See bottom for copyright. Available online at
    http://classics.mit.edu//Tacitus/annals.html

The Annals
By Tacitus

Translated by Alfred John Church and William Jackson Brodribb

----------------------------------------------------------------------

BOOK I

Rome at the beginning was ruled by kings.

----------------------------------------------------------------------

BOOK II

The second book opens here.

----------------------------------------------------------------------

Copyright statement:
The Internet Classics Archive by Daniel C. Stevenson, Web Atomics.
"""

PG = """The Project Gutenberg eBook of Test
*** START OF THE PROJECT GUTENBERG EBOOK TEST ***

Real body text.

*** END OF THE PROJECT GUTENBERG EBOOK TEST ***
trailing legal
"""


def test_strip_ica_keeps_body_drops_header_footer_and_rules() -> None:
    body = strip_ica(ICA)
    assert body.startswith("BOOK I")
    assert "BOOK II" in body
    assert "Internet Classics Archive" not in body  # header gone
    assert "Copyright statement" not in body  # footer gone
    assert "----" not in body  # internal rule lines stripped


def test_strip_ica_raises_without_rules() -> None:
    with pytest.raises(ValueError, match="rule line"):
        strip_ica("Provided by The Internet Classics Archive.\nNo rules here.")


def test_clean_raw_dispatches_gutenberg() -> None:
    assert clean_raw(PG) == "Real body text."


def test_clean_raw_dispatches_ica() -> None:
    assert clean_raw(ICA).startswith("BOOK I")


def test_clean_raw_handles_crlf_ica() -> None:
    assert clean_raw(ICA.replace("\n", "\r\n")).startswith("BOOK I")


def test_clean_raw_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unrecognized source format"):
        clean_raw("just some text with no recognizable markers")


# A miniature archive.org djvu scan: Google preamble, garbled title page, a
# running head + page number, two narrative paragraphs (one with a soft-hyphen
# wrap), a digit-heavy index entry, and a catalogue advertisement.
PARA1 = (
    "The Romans, dividing Italy into many parts, made each of them a province, "
    "and over each they set a governor or a praetor, so that the whole was ruled "
    "with order, and the tribute was gathered without complaint from the cities."
)
DJVU = """This is a digital copy of a book preserved by Google and made available online.

Whether a book is in the public domain may vary from country to country.

THE HISTORICAL LIBRARY

OF DIODORUS

123 RUNNING HEAD OF THE BOOK 45

The Romans, dividing Italy into many parts, made each of them a prov-
ince, and over each they set a governor or a praetor, so that the whole was
ruled with order, and the tribute was gathered without complaint from the cities.

Carthage, 146; Corinth, 146; Numantia, 133; Spain, 205, 206, 209, 210, 211.

New Webster Dictionary. Translated by J. Smith. 5 vols. 3s. 6d. net. cloth.
"""


def test_strip_djvu_keeps_prose_drops_furniture() -> None:
    body = strip_djvu(DJVU)
    assert PARA1 in body  # soft hyphen "prov-\nince" rejoined to "province"
    assert "province" in body
    assert "digital copy" not in body  # Google preamble dropped
    assert "public domain" not in body
    assert "RUNNING HEAD" not in body  # running head + page numbers dropped
    assert "Carthage, 146" not in body  # digit-heavy index entry dropped
    assert "New Webster" not in body  # catalogue advertisement dropped


def test_strip_djvu_drops_page_reference_appendix() -> None:
    appendix = (
        "P. 823, Sect. 5.—Sources: Primary: inscriptions; Aeschines, Against Ctesiphon; "
        "Demosthenes, On the Crown; secondary, the modern histories of the period as cited."
    )
    body = strip_djvu(f"{PARA1}\n\n{appendix}\n")
    assert PARA1 in body
    assert "Sources: Primary" not in body  # back-matter note dropped despite being long prose


def test_strip_djvu_drops_pp_plural_page_reference() -> None:
    # Bury's notes open with "Pp." (plural) as well as "P." — both are back-matter.
    note = (
        ": Pp. 331, 332.—Harbours of the Munychian peninsula: I have followed the current view "
        "as to the identity of the harbours, though the matter is far from certain in the sources."
    )
    body = strip_djvu(f"{PARA1}\n\n{note}\n")
    assert PARA1 in body
    assert "Munychian peninsula" not in body  # leading OCR colon before "Pp." still caught


def test_strip_djvu_raises_when_no_prose() -> None:
    with pytest.raises(ValueError, match="no prose blocks"):
        strip_djvu("INDEX\n\n123\n\nTHE END\n")


def test_clean_raw_routes_djvu_by_filename() -> None:
    body = clean_raw(DJVU, "diodorus-booth-1814-v2-books-11-20.txt")
    assert "province" in body
    assert "digital copy" not in body
    # same content without a known djvu filename is unrecognized
    with pytest.raises(ValueError, match="unrecognized source format"):
        clean_raw(DJVU, "some-unknown-file.txt")
