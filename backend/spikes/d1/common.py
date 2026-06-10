"""Shared, framework-NEUTRAL pieces of the D1 spike: corpus loading, the
embedding prefix policy, questions, and the answer prompt.

Everything framework-specific stays in the two spike scripts so each framework
is judged on its own ergonomics. This module only guarantees both spikes see
identical inputs (books, chunk params, prefix, prompt, questions).
"""

import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[3] / "corpus" / "raw"

LLAMA_SWAP_BASE = "http://127.0.0.1:8080/v1"
EMBED_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "gemma-12b-16k"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K = 6

# Qwen3-Embedding instruction policy: queries get the instruction, documents do
# NOT (docs/embeddings.md §6 footgun 1). Both spikes must route every query
# embedding through this prefix.
QWEN3_QUERY_PREFIX = (
    "Instruct: Given a search query, retrieve relevant passages "
    "from ancient history texts that answer the query\nQuery:"
)

BOOKS = [
    {
        "file": "pg1170-xenophon-anabasis.txt",
        "author": "Xenophon",
        "title": "Anabasis",
    },
    {
        "file": "pg6400-suetonius-twelve-caesars.txt",
        "author": "Suetonius",
        "title": "The Lives of the Twelve Caesars",
    },
]

QUESTIONS = [
    "How did Julius Caesar die, and what is reported about his final moments?",
    "Why did Cyrus the Younger march against his brother Artaxerxes?",
    "What omens are said to have preceded Caesar's assassination?",
    # Out of scope — the correct behavior is an explicit refusal.
    "What did Napoleon do at the Battle of Austerlitz?",
]

ANSWER_PROMPT = """\
You are a careful research assistant answering questions strictly from the \
provided source passages.

Rules:
- Use ONLY the passages below. Cite every factual claim with the passage \
number in square brackets, e.g. [2].
- If the passages do not contain the information needed, reply exactly: \
"I don't have a source for that." Do not use outside knowledge.

Passages:
{context}

Question: {question}

Answer:"""

_START_RE = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n")
_END_RE = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK")


@dataclass
class Book:
    author: str
    title: str
    text: str


def load_books() -> list[Book]:
    """Load spike books with Gutenberg boilerplate stripped (spike-grade cleaning)."""
    books: list[Book] = []
    for spec in BOOKS:
        raw = (CORPUS_DIR / spec["file"]).read_text(encoding="utf-8")
        start = _START_RE.search(raw)
        end = _END_RE.search(raw)
        body = raw[start.end() : end.start()] if start and end else raw
        books.append(Book(author=spec["author"], title=spec["title"], text=body.strip()))
    return books


def format_context(passages: list[tuple[str, str]]) -> str:
    """passages: list of (source_label, text) -> numbered context block."""
    return "\n\n".join(
        f"[{i}] ({label})\n{text}" for i, (label, text) in enumerate(passages, start=1)
    )
