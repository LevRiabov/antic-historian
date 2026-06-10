"""Corpus ingestion: per-format parsers -> normalized document tree -> chunking -> embedding.

Architecture: docs/chunking.md. Per-book uniqueness lives in parsers; the chunker is uniform.
"""
