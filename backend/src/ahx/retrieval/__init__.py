"""Retrieval stack: dense / hybrid / rerank. One module per technique, ablation-friendly.

Technique catalogue and build order: docs/rag-techniques.md.
Embedding calls MUST go through one module that owns the query/document prefix
policy (parity footguns: docs/embeddings.md §6).
"""
