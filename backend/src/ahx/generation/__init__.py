"""Generation layer: prompt assembly, citation contract, ask pipeline.

The citation rule that defines this package (phase-3-plan.md, decision 3):
citations are DERIVED from retrieval metadata — the LLM only places [n]
markers in its text. It can misplace a marker (the faithfulness judge
catches that); it cannot invent a source.
"""
