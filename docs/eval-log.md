# Eval Log — append-only record of measured results

> One entry per significant eval run or finding. Raw run records (per-question
> forensics) live in `backend/evals/runs/*.json`; this log is the human-readable
> narrative over them. This file is case-study source material — keep entries
> honest, dated, and tied to a run record.

---

## 2026-06-11 — Baseline: dense-v1 (Phase 2/3 floor)

**Config:** qwen3-embedding-0.6b (local, provisional pending D2) · chunking structural-v1
(500/50, division walls) · 30,187 chunks / 38 works (~11M tokens) · naive dense top-k,
no rerank, no enrichment · golden set v2.0: 72 questions, 154 gold spans, all human-reviewed.
**Run record:** `backend/evals/runs/2026-06-11T18-23-15Z-dense-v1.json`

| category | n | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---|---|---|---|---|---|
| literal | 11 | 9.1% | 69.7% | 69.7% | 69.7% | 0.447 |
| synonym | 10 | 20.0% | 30.0% | 46.7% | 51.7% | 0.431 |
| multi-hop | 10 | 0.0% | 6.7% | 11.7% | 21.7% | 0.055 |
| synthesis | 10 | 5.0% | 15.8% | 25.8% | 29.8% | 0.362 |
| cross-book | 10 | 14.2% | 29.2% | 29.2% | 52.5% | 0.495 |
| contradiction | 11 | 13.6% | 54.5% | 54.5% | 59.1% | 0.427 |
| **overall** | **62** | **10.3%** | **35.2%** | **40.3%** | **48.0%** | **0.372** |

**Findings:**

1. **Reproducibility across scale: overall recall@5 = 35.2% vs rag-historian's naive
   baseline 35.1%** — same architecture (naive dense, 500-token chunks), measured on a
   corpus ~30× larger (38 works vs 4) with a fresh, independently-authored question set
   and a different embedder. The naive-dense floor appears to be a property of the
   architecture, not the corpus. Strong validation that both eval harnesses measure the
   same thing.
2. **Multi-hop is the catastrophe (6.7% @5, MRR 0.055):** one query embedding can't chase
   two facts — each hop's passage competes with the other's in a single similarity
   ranking. Known fixes, in our Phase 4/5 plan: agent loop (multi-step search), query
   decomposition, RAPTOR.
3. **Cross-book is rerank-bait:** 52.5% @20 vs 29.2% @5 — the right passages reach the
   pool but rank too low. The clearest "known fix" signal in the table (cross-encoder
   rerank of top-50).
4. **Literal plateaus after k=5** (69.7% at @5=@10=@20): misses are vocabulary/entity
   misses, not ranking misses — contextual retrieval's target.
5. **Synonym tax measured:** 30% @5 — modern-English questions vs Victorian translations.
6. **Synthesis confirms the prior** (15.8% @5; was 18.7% on the small corpus) —
   distributed answers stay retrieval-hard at every scale.

**Next levers (Phase 4 order, per docs/rag-techniques.md):** contextual retrieval →
cross-encoder rerank → hybrid BM25/RRF re-test → agent loop (Phase 5) → RAPTOR.
