# Chunking Strategy at Corpus Scale

> **Status:** planning input for Phase 1 (ingestion) and the Phase 4 chunking re-study.
> **Written:** 2026-06-10. Prior evidence: rag-historian Module 6.1 (4 books, 950 chunks) — naive 500/50 section-aware chunking beat finer (300), coarser (chapter-level ~6k), and parent-child variants. That finding may not transfer to dozens of heterogeneous books — re-measure.

---

## 1. The reframe: chunking at scale is a parsing problem

With 4 books you can eyeball every quirk. With dozens you cannot hand-tune per book — but you also must not run a one-size-fits-all splitter over raw Gutenberg text (page headers, translator footnotes, tables of contents, and inconsistent heading conventions would all leak into chunks).

The architecture that scales is **two layers with a hard interface between them:**

```
raw book file ──(per-format parser)──▶ normalized document tree ──(ONE uniform chunker)──▶ chunks
```

- **Layer 1 — normalization (where the per-book uniqueness lives).** Parse each book into a common hierarchical document model: `work → book/part → chapter → section → paragraph`, plus work-level metadata (author, title, tier: primary/secondary, date written, translator, edition). Strip boilerplate, footnotes (see §4), TOCs, page artifacts here — *before* chunking.
- **Layer 2 — chunking (uniform, boring, tested once).** A single structure-aware chunker that operates on the normalized tree and never needs to know which book it's processing.

All per-book effort goes into parser adapters; the chunker stays one well-tested function. In practice Gutenberg-era texts cluster into a handful of structural archetypes (numbered classical books/chapters, letter collections, speech collections, modern monographs with headings) — expect ~3–5 adapters, not ~40.

## 2. The chunker itself (baseline)

Proven shape from rag-historian, kept as the v1 baseline:

- **Recursive structure-aware packing:** split on structural boundaries in priority order (section → paragraph → sentence), then pack units greedily up to the target size. Yes — a 2,000-token chapter becomes ~4 chunks; the packing just never splits mid-sentence and never merges across a section/chapter boundary (hard walls).
- **Target ~500 tokens, ~50-token overlap** (overlap only between chunks within the same section).
- **Every chunk carries:** `char_start`/`char_end` into the normalized text (chunking-invariant gold spans + source viewer), `chunking_version` tag (variants coexist in the store), and the full metadata + locator path (§3).
- Oversized single paragraphs (rare in this corpus) fall through to sentence-packing; never hard-cut mid-word/mid-sentence.

## 3. The corpus's special asset: canonical citation schemes

Classical texts have stable, universal reference systems: *Gallic War* 4.25, Plutarch *Caesar* 32.4, Cicero *Ad Atticum* 7.11. These survive across every edition and translation.

**Capture the canonical locator on every chunk** (`work=BG, book=4, chapter=25`). This buys, almost for free:

1. **Professional citations** — "Caesar, *BG* 4.25" in answers instead of "chunk #4812". Directly serves the buyer-facing citation feature, and historians/reviewers can verify against any edition.
2. **Cross-source alignment** — secondary studies cite primary sources by these same locators ("see *BG* 4.25"); preserved locators make cross-referencing and contradiction features tractable later.
3. **Cheap eval authoring** — golden-set gold spans can be written as locators and resolved to char spans mechanically.

This requires the Layer-1 parsers to recognize numbered book/chapter structure — the main reason normalization deserves real effort.

## 4. Heterogeneity: primary vs secondary sources

| Property | Ancient primary sources | Modern studies |
|---|---|---|
| Structure | Canonical book/chapter/section numbering | Author's headings, chapters, footnotes |
| Prose | Continuous narrative; pronoun-heavy ("he then marched…") | Topic-structured; self-contextualizing headings |
| Chunk risk | Chunks lose antecedents (who is "he"? which river?) | Footnotes/citations polluting chunk text |
| Mitigation | **Contextual retrieval notes** (§5) — disambiguate entities | Strip footnotes to a side-channel at parse time; prepend heading path to chunk text |
| Metadata | tier=primary, vantage/bias note | tier=secondary, publication date |

The `tier` field also powers later features: source-isolated search, primary-vs-secondary weighting, and the contradiction story.

## 5. Enhancements — enter through the Phase 4 ablation door

Ranked by expected ROI given prior evidence:

1. **Contextual retrieval notes** (proven +16 recall@5 at small scale): LLM writes 1–2 sentences situating each chunk in its document ("From Book 4 of Caesar's Gallic War; 'he' = Caesar; describes the first crossing into Britain, 55 BC"), embedded with the chunk. *More* valuable at this scale — cross-book entity ambiguity ("the consul", "the war") grows with corpus size. Cost: one cheap-LLM pass over the corpus at ingest, one-time.
2. **Heading-path prefixing** for secondary sources (near-free: prepend "Work › Chapter › Section" to chunk text before embedding).
3. **Chunk-size re-study** (cheap to run once the eval harness exists: 300/500/800 variants under `chunking_version` tags). Prior winner was 500; verify it holds with modern-prose books in the mix.
4. **Semantic chunking** (embedding-similarity breakpoints): one ablation arm, low expected ROI — published comparisons show small/mixed gains over recursive structural chunking at notably higher ingest complexity. Include only to have the receipt.
5. **Parent-child retrieval**: regressed last time (completeness 3.22 → 2.67); re-test only if eval shows answers starved for context around retrieved chunks.
6. **RAPTOR / hierarchical summaries**: not chunking — a retrieval-architecture experiment over the chunk layer (targets synthesis questions). Phase 4, separate ablation.

## 6. QA at scale (replaces eyeballing every book)

Per-book automated report at ingest: chunk-size histogram, count of boundary violations (chunks crossing sections), % boilerplate stripped, unresolved-locator count, plus N random chunks sampled for human spot-check. A book fails loudly, not silently. Timebox per-book parser fixes; prefer dropping a hopelessly messy edition over hand-patching it — cleaner substitute editions usually exist (multiple translations per work on Gutenberg/Perseus/LacusCurtius).

## 7. Decision summary

- **v1 (Phase 1):** per-format parsers → normalized tree with canonical locators → structure-aware 500/50 packing, hard section walls, `chunking_version=naive-v1`. This unblocks everything downstream.
- **Phase 4:** contextual notes first (highest expected ROI), then chunk-size re-study, then semantic/parent-child arms only as ablation receipts.
- **Non-goals:** per-book hand-tuned chunking; LLM-driven "agentic chunking" at ingest (cost/complexity without evidence); multiple simultaneous granularities in v1 (RAPTOR may add a coarse layer later, measured).
