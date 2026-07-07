# RAG Techniques — Full Menu, Interactions, and Our Combo (Phase 4 map)

> **Status:** the technique catalogue for the Phase 4 ablations.
> **Written:** 2026-06-10. Evidence tags: **[proven]** / **[rejected]** = measured in rag-historian (4 books / 950 chunks / 50-question golden set) — scale-sensitive findings get re-tested; **[lit]** = literature claims, unmeasured by us.
> Costs assume our shape: ~35k chunks, ~15M token corpus, one-time local-GPU ingest, low query volume, CPU/API serving.

---

## 1. The mental model

Every RAG technique lives at one of five stages. Techniques at the same stage usually **compete**; techniques at different stages usually **compose**:

```
A. INGEST (document-side, one-time $)  →  B. QUERY TRANSFORM (per-query $)
→  C. RETRIEVAL CORE  →  D. POST-RETRIEVAL  →  E. GENERATION/ORCHESTRATION
```

Two cost ledgers, and they are not symmetric:
- **Ingest-time cost** — paid once, on our GPU, offline. Cheap for us. This is where rag-historian's biggest wins lived ("invest in the document side").
- **Query-time cost** — paid on every request, in latency and tokens, forever. Expensive for us. Query-side cleverness must clear a higher bar.

---

## 2. Stage A — Ingest / document-side

| Technique | What it does | Ingest cost | Query cost | Verdict for us |
|---|---|---|---|---|
| **Contextual retrieval** **[proven, +16 recall@5]** | LLM writes 1–2 sentences situating each chunk (resolves "he", "the war", names the book/campaign); note is embedded with the chunk | 1 cheap-LLM call/chunk (~35k calls, few $ or free local; prompt-cache the per-document prefix) | **Zero** | **Core. Build first.** Matters *more* at scale (cross-book ambiguity) |
| **Heading-path prefixing** | Prepend "Work › Book › Chapter" to chunk text before embedding | ~Zero | Zero | **Core.** Free contextual-lite, esp. for scholarship books |
| **Metadata enrichment** | Extract entities/dates/places per chunk into filterable fields | 1 LLM call/chunk (can ride the contextual call) | Zero (enables filters) | **Core-cheap.** Same pass as contextual notes; powers routing + source-isolation |
| **Sparse indexing (BM25/FTS)** | Keyword inverted index next to vectors | ~Zero (tsvector/GIN) | ~Zero | **Build it** — feeds hybrid (C) |
| **RAPTOR** | Recursively cluster chunks → LLM-summarize → tree of abstraction levels; retrieve across levels | Heavy: ~N LLM summary calls over clusters, repeated up levels (~1–2× corpus tokens through an LLM) | Low (just more candidates in one index) | **Ablation arm.** Targets *synthesis* — our historically weakest category (18.7% recall@5). Static corpus = build once, perfect fit |
| **GraphRAG (entity graph + community summaries)** | Extract entities/relations per chunk → knowledge graph → community detection → summaries; query via graph traversal or community reports | **Heaviest**: ~2–4× corpus tokens through an LLM + graph infra | Medium (graph query + summary retrieval) | **Stretch ablation.** Entity-rich corpus fits (people/battles/alliances), targets multi-hop + "global" questions. But multi-hop already hit 4.89/5 via the agent — must beat that to earn its cost |
| **Parent-child / small-to-big** **[rejected: completeness 3.22→2.67]** | Embed small chunks, hand the LLM their parent section | Low | Low | Re-test only if eval shows context starvation; prior evidence negative |
| **Multi-vector (ColBERT-style late interaction)** | Token-level vectors per chunk; MaxSim scoring | Moderate embed cost, **~10–50× vector storage** | Higher compute/query | **Skip.** Storage blows the 500MB cloud budget; cross-encoder rerank covers the same precision ground at our k |
| **SPLADE / learned sparse** | Neural term-weighted sparse vectors | Moderate (model pass/chunk) | Low | **Skip v1.** BM25 + dense + rerank covers it; revisit only if hybrid arm disappoints |
| **Embed summaries instead of chunks** | Index LLM summaries, return originals | 1 call/chunk | Zero | Subsumed by contextual retrieval (which keeps original text *and* adds context). Skip |

## 3. Stage B — Query transformations (per-query, latency-positive)

| Technique | What it does | Query cost | Verdict for us |
|---|---|---|---|
| **HyDE** **[rejected: −9.7 recall@5]** | LLM writes a hypothetical answer; embed *that* instead of the query | +1 LLM call (~0.5–2s) | **Skip.** *Replaces* the query → discards discriminative terms. Designed for embedder-query mismatch we don't have. Prior result decisive |
| **Multi-query expansion** **[proven but marginal: +2.1 recall@5 at n=5]** | LLM generates query paraphrases; union results | +1 LLM call + n× search (~1s) | Opt-in flag, off by default (as before). Agent loop does this organically anyway |
| **Query decomposition** | Split multi-part question into sub-queries | +1 LLM call + n× search | **Skip as standalone** — the agent loop (E) subsumes it dynamically |
| **Step-back prompting** | Generalize the question, retrieve for the abstraction | +1 LLM call | Skip; RAPTOR addresses the same need at ingest time, without per-query cost |
| **Query routing / classification** | Classify query (literal vs synthesis vs OOS) → pick strategy/model | +1 cheap-LLM call (~200ms) or embedding heuristic | **Phase 6 arm.** Prior router hit 74% with dangerous misroutes on contradictions — needs a better classifier or coarser routes to be safe |
| **Self-query / filter extraction** | Parse "according to Plutarch…" into metadata filters | +1 cheap-LLM call | **Worth an arm** — our metadata is rich (author/tier/locator) and the agent's `search_within_source` tool is the manual version |
| Spelling/archaic-term normalization | Map "Pompey the Great"/variant spellings | ~Zero (dictionary) | Cheap nicety; corpus uses Victorian spellings — handle in eval design first |

## 4. Stage C — Retrieval core

| Technique | What it does | Query cost | Verdict for us |
|---|---|---|---|
| **Dense top-k** | Embed query → ANN search | ~ms | Baseline, always on |
| **Hybrid dense+BM25 with RRF fusion** **[rejected at 950 chunks — re-test at 35k]** | Run both, fuse ranks | ~2× ms-scale searches | **Likely promotion to core.** Rare proper nouns (Vercingetorix, Pharsalus) are exactly where dense misses; bigger heterogeneous corpus amplifies it. Prior rejection was "subsumed by reranker at k=5" — at 35k chunks the candidate *pool* (top-50) needs hybrid even if final top-5 ranking comes from the reranker |
| **Metadata-filtered search** | Vector search within `source=X` / `tier=primary` | ~ms | **Core** — powers source-isolation (the contradiction lever) |
| **MMR (diversity)** | Penalize near-duplicate results | ~ms | Narrow use: multi-source synthesis queries where top-k is one book's near-dupes. Post-rerank only, never before |
| **Iterative/recursive retrieval** | Retrieve → read → retrieve again | Multiplied | This *is* the agent loop (E); don't build twice |

## 5. Stage D — Post-retrieval

| Technique | What it does | Query cost | Verdict for us |
|---|---|---|---|
| **Cross-encoder reranking** **[proven]** | Rescore top-50 → top-5..8 with a cross-encoder (BGE-reranker class, or hosted: Cohere/Voyage/Jina rerank APIs) | +100–500ms (CPU small model / API) | **Core.** Precision engine of the whole pipeline. ⚠ Serving: needs same treatment as embedder (small CPU model or API — rerankers have free/cheap hosted tiers; ablate like D2) |
| **Rerank-on-context alignment** **[proven: bare-text rerank UNDID contextual gains]** | Reranker scores the *contextualized* chunk text, not bare text | Zero extra | **Architectural law:** retrieval and rerank must share representation |
| **Lost-in-the-middle reordering** | Put best chunks at prompt start/end | Zero | Free; one-line. Do it, measure nothing |
| **Deduplication** | Drop near-identical passages (overlap artifacts, parallel translations) | ~Zero | Do it; matters once multiple editions/translations of one work coexist |
| **Contextual compression / summarize-before-stuff** | LLM compresses retrieved chunks | +1 LLM call, real latency | **Skip.** Token prices + prompt caching make raw chunks cheap; compression adds latency and a failure mode (summary drops the cited fact). Citations need verbatim text |
| **CRAG (corrective RAG)** | Grade retrieval quality; on fail → fallback (web search / refuse) | +1 cheap-LLM call on suspicion | Partial adopt: the *grading* idea feeds honest refusal ("retrieval found nothing relevant") without the web-search fallback (closed corpus is the product) |

## 6. Stage E — Generation & orchestration

| Technique | What it does | Query cost | Verdict for us |
|---|---|---|---|
| **Single-shot stuffing** | Retrieve once → generate | 1 LLM call | Baseline; stays as the cheap path (and a routing target) |
| **Agentic RAG (tool loop)** **[proven: +0.64 completeness; +0.78 synthesis, +0.89 contradiction]** | LLM iteratively searches/reads/decides with tools; source-isolation, read-before-cite, abstention, forced-finalize | 5–10× tokens, 20–40s | **Core for hard questions.** The differentiating layer; framework-native this time (D1) |
| **Self-RAG / reflection** | Model critiques its own retrieval/answer mid-generation | Extra tokens | Skip as a named technique — agent loop + judge evals capture the value without specialty fine-tunes |
| **FLARE (anticipatory retrieval)** | Retrieve when generation confidence drops | Complex | Skip; agent loop subsumes |
| **Structured citations + streaming** | Citations as structured data attached to streamed answer | ~Zero | **Core** (chosen in plan); engineering not retrieval |
| **Long-context stuffing (anti-RAG)** | Skip retrieval; stuff whole book(s) into a 1M-ctx model | Huge $/query, no citations granularity | Skip as product; **interesting as one eval row** in the case study ("RAG vs long-context: cost ×N, quality ±?") |

---

## 7. Interactions & contradictions (the part nobody documents)

1. **Reranker vs hybrid — partially redundant, scale decides.** Both fix dense-retrieval ranking errors. At 950 chunks the reranker fully subsumed hybrid **[proven]**. At 35k chunks, the reranker only sees what's *in* the candidate pool — hybrid's job shifts from "fix ranking" to "widen the pool". Expectation: hybrid top-50 → rerank → top-5 wins; measure exactly this.
2. **Contextual retrieval × reranking — compose ONLY if aligned** **[proven]**: rerank the contextualized text or the reranker fights the retriever (47.9% vs 51.6%).
3. **HyDE × everything — anti-synergy.** HyDE replaces the query; stacked with multi-query or rerank it just feeds them worse input **[rejected]**.
4. **RAPTOR vs agent loop — same target (synthesis), opposite ledgers.** RAPTOR pre-computes synthesis at ingest (static, free per-query); the agent synthesizes at query time (dynamic, 5–10× tokens). Both may be redundant: if RAPTOR lifts single-shot synthesis enough, the router can send fewer queries down the expensive agent path. Test RAPTOR *on single-shot first* — that's where its marginal value is visible.
5. **RAPTOR vs GraphRAG — competing global structures.** Thematic abstraction tree vs entity-relation graph. Building both doubles the heaviest ingest costs for overlapping query coverage. Sequence: RAPTOR first (cheaper, targets our measured weakness); GraphRAG only if multi-hop/contradiction evals show headroom the agent isn't closing.
6. **Query transforms × semantic cache (Phase 6) — friction.** Rewriting queries before the cache lookup fragments cache keys; cache on the *raw* query, transform after a miss.
7. **Compression vs prompt caching — caching won.** Cached input tokens are ~10× cheaper; compression burns latency to save what caching already saved, and risks deleting the fact you cite.
8. **MMR vs reranking — opposite objectives.** Rerankers maximize relevance (happily returning 5 near-duplicates); MMR trades relevance for diversity. Never MMR before rerank; optionally after, for synthesis-class queries only.
9. **Agent loop subsumes query decomposition, iterative retrieval, FLARE, most of multi-query.** Don't implement static versions of behaviors the agent does adaptively — measure the agent against the static technique instead.
10. **Filters × ANN recall.** Heavily-filtered vector search (one small source) can degrade HNSW recall or fall back to slow scans depending on engine/plan — at our scale exact scan within a filter is fine; just don't blindly trust default plans (pgvector + `WHERE` needs a look at `EXPLAIN`).

---

## 8. The combo for this project

**Production pipeline (target state):**

```
query ──▶ semantic cache (raw query) ──▶ [router: easy/OOS → single-shot · hard → agent]
   single-shot path:  hybrid top-50 (dense-on-contextualized + BM25, RRF, metadata filters)
                      ──▶ cross-encoder rerank (contextualized text) ──▶ top 5–8
                      ──▶ generate with structured streamed citations
   agent path:        same retrieval stack exposed as tools
                      (search_corpus · search_within_source · read_chunk · finalize)
RAPTOR summary nodes live in the same index, retrievable by both paths.
```

**Build/ablate order (each step measured against the previous on the golden set):**

1. **Baseline**: dense top-k → single-shot, streaming citations. *(Phase 3)*
2. **+ Contextual notes & heading prefixes + metadata enrichment** — one combined ingest pass. Expected: the big lift (was +16 recall@5).
3. **+ Cross-encoder rerank** (aligned on contextualized text) + lost-in-middle ordering + dedup.
4. **+ Hybrid BM25/RRF** before rerank — the headline *re-test at scale*.
5. **+ Agent loop** on the hard path. Expected: synthesis/contradiction lift (was +0.64 completeness).
6. **+ RAPTOR** arm (synthesis-targeted; evaluate on single-shot first, then with agent).
7. **Optional arms, in order of expected ROI:** self-query filter extraction → router (Phase 6, with cost data) → GraphRAG (only if multi-hop/contradiction headroom remains) → SPLADE/ColBERT (only if hybrid disappoints — unlikely).
8. **Case-study row:** long-context stuffing comparison (cost vs quality, one table row of honest engineering).

**Standing skips (with reasons on file):** HyDE (−9.7, wrong tool for single-domain corpus), contextual compression (caching beats it), Self-RAG/FLARE (agent subsumes), embed-summaries (contextual subsumes), parent-child (regressed; revisit only on context-starvation evidence), ColBERT (storage budget).

---

## 9. Performance summary — what each core piece costs us

| Stage | Addition | One-time ingest | Per-query latency | Per-query $ |
|---|---|---|---|---|
| Ingest | Contextual + metadata pass | ~35k cheap-LLM calls (~$2–10 hosted, ~$0 local) + re-embed | — | — |
| Ingest | RAPTOR tree | ~1–2× corpus tokens through LLM (~$5–20 hosted ⚠est.) | — | — |
| Ingest | GraphRAG | ~2–4× corpus tokens (~$15–50 ⚠est.) + graph store | — | — |
| Retrieve | Query embed (CPU) | — | ~20–150ms | $0 |
| Retrieve | Hybrid + RRF | — | +~10–50ms | $0 |
| Rerank | Cross-encoder top-50 (CPU/API) | — | +100–500ms | ~$0–0.001 |
| Generate | Single-shot | — | 1–4s | ~$0.001–0.005 |
| Generate | Agent loop | — | 20–40s | ~$0.02–0.05 (planning est.; measured in prod 2026-06-17: ≈$0.01/query, see eval-log) |

The shape to notice: **everything that made rag-historian good is either free or one-time at ingest** — the per-query bill is dominated by generation, which is exactly what the router + caching (Phase 6) attack.
