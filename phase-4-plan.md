# Phase 4 — Advanced retrieval ablations (the heart of the project)

> Plan written 2026-06-12, after the Phase 3 exit (commit `c1b1405`). Scope per
> [project-plan.md](project-plan.md) and the pre-committed build order in
> [docs/rag-techniques.md §8](docs/rag-techniques.md). Every technique enters through the
> ablation door: implement → measure → keep/reject → decision note. Rejections with
> receipts are case-study content.
>
> **Exit criteria:** (1) gate D2 decided (ADR-002); (2) every core arm (embedder,
> contextual, rerank, hybrid) measured against the previous step with a keep/reject note;
> (3) the production single-shot pipeline assembled from winners; (4) the ablation table
> in eval-log.md tells the whole story.

## The baseline we measure against (locked, commit c1b1405)

| tier | metric | value |
|---|---|---|
| retrieval (dense-v1, top-20) | recall@5 / recall@20 / MRR | 35.2% / 48.0% / 0.372 |
| generation (top-5, gemma-12b) | citation recall / precision | 32.0% / 54.1% |
| generation (judge-v2) | faithfulness / completeness | 4.72 / 3.07 |
| behavior | OOS refusal / false refusals | 100% / 25.8% |

Noise floors (documented in eval-log): ±1 question on retrieval metrics;
±0.2 on judge scores. Single-question movements are not findings.

**The three numbers Phase 4 exists to move:** retrieval recall@5 (ceiling for
everything), false refusal rate (24% of in-scope questions die to honest "sources don't
say" — retrieval failures wearing a polite mask), completeness (3.07 — answers are
faithful but thin because the right passages aren't co-located in the prompt).

## Measurement protocol (every arm, no exceptions)

1. `ahx eval run --retriever <label>` — retrieval tier, free, the iteration loop.
2. `ahx eval generate --label <label> --judge` — generation tier, ~$0.03/run.
3. Eval-log entry with keep/reject decision + run-record paths.
4. Retriever labels are append-only versioned: `dense-v1` → `dense-ctx-v1` →
   `rerank-v1` → `hybrid-rerank-v1`. A label change = a measured config change.
5. After ANY embedding runtime/model change: `ahx ingest parity` (rule #3).

## Workstream 4.0 — Gate D2: the embedder ablation (carried over from Phase 2)

Shortlist per [docs/embeddings.md §5](docs/embeddings.md): **voyage-4-nano**
(front-runner: Apache 2.0, CPU-class, shared embedding space with hosted Voyage 4 —
asymmetric-upgrade escape hatch), **qwen3-embedding-0.6b** (incumbent — already proven on
llama-swap, strongest CPU-class raw quality), **gte-modernbert-base** (insurance: 149M,
no prefixes, tiny-container fallback), plus **one hosted ceiling reference**
(voyage-4-lite via the 200M-token free tier) to learn how much quality local leaves on
the table.

Mechanics:

- **Bare chunks, sequentially.** Embed the same structural-v1 chunks per candidate →
  directly comparable to the locked dense-v1 baseline. One candidate at a time:
  re-init schema with the model's dims (768/1024/2048 — `EMBED_DIM` becomes
  config-driven), `ahx ingest load`, parity fixture, `ahx eval run`. Corpus re-embeds
  are minutes on the 5070 Ti — sequential is simpler than multi-column storage.
- **The one embedding module grows per-model prefix policies** (nano:
  query/document prompts; gte: none; qwen3: current instruction) — policy keyed by
  model name, nothing else changes (rule #3).
- **Runtime risk (nano):** community GGUF/ONNX ports are unverified — parity-check
  against the reference implementation (sentence-transformers, GPU) at cosine ≥ 0.999
  before trusting llama-swap serving. If ports fail parity, that's a real ops point
  *for* the incumbent qwen3 (official GGUF, already running).
- **Decision criteria (written before results):** golden-set recall@5/MRR per category
  (primary), CPU query latency on target-host-class hardware, RAM, ops complexity,
  license. Winner → **ADR-002**. Thin margins (≤ noise floor) → incumbent wins by ops.
- After 4.1 lands, re-run the top-2 on *contextualized* text if the margin was thin —
  what we ship is embeddings-of-contextualized-chunks, not bare chunks.

## Workstream 4.1 — Contextual retrieval + heading prefixes + metadata (one ingest pass)

The expected big lever (**[proven +16 recall@5]** at small scale; cross-book ambiguity
should make it matter *more* here — "he marched on the city" needs to know which book,
which war).

> **Build settled 2026-06-13** (corpus is now **46,170 chunks / 62 works**, not 30k):
> `ahx ingest enrich` (module `ahx/ingest/enrich.py`). Three decoupled passes —
> **enrich → disk cache → embed** — so the expensive LLM pass is paid once and never
> repeats on a re-embed. See the enrich-mechanics block below.

- One LLM pass over the 46k chunks producing, in one grammar-constrained JSON reply: a
  1–2 sentence **context note** (situates the chunk: work, campaign, who "he" is) +
  **entities/dates metadata** (JSONB, powers Phase 5 source-isolation + self-query).
  **Heading-path prefix** (`Work > Book > Chapter`) is free string assembly.
- **What gets embedded and stored:** `context_note + heading_path + chunk_text` becomes
  the chunk's *retrieval representation* (`ChunkRow.retrieval_text`) — its own column,
  because the reranker (4.2) must score exactly this text (**alignment law, rule #4**).
  The *generation* prompt and citations continue to show the original `text` + locator.
- **Note-generation model — DECIDED: local gemma-4-12B** (`gemma-12b-enrich` llama-swap
  profile, `-np` parallel slots). Windowed context, NOT whole-document: local 16k ctx
  can't hold a 290k-token book, and the heading path carries cross-book disambiguation
  for free — each call sees work/section headers + the chunk + its immediate neighbors.
  Cost $0, ~1.4 chunks/s warm on the 5070 Ti ⇒ ~9h, run unattended overnight (resumable).
  **Rejected hosted deepseek-v4-flash** (~$5, ~1h): the whole-doc recipe that earned the
  +16 would cost ~$390+ even *with* caching here (each book re-read once per chunk =
  27.8B cache-read tokens), and windowed-local is free; receipt for the case study.
- **Enrich mechanics (durability is the point — 46k chunks is too much to repeat):**
  - **Cached to `corpus/enriched/pgNNNN.jsonl`**, keyed by `enrichment_version`
    (`enrich-v1`). Every later re-embed (D2 follow-ups, dim changes, the 4.2 rerank arm)
    reads this cache; the LLM runs **once per version, ever**.
  - **Resumable + crash-safe:** results appended+flushed per chunk; a re-run skips chunks
    already done at the current version. Writes only to disk (no DB) so a crash/power-cut
    costs only the in-flight calls.
  - **Robust unattended:** grammar-constrained JSON (no malformed-output failure mode);
    bounded arrays + note length (no max-token truncation); retry-with-backoff on
    transient 503s (model cold-load / ttl reload mid-run).
- Schema: new nullable columns on `ChunkRow` (`context_note`, `retrieval_text`,
  `enrichment_version`, `entities`, `dates`); rebuild via `db reset-chunks` + full reload
  (the loader joins the enriched cache at embed time; bare-text fallback = dense-v1 when a
  chunk isn't enriched). Alembic still deferred — corpus regenerates from files.
- Measure: retrieval (`dense-ctx-v1`) vs the **55.0% recall@5** floor + generation tier vs
  **4.45 completeness**. Watch literal (vocabulary misses, e.g. lit-004) and cross-book.

## Workstream 4.2 — Cross-encoder rerank (representation-aligned)

The precision engine (**[proven]**), with the hard-won law baked in: **the reranker
scores the contextualized text** — bare-text rerank UNDID contextual gains in
rag-historian (47.9% vs 51.6%).

- Pipeline shape: dense top-50 → rerank → top-5..8. Implemented as a new `Retriever`
  (the 3.3 protocol pays off: the ask pipeline doesn't change).
- **Local candidates already sitting on llama-swap:** `qwen3-reranker-0.6b`,
  `bge-reranker-v2-m3`. Hosted candidates (Voyage/Cohere/Jina rerank free tiers —
  verify prices/limits at implementation, same D2-style serving constraint).
- Free riders in the same arm, per the menu: **lost-in-the-middle reordering**
  (one line, unmeasured by design) and **near-duplicate dedup** (matters: 500/50
  overlap means adjacent chunks both covering one span).
- Measure: recall@5 of the reranked list vs `dense-ctx-v1`; cross-book is the
  rerank-bait category (52.5% @20 vs 29.2% @5 at baseline — the pool has the answers,
  the ranking doesn't).

## Workstream 4.3 — Hybrid BM25 + RRF (the headline re-test at scale)

Rejected at 950 chunks (**reranker subsumed it**), expected to flip at 30k: the
reranker only sees what's in the pool, and hybrid's job here is *widening the pool* —
rare proper nouns (Vercingetorix, Pharsalus) are exactly where dense misses.

- Postgres FTS (`tsvector` + GIN) next to the vectors — one store, joinable (D3
  default). RRF fusion of dense top-50 + BM25 top-50 → rerank → top-5.
- **New metric lens this arm needs: pool recall@50** (did the answer reach the
  reranker at all?) — distinguishes "hybrid widened the pool" from "rerank fixed the
  order". Small harness addition, measured like everything else.
- **Gate D3 check rides here:** if DIY Postgres FTS+RRF clearly underperforms or
  fights the planner (watch `EXPLAIN` on filtered ANN — interaction #10), the Qdrant
  challenger gets its trial. Default expectation: Postgres holds.
- Victorian-spelling caveat: BM25 matches exact tokens; archaic spellings may need a
  small normalization dictionary — only if the eval shows it (synonym category is the
  canary).

## Workstream 4.4 — Conditional arms (each needs an evidence trigger, not enthusiasm)

| Arm | Trigger | Notes |
|---|---|---|
| **RAPTOR** | synthesis/completeness still weak after 4.1–4.3 | Targets synthesis (15.8% @5). Heavy ingest (~1–2× corpus tokens through an LLM; local gemma or ~$5–20 hosted ⚠est.). Evaluate on single-shot first (interaction #4: it competes with the Phase 5 agent for the same queries) |
| **Chunking re-study** | context starvation / oversize signals in forensics | Prior: 500/50 optimal at small scale, unknown at 30k. Structural-v2 candidates only with a hypothesis |
| **Self-query filter extraction** | after metadata lands (4.1) | "according to Plutarch…" → metadata filter; cheap arm, rich metadata |
| **D5 chat-model row** | anytime, ~free | gemma-12b vs qwen-9b (already on llama-swap) on the generation tier — does completeness 3.07 move with the model or is it retrieval-bound? One config change per run |
| **GraphRAG** | multi-hop headroom remains AFTER the Phase 5 agent | Heaviest ingest; must beat the agent on multi-hop to earn it (prior agent hit 4.89/5) |
| **Long-context anti-RAG row** | case-study material, once, at the end | One honest table row: cost ×N vs quality ± |

**Standing skips (receipts on file, do not relitigate):** HyDE (−9.7), contextual
compression, Self-RAG/FLARE, embed-summaries, parent-child, ColBERT, SPLADE.

## Sequencing

```
4.0 D2 gate (bare chunks; ADR-002)
 └─→ 4.1 contextual + metadata pass (re-embed with winner; dense-ctx-v1)
      └─→ 4.2 rerank, aligned (rerank-v1)
           └─→ 4.3 hybrid BM25/RRF + pool-recall lens (hybrid-rerank-v1)  → D3 check
                └─→ 4.4 conditional arms, evidence-triggered
```

Strictly serial for the core chain — each arm is measured against the previous one, so
parallel arms would confound attribution. The 4.4 conditional arms can interleave.

## Out of scope (resist the pull)

- **Agent loop, query decomposition** — Phase 5 (and the agent subsumes the static
  versions; interaction #9).
- **Router, semantic cache, cost tracking, Langfuse** — Phase 6.
- **Multi-query expansion** — prior result marginal (+2.1); the agent does it
  organically. Revisit only as a Phase 6 routing question.

## Definition of done

- [ ] ADR-002 (D2) written; corpus on the winning embedder, parity fixture updated
- [ ] Contextual/metadata pass: measured keep/reject vs dense-v1, enrichment versioned
- [ ] Rerank arm: measured on contextualized text, reranker choice justified
      (local vs hosted, latency on target hardware)
- [ ] Hybrid arm: measured incl. pool-recall@50; D3 confirmed or challenged
- [ ] Production single-shot pipeline = measured winners, behind the same `Retriever`
      protocol, served by `POST /ask` unchanged
- [ ] eval-log ablation table: one row per arm, baseline → final, per category
- [ ] All four CI checks green; everything committed phase-style
