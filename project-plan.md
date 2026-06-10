# Antic Historian — Project Plan

A production-grade RAG system over a large corpus of ancient-history primary sources (Caesar / late Roman Republic era, several dozen books). Two goals, in order:

1. **Portfolio asset** — a deployed, polished app + case study that convinces freelance clients: *"this person builds reliable, measurable, cost-controlled LLM systems."* (Buyer-facing feature thinking lives in [module-10-build-plan.md](module-10-build-plan.md) — still valid as the UX/feature North Star.)
2. **Learning vehicle** — production Python AI engineering: the standard toolchain, a mainstream RAG/agent framework, and advanced retrieval techniques chosen *by measurement, not fashion*.

This is **not** a port of `../rag-historian`. That repo was the hand-rolled learning lab; this one uses industry-standard tooling on a 10×+ larger corpus, with a fresh golden set.

---

## What we carry over from rag-historian (knowledge, not code)

The previous project's biggest asset is its **eval-driven methodology** and ablation data. We inherit the lessons, but re-verify anything scale-sensitive:

| Prior finding (950 chunks, 4 books) | Transfers to ~10–15k chunks, dozens of books? |
|---|---|
| Contextual retrieval = biggest lever (+16 recall@5) | Likely yes — re-measure (ingest cost grows linearly) |
| Rerank must score the same representation as retrieval | Yes — architectural principle, scale-independent |
| Hybrid BM25 subsumed by reranker at k=5 | **Likely flips** — rare-term matching matters more in a big heterogeneous corpus. Re-test. |
| HyDE hurts (−9.7 recall@5) | Probably still true, low priority to re-test |
| Naive 500/50 chunking optimal | **Unknown at scale** — re-test; semantic/structural chunking may win with diverse books |
| Prompt-tuning below noise floor at n≈9 per category | Yes — fix with a *larger golden set* this time |
| Agent loop wins on synthesis/contradiction | Yes — and matters more with more sources |
| Measurement bugs move numbers more than techniques | Yes — budget time for eval-harness hardening |

Retrieval was the previous project's weak point (51.6% recall@5 at best). With a much larger corpus it gets harder, which is exactly why this project's headline is **advanced retrieval, chosen by ablation** (RAPTOR, GraphRAG, hybrid, contextual, etc.).

---

## Decisions — settled

| Area | Decision | Rationale |
|---|---|---|
| Backend language | **Python** | Industry standard for AI engineering; the learning goal |
| Python toolchain | **uv, ruff, pyright, pytest, pydantic v2 + pydantic-settings** | The modern production stack; closest analogues to pnpm/biome/tsc/zod |
| API | **FastAPI** (async, streaming via SSE) | De-facto standard for ML services |
| RAG layer | **A mainstream framework** (not hand-rolled) | Custom version already built in TS; framework knowledge is the résumé gap. *Which* framework → Gate D1 |
| Frontend language | **TypeScript** | User's home turf; polish sells the demo. Next.js vs SPA → Gate D4 |
| LLM strategy | **Provider-agnostic from day one**, local → frontier swappable | Via the framework's LLM abstraction or LiteLLM underneath; config-switchable, never hardcoded |
| Vector DB / embeddings | **Deployable required** (no local-only at query time) | Specific choice → Gates D2/D3, decided by measurement |
| Methodology | **Evals first.** No technique ships without golden-set evidence | The proven differentiator from rag-historian |
| Observability | **Langfuse** (tracing, prompt mgmt, cost) | Already known, free tier, Python SDK |

## Decisions — open (decision gates)

| Gate | Decision | When | How we decide |
|---|---|---|---|
| **D1** | ~~RAG/agent framework~~ **DECIDED 2026-06-10** ([ADR-001](docs/adr/001-d1-framework.md)) | — | **LlamaIndex for the RAG layer** (Phases 1–4) + **LangGraph for agent orchestration** (Phase 5, mini-recheck vs LlamaIndex Workflows at phase start), behind a thin project-owned interface (no framework types in eval harness/API/ablation modules). Decided by twin spikes on 2 books with local models — see `backend/spikes/d1/`. |
| **D2** | Embedding model | Phase 2 (provisional pick in Phase 1) | **Hard constraint:** queries are embedded at request time in production → model must run on a cheap CPU tier *or* be a hosted API with a viable free tier. Landscape research + shortlist: [docs/embeddings.md](docs/embeddings.md). Working hypothesis: **voyage-4-nano** (Apache 2.0, CPU-class, shared embedding space with hosted Voyage 4 family), ablated against Qwen3-Embedding-0.6B, gte-modernbert-base, and a hosted ceiling reference, on golden-set retrieval metrics. Re-embedding ~15k chunks is cheap, so the provisional pick is low-risk. The ablation itself becomes case-study content. |
| **D3** | Vector store | After Phase 1 ingestion (real corpus size known) | Landscape research + shortlist: [docs/vector-stores.md](docs/vector-stores.md). Default: **Postgres + pgvector** (Docker locally, Neon free tier in prod — one store for vectors and relational data, full local/cloud parity). Challenger: Qdrant (only if the hybrid-search ablation shows DIY Postgres FTS+RRF underperforming native fusion). Escape hatch: embedded LanceDB. |
| **D4** | Frontend: Next.js vs Vite SPA | Before Phase 7 | Backend is FastAPI either way, so no need for Next API routes. SPA is simpler; Next gives SSR landing (fast first paint for the 90-second demo) + free Vercel hosting. Decide when UI work starts. |
| **D5** | LLM lineup (chat, agent, judge, cheap-tier) | Continuous | Abstraction makes this reversible; ablate models against the golden set as part of Phase 4/5. |

---

## Corpus (fresh start)

- **Scope:** several dozen books around Caesar / late Republic — primary sources (Caesar, Cicero's letters & speeches, Sallust, Plutarch lives, Suetonius, Appian, Cassius Dio, Velleius…) plus possibly tiered secondary works. Public domain translations (Project Gutenberg, Perseus, LacusCurtius).
- **Estimated scale:** ~30–50 books ≈ 3–6M tokens ≈ 10–20k chunks. Small enough for any candidate store; large enough that retrieval quality is a real problem worth solving.
- **Ingestion pipeline is a first-class deliverable:** acquisition → cleaning (Gutenberg boilerplate, footnotes, OCR noise) → structural parsing (book/chapter/section) → metadata (author, work, tier, date written, vantage/bias) → chunking → embedding. Reproducible, idempotent, versioned (chunking_version pattern from rag-historian worked well).
- Source-level metadata (tier, date, vantage) is what powers the contradiction/cross-source features later — capture it at ingest, don't bolt it on.

## Golden set v2 (fresh)

Hand-writing ideal answers for dozens of books doesn't scale like it did for 4. Plan:

- **Target ~100–150 questions** (fixes the n≈9-per-category noise-floor problem).
- **Pipeline:** synthetic generation from sampled chunks (LLM drafts Q + gold spans) → **manual curation pass** (quality bar stays human) → categories: literal, synonym, multi-hop, synthesis, contradiction, out-of-scope, + new **cross-book synthesis** category the larger corpus enables.
- **Gold = character spans** in cleaned source text (chunking-invariant — proven design, keep it).
- **Two metric tiers:** retrieval-only (recall@k, MRR — free, fast, run constantly) and generation (LLM-as-judge: faithfulness, completeness, refusal — use a strong judge from day one; Haiku-judge miscalibration was a known footgun).

---

## Phases

Each phase has an exit criterion; gates fire where marked. Capture case-study assets (screenshots, eval tables, traces) **as you go**.

### Phase 0 — Scaffolding + framework spike → **Gate D1**
Repo setup (uv, ruff, pyright, pytest, pre-commit, CI skeleton, docker-compose for local deps). Spike the 2–3 framework candidates on a toy 2-book pipeline. Pick the framework with a written decision note (ADR-style — these notes accumulate into the case study).
**Exit:** toolchain runs in CI; framework chosen; toy pipeline streams a cited answer.

### Phase 1 — Corpus + ingestion
Acquire and clean the full corpus; build the ingestion pipeline; provisional embedding pick; ingest into a provisionally-chosen store. → **Gate D3** at the end, once real scale and access patterns are known.
**Exit:** full corpus queryable; ingestion reproducible from raw files with one command.

### Phase 2 — Eval harness + golden set v2 → **Gate D2**
Port the *methodology* (not code) from rag-historian: golden-set schema, retrieval metrics, judge rubrics. Build the synthetic-generation + curation pipeline. Run the embedding-model ablation.
**Exit:** `make eval` produces a per-category scorecard; baseline numbers recorded.

### Phase 3 — Baseline RAG end-to-end
Naive-retrieval → generation → **streaming structured citations** through FastAPI. This is the reference implementation every later technique is measured against.
**Exit:** API endpoint streams cited answers; baseline eval row locked in.

### Phase 4 — Advanced retrieval ablations *(the heart of the project)*
Candidates, roughly in expected-ROI order: **contextual retrieval** (proven), **reranking** (proven, representation-aligned), **hybrid BM25 + RRF** (likely valuable at this scale), **chunking re-study**, **RAPTOR** (targets synthesis — prior weak spot), **GraphRAG / property graph** (targets multi-hop/contradiction across many books), query routing/expansion. Each gets: implement → eval → keep/reject decision note. Rejections are case-study content too.
**Exit:** a measured, justified retrieval stack; ablation table.

### Phase 5 — Agentic layer
Multi-step agent (framework-native, e.g. LangGraph or LlamaIndex workflows): corpus search, source-isolated search, read-before-cite, abstention, forced-finalize. Re-run golden set; agent vs single-shot comparison.
**Exit:** agent beats single-shot on synthesis/contradiction categories with evidence.

### Phase 6 — Production layer
Semantic caching, prompt caching, model routing (cheap/expensive), rate limiting + per-session caps, fallback chain with visible indicator, cost tracking per request, Langfuse tracing end-to-end, guardrails.
**Exit:** the buyer-checklist features from module-10-build-plan.md all demonstrable.

### Phase 7 — Frontend + deploy → **Gate D4**
TS frontend (chat with inline expandable citations, evals tab, cost display, how-it-works page, refusal-trigger demo prompt). Deploy: frontend on Vercel/CF Pages, API on a free/cheap CPU tier (Fly.io / Render / HF Spaces).
**Exit:** live URL passes the 90-second demo test from module-10-build-plan.md.

### Phase 8 — Case study + launch polish
Write the case study (lead with trust story, close with capability demo; ablation data as the rigor section). README with architecture diagram, published evals, demo gif. Collect the screen-shareable assets: a trace, a fallback firing, a clean refusal, the cost readout.
**Exit:** public repo + live demo + case study you'd send to a client.

---

## Risks / watch-list

- **Corpus cleaning is a time sink.** Dozens of Gutenberg texts = dozens of formatting quirks. Timebox per book; prefer fewer, cleaner books over more, dirtier ones.
- **Framework friction.** First time with any of these frameworks; expect the abstraction tax rag-historian avoided. Mitigation: the D1 spike, and keeping eval harness framework-independent so we can swap if needed.
- **Free-tier ceilings.** Larger corpus + hosted embeddings + frontier judges → eval runs cost real money. Use retrieval-only metrics (free) for iteration; full judge runs at phase boundaries.
- **Eval-harness bugs masquerade as findings** (cost 30+ min each last time: citation format, forced-finalize, judge calibration). Treat the harness as production code: tests, fixtures, versioned runs.
- **Scope creep on techniques.** "All of them and more" is the ambition, but each must enter through the Phase 4 ablation door — implement, measure, decide, document. No unmeasured features.
