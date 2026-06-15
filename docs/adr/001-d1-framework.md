# ADR-001 — Gate D1: RAG/Agent Framework

**Date:** 2026-06-10 · **Status:** accepted
**Decision:** **LlamaIndex for the RAG layer** (ingestion, indexing, retrieval, rerank slots — Phases 1–4) **+ LangGraph for agent orchestration** (Phase 5), behind a thin project-owned interface. Haystack not spiked (rationale below).

## Context

Gate D1 (project-plan.md): pick the framework(s) by spike, not mindshare. Criteria: control/transparency, advanced-retrieval support, streaming + citation ergonomics, market recognition, docs quality. The user has never used either framework — learning value counts.

## What we did

Two self-contained scripts ([backend/spikes/d1/](../../backend/spikes/d1/)) building the identical pipeline: 2 books (Anabasis + Twelve Caesars, ~2MB) → token chunking 500/50 → local qwen3-embedding-0.6b with Qwen3 query-prefix policy → in-memory store with persistence → top-6 retrieval → streamed, citation-forced answer from local gemma-12b → 3 in-scope questions + 1 refusal trigger. Identical prompts, models, and prefix policy (`common.py`); only the framework varies.

## Observations

| | LlamaIndex (core 0.14) | LangChain 1.x + LangGraph 1.2 |
|---|---|---|
| Pipeline LOC | ~140 | ~165 |
| Build (1,111 / 1,335 chunks) | 34.1s | 30.8s |
| Retrieval latency | 46–83ms | ~same |
| Ergonomics highlights | `VectorStoreIndex.from_documents(transformations=[...])` concise; persistence built-in; retriever API clean; `stream_chat` trivial | `StateGraph` is explicit and readable — nodes are plain functions over a typed state; streaming via `stream_mode=["updates","messages"]` is powerful |
| Friction | Plugin-package sprawl (3 packages for core+2 integrations); global `Settings` is a footgun (avoided by passing models explicitly); `OpenAILike` needs `is_chat_model`/`context_window` spelled out | `check_embedding_ctx_length=False` required for non-OpenAI embedding endpoints (the §6 footgun, met in the wild); state-schema strictness threw on an undeclared key; stream payload shapes underdocumented |
| Prefix-policy override (our rule #3) | Subclass, override `_get_query_embedding` — clean | Subclass, override `embed_query` — clean |
| Both | Citation forcing was manual prompt work in both — neither gives structured citations for free at this level; refusal worked in both |

**The decisive criterion — Phase 4 ablation support:** LlamaIndex has first-class building blocks for almost our whole technique menu: ingestion `transformations` pipeline (slot for contextual notes), `NodePostprocessor` chain (slot for rerankers — BGE/Cohere/Voyage integrations exist), RAPTOR llama-pack, `PropertyGraphIndex` (GraphRAG), hybrid retrievers + fusion. On the LangChain side these are mostly assemble-it-yourself. Conversely, **LangGraph is the stronger agent substrate** (explicit state, checkpointing, interrupts, streaming per node) and the single most-demanded orchestration name in job posts.

**Honest spike caveat:** retrieval *quality* differed on Q1 (Julius's death) — LangChain's splitter happened to isolate the right chapter; LlamaIndex's pulled other-emperor passages. That's **chunking-boundary sensitivity, not framework quality** (different default splitters → different boundaries), and it previews two Phase 1/4 priorities: our own uniform chunker (chunking.md) and contextual notes (both spikes mixed up "which Caesar" to some degree — the corpus-scale ambiguity problem on full display, in miniature).

## Decision

1. **LlamaIndex** owns ingest/index/retrieve/rerank (Phases 1–4).
2. **LangGraph** owns the agent loop (Phase 5) — re-confirm with a mini-check against LlamaIndex Workflows when Phase 5 starts; switching cost is low behind (3).
3. **Thin waist:** the eval harness, API layer, and ablation modules depend on *our* pydantic interfaces (retriever in → scored passages out), never on framework types. Frameworks stay at the edges and remain swappable.
4. Haystack: not spiked. Weakest of the three on RAPTOR/GraphRAG coverage and market recognition; two strong candidates produced a clear answer — a third spike wasn't going to change the outcome. Revisit only if both choices fail us.
5. Spike code stays in `backend/spikes/d1/` as the ADR's evidence; `spike` dep group gets promoted to main deps as Phase 1 adopts LlamaIndex.

## Consequences

- Two framework dependencies, one per layer — justified by each being best-of-breed for its layer and by the thin waist keeping them replaceable.
- Known LlamaIndex risks accepted: global-Settings footgun (banned by convention — models passed explicitly), plugin sprawl (deps pinned via uv lock).
- The wrong-Caesar retrieval failure is the first entry in the case-study evidence log.

## Phase 5 recheck (2026-06-15) — LangGraph confirmed

Decision point 2 above promised a mini-recheck of LangGraph vs LlamaIndex Workflows when
the agent layer started. Done; **LangGraph confirmed**, faster than expected because the
counter-argument collapsed:

- **No LlamaIndex investment to protect.** Across Phases 1–4 the thin waist won completely —
  `grep llama_index src/` returns nothing. Every retrieval ablation (contextual notes,
  rerank-on-aligned-text, RRF) was written as plain functions behind our own pydantic
  protocols; LlamaIndex was spiked but never adopted in production code. So "use LlamaIndex
  Workflows because we already use LlamaIndex" is false — both frameworks are fresh deps.
- **Our loop doesn't use framework tool-calling.** Phase 5 uses a *grammar-constrained ReAct*
  loop: the think step is our own `llm.complete(..., response_format=<GBNF schema>)`, not
  `bind_tools`. We need the framework only for loop scaffolding — typed state, conditional
  edges, checkpointing, per-node streaming — which is LangGraph's core competency and the
  résumé-valuable name (a stated project goal). LlamaIndex Workflows' event model adds nothing
  here.

`langgraph` promoted from the (now-removed) `spike` dependency group into main `dependencies`
(1.2.4, same major line the D1 spike validated). Framework types stay inside `ahx.agent`,
never in the eval harness or API (the thin-waist rule, unchanged).
