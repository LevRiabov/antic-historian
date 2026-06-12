# ADR-002 — Gate D2: Embedding Model

**Date:** 2026-06-12 · **Status:** accepted
**Decision:** **qwen3-embedding-8b, hosted via OpenRouter pinned to Nebius**, MRL-truncated
4096→**1024 dims** + L2-renorm, $0.01/M tokens. Local **qwen3-embedding-0.6b stays as the
documented fallback** (measured floor: 35.2% recall@5). The small-local shortlist arms
(voyage-4-nano, gte-modernbert-base) were **not run** — rationale below.

## Context

Gate D2 (project-plan.md, docs/embeddings.md): queries are embedded at request time in
production → CPU-class local or hosted API. Shortlist assumed local-vs-local with 1–3
point quality gaps where ops would decide, plus a hosted "ceiling reference" to learn what
local leaves on the table. Decision criteria (written first): golden-set recall/MRR per
category (primary), query latency, RAM, ops complexity, license/lock-in.

## What we measured (all on identical structural-v1 chunks, bare text, golden set v2.0)

| arm | config | recall@5 | recall@20 | MRR | query latency |
|---|---|---|---|---|---|
| baseline | qwen3-0.6b local (llama-swap GPU) | 35.2% | 48.0% | 0.372 | ~104ms mean |
| arm 1 | qwen3-8b OpenRouter unpinned, 1024d | 53.2% | 74.9% | 0.519 | 3.3s mean, 8.8s p95 (!) |
| arm 2 | qwen3-8b Nebius-pinned, 2000d | 54.3% | 75.8% | 0.529 | — |
| **final** | **qwen3-8b Nebius-pinned, 1024d** | **53.2%** | **74.9%** | **0.522** | **~0.9s consistent** |

Run records: `backend/evals/runs/2026-06-12T*-dense-8b-*.json`. Corpus re-embed: ~$0.11 /
~70 min per arm. Total gate spend ≈ $0.36.

Key facts the arms established:

1. **The embedder was the binding constraint, not chunking/enrichment:** +18.0 recall@5
   from the model swap alone — larger than the expected headline lever (contextual
   retrieval, +16 at small scale). Synonym +41.7 points (the Victorian-translation
   vocabulary tax, largely paid by the 13× larger model).
2. **MRL truncation 4096→1024 is free on this corpus:** 2000d scored +1.1 — inside the
   ±1-question noise floor (pre-registered rule: within noise → ship smaller). 1024d =
   ~120MB vectors, comfortable in the 500MB free-tier budget; 2000d would crowd it.
   Re-check dims once on contextualized text when 4.1 re-embeds anyway.
3. **Unpinned OpenRouter is a parity + latency hazard:** provider roulette mixed runtimes
   (incl. an fp8 endpoint at 70% uptime) into one corpus and produced 5–8s latency
   spikes. Pinned probes: Nebius ~0.9s consistent / 100% uptime-30d; DeepInfra 4–7s.
   Pinning reproduced arm 1's numbers exactly → the mix hadn't measurably hurt quality,
   but reproducibility now holds by construction.

## Why the small-local arms were skipped (a measured decision, not a skipped one)

The shortlist's premise — "local CPU models within a few points of hosted, ops decides" —
was falsified by arm 1: the gap to close became ~18 points. No 149M–340M model plausibly
closes that on out-of-distribution Victorian prose when the 595M incumbent sits at 35.2%.
Running two more corpus embeds + building a serving sidecar to confirm a foregone
conclusion is ritual, not measurement. What we keep instead:

- **qwen3-0.6b** remains fully measured and llama-swap-served — the fallback if the
  hosted dependency ever becomes unacceptable, with a known cost (−18 recall@5).
- **voyage-4-nano's shared-embedding-space trick** (local CPU queries against a
  hosted-embedded corpus, one space) is noted as the escape hatch to evaluate IF
  query-time API dependency becomes a product problem. Not worth an arm today.

## Consequences

- **Accepted: query-time API dependency.** ~0.9s added to time-to-sources (was ~0.1s
  local), $0.01/M (≈ $0/month at our volume), OpenRouter+Nebius availability risk.
  Mitigations: open weights (same model servable on DeepInfra/Nebius direct, or GPU
  self-host — no proprietary lock-in), measured local fallback, provider pinned via
  config (`AHX_EMBED_PROVIDER`), `allow_fallbacks=false` fails loudly rather than
  silently switching runtimes.
- **Embedding config is now part of measurement provenance:** model + dims + provider +
  MRL flag all live in Settings; the parity fixture records the model; run records carry
  `embed_model`. Any change re-runs `ahx ingest parity` (rule #3).
- Ingest-side embeds also go through the hosted API (~$0.11/re-embed) — fine per the
  cost ledger (ingest is one-time), and it keeps corpus/query parity by construction.
- The Phase 4 plan inherits a 53.2%/74.9% floor: the rerank arm gets a far richer top-50
  pool (contradiction recall@20 = 90.9–100%) than the plan assumed.
