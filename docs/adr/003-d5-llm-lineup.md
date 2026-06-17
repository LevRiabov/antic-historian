# ADR-003 — Gate D5: Production LLM Lineup

**Date:** 2026-06-16 · **Status:** accepted (agent + judge settled; fast-path / cheap-tier /
fallback deferred to Phase 6 — see Open)
**Decision:** **Agent reasoner = deepseek/deepseek-v4-pro** (hosted, OpenRouter), driving the
grammar-constrained ReAct loop. **Eval judge = split** — **moonshotai/kimi-k2.6** for
refusal/faithfulness/completeness + **qwen/qwen3.7-max** for attribution only. **Retrieval ships
rerank-free** (`dense-ctx-v1`, contextual dense — the Phase-4.2 cohere-pro KEEP is overturned).
The single-shot default model, the cheap/demo tier, and the fallback order are **deliberately left
open** for the Phase-6 routing/fallback ablations.

## Context

Gate D5 (project-plan.md): the production LLM lineup — chat, agent, judge, cheap-tier — decided
by ablation against the golden set, kept reversible by the provider-agnostic seam (ADR-001: business
logic depends on the `ChatModel` Protocol, model = two Settings values). Two hard constraints frame
it: **judge calibration must precede any model claim** (rule #5 — a weak/unstable judge invents
findings), and **production deploys to a no-GPU tier**, so every query-time model is hosted (the
expensive ledger). D5 is the one gate marked "continuous"; this ADR records what the Phase-4/5
ablations settled and what the Phase-6 production work still has to decide.

## What we measured

**1. Agent reasoner: gemma-12b → deepseek-v4-pro.** Same agent (agent-v2 prompt), same retrieval
(`dense-ctx-v1`), same split judge — a clean model comparison (the gemma baseline was re-scored by
the identical judge). Run records: `gen-agent-v3-deepseek-pro` (2026-06-15T18-01-51Z) vs the gemma
probe-a re-score.

| metric | gemma agent | **deepseek-pro** | Δ |
|---|---|---|---|
| OOS refusal accuracy | 73–77% | **100% (26/26)** | the headline trust fix |
| attribution | 4.31 | **4.67** | +0.36 |
| completeness | 4.68 | **4.96** | +0.28 |
| faithfulness | 4.63 | 4.61 | flat |

The two documented gemma model-limits — **source-absent OOS substitution** (answering an absent named
work from secondary mentions) and **namesake/date conflation** (two Pausaniases, Mantinea 418-vs-362)
— were *strength* limits, not prompt gaps (prompt-v2 recovered only 1/8). deepseek-pro clears both.

**2. Agent prompt hardened to recover the cost — agent-v4 (KEEP).** The model swap raised in-scope
false-refusal to 18.5% (deepseek abstains rather than half-assemble a distributed cross-book/synthesis
answer). agent-v4 (graded escape hatch + visible step-budget + forced-synthesis-at-bound) closed it
without touching quality. Run record: `gen-agent-v4-deepseek` (2026-06-16T08-01-40Z).

| metric | agent-v2 | **agent-v4** |
|---|---|---|
| in-scope false-refusal | 18.5% (25) | **3.0% (4)** |
| — cross-book / synthesis | 13/28 · 7/18 | **1/28 · 1/18** |
| faithfulness / completeness / attribution | 4.61 / 4.96 / 4.67 | 4.52 / 4.94 / 4.71 (within noise) |

A later context-compaction pass (**agent-v5.1**, `keep_ids` relevance filter) cut **prompt tokens −11%**
with quality held; the false-refusal axis was found **too high-variance to rate from a single run**
(~90% non-reproducible run-to-run — a measurement property, deferred to a budgeted multi-sample protocol).
Run record: `gen-agent-v5` (2026-06-16T13-07-18Z).

**3. Judge: split kimi-k2.6 + qwen3.7-max (calibrated before any D5 claim).** A two-pass variance probe
on frozen answers found the **attribution** rubric judge-noise-limited on a flash-tier judge — re-scoring
identical text swung 0.66/question with full 1↔5 flips, swallowing the agent's +0.36. Routing attribution
to a stronger, different-family model fixed it. Run records: probe-a/probe-b (2026-06-15T15/16).

| dimension | flash judge | **kimi/qwen split** |
|---|---|---|
| attribution mean \|Δ\|/question | 0.66 | **0.153** |
| attribution 1↔5 flips | ~25 | **4** |
| faithfulness \|Δ\| (kimi) / completeness \|Δ\| (kimi) | — | 0.211 / 0.123 |

Attribution is now scored *as stably as* faith/compl. Constraint recorded: **a judge must be ≥ the
generated model's tier**; both judges are frontier-tier and a different family from the agent.

**4. Retrieval ships rerank-free (`dense-ctx-v1`).** The Phase-4.2 cohere-pro reranker was overturned:
it was marginal over the 8B embedder (+1.7 @5) and a query-time cost forever (the expensive ledger),
and the agent runs on contextual dense throughout the D5 work. (Receipt-gap noted: the reranker-drop
was confounded with the model swap — no rerank-ON deepseek arm was run — but the cb-001 probe confirmed
4.2's "cross-book not rerankable" finding, so the lever there is agent search strategy, not rerank.)

## Decision

1. **Agent reasoner = `deepseek/deepseek-v4-pro`** (OpenRouter), grammar-ReAct via
   `llm.complete(response_format=<schema>)`. The agent is the opt-in **"deep mode"** (phase-6-plan.md):
   it wins synthesis/contradiction but costs ~64–75s and ~19k prompt tokens/query.
2. **Judge = split:** `moonshotai/kimi-k2.6` (refusal/faithfulness/completeness) +
   `qwen/qwen3.7-max` (attribution only). Eval-time only — never on the request path.
3. **Retrieval = `dense-ctx-v1`, no reranker.** Supersedes the Phase-4.2 cohere-pro KEEP.
4. **Embeddings = `qwen3-embedding-8b` / Nebius / 1024d** — unchanged (ADR-002).
5. **Provider-agnostic, by config.** Every model above is `AHX_*` Settings; no provider/model in
   business logic. Swapping any is two values, zero code (ADR-001 thin waist).

## Open — deferred to Phase 6 (the production layer)

These are genuinely undecided and enter through their workstream's measurement door, not this ADR:

- **Single-shot default (fast-path) model** — the API default is single-shot (phase-6-plan.md); which
  model serves it (deepseek-v4-pro vs a cheaper tier) is a cost/quality call for **6.5**.
- **Cheap / demo tier** — for the public, abusable demo default. Measured 2-tier routing ablation, **6.5**.
- **Fallback order** — an ordered, distinct-provider chain so one outage ≠ total outage. **6.4** (the
  `CompositeChatModel` + `served_by` indicator).
- **Prompt caching** — only if the chosen lineup includes a provider with explicit cache control. **6.6**.

## Consequences

- **Query-time cost + latency are the agent's, and they are real.** ~$2.14 generation per 161-question
  run (≈ $0.013/question) + judge $2–4 (eval-only); ~64–75s and ~19k prompt tokens per agent query.
  This is *why* the app defaults to single-shot and why rate-limiting + per-session caps are load-bearing
  (6.4), not cosmetic — a public demo on this reasoner is abusable.
- **The judge is an eval dependency, not a serving one** — it never runs on `/ask`, so its cost/latency
  is bounded to phase-boundary runs.
- **Reversible by construction.** All open weights or hosted-swappable; the seam means the 6.4/6.5
  decisions can change the lineup without code changes.
- **Receipt-gaps carried forward (honest):** the reranker-drop is un-isolated from the model swap, and
  the agent false-refusal rate is single-run-unstable — both pre-registered in eval-log.md for the
  budgeted follow-ups, not hidden.
