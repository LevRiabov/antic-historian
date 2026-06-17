# Phase 6 — Production layer

> Plan written 2026-06-16, after the Phase 5.1 agent stabilization (commit `0cae597`).
> Scope per [project-plan.md §Phase 6](project-plan.md) and the buyer checklist in
> [module-10-build-plan.md](module-10-build-plan.md) (Tier 1–2). Production features are
> not techniques behind the Phase-4 ablation door — they ship to make engineering rigor
> *visible in the product in ~90 seconds*. Where a production choice trades quality for
> cost/latency (model routing), it still enters through the ablation door: measure → keep/
> reject → note. Rejections with receipts are case-study content.
>
> **Exit criteria:** (1) gate **D5** decided (ADR-003: production LLM lineup + fallback
> order); (2) the deployed `/ask` carries USD cost + `served_by` on every answer; (3) a
> forced fallback produces a visible indicator; (4) rate limit + session cap demonstrable;
> (5) one Langfuse trace per request (retrieve → tokens → cost → latency) screenshot-ready;
> (6) the buyer checklist (module-10 Tier 1–2) all demonstrable; (7) case-study assets
> captured as we go (trace, fallback firing, cost readout, clean refusal).

## Settled going in (decisions already made for Phase 6)

- **What the app serves:** **single-shot is the default fast path; the agent is an opt-in
  "deep mode" that streams its reasoning steps** (thought → action → observation). Rationale:
  the agent wins synthesis/contradiction but costs ~75 s latency + ~19k prompt tokens/query
  (eval-log 2026-06-16) — wrong as a default for a public, abusable demo, but its visible
  search-read-cite loop is the best "how it works" artifact we have. Streaming the loop turns
  the 75 s into the demo. Requires teaching the agent path to stream (see 6.7).
- **Semantic caching: CUT from launch scope** (rejection-with-receipts — see Appendix A).
- **Prompt caching: conditional on D5** (provider-specific — see 6.6).

## The two gating decisions

### D5 — production LLM lineup (gates 6.5 routing, 6.4 fallback, 6.6 prompt-cache)
The eval log has effectively converged: **deepseek-v4-pro = the standing agent KEEP**
(agent-v4/v5.1), judge = kimi-k2.6 + qwen3.7-max split. Phase 6's job is to write that down
as **ADR-003** with the full request-time lineup:

| role | front-runner | notes |
|---|---|---|
| chat / agent (quality) | deepseek-v4-pro | the standing KEEP; serves "deep mode" + single-shot quality tier |
| cheap / demo tier | TBD (measured in 6.5) | candidate for the abusable public default if quality holds |
| fallback order | primary → 2 alternates | distinct providers so one outage ≠ total outage (6.4) |
| embed | qwen3-embedding-8b / Nebius | settled (ADR-002), query-time cost |
| retrieval | `dense-ctx-v1` (contextual dense, **no reranker**) | the reranker was dropped at D5 — the 8B embedder out-ranks a weak cross-encoder and SOTA cohere-pro was marginal+paid; production = contextual dense |

D5 stays "provider-agnostic in code" (ADR-001) — the lineup is config, the ADR records *which*
config ships and why.

## The cost-of-a-query risk (drives 6.4 priority)

Every production query is all-hosted: embed (qwen3-8B/Nebius) + agent/chat LLM. No reranker
(dropped at D5), so embed cost is small and the LLM dominates — the agent path is ~19k prompt
tokens × multiple steps on deepseek-v4-pro. A public demo with no caps can run up real money
fast. This makes **6.4 (rate limit + caps) load-bearing, not nice-to-have**, and makes a cheaper
"demo tier" (6.5) a real consideration for the default path. Ingest-time spend is the cheap
ledger; this is the expensive one (CLAUDE.md §Conventions).

## Measurement protocol (the one workstream that touches quality)

Only **6.5 model routing** moves quality, so only it goes through the full ablation door:
1. `ahx eval generate --label <tier>-v1 --judge` per candidate tier (single-shot + agent).
2. Eval-log entry: cost/quality table, keep/reject note for the public-default tier.
3. Everything else (tracing, cost, fallback, limits, guardrails) is verified by tests +
   a demonstrable trace/asset, not by a golden-set run.

---

## Workstream 6.0 — Close D5 → ADR-003 ✅ DONE (2026-06-16)
[ADR-003](docs/adr/003-d5-llm-lineup.md) written: agent = deepseek-v4-pro, split judge
(kimi-k2.6 + qwen3.7-max), retrieval rerank-free (`dense-ctx-v1`), embeddings unchanged
(ADR-002). The fast-path model, cheap/demo tier, and fallback order are explicitly deferred to
6.5/6.4 (they need the routing/fallback ablations). Lineup is config (`AHX_*`), not hardcoded.

## Workstream 6.1 — Langfuse tracing (the substrate, build first)
One root span per request; child spans for embed / retrieve / each LLM call. Wrap at the
existing thin seams — the `ChatModel` Protocol ([llm.py](backend/src/ahx/llm.py)) and the
`Retriever` callable ([pipeline.py](backend/src/ahx/generation/pipeline.py)) — so no business
logic imports Langfuse (thin-waist rule, ADR-001). Async/background flush — **rule #7, never
block the event loop**. Cost (6.2) and `served_by` (6.4) attach as span attributes once they
exist, so this lands first.
**Exit:** a screenshot-ready trace for one `/ask`: retrieve → tokens → cost → latency.

## Workstream 6.2 — Cost tracking per request ✅ DONE (2026-06-16)
[ahx/pricing.py](backend/src/ahx/pricing.py): price table FETCHED from OpenRouter (not hand-typed,
rule #6) via `ahx pricing refresh` → committed dated snapshot `pricing_snapshot.json` (fetched 2026-06-16:
deepseek-v4-pro $0.435/$0.870 per M, etc.). `cost_for()` computes generation USD; local models (bare
id) = $0, unknown hosted = unpriced/None (never silent $0). `Cost` rides on `DoneEvent` → the SSE
`done` event + the 6.1 trace metadata (`cost_usd`/`cost_priced`). Single-shot + agent paths both fill
it. Tests: tests/test_pricing.py. Gates green (153 tests). Verified: real /ask → cost on SSE + trace.
**Scope:** generation tokens only — query-embed cost (~$2e-7) excluded (negligible + EmbeddingClient
doesn't surface usage); cache-read prices captured for 6.6 readiness.

## Workstream 6.3 — Guardrails (mostly framing + small adds)
The **citation/refusal audit already is the output guardrail** — `MarkerAudit` + `_is_refusal`
([pipeline.py](backend/src/ahx/generation/pipeline.py)) catch ungrounded/hallucinated citations
and detect refusals. Frame it as such; add only: a lightweight input check (length already done
in [app.py](backend/src/ahx/api/app.py); add cheap off-topic/injection rejection) and a test
asserting refusal fires when sources are empty. **No heavyweight guardrails framework.**
**Exit:** input guard + the existing audit documented as the guardrail story; tests green.

> **Companion learning track:** [phase-6-3-security-plan.md](phase-6-3-security-plan.md) (6.3-lab)
> red-teams the attacks this app actually exposes (system-prompt extraction, scope-escape,
> grounding-bypass) with an Attack-Success-Rate metric + ablated defenses, and documents the full
> production playbook (uploads/sensitive-tools/trifecta) as an appendix. 6.3 stays lean; the lab
> is where the security *learning* lives.

## Workstream 6.4 — Fallback chain + visible indicator ✅ DONE (2026-06-17)
`CompositeChatModel` ([llm.py](backend/src/ahx/llm.py)) wraps the D5 lineup `[primary,
*AHX_CHAT_FALLBACKS]` and **falls over only before the first delta** (peek-the-first-stream-
event; once tokens ship a mid-stream failure propagates, can't switch) — the cross-provider
layer above each model's own retry/backoff. `served_by` rides on `StreamEnd`/`ChatResult` (NOT
shared instance state — the composite is shared across concurrent requests): each model stamps
its own id, the composite passes the served one through, and the pipeline prices cost + labels
the SSE `done` event + Langfuse trace by the model that ACTUALLY served. Lineup config (ADR-003):
served primary = deepseek-v4-pro, backups kimi-k2.6 + qwen3.7-max (distinct families; shared
OpenRouter gateway, noted as the residual single-point). **Rate limit + per-session cap**
([api/limits.py](backend/src/ahx/api/limits.py)): a small CUSTOM in-memory limiter (not slowapi —
the per-session "N of M left" counter is bespoke) behind a `RateLimiter` interface (Redis = the
documented scale path); an IP sliding window (abuse) + a lifetime session cap (free-tier, keyed on
a client `X-Session-Id` header) enforced as an `/ask` dependency → a structured **429** (+
`Retry-After`) BEFORE the stream opens (no model spend on a rejected request). "N of M left" rides
a new `meta` SSE event. Verified by tests (fallover `served_by` + mid-stream-propagate + all-down,
limiter arithmetic + reject-without-consume, route 429s); no golden-set run — not a quality lever.
**Commit `1ed986a`.**

## Workstream 6.5 — Model routing (cheap/expensive) — through the ablation door
A **measured 2-tier switch** (quality vs cheap), NOT a clever query-complexity router (that's an
unmeasured technique and a regression risk). Run the golden set on each tier (single-shot +
agent), publish the cost/quality table, and pick the public-default tier with a keep/reject
note. The table is the deliverable — it's the "I measured before I shipped" case-study moment.
**Exit:** eval-log entry with the cost/quality table; the public default tier chosen with a note.

## Workstream 6.6 — Prompt caching (conditional on D5)
Only worthwhile if the D5 lineup includes a provider with explicit cache control (e.g. Anthropic
`cache_control` on the static system/rubric block — retrieved context changes per query, so only
the system prompt caches). Modest token-cost win; a few lines if supported, skipped otherwise.
Measure the token-cost delta if built.
**Exit:** either a measured token-cost delta, or a one-line note that the chosen lineup doesn't
support it (rejection-with-receipts).

## Workstream 6.7 — Stream the agent ("deep mode" over the API) ✅ DONE (2026-06-17)
`mode: "deep"` on `/ask` streams the ReAct loop live then the cited answer — order: `meta → step*
→ sources → delta* → done`, with a new `step` SSE event (`thought`/`tool`/`args`/`observation`/
`chunk_ids`/`searches_left`). `astream_agent` ([graph.py](backend/src/ahx/agent/graph.py)) drives
`graph.astream(stream_mode="values")`, yielding each completed `Step` as it lands then the final
`AgentState` — the LangGraph type stays inside the boundary file (new `AgentGraph` alias);
`invoke_agent` (the eval path) is untouched. `stream_agent_events` + `make_agent_streamer`
([runner.py](backend/src/ahx/agent/runner.py)) emit the `StepEvent`s then reuse the existing
`build_agent_events` for the SAME `sources`/`done` as single-shot — so **eval == served**: the
answer is the grammar-decided text the judge scored, and the deltas are a *cosmetic* word-stream of
it (true token-streaming was the rejected option — it would diverge served from measured + need a
paid re-validation). The guard was factored into `guard_stream`, shared by both paths (input block
short-circuits BEFORE the expensive loop; output validation on the final answer). Built once in
`lifespan` over the traced+composite chat → deep steps are traced and fall over across providers;
served on `dense-ctx-v1` (the D5 KEEP, no paid reranker); `served_by` + `cost` ride the agent
`done` event (6.4/6.2); a `503` fires if deep is requested when unavailable. **Verified live** (one
deepseek-v4-pro deep query): streamed `search`/`read`/`list_sources` steps, 14 citations across 6
authors, `served_by=deepseek/deepseek-v4-pro`, `cost=$0.0101` priced, 369 deltas re-joining exactly
to the answer. Tests: `astream_agent`, `stream_agent_events`, deep-mode route + 503. **Commit `588bbdf`.**

---

## Appendix A — Semantic caching: considered, rejected (receipts)

**What it is.** A normal cache keys on the exact query string (`Map<string, Answer>`). A semantic
cache keys on *meaning*: embed the incoming query (machinery we already have via `EmbeddingClient`),
cosine-compare against vectors of previously-answered queries, and if the nearest scores **≥ a
threshold**, return its cached answer.

**Why we reject it.** The entire decision is one threshold, and cosine similarity measures topical
closeness, not answer-equivalence. The things that flip an answer are exactly what embeddings smear
together — and they map onto our hardest categories:
- **Negation:** "Did Caesar cross the Rubicon?" vs "Did Caesar *not* cross?" — cosine ~0.98, opposite answer.
- **Entity swap:** "How did *Caesar* die?" vs "How did *Pompey* die?" — high cosine, different answer.
- **Scope/date:** "Cicero in 63 BC" vs "...in 44 BC" — near-identical vectors.

Low threshold → false hits serve a confident-but-wrong answer on the system whose whole pitch is
faithfulness. High threshold (≥0.97) → ~0 hit rate at demo traffic anyway. No setting is both safe
and useful for this query distribution. **Cut from launch; documented here as case-study content.**
