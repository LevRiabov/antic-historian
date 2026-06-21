# Module 10 Build Plan — Production RAG Agent (Buyer-Facing Demo)

The goal of this app is **not** to be a great history tool. It's to make a potential client think: *"This person builds reliable, measurable, cost-controlled LLM systems — they'd do the same for my product."* The history domain is just the vehicle.

Three rules govern every decision below:

1. **The buyer is not a historian.** Every feature must map to a fear a buyer has about *their own* LLM product (hallucination, cost, "is it any good," reliability).
2. **Rigor must be visible in the UI in ~90 seconds.** If the buyer has to read the README to be impressed, you've lost. Surface the engineering *in the product.*
3. **Engineering legibility > history depth.** The 50th book impresses nobody. A clickable citation and a live evals tab impress everyone who's hiring.

---

## What this app must prove (the buyer's checklist)

| Buyer's fear about their own product | The feature that answers it |
|---|---|
| "It hallucinates / I can't trust it" | Inline citations on every answer + honest out-of-scope refusal |
| "I have no idea if it's actually good" | An **evals tab inside the live app** (faithfulness, completeness, refusal %) |
| "It'll cost me a fortune" | Per-request cost shown live (per query, per article) |
| "It'll fall over in production" | Visible fallback chain + tracing/observability |
| "Is it production-grade or a toy?" | Rate limiting, monitoring, clean architecture, an "how it works" page |

---

## Scope: two modes, two jobs

- **Mode 1 — Chat** = your **trust demo.** Ask a historical question → grounded answer with citations + suggested follow-ups. This is what converts skeptics. *Lead with this.*
- **Mode 2 — Article builder** = your **capability demo.** Topic in → ~10-page professional article, fully cited, with the cost shown. This is the "wow, it makes something substantial for two cents" moment.

If time-constrained, **Mode 1 is the priority.** Mode 2 is the headline-grabber but the trust signals close the deal.

---

## INCLUDE — ranked by buyer impact

### Tier 1 — the conversion core (build first, do not ship without these)
- [ ] **Inline citations on every chat answer** — show book + passage, expandable to the actual quoted source text. This is your faithfulness story in one glance.
- [ ] **Honest out-of-scope refusal** — when asked something not in the corpus, it says "I don't have a source for that" instead of inventing. Add a **suggested prompt that deliberately triggers a refusal** so the buyer *sees* it happen.
- [ ] **Evals tab in the live app** — faithfulness 4.40, completeness 4.62, refusal 96% on a 50-question golden set, plus one paragraph on methodology and the question categories (literal / synonym / multi-hop / synthesis / out-of-scope). Own the n=50 explicitly; note it's curated and expandable.
- [ ] **Per-request cost display** — "this answer cost $0.0008" / "this 10-page article cost $0.02." A live counter. Memorable and de-risking.
- [ ] **Frictionless landing** — states your positioning line, offers 3–4 "try these" prompts (one of them the refusal trigger). Guides the 90-second demo so the buyer hits the impressive parts without hunting.

### Tier 2 — strong differentiators (build second)
- [ ] **Tracing / observability** — Langfuse free tier: a trace per request (retrieval → tokens → cost → latency). Doesn't need to be user-facing, but you must be able to screen-share/screenshot it for the case study.
- [ ] **Visible fallback chain** — Gemini Flash → Groq → OpenRouter, with a small indicator when a fallback fires. Your Module 9 reliability skill, shown not claimed.
- [ ] **Rate limiting + query caps** — keeps you inside free tiers *and* reads as a production signal.
- [ ] **"How it works" page** — one screen: architecture diagram, the technique stack, the "$0/month, here's how" story.

### Tier 3 — only if cheap and fast (don't let these block launch)
- [ ] Suggested next questions (good product instinct, but not a trust signal — keep the logic simple).
- [ ] Light UI polish: clean typography, dark mode. Not avatars/animations.

---

## Architecture (keep it ~$0/month)

- **Frontend:** Next.js on Vercel (free).
- **RAG service / API:** FastAPI (or similar) on a free CPU tier — Hugging Face Spaces / Fly.io / Render. No GPU in production.
- **Corpus:** static. Embed **once locally** (BGE-M3 on the 5070 Ti) → ship a file-based vector store (sqlite-vec / LanceDB) bundled with the app. No DB to host.
- **Generation:** free hosted fallback chain — Gemini Flash → Groq → OpenRouter.
- **Tracing:** Langfuse free tier (or self-hosted).
- **Guardrails:** rate limiting + per-session query caps to stay inside free tiers and prevent abuse.

*Free tiers are for the demo, not an SLA — that's fine. Inference becomes a paid line item only if this ever becomes a real product.*

---

## Build sequence

- [ ] **A — Core chat end to end.** Port the console RAG logic into the service; minimal UI; answers returning with citation data attached.
- [ ] **B — The visible-rigor layer.** Citation display, out-of-scope refusal + demo prompt, cost counter, evals tab. *(This is where most of the buyer value lives — don't rush past it to build features.)*
- [ ] **C — Reliability + observability.** Fallback chain with indicator, Langfuse tracing, rate limits/caps.
- [ ] **D — Article builder mode.** Topic → ~10-page cited article, cost shown.
- [ ] **E — Polish + deploy.** Landing page with positioning + try-these prompts, "how it works" page, README (architecture diagram, eval results, demo gif).
- [ ] **F — Capture case-study assets** *(do this as you go, not at the end):* screenshots of a trace, the eval table, a cost readout, a fallback firing, and a clean refusal. These become the case study.

---

## Definition of "demo-ready"

- [ ] Live URL, no login, loads fast.
- [ ] A first-time visitor, with zero instructions, can within 90 seconds: ask a question and see a cited answer, trigger a refusal, open the evals tab, and see a cost number.
- [ ] Repo is public with README, architecture diagram, published evals, demo gif, permissive license.
- [ ] You can screen-share a trace and a fallback on demand.

---

## Feeds the case study (the payoff)

Everything above doubles as case-study material. Your **Module 1–9 ablation data** ("technique X moved faithfulness +0.3, technique Y did nothing") becomes the *justification* section: "Before building production I measured these techniques against my golden set — here's what I kept and why." That turns your learning work into professional rigor, inside the asset that actually converts.

Case study structure: **lead with the chat trust story, close with the article-builder wow.** Business/value framing first, the measured engineering second.
