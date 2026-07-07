# Case study — Antique Historian

> A production RAG system over 62 ancient-history primary sources that **answers from its
> sources or honestly says it can't** — and proves it with published, reproducible evals.
>
> **Live demo:** https://historian.loroplanner.com · **Code:** this repository ·
> **Receipts:** every number below links to [docs/eval-log.md](docs/eval-log.md)

---

## The one-paragraph version

Most RAG demos look great until you ask the question the corpus can't answer — then they
invent something. Antique Historian was built to be the opposite: on a 161-question curated
golden set it **answers 100% of in-scope questions** (0% false-refusal), **correctly
declines 96% of out-of-scope ones**, scores **4.41/5 faithfulness** and **4.91/5
completeness** under an LLM judge, and survives a 40-prompt prompt-injection suite at **0%
attack-success** (15–18% undefended). Every technique in the stack — and several that were
*rejected* — earned its place through a measured ablation, logged with a date and a run
record. The result is a system whose trustworthiness is a number, not a claim.

---

## The problem

Question-answering over a large, heterogeneous corpus of historical primary sources is a
high-stakes retrieval problem: the sources disagree with each other, span Victorian
translations of Greek and Latin, and frequently *don't* contain the answer at all. In that
setting the dangerous failure isn't a wrong ranking — it's a **fluent, confident
fabrication** when the right passage wasn't retrieved. A system that's right 90% of the time
and silently makes up the other 10% is worse than useless for any serious use, because you
can't tell the 90 from the 10.

So the design target was never "high recall." It was **calibrated trust**: grounded answers
with verifiable citations when the evidence exists, and an honest refusal when it doesn't —
both *measured*, on a fixed test set, every time the system changes.

## Why naive RAG isn't enough (the floor)

The baseline — dense retrieval, 500-token chunks, top-5 into a local 12B — scored
**recall@5 = 35.2%**. (Notably, that reproduced a prior project's naive-dense floor almost
exactly on a corpus ~30× larger and an independently authored question set — the naive floor
is a property of the *architecture*, not the corpus.) At that retrieval quality the generator
behaved exactly as feared in the worst categories: it either refused honestly (because the
sources it was shown genuinely lacked the answer) or stitched together a plausible-but-thin
answer. Closing that gap — and proving it stayed closed — is the whole project.

## The approach: gates, not vibes

The core methodology is one rule: **no technique ships without golden-set evidence.** Each
candidate enters through an ablation door — *implement → measure → keep or reject → write a
dated decision note* — and the entire history lives in an append-only
[eval log](docs/eval-log.md). Framework, embedder, vector store, and model choices were made
at explicit decision gates (D1–D5) with written criteria, not adopted because they were
fashionable.

This is the part that's hard to fake and easy to verify: the repo contains the *receipts* for
every choice, including the ones that didn't work out.

---

## The trust result

Measured on the latest published run (`agent-v8` / `judge-v3.6`, 2026-06-18), over 161
questions — 135 in-scope across six difficulty categories (literal, synonym, multi-hop,
synthesis, cross-book, contradiction) plus 26 out-of-scope traps:

| Metric | Result | What it means |
|---|---|---|
| **In-scope false-refusal** | **0.0%** (135/135 answered) | It doesn't dodge answerable questions |
| **Out-of-scope honest refusal** | **96.2%** | It declines what it can't ground, instead of inventing |
| **Faithfulness** (LLM judge, 1–5) | **4.41** | Answers stay grounded in the cited sources |
| **Completeness** (1–5) | **4.91** | Answers are materially complete |
| **Attribution** (1–5) | **4.63** | It correctly says *which* source said what |
| **Citation span recall** | **58.2%** | Share of gold evidence spans actually cited |
| **Prompt-injection ASR** | **0%** (15–18% undefended) | The defense stack holds |

The out-of-scope number is the one I'd point a skeptic at first. The hardest traps are
questions about a *named work the corpus doesn't contain* but *does mention* (e.g. "what does
Sappho's poetry express?" when only a secondary footnote about Sappho is in the corpus). A
weaker model substitutes the adjacent material and sounds authoritative; the shipped model
reasons *"this source summarizes the work but is not its text"* and refuses with provenance.
That behavior was **achieved by measurement** — it's the headline result of the D5 model gate
(see appendix), not a lucky prompt.

## Production engineering

Trust is necessary but not sufficient — a buyer also needs to know it won't fall over or run
up a bill. What's shipped and demonstrable:

- **Two cost/quality modes**, user-chosen and measured end-to-end: a single-shot **fast path**
  (~$0.002–0.003/query) and an agent **deep mode** (~$0.01/query) for hard distributed
  questions. A per-query complexity router was *considered and rejected* — there was no
  measured gap to exploit once the strong model was the default, and a misroute is a
  regression surface.
- **Live per-request cost**, **per-IP rate limiting + per-session caps**, a **provider
  fallback chain**, and **Langfuse tracing** end-to-end (every eval result also carries a
  `trace_id`, so a failed question links straight to its trace).
- **A defense stack in code, not just prompt** — input blocklist + output validation +
  grounding gate — measured at 0% attack-success on a dedicated security eval tier.
- **Deployed** as a single Docker Compose project (SPA + FastAPI + Postgres/pgvector) behind
  an nginx edge on **AWS EC2**, mem-capped to co-tenant a shared box. Async end-to-end (SSE
  streaming; no sync I/O on the event loop).

**On the security claim, honestly:** the 40-prompt suite covers the common attack classes
(prompt-extraction, scope-escape, grounding-bypass, citation-forgery, fake-source-injection),
but it is *representative, not exhaustive* — the prompts are deliberately straightforward. A
determined adversary (novel multi-turn, obfuscated, or encoding-based injections) is out of
scope, and a production deployment with a real threat model would want more: dedicated
guardrail models, continuous red-teaming, output classifiers. That hardening is achievable
but expensive, and unnecessary for a portfolio demo. The honest claim is narrow and true —
the common attack classes are closed at 0%, with the defense living in **code and
architecture rather than prompt** (the part that actually generalizes). Going further is a
known, deliberately deferred cost, not a blind spot.

---

## What I rejected, and the bugs I caught

This section is the point. Anyone can list techniques they added; senior judgment shows in
what you *don't* add and the measurement errors you catch before they mislead you.

**Techniques rejected with receipts:**

- **Hybrid BM25 + RRF** — the 2024-era "hybrid always wins at scale" advice. Pre-registered as
  *expected to flip to a win* on this larger corpus. It didn't: recall@50 was **byte-identical**
  to dense, category by category. A strong embedder (qwen3-8b) already had every keyword-findable
  passage; where dense missed, the answer was *distributed*, not keyword-findable, so BM25
  couldn't reach it either. **Rejected** — and the falsified prediction is more valuable than a
  confirmed one.
- **Cross-encoder reranking** — only a SOTA hosted reranker beat the no-rerank baseline, and only
  by ~1.7 points at recall@5, for a permanent paid per-query dependency. **Dropped** at the model
  gate as marginal-and-paid.
- **A per-query routing classifier** and a **separate cheap served tier** — rejected as unmeasured
  complexity with a regression surface, once the cost evidence showed the strong model was cheap
  enough to be the default.

**Measurement bugs caught (each would have led to a wrong conclusion):**

- A recall metric that **penalized adding corroborating sources** — a perverse incentive against
  exactly the diligence the project values. Redesigned to per-requirement-group recall, with a
  control proving the fix was surgical (zero movement on the categories it shouldn't touch).
- A judge that scored **correct-but-miscited the same as invented** — separated misattribution
  from fabrication.
- An attribution metric so **noisy it couldn't be trusted** (re-scoring identical answers swung
  0.66/question with full 1↔5 flips). Fixed by routing attribution to a stronger, *stable* judge
  before any decision rode on it — noise dropped to 0.15/question.
- Network errors and rerank-API failures **silently counted as model refusals**, inflating the
  failure rate. A prompt-injection classifier that **counted a quoted falsehood the model was
  rejecting** as a successful attack. A judge prompt that **hard-coded a test question's answer**
  as its example (overfitting the judge to the test) — caught and replaced with a synthetic one.

The recurring lesson, stated in the project's own rules: *measurement bugs moved the numbers more
than real techniques did.* The eval harness is treated as production code — typed records,
versioned rubrics, tests — precisely so those bugs surface as bugs, not as "findings."

## Can the results be better? Yes — and here's the cost/benefit

The residual failure modes aren't mysterious; they're named and measured: **parametric
embellishment** (a grounded answer plus one unsourced-but-true specific), **cross-source
misattribution** (right facts, wrong source label), and a **~58% citation-recall ceiling**.
Each has a lever, and each lever has a price — naming the price is the point.

- **A verify-and-fix critic pass** — a second agent reviews and corrects the final answer (catching
  the unsourced specific or the mislabeled source) before it reaches the user. The most direct
  attack on both residuals. Cost: roughly **2× the query, forever.**
- **A stronger generator** — would chip at embellishment and misattribution; same forever-cost
  shape, paid per query.
- **A clean-writer arm** — compose the final answer from deduped, author-labelled evidence
  (scoped, not built). Targets misattribution specifically.
- **A retrieval-coverage / query-decomposition study** — lift the citation-recall ceiling. An
  *ingest-time* (one-time, cost-aligned) lever, and the cheapest of the four.

And — just as important — what would **not** help: **RAPTOR and GraphRAG.** Both are fashionable
and both are expensive to build, but the measurement diagnosed the hard-category headroom as the
agent's *distributed-evidence assembly*, not a retrieval-structure problem these solve. Building
them would be real effort spent against a bottleneck the data says isn't there — the same
discipline that rejected hybrid BM25.

None of these ships in the portfolio version on purpose: each is a real cost (query-time spend
forever, or build time) against residuals that are already small and well-characterized. Deciding
*not* to gold-plate — and being able to say exactly what the next dollar would buy — is itself part
of what the project demonstrates.

## What this demonstrates

The transferable skill set, shown rather than claimed:

- **Grounded generation with honest abstention** — the calibrated-trust property any high-stakes
  retrieval domain (legal, medical, compliance, internal support) needs before it can ship.
- **Decisions defended by data** — an ablation discipline that keeps a system from accreting
  unmeasured complexity, and a paper trail a reviewer can audit.
- **Production hardening** — cost control, rate limiting, fallback, tracing, a security tier, and
  a real deployment.
- **Measurement honesty** — the rarest of the four: knowing when a number is lying to you.

The domain here is antiquity, but nothing about the engineering is. The same harness, gates, and
trust properties transfer directly to any corpus where being *wrong* is worse than being silent.

---

## Appendix — the full arc (for the technical reader)

### Corpus & golden set

62 EU-public-domain works (Project Gutenberg; Herodotus to Gibbon), ~46,170 embedded chunks.
Golden set: 161 questions (135 in-scope + 26 out-of-scope), 440 gold spans defined as
*character spans in the cleaned source text* (chunking-invariant by design). Categories chosen
to isolate distinct failure modes: literal, synonym (modern English vs Victorian translation),
multi-hop, synthesis, cross-book, contradiction, and out-of-scope traps.

### Retrieval ablations

| Step | Change | Result |
|---|---|---|
| Baseline | naive dense, top-5 | recall@5 **35.2%** |
| **D2 gate** | embedder → qwen3-embedding-8b (1024d, Nebius) | **+18 recall@5 → 53%**; synonym +41.7 — *the embedder was the binding constraint* |
| Metric fix | per-span → per-requirement-group recall | measurement-honesty fix (isolated to redundant-source categories) |
| Full-set re-baseline | 135 in-scope questions | recall@5 **55%** |
| 4.1 contextual | embed `context_note + heading_path + text` | synthesis **+18.2 @5** (the worst category, transformed); a coupled attribution tax |
| 4.2 rerank | 5 cross-encoders measured | only SOTA hosted reranker beats no-rerank (+1.7), and it's paid → later **dropped** |
| 4.3 hybrid BM25 | dense + BM25 via RRF | **rejected** — recall@50 byte-identical; strong embedder subsumes BM25 |

**Shipped retrieval:** `dense-ctx-v1` — contextual dense embeddings, no hybrid, no rerank. The
remaining headroom on the hard categories was diagnosed as *architectural* (assembling
distributed evidence), which is the agent's job, not a ranking knob's.

### Generation & the agent

| Step | Change | Result |
|---|---|---|
| Baseline | single-shot, local 12B | faithful but incomplete; quality tracked the retrieval ceiling ~1:1 |
| Phase 5 agent | LangGraph grammar-ReAct (search / read / finalize, forced-finalize bound) | synthesis completeness 3.53 → **4.44**; multi-hop regressed; OOS abstention dropped — *model-strength limits, on the local model* |
| Split-judge calibration | route attribution to a stronger, stable judge | judge noise 0.66 → **0.15** /question — *unblocked the model gate* |
| **D5 gate** | agent reasoner → deepseek-v4-pro | **OOS abstention 73% → 100%**; attribution +0.36; completeness +0.28 |
| Prompt saga v3→v8 | five iterations | v3 rejected-with-receipt; v4 cut false-refusal **18.5% → 3.0%**; v6 showed *quote-pinning backfires* (induces fabrication); v8 froze the prompt at production quality |

**Why the prompt was frozen:** the v3→v8 sequence showed each edit sliding along a *fixed*
coverage/faithfulness frontier set by the model+retrieval, not moving it. The two residual
failure modes — parametric embellishment (one unsourced-but-true specific) and cross-source
misattribution — were demonstrated to be **model-strength limits, not prompt-reachable** (the
anti-embellishment prompt edit didn't move faithfulness and added a fabrication flavor). Knowing
when to *stop* tuning is itself a measured decision here.

### Stack & decision gates

FastAPI (async/SSE) · LangGraph grammar-ReAct agent · Postgres + pgvector · qwen3-embedding-8b
(hosted) · deepseek-v4-pro · split LLM-judge (kimi-k2.6 + qwen3.7-max) · Vite/React SPA ·
Langfuse · Docker Compose on AWS EC2. Gates: **D1** framework (LlamaIndex RAG layer + LangGraph
agent, behind a project-owned interface) · **D2** embedder · **D3** vector store · **D4**
frontend · **D5** LLM lineup. ADRs and full rationale live in [docs/adr/](docs/adr/) and
[project-plan.md](project-plan.md).
