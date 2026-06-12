# Eval Log — append-only record of measured results

> One entry per significant eval run or finding. Raw run records (per-question
> forensics) live in `backend/evals/runs/*.json`; this log is the human-readable
> narrative over them. This file is case-study source material — keep entries
> honest, dated, and tied to a run record.

---

## 2026-06-11 — Baseline: dense-v1 (Phase 2/3 floor)

**Config:** qwen3-embedding-0.6b (local, provisional pending D2) · chunking structural-v1
(500/50, division walls) · 30,187 chunks / 38 works (~11M tokens) · naive dense top-k,
no rerank, no enrichment · golden set v2.0: 72 questions, 154 gold spans, all human-reviewed.
**Run record:** `backend/evals/runs/2026-06-11T18-23-15Z-dense-v1.json`

| category | n | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---|---|---|---|---|---|
| literal | 11 | 9.1% | 69.7% | 69.7% | 69.7% | 0.447 |
| synonym | 10 | 20.0% | 30.0% | 46.7% | 51.7% | 0.431 |
| multi-hop | 10 | 0.0% | 6.7% | 11.7% | 21.7% | 0.055 |
| synthesis | 10 | 5.0% | 15.8% | 25.8% | 29.8% | 0.362 |
| cross-book | 10 | 14.2% | 29.2% | 29.2% | 52.5% | 0.495 |
| contradiction | 11 | 13.6% | 54.5% | 54.5% | 59.1% | 0.427 |
| **overall** | **62** | **10.3%** | **35.2%** | **40.3%** | **48.0%** | **0.372** |

**Findings:**

1. **Reproducibility across scale: overall recall@5 = 35.2% vs rag-historian's naive
   baseline 35.1%** — same architecture (naive dense, 500-token chunks), measured on a
   corpus ~30× larger (38 works vs 4) with a fresh, independently-authored question set
   and a different embedder. The naive-dense floor appears to be a property of the
   architecture, not the corpus. Strong validation that both eval harnesses measure the
   same thing.
2. **Multi-hop is the catastrophe (6.7% @5, MRR 0.055):** one query embedding can't chase
   two facts — each hop's passage competes with the other's in a single similarity
   ranking. Known fixes, in our Phase 4/5 plan: agent loop (multi-step search), query
   decomposition, RAPTOR.
3. **Cross-book is rerank-bait:** 52.5% @20 vs 29.2% @5 — the right passages reach the
   pool but rank too low. The clearest "known fix" signal in the table (cross-encoder
   rerank of top-50).
4. **Literal plateaus after k=5** (69.7% at @5=@10=@20): misses are vocabulary/entity
   misses, not ranking misses — contextual retrieval's target.
5. **Synonym tax measured:** 30% @5 — modern-English questions vs Victorian translations.
6. **Synthesis confirms the prior** (15.8% @5; was 18.7% on the small corpus) —
   distributed answers stay retrieval-hard at every scale.

**Next levers (Phase 4 order, per docs/rag-techniques.md):** contextual retrieval →
cross-encoder rerank → hybrid BM25/RRF re-test → agent loop (Phase 5) → RAPTOR.

---

## 2026-06-11 — Refactor verification: retrieval promoted to shared module (Phase 3.1)

`dense_retrieve` moved from the eval harness into `ahx/retrieval/dense.py` (one
implementation now shared by evals, CLI, and the upcoming API; sync/async variants share
one SQL builder). Re-run produced **identical numbers to the baseline above in every
cell** — refactor confirmed a behavioral no-op.
**Run record:** `backend/evals/runs/2026-06-11T19-04-29Z-dense-v1.json`

Same day, run records restructured to the proven rag-historian layout: aggregates
(overall + by-category) at the top, then per-question results with question text, ideal
answer, gold_chunk_ids vs retrieved_chunk_ids, similarities, latency_ms, recall@k, MRR.
Numbers re-verified identical at each step:
`backend/evals/runs/2026-06-11T19-54-52Z-dense-v1.json` is the first record in the final
format (the 19-35-45Z record is an intermediate verbose format, superseded).

---

## 2026-06-12 — Baseline: gen-baseline-v1 (Phase 3 generation floor, mechanical tier)

**Config:** full ask pipeline — dense-v1 retrieval (top-5) → prompt baseline-v1 →
gemma-12b-16k (local llama-swap, temp 0) → structured citations · golden set v2.0,
all 72 questions (out-of-scope measurable for the first time) · mechanical metrics only,
**no judge** (judge layer implemented but gated on a configured strong judge — AHX_JUDGE_*).
**Run record:** `backend/evals/runs/2026-06-12T08-26-35Z-gen-baseline-v1.json`

| category | n | refused | cit recall | cit precision | mean latency |
|---|---|---|---|---|---|
| literal | 11 | 2 | 65.2% | 75.9% | 1.6s |
| synonym | 10 | 4 | 25.0% | 58.3% | 2.5s |
| multi-hop | 10 | 6 | 6.7% | 50.0% | 1.6s |
| synthesis | 10 | 1 | 15.8% | 41.7% | 3.2s |
| cross-book | 10 | 0 | 29.2% | 34.3% | 5.2s |
| contradiction | 11 | 2 | 45.5% | 47.2% | 2.9s |
| out-of-scope | 10 | 10 | — | — | 1.3s |
| **overall** | **72** | | **32.0%** | **50.6%** | **2.6s** |

cit recall = gold spans covered by chunks the model *cited* (not merely retrieved);
cit precision = used markers pointing at a gold-covering chunk. False refusal rate
(in-scope): **24.2%** (15/62). Refusal accuracy (out-of-scope): **100%** (10/10, all with
the exact contract sentence). Mean completion: 124 tokens.

**Findings:**

1. **The abstention contract works at temperature 0:** 10/10 out-of-scope refusals,
   verbatim sentence, zero hallucinated answers — the headline trust number for the
   case study, measured on day one of the generation tier.
2. **Citation recall (32.0%) sits just under retrieval recall@5 (35.2%) — retrieval is
   the ceiling.** Generation loses only ~3pp by citing the wrong subset of what it was
   given. Confirms Phase 4's premise: retrieval gains should convert ~1:1 to cited-answer
   gains.
3. **False refusals concentrate exactly where retrieval fails:** multi-hop 6/10 refused
   (retrieval recall@5 there: 6.7%), synonym 4/10 (30%). These are *honest* refusals —
   the sources shown to the model genuinely lacked the answer. The visible cost of bad
   retrieval at the generation tier is silence, not hallucination; Phase 4 retrieval work
   should convert refusals into cited answers.
4. **Citation precision 50.6%:** half the markers point at non-gold chunks. Partly
   generous citing (cross-book answers often cite all 5 sources → 34.3% precision),
   partly gold-adjacent context. A judge tier is needed to separate "wrong citation"
   from "correct citation for a correct non-gold-path claim" — mechanical precision
   undercounts by design.
5. **Cost profile:** mean 2.6s/answer, ~124 completion tokens on a local 12B at 16k
   context. Cross-book is slowest (5.2s — longer answers citing many sources).
6. **Judge tier (faithfulness/completeness vs ideal answers) is pending** a configured
   strong judge model (D5 decision: which hosted frontier model + key). Harness, rubrics
   v1, and record fields are in place; this entry's row gets a judged companion at the
   phase boundary.

---

## 2026-06-12 — Judged baseline: gen-baseline-v1-judged (Phase 3 exit row)

**Config:** identical pipeline to gen-baseline-v1 above · judge = **deepseek/deepseek-v4-flash
via OpenRouter** ($0.098/M in, $0.196/M out, verified 2026-06-12; full run ≈ $0.03) ·
rubrics v1 (faithfulness vs cited sources, completeness vs ideal_answer, 1-5).
**Run record:** `backend/evals/runs/2026-06-12T09-27-11Z-gen-baseline-v1-judged.json`

| category | n | refused | cit recall | cit precision | faith | compl |
|---|---|---|---|---|---|---|
| literal | 11 | 2 | 65.2% | 75.9% | 5.00 | 4.33 |
| synonym | 10 | 4 | 25.0% | 58.3% | 4.33 | 3.67 |
| multi-hop | 10 | 7 | 6.7% | 100.0%* | 5.00* | 1.67* |
| synthesis | 10 | 1 | 15.8% | 41.7% | 3.44 | 2.67 |
| cross-book | 10 | 0 | 29.2% | 39.3% | 4.80 | 2.78 |
| contradiction | 11 | 2 | 45.5% | 53.1% | 4.56 | 3.33 |
| out-of-scope | 10 | 10 | — | — | — | — |
| **overall** | **72** | | **32.0%** | **54.1%** | **4.48** | **3.22** |

\* multi-hop judged on n=3 answered questions — noise.

**Findings:**

1. **Faithful but incomplete — the baseline's character in two numbers: faith 4.48,
   compl 3.22.** When gemma answers, it sticks to the sources; what it lacks is
   material. Completeness craters exactly in the distributed-answer categories
   (multi-hop 1.67, synthesis 2.67, cross-book 2.78) — single-shot top-5 stuffing
   cannot assemble what retrieval didn't co-locate. Matches the human read
   ("mostly incomplete but still answers").
2. **Judge quality (flash-tier risk check):** verdicts cite specific, checkable
   evidence — e.g. catching answers that attribute Cassius Dio's text to Plutarch
   (synth-001), and fabricated bridge-construction details (syn-007). 1 parse failure
   in 92 calls. Human spot-check of lowest/highest verdicts found them defensible;
   full calibration vs a frontier judge deferred until a decision rides on a small
   difference.
3. **Re-run variance at temperature 0 is not zero:** between the two same-config runs,
   false refusals moved 15→16 and one answer dropped its marker (con-003) —
   llama.cpp batching nondeterminism. Treat single-question effects as noise floor in
   Phase 4 ablations; only multi-question movements are signal.
4. **Phase 3 exit satisfied:** API streams cited answers (verified live); retrieval +
   generation baseline rows locked. ~~Phase 4 measures against this row~~ — superseded
   same day by the judge-v2 rejudge below (measurement fix, not a pipeline change).

---

## 2026-06-12 — Judge rubric v2: misattribution ≠ fabrication (rejudge of frozen answers)

**Why (measurement-bug class, rule #5):** external verification of the two faithfulness=1
verdicts above showed both answers were *grounded but miscited* — syn-007's "unsupported"
details sat verbatim in retrieved-but-uncited Herodotus; synth-001 presented Cassius Dio's
text as "Plutarch portrays". Under judge-v1 the judge saw only **cited** chunks, so
correct-but-miscited scored identical to invented — double-counting what
citation_precision already measures (same failure family as rag-historian's judge
punishing answers that out-sourced the gold).

**Change (judge-v2):** the faithfulness judge sees ALL retrieved passages exactly as the
answer model saw them (numbered, authors visible, cited ones flagged); grounded content
with wrong marker/author caps at 4; invention drives 1-3. Completeness rubric unchanged.
**Isolation:** new `ahx eval rejudge` re-scored the FROZEN answers of the judged baseline —
zero generation drift; only the judge moved.
**Run record:** `backend/evals/runs/2026-06-12T10-06-34Z-gen-baseline-v1-judge-v2.json`

| metric | judge-v1 | judge-v2 | reading |
|---|---|---|---|
| faithfulness | 4.48 | **4.72** | misattribution reclassified out of "fabrication" |
| completeness | 3.22 | **3.07** | rubric unchanged → ±0.15 ≈ judge noise floor |

**Verdict movements (same answers):** syn-007 1→5 (details were in uncited [2]/[3]);
synth-001 1→4 (Dio-as-Plutarch correctly capped as misattribution); synth-010 3→**1**
(sharper, not noisier: "support" was a passage about Tiberius *Gracchus*, not the emperor
— wrong-person content v1 couldn't see); con-003 1 (answer claimed sources lack the info
while retrieved [4] contained it). The low-faithfulness set now isolates genuine failures.

**Phase 4 measures against THIS row:** cit recall 32.0% · cit precision 54.1% ·
faithfulness 4.72 (judge-v2) · completeness 3.07 · refusal accuracy 100% ·
false refusals 25.8%. Judge-score deltas under ~0.2 are noise; records carry
`judge_rubric` so v1/v2 scores can't be silently compared.
