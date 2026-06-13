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

---

## 2026-06-12 — D2 ablation arm 1: qwen3-embedding-8b hosted (dense-8b-1024-v1)

**Config:** qwen/qwen3-embedding-8b via OpenRouter ($0.01/M, verified 2026-06-12),
MRL-truncated 4096→1024 dims + L2-renorm · same chunks (structural-v1), same golden set,
bare-chunk embedding — directly comparable to the dense-v1 baseline. Corpus re-embed:
~$0.11, 83 min (hosted, batch 32).
**Run record:** `backend/evals/runs/2026-06-12T12-11-37Z-dense-8b-1024-v1.json`

| category | n | recall@5 (vs 0.6b) | recall@20 (vs 0.6b) | MRR (vs 0.6b) |
|---|---|---|---|---|
| literal | 11 | **83.3%** (69.7) | 87.9% (69.7) | 0.818 (0.447) |
| synonym | 10 | **71.7%** (30.0) | 85.0% (51.7) | 0.667 (0.431) |
| multi-hop | 10 | 23.3% (6.7) | 55.0% (21.7) | 0.287 (0.055) |
| synthesis | 10 | 19.2% (15.8) | 54.2% (29.8) | 0.338 (0.362) |
| cross-book | 10 | 44.2% (29.2) | 73.3% (52.5) | 0.480 (0.495) |
| contradiction | 11 | **72.7%** (54.5) | 90.9% (59.1) | 0.498 (0.427) |
| **overall** | **62** | **53.2% (35.2)** | **74.9% (48.0)** | **0.519 (0.372)** |

**Findings:**

1. **+18.0 recall@5 from the embedder alone** — bigger than the expected headline lever
   (contextual retrieval was +16 at small scale). Embedder quality was the binding
   constraint, not chunking or enrichment.
2. **Synonym +41.7 points (30.0→71.7)** — the 8B bridges modern-English questions to
   Victorian translations; this was the "synonym tax" and a big model largely pays it.
3. **recall@20 = 74.9%** transforms the Phase 4 plan: the rerank arm (4.2) now has a
   rich pool — contradiction @20 is 90.9% with @5 at 72.7%, classic rerank-bait.
4. **Caveats for the gate:** hosted (API dependency at query time — latency via
   OpenRouter to be compared against local CPU candidates), and this measures the 8B
   *truncated to 1024* — a 2048-dim arm is one env-var away if the gate gets close.
   Synthesis stayed flat at @5 (15.8→19.2) — distributed answers remain a
   retrieval-architecture problem, not an embedder problem.
5. **Open:** local CPU-class candidates (voyage-4-nano, gte-modernbert-base) still
   unmeasured — the gate question is now "does any local model get close enough to
   53.2% to justify zero API dependency?"

---

## 2026-06-12 — D2 arm 2: dims + provider pinning (dense-8b-2000-nebius-v1)

**Why:** (a) quantify MRL truncation loss (4096-native model; pgvector HNSW caps at
2000 dims, so 2000 is the max indexable size); (b) pin one provider — the unpinned arm-1
corpus was embedded by OpenRouter's provider mix (incl. an fp8 endpoint at 70% uptime,
the source of 5–8s latency spikes; probes: Nebius ~0.9s consistent, DeepInfra 4–7s).
**Config delta vs arm 1:** dims 1024→2000, provider pinned to Nebius. (Two changes at
once — attribution confounded by design; the gate ships a config, not an attribution.)
**Run record:** `backend/evals/runs/2026-06-12T14-11-20Z-dense-8b-2000-nebius-v1.json`

| | recall@5 | recall@10 | recall@20 | MRR |
|---|---|---|---|---|
| arm 1 (1024, unpinned) | 53.2% | 61.8% | 74.9% | 0.519 |
| arm 2 (2000, Nebius) | **54.3%** | 63.0% | 75.8% | 0.529 |

**Findings:** +1.1 recall@5 — at/below the ±1-question noise floor. **MRL truncation
4096→1024 is effectively free on this corpus**; the provider mix didn't measurably hurt
arm 1 either. Contradiction recall@20 reached 100% (every contradiction question now has
its evidence inside a top-50 rerank pool). Per the pre-stated rule (within noise → ship
the smaller vectors), the gate-final config re-embeds at **1024 dims, Nebius-pinned** —
half the storage (~120MB vectors, comfortable in a 500MB free-tier budget).

---

## 2026-06-12 — Gate D2 CLOSED: dense-8b-1024-nebius-v1 is the new retrieval floor

**Run record:** `backend/evals/runs/2026-06-12T15-21-10Z-dense-8b-1024-nebius-v1.json`
Gate-final config (qwen3-8b · Nebius-pinned · 1024d) reproduced arm 1 category-for-
category: **53.2% recall@5 · 74.9% recall@20 · MRR 0.522**, query embed ~0.9s consistent.
Decision + full rationale: [ADR-002](adr/002-d2-embeddings.md). Total gate spend ≈ $0.36.

**This row replaces dense-v1 (35.2%) as the Phase 4 comparison floor.** Generation-tier
re-baseline on the new corpus is pending (the old gen baseline was measured on 0.6b
retrieval) — run before the 4.1 contextual arm so generation deltas stay attributable.

---

## 2026-06-12 — Generation re-baseline on D2 corpus (gen-dense-8b-judged)

**Config:** identical generation pipeline (gemma-12b-16k, prompt baseline-v1, top-5,
judge-v2 deepseek-v4-flash) — only retrieval changed (dense-v1 0.6b → dense-8b-1024-nebius).
**Run record:** `backend/evals/runs/2026-06-12T16-56-58Z-gen-dense-8b-judged.json`

| metric | on 0.6b retrieval | on 8b retrieval | Δ |
|---|---|---|---|
| citation recall | 32.0% | **49.1%** | +17.1 |
| citation precision | 54.1% | 52.7% | ~flat |
| faithfulness (judge-v2) | 4.72 | 4.79 | noise |
| completeness | 3.07 | **3.61** | **+0.54** |
| false refusal rate | 25.8% | **8.1%** | −17.7 (15→5 questions) |
| OOS refusal accuracy | 100% | **100%** | held |
| mean latency | 2.8s | 5.4s | +2.6s (hosted embed + longer answers) |

**Findings:**

1. **Retrieval gains converted ~1:1 into cited-answer gains** (+18.0 retrieval recall@5 →
   +17.1 citation recall) — the Phase 4 premise, now measured twice from opposite sides.
2. **The honest-refusal hypothesis confirmed:** false refusals collapsed 25.8%→8.1% with
   ZERO prompt/model changes — those refusals were retrieval failures wearing a polite
   mask, exactly as the Phase 3 baseline entry predicted. Literal/synthesis/cross-book/
   contradiction now answer 100% of in-scope questions.
3. **Completeness +0.54 (3.07→3.61)** — well above the ±0.2 judge noise floor. Better
   sources = fuller answers, same model. Remaining gap is concentrated where retrieval
   still misses: multi-hop (3.33, still 4 refusals, retrieval@5 23.3%) and
   synthesis/cross-book (compl 3.00) — the 4.1/4.2/RAPTOR targets.
4. **Abstention contract intact:** 10/10 OOS refusals with richer (more tempting)
   wrong-context sources — the trust property survived the retrieval upgrade.
5. **Latency cost is real:** mean 5.4s/answer (hosted query embed ~1s + longer, fuller
   answers at 138 mean completion tokens). The Phase 6 router/caching arms own this.

---

## 2026-06-13 — Recall redesign: per-requirement-group, not per-span (measurement fix)

**Why (measurement-bug class, rule #5):** the corpus grew with many corroborating sources,
so a single fact now recurs across works (Caesar's 23 wounds: Suetonius, Appian, Plutarch,
Livy, Smith). The old recall counted **every** gold span as independently required
(`covered_spans / total_spans`), so a literal question with 5 equivalent attestations
capped at 0.2 even when retrieval surfaced the one best passage. Worse, it created a
**perverse incentive**: adding more corroborating sources — exactly the diligence this
project values — *lowered* the score (bigger denominator, same coverage). MRR already did
the right thing (any-of), so MRR and recall disagreed, and recall was the broken one.

**Change:** recall is now per **requirement group**, not per span (docs/golden-set.md §4a).
Each span declares a `groups` list — the answer requirement(s) it satisfies. Spans sharing
a label are *alternatives* (any one covers the requirement); distinct labels are
*conjunctive*; a span may satisfy several at once (a chunk answering two hops). Empty
`groups` = a singleton requirement (the back-compatible default). Recall@k = requirements
with ≥1 covering span in top-k / total requirements. Grouping applied to `literal`/`synonym`
(alternatives → one requirement), `multi-hop` (per-hop), `contradiction` (per-version);
`cross-book`/`synthesis` left ungrouped — there each load-bearing span is a genuine coverage
target. Golden set also grew this day: **154 → 240 gold spans** (new public-domain sources),
72 questions unchanged.
**Isolation:** scored the *identical current set* on *one fresh retrieval* both ways — the
delta below is the metric alone, zero retrieval change.
**Config:** qwen3-embedding-8b · Nebius-pinned · 1024d · structural-v1 · naive dense top-20,
no rerank (the D2 gate-final retrieval; query embed ≈ 0.9s). Run cost ≈ $0.00 (62 short
query embeds).
**Run record:** `backend/evals/runs/2026-06-13T11-22-00Z-dense-v1.json` (retriever label is
the default `dense-v1`; embedder field confirms the 8b corpus).

| category | n | old r@5 (per-span) | new r@5 (per-group) | Δ@5 | new r@1 | new r@20 | new MRR |
|---|---|---|---|---|---|---|---|
| contradiction | 11 | 0.647 | **0.841** | +0.194 | 18.2% | 95.5% | 0.568 |
| literal | 11 | 0.726 | **0.909** | +0.183 | 72.7% | 90.9% | 0.818 |
| synonym | 10 | 0.750 | **0.900** | +0.150 | 60.0% | 90.0% | 0.678 |
| multi-hop | 10 | 0.273 | **0.350** | +0.077 | 25.0% | 70.0% | 0.522 |
| cross-book | 10 | 0.463 | 0.463 | +0.000 | 12.3% | 75.0% | 0.725 |
| synthesis | 10 | 0.167 | 0.167 | +0.000 | 5.0% | 44.2% | 0.319 |
| **overall** | **62** | **0.510** | **0.614** | **+0.103** | **32.6%** | **78.1%** | **0.608** |

**The three-number story (set growth × metric, both isolated):**
D2 floor (old set 154 spans, old metric) **53.2%** → current set (240 spans, old metric)
**51.0%** → current set, new metric **61.4%**. The middle step is the perverse incentive in
the raw: adding corroborating sources *dropped* per-span recall 2.2pp. The redesign both
reverses that and re-bases the floor.

**Findings:**

1. **The lift is the measurement getting honest, not retrieval improving.** +10.3 overall
   recall@5, landing *only* on the redundant-source categories (contradiction +19.4,
   literal +18.3, synonym +15.0) and **exactly 0.0** on the two ungrouped coverage
   categories. cross-book and synthesis are byte-identical old-vs-new — the control that
   proves the change is surgical, not a global inflation.
2. **Effect concentrates at low k** (+10.3 @5 vs +7.0 @20) — where it matters, since the
   generator is fed top-5. By @20 even per-span recall catches the redundant alternatives.
3. **The honest metric exposes three real full-misses** previously blended into the
   redundant-source noise, all single-requirement questions, all on ingested works (153/
   463/275 chunks present — not ingestion bugs): **syn-006** (Cambyses "sacred disease" —
   synonym gap the 8b doesn't bridge), **lit-004** (retrieves Tacitus' Germania at rank 2
   but the wrong chunk — chunk-granularity/rerank), **mh-005** (both hops missed). These
   are clean Phase-4 rerank/chunking targets.
4. **multi-hop's gains are real but partial** (+7.7 @5): grouping per-hop means a question
   that finds one hop's passage now scores 0.5 honestly instead of a diluted per-span
   fraction; several sit at exactly 0.5 @20 (one hop found, one not) — still the hardest
   single-retrieval category.
5. **synthesis/cross-book unchanged and still hard** (synthesis @5 0.167, the rag-historian
   prior): distributed answers are a retrieval-architecture problem, untouched by a metric
   that was never wrong for them.
6. **Three grouping judgment calls flagged for review** in the YAML (`GROUPING JUDGMENT`):
   syn-005 (wave + its cause as one requirement), con-005 (Rawlinson tagged both versions),
   con-010 (Lycurgus meta-contradiction facet split).

**Phase 4 measures retrieval against THIS row** (per-group, 240-span set): recall@5 **61.4%**
· recall@20 78.1% · MRR 0.608. The old per-span dense-8b row (53.2%) is superseded — it
measured a different set with a metric that mismodeled redundancy. Records carry per-span
`groups` so a future per-span vs per-group reading can't be silently confused.
