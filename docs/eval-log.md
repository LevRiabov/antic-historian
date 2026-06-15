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

---

## 2026-06-13 — Retrieval re-baseline on the full golden set (dense-v1, 135 in-scope)

**Why:** the golden set reached its final size (161 questions / 440 gold spans, all spans
resolving) — re-measure the naive-dense floor on the complete set before any Phase 4
technique lands, so the comparison row reflects the corpus and questions Phase 4 will
actually be judged on. Same retrieval config as the D2 gate-final row; only the question
set grew (62 → 135 in-scope; out-of-scope excluded from the retrieval tier by design).
**Config:** qwen3-embedding-8b · Nebius-pinned · 1024d · structural-v1 · naive dense top-20,
no rerank · per-requirement-group recall. Run cost ≈ $0.00 (135 short query embeds).
**Run record:** `backend/evals/runs/2026-06-13T18-31-18Z-dense-v1.json`

| category | n | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---|---|---|---|---|---|
| literal | 23 | 65.2% | 87.0% | 95.7% | 95.7% | 0.757 |
| synonym | 23 | 52.2% | 91.3% | 95.7% | 95.7% | 0.665 |
| multi-hop | 24 | 16.7% | 31.2% | 45.8% | 52.1% | 0.415 |
| synthesis | 18 | 2.8% | 13.4% | 16.2% | 30.1% | 0.223 |
| cross-book | 28 | 5.3% | 34.4% | 47.6% | 62.2% | 0.460 |
| contradiction | 19 | 15.8% | 72.4% | 80.3% | 86.8% | 0.548 |
| **overall** | **135** | **26.7%** | **55.0%** | **64.1%** | **71.0%** | **0.519** |

**Findings:**

1. **The floor held at scale.** Overall recall@20 is unchanged from the 62-question
   per-group row (78.1% → here 71.0% — the drop is the harder, larger synthesis/cross-book
   tails, now 18+28 questions vs 10+10) and recall@5 stayed in band (61.4% → 55.0% as the
   weak categories doubled in weight). Doubling the question count introduced no broken
   questions and did not move the architecture's signature — the baseline is stable.
2. **The split is the same story, sharper:** single-fact lookup is strong (literal/synonym
   ~87–91% @5, plateauing by @10), distributed answers are the floor (synthesis 13.4% @5,
   multi-hop 31.2%, cross-book 34.4%). cross-book stays rerank-bait (34.4% @5 → 62.2% @20).
3. **literal/synonym misses triaged (5 total), all real, three distinct classes** —
   the residue behind the 87–91%:
   - **lit-004** (Tacitus, "greatest disgrace… abandoned their shields"): lexical gap,
     *not retrieved at all* (rank None @20). The answer passage's signal is
     "shields"/"reproach and infamy", the query says "greatest disgrace" — dense can't
     bridge it. → hybrid BM25 + rerank.
   - **lit-018 / lit-021 / syn-018** (rank 7–10, recovered @20): correct chunk in the pool
     but outranked by same-book passages on the same topic. → cross-encoder rerank.
   - **syn-006** (Cambyses "sacred disease"): chunk-boundary split. The identifying subject
     ("Cambyses had from his birth") and the keyword ("sacred disease") landed in adjacent
     chunks at the 535217 boundary; the keyword chunk (277) retrieved at rank 1 but *lacks
     the subject*, the full-answer chunk (276) missed top-20. → chunk overlap / boundary
     tuning. (Same syn-006 first surfaced in the 06-13 recall-redesign entry; mechanism now
     pinned to the chunk wall, not just the synonym gap.)

**Phase 4 retrieval ablations measure against THIS row** (full 135-question set, per-group):
recall@5 **55.0%** · recall@20 71.0% · MRR 0.519.

---

## 2026-06-13 — Generation re-baseline on the full set, judged (gen-baseline-v2-judge-v3.1)

**Config:** full ask pipeline — dense-8b-1024-nebius retrieval (top-5) → prompt
**baseline-v2** (instructs the model to surface disagreement / attribute multi-source
synthesis) → gemma-12b-16k (local llama-swap, temp 0) → structured citations · judge =
deepseek/deepseek-v4-flash via OpenRouter, **rubric judge-v3.1** (faithfulness +
completeness + attribution, 1-5; semantic-refusal yes/no). Golden set: all **161 questions**
(135 in-scope + 26 out-of-scope). First judged generation row on the full set and on the
baseline-v2 prompt + judge-v3.1 rubric — supersedes the 72-question judged rows above as the
Phase 4 generation floor. Run ≈ 33 min; judge cost ≈ $0.08 (⚠ estimate — deepseek-v4-flash
$0.098/M in · $0.196/M out verified 2026-06-12; judge token counts not stored in the record).
**Run record:** `backend/evals/runs/2026-06-13T19-52-16Z-gen-baseline-v2-judge-v3.1-full.json`

| category | n | refused | refusal-OK | cit recall | cit prec | faith | compl | attrib |
|---|---|---|---|---|---|---|---|---|
| literal | 23 | 2 | 91.3% | 74.9% | 63.8% | 5.00 | 4.90 | 5.00 |
| synonym | 23 | 4 | 82.6% | 58.0% | 65.5% | 5.00 | 4.89 | 4.26 |
| multi-hop | 24 | 12 | 50.0% | 11.0% | 25.0% | 4.92 | 4.67 | 5.00 |
| synthesis | 18 | 3 | 83.3% | 13.4% | 18.0% | 4.87 | 3.53 | 3.67 |
| cross-book | 28 | 0 | 100.0% | 33.5% | 27.6% | 4.93 | 4.21 | 3.93 |
| contradiction | 19 | 0 | 100.0% | 58.5% | 46.8% | 4.58 | 4.42 | 4.05 |
| out-of-scope | 26 | 22 | 84.6% | — | — | — | — | — |
| **overall** | **161** | | **84.6%** oos / **15.6%** false-refusal | **41.6%** | **42.3%** | **4.89** | **4.45** | **4.28** |

refusal-OK = `refusal_correct` (in-scope: correctly answered; oos: correctly refused, semantic
judge). false refusal rate (in-scope) **15.6%** · mean completion **204 tokens** · mean
latency **6.2s**.

**Findings:**

1. **Faithful at baseline, before any optimization: 4.89 overall, ≥4.87 in every in-scope
   category.** The generator passes retrieval through honestly — it does not invent. This is
   the headline trust number and it is already strong with naive retrieval and a local 12B.
2. **Quality is retrieval-bound, measured a third time.** Citation span recall (41.6%) tracks
   the retrieval ceiling, and completeness moves with it cell-by-cell: literal (cit-recall
   74.9%) → compl 4.90; synthesis (cit-recall 13.4%) → compl 3.53. Generation adds no new
   loss beyond what retrieval withheld. Phase 4 retrieval gains should convert ~1:1 into
   completeness, as the prior re-baseline showed for citation recall.
3. **Attribution is the one generation-side lever** (overall 4.28; weak exactly on the
   multi-source categories — synthesis 3.67, cross-book 3.93, contradiction 4.05; perfect
   where one source suffices — literal 5.00, multi-hop 5.00). baseline-v2 asks for
   disagreement-surfacing and per-source attribution; the model under-delivers when it must
   weave several sources. Partly independent of retrieval — a prompt/generation target.
4. **False refusals are honest refusals, concentrated where retrieval starves.** multi-hop
   refused 12/24 (cit-recall 11%) — when the chain isn't co-located the model abstains
   rather than guess. The 15.6% in-scope false-refusal rate is a retrieval-coverage symptom,
   not a generation defect; fix retrieval (decomposition / multi-query) and these convert to
   cited answers.
5. **The abstention contract holds — except on the hardest source-absent traps.** Refusal
   behavior, by OOS subtype: far-from-corpus 10/10, false-premise 8/8 (the semantic judge
   correctly credits premise-corrections-with-citations as refusals — e.g. oos-014 cites
   markers yet scores refusal-correct because the answer corrects "Augustus was assassinated"),
   but **source-absent only 4/8**. The 4 misfires (oos-019/021/023/024, the
   "GENUINE ANTIQUITY, SOURCE ABSENT" group) re-audited as **correctly authored, model
   failures** — see below.

**OOS re-audit — the 4 source-absent misfires (no golden-set change):**

The trap these questions are built for ("the named primary source is absent, but a
*secondary* source discusses it — must refuse, not substitute") caught a real abstention
gap. Cited-chunk forensics:

- **oos-024 (Pliny's Natural History)** — worst: the answer lists the NH's structure
  ("the heavens, the elements, the stars, planets and their orbital periods") but that text
  is *not in any cited chunk* (Smith on Seneca; Suetonius on "natural knowledge"). The run's
  one genuine faithfulness breach — parametric knowledge with irrelevant markers attached.
- **oos-019 (Plato's Republic)** / **oos-023 (Sappho)** — substitution: Bury's account of
  Plato's failed political venture *in Syracuse* dressed up as "the Republic's ideal state";
  a one-line Grote Greek footnote ("impassioned love-songs") dressed up as "what Sappho's
  poetry expresses." Adjacent material about the figure, presented as the absent work.
- **oos-021 (Behistun inscription)** — borderline, flagged for review: the cited Rawlinson
  passage genuinely reports the exact deeds the inscription records (Gobryas/Susiana, the
  Sacae) — Rawlinson decoded Behistun. Only the "via the inscription itself" framing makes
  it out-of-scope; the content overlap may make this question too subtle.

**The precise gap:** on source-attributed questions ("what does X's *work* say…") the model
does not distinguish "the corpus contains X's text" from "the corpus mentions X." A
refusal-policy clause (refuse unless the named work is itself among the retrieved passages)
is a clean Phase 4 generation lever. Authoring verdict: keep all four; review oos-021.

**Phase 4 generation ablations measure against THIS row** (161 questions, baseline-v2 prompt,
judge-v3.1): faithfulness **4.89** · completeness **4.45** · attribution **4.28** ·
citation recall 41.6% · citation precision 42.3% · OOS refusal accuracy 84.6% · in-scope
false-refusal 15.6%. The 72-question judged rows are superseded (different set, pre-baseline-v2
prompt, pre-v3.1 rubric); records carry `prompt_version` + `judge_rubric` so they can't be
silently compared.

---

## 2026-06-14 — Phase 4.1 contextual retrieval: dense-ctx-v1 + gen-ctx-v1 (the big arm)

**What changed (one variable):** the chunk's *retrieval representation*. Each chunk's
embedding now covers `context_note + heading_path + chunk_text` instead of bare text —
a 1–2 sentence LLM situating note + `Author, Title > BOOK > chapter` prefix. Notes
generated locally by **gemma-4-12B** (`ahx ingest enrich`, `enrichment_version=enrich-v1`,
46,159/46,170 chunks; 11 enumeration-dense index/catalog chunks fall back to bare text).
Everything else identical to the floors: qwen3-8b/Nebius/1024d, structural-v1, per-group
recall; generation = gemma-12b-16k, baseline-v2 prompt, top-5, judge deepseek-v4-flash
v3.1. **Generation reads the original `text`, never the note** (verified: dense retriever
returns `chunk.text`) — so generation deltas are purely a *chunk-selection* effect.
**Run records:** `2026-06-14T14-59-55Z-dense-ctx-v1.json` (retrieval),
`2026-06-14T14-57-37Z-gen-ctx-v1.json` (generation, judged).

### Retrieval — dense-ctx-v1 vs dense-v1 floor (135 in-scope, per-group)

| category | n | recall@5 | Δ@5 | recall@1 | Δ@1 | recall@20 | MRR | ΔMRR |
|---|---|---|---|---|---|---|---|---|
| literal | 23 | 95.7% | **+8.7** | 69.6% | +4.4 | 95.7% | 0.789 | +0.03 |
| synonym | 23 | 91.3% | +0.0 | 60.9% | +8.7 | 100.0% | 0.726 | +0.06 |
| multi-hop | 24 | 33.3% | +2.1 | 20.8% | +4.1 | 52.1% | 0.484 | +0.07 |
| synthesis | 18 | 31.6% | **+18.2** | 12.4% | +9.6 | 53.9% | 0.444 | +0.22 |
| cross-book | 28 | 25.4% | **−9.0** | 6.1% | +0.8 | 60.5% | 0.412 | −0.05 |
| contradiction | 19 | 67.1% | **−5.3** | 26.3% | +10.5 | 86.8% | 0.639 | +0.09 |
| **overall** | **135** | **56.7%** | **+1.7** | **32.5%** | **+5.8** | **74.5%** | **0.579** | **+0.060** |

### Generation — gen-ctx-v1 vs gen-baseline-v2 floor (161 questions)

| metric | floor | ctx | Δ |
|---|---|---|---|
| faithfulness | 4.89 | 4.81 | −0.08 (noise) |
| completeness | 4.45 | 4.30 | −0.15 (see finding 1) |
| attribution | 4.28 | 3.98 | **−0.30** (see finding 2) |
| citation recall | 41.6% | 43.8% | +2.2 |
| citation precision | 42.3% | 43.9% | +1.6 |
| in-scope false-refusal | 15.6% | **7.4%** | **−8.2pp** |
| OOS refusal accuracy | 84.6% | 84.6% | held |

**Findings:**

1. **The "completeness drop" is a selection effect, not a regression (Simpson's paradox).**
   On the **113 questions both runs answered**, completeness is flat (4.46→4.40), faithfulness
   flat (4.89→4.87), citation recall flat (47.9%→48.1%). What changed: ctx **converted 12
   previously-refused questions into answers** (refusals 21→10; 12 newly-answered, 1 newly-
   refused), and those 12 — the retrieval-starved hard ones (5 multi-hop, 3 synthesis, 3
   synonym, 1 literal) — score only **3.42** completeness, dragging the *average* down while
   every prior answer held. Answering 12 refusals as cited answers is a win wearing the
   disguise of a regression.

2. **The attribution drop (−0.30, real on the apples-to-apples set) is the cross-book/
   contradiction retrieval regression propagating downstream — not a generation defect.**
   **All 24 questions with a ≥2-point attribution drop had a changed top-5 retrieval set
   (24/24).** Mechanism, from the judge's reasons: contextual retrieval reshuffles *which*
   chunks reach top-5 → a different, often richer multi-source set → on source-attributed
   questions the model **conflates which source said what** (cb-016 reversed Scipio
   preservation→destruction vs the new [1]; con-001 tagged Suetonius' content as Plutarch's;
   lit-006 cross-wired two epitaph versions across Herodotus/other). **Faithfulness held
   (4.87)** — it is *misattribution, not fabrication* (the judge-v2 distinction, now seen
   from the other side: grounded content, wrong source label).

3. **The synthesis WIN and the attribution COST share one cause.** The richer/more-diverse
   source set contextual notes surface is exactly what lifts synthesis (+18.2 @5, the
   project's historically worst category) — and exactly what taxes per-source attribution.
   More sources = better coverage, harder bookkeeping.

4. **Headline overall recall@5 +1.7 (≈noise) hides large opposing internals:** synthesis
   +18.2 and literal +8.7 vs cross-book −9.0 and contradiction −5.3. But ranking sharpened
   broadly — recall@1 +5.8 and MRR +0.060 (both above noise), recall@20 +3.5. The cross-book/
   contradiction loss is a *top-5 ordering* problem (recall@1 rose for both; recall@20 held) —
   the pool keeps the answers, the @5 rank slips.

**Verdict: provisional KEEP, pending 4.2.** Contextual retrieval is a net positive — synthesis
transformed, literal up, ranking sharper, 12 refusals converted — with a coupled, *recoverable*
attribution tax. The two costs (cross-book/contradiction @5 regression; source-conflation in
attribution) are precisely what **4.2 cross-encoder rerank on the contextualized text** targets:
re-order top-50→top-5 to land the genuinely-best, cleanest source set. If rerank recovers
cross-book @5 and attribution while preserving synthesis, contextual + rerank ship together.
A later generation-side lever (prompt v3: per-claim author attribution discipline) and a D5
model-strength arm (does a stronger generator attribute better across overlapping sources?)
are the follow-ups if the tax survives rerank.

---

## 2026-06-14 — Phase 4.2 cross-encoder rerank: 5 rerankers measured (rerank-*-v1)

**What changed (one variable):** a cross-encoder re-orders the dense candidate pool before
top-k. Pipeline `dense top-50 → near-dup dedup → rerank on retrieval_text → top-k`, a new
`Retriever` behind the same protocol (the ask pipeline/API unchanged). **Alignment law
(rule #4) enforced:** the reranker scores the SAME contextualized text that was embedded
(`retrieval_text` = context_note + heading_path + chunk_text), never bare `text`. Everything
else identical to the dense-ctx-v1 / gen-ctx-v1 rows: qwen3-8b/Nebius/1024d, structural-v1,
enrich-v1, per-group recall; generation = gemma-12b-16k, baseline-v2, top-5, judge
deepseek-v4-flash v3.1. Candidate pool N=50, dedup on. Total arm spend ≈ $0.90 (local
rerankers $0; cohere v3.5 retrieval ≈ $0.14, pro retrieval ≈ $0.34, pro judged generation
≈ $0.42 — ⚠ estimates from Cohere per-search pricing via OpenRouter, 2026-06-14).
**Run records:** `2026-06-14T17-53-56Z-rerank-qwen3-v1.json`,
`…17-58-57Z-rerank-bge-v1.json`, `…18-12-33Z-rerank-cohere-v35-v1.json`,
`…18-18-58Z-rerank-cohere-pro-v1.json`, `…19-45-36Z-gen-rerank-pro-v1.json`.

### Retrieval — 135 in-scope, per-group, vs the dense-ctx-v1 (56.7% @5) and dense-v1 (55.0%) floors

| candidate | host | r@1 | **r@5** | r@20 | MRR | cross-book @5 |
|---|---|---|---|---|---|---|
| dense-v1 (bare, floor) | — | 26.7 | 55.0 | 71.0 | 0.519 | **34.4** |
| dense-ctx-v1 (no rerank) | — | 32.5 | 56.7 | 74.5 | 0.579 | 25.4 |
| rerank qwen3-reranker-0.6b | local | 29.4 | 53.7 | 72.5 | 0.537 | 22.0 |
| rerank bge-reranker-v2-m3 | local | 29.5 | 54.6 | 72.6 | 0.548 | 27.9 |
| rerank cohere/rerank-v3.5 | OpenRouter | **34.5** | 56.7 | 71.3 | 0.585 | 26.4 |
| rerank cohere/rerank-4-pro | OpenRouter | 34.1 | **58.4** | 74.0 | **0.594** | 26.0 |

### Generation — 161 questions, baseline-v2, judge-v3.1, vs the two gen floors

| metric | gen-baseline-v2 | gen-ctx-v1 | gen-rerank-pro-v1 |
|---|---|---|---|
| faithfulness | 4.89 | 4.81 | 4.82 |
| completeness | 4.45 | 4.30 | **4.41** |
| attribution | 4.28 | **3.98** | **4.18** |
| citation recall | 41.6% | 43.8% | **46.8%** |
| citation precision | 42.3% | 43.9% | 38.2% |
| in-scope false-refusal | 15.6% | 7.4% | **3.0%** |
| OOS refusal accuracy | 84.6% | 84.6% | 80.8% |

**Findings:**

1. **Reranker quality is the binding constraint — the inverse of D2.** Only SOTA Cohere Pro
   beats the no-rerank contextual baseline (+1.7 @5, +0.015 MRR); both LOCAL rerankers
   *regress* it (qwen3-0.6b −3.0, bge −2.1 @5, both above the ±0.7pp noise floor) and even
   dip recall@20 — a 0.6b/m3 cross-encoder is weaker than the 8B embedder's own ranking and
   adds noise. In D2 the embedder was the constraint; here the embedder is strong enough that
   only a strong reranker helps. Shipping rerank therefore means a *paid hosted* dependency.
2. **Cross-book is NOT rerankable — not even by Pro (26.0 vs floor 34.4).** The whole
   provisional-KEEP rationale ("rerank recovers the 4.1 cross-book @5 tax") FAILS: every
   reranker leaves cross-book below floor though the pool holds the answers (cross-book @20
   50–58%). Cross-book is a candidate-*pool* problem, not an ordering one — a hybrid-BM25 /
   agent target (4.3+), closed here with a receipt.
3. **The generation tier is where rerank earns its keep.** Pro recovers ~⅔ of the −0.30
   attribution tax (ctx 3.98 → 4.18; baseline 4.28), restores completeness to baseline
   (4.30 → 4.41), posts the best citation recall measured (46.8%) and the lowest false-refusal
   ever (15.6 → 7.4 → 3.0%). Faithfulness flat (4.82, noise). The attribution figure likely
   *understates* recovery: rerank converted still more refusals into answers (the 4.1
   Simpson's-paradox selection effect — newly-answered hard questions attribute worse and
   drag the mean).
4. **Costs of rerank-pro:** citation precision −5.7 (42.3/43.9 → 38.2 — more markers on the
   richer reranked sets, a precision/recall trade the markers-audit already isolates) and OOS
   refusal accuracy −3.8 (one question; the abstention contract essentially holds). Mean
   completion 250 tokens (up from 204), mean latency 6.9s (hosted rerank roundtrip + longer
   answers) — a real per-query cost, forever.
5. **The "identical 82.6% literal" across qwen3-0.6b/bge/v3.5 was reranker strength, not a
   dedup bug:** Pro recovered literal @5 to 91.3% from the same deduped pool the others scored
   82.6% on — so dedup did not drop the covering chunk; the weak rerankers parked it at rank
   6–10. Dedup exonerated (the dedup-off control stays available but is no longer indicated).
6. **qwen3-reranker-8b ruled out (not measured):** the only local GGUF on hand (mradermacher
   Q4_K_S) is a generic causal-LM conversion, not a rerank-purpose build like the ggml-org
   0.6b — llama.cpp's `--reranking` can't read its yes/no logit (smoke test: degenerate
   scores ~1e-27, wrong order). A strong-local 8B data point would need a rerank-purpose
   conversion.

**Verdict: KEEP — contextual + rerank (cohere/rerank-4-pro) ship together.** Net positive
across the metrics that matter (attribution recovered, completeness restored, citation recall
+3.0, false-refusal 3.0%, retrieval +1.7 @5 / +MRR), at the price of a paid query-time
dependency and minor citation-precision/OOS costs. The 4.1 provisional KEEP is now confirmed.

**Phase 4 measures against THIS as the rerank floor.** Open follow-ups, not blockers:
(a) **v3.5 generation check** — v3.5 tied Pro on retrieval @5-equivalents (56.7 @5, 34.5 @1,
0.585 MRR) at ~2.5× lower per-query cost; if it captures most of Pro's generation gain it is
the better production pick (the cost-ledger call). (b) **cross-book → 4.3 hybrid BM25/RRF**
(widen the pool) and the Phase 5 agent. (c) prompt-v3 attribution discipline + a D5
stronger-generator arm for the residual attribution gap (contradiction attribution still
3.84). (d) dedup-off and N=100-pool sensitivity runs remain available but unindicated.

---

## 2026-06-14 — Phase 4.3 hybrid BM25/RRF: REJECTED — the 8B embedder subsumes BM25

**What changed (one variable):** add a sparse (BM25-class) retriever next to dense and fuse
by Reciprocal Rank Fusion. Postgres FTS — a `text_tsv` generated `tsvector` column (over the
verbatim `text`, not retrieval_text) + GIN index, added IN PLACE (`ahx db ensure-fts`, no
re-embed); scored with `ts_rank_cd`. `dense top-50 + BM25 top-50 → RRF (k=60) → top-k`, a new
`hybrid` retriever label. Dense base = the existing `dense-ctx-v1` vectors; everything else
identical. **New metric: recall@50 = pool-recall** (did the answer reach the candidate pool
at all — distinguishes "BM25 widened the pool" from "fusion reordered"). Both arms run at
`--top-k 50`. Run cost ≈ $0 (135 query embeds; BM25 is local Postgres).
**Run records:** `2026-06-14T20-21-45Z-hybrid-v1.json`, `…20-24-19Z-dense-ctx-v1.json`.

### Retrieval — hybrid-v1 vs dense-ctx-v1, both top-k 50 (135 in-scope, per-group)

| metric | dense-ctx-v1 | hybrid-v1 | Δ |
|---|---|---|---|
| recall@1 | 32.5 | 30.7 | −1.8 |
| recall@5 | 56.7 | 56.3 | −0.4 |
| recall@10 | 67.2 | 67.0 | −0.2 |
| recall@20 | 74.5 | 74.5 | 0.0 |
| **recall@50 (pool)** | **82.4** | **82.4** | **0.0** |
| MRR | 0.580 | 0.561 | −0.019 |
| cross-book @50 | 76.1 | 76.1 | 0.0 |

**Findings:**

1. **BM25 adds ZERO pool coverage — recall@50 is byte-identical, category by category**
   (overall 82.4, cross-book 76.1, literal 100, synonym 100, …). Every answer BM25 could
   find by exact-token match, the 8B embedder already had in its top-50. The hypothesized
   win (rare proper nouns, Victorian spellings, lit-004's lexical gap) does not materialize:
   where dense@50 misses, the answer is *distributed* (synthesis/cross-book), not keyword-
   findable — so BM25 can't reach it either.
2. **Fusion slightly DEGRADES the top ranks** (@1 −1.8, MRR −0.019, contradiction @5
   67.1→64.5): RRF injects keyword-noise chunks that displace well-ranked dense hits. Not a
   no-op (the moved cells prove fusion ran) — just net-negative. BM25 works (smoke test
   found Vercingetorix); it simply has nothing to add over a strong dense retriever here.
3. **The pre-registered prediction is FALSIFIED** (rag-techniques.md / interaction #1:
   "hybrid rejected at 950 chunks — *expected to flip to a win at 35k*"). It did NOT flip.
   Same lesson as the 950-chunk reranker-subsumes-hybrid result, one layer up and at 46k
   scale: **a SOTA embedder subsumes BM25.** The 2024-era hybrid-always-wins advice assumes
   a weak/lexical first stage; with qwen3-8B that assumption is false on this corpus.
4. **hybrid+rerank not worth running:** rerank cannot surface what is not in the pool, and
   the pool is unchanged vs dense-ctx — so composing them adds only cost.

**Verdict: REJECT hybrid BM25/RRF.** No pool-coverage gain, slight top-rank/MRR regression,
$0 saved by not shipping it. The D3 Postgres-FTS machinery (tsvector + GIN) stays in the
schema — cheap, idle, and reusable if a self-query/keyword-filter arm ever wants it — but it
is not in the retrieval path. **This closes the core Phase-4 retrieval chain:** the binding
constraint was the embedder (D2, +18 @5); contextual is marginal, rerank is marginal+paid,
hybrid is null. The remaining headroom on the hard categories (synthesis/cross-book/multi-hop)
is **architectural, not retrieval-ranking** — it belongs to the Phase 5 agent (multi-step
search assembles distributed answers), not to more single-query retrieval technique. Optional
unrun control: BM25-alone (`sparse-v1`) to quantify how far lexical trails dense — case-study
color, not a decision input.

---

## 2026-06-15 — Phase 5.1 agentic layer: gen-agent-v2 (KEEP for the distributed-answer path)

**What changed (the architecture, not a ranking knob):** a single grammar-constrained ReAct
agent (LangGraph; D1 recheck confirmed LangGraph, no LlamaIndex in prod) replaces single-shot
generation on the SAME retrieval stack. Each turn the model emits a pydantic `Decision` under a
GBNF grammar (llama.cpp) — `search` (whole-corpus rerank-pro OR `pg_id` source-isolated dense) ·
`read(chunk_id)` · `list_sources` · `find_quote` · `finalize` — looping think→act with a
`max_steps` forced-finalize bound. The citation adapter renumbers the agent's `[c<id>]` markers
to the standard `[n]` table, so the eval/judge consume it byte-identically to single-shot
(agent-vs-single-shot is a like-for-like comparison). **Config:** agent prompt **agent-v2**,
`gemma-12b-100k` (large context so full-text search history fits — see infra notes), cohere/
rerank-4-pro (retry-hardened), judge deepseek-v4-flash **judge-v3.1**, all 161 questions,
concurrency 4. Run ≈ 35 min; spend ≈ $1.5 (⚠ est. — cohere per-search × multi-search/question
+ judge).
**Run record:** `backend/evals/runs/2026-06-15T12-18-31Z-gen-agent-v2.json`

### Agent vs single-shot floor (gen-rerank-pro-v1, same rerank-pro retrieval, judge-v3.1)

| metric | single-shot floor | **agent-v2** | Δ |
|---|---|---|---|
| faithfulness | 4.82 | 4.84 | +0.02 (noise) |
| completeness | 4.41 | 4.53 | +0.12 |
| **attribution** | 4.18 | **4.50** | **+0.32** |
| citation recall | 46.8% | 45.4% | −1.4 (parity) |
| citation precision | 38.2% | 41.4% | +3.2 |
| in-scope false-refusal | 3.0% | 3.0% | 0 |
| OOS refusal accuracy | 80.8% | **73.1%** | **−7.7** |
| mean completion tokens | 250 | 660 | 2.6× |
| mean latency | 6.9s | 30.7s | 4.5× |

### Per-category attribution / completeness (vs single-shot per-category gen-baseline-v2-v3.1)

| category | attrib ss→agent | compl ss→agent | read |
|---|---|---|---|
| synthesis | 3.67 → **4.56** | 3.53 → **4.44** | transformed — the project's worst category |
| cross-book | 3.93 → **4.31** | 4.21 → **4.54** | win on both |
| contradiction | 4.05 → **4.47** | 4.42 → 4.32 | attribution win, completeness flat |
| multi-hop | 5.00 → **3.78** | 4.67 → **4.13** | **REGRESSED** — the agent's weak spot |
| literal/synonym | ~5.00 → ~4.95 | ~4.9 → ~4.87 | parity (already strong single-shot) |

**Findings:**

1. **Attribution is the win, concentrated exactly on the distributed-answer categories.** +0.32
   overall; per-category synthesis +0.89, contradiction +0.42, cross-book +0.38. Source-isolated
   searching surfaces each source and the agent names it — the contradiction/synthesis weakness
   single-shot couldn't touch (Phase 4 closed retrieval-ranking; this is the architectural lever
   the 4.3 entry pointed to). **Synthesis is transformed** (compl 3.53→4.44): multi-step search
   assembles distributed answers, as predicted. Faithfulness stays highest-tier (4.84); in-scope
   quality is at parity (false-refusal 3.0%, citation recall ≈ floor).

2. **Multi-hop REGRESSED — the one in-scope category the agent hurts** (attrib 5.00→3.78, compl
   4.67→4.13, cit-recall 0.23). Mechanism: single-shot's tight top-5 keeps each hop's single
   source cleanly attributable; the agent pulls many sources across several searches and
   **conflates which source supports which hop** (misattribution drives attribution to 3.78).
   More sources = better synthesis coverage, harder per-hop bookkeeping — the same coupling the
   4.1 contextual entry saw, now from the agent side. A Phase 5.2 target (per-hop attribution
   discipline / a stronger reasoner).

3. **OOS refusal regressed to 73.1% (−7.7) — a MODEL limit, not a prompt gap (receipt).** The
   agent over-answers source-absent questions by substituting adjacent secondary material
   (Sappho's themes from a Grote footnote; Plato's *Republic* from Bury's commentary). Tracing
   the model proves it is **not** a detection failure and **not** a prompt-clarity gap: on
   oos-023 the model reasons *"Sappho's own works are not listed as primary sources"* — correctly
   detecting absence — then **answers anyway from the secondary mention.** prompt-v2 sharpened the
   rule and added a refusal-with-provenance form; it recovered exactly 1 of 8 (gemma-12b
   acknowledges the abstention precondition but overrides it under the pull of relevant-looking
   material). **Disciplined abstention on source-absent questions is a model-strength frontier
   (D5 arm), not a prompt fix** — the cleanest case-study evidence that some failures are the
   model, measured directly.

4. **Cost is real and is the whole reason for a router.** 2.6× completion tokens, 4.5× latency
   (30.7s mean; the forced-finalize multi-hop/synthesis questions run 80–120s). The agent is the
   expensive path — Phase 6's router sends only hard distributed-answer questions down it, easy/
   literal/OOS stay single-shot.

5. **Agent runs carry stochastic-tool-path variance beyond single-shot's temp-0 noise.** A prior
   (infra-noisy) run of the same config scored attribution 4.81 vs this run's 4.50 — different
   search paths surface different sources. Treat single-question movements and sub-0.3 per-
   category deltas as agent noise; the architecture-level signal (attribution up, synthesis
   transformed, multi-hop down) is stable.

**Two measurement bugs found and fixed en route (rule #5), both worth the receipt:**
- **Unbounded `cited_chunk_ids` → run crash.** The first full run died when gemma emitted a
  runaway integer list until the JSON truncated → `ValidationError` → uncaught → whole run lost.
  Fix: dropped the (unused) field; defensive parse in `think` ends a degenerate generation as a
  refusal, never a crash.
- **Transient cohere-rerank failures under concurrency → 6 false "refusals".** Concurrent searches
  rate-limited OpenRouter; `RerankClient` read the error body as `payload["results"]` → KeyError,
  recorded as failed refusals (inflated false-refusal 3.0%→7.4%, deflated recall). Fix: rerank
  retry with backoff + a clear `RerankError`. The clean re-run confirms: 0 errors, false-refusal
  back to 3.0%. (A per-question resilience guard in the eval also records-and-continues so one
  failure can't abort a concurrent run.)

**Verdict: KEEP — the agent is the distributed-answer path, routed.** The 5.1 exit (beat
single-shot on synthesis + contradiction with evidence) is **met**: synthesis transformed,
contradiction/cross-book attribution up, in-scope quality at parity. It is NOT a universal
replacement — multi-hop attribution regresses and OOS abstention drops (both partly model-strength
limits), and it costs ~4.5× latency. Production shape: Phase 6 router → single-shot for easy/
literal/OOS, agent for synthesis/cross-book/contradiction. **Open follow-ups (Phase 5.2):** D5
stronger-reasoner arm (the lever for OOS abstention AND multi-hop attribution — needs verified
structured-output support to keep grammar-ReAct); per-hop attribution discipline (prompt-v3);
reflection/critic edge. **This row is the agent baseline; technique arms measure against it.**

---

## 2026-06-15 — Pre-D5 forensic audit of gen-agent-v2 + judge-v3.2 (two measurement fixes)

**Why:** before opening the D5 stronger-reasoner arm (the production model; gemma-12b is
local-only), audit every failed answer in the agent baseline to separate model failures from
*judge* failures — and to confirm the harness can fairly measure what the smarter model is
expected to improve. Forensic read of all 11 in-scope failures (4 refusals + 7 weak) and all 7
OOS leaks in the gen-agent-v2 record (the judge-v3.1 row, 2026-06-15T12-18-31Z).

### Finding 1 — one real judge bug (OOS), fixed in judge-v3.2

The semantic-refusal judge stored **only a yes/no bit, no rationale** — so every OOS verdict
(26/26) was unauditable (`judge_notes` empty). Worse, its narrow wording ("states the sources
lack the info") credited a false-premise *correction* only when phrased as an explicit
abstention: **oos-013** ("which Persian king did Caesar beat at Gaugamela?") was answered with a
textbook premise-rejection ("the sources don't place Caesar there; Gaugamela was Alexander over
Darius") yet scored as a non-refusal — inconsistent with oos-014, credited on the same pattern.

**judge-v3.2 (rule #5):** refusal call returns JSON `{"refusal", "reason"}` (reason stored, OOS
now auditable) and the rubric explicitly credits a premise-correction-without-supplying-the-fact
as a refusal, while keeping secondary-material substitution a leak. **Methodology guard:** the
first draft of the new prompt pasted oos-013's real question+answer as the illustrative example —
caught and replaced with a synthetic non-corpus example (Hannibal-on-the-Moon). A judge prompt
that hard-codes a held-out item's answer overfits the judge to the test and voids that
measurement; illustrations in judge prompts must be synthetic (the three scoring rubrics already
follow this — abstract `X`/`Y` placeholders).
**Run record:** `backend/evals/runs/2026-06-15T13-44-11Z-gen-agent-v2-judge-v3.2.json` (rejudge
of the frozen gen-agent-v2 answers; 0 answers differ — judge-only isolation).

| | v3.1 | v3.2 | reading |
|---|---|---|---|
| OOS refusal accuracy | 73.1% | **76.9%** | oos-013 flipped (19→20/26), the *only* refusal change |
| in-scope false-refusal | 3.0% | 3.0% | no in-scope answer misread as a refusal |
| in-scope answered | 131/135 | 131/135 | unchanged |
| faithfulness | 4.84 | 4.79 | reproduced within noise |
| completeness | 4.53 | 4.48 | reproduced within noise |

The remaining 5 OOS leaks (oos-019/020/023/024/025) re-audited as **genuine model failures** —
the agent substitutes secondary material for an absent named work (oos-024 also a faithfulness
breach: recited the *Natural History*'s 37-book structure, present in no cited chunk). oos-021
(Behistun) stays flagged as a too-subtle question (Rawlinson decoded the inscription and reports
its content). **No in-scope judge errors found** — several low scores are *impressively* precise
(caught the 207 BC-vs-362 BC Mantinea swap in con-018; the regent-vs-king Pausanias conflation in
mh-016).

### Finding 2 — in-scope failures are one mechanism: namesake/date conflation

8 of 11 in-scope failures share a signature — **faithfulness 5, completeness 1**: every sentence
is grounded in *some* retrieved passage, but the answer is assembled around the **wrong entity**.
Two wives of Claudius (mh-002, Agrippina for Messalina), two statesmen (mh-008, Themistocles for
Pericles), one general's two battles (mh-014, Leuctra for Mantinea), two men named Pausanias
(mh-016), two battles of Mantinea (con-018). Not retrieval (right passages in pool), not the
judge (scores the defect correctly), not the prompt — a **disambiguation limit of the 12B**. The
other 3: cb-005/cb-017 are false refusals on answerable cross-book *breadth* (the agent churns
99–105s / 1000–1800 tokens then quits assembling — an agent-synthesis/`max_steps` limit);
syn-006 is the one honestly-correct refusal (documented chunk-boundary split of *Cambyses* +
*sacred disease*). This is the D5 reasoner's target, now with a precise mechanism.

### Finding 3 — attribution is judge-noise-limited (the blocker for the D5 arm)

The v3.2 rejudge is a controlled judge-variance probe: **identical frozen answers, byte-identical
scoring rubrics, re-scored** (only the refusal prompt changed; judge already at temperature 0).
Faithfulness/completeness reproduced at the ±0.2 floor; **attribution did not.**

| dimension | % questions changed | mean \|Δ\|/question | aggregate Δ |
|---|---|---|---|
| faithfulness | 10% | 0.20 | −0.05 |
| completeness | 9% | 0.22 | −0.05 |
| **attribution** | **19%** | **0.66** | **−0.41** |

Not jitter — **full 1↔5 flips on 25 questions** (con-013/con-018/cb-002/cb-003/cb-008 all 5→1),
concentrated on the multi-source categories. The flash judge cannot *stably* answer "do these
sources disagree, and did the answer attribute each in prose?" — itself a hard reasoning task.
**Consequence:** the agent's headline "+0.32 attribution win over single-shot" and the "multi-hop
attribution regression (5.00→3.78)" both sit **inside the judge's own 0.41 aggregate swing** — not
reliably real. A frontier judge for the attribution rubric (or N≥3 self-consistency) is a
**prerequisite** before the D5 model is measured on attribution; faithfulness/completeness are
ready as-is. This is the eval log's pre-registered "calibrate vs a frontier judge when a decision
rides on a small difference" — D5 is that decision.

**Verdict:** gemma-12b agent baseline locked and honest (judge-v3.2). Still a strong portfolio
result — faithfulness 4.79, 131/135 in-scope answered, synthesis transformed. Two clean pre-D5
work items: (1) stabilise the attribution measurement (frontier/ensemble judge); (2) the D5
stronger-reasoner arm targets namesake/date conflation (the 8-question mechanism above).
