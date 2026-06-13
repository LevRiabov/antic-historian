# Golden Set v2 — Format & Authoring Guide

> **Status:** format frozen, set under construction (semi-automatic authoring by the user + an MCP-equipped Claude).
> Files: `backend/evals/golden/<category>.yaml` · Validate: `uv run ahx eval validate` (in `backend/`).
> The golden set is production code (rule #5): schema-validated, git-reviewed, versioned.

## 1. Why this exists, in one paragraph

Every retrieval/generation decision in this project is justified by numbers from this set
(rule #1). Two metric tiers consume it: **retrieval metrics** (recall@k, MRR — computed from
gold spans, free, run constantly) and **judge metrics** (faithfulness / completeness / refusal
— computed from ideal answers by an LLM judge, costly, run at phase boundaries). Recall alone
is NOT enough — rag-historian's synthesis category was retrieval-flat but generation-bound;
only the judge tier caught it.

## 2. Size targets

| Version | Per category | Total | Unlocks |
|---|---|---|---|
| **v2.0** | 10 | 70 | Phase 3 baseline numbers; harness end-to-end |
| **v2.1** | 20 | 140 | Phase 4 ablations — ±0.3-effects per category become measurable (rag-historian's n≈9 noise-floor lesson) |

Industry context: there is no single standard; serious production teams curate 50–200 questions
for a focused domain. Below ~10/category, per-category claims are anecdotes; beyond ~300 total,
diminishing returns at our corpus size.

## 3. Categories (6 proven + 1 new)

| Category | Tests | Gold spans |
|---|---|---|
| `literal` | Basic retrieval: answer sits verbatim in one passage | 1 (occasionally 2) |
| `synonym` | Vocabulary mismatch: modern question vs Victorian translation ("epilepsy" / "falling sickness") | 1–2 |
| `multi-hop` | Combining facts from different passages | one per hop |
| `synthesis` | Distributed answer within one work ("How does Caesar portray the Gauls?") | the 3–6 most load-bearing passages |
| `cross-book` **(new)** | Synthesis across works — needs ≥2 pg_ids | 2–6, ≥2 works |
| `contradiction` | Sources disagree; answer must attribute versions | the disagreeing passages, ≥2 works |
| `out-of-scope` | Honest refusal | none (and no ideal_answer) |

## 4. The format (one YAML list per category file)

```yaml
- id: lit-001                  # category prefix + running number, unique across ALL files
  category: literal
  question: How many wounds did Julius Caesar receive when he was assassinated?
  ideal_answer: >-             # what a 5/5 answer contains; concise, factual
    According to Suetonius, Caesar was stabbed with twenty-three wounds. ...
  gold_spans:
    - pg_id: 6400              # which work (manifest id)
      quote: >-                # EXACT quote from the text — see rules below
        He was stabbed with three and twenty wounds, uttering a groan only,
        but no cry, at the first wound
      note: Suetonius, Julius LXXXII        # optional human note
  notes: anything for future-you            # optional
  status: draft                # draft -> reviewed (human-checked)
```

## 4a. Requirement groups (`groups`) — how recall reads multiple spans

A question often has several gold spans, and they relate to the answer in one of
two ways. Retrieval recall must know which:

- **Alternatives (any-of)** — the same fact recurs across works ("Caesar's
  twenty-three wounds" in Suetonius, Appian, Plutarch, Livy, Smith). Retrieving
  *any one* fully answers; surfacing one shouldn't score 0.2.
- **Required (all-of)** — distinct facts the answer must combine (a multi-hop
  question's two hops; a contradiction's two versions; the load-bearing passages
  of a synthesis). Each is independently required.

`groups` on a span is the list of **answer requirements it satisfies**. Recall is
computed per requirement, not per span:

```
recall@k = (requirements with ≥1 covering span in top-k) / (total requirements)
```

Rules:
1. Spans **sharing a label are alternatives** — any one covers that requirement.
   So all five wound-count spans get `groups: [wounds]` → one requirement → a
   single hit scores 1.0.
2. **Distinct labels are conjunctive** — `groups: [hop1]` vs `groups: [hop2]`
   are two requirements; you need a covering span for each.
3. A span may satisfy **several requirements at once** — a chunk that answers
   both hops gets `groups: [hop1, hop2]`; retrieving it alone covers both.
4. **No `groups` = a singleton requirement** (independently required). This is
   the back-compatible default and the right model for `cross-book` /
   `synthesis`, where each load-bearing span is genuinely its own coverage
   target. Single-span questions need no labels.

Per category:

| Category | Grouping |
|---|---|
| `literal`, `synonym` | all spans one label (alternatives) — usually a single requirement |
| `multi-hop` | one label per hop (`hop1`, `hop2`); a combined chunk lists both |
| `contradiction` | one label per version; group only *alternative attestations of the same version*; a passage stating both versions lists both |
| `cross-book`, `synthesis` | no labels — each span its own requirement (coverage) |

Labels are scoped to the question (reused freely across questions). Pick readable
names (`wounds`, `hop1`, `apotheosis`) — they show up in run records and diffs.

**Quote rules — the part that matters:**
1. The quote must occur **exactly once** in that work. Too short → "ambiguous"; lengthen it.
2. Wording must be **verbatim** (the resolver tolerates any whitespace/line-break differences,
   nothing else). Don't trust an LLM's memory of a quote — LLMs paraphrase; always verify with
   the `find_quote` MCP tool or `ahx eval validate`.
3. The span is **where the answer lives**, not merely on-topic text.
4. Offsets are never written by hand — quotes are resolved to char offsets at eval time,
   so parser/normalization fixes can't silently invalidate the set.

## 5. The semi-automatic workflow (MCP)

The repo's `.mcp.json` exposes the corpus to any Claude instance opened in this repo
(requires: `docker compose up -d` + llama-swap running):

| Tool | Use |
|---|---|
| `list_sources` | What's in the corpus (38 works, pg_ids, chunk counts) |
| `search_corpus(query, top_k, pg_id?)` | Find candidate passages; optionally inside one work |
| `read_passage(pg_id, start, end, pad)` | Read exact text + surrounding context |
| `find_quote(pg_id, quote)` | **Verify a quote resolves uniquely BEFORE putting it in YAML** |

Suggested loop per category: prompt a Claude instance to (1) pick a topic with good corpus
coverage via `search_corpus`, (2) draft a question of the right category shape, (3) locate and
`find_quote`-verify the gold spans, (4) emit the YAML block. **You review each question**
(is it natural? is the ideal answer right? are spans where the answer lives?), set
`status: reviewed`, and run `uv run ahx eval validate`.

Authoring guidance:
- Write questions a curious reader would ask — not retrieval-shaped keyword strings.
- Vary difficulty and works; don't let one book dominate a category.
- For contradiction/cross-book: Caesar-era works overlap most (Suetonius, Plutarch, Dio,
  Caesar himself, Mommsen/Gibbon as scholarship voices) — richest hunting ground.
- OOS questions: adjacent-but-absent beats absurd ("Bayeux Tapestry" beats "JavaScript").

## 6. Validation

`uv run ahx eval validate` checks: schema, id uniqueness, every non-OOS question has spans,
every quote resolves uniquely; prints per-category counts vs targets. CI runs it once the set
stabilizes. Resolution failures are listed with the reason (`not-found` = wording drift,
`ambiguous` = lengthen the quote).
