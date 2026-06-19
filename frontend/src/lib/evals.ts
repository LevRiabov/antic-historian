/*
 * Pure transforms for the /evals golden-set page — joining the two eval runs,
 * deriving the aggregate tiles + per-category table, and filtering the rows.
 * Framework-free and side-effect-free (no React, no fetch), like lib/sources.ts.
 *
 * One table from two handlers: the GenerationRun is the spine (it holds all
 * questions, including the out-of-scope ones the RetrievalRun skips by design),
 * and recall/MRR attach by question_id where the RetrievalRun has them.
 */
import type {
  Category,
  GenAggregates,
  GenerationRun,
  RetrievalAggregates,
  RetrievalRun,
  SourceOut,
} from "./types";

export const CAT_ORDER: readonly Category[] = [
  "literal",
  "synonym",
  "multi-hop",
  "synthesis",
  "cross-book",
  "contradiction",
  "out-of-scope",
];

export interface CategoryMeta {
  color: string;
  desc: string;
}

// Carried verbatim from the design mock (golden-01-table.html): a stable hue +
// one-line gloss per category, used by the badges, tabs, and detail sidecard.
export const CAT_META: Record<Category, CategoryMeta> = {
  literal: { color: "#3a6ea5", desc: "Answer sits verbatim in one passage." },
  synonym: {
    color: "#6b8e23",
    desc: "Modern wording vs Victorian translation (vocabulary mismatch).",
  },
  "multi-hop": { color: "#8a5a2b", desc: "Combine facts from different passages." },
  synthesis: { color: "#a05a8a", desc: "Distributed answer within one work." },
  "cross-book": { color: "#2f8f8f", desc: "Synthesis across two or more works." },
  contradiction: {
    color: "#b5621f",
    desc: "Sources disagree; the answer must attribute each version.",
  },
  "out-of-scope": {
    color: "#8b9098",
    desc: "Not in the corpus — an honest refusal is the correct answer.",
  },
};

/* ---- metric tone (good / amber / refuse / danger), thresholds from the mock ---- */

// The shared metric-chip tone. `danger` is unused by the eval metrics (it's the
// security page's breach tone) but lives here so one MetricChip serves both pages.
export type MetricTone = "good" | "amber" | "refuse" | "danger";

/** recall is a percentage (0-100). */
export function recallTone(pct: number): MetricTone {
  return pct >= 85 ? "good" : pct >= 60 ? "amber" : "refuse";
}
/** judge scores are 1-5. */
export function scoreTone(score: number): MetricTone {
  return score >= 4.5 ? "good" : score >= 3.5 ? "amber" : "refuse";
}
export function mrrTone(mrr: number): MetricTone {
  return mrr >= 0.7 ? "good" : mrr >= 0.45 ? "amber" : "refuse";
}

export function fmtPct(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}
export function fmtScore(score: number): string {
  return score.toFixed(1);
}
export function fmtMrr(mrr: number): string {
  return mrr.toFixed(2);
}

/* ---- the merged per-question row ---- */

export interface GoldenRow {
  id: string;
  category: Category;
  question: string;
  ideal: string; // ideal_answer; "" for out-of-scope
  answer: string; // the model's produced answer
  // retrieval tier — null for out-of-scope (the RetrievalRun skips those)
  recall5: number | null; // percentage 0-100
  recall20: number | null;
  mrr: number | null;
  // judge tier — null until/unless the judge layer ran, or N/A (attribution on
  // single-fact categories)
  faithfulness: number | null;
  completeness: number | null;
  attribution: number | null;
  // refusal — meaningful for out-of-scope (and tracked everywhere)
  refusalExpected: boolean;
  refusalCorrect: boolean;
  // chunk identity for the detail panel (ids matter more than the locator)
  goldChunkIds: number[];
  goldPgIds: number[]; // distinct works behind the gold spans
  citedChunkIds: number[];
  // the answer's [n] markers index into this 1-based: marker n -> retrievedChunkIds[n-1]
  // (the prompt's numbering invariant). Lets a citation pill open the right passage.
  retrievedChunkIds: number[];
  judgeNotes: string;
}

/** The three judged dimensions, in the order the backend emits them. */
export const JUDGE_DIMENSIONS = ["faithfulness", "completeness", "attribution"] as const;

export interface JudgeNote {
  label: string; // capitalized dimension, or "" for an unlabeled note
  text: string;
}

/** judge_notes arrives as "faithfulness: … | completeness: … | attribution: …"
 *  (joined per-dimension on the backend, generation.py). Split it back into the
 *  labeled parts. A note that isn't one of the known dimensions (e.g. the
 *  out-of-scope "refusal=…: …" reason, or a "RUN ERROR: …" line) comes back as a
 *  single unlabeled note. */
export function parseJudgeNotes(notes: string): JudgeNote[] {
  const trimmed = notes.trim();
  if (!trimmed) return [];
  return trimmed.split(" | ").map((part) => {
    const idx = part.indexOf(": ");
    const label = idx === -1 ? "" : part.slice(0, idx);
    if ((JUDGE_DIMENSIONS as readonly string[]).includes(label)) {
      return { label: label.charAt(0).toUpperCase() + label.slice(1), text: part.slice(idx + 2) };
    }
    return { label: "", text: part };
  });
}

function recallPct(recall: Record<string, number>, k: string): number | null {
  const v = recall[k];
  return v === undefined ? null : v * 100;
}

/** Join the two runs by question_id. The generation run drives the row set (it
 *  has every question, incl. out-of-scope); retrieval metrics attach where the
 *  retrieval run has that question. Returns rows in the generation run's order. */
export function mergeGolden(rag: RetrievalRun, agent: GenerationRun): GoldenRow[] {
  const ragById = new Map(rag.results.map((r) => [r.question_id, r]));
  return agent.results.map((g) => {
    const r = ragById.get(g.question_id);
    const goldPgIds = r ? [...new Set(r.gold_spans.map((s) => s.pg_id))] : [];
    return {
      id: g.question_id,
      category: g.category,
      question: g.question,
      ideal: g.ideal_answer,
      answer: g.answer,
      recall5: r ? recallPct(r.recall, "5") : null,
      recall20: r ? recallPct(r.recall, "20") : null,
      mrr: r ? r.mrr : null,
      faithfulness: g.faithfulness,
      completeness: g.completeness,
      attribution: g.attribution,
      refusalExpected: g.refusal_expected,
      refusalCorrect: g.refusal_correct,
      goldChunkIds: r ? r.gold_chunk_ids : [],
      goldPgIds,
      citedChunkIds: g.cited_chunk_ids,
      retrievedChunkIds: g.retrieved_chunk_ids,
      judgeNotes: g.judge_notes,
    };
  });
}

/* ---- answer tiers (good / poor / failed) ----
 *
 *  A judged answer falls into one of three tiers, governed by its WEAKEST judged
 *  dimension (faithfulness / completeness / attribution):
 *    - good   — every dimension scored > 3 (a genuinely strong answer)
 *    - poor   — scored >= 3 everywhere but >= one dimension is exactly 3
 *               (a 3/5 is still a real answer, just a weak one — we surface it
 *               rather than hide it inside the pass rate)
 *    - failed — any dimension scored < 3, no answer was produced, or an
 *               out-of-scope question wasn't refused
 *  "good" and "poor" together are the ADEQUATE (>= 3) answers — the ones that
 *  count as correctly answered. A correct refusal is a "good" answer (it's the
 *  right behaviour); a missed refusal is "failed". A null dimension (unjudged, or
 *  attribution N/A on single-fact questions) never drags a row down a tier. */
export type AnswerTier = "good" | "poor" | "failed";

export function answerTier(row: GoldenRow): AnswerTier {
  if (row.category === "out-of-scope") return row.refusalCorrect ? "good" : "failed";
  if (row.answer.trim() === "") return "failed"; // expected an answer, got none
  const scores = [row.faithfulness, row.completeness, row.attribution].filter(
    (s): s is number => s !== null,
  );
  if (scores.length === 0) return "good"; // answered but unjudged — counts as passing
  const weakest = Math.min(...scores);
  if (weakest < 3) return "failed";
  return weakest > 3 ? "good" : "poor";
}

/** A row counts as a failure when its tier is "failed" (see answerTier): an
 *  out-of-scope question that wasn't refused, an expected answer that wasn't
 *  produced, or any judged dimension below 3. A 3/5 is "adequate", not a failure. */
export function isFailure(row: GoldenRow): boolean {
  return answerTier(row) === "failed";
}

/* ---- filtering ---- */

export type EvalFilter = "all" | Category;

export interface EvalView {
  filter: EvalFilter;
  query: string;
  failuresOnly: boolean;
}

export function selectRows(rows: readonly GoldenRow[], view: EvalView): GoldenRow[] {
  const q = view.query.trim().toLowerCase();
  return rows.filter((row) => {
    if (view.filter !== "all" && row.category !== view.filter) return false;
    if (view.failuresOnly && !isFailure(row)) return false;
    if (q && !row.id.toLowerCase().includes(q) && !row.question.toLowerCase().includes(q)) {
      return false;
    }
    return true;
  });
}

export type CategoryCounts = Record<EvalFilter, number>;

export function categoryCounts(rows: readonly GoldenRow[]): CategoryCounts {
  const counts = { all: rows.length } as CategoryCounts;
  for (const cat of CAT_ORDER) counts[cat] = 0;
  for (const row of rows) counts[row.category] += 1;
  return counts;
}

/* ---- aggregate stat bar (top tiles) ---- */

/** Plain-language definition of a failing question — used in the stat-bar hint and
 *  the failures-only toggle so both explain the same rule (kept in sync with
 *  isFailure above). */
export const FAILURE_HINT =
  "A question fails if it's out-of-scope but wasn't refused, if an answer was " +
  "expected but none was produced, or if faithfulness, completeness, or attribution " +
  "scored below 3. A score of exactly 3 is a poor-but-adequate answer, not a failure.";

export interface StatTile {
  label: string;
  value: string;
  tone: MetricTone | null; // null = neutral (no color)
  sub: string;
  hint?: string; // optional "ⓘ" tooltip explaining the metric
}

/** Tier breakdown over the full (unfiltered) set. `adequate` (= good + poor) is
 *  the count that passes — i.e. scored >= 3 everywhere (or refused correctly);
 *  `poor` is the subset that only scraped a 3 somewhere. */
export function countOutcomes(rows: readonly GoldenRow[]): {
  total: number;
  good: number;
  poor: number;
  adequate: number; // good + poor — the answers that count as correct (>= 3)
  failed: number;
} {
  let good = 0;
  let poor = 0;
  let failed = 0;
  for (const row of rows) {
    const tier = answerTier(row);
    if (tier === "good") good += 1;
    else if (tier === "poor") poor += 1;
    else failed += 1;
  }
  return { total: rows.length, good, poor, adequate: good + poor, failed };
}

function pctOrDash(value: number | null): { value: string; tone: MetricTone | null } {
  return value === null ? { value: "—", tone: null } : { value: fmtPct(value), tone: recallTone(value * 100) };
}
function scoreOrDash(value: number | null): { value: string; tone: MetricTone | null } {
  return value === null
    ? { value: "—", tone: null }
    : { value: `${fmtScore(value)}/5`, tone: scoreTone(value) };
}

export function buildStatTiles(
  rag: RetrievalAggregates,
  agent: GenAggregates,
  rows: readonly GoldenRow[],
): StatTile[] {
  const r5 = recallPct(rag.recall, "5");
  const r20 = recallPct(rag.recall, "20");
  const { total, good, poor, adequate, failed } = countOutcomes(rows);
  const goodPct = total ? (good / total) * 100 : null;
  const adequatePct = total ? (adequate / total) * 100 : null;
  return [
    { label: "Questions", value: String(agent.questions), tone: null, sub: `${CAT_ORDER.length} categories` },
    {
      label: "Good answers",
      value: goodPct === null ? "—" : `${Math.round(goodPct)}%`,
      tone: goodPct === null ? null : recallTone(goodPct),
      sub: `${good} of ${total} · every dimension > 3`,
      hint:
        "Scored above 3 on every judged dimension (faithfulness, completeness, " +
        "attribution) — a genuinely strong answer. A correct refusal counts here too.",
    },
    {
      label: "Adequate answers",
      value: adequatePct === null ? "—" : `${Math.round(adequatePct)}%`,
      tone: adequatePct === null ? null : recallTone(adequatePct),
      sub: `${adequate} of ${total} · ${poor} poor (=3) · ${failed} failed`,
      hint:
        "Scored at least 3 on every dimension, so it still counts as correctly " +
        "answered — but a 3/5 is a poor, weak answer. Anything below 3 is a failure.",
    },
    {
      label: "Recall@5",
      value: r5 === null ? "—" : `${Math.round(r5)}%`,
      tone: r5 === null ? null : recallTone(r5),
      sub: "gold reqs in top 5",
    },
    {
      label: "Recall@20",
      value: r20 === null ? "—" : `${Math.round(r20)}%`,
      tone: r20 === null ? null : recallTone(r20),
      sub: "gold reqs in top 20",
    },
    { label: "MRR", value: fmtMrr(rag.mrr), tone: mrrTone(rag.mrr), sub: "rank of first gold hit" },
    { label: "Faithfulness", ...scoreOrDash(agent.faithfulness), sub: "claims traceable to source" },
    { label: "Completeness", ...scoreOrDash(agent.completeness), sub: "all required facts present" },
    { label: "Attribution", ...scoreOrDash(agent.attribution), sub: "each version credited right" },
    { label: "Refusal accuracy", ...pctOrDash(agent.refusal_accuracy_oos), sub: "out-of-scope handled" },
  ];
}

/* ---- per-category aggregate table ---- */

export interface CategoryAggRow {
  category: Category | "overall";
  label: string;
  color: string | null;
  // metrics; null cells render as a muted dash
  recall5: number | null; // percentage
  recall20: number | null;
  mrr: number | null;
  faithfulness: number | null;
  completeness: number | null;
  attribution: number | null;
  // out-of-scope has no retrieval/judge scores, only a refusal accuracy
  refusalAccuracy: number | null; // fraction
}

export function buildCategoryAggRows(
  rag: RetrievalAggregates,
  agent: GenAggregates,
): CategoryAggRow[] {
  const rows: CategoryAggRow[] = CAT_ORDER.map((cat) => {
    const r = rag.by_category[cat];
    const g = agent.by_category[cat];
    const isOos = cat === "out-of-scope";
    return {
      category: cat,
      label: cat,
      color: CAT_META[cat].color,
      recall5: r ? recallPct(r.recall, "5") : null,
      recall20: r ? recallPct(r.recall, "20") : null,
      mrr: r ? r.mrr : null,
      faithfulness: g ? g.faithfulness : null,
      completeness: g ? g.completeness : null,
      attribution: g ? g.attribution : null,
      refusalAccuracy: isOos && g ? g.refusal_correct : null,
    };
  });
  rows.push({
    category: "overall",
    label: "overall",
    color: null,
    recall5: recallPct(rag.recall, "5"),
    recall20: recallPct(rag.recall, "20"),
    mrr: rag.mrr,
    faithfulness: agent.faithfulness,
    completeness: agent.completeness,
    attribution: agent.attribution,
    refusalAccuracy: null,
  });
  return rows;
}

/* ---- gold-source labelling ---- */

/** pg_id -> "Author, Work" for naming the gold chunks in the detail panel. */
export function worksByPgId(sources: readonly SourceOut[]): Map<number, string> {
  return new Map(sources.map((s) => [s.pg_id, `${s.author}, ${s.title}`]));
}
