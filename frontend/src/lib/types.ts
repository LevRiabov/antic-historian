/*
 * SSE event payloads — the TS mirror of the backend pydantic models. Keep these
 * in lockstep with:
 *   ahx/generation/pipeline.py  (SourcesEvent, DeltaEvent, StepEvent, DoneEvent)
 *   ahx/generation/citations.py (Citation, MarkerAudit)
 *   ahx/llm.py (Usage), ahx/pricing.py (Cost), ahx/api/limits.py (SessionStatus)
 * The wire order is: meta -> sources -> (step* in deep mode) -> delta* -> done.
 */

export type AskMode = "fast" | "deep";

export type SourceCategory = "primary" | "scholarship";

/** One corpus work as GET /sources returns it. Mirrors ahx/api/sources.py:SourceOut.
 *  Note: the API is 1:1 with the DB, so a multi-volume set appears as its separate
 *  volumes (one row each), not a single grouped row — `chunks` is the auditable
 *  passage count per work. */
export interface SourceOut {
  pg_id: number;
  author: string;
  title: string;
  translator: string;
  category: SourceCategory;
  pd_basis: string; // EU public-domain justification
  source: string; // derived publisher label, e.g. "Project Gutenberg"
  landing_url: string; // canonical source page for the "↗" link
  chunks: number; // retrievable passages in the DB
}

/** One corpus passage as GET /chunks returns it. Mirrors ahx/api/chunks.py:ChunkOut.
 *  Same readable shape as Citation (minus the per-answer marker/score), so the
 *  citation drawer renders identically whether fed by a live answer or by id. */
export interface ChunkOut {
  chunk_id: number;
  pg_id: number;
  author: string;
  work_title: string;
  locator: string[];
  heading: string | null;
  text: string;
  char_start: number;
  char_end: number;
  pd_basis: string;
}

export interface Citation {
  marker: number; // the [n] the answer text refers to
  chunk_id: number;
  pg_id: number;
  author: string;
  work_title: string;
  locator: string[];
  text: string;
  score: number;
  char_start: number;
  char_end: number;
}

export interface MarkerAudit {
  used: number[]; // distinct valid markers, in order of first appearance
  dangling: number[]; // markers that point at no source
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
}

export interface Cost {
  usd: number | null; // null when the model is unpriced — never silently 0
  input_tokens: number;
  output_tokens: number;
  model: string;
  priced: boolean;
}

/** `limit === 0` means uncapped (don't render a badge). */
export interface SessionStatus {
  limit: number;
  remaining: number;
}

export interface SourcesEvent {
  citations: Citation[];
  prompt_version: string;
}

export interface DeltaEvent {
  text: string;
}

/** One live ReAct step — emitted only in deep mode. */
export interface StepEvent {
  index: number; // 1-based step number
  thought: string;
  tool: string; // "search" | "read" | "list_sources"
  args: Record<string, unknown>;
  observation: string;
  chunk_ids: number[] | null;
  searches_left: number;
}

export interface DoneEvent {
  answer: string;
  refused: boolean;
  markers: MarkerAudit;
  usage: Usage | null;
  cost: Cost | null;
  served_by: string | null;
  blocked: boolean;
}

/** Discriminated union of the named SSE events the client handles. */
export type AskEvent =
  | { event: "meta"; data: SessionStatus }
  | { event: "sources"; data: SourcesEvent }
  | { event: "step"; data: StepEvent }
  | { event: "delta"; data: DeltaEvent }
  | { event: "done"; data: DoneEvent };

/* -----------------------------------------------------------------------------
 * Eval-run records — the TS mirror of the published golden-set runs. Keep these
 * in lockstep with:
 *   ahx/evals/retrieval.py  (RetrievalRun, QuestionResult, Aggregates)
 *   ahx/evals/generation.py (GenerationRun, GenQuestionResult, GenAggregates)
 * The /evals page joins one RetrievalRun (recall@k/MRR, in-scope only) and one
 * GenerationRun (answer + judge scores, incl. out-of-scope) by question_id.
 * Only the fields the page reads are mirrored; the records carry more.
 * -------------------------------------------------------------------------- */

export type Category =
  | "literal"
  | "synonym"
  | "multi-hop"
  | "synthesis"
  | "cross-book"
  | "contradiction"
  | "out-of-scope";

export const CATEGORIES: readonly Category[] = [
  "literal",
  "synonym",
  "multi-hop",
  "synthesis",
  "cross-book",
  "contradiction",
  "out-of-scope",
];

/** A gold span's locator within a work. `pg_id` lets us name the work via /sources. */
export interface GoldSpanRef {
  pg_id: number;
  char_start: number;
  char_end: number;
  groups: string[];
}

/** recall is a k -> recall@k map; JSON serializes the int keys as STRINGS, so
 *  read it as `recall["5"]`, never `recall[5]`. */
export type RecallByK = Record<string, number>;

export interface RetrievalQuestionResult {
  question_id: string;
  category: Category;
  question: string;
  ideal_answer: string;
  gold_spans: GoldSpanRef[];
  gold_chunk_ids: number[];
  retrieved_chunk_ids: number[];
  recall: RecallByK;
  first_hit_rank: number | null;
  mrr: number;
  latency_ms: number;
}

export interface RetrievalCategoryAggregate {
  count: number;
  recall: RecallByK;
  mrr: number;
}

export interface RetrievalAggregates {
  recall: RecallByK; // mean over all in-scope questions
  mrr: number;
  by_category: Record<string, RetrievalCategoryAggregate>;
}

export interface RetrievalRun {
  created_at: string;
  retriever: string;
  embed_model: string;
  top_k: number;
  aggregates: RetrievalAggregates;
  results: RetrievalQuestionResult[];
}

export interface GenQuestionResult {
  question_id: string;
  category: Category;
  question: string;
  ideal_answer: string;
  answer: string;
  refused: boolean;
  refusal_expected: boolean; // true only for out-of-scope
  refusal_correct: boolean;
  markers_used: number[];
  retrieved_chunk_ids: number[];
  cited_chunk_ids: number[];
  faithfulness: number | null; // 1-5, judge layer; null if unjudged / out-of-scope
  completeness: number | null;
  attribution: number | null; // null where not applicable (single-fact categories)
  judge_notes: string;
  latency_ms: number;
}

export interface GenCategoryAggregate {
  count: number;
  refused: number;
  refusal_correct: number; // fraction
  faithfulness: number | null;
  completeness: number | null;
  attribution: number | null;
  mean_latency_ms: number;
}

export interface GenAggregates {
  questions: number;
  refusal_accuracy_oos: number | null;
  false_refusal_rate: number;
  faithfulness: number | null;
  completeness: number | null;
  attribution: number | null;
  mean_latency_ms: number;
  by_category: Record<string, GenCategoryAggregate>;
}

export interface GenerationRun {
  created_at: string;
  label: string;
  chat_model: string;
  embed_model: string;
  prompt_version: string;
  top_k: number;
  engine: string; // "single-shot" | "agent"
  judge_model: string | null;
  judge_rubric: string | null;
  aggregates: GenAggregates;
  results: GenQuestionResult[];
}

/* -----------------------------------------------------------------------------
 * Security-audit records — the TS mirror of ahx/evals/security.py. The /security
 * page joins the latest baseline (no defence) and defended (defence-stack) runs by
 * attack id: per-attack before/after, plus ASR (attack success rate) aggregates.
 * -------------------------------------------------------------------------- */

export type AttackCategory =
  | "extraction"
  | "scope-escape"
  | "grounding-bypass"
  | "citation-forgery"
  | "fake-source-injection";

export const ATTACK_CATEGORIES: readonly AttackCategory[] = [
  "extraction",
  "scope-escape",
  "grounding-bypass",
  "citation-forgery",
  "fake-source-injection",
];

export interface AttackResult {
  id: string;
  category: AttackCategory;
  attack_prompt: string;
  answer: string;
  refused: boolean;
  markers_used: number[];
  markers_dangling: number[]; // forged/invalid [N]s (citation-forgery signal)
  succeeded: boolean; // true = the attack breached the assistant
  latency_ms: number;
}

export interface CategoryASR {
  count: number;
  successes: number;
  asr: number; // successes / count — lower is better
}

export interface SecurityAggregates {
  attacks: number;
  successes: number;
  asr: number;
  by_category: Record<string, CategoryASR>;
}

export interface SecurityRun {
  created_at: string;
  label: string;
  chat_model: string;
  prompt_version: string;
  retriever: string;
  defense: string; // "baseline" | "defense-stack"
  aggregates: SecurityAggregates;
  results: AttackResult[];
}
