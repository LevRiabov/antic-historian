/*
 * Runtime validation for the JSON API responses (zod). types.ts stays the
 * hand-written contract mirror of the backend pydantic models; these schemas
 * validate the wire against it so a backend schema drift surfaces as a clear,
 * caught error at the fetch boundary — handled by the routes' React Query error
 * states — instead of an undefined-access crash deep inside a render.
 *
 * Each parse function's declared return type is the matching interface from
 * types.ts, so tsc fails the build if a schema stops producing that shape: the
 * interface and the schema can't silently drift apart. Object schemas are
 * non-strict (zod's default), so EXTRA backend fields pass through untouched —
 * the records carry more than the page reads (see types.ts), and that's fine.
 */
import { z } from "zod";

import type {
  ChunkOut,
  GenerationRun,
  RetrievalRun,
  SecurityRun,
  SourceOut,
} from "./types";

const sourceCategory = z.enum(["primary", "scholarship"]);
const category = z.enum([
  "literal",
  "synonym",
  "multi-hop",
  "synthesis",
  "cross-book",
  "contradiction",
  "out-of-scope",
]);
const attackCategory = z.enum([
  "extraction",
  "scope-escape",
  "grounding-bypass",
  "citation-forgery",
  "fake-source-injection",
]);

/** k -> recall@k; JSON serializes the int keys as strings (see types.ts RecallByK). */
const recallByK = z.record(z.string(), z.number());

const sourceOutSchema = z.object({
  pg_id: z.number(),
  author: z.string(),
  title: z.string(),
  translator: z.string(),
  category: sourceCategory,
  pd_basis: z.string(),
  source: z.string(),
  landing_url: z.string(),
  chunks: z.number(),
});

const chunkOutSchema = z.object({
  chunk_id: z.number(),
  pg_id: z.number(),
  author: z.string(),
  work_title: z.string(),
  locator: z.array(z.string()),
  heading: z.string().nullable(),
  text: z.string(),
  char_start: z.number(),
  char_end: z.number(),
  pd_basis: z.string(),
});

const goldSpanRefSchema = z.object({
  pg_id: z.number(),
  char_start: z.number(),
  char_end: z.number(),
  groups: z.array(z.string()),
});

const retrievalQuestionResultSchema = z.object({
  question_id: z.string(),
  category,
  question: z.string(),
  ideal_answer: z.string(),
  gold_spans: z.array(goldSpanRefSchema),
  gold_chunk_ids: z.array(z.number()),
  retrieved_chunk_ids: z.array(z.number()),
  recall: recallByK,
  first_hit_rank: z.number().nullable(),
  mrr: z.number(),
  latency_ms: z.number(),
});

const retrievalCategoryAggregateSchema = z.object({
  count: z.number(),
  recall: recallByK,
  mrr: z.number(),
});

const retrievalAggregatesSchema = z.object({
  recall: recallByK,
  mrr: z.number(),
  by_category: z.record(z.string(), retrievalCategoryAggregateSchema),
});

const retrievalRunSchema = z.object({
  created_at: z.string(),
  retriever: z.string(),
  embed_model: z.string(),
  top_k: z.number(),
  aggregates: retrievalAggregatesSchema,
  results: z.array(retrievalQuestionResultSchema),
});

const genQuestionResultSchema = z.object({
  question_id: z.string(),
  category,
  question: z.string(),
  ideal_answer: z.string(),
  answer: z.string(),
  refused: z.boolean(),
  refusal_expected: z.boolean(),
  refusal_correct: z.boolean(),
  markers_used: z.array(z.number()),
  retrieved_chunk_ids: z.array(z.number()),
  cited_chunk_ids: z.array(z.number()),
  faithfulness: z.number().nullable(),
  completeness: z.number().nullable(),
  attribution: z.number().nullable(),
  judge_notes: z.string(),
  latency_ms: z.number(),
});

const genCategoryAggregateSchema = z.object({
  count: z.number(),
  refused: z.number(),
  refusal_correct: z.number(),
  faithfulness: z.number().nullable(),
  completeness: z.number().nullable(),
  attribution: z.number().nullable(),
  mean_latency_ms: z.number(),
});

const genAggregatesSchema = z.object({
  questions: z.number(),
  refusal_accuracy_oos: z.number().nullable(),
  false_refusal_rate: z.number(),
  faithfulness: z.number().nullable(),
  completeness: z.number().nullable(),
  attribution: z.number().nullable(),
  mean_latency_ms: z.number(),
  by_category: z.record(z.string(), genCategoryAggregateSchema),
});

const generationRunSchema = z.object({
  created_at: z.string(),
  label: z.string(),
  chat_model: z.string(),
  embed_model: z.string(),
  prompt_version: z.string(),
  top_k: z.number(),
  engine: z.string(),
  judge_model: z.string().nullable(),
  judge_rubric: z.string().nullable(),
  aggregates: genAggregatesSchema,
  results: z.array(genQuestionResultSchema),
});

const attackResultSchema = z.object({
  id: z.string(),
  category: attackCategory,
  attack_prompt: z.string(),
  answer: z.string(),
  refused: z.boolean(),
  markers_used: z.array(z.number()),
  markers_dangling: z.array(z.number()),
  succeeded: z.boolean(),
  latency_ms: z.number(),
});

const categoryAsrSchema = z.object({
  count: z.number(),
  successes: z.number(),
  asr: z.number(),
});

const securityAggregatesSchema = z.object({
  attacks: z.number(),
  successes: z.number(),
  asr: z.number(),
  by_category: z.record(z.string(), categoryAsrSchema),
});

const securityRunSchema = z.object({
  created_at: z.string(),
  label: z.string(),
  chat_model: z.string(),
  prompt_version: z.string(),
  retriever: z.string(),
  defense: z.string(),
  aggregates: securityAggregatesSchema,
  results: z.array(attackResultSchema),
});

/* Parse helpers — the declared return type pins each schema to its types.ts
 * interface (tsc errors if a schema stops covering it). They throw a ZodError on a
 * shape mismatch, which the API client wraps into a thrown Error for React Query. */
export function parseSources(data: unknown): SourceOut[] {
  return z.array(sourceOutSchema).parse(data);
}

export function parseChunks(data: unknown): ChunkOut[] {
  return z.array(chunkOutSchema).parse(data);
}

export function parseRagEval(data: unknown): RetrievalRun {
  return retrievalRunSchema.parse(data);
}

export function parseAgentEval(data: unknown): GenerationRun {
  return generationRunSchema.parse(data);
}

export function parseSecurityRun(data: unknown): SecurityRun {
  return securityRunSchema.parse(data);
}
