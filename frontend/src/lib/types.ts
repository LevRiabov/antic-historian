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
