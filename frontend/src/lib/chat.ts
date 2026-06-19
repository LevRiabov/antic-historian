/*
 * Chat domain model + pure helpers for the /ask page. A "turn" is one
 * question→answer exchange; the SSE stream (meta → sources → step* → delta* →
 * done) mutates a turn in place as it arrives. Framework-free; the React page
 * owns the state, this module just shapes + formats it.
 */
import type {
  AskEvent,
  AskMode,
  Citation,
  Cost,
  DoneEvent,
  SessionStatus,
  StepEvent,
} from "./types";

export type TurnStatus = "streaming" | "done" | "error";

export interface Turn {
  id: string;
  question: string;
  mode: AskMode;
  status: TurnStatus;
  answer: string; // accumulated delta text
  reasoning: string; // accumulated live chain-of-thought (display-only; reasoning models)
  sources: Citation[]; // from the `sources` event (full citation data — no fetch needed)
  steps: StepEvent[]; // live ReAct steps (deep mode only)
  done: DoneEvent | null; // terminal event
  session: SessionStatus | null; // the `meta` budget for this turn
  error: string | null;
  elapsedMs: number | null; // client-measured send→done latency
}

/** A unique id for a turn. crypto.randomUUID is only defined in a secure context
 *  (HTTPS / localhost); over plain HTTP it's undefined, so fall back to a
 *  good-enough random id — a turn id is a client-only React key, not a secret. */
export function newTurnId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `t-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function newTurn(id: string, question: string, mode: AskMode): Turn {
  return {
    id,
    question,
    mode,
    status: "streaming",
    answer: "",
    reasoning: "",
    sources: [],
    steps: [],
    done: null,
    session: null,
    error: null,
    elapsedMs: null,
  };
}

/** Shown when the stream fails mid-answer (the backend's terminal `error` frame, or
 *  any other case the reducer marks failed). Friendly, not the raw backend detail. */
export const STREAM_ERROR_MESSAGE =
  "The answer stream failed before it finished. Please try again.";

/* The wire order is meta → sources → (step* in deep mode) → delta* → (done | error).
 * `applyAskEvent` folds one event into the turn; the React page just stores the
 * result. Keeping it a pure function (not inline in the component) is what lets the
 * full stream-reducer contract — including the terminal `error` frame — be unit
 * tested. `elapsedMs` is the caller's client-measured send→event latency; it's only
 * recorded on the terminal events (done/error) and ignored otherwise. */
export function applyAskEvent(turn: Turn, ev: AskEvent, elapsedMs: number): Turn {
  switch (ev.event) {
    case "meta":
      return { ...turn, session: ev.data };
    case "sources":
      return { ...turn, sources: ev.data.citations };
    case "step":
      return { ...turn, steps: [...turn.steps, ev.data] };
    case "delta":
      return { ...turn, answer: turn.answer + ev.data.text };
    case "reasoning":
      return { ...turn, reasoning: turn.reasoning + ev.data.text };
    case "done":
      return { ...turn, done: ev.data, status: "done", elapsedMs };
    case "error":
      // Terminal failure: keep any partial answer for context but mark the turn
      // failed so the UI shows an error, never a truncated answer as success.
      return { ...turn, status: "error", error: STREAM_ERROR_MESSAGE, elapsedMs };
  }
}

/** Whether this turn is an honest refusal (no source in corpus). Known only once
 *  the done event arrives. */
export function isRefusal(turn: Turn): boolean {
  return turn.done?.refused ?? false;
}

export function citationByMarker(sources: readonly Citation[], marker: number): Citation | undefined {
  return sources.find((c) => c.marker === marker);
}

/** Cited (distinct markers actually used) vs retrieved (passages shown to the model). */
export function citedRetrieved(turn: Turn): { cited: number; retrieved: number } {
  return { cited: turn.done?.markers.used.length ?? 0, retrieved: turn.sources.length };
}

export function formatCost(cost: Cost | null): string {
  // Never silently show $0 for an unpriced model — that's a real distinction (rule #6).
  if (!cost || cost.usd === null) return "—";
  return `$${cost.usd.toFixed(4)}`;
}

export function formatLatency(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export interface SessionTotals {
  costUsd: number | null; // null until at least one priced answer
  avgLatencyMs: number | null;
  servedBy: string | null;
  answered: number;
}

/** Roll up the session instruments from completed turns (all client-derived from
 *  the per-answer done events — real numbers, not asserted). */
export function sessionTotals(turns: readonly Turn[]): SessionTotals {
  let cost = 0;
  let anyPriced = false;
  let servedBy: string | null = null;
  let answered = 0;
  const latencies: number[] = [];
  for (const t of turns) {
    if (t.done) {
      answered += 1;
      if (t.done.cost?.usd != null) {
        cost += t.done.cost.usd;
        anyPriced = true;
      }
      if (t.done.served_by) servedBy = t.done.served_by;
    }
    if (t.elapsedMs != null) latencies.push(t.elapsedMs);
  }
  return {
    costUsd: anyPriced ? cost : null,
    avgLatencyMs: latencies.length
      ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length)
      : null,
    servedBy,
    answered,
  };
}

/** The landing-state starter prompts (the last one demonstrates an honest refusal). */
export interface Suggestion {
  kind: string; // small uppercase label
  text: string;
  refusal?: boolean;
}

// Curated from the golden set so the demo's first impression is reliable: each of
// the first three is a top-scoring in-scope question (faithfulness/completeness/
// attribution all 5/5 in the published gen-agent-v8 run that /evals/agent serves),
// spanning the difficulty range — a single fact, a multi-hop chain, and a
// cross-source synthesis. The fourth is a golden out-of-scope question the eval
// confirms is correctly refused (refused + refusal_correct), so the "honest refusal"
// demo lands every time. If the published run changes, re-pick from the new top scorers.
export const SUGGESTIONS: readonly Suggestion[] = [
  {
    kind: "Simple fact",
    text: "How many wounds did Julius Caesar receive when he was assassinated?",
  },
  {
    kind: "Multi-hop",
    text: "What city did Alexander found in memory of the horse he had tamed as a boy?",
  },
  {
    kind: "Cross-source synthesis",
    text: "How do the sources describe the final destruction of Carthage?",
  },
  {
    kind: "See a refusal",
    text: "What was the Antikythera mechanism used for?",
    refusal: true,
  },
];
