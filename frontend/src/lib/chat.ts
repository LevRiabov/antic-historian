/*
 * Chat domain model + pure helpers for the /ask page. A "turn" is one
 * question→answer exchange; the SSE stream (meta → sources → step* → delta* →
 * done) mutates a turn in place as it arrives. Framework-free; the React page
 * owns the state, this module just shapes + formats it.
 */
import type { AskMode, Citation, Cost, DoneEvent, SessionStatus, StepEvent } from "./types";

export type TurnStatus = "streaming" | "done" | "error";

export interface Turn {
  id: string;
  question: string;
  mode: AskMode;
  status: TurnStatus;
  answer: string; // accumulated delta text
  sources: Citation[]; // from the `sources` event (full citation data — no fetch needed)
  steps: StepEvent[]; // live ReAct steps (deep mode only)
  done: DoneEvent | null; // terminal event
  session: SessionStatus | null; // the `meta` budget for this turn
  error: string | null;
  elapsedMs: number | null; // client-measured send→done latency
}

export function newTurn(id: string, question: string, mode: AskMode): Turn {
  return {
    id,
    question,
    mode,
    status: "streaming",
    answer: "",
    sources: [],
    steps: [],
    done: null,
    session: null,
    error: null,
    elapsedMs: null,
  };
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

export const SUGGESTIONS: readonly Suggestion[] = [
  { kind: "Try this", text: "Did Caesar cross the Rubicon, and what did he say?" },
  { kind: "Try this", text: "How did the Battle of Cannae unfold?" },
  { kind: "Try this", text: "Why did the Roman Republic fall?" },
  { kind: "See a refusal", text: "What did Caesar think of the printing press?", refusal: true },
];
