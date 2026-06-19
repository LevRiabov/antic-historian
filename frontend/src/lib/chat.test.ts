import { describe, expect, it } from "vitest";

import {
  citationByMarker,
  citedRetrieved,
  formatCost,
  formatLatency,
  isRefusal,
  newTurn,
  newTurnId,
  sessionTotals,
  type Turn,
} from "@/lib/chat";
import type { Citation, Cost, DoneEvent } from "@/lib/types";

function cost(usd: number | null): Cost {
  return { usd, input_tokens: 0, output_tokens: 0, model: "m", priced: usd !== null };
}

function done(over: Partial<DoneEvent> = {}): DoneEvent {
  return {
    answer: "a",
    refused: false,
    markers: { used: [], dangling: [] },
    usage: null,
    cost: null,
    served_by: null,
    blocked: false,
    ...over,
  };
}

/** A completed turn with the given done payload + client-measured latency. */
function settledTurn(over: Partial<DoneEvent>, elapsedMs: number | null = null): Turn {
  const t = newTurn("id", "q", "fast");
  return { ...t, status: "done", done: done(over), elapsedMs };
}

function citation(marker: number): Citation {
  return {
    marker,
    chunk_id: marker * 10,
    pg_id: 1,
    author: "Caesar",
    work_title: "Commentaries",
    locator: [],
    text: "…",
    score: 0.9,
    char_start: 0,
    char_end: 1,
  };
}

describe("newTurn", () => {
  it("starts streaming with empty accumulators", () => {
    const t = newTurn("x", "why?", "deep");
    expect(t).toMatchObject({ id: "x", question: "why?", mode: "deep", status: "streaming" });
    expect(t.answer).toBe("");
    expect(t.sources).toEqual([]);
    expect(t.steps).toEqual([]);
    expect(t.done).toBeNull();
    expect(t.elapsedMs).toBeNull();
  });
});

describe("newTurnId", () => {
  it("returns distinct non-empty ids on each call", () => {
    const a = newTurnId();
    const b = newTurnId();
    expect(a).toBeTruthy();
    expect(a).not.toBe(b);
  });
});

describe("formatCost", () => {
  it("renders priced cost to four decimals", () => {
    expect(formatCost(cost(0.0012))).toBe("$0.0012");
    expect(formatCost(cost(1.23456))).toBe("$1.2346"); // rounds
  });
  it("never shows $0 for an unpriced model — dashes instead (rule #6)", () => {
    expect(formatCost(cost(null))).toBe("—");
    expect(formatCost(null)).toBe("—");
  });
  it("renders a real zero cost distinctly from unpriced", () => {
    expect(formatCost(cost(0))).toBe("$0.0000");
  });
});

describe("formatLatency", () => {
  it("uses ms below one second", () => {
    expect(formatLatency(0)).toBe("0ms");
    expect(formatLatency(999)).toBe("999ms");
  });
  it("switches to seconds at the 1000ms boundary", () => {
    expect(formatLatency(1000)).toBe("1.0s");
    expect(formatLatency(1500)).toBe("1.5s");
    expect(formatLatency(12345)).toBe("12.3s");
  });
});

describe("isRefusal", () => {
  it("is false before the done event", () => {
    expect(isRefusal(newTurn("a", "q", "fast"))).toBe(false);
  });
  it("reflects done.refused", () => {
    expect(isRefusal(settledTurn({ refused: true }))).toBe(true);
    expect(isRefusal(settledTurn({ refused: false }))).toBe(false);
  });
});

describe("citationByMarker", () => {
  it("finds the citation with a matching marker", () => {
    const sources = [citation(1), citation(2)];
    expect(citationByMarker(sources, 2)?.chunk_id).toBe(20);
  });
  it("returns undefined for a dangling marker", () => {
    expect(citationByMarker([citation(1)], 9)).toBeUndefined();
  });
});

describe("citedRetrieved", () => {
  it("counts distinct used markers vs retrieved sources", () => {
    const t = settledTurn({ markers: { used: [1, 2], dangling: [] } });
    const withSources: Turn = { ...t, sources: [citation(1), citation(2), citation(3)] };
    expect(citedRetrieved(withSources)).toEqual({ cited: 2, retrieved: 3 });
  });
  it("is zero/zero before done with no sources", () => {
    expect(citedRetrieved(newTurn("a", "q", "fast"))).toEqual({ cited: 0, retrieved: 0 });
  });
});

describe("sessionTotals", () => {
  it("is all-null for an empty conversation", () => {
    expect(sessionTotals([])).toEqual({
      costUsd: null,
      avgLatencyMs: null,
      servedBy: null,
      answered: 0,
    });
  });

  it("sums priced costs and counts answered turns", () => {
    const turns = [
      settledTurn({ cost: cost(0.01), served_by: "deepseek" }, 1000),
      settledTurn({ cost: cost(0.02), served_by: "deepseek" }, 3000),
    ];
    const t = sessionTotals(turns);
    expect(t.answered).toBe(2);
    expect(t.costUsd).toBeCloseTo(0.03, 10);
    expect(t.avgLatencyMs).toBe(2000);
    expect(t.servedBy).toBe("deepseek");
  });

  it("keeps costUsd null when every answer is unpriced (not $0)", () => {
    const turns = [settledTurn({ cost: cost(null) }, 500)];
    const t = sessionTotals(turns);
    expect(t.costUsd).toBeNull();
    expect(t.answered).toBe(1);
    expect(t.avgLatencyMs).toBe(500);
  });

  it("ignores still-streaming turns in the answered count but uses their latency if set", () => {
    const streaming = newTurn("s", "q", "fast");
    const turns = [settledTurn({ cost: cost(0.05) }, 1000), streaming];
    const t = sessionTotals(turns);
    expect(t.answered).toBe(1); // only the done turn
    expect(t.avgLatencyMs).toBe(1000); // streaming turn had null elapsedMs
  });

  it("takes the most recent served_by", () => {
    const turns = [
      settledTurn({ served_by: "primary" }, 100),
      settledTurn({ served_by: "fallback" }, 100),
    ];
    expect(sessionTotals(turns).servedBy).toBe("fallback");
  });
});
