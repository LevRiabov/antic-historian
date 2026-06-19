import { afterEach, describe, expect, it, vi } from "vitest";

import { AskError, askStream } from "@/lib/sse";
import type { AskEvent } from "@/lib/types";

/* Drive the real askStream generator against a mocked streaming fetch. This is
 * the public contract — frame splitting, CRLF normalization, partial-chunk
 * stitching, keepalive/unknown-event filtering, and the trailing-frame flush all
 * run for real; only the network is faked. */

const enc = new TextEncoder();

/** A 200 response whose body streams the given raw strings as separate chunks
 *  (each chunk is one `reader.read()`), letting us split frames across reads. */
function streamResponse(chunks: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  return { ok: true, status: 200, body } as unknown as Response;
}

function errorResponse(status: number, json?: unknown): Response {
  return {
    ok: false,
    status,
    body: null,
    json: async () => {
      if (json === undefined) throw new Error("no json");
      return json;
    },
    text: async () => "error body",
  } as unknown as Response;
}

function mockFetch(res: Response) {
  const fn = vi.fn().mockResolvedValue(res);
  vi.stubGlobal("fetch", fn);
  return fn;
}

async function collect(gen: AsyncGenerator<AskEvent>): Promise<AskEvent[]> {
  const out: AskEvent[] = [];
  for await (const ev of gen) out.push(ev);
  return out;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("askStream — request shaping", () => {
  it("POSTs JSON with mode/top_k defaults and the SSE Accept header", async () => {
    const fetchFn = mockFetch(streamResponse(["event: delta\ndata: {\"text\":\"hi\"}\n\n"]));
    await collect(askStream({ question: "  who?  " }));

    expect(fetchFn).toHaveBeenCalledOnce();
    const [url, init] = fetchFn.mock.calls[0]!;
    expect(String(url)).toMatch(/\/ask$/);
    expect(init.method).toBe("POST");
    expect(init.headers.Accept).toBe("text/event-stream");
    expect(JSON.parse(init.body)).toEqual({ question: "  who?  ", mode: "fast", top_k: 5 });
  });

  it("passes through an explicit mode and topK", async () => {
    const fetchFn = mockFetch(streamResponse(["event: delta\ndata: {\"text\":\"x\"}\n\n"]));
    await collect(askStream({ question: "q", mode: "deep", topK: 12 }));
    expect(JSON.parse(fetchFn.mock.calls[0]![1].body)).toEqual({
      question: "q",
      mode: "deep",
      top_k: 12,
    });
  });
});

describe("askStream — frame parsing", () => {
  it("yields multiple frames packed into a single chunk, in order", async () => {
    mockFetch(
      streamResponse([
        'event: meta\ndata: {"limit":0,"remaining":0}\n\n' +
          'event: delta\ndata: {"text":"a"}\n\n',
      ]),
    );
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([
      { event: "meta", data: { limit: 0, remaining: 0 } },
      { event: "delta", data: { text: "a" } },
    ]);
  });

  it("stitches a frame that is split across two reads", async () => {
    mockFetch(streamResponse(['event: delta\ndata: {"text":"He', 'llo"}\n\n']));
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([{ event: "delta", data: { text: "Hello" } }]);
  });

  it("normalizes CRLF frame delimiters (sse_starlette emits \\r\\n\\r\\n)", async () => {
    mockFetch(streamResponse(['event: meta\r\ndata: {"limit":5,"remaining":4}\r\n\r\n']));
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([{ event: "meta", data: { limit: 5, remaining: 4 } }]);
  });

  it("joins multi-line data fields with a newline", async () => {
    // Two data: lines join to `{"text":\n"hi"}`, which is valid JSON.
    mockFetch(streamResponse(['event: delta\ndata: {"text":\ndata: "hi"}\n\n']));
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([{ event: "delta", data: { text: "hi" } }]);
  });

  it("ignores keepalive comments and frames with no data line", async () => {
    mockFetch(
      streamResponse([": keepalive\n\n", 'event: delta\ndata: {"text":"a"}\n\n', ": ping\n\n"]),
    );
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([{ event: "delta", data: { text: "a" } }]);
  });

  it("drops events whose name is not in the known set", async () => {
    mockFetch(
      streamResponse(['event: bogus\ndata: {"x":1}\n\nevent: delta\ndata: {"text":"a"}\n\n']),
    );
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([{ event: "delta", data: { text: "a" } }]);
  });

  it("flushes a final frame not terminated by a blank line", async () => {
    mockFetch(streamResponse(['event: done\ndata: {"answer":"x","refused":false}']));
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([{ event: "done", data: { answer: "x", refused: false } }]);
  });

  it("skips a frame with malformed JSON but keeps the surrounding stream", async () => {
    mockFetch(
      streamResponse([
        'event: delta\ndata: {"text":"ok"}\n\n' +
          "event: delta\ndata: {not json}\n\n" +
          'event: done\ndata: {"answer":"x","refused":false}\n\n',
      ]),
    );
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([
      { event: "delta", data: { text: "ok" } },
      { event: "done", data: { answer: "x", refused: false } },
    ]);
  });

  it("yields the backend's terminal `error` frame instead of dropping it", async () => {
    // Regression: "error" must be a known event, else a failed/timed-out stream
    // looks like a clean close and a truncated answer is shown as success.
    mockFetch(
      streamResponse([
        'event: delta\ndata: {"text":"half"}\n\n' +
          'event: error\ndata: {"detail":"the answer stream failed"}\n\n',
      ]),
    );
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([
      { event: "delta", data: { text: "half" } },
      { event: "error", data: { detail: "the answer stream failed" } },
    ]);
  });

  it("strips a single leading space after the field colon", async () => {
    // "data: x" -> "x" (one space stripped), but interior spacing is preserved.
    mockFetch(streamResponse(['event: delta\ndata: {"text":"a b"}\n\n']));
    const events = await collect(askStream({ question: "q" }));
    expect(events).toEqual([{ event: "delta", data: { text: "a b" } }]);
  });
});

describe("askStream — error handling", () => {
  it("throws AskError carrying the status and parsed JSON body on non-2xx", async () => {
    mockFetch(errorResponse(429, { detail: { error: "session_cap_reached" } }));
    const err = await collect(askStream({ question: "q" })).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(AskError);
    const asked = err as AskError;
    expect(asked.status).toBe(429);
    expect(asked.body).toEqual({ detail: { error: "session_cap_reached" } });
  });

  it("falls back to text when the error body is not JSON", async () => {
    mockFetch(errorResponse(503));
    const err = (await collect(askStream({ question: "q" })).catch((e: unknown) => e)) as AskError;
    expect(err).toBeInstanceOf(AskError);
    expect(err.status).toBe(503);
    expect(err.body).toBe("error body");
  });
});
