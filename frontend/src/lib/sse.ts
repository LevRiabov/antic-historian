/*
 * Streaming client for POST /ask.
 *
 * The browser's native EventSource only supports GET, but /ask is a POST (it
 * carries a JSON body) that replies with an SSE stream. So we drive it with
 * fetch + a ReadableStream reader and parse the `event:`/`data:` frames by hand.
 * Exposed as an async generator so callers can `for await` the typed events and
 * cancel via an AbortSignal.
 */
import { API_BASE } from "./api";
import type { AskEvent, AskMode } from "./types";

export interface AskParams {
  question: string;
  mode?: AskMode;
  topK?: number;
  signal?: AbortSignal;
}

/** Raised for a non-2xx response. `status` lets callers special-case 429 (rate
 *  limit / session cap) and 503 (deep mode unavailable). */
export class AskError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message);
    this.name = "AskError";
  }
}

const KNOWN_EVENTS = new Set(["meta", "sources", "step", "delta", "done"]);

export async function* askStream(params: AskParams): AsyncGenerator<AskEvent> {
  const { question, mode = "fast", topK = 5, signal } = params;

  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, mode, top_k: topK }),
    signal,
  });

  if (!res.ok || !res.body) {
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      body = await res.text().catch(() => undefined);
    }
    throw new AskError(`/ask failed: ${res.status}`, res.status, body);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      // Normalize CRLF -> LF: sse_starlette delimits frames with "\r\n\r\n", so a
      // bare "\n\n" search would never match. Re-normalizing the whole remaining
      // buffer each read also stitches a CRLF that straddled two chunks.
      buffer = (buffer + value).replace(/\r\n/g, "\n");

      // SSE frames are separated by a blank line. Process every complete frame
      // in the buffer; keep the trailing partial for the next chunk.
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
      }
    }
    // Flush a final frame that wasn't terminated by a blank line.
    const tail = parseFrame(buffer);
    if (tail) yield tail;
  } finally {
    reader.cancel().catch(() => undefined);
  }
}

function parseFrame(frame: string): AskEvent | null {
  let eventName = "message";
  const dataLines: string[] = [];

  for (const raw of frame.split("\n")) {
    const line = raw.replace(/\r$/, "");
    if (line.startsWith(":") || line.length === 0) continue; // comment / keepalive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") eventName = value;
    else if (field === "data") dataLines.push(value);
  }

  if (!KNOWN_EVENTS.has(eventName) || dataLines.length === 0) return null;
  const data: unknown = JSON.parse(dataLines.join("\n"));
  // The event name is the discriminant; the matching data shape is guaranteed by
  // the backend contract (lib/types.ts mirrors the pydantic models).
  return { event: eventName, data } as AskEvent;
}
