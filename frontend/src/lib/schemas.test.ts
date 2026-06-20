import { describe, expect, it } from "vitest";

import { parseChunks, parseSources } from "./schemas";

/* The schemas are the fetch-boundary guard (lib/api.ts): a valid response parses
 * into the typed shape, and a drifted one throws a clear error instead of letting an
 * undefined field crash a later render. We pin both directions here. */

const validSource = {
  pg_id: 1,
  author: "Plutarch",
  title: "Lives",
  translator: "Dryden",
  category: "primary",
  pd_basis: "EU PD",
  source: "Project Gutenberg",
  landing_url: "https://example.org",
  chunks: 42,
};

describe("parseSources", () => {
  it("accepts a well-formed response and passes through unknown extra fields", () => {
    const parsed = parseSources([{ ...validSource, extra_backend_field: "ignored" }]);
    expect(parsed[0]?.author).toBe("Plutarch");
  });

  it("rejects a wrong field type (drift surfaces as a thrown error, not a crash)", () => {
    expect(() => parseSources([{ ...validSource, chunks: "not-a-number" }])).toThrow();
  });

  it("rejects an unknown category enum value", () => {
    expect(() => parseSources([{ ...validSource, category: "tertiary" }])).toThrow();
  });

  it("rejects a missing required field", () => {
    const withoutAuthor = { ...validSource } as Record<string, unknown>;
    delete withoutAuthor.author;
    expect(() => parseSources([withoutAuthor])).toThrow();
  });
});

describe("parseChunks", () => {
  it("accepts a nullable heading as null", () => {
    const parsed = parseChunks([
      {
        chunk_id: 7,
        pg_id: 1,
        author: "Plutarch",
        work_title: "Lives",
        locator: ["Caesar", "66"],
        heading: null,
        text: "…",
        char_start: 0,
        char_end: 10,
        pd_basis: "EU PD",
      },
    ]);
    expect(parsed[0]?.heading).toBeNull();
  });
});
