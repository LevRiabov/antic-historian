import { describe, expect, it } from "vitest";

import { categoryCounts, selectSources, summarize, type SourcesView } from "@/lib/sources";
import type { SourceCategory, SourceOut } from "@/lib/types";

function src(over: Partial<SourceOut> = {}): SourceOut {
  return {
    pg_id: 1,
    author: "Author",
    title: "Title",
    translator: "",
    category: "primary" as SourceCategory,
    pd_basis: "",
    source: "Project Gutenberg",
    landing_url: "",
    chunks: 0,
    ...over,
  };
}

const corpus: SourceOut[] = [
  src({ pg_id: 1, author: "Caesar", title: "Gallic War", category: "primary", chunks: 120 }),
  src({ pg_id: 2, author: "Caesar", title: "Civil War", category: "primary", chunks: 80 }),
  src({ pg_id: 3, author: "Mommsen", title: "History of Rome", category: "scholarship", chunks: 200 }),
];

const view = (over: Partial<SourcesView> = {}): SourcesView => ({
  filter: "all",
  query: "",
  sortKey: null,
  sortDir: "asc",
  ...over,
});

describe("summarize", () => {
  it("counts works, distinct authors, and total passages", () => {
    expect(summarize(corpus)).toEqual({ works: 3, authors: 2, passages: 400 });
  });
  it("is zeroed for an empty corpus", () => {
    expect(summarize([])).toEqual({ works: 0, authors: 0, passages: 0 });
  });
});

describe("categoryCounts", () => {
  it("splits primary vs scholarship and totals all", () => {
    expect(categoryCounts(corpus)).toEqual({ all: 3, primary: 2, scholarship: 1 });
  });
});

describe("selectSources", () => {
  it("filters by category", () => {
    const rows = selectSources(corpus, view({ filter: "scholarship" }));
    expect(rows.map((r) => r.pg_id)).toEqual([3]);
  });

  it("filters by free-text over author and title, case-insensitively", () => {
    expect(selectSources(corpus, view({ query: "caesar" })).length).toBe(2);
    expect(selectSources(corpus, view({ query: "GALLIC" })).map((r) => r.pg_id)).toEqual([1]);
    expect(selectSources(corpus, view({ query: "rome" })).map((r) => r.pg_id)).toEqual([3]);
  });

  it("trims the query and treats whitespace-only as no filter", () => {
    expect(selectSources(corpus, view({ query: "   " })).length).toBe(3);
  });

  it("sorts ascending and descending by a key", () => {
    const asc = selectSources(corpus, view({ sortKey: "title", sortDir: "asc" }));
    expect(asc.map((r) => r.title)).toEqual(["Civil War", "Gallic War", "History of Rome"]);
    const desc = selectSources(corpus, view({ sortKey: "title", sortDir: "desc" }));
    expect(desc.map((r) => r.title)).toEqual(["History of Rome", "Gallic War", "Civil War"]);
  });

  it("leaves order untouched when no sortKey is set", () => {
    const rows = selectSources(corpus, view());
    expect(rows.map((r) => r.pg_id)).toEqual([1, 2, 3]);
  });

  it("never mutates the input array", () => {
    const before = corpus.map((s) => s.pg_id);
    selectSources(corpus, view({ sortKey: "title", sortDir: "desc" }));
    expect(corpus.map((s) => s.pg_id)).toEqual(before);
  });
});
