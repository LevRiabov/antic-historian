/*
 * Pure transforms for the /sources page — filtering, sorting, and the summary
 * stats. Kept framework-free and side-effect-free (no React, no fetch) so the
 * table logic is trivially testable and the component stays declarative.
 */
import type { SourceCategory, SourceOut } from "./types";

export type SourceFilter = "all" | SourceCategory;
export type SortKey = "author" | "title";
export type SortDir = "asc" | "desc";

export interface SourcesView {
  filter: SourceFilter;
  query: string;
  sortKey: SortKey | null;
  sortDir: SortDir;
}

export interface SourcesSummary {
  works: number; // rows as the API returns them (one per volume of a set)
  authors: number; // distinct authors
  passages: number; // total retrievable chunks across the corpus
}

export function summarize(sources: readonly SourceOut[]): SourcesSummary {
  const authors = new Set(sources.map((s) => s.author));
  const passages = sources.reduce((sum, s) => sum + s.chunks, 0);
  return { works: sources.length, authors: authors.size, passages };
}

export interface CategoryCounts {
  all: number;
  primary: number;
  scholarship: number;
}

export function categoryCounts(sources: readonly SourceOut[]): CategoryCounts {
  let primary = 0;
  for (const s of sources) if (s.category === "primary") primary++;
  return { all: sources.length, primary, scholarship: sources.length - primary };
}

/** Filter by category + free-text (author/title), then optionally sort. Returns a
 *  new array; never mutates the input. */
export function selectSources(sources: readonly SourceOut[], view: SourcesView): SourceOut[] {
  const q = view.query.trim().toLowerCase();
  const rows = sources.filter((s) => {
    if (view.filter !== "all" && s.category !== view.filter) return false;
    if (q && !s.author.toLowerCase().includes(q) && !s.title.toLowerCase().includes(q)) {
      return false;
    }
    return true;
  });

  if (view.sortKey) {
    const key = view.sortKey;
    const dir = view.sortDir === "asc" ? 1 : -1;
    rows.sort((a, b) => a[key].localeCompare(b[key]) * dir);
  }
  return rows;
}
