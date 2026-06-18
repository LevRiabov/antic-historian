import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { SourceCards } from "@/components/sources/SourceCards";
import { SourcesTable } from "@/components/sources/SourcesTable";
import { SummaryBar } from "@/components/sources/SummaryBar";
import { FilterTabs, type TabItem } from "@/components/ui/FilterTabs";
import { SearchInput } from "@/components/ui/SearchInput";
import { getSources } from "@/lib/api";
import {
  categoryCounts,
  selectSources,
  summarize,
  type SortKey,
  type SourceFilter,
} from "@/lib/sources";

export function Sources() {
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["sources"],
    queryFn: ({ signal }) => getSources(signal),
  });

  const [filter, setFilter] = useState<SourceFilter>("all");
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const sources = useMemo(() => data ?? [], [data]);
  const summary = useMemo(() => summarize(sources), [sources]);
  const counts = useMemo(() => categoryCounts(sources), [sources]);
  const rows = useMemo(
    () => selectSources(sources, { filter, query, sortKey, sortDir }),
    [sources, filter, query, sortKey, sortDir],
  );

  // Click a sorted column again to flip direction; a new column starts ascending.
  function handleSort(key: SortKey) {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  const tabs: TabItem<SourceFilter>[] = [
    { value: "all", label: "All", count: counts.all },
    { value: "primary", label: "Primary sources", count: counts.primary },
    { value: "scholarship", label: "Foundational scholarship", count: counts.scholarship },
  ];

  return (
    <section>
      <header className="mb-6">
        <h1 className="font-serif text-3xl font-semibold tracking-[0.2px] text-ink">
          The corpus
        </h1>
        <p className="mt-2 mb-4 max-w-2xl text-[15.5px] text-ink-soft">
          Every answer is grounded in these public-domain sources — nothing else. Here is the
          complete dataset.
        </p>
        {!isLoading && !isError && <SummaryBar summary={summary} />}
      </header>

      {isError ? (
        <ErrorPanel message={error instanceof Error ? error.message : "Request failed"} onRetry={() => void refetch()} />
      ) : isLoading ? (
        <TableSkeleton />
      ) : (
        <>
          <div className="my-4 flex flex-wrap items-center gap-3.5">
            <FilterTabs items={tabs} value={filter} onChange={setFilter} ariaLabel="Filter by type" />
            <div className="w-full sm:ml-auto sm:w-auto">
              <SearchInput
                value={query}
                onChange={setQuery}
                placeholder="Search author or work…"
                ariaLabel="Search author or work"
              />
            </div>
          </div>

          {/* Sorting lives in the table header on desktop; the card view has no
              header, so expose it here for small screens. */}
          <div className="mb-3 flex items-center gap-2 text-xs text-ink-soft lg:hidden">
            <span>Sort</span>
            <MobileSortButton label="Author" col="author" {...{ sortKey, sortDir, onSort: handleSort }} />
            <MobileSortButton label="Work" col="title" {...{ sortKey, sortDir, onSort: handleSort }} />
          </div>

          <SourcesTable rows={rows} sortKey={sortKey} sortDir={sortDir} onSort={handleSort} />
          <SourceCards rows={rows} />

          <footer className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2.5 border-t border-line pt-5 text-[13px] text-ink-soft">
            <span className="inline-flex items-center gap-2 font-semibold text-ink">
              <span className="text-[11px] text-good" aria-hidden>
                ●
              </span>
              EU-hosted · GDPR-native
            </span>
            <span className="max-w-2xl text-ink-faint">
              Public-domain basis = author/translator died &gt;70 years ago (EU rule). This repo
              documents the basis per work.
            </span>
          </footer>
        </>
      )}
    </section>
  );
}

function MobileSortButton({
  label,
  col,
  sortKey,
  sortDir,
  onSort,
}: {
  label: string;
  col: SortKey;
  sortKey: SortKey | null;
  sortDir: "asc" | "desc";
  onSort: (key: SortKey) => void;
}) {
  const active = sortKey === col;
  return (
    <button
      type="button"
      onClick={() => onSort(col)}
      aria-pressed={active}
      className={`inline-flex items-center gap-1 rounded-full px-3 py-1 font-medium transition-colors ${
        active
          ? "bg-accent-soft text-accent-ink"
          : "bg-surface text-ink-soft ring-1 ring-inset ring-line-strong"
      }`}
    >
      {label}
      <span className="text-[10px]" aria-hidden>
        {!active ? "⇅" : sortDir === "asc" ? "▲" : "▼"}
      </span>
    </button>
  );
}

function TableSkeleton() {
  return (
    <div className="mt-4 animate-pulse rounded-2xl border border-line bg-surface p-4 shadow-card">
      <div className="h-9 rounded bg-line/60" />
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="mt-3 h-6 rounded bg-line/40" />
      ))}
    </div>
  );
}

function ErrorPanel({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="mt-4 rounded-2xl border border-line bg-surface p-8 text-center shadow-card">
      <p className="text-sm text-refuse">Couldn’t load the corpus.</p>
      <p className="mt-1 font-mono text-xs text-ink-faint">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-4 rounded-full bg-accent-soft px-4 py-1.5 text-sm font-semibold text-accent-ink hover:bg-accent/10"
      >
        Retry
      </button>
    </div>
  );
}
