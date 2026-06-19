import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { CategoryAggTable } from "@/components/evals/CategoryAggTable";
import { EvalStatBar } from "@/components/evals/EvalStatBar";
import { GoldenTable } from "@/components/evals/GoldenTable";
import { FilterTabs, type TabItem } from "@/components/ui/FilterTabs";
import { SearchInput } from "@/components/ui/SearchInput";
import { ErrorPanel, TableSkeleton } from "@/components/ui/StatePanels";
import { getAgentEval, getRagEval, getSources } from "@/lib/api";
import {
  buildCategoryAggRows,
  buildStatTiles,
  categoryCounts,
  CAT_META,
  CAT_ORDER,
  FAILURE_HINT,
  mergeGolden,
  selectRows,
  worksByPgId,
  type EvalFilter,
} from "@/lib/evals";

export function Evals() {
  // Two handlers build the table; /sources only labels the gold chunks in the
  // detail panel, so it's non-blocking (an empty map just falls back to "pg N").
  const rag = useQuery({ queryKey: ["eval-rag"], queryFn: ({ signal }) => getRagEval(signal) });
  const agent = useQuery({
    queryKey: ["eval-agent"],
    queryFn: ({ signal }) => getAgentEval(signal),
  });
  const sources = useQuery({ queryKey: ["sources"], queryFn: ({ signal }) => getSources(signal) });

  const [filter, setFilter] = useState<EvalFilter>("all");
  const [query, setQuery] = useState("");
  const [failuresOnly, setFailuresOnly] = useState(false);

  const allRows = useMemo(
    () => (rag.data && agent.data ? mergeGolden(rag.data, agent.data) : []),
    [rag.data, agent.data],
  );
  const counts = useMemo(() => categoryCounts(allRows), [allRows]);
  const rows = useMemo(
    () => selectRows(allRows, { filter, query, failuresOnly }),
    [allRows, filter, query, failuresOnly],
  );
  const works = useMemo(() => worksByPgId(sources.data ?? []), [sources.data]);
  const tiles = useMemo(
    () =>
      rag.data && agent.data
        ? buildStatTiles(rag.data.aggregates, agent.data.aggregates, allRows)
        : [],
    [rag.data, agent.data, allRows],
  );
  const catRows = useMemo(
    () =>
      rag.data && agent.data
        ? buildCategoryAggRows(rag.data.aggregates, agent.data.aggregates)
        : [],
    [rag.data, agent.data],
  );

  const isLoading = rag.isLoading || agent.isLoading;
  const isError = rag.isError || agent.isError;
  const error = rag.error ?? agent.error;

  const tabs: TabItem<EvalFilter>[] = [
    { value: "all", label: "All", count: counts.all },
    ...CAT_ORDER.map((cat) => ({
      value: cat,
      label: cat,
      count: counts[cat],
      color: CAT_META[cat].color,
    })),
  ];

  return (
    <section>
      <header className="mb-2">
        <h1 className="font-serif text-3xl font-semibold tracking-[0.2px] text-ink">
          Golden set — how it performs, measured
        </h1>
        <p className="mt-2 max-w-3xl text-[15.5px] text-ink-soft">
          Every retrieval and generation decision in this project is justified against this question
          set. Browse the full set — including the hard categories and the failures.
        </p>
      </header>

      {isError ? (
        <ErrorPanel
          title="Couldn’t load the eval runs."
          message={error instanceof Error ? error.message : "Request failed"}
          onRetry={() => {
            void rag.refetch();
            void agent.refetch();
          }}
        />
      ) : isLoading ? (
        <TableSkeleton rows={10} />
      ) : (
        <>
          <EvalStatBar tiles={tiles} />
          <CategoryAggTable rows={catRows} />

          <div className="mt-5 flex flex-wrap items-center gap-3 border-b border-line pb-3">
            <FilterTabs items={tabs} value={filter} onChange={setFilter} ariaLabel="Filter by category" />
            <div className="w-full sm:ml-auto sm:w-auto">
              <SearchInput
                value={query}
                onChange={setQuery}
                placeholder="Search by id or question…"
                ariaLabel="Search by id or question"
              />
            </div>
            <FailuresToggle on={failuresOnly} onToggle={() => setFailuresOnly((v) => !v)} />
            <span className="font-mono text-[12.5px] text-ink-faint sm:ml-auto">
              Showing {rows.length} of {allRows.length}
            </span>
          </div>

          <GoldenTable rows={rows} works={works} />

          <p className="mt-4 text-[12.5px] leading-relaxed text-ink-faint">
            <b className="font-semibold text-ink-soft">Two metric tiers:</b> retrieval (recall@k,
            MRR — free, run constantly) and judge (faithfulness / completeness / attribution /
            refusal — LLM-judged at phase boundaries). Out-of-scope questions have no retrieval
            metrics; an honest refusal is the correct answer.
          </p>
        </>
      )}
    </section>
  );
}

/** A small switch that narrows the table to failing rows (see isFailure). */
function FailuresToggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      onClick={onToggle}
      title={FAILURE_HINT}
      className={`inline-flex items-center gap-2.5 rounded-lg border px-3 py-1.5 text-[13.5px] font-medium shadow-card transition-colors ${
        on ? "border-refuse text-refuse" : "border-line text-ink-soft hover:border-line-strong"
      }`}
    >
      <span
        className={`relative h-[19px] w-[34px] flex-none rounded-full transition-colors ${
          on ? "bg-refuse" : "bg-line-strong"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 h-[15px] w-[15px] rounded-full bg-white shadow transition-transform ${
            on ? "translate-x-[15px]" : ""
          }`}
        />
      </span>
      Failures only
    </button>
  );
}
