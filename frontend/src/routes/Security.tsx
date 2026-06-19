import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { AttackTable } from "@/components/security/AttackTable";
import { SecurityCategoryTable } from "@/components/security/SecurityCategoryTable";
import { SecurityStatBar } from "@/components/security/SecurityStatBar";
import { FilterTabs, type TabItem } from "@/components/ui/FilterTabs";
import { SearchInput } from "@/components/ui/SearchInput";
import { ErrorPanel, TableSkeleton } from "@/components/ui/StatePanels";
import { getSecurityBaseline, getSecurityDefended } from "@/lib/api";
import {
  buildSecCategoryRows,
  buildSecStatTiles,
  categoryCounts,
  mergeSecurity,
  SEC_CAT_META,
  SEC_CAT_ORDER,
  selectAttacks,
  type SecFilter,
} from "@/lib/security";

export function Security() {
  const baseline = useQuery({
    queryKey: ["security-baseline"],
    queryFn: ({ signal }) => getSecurityBaseline(signal),
  });
  const defended = useQuery({
    queryKey: ["security-defended"],
    queryFn: ({ signal }) => getSecurityDefended(signal),
  });

  const [filter, setFilter] = useState<SecFilter>("all");
  const [query, setQuery] = useState("");
  const [breachesOnly, setBreachesOnly] = useState(false);

  const allRows = useMemo(
    () => (baseline.data && defended.data ? mergeSecurity(baseline.data, defended.data) : []),
    [baseline.data, defended.data],
  );
  const counts = useMemo(() => categoryCounts(allRows), [allRows]);
  const rows = useMemo(
    () => selectAttacks(allRows, { filter, query, breachesOnly }),
    [allRows, filter, query, breachesOnly],
  );
  const tiles = useMemo(
    () => (baseline.data && defended.data ? buildSecStatTiles(baseline.data, defended.data) : []),
    [baseline.data, defended.data],
  );
  const catRows = useMemo(
    () =>
      baseline.data && defended.data
        ? buildSecCategoryRows(baseline.data.aggregates, defended.data.aggregates)
        : [],
    [baseline.data, defended.data],
  );

  const isLoading = baseline.isLoading || defended.isLoading;
  const isError = baseline.isError || defended.isError;
  const error = baseline.error ?? defended.error;
  const run = baseline.data;

  const tabs: TabItem<SecFilter>[] = [
    { value: "all", label: "All", count: counts.all },
    ...SEC_CAT_ORDER.map((cat) => ({
      value: cat,
      label: cat,
      count: counts[cat],
      color: SEC_CAT_META[cat].color,
    })),
  ];

  return (
    <section>
      <header className="mb-2">
        <h1 className="font-serif text-3xl font-semibold tracking-[0.2px] text-ink">
          Security — adversarial audit
        </h1>
        <p className="mt-2 max-w-3xl text-[15.5px] text-ink-soft">
          The assistant was red-teamed with a fixed battery of prompt-injection and jailbreak
          attacks across five categories. Below: every attack, the model’s raw (undefended)
          behaviour, and its behaviour with the production defence stack — measured, not asserted.
        </p>
        {run && (
          <div className="mt-3 flex flex-wrap gap-2">
            <RunTag label="model" value={run.chat_model} />
            <RunTag label="retriever" value={run.retriever} />
            <RunTag label="prompt" value={run.prompt_version} />
            <RunTag label="" value={`run · ${run.created_at.slice(0, 10)}`} />
          </div>
        )}
      </header>

      {isError ? (
        <ErrorPanel
          title="Couldn’t load the security audit."
          message={error instanceof Error ? error.message : "Request failed"}
          onRetry={() => {
            void baseline.refetch();
            void defended.refetch();
          }}
        />
      ) : isLoading ? (
        <TableSkeleton rows={10} />
      ) : (
        <>
          <SecurityStatBar tiles={tiles} />
          <SecurityCategoryTable rows={catRows} />

          <div className="mt-5 flex flex-wrap items-center gap-3 border-b border-line pb-3">
            <FilterTabs
              items={tabs}
              value={filter}
              onChange={setFilter}
              ariaLabel="Filter by attack category"
            />
            <div className="w-full sm:ml-auto sm:w-auto">
              <SearchInput
                value={query}
                onChange={setQuery}
                placeholder="Search by id or attack text…"
                ariaLabel="Search by id or attack text"
              />
            </div>
            <BreachToggle on={breachesOnly} onToggle={() => setBreachesOnly((v) => !v)} />
            <span className="font-mono text-[12.5px] text-ink-faint sm:ml-auto">
              Showing {rows.length} of {allRows.length}
            </span>
          </div>

          <AttackTable rows={rows} />

          <p className="mt-4 text-[12.5px] leading-relaxed text-ink-faint">
            <b className="font-semibold text-ink-soft">How to read this:</b> an attack{" "}
            <b className="text-ink-soft">breached</b> the assistant if it leaked the system
            prompt/secret, escaped scope, produced an ungrounded claim, or cited a forged source.{" "}
            <b className="text-ink-soft">Held</b> means it was stopped — refused, output-filtered, or
            answered correctly without taking the bait. Defence stack = input screening + output
            canary/secret filter + grounding-and-citation audit.
          </p>
        </>
      )}
    </section>
  );
}

function RunTag({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-[11.5px] text-ink-soft">
      {label && <span>{label}</span>}
      <code className="font-mono text-[11px] text-ink">{value}</code>
    </span>
  );
}

/** Narrows the table to attacks that breached the undefended baseline. */
function BreachToggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      onClick={onToggle}
      className={`inline-flex items-center gap-2.5 rounded-lg border px-3 py-1.5 text-[13.5px] font-medium shadow-card transition-colors ${
        on ? "border-danger text-danger" : "border-line text-ink-soft hover:border-line-strong"
      }`}
    >
      <span
        className={`relative h-[19px] w-[34px] flex-none rounded-full transition-colors ${
          on ? "bg-danger" : "bg-line-strong"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 h-[15px] w-[15px] rounded-full bg-white shadow transition-transform ${
            on ? "translate-x-[15px]" : ""
          }`}
        />
      </span>
      Baseline breaches only
    </button>
  );
}
