import type { SourcesSummary } from "@/lib/sources";

// The pill of corpus stats under the page title. Uses `chunks`-derived `passages`
// instead of the mock's "volumes" — the API is per-volume, and passage count is
// the honest, auditable corpus size for a RAG demo.
export function SummaryBar({ summary }: { summary: SourcesSummary }) {
  const fmt = new Intl.NumberFormat("en-US");
  return (
    <div className="inline-flex flex-wrap items-center rounded-full border border-line bg-surface px-1 py-1.5 text-[13.5px] shadow-card">
      <Stat value={fmt.format(summary.works)} label="works" />
      <Stat value={fmt.format(summary.authors)} label="authors" divider />
      <Stat value={fmt.format(summary.passages)} label="passages" divider />
      <Stat value="100%" label="public domain" divider accent />
    </div>
  );
}

function Stat({
  value,
  label,
  divider = false,
  accent = false,
}: {
  value: string;
  label: string;
  divider?: boolean;
  accent?: boolean;
}) {
  return (
    <span className={`px-4 text-ink-soft ${divider ? "border-l border-line" : ""}`}>
      <b className={`font-semibold tabular-nums ${accent ? "text-good" : "text-ink"}`}>
        {value}
      </b>{" "}
      {label}
    </span>
  );
}
