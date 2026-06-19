import { Link } from "react-router-dom";

import { formatLatency, type SessionTotals } from "@/lib/chat";
import type { SessionStatus } from "@/lib/types";

/** Compact session instruments — all client-derived from the per-answer done
 *  events (real spend/latency, not asserted). Shown once the conversation starts. */
export function InstrumentBar({
  totals,
  session,
}: {
  totals: SessionTotals;
  session: SessionStatus | null;
}) {
  const capped = session !== null && session.limit > 0;
  return (
    <div className="mb-5 flex flex-wrap items-center gap-2">
      <Chip k="session" v={totals.costUsd === null ? "—" : `$${totals.costUsd.toFixed(4)}`} />
      <Chip k="avg" v={totals.avgLatencyMs === null ? "—" : formatLatency(totals.avgLatencyMs)} />
      {totals.servedBy && (
        <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-2.5 py-1 text-[12px] text-ink-soft">
          <span className="h-1.5 w-1.5 rounded-full bg-good" aria-hidden />
          served by <code className="font-mono text-[11.5px] text-ink">{totals.servedBy}</code>
        </span>
      )}
      {capped && (
        <Chip
          k="queries"
          v={`${session.limit - session.remaining} of ${session.limit}`}
          title="Per-session query cap — keeps the public demo inside free tiers"
        />
      )}
      <Link
        to="/evals"
        className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-good/25 bg-good-soft px-3 py-1 text-[12px] font-medium text-good transition-colors hover:bg-good/15"
      >
        Evals scorecard →
      </Link>
    </div>
  );
}

function Chip({ k, v, title }: { k: string; v: string; title?: string }) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-[12px] text-ink-soft"
    >
      {k} <span className="font-mono font-semibold text-ink">{v}</span>
    </span>
  );
}
