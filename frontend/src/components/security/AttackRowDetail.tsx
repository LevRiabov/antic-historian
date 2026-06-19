import type { ReactNode } from "react";

import { ColorBadge } from "@/components/ui/ColorBadge";
import { OutcomeChip } from "@/components/security/OutcomeChip";
import { baseLeakNote, fmtLatency, SEC_CAT_META, type AttackRow } from "@/lib/security";

/** Render an answer's [n] citation markers as static superscript pills. Unlike the
 *  golden page these aren't clickable — the audit records keep marker ids but not
 *  the retrieved-chunk mapping needed to open a passage. */
function renderAnswer(text: string): ReactNode[] {
  return text.split(/(\[\d+\])/g).map((part, i) => {
    const m = /^\[(\d+)\]$/.exec(part);
    if (!m) return <span key={i}>{part}</span>;
    return (
      <sup
        key={i}
        className="mx-px inline-flex h-3.5 min-w-3.5 items-center justify-center rounded border border-accent/30 bg-accent-soft px-1 align-super text-[9.5px] font-bold text-accent-ink"
      >
        {m[1]}
      </sup>
    );
  });
}

export function AttackRowDetail({ row }: { row: AttackRow }) {
  const meta = SEC_CAT_META[row.category];
  const leak = baseLeakNote(row);

  return (
    <div className="grid gap-6 px-6 py-5 md:grid-cols-[1fr_300px]">
      <div>
        <Label>Attack prompt</Label>
        <div className="rounded-lg border border-line border-l-[3px] border-l-ink-faint bg-surface px-3.5 py-3 font-mono text-[13px] leading-relaxed whitespace-pre-wrap text-ink">
          {row.attack}
        </div>

        <Label className="mt-4">Response — baseline vs defended</Label>
        <div className="grid gap-3.5 sm:grid-cols-2">
          {/* Baseline */}
          <div>
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-[11px] font-bold tracking-wide text-danger uppercase">
                Baseline · no defence
              </span>
              <OutcomeChip breached={row.baseSucceeded} />
            </div>
            <div
              className={`rounded-lg border border-l-[3px] px-3.5 py-3 text-[13.5px] leading-relaxed ${
                row.baseSucceeded
                  ? "border-line border-l-danger bg-danger-soft text-ink"
                  : "border-line border-l-good bg-surface text-ink"
              }`}
            >
              {renderAnswer(row.baseAnswer)}
            </div>
            {leak && (
              <div className="mt-2 inline-flex items-center gap-1.5 rounded-full border border-danger/25 bg-danger-soft px-2.5 py-1 text-[11.5px] font-semibold text-danger">
                ⚠ {leak}
              </div>
            )}
          </div>

          {/* Defended */}
          <div>
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-[11px] font-bold tracking-wide text-good uppercase">
                Defended · defence-stack
              </span>
              <OutcomeChip breached={row.defSucceeded} />
            </div>
            <div
              className={`rounded-lg border border-l-[3px] border-l-good px-3.5 py-3 text-[13.5px] leading-relaxed ${
                row.defBlocked
                  ? "border-line bg-surface text-ink-soft italic"
                  : "border-line bg-surface text-ink"
              }`}
            >
              {row.defBlocked ? row.defAnswer : renderAnswer(row.defAnswer)}
            </div>
            <div className="mt-2 inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-0.5 font-mono text-[11px] text-ink-faint">
              stopped by · {row.defBlocked ? "output filter" : "grounding & citation audit"}
            </div>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <MetaPill>
            baseline latency <code className="font-mono text-ink">{fmtLatency(row.baseLatencyMs)}</code>
          </MetaPill>
          <MetaPill>
            refused (baseline):{" "}
            <code className="font-mono text-ink">{row.baseRefused ? "yes" : "no"}</code>
          </MetaPill>
          {row.baseDangling.length > 0 && (
            <MetaPill>
              dangling markers{" "}
              <code className="font-mono text-ink">{row.baseDangling.join(", ")}</code>
            </MetaPill>
          )}
        </div>
      </div>

      <aside className="self-start rounded-xl border border-line bg-surface p-4">
        <ColorBadge label={row.category} color={meta.color} />
        <p className="mt-2 text-[13px] leading-relaxed text-ink-soft">{meta.desc}</p>
        <div className="mt-3 border-t border-line pt-3">
          <Label>Net verdict</Label>
          <OutcomeChip breached={row.defSucceeded} />
        </div>
      </aside>
    </div>
  );
}

function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`mb-1.5 text-[11px] font-bold tracking-wide text-ink-faint uppercase ${className}`}
    >
      {children}
    </div>
  );
}

function MetaPill({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full border border-line bg-surface px-2.5 py-1 text-[11.5px] text-ink-soft">
      {children}
    </span>
  );
}
