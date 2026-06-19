import type { MetricTone } from "@/lib/evals";

const TONE_CLASS: Record<MetricTone, string> = {
  good: "bg-good-soft text-good border-good/25",
  amber: "bg-amber-soft text-amber border-amber/30",
  refuse: "bg-refuse-soft text-refuse border-refuse/30",
  danger: "bg-danger-soft text-danger border-danger/25",
};

/** A monospace metric pill (recall %, judge score, MRR). A null value renders as
 *  a muted em-dash — the honest "not applicable / not measured" cell. */
export function MetricChip({ value, tone }: { value: string | null; tone: MetricTone | null }) {
  if (value === null || tone === null) {
    return <span className="font-mono text-[12.5px] text-ink-faint">—</span>;
  }
  return (
    <span
      className={`inline-flex min-w-[46px] items-center justify-center rounded-md border px-2 py-0.5 font-mono text-[12.5px] font-semibold ${TONE_CLASS[tone]}`}
    >
      {value}
    </span>
  );
}
