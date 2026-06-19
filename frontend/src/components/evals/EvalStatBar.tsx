import type { StatTile } from "@/lib/evals";

const TONE_TEXT = {
  good: "text-good",
  amber: "text-amber",
  refuse: "text-refuse",
  danger: "text-danger",
} as const;

/** The row of aggregate tiles above the table — corpus-wide retrieval + judge
 *  numbers, straight from the two runs' `aggregates` (real, not illustrative). */
export function EvalStatBar({ tiles }: { tiles: readonly StatTile[] }) {
  return (
    <section className="mt-5 flex flex-wrap gap-3">
      {tiles.map((t) => (
        <div
          key={t.label}
          className="flex-1 basis-[150px] rounded-xl border border-line bg-surface p-3.5 shadow-card"
        >
          <div className="flex items-center gap-1 text-[11.5px] font-semibold tracking-wide text-ink-faint uppercase">
            {t.label}
            {t.hint && (
              <span
                tabIndex={0}
                role="img"
                aria-label={t.hint}
                title={t.hint}
                className="grid h-3.5 w-3.5 cursor-help place-items-center rounded-full border border-line-strong text-[9px] normal-case"
              >
                i
              </span>
            )}
          </div>
          <div className={`mt-1 font-mono text-2xl font-semibold ${t.tone ? TONE_TEXT[t.tone] : "text-ink"}`}>
            {t.value}
          </div>
          <div className="mt-0.5 text-[11.5px] text-ink-faint">{t.sub}</div>
        </div>
      ))}
    </section>
  );
}
