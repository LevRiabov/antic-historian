import type { SecStatTile } from "@/lib/security";

const TONE_TEXT = {
  good: "text-good",
  amber: "text-amber",
  refuse: "text-refuse",
  danger: "text-danger",
} as const;

/** The aggregate tiles above the table — real ASR numbers from the two runs. The
 *  defended-ASR tile is the headline (hero), tinted green when the defence holds. */
export function SecurityStatBar({ tiles }: { tiles: readonly SecStatTile[] }) {
  return (
    <section className="mt-5 flex flex-wrap gap-3">
      {tiles.map((t) => (
        <div
          key={t.label}
          className={`flex-1 basis-[150px] rounded-xl border p-3.5 shadow-card ${
            t.hero
              ? "border-good/30 bg-gradient-to-b from-surface to-good-soft/50"
              : "border-line bg-surface"
          }`}
        >
          <div className="text-[11.5px] font-semibold tracking-wide text-ink-faint uppercase">
            {t.label}
          </div>
          <div
            className={`mt-1 font-mono font-semibold ${t.mono ? "text-[15px]" : "text-2xl"} ${
              t.tone ? TONE_TEXT[t.tone] : "text-ink"
            } ${t.mono ? "break-all" : ""}`}
          >
            {t.value}
          </div>
          <div className="mt-0.5 text-[11.5px] text-ink-faint">{t.sub}</div>
        </div>
      ))}
    </section>
  );
}
