import { SUGGESTIONS } from "@/lib/chat";

/** The empty state: a short pitch + starter prompts (one shows an honest refusal).
 *  Picking a chip sends it immediately. */
export function Landing({ onPick }: { onPick: (text: string) => void }) {
  return (
    <section className="px-2 py-8 text-center">
      <h1 className="font-serif text-3xl font-bold text-ink">Ask the ancient world.</h1>
      <p className="mx-auto mt-2.5 mb-6 max-w-xl text-[15.5px] text-ink-soft">
        Every answer is drawn strictly from public-domain primary sources, with inline citations you
        can open and read. No invented history.
      </p>
      <div className="flex flex-wrap justify-center gap-2.5">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.text}
            type="button"
            onClick={() => onPick(s.text)}
            className={`max-w-[330px] rounded-xl border bg-surface px-3.5 py-2.5 text-left text-[13.5px] text-ink shadow-card transition-transform hover:-translate-y-0.5 hover:border-accent ${
              s.refusal ? "border-dashed border-line-strong" : "border-line-strong"
            }`}
          >
            <span
              className={`mb-0.5 block text-[11px] tracking-wide uppercase ${
                s.refusal ? "text-refuse" : "text-ink-faint"
              }`}
            >
              {s.kind}
            </span>
            {s.text}
          </button>
        ))}
      </div>
    </section>
  );
}
