import type { ReactNode } from "react";

import { DeepPanel } from "@/components/chat/DeepPanel";
import { FoundChunks } from "@/components/chat/FoundChunks";
import { ReasoningPanel } from "@/components/chat/ReasoningPanel";
import { useElapsedSeconds } from "@/components/chat/useElapsedSeconds";
import {
  citationByMarker,
  citedRetrieved,
  formatCost,
  formatLatency,
  isRefusal,
  type Turn,
} from "@/lib/chat";
import type { Citation } from "@/lib/types";

/*
 * One assistant answer: the avatar + the answer body. While streaming it shows a
 * live indicator (and the deep-mode trace, auto-opened); when done it shows the
 * answer with clickable [n] citations, a meta row (cost / latency / sources /
 * served-by), and — for deep turns — the collapsible reasoning trace.
 */
export function AnswerMessage({
  turn,
  onOpenCitation,
}: {
  turn: Turn;
  // Opens the drawer on the turn's full passage list, positioned at `index` — so the
  // drawer can page through every retrieved passage, not just the one clicked.
  onOpenCitation: (sources: readonly Citation[], index: number) => void;
}) {
  // While streaming, show the accumulating deltas; once done, show the authoritative
  // SERVED answer (done.answer) — which may differ from the deltas when the output
  // filter withheld/redacted the response (done.blocked).
  const text = turn.done ? turn.done.answer : turn.answer;
  const refused = isRefusal(turn);
  const { cited, retrieved } = citedRetrieved(turn);
  const firstSource = turn.sources.find((c) => turn.done?.markers.used.includes(c.marker)) ?? turn.sources[0];

  return (
    <div className="flex items-start gap-3.5">
      <div className="mt-0.5 grid h-[30px] w-[30px] flex-none place-items-center rounded-lg bg-gradient-to-br from-[#9c6a39] to-[#6b4420] font-serif text-[15px] font-bold text-white">
        A
      </div>
      <div className="min-w-0 flex-1">
        {/* Deep trace sits ABOVE the answer (collapsed by default, with a live
            step-by-step status); the answer renders below it. Shown for the whole
            deep turn so the live progress is visible from the first moment. */}
        {turn.mode === "deep" && turn.status !== "error" && <DeepPanel turn={turn} />}

        {/* Live chain-of-thought (fast path / reasoning models). Self-hides when the
            model emits none, so it's safe to render unconditionally. */}
        {turn.status !== "error" && <ReasoningPanel turn={turn} />}

        {turn.status === "error" ? (
          <div className="rounded-2xl border border-refuse/30 bg-refuse-soft px-4 py-3 text-[14.5px] text-refuse">
            {turn.error}
          </div>
        ) : refused ? (
          <Refusal text={text} />
        ) : text ? (
          <div className="text-[15.5px] leading-relaxed text-ink">
            <AnswerBody text={text} sources={turn.sources} onOpenCitation={onOpenCitation} />
            {turn.status === "streaming" && <Caret />}
          </div>
        ) : turn.mode === "deep" ? (
          // The deep panel above already shows live progress — no duplicate spinner.
          null
        ) : (
          // Fast path, still composing: the status line + the retrieved passages as
          // cards, so the (often long, reasoning-model) wait is reading time, not a void.
          <>
            <Thinking sources={turn.sources.length} />
            <FoundChunks
              sources={turn.sources}
              onOpen={(index) => onOpenCitation(turn.sources, index)}
            />
          </>
        )}

        {turn.done && turn.status !== "error" && (
          <div className="mt-3.5 flex flex-wrap items-center gap-2 text-[12px]">
            {turn.done.blocked && (
              <Pill tone="good">blocked by security filter</Pill>
            )}
            <Pill>
              this answer cost <b className="font-semibold text-ink">{formatCost(turn.done.cost)}</b>
            </Pill>
            {turn.elapsedMs != null && <Pill>{formatLatency(turn.elapsedMs)}</Pill>}
            {retrieved > 0 && (
              <button
                type="button"
                onClick={() =>
                  firstSource && onOpenCitation(turn.sources, turn.sources.indexOf(firstSource))
                }
                className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-ink-soft transition-colors hover:border-accent hover:text-accent-ink"
              >
                <SourcesIcon />
                {cited} cited · {retrieved} retrieved
              </button>
            )}
            {turn.done.served_by && (
              <Pill>
                <span className="h-1.5 w-1.5 rounded-full bg-good" aria-hidden />
                served by <code className="font-mono text-[11.5px] text-ink">{turn.done.served_by}</code>
              </Pill>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** Render the answer, turning [n] markers into clickable citation pills. A marker
 *  with no matching source (dangling) stays a static pill. */
function AnswerBody({
  text,
  sources,
  onOpenCitation,
}: {
  text: string;
  sources: readonly Citation[];
  onOpenCitation: (sources: readonly Citation[], index: number) => void;
}) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <>
      {parts.map((part, i) => {
        const m = /^\[(\d+)\]$/.exec(part);
        if (!m) return <span key={i}>{part}</span>;
        const marker = Number(m[1]);
        const citation = citationByMarker(sources, marker);
        const base =
          "mx-px inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-[5px] border border-accent/30 bg-accent-soft px-1 align-super text-[10.5px] font-bold text-accent-ink";
        if (!citation) {
          return (
            <sup key={i} className={base} title="no source for this marker">
              {marker}
            </sup>
          );
        }
        return (
          <sup key={i}>
            <button
              type="button"
              onClick={() => onOpenCitation(sources, sources.indexOf(citation))}
              className={`${base} cursor-pointer transition-colors hover:bg-[#e9d9c6]`}
              title={`${citation.author}, ${citation.work_title}`}
            >
              {marker}
            </button>
          </sup>
        );
      })}
    </>
  );
}

function Refusal({ text }: { text: string }) {
  return (
    <div className="rounded-2xl border border-refuse/25 bg-refuse-soft px-4 py-3.5">
      <div className="mb-1.5 flex items-center gap-2 text-[14px] font-semibold text-refuse">
        <WarnIcon /> No source in the corpus
      </div>
      <p className="text-[14.5px] leading-relaxed text-[#5a4636]">{text}</p>
      <span className="mt-2.5 inline-flex items-center gap-1.5 rounded-full border border-good/25 bg-good-soft px-2.5 py-1 text-[12px] text-good">
        <CheckIcon /> The system working as intended — grounded or silent.
      </span>
    </div>
  );
}

/* The fast-path pre-answer indicator. Two things keep a slow time-to-first-token from
 * looking frozen: the status advances the moment retrieval lands (sources arrived →
 * "Read N passages · composing the answer…"), and a live seconds counter proves the
 * stream is alive. The counter only appears past 3s so a quick answer doesn't flash it. */
function Thinking({ sources }: { sources: number }) {
  const seconds = useElapsedSeconds(true);
  const label =
    sources > 0
      ? `Read ${sources} passage${sources === 1 ? "" : "s"} · composing the answer…`
      : "Searching the corpus…";
  return (
    <div className="flex items-center gap-2 text-[14px] text-ink-faint">
      <span className="flex gap-1" aria-hidden>
        <Dot /> <Dot /> <Dot />
      </span>
      {label}
      {seconds >= 3 && <span className="font-mono text-[12px] text-ink-faint">{seconds}s</span>}
    </div>
  );
}

function Caret() {
  return <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-accent" aria-hidden />;
}

function Dot() {
  return <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-faint" />;
}

function Pill({ children, tone }: { children: ReactNode; tone?: "good" }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 ${
        tone === "good" ? "border-good/25 bg-good-soft text-good" : "border-line bg-surface text-ink-soft"
      }`}
    >
      {children}
    </span>
  );
}

function SourcesIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  );
}

function WarnIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 8v5M12 16h.01" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}
