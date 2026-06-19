import { useEffect, useRef, useState } from "react";

import type { Turn } from "@/lib/chat";

/*
 * The model's live chain-of-thought. A reasoning model (deepseek-v4-pro) "thinks" for
 * many seconds before the first answer token — without this the user stares at a void.
 * Streaming the `reasoning` events here turns that wait into visible progress (the o1
 * pattern). Open while the model is working, auto-collapses once the answer lands (a
 * manual toggle sticks). Renders nothing when the model emits no reasoning, so a
 * non-reasoning model is unaffected. Display-only: this is never the served answer.
 */
export function ReasoningPanel({ turn }: { turn: Turn }) {
  const streaming = turn.status === "streaming";
  // Follows `streaming` by default (open while thinking, closed after); a click pins it.
  const [override, setOverride] = useState<boolean | null>(null);
  const open = override ?? streaming;
  const bodyRef = useRef<HTMLDivElement>(null);

  // Keep the newest thought in view while it streams.
  useEffect(() => {
    if (open && streaming && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [turn.reasoning, open, streaming]);

  if (!turn.reasoning) return null;

  return (
    <div className="mb-3.5 overflow-hidden rounded-xl border border-line bg-[#fcfbf9]">
      <button
        type="button"
        onClick={() => setOverride(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left text-[13px] text-ink-soft"
      >
        <span className="rounded border border-accent/30 bg-accent-soft px-1.5 py-0.5 text-[10.5px] font-bold tracking-wide text-accent-ink uppercase">
          Reasoning
        </span>
        {streaming ? (
          <span className="inline-flex items-center gap-1.5">
            <span className="flex gap-1" aria-hidden>
              <Dot /> <Dot /> <Dot />
            </span>
            Thinking…
          </span>
        ) : (
          <span>Thought process</span>
        )}
        <span
          className={`ml-auto text-ink-faint transition-transform ${open ? "rotate-90" : ""}`}
          aria-hidden
        >
          ▸
        </span>
      </button>

      {open && (
        <div
          ref={bodyRef}
          className="max-h-48 overflow-y-auto border-t border-line px-4 py-3 text-[13px] leading-relaxed whitespace-pre-wrap text-ink-soft"
        >
          {turn.reasoning}
        </div>
      )}
    </div>
  );
}

function Dot() {
  return <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-faint" />;
}
