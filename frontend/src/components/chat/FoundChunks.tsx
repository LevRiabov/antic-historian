import { useState } from "react";

import type { Citation } from "@/lib/types";

/*
 * The retrieved passages, shown WHILE the model is still composing (fast path). The
 * `sources` event lands seconds in, but the answer can take far longer — on a
 * reasoning model the model "thinks" with nothing visible. Rather than a bare
 * spinner, surface the real grounding the moment it arrives: the user can read the
 * source material during the wait instead of staring at dots. Each card expands
 * inline; "Open in reader" hands off to the full drawer (with its pager).
 */
export function FoundChunks({
  sources,
  onOpen,
}: {
  sources: readonly Citation[];
  onOpen: (index: number) => void;
}) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-2.5 flex flex-col gap-1.5">
      {sources.map((c, i) => (
        <FoundCard key={c.chunk_id} citation={c} onOpen={() => onOpen(i)} />
      ))}
    </div>
  );
}

function FoundCard({ citation, onOpen }: { citation: Citation; onOpen: () => void }) {
  const [open, setOpen] = useState(false);
  const locator = citation.locator.join(" · ");
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-[#fcfbf9]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-baseline gap-2 px-3 py-2 text-left text-[12.5px]"
      >
        <span
          className={`mt-0.5 flex-none text-ink-faint transition-transform ${open ? "rotate-90" : ""}`}
          aria-hidden
        >
          ▸
        </span>
        <span className="min-w-0 flex-1">
          <span className="font-semibold text-ink">{citation.work_title}</span>
          <span className="text-ink-faint"> — {citation.author}</span>
          {locator && <span className="text-ink-faint"> · {locator}</span>}
        </span>
      </button>
      {open && (
        <div className="border-t border-line px-3 py-2.5">
          <blockquote className="max-h-44 overflow-y-auto border-l-2 border-accent pl-3 font-serif text-[13.5px] leading-relaxed text-[#2b2620]">
            {citation.text}
          </blockquote>
          <button
            type="button"
            onClick={onOpen}
            className="mt-2 text-[11.5px] font-medium text-accent-ink underline-offset-2 hover:underline"
          >
            Open in reader ↗
          </button>
        </div>
      )}
    </div>
  );
}
