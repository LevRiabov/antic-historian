import { useRef, type KeyboardEvent } from "react";

import type { SessionStatus } from "@/lib/types";

/*
 * The bottom composer: an auto-growing textarea, a send button, and the deep-mode
 * switch. Enter sends; Shift+Enter inserts a newline. The parent owns the text +
 * mode and the in-flight state (send is disabled mid-stream).
 */
export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  deep,
  onToggleDeep,
  busy,
  session,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  deep: boolean;
  onToggleDeep: () => void;
  busy: boolean;
  session: SessionStatus | null;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  function grow() {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!busy && value.trim()) onSend();
    }
  }

  const capped = session !== null && session.limit > 0;

  return (
    <div className="sticky bottom-0 z-10 bg-gradient-to-t from-bg from-65% to-transparent pt-3 pb-4">
      <div className="rounded-2xl border border-line-strong bg-surface p-2 pl-4 shadow-[0_6px_30px_rgba(31,35,40,0.08)]">
        <div className="flex items-end gap-2.5">
          <textarea
            ref={ref}
            rows={1}
            value={value}
            disabled={busy}
            onChange={(e) => {
              onChange(e.target.value);
              grow();
            }}
            onKeyDown={onKeyDown}
            placeholder="Ask about the ancient Greco-Roman world…"
            className="max-h-40 flex-1 resize-none bg-transparent py-2 text-[15px] leading-relaxed text-ink outline-none placeholder:text-ink-faint disabled:opacity-60"
          />
          {/* Mid-stream the send button becomes a Stop button: on a slow provider a
              user must be able to bail out of a long answer, not just wait it out. */}
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop generating"
              title="Stop generating"
              className="grid h-9.5 w-9.5 flex-none place-items-center rounded-xl bg-ink p-2 text-white shadow-[0_2px_8px_rgba(31,35,40,0.25)] transition-colors hover:bg-ink-soft"
            >
              <StopIcon />
            </button>
          ) : (
            <button
              type="button"
              onClick={onSend}
              disabled={!value.trim()}
              aria-label="Send"
              className="grid h-9.5 w-9.5 flex-none place-items-center rounded-xl bg-accent p-2 text-white shadow-[0_2px_8px_rgba(138,90,43,0.35)] transition-colors hover:bg-accent-ink disabled:cursor-not-allowed disabled:opacity-50"
            >
              <SendIcon />
            </button>
          )}
        </div>

        <div className="mt-1 flex items-center gap-3 border-t border-line pt-2">
          <button
            type="button"
            role="switch"
            aria-checked={deep}
            onClick={onToggleDeep}
            className="inline-flex items-center gap-2.5 select-none"
            title="Deep mode streams the agent's reasoning + the pipeline trace"
          >
            <span
              className={`relative h-[22px] w-[38px] flex-none rounded-full transition-colors ${
                deep ? "bg-accent" : "bg-line-strong"
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 h-[18px] w-[18px] rounded-full bg-white shadow transition-transform ${
                  deep ? "translate-x-4" : ""
                }`}
              />
            </span>
            <span className={`text-[13px] ${deep ? "font-semibold text-accent-ink" : "text-ink-soft"}`}>
              Deep mode
            </span>
            <span className="hidden text-[11px] text-ink-faint sm:inline">
              streams the agent’s reasoning
            </span>
          </button>

          <div className="ml-auto flex items-center gap-3 text-[11.5px] text-ink-faint">
            {capped && (
              <span className="font-mono" title="Per-session query cap (keeps the public demo inside free tiers)">
                {session.remaining} of {session.limit} left
              </span>
            )}
            <span className="inline-flex items-center gap-1.5">
              <span className="text-good" aria-hidden>
                ●
              </span>
              EU-hosted · GDPR-native
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function SendIcon() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  );
}
