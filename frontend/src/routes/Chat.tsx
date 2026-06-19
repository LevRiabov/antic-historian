import { useEffect, useRef, useState } from "react";

import { AnswerMessage } from "@/components/chat/AnswerMessage";
import { Composer } from "@/components/chat/Composer";
import { InstrumentBar } from "@/components/chat/InstrumentBar";
import { Landing } from "@/components/chat/Landing";
import { SourceDrawer } from "@/components/SourceDrawer";
import { applyAskEvent, newTurn, newTurnId, sessionTotals, type Turn } from "@/lib/chat";
import { AskError, askStream } from "@/lib/sse";
import type { AskMode, Citation, SessionStatus } from "@/lib/types";

export function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [deep, setDeep] = useState(false);
  const [busy, setBusy] = useState(false);
  const [session, setSession] = useState<SessionStatus | null>(null);
  // The citation drawer pages through one turn's passages: the full list + the
  // currently-shown index. Opened from an inline [n], the cited chip, or a found-card.
  const [active, setActive] = useState<{ sources: readonly Citation[]; index: number } | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Cancel an in-flight stream if the user navigates away mid-answer.
  useEffect(() => () => abortRef.current?.abort(), []);

  // Keep the latest turn in view as the conversation grows.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  function patchTurn(id: string, fn: (t: Turn) => Turn) {
    setTurns((prev) => prev.map((t) => (t.id === id ? fn(t) : t)));
  }

  async function send(raw: string) {
    const question = raw.trim();
    if (!question || busy) return;

    const id = newTurnId();
    const mode: AskMode = deep ? "deep" : "fast";
    setTurns((prev) => [...prev, newTurn(id, question, mode)]);
    setInput("");
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const started = performance.now();

    try {
      for await (const ev of askStream({ question, mode, signal: controller.signal })) {
        // The `meta` budget also drives the page-level composer/instrument bar, so
        // mirror it into top-level state; the rest is folded into the turn by the
        // (unit-tested) reducer — including the terminal `error` frame.
        if (ev.event === "meta") setSession(ev.data);
        patchTurn(id, (t) => applyAskEvent(t, ev, Math.round(performance.now() - started)));
      }
      // Stream closed without a done event (e.g. aborted): settle the turn.
      patchTurn(id, (t) =>
        t.status === "streaming"
          ? { ...t, status: "done", elapsedMs: Math.round(performance.now() - started) }
          : t,
      );
    } catch (err) {
      if (controller.signal.aborted) {
        patchTurn(id, (t) => ({ ...t, status: "done" }));
      } else {
        patchTurn(id, (t) => ({ ...t, status: "error", error: errorMessage(err) }));
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  // Cancel the in-flight stream but keep the conversation. The aborted path in send()
  // settles the current turn (status "done", partial answer retained).
  function stop() {
    abortRef.current?.abort();
  }

  function reset() {
    abortRef.current?.abort();
    setTurns([]);
    setInput("");
  }

  const totals = sessionTotals(turns);

  return (
    <div className="mx-auto flex min-h-[calc(100vh-9rem)] max-w-3xl flex-col">
      {turns.length === 0 ? (
        <div className="flex-1">
          <Landing onPick={send} />
        </div>
      ) : (
        <div className="flex-1">
          <div className="mb-4 flex items-center justify-between">
            <InstrumentBar totals={totals} session={session} />
          </div>
          <div className="flex flex-col gap-7">
            {turns.map((turn) => (
              <div key={turn.id}>
                <div className="flex justify-end">
                  <div className="max-w-[80%] rounded-[16px_16px_4px_16px] border border-line bg-user-bubble px-4 py-3 text-[15px] leading-relaxed text-ink">
                    {turn.question}
                  </div>
                </div>
                <div className="mt-6">
                  <AnswerMessage
                    turn={turn}
                    onOpenCitation={(sources, index) => setActive({ sources, index })}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-6 text-center">
            <button
              type="button"
              onClick={reset}
              disabled={busy}
              className="text-[12.5px] text-ink-faint underline-offset-2 hover:text-accent-ink hover:underline disabled:opacity-50"
            >
              New conversation
            </button>
          </div>
          <div ref={endRef} />
        </div>
      )}

      <Composer
        value={input}
        onChange={setInput}
        onSend={() => send(input)}
        onStop={stop}
        deep={deep}
        onToggleDeep={() => setDeep((v) => !v)}
        busy={busy}
        session={session}
      />

      <SourceDrawer
        open={active !== null}
        onClose={() => setActive(null)}
        marker={active ? (active.sources[active.index]?.marker ?? undefined) : undefined}
        passage={active ? (active.sources[active.index] ?? null) : null}
        loading={false}
        error={false}
        index={active?.index}
        count={active?.sources.length}
        onPrev={() =>
          setActive((a) => (a && a.index > 0 ? { ...a, index: a.index - 1 } : a))
        }
        onNext={() =>
          setActive((a) =>
            a && a.index < a.sources.length - 1 ? { ...a, index: a.index + 1 } : a,
          )
        }
      />
    </div>
  );
}

/** Map a failed /ask into a user-facing line — special-casing the cap + deep-mode 503. */
function errorMessage(err: unknown): string {
  if (err instanceof AskError) {
    if (err.status === 429) {
      const detail = (err.body as { detail?: { error?: string } } | undefined)?.detail;
      if (detail?.error === "session_cap_reached") {
        return "You’ve reached this demo’s per-session query limit. Thanks for trying it!";
      }
      return "Rate limit reached — please wait a moment, then try again.";
    }
    if (err.status === 503) {
      return "Deep mode is temporarily unavailable. Try again, or switch Deep mode off.";
    }
    return `The request failed (${err.status}). Please try again.`;
  }
  return "Something went wrong reaching the server. Please try again.";
}
