import { useState } from "react";

import { useElapsedSeconds } from "@/components/chat/useElapsedSeconds";
import { formatCost, formatLatency, type Turn } from "@/lib/chat";
import type { StepEvent } from "@/lib/types";

/*
 * Deep-mode panel: the agent's ReAct reasoning (thought → action → observation)
 * plus the real token/cost/latency callouts from the done event. Collapsed by
 * default — but collapsed it shows a LIVE, growing one-line-per-step list so you
 * can watch the agent work through several searches (each step takes ~15s; showing
 * only the latest made it look frozen). NB: no pipeline-stage waterfall — the SSE
 * stream carries no per-stage timings, so we show only numbers we actually have.
 */
export function DeepPanel({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState(false);
  const usage = turn.done?.usage ?? null;
  const working = turn.status === "streaming";

  return (
    <div className="mb-3.5 overflow-hidden rounded-xl border border-line bg-[#fcfbf9]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left text-[13px] text-ink-soft"
      >
        <span className="rounded border border-accent/30 bg-accent-soft px-1.5 py-0.5 text-[10.5px] font-bold tracking-wide text-accent-ink uppercase">
          Deep mode
        </span>
        <span>
          Agent reasoning · {turn.steps.length} step{turn.steps.length === 1 ? "" : "s"}
          {turn.sources.length > 0 && ` · ${turn.sources.length} passages`}
        </span>
        <span
          className={`ml-auto text-ink-faint transition-transform ${open ? "rotate-90" : ""}`}
          aria-hidden
        >
          ▸
        </span>
      </button>

      {/* Collapsed: a live, compact step-by-step progress list. */}
      {!open && (
        <div className="border-t border-line px-3.5 py-2.5">
          {turn.steps.length > 0 ? (
            <ol className="flex flex-col gap-1">
              {turn.steps.map((step) => (
                <CompactStep key={step.index} step={step} />
              ))}
              {working && <WorkingLine />}
            </ol>
          ) : (
            <WorkingLine label="Searching the corpus…" />
          )}
        </div>
      )}

      {/* Expanded: the full reasoning timeline + the real callouts. */}
      {open && (
        <div className="border-t border-line px-4 pt-2 pb-4">
          {turn.steps.length > 0 ? (
            <ol className="relative ml-1.5 border-l-2 border-line-strong pl-5">
              {turn.steps.map((step) => (
                <li key={step.index} className="relative">
                  <span
                    className="absolute top-3 -left-[26px] h-2.5 w-2.5 rounded-full border-2 border-accent bg-surface"
                    aria-hidden
                  />
                  <FullStep step={step} />
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-[13px] text-ink-faint">No reasoning steps were recorded yet.</p>
          )}

          {turn.done && (
            <div className="mt-4 flex flex-wrap gap-2">
              <Callout k="latency" v={turn.elapsedMs != null ? formatLatency(turn.elapsedMs) : "—"} />
              <Callout k="prompt tok" v={usage ? usage.prompt_tokens.toLocaleString() : "—"} />
              <Callout k="completion" v={usage ? usage.completion_tokens.toLocaleString() : "—"} />
              <Callout k="cost" v={formatCost(turn.done.cost)} accent />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const clamp = (s: string, n: number) => {
  const flat = s.replace(/\s+/g, " ").trim();
  return flat.length > n ? `${flat.slice(0, n)}…` : flat;
};

function argSummary(step: StepEvent): string {
  const a = step.args;
  if (typeof a.query === "string") return `"${clamp(a.query, 44)}"`;
  if (a.pg_id != null) return `pg ${String(a.pg_id)}`;
  const entries = Object.entries(a);
  return entries.length ? clamp(entries.map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(", "), 44) : "";
}

function resultSummary(step: StepEvent): string {
  if (step.chunk_ids && step.chunk_ids.length > 0) return `${step.chunk_ids.length} passages`;
  return clamp(step.observation, 52);
}

/** One collapsed step: a numbered badge, the action, and a short result. The full
 *  thought is the hover title (kept off-screen to stay one line). */
function CompactStep({ step }: { step: StepEvent }) {
  return (
    <li className="flex items-baseline gap-2 text-[12.5px]" title={step.thought}>
      <span className="grid h-4 w-4 flex-none place-items-center rounded-full bg-accent-soft font-mono text-[10px] font-bold text-accent-ink">
        {step.index}
      </span>
      <code className="truncate font-mono text-[12px] text-accent-ink">
        {step.tool}({argSummary(step)})
      </code>
      <span className="flex-none text-ink-faint">→ {resultSummary(step)}</span>
    </li>
  );
}

/* A live "still working" line for the deep panel. The seconds counter (past 3s) is
 * what stops a long step — each search runs ~15s — from looking hung. */
function WorkingLine({ label = "working…" }: { label?: string }) {
  const seconds = useElapsedSeconds(true);
  return (
    <li className="flex items-center gap-2 text-[12.5px] text-ink-faint">
      <span className="flex gap-1" aria-hidden>
        <Dot /> <Dot /> <Dot />
      </span>
      {label}
      {seconds >= 3 && <span className="font-mono text-[11.5px]">{seconds}s</span>}
    </li>
  );
}

function Dot() {
  return <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-ink-faint" />;
}

/** A step's full thought / action / observation (expanded view). */
function FullStep({ step }: { step: StepEvent }) {
  const argText = Object.entries(step.args)
    .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
    .join(", ");
  return (
    <div className="py-1.5 text-[13px] leading-relaxed">
      {step.thought && (
        <p>
          <Tag>Thought</Tag>
          {step.thought}
        </p>
      )}
      <p className="mt-0.5">
        <Tag>Action</Tag>
        <code className="rounded bg-[#f4f1ec] px-1.5 py-0.5 font-mono text-[12px] text-accent-ink">
          {step.tool}({argText})
        </code>
      </p>
      <p className="mt-0.5">
        <Tag tone="good">Observation</Tag>
        {clamp(step.observation, 280)}
        {step.chunk_ids && step.chunk_ids.length > 0 && (
          <span className="text-ink-faint"> · {step.chunk_ids.length} passages</span>
        )}
      </p>
    </div>
  );
}

function Tag({ children, tone }: { children: React.ReactNode; tone?: "good" }) {
  return (
    <span
      className={`mr-1.5 text-[11px] font-bold tracking-wide uppercase ${
        tone === "good" ? "text-good" : "text-accent-ink"
      }`}
    >
      {children}
    </span>
  );
}

function Callout({ k, v, accent }: { k: string; v: string; accent?: boolean }) {
  return (
    <div className="min-w-[84px] rounded-lg border border-line bg-surface px-3 py-1.5">
      <div className="text-[10px] tracking-wide text-ink-faint uppercase">{k}</div>
      <div className={`mt-0.5 font-mono text-[13.5px] font-semibold ${accent ? "text-accent-ink" : "text-ink"}`}>
        {v}
      </div>
    </div>
  );
}
