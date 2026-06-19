import type { ReactNode } from "react";

import { CategoryBadge } from "@/components/evals/CategoryBadge";
import { MetricChip } from "@/components/evals/MetricChip";
import {
  CAT_META,
  fmtMrr,
  fmtScore,
  mrrTone,
  parseJudgeNotes,
  recallTone,
  scoreTone,
  type GoldenRow,
  type MetricTone,
} from "@/lib/evals";

/** Split an answer on its [n] citation markers, rendering each as a clickable
 *  superscript pill. Marker n resolves to retrievedChunkIds[n-1]; a marker that
 *  points at no chunk (dangling) renders as a static pill. No HTML injection — we
 *  map text/markers to React nodes. */
function renderAnswer(
  text: string,
  retrievedChunkIds: number[],
  onOpenCitation: (chunkId: number, marker: number) => void,
): ReactNode[] {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const m = /^\[(\d+)\]$/.exec(part);
    if (!m) return <span key={i}>{part}</span>;
    const marker = Number(m[1]);
    const chunkId = retrievedChunkIds[marker - 1];
    const base =
      "mx-px inline-flex h-4 min-w-4 items-center justify-center rounded border border-accent/30 bg-accent-soft px-1 align-super text-[10px] font-bold text-accent-ink";
    if (chunkId === undefined) {
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
          onClick={() => onOpenCitation(chunkId, marker)}
          className={`${base} cursor-pointer transition-colors hover:bg-[#e9d9c6]`}
          title="View the cited passage"
        >
          {marker}
        </button>
      </sup>
    );
  });
}

export function GoldenRowDetail({
  row,
  works,
  onOpenCitation,
}: {
  row: GoldenRow;
  works: Map<number, string>;
  onOpenCitation: (chunkId: number, marker?: number) => void;
}) {
  const meta = CAT_META[row.category];
  const sidecard = (
    <aside className="self-start rounded-xl border border-line bg-surface p-4">
      <CategoryBadge category={row.category} />
      <p className="mt-2 text-[13px] leading-relaxed text-ink-soft">{meta.desc}</p>
    </aside>
  );

  if (row.category === "out-of-scope") {
    const pass = row.refusalCorrect;
    return (
      <div className="grid gap-6 px-6 py-5 md:grid-cols-[1fr_320px]">
        <div>
          <Label>Model answer</Label>
          <div className="mt-1.5 rounded-lg border border-line border-l-[3px] border-l-refuse bg-surface px-3.5 py-3 text-[14.5px] leading-relaxed text-ink-soft">
            {renderAnswer(row.answer, row.retrievedChunkIds, onOpenCitation)}
          </div>
          <Label className="mt-4">Expected behaviour</Label>
          <p className="font-serif text-[15.5px] leading-relaxed text-ink-soft">
            No source in corpus — the correct behaviour is to refuse.
          </p>
          <Label className="mt-3.5">Verdict</Label>
          <span
            className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1 text-[12.5px] font-semibold ${
              pass
                ? "border-good/25 bg-good-soft text-good"
                : "border-refuse/30 bg-refuse-soft text-refuse"
            }`}
          >
            {pass ? "✓ Refused correctly" : "✗ Failed to refuse"}
          </span>

          {row.judgeNotes.trim() && (
            <>
              <Label className="mt-4">Judge notes</Label>
              <p className="rounded-lg border border-line border-l-[3px] border-l-accent-ink/40 bg-surface px-3.5 py-3 text-[13.5px] leading-relaxed whitespace-pre-line text-ink-soft">
                {row.judgeNotes}
              </p>
            </>
          )}
        </div>
        {sidecard}
      </div>
    );
  }

  const workNames = row.goldPgIds.map((id) => works.get(id) ?? `pg ${id}`);
  const cited = new Set(row.citedChunkIds);
  const goldSet = new Set(row.goldChunkIds);
  const judgeNotes = parseJudgeNotes(row.judgeNotes);

  return (
    <div className="grid gap-6 px-6 py-5 md:grid-cols-[1fr_320px]">
      <div>
        <Label>Model answer</Label>
        <div className="mt-1.5 rounded-lg border border-line border-l-[3px] border-l-accent bg-surface px-3.5 py-3 text-[14.5px] leading-relaxed text-ink">
          {renderAnswer(row.answer, row.retrievedChunkIds, onOpenCitation)}
        </div>

        {row.ideal && (
          <>
            <Label className="mt-4">Ideal answer</Label>
            <p className="font-serif text-[15.5px] leading-relaxed text-ink">{row.ideal}</p>
          </>
        )}

        <Label className="mt-4">Gold sources</Label>
        {workNames.length > 0 ? (
          <div className="mt-1.5 flex flex-wrap gap-2">
            {workNames.map((name) => (
              <span
                key={name}
                className="inline-flex items-center gap-1.5 rounded-md border border-line-strong bg-accent-soft px-2.5 py-1 font-mono text-[12.5px] text-accent-ink"
              >
                <span className="font-bold text-accent" aria-hidden>
                  §
                </span>
                {name}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-[13px] text-ink-faint">No gold spans recorded.</p>
        )}

        {/* Chunk ids are the auditable identity (the locator is not surfaced):
            ✓ marks a gold chunk the answer actually cited. Click to read it. */}
        {row.goldChunkIds.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {row.goldChunkIds.map((id) => {
              const hit = cited.has(id);
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => onOpenCitation(id)}
                  title={hit ? "cited by the answer — view passage" : "view passage"}
                  className={`inline-flex cursor-pointer items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[11.5px] transition-colors hover:border-accent hover:bg-accent-soft ${
                    hit ? "border-good/25 bg-good-soft text-good" : "border-line bg-bg text-ink-faint"
                  }`}
                >
                  chunk {id}
                  {hit && <span aria-hidden>✓</span>}
                </button>
              );
            })}
          </div>
        )}

        {/* The chunks the answer actually cited — the model sometimes grounds in
            passages outside the gold set. ✓ = the cited chunk is in the gold set;
            "off-gold" = a different source the model chose. Click to read it. */}
        {row.citedChunkIds.length > 0 && (
          <>
            <Label className="mt-4">Cited by the answer</Label>
            <div className="mt-1.5 flex flex-wrap gap-1.5">
              {row.citedChunkIds.map((id) => {
                const inGold = goldSet.has(id);
                return (
                  <button
                    key={id}
                    type="button"
                    onClick={() => onOpenCitation(id)}
                    title={
                      inGold ? "in the gold set — view passage" : "not in the gold set — view passage"
                    }
                    className={`inline-flex cursor-pointer items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[11.5px] transition-colors hover:border-accent hover:bg-accent-soft ${
                      inGold
                        ? "border-good/25 bg-good-soft text-good"
                        : "border-amber/30 bg-amber-soft text-amber"
                    }`}
                  >
                    chunk {id}
                    <span aria-hidden>{inGold ? "✓" : "✕"}</span>
                    <span className="font-sans text-[10px] tracking-wide uppercase opacity-80">
                      {inGold ? "in gold" : "off-gold"}
                    </span>
                  </button>
                );
              })}
            </div>
          </>
        )}

        <Label className="mt-4">Results</Label>
        <div className="mt-1">
          <MetricRow
            value={pct(row.recall5)}
            tone={tone(row.recall5, recallTone)}
            name="Recall@5"
            gloss="share of gold requirements retrieved in the top 5"
          />
          <MetricRow
            value={pct(row.recall20)}
            tone={tone(row.recall20, recallTone)}
            name="Recall@20"
            gloss="share retrieved in the top 20"
          />
          <MetricRow
            value={row.mrr === null ? null : fmtMrr(row.mrr)}
            tone={tone(row.mrr, mrrTone)}
            name="MRR"
            gloss="mean reciprocal rank of the first gold passage"
          />
          <MetricRow
            value={score(row.faithfulness)}
            tone={tone(row.faithfulness, scoreTone)}
            name="Faithfulness"
            gloss="every claim traceable to a cited source"
          />
          <MetricRow
            value={score(row.completeness)}
            tone={tone(row.completeness, scoreTone)}
            name="Completeness"
            gloss="all required facts present in the answer"
          />
          <MetricRow
            value={score(row.attribution)}
            tone={tone(row.attribution, scoreTone)}
            name="Attribution"
            gloss={
              row.attribution === null
                ? "not applicable (single-source factual question)"
                : "each competing version credited to the right source"
            }
          />
        </div>

        {judgeNotes.length > 0 && (
          <>
            <Label className="mt-4">Judge notes</Label>
            <div className="mt-1.5 space-y-2">
              {judgeNotes.map((note, i) => (
                <div
                  key={note.label || i}
                  className="rounded-lg border border-line border-l-[3px] border-l-accent-ink/40 bg-surface px-3.5 py-2.5"
                >
                  {note.label && (
                    <div className="mb-0.5 text-[11px] font-bold tracking-wide text-accent-ink uppercase">
                      {note.label}
                    </div>
                  )}
                  <p className="text-[13.5px] leading-relaxed whitespace-pre-line text-ink-soft">
                    {note.text}
                  </p>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
      {sidecard}
    </div>
  );
}

function pct(value: number | null): string | null {
  return value === null ? null : `${Math.round(value)}%`;
}
function score(value: number | null): string | null {
  return value === null ? null : fmtScore(value);
}
function tone(value: number | null, fn: (n: number) => MetricTone): MetricTone | null {
  return value === null ? null : fn(value);
}

function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`mb-1.5 text-[11px] font-bold tracking-wide text-ink-faint uppercase ${className}`}
    >
      {children}
    </div>
  );
}

function MetricRow({
  value,
  tone,
  name,
  gloss,
}: {
  value: string | null;
  tone: MetricTone | null;
  name: string;
  gloss: string;
}) {
  return (
    <div className="flex items-baseline gap-2.5 border-t border-line py-2 first:border-0">
      <span className="flex-none">
        <MetricChip value={value} tone={tone} />
      </span>
      <span>
        <span className="text-[13.5px] font-semibold text-ink">{name}</span>{" "}
        <span className="text-[12.5px] text-ink-soft">— {gloss}</span>
      </span>
    </div>
  );
}
