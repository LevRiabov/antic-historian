import type { ReactNode } from "react";

import { Pill } from "@/components/ui/Pill";
import type { SourceOut } from "@/lib/types";

const PASSAGES_FMT = new Intl.NumberFormat("en-US");

// Stacked-card layout for the corpus — the small-screen counterpart to
// SourcesTable (six columns can't fit a phone without cutting off). Same data,
// shown vertically; the table takes over at lg.
export function SourceCards({ rows }: { rows: readonly SourceOut[] }) {
  if (rows.length === 0) {
    return (
      <div className="rounded-2xl border border-line bg-surface px-5 py-12 text-center text-sm text-ink-faint shadow-card lg:hidden">
        No sources match your filter.
      </div>
    );
  }

  return (
    <ul className="flex flex-col gap-3 lg:hidden">
      {rows.map((s) => (
        <li
          key={s.pg_id}
          className="rounded-2xl border border-line bg-surface p-4 shadow-card"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="font-serif text-[15.5px] font-semibold text-ink">{s.author}</div>
              <div className="mt-0.5 text-sm text-ink">{s.title}</div>
            </div>
            <Pill variant={s.category === "primary" ? "accent" : "good"}>
              {s.category === "primary" ? "Primary" : "Scholarship"}
            </Pill>
          </div>

          <dl className="mt-3 space-y-1.5 text-[13px]">
            <Row label="Translator">
              <span className="text-ink-soft">{s.translator}</span>
            </Row>
            <Row label="Public domain">
              <span className="text-ink-soft">
                <span className="mr-1 text-good" aria-hidden>
                  ✓
                </span>
                {s.pd_basis}
              </span>
            </Row>
            <Row label="Source">
              <SourceLink label={s.source} url={s.landing_url} />
            </Row>
          </dl>

          {s.chunks > 0 && (
            <div className="mt-3 border-t border-line pt-2.5 text-[12px] text-ink-faint tabular-nums">
              {PASSAGES_FMT.format(s.chunks)} retrievable passages
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-2">
      <dt className="w-24 shrink-0 text-ink-faint">{label}</dt>
      <dd className="min-w-0 flex-1">{children}</dd>
    </div>
  );
}

function SourceLink({ label, url }: { label: string; url: string }) {
  if (!url) return <span className="text-accent-ink">{label}</span>;
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-1 text-accent-ink underline decoration-line-strong underline-offset-2"
    >
      {label}
      <span className="text-[11px] text-ink-faint" aria-hidden>
        ↗
      </span>
    </a>
  );
}
