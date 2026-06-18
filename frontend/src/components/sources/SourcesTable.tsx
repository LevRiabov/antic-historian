import type { ReactNode } from "react";

import { Pill } from "@/components/ui/Pill";
import type { SortDir, SortKey } from "@/lib/sources";
import type { SourceOut } from "@/lib/types";

const PASSAGES_FMT = new Intl.NumberFormat("en-US");

export function SourcesTable({
  rows,
  sortKey,
  sortDir,
  onSort,
}: {
  rows: readonly SourceOut[];
  sortKey: SortKey | null;
  sortDir: SortDir;
  onSort: (key: SortKey) => void;
}) {
  return (
    <div className="hidden overflow-x-auto rounded-2xl border border-line bg-surface shadow-card lg:block">
      <table className="w-full min-w-[920px] border-collapse">
        <thead>
          <tr className="bg-[#f7f5f1]">
            <SortHeader label="Author" col="author" {...{ sortKey, sortDir, onSort }} />
            <SortHeader label="Work" col="title" {...{ sortKey, sortDir, onSort }} />
            <HeadCell>Translator</HeadCell>
            <HeadCell>Type</HeadCell>
            <HeadCell>Public-domain basis</HeadCell>
            <HeadCell>Source</HeadCell>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={6} className="px-5 py-12 text-center text-sm text-ink-faint">
                No sources match your filter.
              </td>
            </tr>
          ) : (
            rows.map((s) => (
              <tr
                key={s.pg_id}
                className="border-b border-line last:border-0 even:bg-[#fcfbf9] hover:bg-accent-soft"
              >
                <td className="px-4 py-3 align-top font-serif text-[15.5px] font-semibold whitespace-nowrap text-ink">
                  {s.author}
                </td>
                <td className="min-w-[240px] px-4 py-3 align-top text-sm">
                  <span className="text-ink">{s.title}</span>
                  {s.chunks > 0 && (
                    <span className="ml-2 inline-block rounded-full bg-bg align-[1px] px-2 py-0.5 text-[11px] font-medium whitespace-nowrap text-ink-faint ring-1 ring-inset ring-line">
                      {PASSAGES_FMT.format(s.chunks)} passages
                    </span>
                  )}
                </td>
                <td className="max-w-[220px] px-4 py-3 align-top text-[13.5px] text-ink-soft">
                  {s.translator}
                </td>
                <td className="px-4 py-3 align-top">
                  <Pill variant={s.category === "primary" ? "accent" : "good"}>
                    {s.category === "primary" ? "Primary" : "Scholarship"}
                  </Pill>
                </td>
                <td className="max-w-[230px] px-4 py-3 align-top text-[12.8px] leading-snug text-ink-soft">
                  <span className="mr-1.5 text-good" aria-hidden>
                    ✓
                  </span>
                  {s.pd_basis}
                </td>
                <td className="px-4 py-3 align-top">
                  <SourceChip label={s.source} url={s.landing_url} />
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function HeadCell({ children }: { children: ReactNode }) {
  return (
    <th className="border-b border-line-strong px-4 py-3 text-left text-[11.5px] font-semibold tracking-wider whitespace-nowrap text-ink-soft uppercase">
      {children}
    </th>
  );
}

function SortHeader({
  label,
  col,
  sortKey,
  sortDir,
  onSort,
}: {
  label: string;
  col: SortKey;
  sortKey: SortKey | null;
  sortDir: SortDir;
  onSort: (key: SortKey) => void;
}) {
  const active = sortKey === col;
  const caret = !active ? "⇅" : sortDir === "asc" ? "▲" : "▼";
  return (
    <th
      aria-sort={active ? (sortDir === "asc" ? "ascending" : "descending") : "none"}
      className="border-b border-line-strong px-4 py-3 text-left text-[11.5px] font-semibold tracking-wider whitespace-nowrap text-ink-soft uppercase"
    >
      <button
        type="button"
        onClick={() => onSort(col)}
        className="inline-flex items-center gap-1.5 uppercase hover:text-accent-ink"
      >
        {label}
        <span className={`text-[10px] ${active ? "text-accent" : "text-ink-faint"}`} aria-hidden>
          {caret}
        </span>
      </button>
    </th>
  );
}

function SourceChip({ label, url }: { label: string; url: string }) {
  const inner = (
    <>
      {label}
      {url && (
        <span className="text-[11px] text-ink-faint" aria-hidden>
          ↗
        </span>
      )}
    </>
  );
  const base =
    "inline-flex items-center gap-1 rounded-full border border-line-strong bg-surface px-2.5 py-0.5 text-[12.5px] whitespace-nowrap text-accent-ink";
  if (!url) return <span className={base}>{inner}</span>;
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className={`${base} transition-colors hover:border-accent hover:bg-accent-soft`}
    >
      {inner}
    </a>
  );
}
