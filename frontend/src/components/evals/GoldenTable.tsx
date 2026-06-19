import { Fragment, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";

import { CategoryBadge } from "@/components/evals/CategoryBadge";
import { GoldenRowDetail } from "@/components/evals/GoldenRowDetail";
import { MetricChip } from "@/components/evals/MetricChip";
import { SourceDrawer } from "@/components/SourceDrawer";
import { getChunks } from "@/lib/api";
import {
  fmtMrr,
  fmtScore,
  mrrTone,
  recallTone,
  scoreTone,
  type GoldenRow,
  type MetricTone,
} from "@/lib/evals";

// One row per golden question; click to expand the detail panel. Six metric
// columns grouped Retrieval | Judge. Out-of-scope rows have no metrics — they
// span those columns with the refusal verdict instead.
export function GoldenTable({
  rows,
  works,
}: {
  rows: readonly GoldenRow[];
  works: Map<number, string>;
}) {
  const [open, setOpen] = useState<string | null>(null);
  // The cited passage shown in the drawer; fetched by id (the records carry only ids).
  const [citation, setCitation] = useState<{ chunkId: number; marker?: number } | null>(null);
  const chunkQuery = useQuery({
    queryKey: ["chunk", citation?.chunkId],
    queryFn: ({ signal }) => getChunks([citation!.chunkId], signal),
    enabled: citation !== null,
    staleTime: Infinity, // a published passage is immutable; cache it across opens
  });

  return (
    <div className="mt-4 overflow-hidden rounded-2xl border border-line bg-surface shadow-card">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[1060px] border-collapse">
          <thead>
            <tr>
              <th colSpan={4} />
              <GroupHead className="text-[#3a6ea5]">Retrieval</GroupHead>
              <GroupHead className="text-accent-ink">Judge</GroupHead>
            </tr>
            <tr>
              <Th className="w-[30px]" />
              <Th className="w-[88px]">ID</Th>
              <Th className="w-[132px]">Category</Th>
              <Th>Question</Th>
              <Th center sep className="w-[66px]">
                R@5
              </Th>
              <Th center className="w-[66px]">
                R@20
              </Th>
              <Th center className="w-[64px]">
                MRR
              </Th>
              <Th center sep className="w-[70px]">
                Faithful
              </Th>
              <Th center className="w-[74px]">
                Complete
              </Th>
              <Th center className="w-[78px]">
                Attrib.
              </Th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={10} className="px-5 py-12 text-center text-sm text-ink-faint">
                  No questions match these filters.
                </td>
              </tr>
            ) : (
              rows.map((row) => {
                const isOpen = open === row.id;
                return (
                  <Fragment key={row.id}>
                    <tr
                      onClick={() => setOpen(isOpen ? null : row.id)}
                      aria-expanded={isOpen}
                      className={`cursor-pointer border-b border-line transition-colors hover:bg-accent-soft ${
                        isOpen ? "bg-accent-soft" : ""
                      }`}
                    >
                      <td className="px-3 py-3 text-center align-top text-ink-faint">
                        <span
                          className={`inline-block text-[11px] transition-transform ${isOpen ? "rotate-90" : ""}`}
                          aria-hidden
                        >
                          ▸
                        </span>
                      </td>
                      <td className="px-3 py-3 align-top font-mono text-[12.5px] whitespace-nowrap text-ink-soft">
                        {row.id}
                      </td>
                      <td className="px-3 py-3 align-top">
                        <CategoryBadge category={row.category} />
                      </td>
                      <td className="px-3 py-3 align-top">
                        <div className="text-[14.5px] leading-snug text-ink">{row.question}</div>
                      </td>
                      {row.category === "out-of-scope" ? (
                        <td
                          colSpan={6}
                          className="border-l border-line-strong px-3 py-3 text-center align-top whitespace-nowrap"
                        >
                          <RefusalChip pass={row.refusalCorrect} />
                        </td>
                      ) : (
                        <>
                          <MetricTd sep>
                            <Recall pct={row.recall5} />
                          </MetricTd>
                          <MetricTd>
                            <Recall pct={row.recall20} />
                          </MetricTd>
                          <MetricTd>
                            <Mrr value={row.mrr} />
                          </MetricTd>
                          <MetricTd sep>
                            <Score value={row.faithfulness} />
                          </MetricTd>
                          <MetricTd>
                            <Score value={row.completeness} />
                          </MetricTd>
                          <MetricTd>
                            <Score value={row.attribution} />
                          </MetricTd>
                        </>
                      )}
                    </tr>
                    {isOpen && (
                      <tr>
                        <td colSpan={10} className="border-b border-line bg-[#fcfbf9] p-0">
                          <GoldenRowDetail
                            row={row}
                            works={works}
                            onOpenCitation={(chunkId, marker) => setCitation({ chunkId, marker })}
                          />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <SourceDrawer
        open={citation !== null}
        onClose={() => setCitation(null)}
        marker={citation?.marker}
        passage={chunkQuery.data?.[0] ?? null}
        loading={chunkQuery.isLoading}
        error={chunkQuery.isError}
      />
    </div>
  );
}

function GroupHead({ children, className }: { children: ReactNode; className: string }) {
  return (
    <th
      colSpan={3}
      className={`border-l border-line-strong px-3 pt-2.5 pb-1 text-center text-[11px] font-bold tracking-wide uppercase ${className}`}
    >
      {children}
    </th>
  );
}

function Th({
  children,
  center,
  sep,
  className = "",
}: {
  children?: ReactNode;
  center?: boolean;
  sep?: boolean;
  className?: string;
}) {
  return (
    <th
      className={`border-b border-line-strong bg-[#f6f4f0] px-3 py-2.5 text-[11px] font-bold tracking-wider whitespace-nowrap text-ink-faint uppercase ${
        center ? "text-center" : "text-left"
      } ${sep ? "border-l border-line-strong" : ""} ${className}`}
    >
      {children}
    </th>
  );
}

function MetricTd({ children, sep }: { children: ReactNode; sep?: boolean }) {
  return (
    <td
      className={`px-3 py-3 text-center align-top whitespace-nowrap ${sep ? "border-l border-line-strong" : ""}`}
    >
      {children}
    </td>
  );
}

function RefusalChip({ pass }: { pass: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1 text-[12.5px] font-semibold ${
        pass ? "border-good/25 bg-good-soft text-good" : "border-refuse/30 bg-refuse-soft text-refuse"
      }`}
    >
      {pass ? "Refusal ✓" : "Refusal ✗"}
    </span>
  );
}

function Recall({ pct }: { pct: number | null }) {
  return pct === null ? (
    <MetricChip value={null} tone={null} />
  ) : (
    <MetricChip value={`${Math.round(pct)}%`} tone={recallTone(pct)} />
  );
}
function Score({ value }: { value: number | null }) {
  return value === null ? (
    <MetricChip value={null} tone={null} />
  ) : (
    <MetricChip value={fmtScore(value)} tone={scoreTone(value) as MetricTone} />
  );
}
function Mrr({ value }: { value: number | null }) {
  return value === null ? (
    <MetricChip value={null} tone={null} />
  ) : (
    <MetricChip value={fmtMrr(value)} tone={mrrTone(value)} />
  );
}
