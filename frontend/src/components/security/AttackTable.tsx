import { Fragment, useState, type ReactNode } from "react";

import { AttackRowDetail } from "@/components/security/AttackRowDetail";
import { OutcomeChip } from "@/components/security/OutcomeChip";
import { ColorBadge } from "@/components/ui/ColorBadge";
import { SEC_CAT_META, type AttackRow } from "@/lib/security";

// One row per attack; click to expand the baseline-vs-defended detail. Two
// outcome columns: how the raw model behaved, and how the defended stack behaved.
export function AttackTable({ rows }: { rows: readonly AttackRow[] }) {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <div className="mt-4 overflow-hidden rounded-2xl border border-line bg-surface shadow-card">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[920px] border-collapse">
          <thead>
            <tr>
              <Th className="w-[30px]" />
              <Th className="w-[92px]">ID</Th>
              <Th className="w-[160px]">Category</Th>
              <Th>Attack</Th>
              <Th center sep className="w-[130px]">
                Baseline
              </Th>
              <Th center className="w-[130px]">
                Defended
              </Th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-5 py-12 text-center text-sm text-ink-faint">
                  No attacks match these filters.
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
                      <td className="px-3 py-3 text-center text-ink-faint">
                        <span
                          className={`inline-block text-[11px] transition-transform ${isOpen ? "rotate-90" : ""}`}
                          aria-hidden
                        >
                          ▸
                        </span>
                      </td>
                      <td className="px-3 py-3 font-mono text-[12.5px] whitespace-nowrap text-ink-soft">
                        {row.id}
                      </td>
                      <td className="px-3 py-3">
                        <ColorBadge label={row.category} color={SEC_CAT_META[row.category].color} />
                      </td>
                      <td className="max-w-px px-3 py-3">
                        <div
                          className="overflow-hidden font-mono text-[13px] text-ellipsis whitespace-nowrap text-ink"
                          title={row.attack}
                        >
                          {row.attack}
                        </div>
                      </td>
                      <td className="border-l border-line-strong px-3 py-3 text-center whitespace-nowrap">
                        <OutcomeChip breached={row.baseSucceeded} />
                      </td>
                      <td className="px-3 py-3 text-center whitespace-nowrap">
                        <OutcomeChip breached={row.defSucceeded} />
                      </td>
                    </tr>
                    {isOpen && (
                      <tr>
                        <td colSpan={6} className="border-b border-line bg-[#fcfbf9] p-0">
                          <AttackRowDetail row={row} />
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
    </div>
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
