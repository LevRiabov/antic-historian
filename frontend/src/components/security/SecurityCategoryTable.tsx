import { MetricChip } from "@/components/evals/MetricChip";
import { asrTone, breachTone, fmtAsr, type SecCategoryRow } from "@/lib/security";

// ASR by category, grouped under spanning headers like the golden-set table:
//   Baseline · no defence   vs   Defended · defence-stack
// Each side shows ASR % and the raw breach count.
export function SecurityCategoryTable({ rows }: { rows: readonly SecCategoryRow[] }) {
  return (
    <section className="mt-4 overflow-hidden rounded-xl border border-line bg-surface shadow-card">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <h2 className="text-[13px] font-bold tracking-wide text-ink-soft uppercase">
          Attack success rate by category
        </h2>
        <span className="text-xs text-ink-faint">
          ASR = share of attacks that breached the assistant — lower is better
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] border-collapse">
          <thead>
            <tr>
              <th />
              <th />
              <th
                colSpan={2}
                className="border-l border-line-strong px-3.5 pt-2.5 pb-1 text-center text-[11px] font-semibold tracking-wide text-danger uppercase"
              >
                Baseline · no defence
              </th>
              <th
                colSpan={2}
                className="border-l border-line-strong px-3.5 pt-2.5 pb-1 text-center text-[11px] font-semibold tracking-wide text-good uppercase"
              >
                Defended · defence-stack
              </th>
            </tr>
            <tr className="text-[10.5px] font-bold tracking-wide text-ink-faint uppercase">
              <th className="bg-[#f6f4f0] px-3.5 py-2 text-left">Category</th>
              <SubHead>n</SubHead>
              <SubHead sep>ASR</SubHead>
              <SubHead>breaches</SubHead>
              <SubHead sep>ASR</SubHead>
              <SubHead>breaches</SubHead>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.category}
                className={
                  row.category === "overall"
                    ? "bg-[#f6f4f0] font-bold"
                    : "border-t border-line hover:bg-[#fcfbf9]"
                }
              >
                <td className="px-3.5 py-2.5 text-left text-[13px] font-semibold whitespace-nowrap">
                  {row.color && (
                    <span
                      className="mr-2 inline-block h-2.5 w-2.5 rounded-full align-middle"
                      style={{ background: row.color }}
                      aria-hidden
                    />
                  )}
                  {row.label}
                </td>
                <Cell>{row.count}</Cell>
                <Cell sep>
                  <MetricChip value={fmtAsr(row.baseAsr / 100)} tone={asrTone(row.baseAsr)} />
                </Cell>
                <Cell>
                  <MetricChip value={String(row.baseBreaches)} tone={breachTone(row.baseBreaches)} />
                </Cell>
                <Cell sep>
                  <MetricChip value={fmtAsr(row.defAsr / 100)} tone={asrTone(row.defAsr)} />
                </Cell>
                <Cell>
                  <MetricChip value={String(row.defBreaches)} tone={breachTone(row.defBreaches)} />
                </Cell>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SubHead({ children, sep }: { children: React.ReactNode; sep?: boolean }) {
  return (
    <th
      className={`bg-[#f6f4f0] px-3.5 py-2 text-center ${sep ? "border-l border-line-strong" : ""}`}
    >
      {children}
    </th>
  );
}

function Cell({ children, sep }: { children: React.ReactNode; sep?: boolean }) {
  return (
    <td
      className={`px-3.5 py-2.5 text-center whitespace-nowrap ${sep ? "border-l border-line-strong" : ""}`}
    >
      {children}
    </td>
  );
}
