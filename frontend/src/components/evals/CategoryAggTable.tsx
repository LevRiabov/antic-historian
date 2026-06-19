import { MetricChip } from "@/components/evals/MetricChip";
import {
  fmtMrr,
  fmtPct,
  fmtScore,
  mrrTone,
  recallTone,
  scoreTone,
  type CategoryAggRow,
} from "@/lib/evals";

// Two metric tiers, grouped under labeled spanning headers like the mock:
//   Retrieval — recall@5 · recall@20 · MRR   (free, run constantly)
//   Judge     — faithful · complete · attribution   (LLM-judged at phase boundaries)
export function CategoryAggTable({ rows }: { rows: readonly CategoryAggRow[] }) {
  return (
    <section className="mt-4 overflow-hidden rounded-xl border border-line bg-surface shadow-card">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <h2 className="text-[13px] font-bold tracking-wide text-ink-soft uppercase">
          Per-category results
        </h2>
        <span className="flex gap-3.5 text-[11px] text-ink-faint">
          <span>
            <b className="text-ink-soft">Retrieval</b> recall@5 · recall@20 · MRR
          </span>
          <span>
            <b className="text-ink-soft">Judge</b> faithful · complete · attribution
          </span>
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] border-collapse">
          <thead>
            <tr>
              <th />
              <th
                colSpan={3}
                className="border-l border-line-strong px-3.5 pt-2.5 pb-1 text-center text-[11px] font-semibold tracking-wide text-[#3a6ea5] uppercase"
              >
                Retrieval
              </th>
              <th
                colSpan={3}
                className="border-l border-line-strong px-3.5 pt-2.5 pb-1 text-center text-[11px] font-semibold tracking-wide text-accent-ink uppercase"
              >
                Judge
              </th>
            </tr>
            <tr className="text-[10.5px] font-bold tracking-wide text-ink-faint uppercase">
              <th className="bg-[#f6f4f0] px-3.5 py-2 text-left">Category</th>
              <SubHead sep>R@5</SubHead>
              <SubHead>R@20</SubHead>
              <SubHead>MRR</SubHead>
              <SubHead sep>Faithful</SubHead>
              <SubHead>Complete</SubHead>
              <SubHead>Attrib.</SubHead>
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
                {row.refusalAccuracy !== null ? (
                  // out-of-scope: only a refusal-accuracy number; the rest is N/A
                  <>
                    <Cell sep>
                      <MetricChip
                        value={fmtPct(row.refusalAccuracy)}
                        tone={recallTone(row.refusalAccuracy * 100)}
                      />
                      <div className="mt-0.5 text-[10px] text-ink-faint">refusal</div>
                    </Cell>
                    <Cell>
                      <MetricChip value={null} tone={null} />
                    </Cell>
                    <Cell>
                      <MetricChip value={null} tone={null} />
                    </Cell>
                    <Cell sep>
                      <MetricChip value={null} tone={null} />
                    </Cell>
                    <Cell>
                      <MetricChip value={null} tone={null} />
                    </Cell>
                    <Cell>
                      <MetricChip value={null} tone={null} />
                    </Cell>
                  </>
                ) : (
                  <>
                    <Cell sep>
                      <Recall pct={row.recall5} />
                    </Cell>
                    <Cell>
                      <Recall pct={row.recall20} />
                    </Cell>
                    <Cell>
                      <Mrr value={row.mrr} />
                    </Cell>
                    <Cell sep>
                      <Score value={row.faithfulness} />
                    </Cell>
                    <Cell>
                      <Score value={row.completeness} />
                    </Cell>
                    <Cell>
                      <Score value={row.attribution} />
                    </Cell>
                  </>
                )}
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

// percentage / score / mrr cells that gracefully dash when the metric is absent
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
    <MetricChip value={fmtScore(value)} tone={scoreTone(value)} />
  );
}
function Mrr({ value }: { value: number | null }) {
  return value === null ? (
    <MetricChip value={null} tone={null} />
  ) : (
    <MetricChip value={fmtMrr(value)} tone={mrrTone(value)} />
  );
}
