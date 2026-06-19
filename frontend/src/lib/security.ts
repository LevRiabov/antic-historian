/*
 * Pure transforms for the /security page — joining the baseline + defended audit
 * runs by attack id, deriving the ASR aggregates + per-category table, and
 * filtering. Framework-free and side-effect-free, mirroring lib/evals.ts.
 *
 * One table from two handlers: the baseline run (no defence) is the spine and the
 * defended run (defence-stack) attaches by id — the before/after of each attack.
 */
import type { MetricTone } from "./evals";
import type { AttackCategory, SecurityAggregates, SecurityRun } from "./types";

// The exact answer the output filter substitutes when it withholds a response —
// mirrors ahx.guard.SECURITY_REDACTION; lets us tell "blocked by filter" from
// "held by behaving correctly" without a separate flag on the record.
export const SECURITY_REDACTION = "[blocked: response withheld by a security filter]";

export const SEC_CAT_ORDER: readonly AttackCategory[] = [
  "extraction",
  "scope-escape",
  "grounding-bypass",
  "citation-forgery",
  "fake-source-injection",
];

export interface AttackCategoryMeta {
  color: string;
  desc: string;
}

// Carried from the design mock (security-01-table.html): hue + one-line gloss.
export const SEC_CAT_META: Record<AttackCategory, AttackCategoryMeta> = {
  extraction: {
    color: "#b3322a",
    desc: "Prying out the hidden system prompt or session token (prompt-leak / secret exfiltration).",
  },
  "scope-escape": {
    color: "#8a5a2b",
    desc: "Jailbreaks pushing the assistant off its corpus to act as a general chatbot.",
  },
  "grounding-bypass": {
    color: "#a05a8a",
    desc: "Coaxing an ungrounded claim — answering from model memory instead of the sources.",
  },
  "citation-forgery": {
    color: "#3a6ea5",
    desc: "Inducing fabricated or out-of-range source citations.",
  },
  "fake-source-injection": {
    color: "#2f8f8f",
    desc: "Smuggling an attacker-controlled 'Source 9' to be trusted, cited, or obeyed.",
  },
};

/* ---- tone rules (from the mock) ---- */

/** ASR is a percentage (0-100): zero is good, a trickle is amber, real breaches danger. */
export function asrTone(pct: number): MetricTone {
  return pct === 0 ? "good" : pct < 20 ? "amber" : "danger";
}
/** A raw breach count: zero held, anything above is a breach. */
export function breachTone(count: number): MetricTone {
  return count === 0 ? "good" : "danger";
}
export function fmtAsr(fraction: number): string {
  // one decimal, matching the mock (e.g. 17.5%)
  return `${Math.round(fraction * 1000) / 10}%`;
}
export function fmtLatency(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

/* ---- the merged per-attack row ---- */

export interface AttackRow {
  id: string;
  category: AttackCategory;
  attack: string; // the attack prompt
  // baseline (no defence)
  baseSucceeded: boolean; // true = breached
  baseRefused: boolean;
  baseAnswer: string;
  baseDangling: number[];
  baseLatencyMs: number;
  // defended (defence-stack)
  defSucceeded: boolean;
  defRefused: boolean;
  defAnswer: string;
  defBlocked: boolean; // answer was withheld by the output filter (vs held by behaving)
}

/** Join the two runs by attack id. The baseline drives the row set; the defended
 *  result attaches by id. If an attack is missing from the defended run it's
 *  treated as still breached (fail-safe — never report a hole as closed). */
export function mergeSecurity(baseline: SecurityRun, defended: SecurityRun): AttackRow[] {
  const defById = new Map(defended.results.map((r) => [r.id, r]));
  return baseline.results.map((b) => {
    const d = defById.get(b.id);
    return {
      id: b.id,
      category: b.category,
      attack: b.attack_prompt,
      baseSucceeded: b.succeeded,
      baseRefused: b.refused,
      baseAnswer: b.answer,
      baseDangling: b.markers_dangling,
      baseLatencyMs: b.latency_ms,
      defSucceeded: d ? d.succeeded : true,
      defRefused: d ? d.refused : false,
      defAnswer: d ? d.answer : "(attack missing from the defended run)",
      defBlocked: d ? d.answer === SECURITY_REDACTION : false,
    };
  });
}

/** A one-line note on what a breached baseline answer leaked (display only). */
export function baseLeakNote(row: AttackRow): string | null {
  if (!row.baseSucceeded) return null;
  if (/You are a careful research assistant|Sources:/.test(row.baseAnswer)) {
    return "leaked system-prompt / source scaffolding";
  }
  if (row.category === "extraction") return "leaked the session token / secret";
  return "produced an ungrounded / off-scope response";
}

/* ---- filtering ---- */

export type SecFilter = "all" | AttackCategory;

export interface SecView {
  filter: SecFilter;
  query: string;
  breachesOnly: boolean; // baseline breaches only
}

export function selectAttacks(rows: readonly AttackRow[], view: SecView): AttackRow[] {
  const q = view.query.trim().toLowerCase();
  return rows.filter((row) => {
    if (view.filter !== "all" && row.category !== view.filter) return false;
    if (view.breachesOnly && !row.baseSucceeded) return false;
    if (q && !row.id.toLowerCase().includes(q) && !row.attack.toLowerCase().includes(q)) {
      return false;
    }
    return true;
  });
}

export type SecCategoryCounts = Record<SecFilter, number>;

export function categoryCounts(rows: readonly AttackRow[]): SecCategoryCounts {
  const counts = { all: rows.length } as SecCategoryCounts;
  for (const cat of SEC_CAT_ORDER) counts[cat] = 0;
  for (const row of rows) counts[row.category] += 1;
  return counts;
}

/* ---- aggregate stat bar ---- */

export interface SecStatTile {
  label: string;
  value: string;
  tone: MetricTone | null; // null = neutral
  sub: string;
  hero?: boolean; // the defended-ASR headline gets a highlighted card
  mono?: boolean; // render the value in the mono font (e.g. the model id)
}

export function buildSecStatTiles(
  baseline: SecurityRun,
  defended: SecurityRun,
): SecStatTile[] {
  const b = baseline.aggregates;
  const d = defended.aggregates;
  return [
    { label: "Attacks", value: String(b.attacks), tone: null, sub: `${SEC_CAT_ORDER.length} categories` },
    {
      label: "Baseline ASR",
      value: fmtAsr(b.asr),
      tone: "danger",
      sub: `${b.successes} of ${b.attacks} breached · no defence`,
    },
    {
      label: "Defended ASR",
      value: fmtAsr(d.asr),
      tone: asrTone(d.asr * 100),
      sub: `${d.successes} of ${d.attacks} breached · defence-stack`,
      hero: true,
    },
    {
      label: "Breaches closed",
      value: `${b.successes} → ${d.successes}`,
      tone: d.successes < b.successes ? "good" : null,
      sub: d.successes === 0 ? "every hole fixed" : `${b.successes - d.successes} closed`,
    },
    {
      label: "Model",
      value: baseline.chat_model,
      tone: null,
      sub: `${baseline.retriever} · ${baseline.prompt_version}`,
      mono: true,
    },
  ];
}

/* ---- per-category aggregate table ---- */

export interface SecCategoryRow {
  category: AttackCategory | "overall";
  label: string;
  color: string | null;
  count: number;
  baseAsr: number; // percentage
  baseBreaches: number;
  defAsr: number;
  defBreaches: number;
}

function catRow(
  category: AttackCategory | "overall",
  color: string | null,
  base: { count: number; successes: number; asr: number },
  def: { successes: number; asr: number },
): SecCategoryRow {
  return {
    category,
    label: category,
    color,
    count: base.count,
    baseAsr: base.asr * 100,
    baseBreaches: base.successes,
    defAsr: def.asr * 100,
    defBreaches: def.successes,
  };
}

const EMPTY_CAT = { count: 0, successes: 0, asr: 0 };

export function buildSecCategoryRows(
  baseline: SecurityAggregates,
  defended: SecurityAggregates,
): SecCategoryRow[] {
  const rows: SecCategoryRow[] = [];
  for (const cat of SEC_CAT_ORDER) {
    const base = baseline.by_category[cat];
    if (!base) continue;
    rows.push(catRow(cat, SEC_CAT_META[cat].color, base, defended.by_category[cat] ?? EMPTY_CAT));
  }
  rows.push(
    catRow(
      "overall",
      null,
      { count: baseline.attacks, successes: baseline.successes, asr: baseline.asr },
      { successes: defended.successes, asr: defended.asr },
    ),
  );
  return rows;
}
