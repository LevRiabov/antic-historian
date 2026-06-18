import type { ReactNode } from "react";

// Small rounded status badge. Variants are color *intents* (not domain terms) so
// other table pages (evals, security) can reuse them — the caller maps its domain
// value to an intent.
export type PillVariant = "accent" | "good" | "amber" | "neutral";

const VARIANT_CLASS: Record<PillVariant, string> = {
  accent: "bg-accent-soft text-accent-ink ring-line-strong",
  good: "bg-good-soft text-good ring-good/20",
  amber: "bg-amber-soft text-amber ring-amber/20",
  neutral: "bg-bg text-ink-soft ring-line-strong",
};

export function Pill({
  variant = "neutral",
  children,
}: {
  variant?: PillVariant;
  children: ReactNode;
}) {
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-[11.5px] font-semibold whitespace-nowrap ring-1 ring-inset ${VARIANT_CLASS[variant]}`}
    >
      {children}
    </span>
  );
}
