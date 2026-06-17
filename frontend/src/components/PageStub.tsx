import type { ReactNode } from "react";

// Placeholder used by every route until Phase 7 ports the matching design
// mockup. Keeps the shell navigable and visually on-brand in the meantime.
export function PageStub({
  title,
  mockup,
  children,
}: {
  title: string;
  mockup?: string;
  children?: ReactNode;
}) {
  return (
    <section>
      <h1 className="font-serif text-2xl font-bold text-ink">{title}</h1>
      <div className="mt-4 rounded-lg border border-line bg-surface p-6 shadow-card">
        <p className="text-sm text-ink-soft">{children ?? "Coming in Phase 7."}</p>
        {mockup && (
          <p className="mt-3 font-mono text-xs text-ink-faint">
            design reference: frontend/design/{mockup}
          </p>
        )}
      </div>
    </section>
  );
}
