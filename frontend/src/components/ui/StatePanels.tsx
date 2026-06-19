// Shared loading + error states for the data-backed table pages (sources, evals).
// Kept generic so each page passes its own copy text.

/** Pulsing placeholder shaped like a table card while a query is in flight. */
export function TableSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div className="mt-4 animate-pulse rounded-2xl border border-line bg-surface p-4 shadow-card">
      <div className="h-9 rounded bg-line/60" />
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="mt-3 h-6 rounded bg-line/40" />
      ))}
    </div>
  );
}

/** A failed-request panel with the raw error and a retry button. */
export function ErrorPanel({
  title,
  message,
  onRetry,
}: {
  title: string;
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="mt-4 rounded-2xl border border-line bg-surface p-8 text-center shadow-card">
      <p className="text-sm text-refuse">{title}</p>
      <p className="mt-1 font-mono text-xs text-ink-faint">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-4 rounded-full bg-accent-soft px-4 py-1.5 text-sm font-semibold text-accent-ink hover:bg-accent/10"
      >
        Retry
      </button>
    </div>
  );
}
