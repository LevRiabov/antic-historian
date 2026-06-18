// Segmented pill-tabs with optional counts. Generic over the value type so any
// table page can drive it with its own filter union.
export interface TabItem<T extends string> {
  value: T;
  label: string;
  count?: number;
}

export function FilterTabs<T extends string>({
  items,
  value,
  onChange,
  ariaLabel = "Filter",
}: {
  items: readonly TabItem<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className="flex gap-1 rounded-full bg-accent-soft p-1"
    >
      {items.map((item) => {
        const active = item.value === value;
        return (
          <button
            key={item.value}
            role="tab"
            aria-selected={active}
            type="button"
            onClick={() => onChange(item.value)}
            className={`flex items-center gap-1.5 rounded-full px-4 py-1.5 text-[13.5px] whitespace-nowrap transition-colors ${
              active
                ? "bg-surface font-semibold text-accent-ink shadow-card"
                : "text-accent-ink/80 hover:bg-accent/10"
            }`}
          >
            {item.label}
            {item.count !== undefined && (
              <span
                className={`text-xs tabular-nums ${active ? "text-accent" : "text-ink-faint"}`}
              >
                {item.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
