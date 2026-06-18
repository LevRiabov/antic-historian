// Rounded search box with a leading magnifier. Controlled — the parent owns the
// query string. Reusable across the table pages.
export function SearchInput({
  value,
  onChange,
  placeholder = "Search…",
  ariaLabel = "Search",
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  ariaLabel?: string;
}) {
  return (
    <div className="relative w-full sm:w-auto">
      <svg
        className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-ink-faint"
        width="15"
        height="15"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        aria-hidden
      >
        <circle cx="11" cy="11" r="7" />
        <line x1="16.5" y1="16.5" x2="21" y2="21" />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-label={ariaLabel}
        autoComplete="off"
        className="w-full rounded-full border border-line-strong bg-surface py-2 pr-3.5 pl-9 text-sm text-ink shadow-card outline-none transition-colors placeholder:text-ink-faint focus:border-accent focus:ring-3 focus:ring-accent-soft sm:w-64"
      />
    </div>
  );
}
