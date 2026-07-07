import { Suspense, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { ErrorBoundary } from "@/components/ErrorBoundary";
import { HealthBadge } from "@/components/HealthBadge";

const NAV = [
  { to: "/", label: "Ask", end: true },
  { to: "/sources", label: "Sources", end: false },
  { to: "/evals", label: "Evals", end: false },
  { to: "/security", label: "Security", end: false },
  { to: "/how-it-works", label: "How it works", end: false },
];

export function Layout() {
  const [menuOpen, setMenuOpen] = useState(false);
  const { pathname } = useLocation();

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-30 border-b border-line bg-surface/85 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3">
          <span className="font-serif text-[17px] font-bold tracking-[0.2px] text-ink">
            Antique Historian
          </span>

          {/* Desktop nav — inline links, hidden on small screens. */}
          <nav className="hidden items-center gap-1 text-sm md:flex">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end} className={desktopLinkClass}>
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <HealthBadge />
            {/* Hamburger — only below md, where the inline nav is hidden. */}
            <button
              type="button"
              className="-mr-1 rounded-md p-1.5 text-ink-soft hover:bg-accent-soft hover:text-accent-ink md:hidden"
              aria-label="Toggle navigation menu"
              aria-expanded={menuOpen}
              aria-controls="mobile-nav"
              onClick={() => setMenuOpen((open) => !open)}
            >
              <MenuIcon open={menuOpen} />
            </button>
          </div>
        </div>

        {/* Mobile nav — collapsible panel; closes on selection. */}
        {menuOpen && (
          <nav id="mobile-nav" className="border-t border-line px-3 py-2 md:hidden">
            <ul className="flex flex-col gap-1">
              {NAV.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    className={mobileLinkClass}
                    onClick={() => setMenuOpen(false)}
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        )}
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">
        {/* Keyed by path so a crash on one page recovers when the user navigates;
            Suspense covers the lazily-loaded route chunks while they download. */}
        <ErrorBoundary key={pathname}>
          <Suspense fallback={<RouteFallback />}>
            <Outlet />
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  );
}

/** Brief placeholder shown while a lazily-loaded route chunk downloads. */
function RouteFallback() {
  return (
    <div className="flex items-center gap-2 py-16 text-sm text-ink-faint" role="status">
      <span className="h-2 w-2 animate-pulse rounded-full bg-ink-faint" aria-hidden />
      Loading…
    </div>
  );
}

function desktopLinkClass({ isActive }: { isActive: boolean }): string {
  return [
    "rounded-md px-3 py-1.5 transition-colors",
    isActive ? "bg-accent-soft text-accent-ink" : "text-ink-soft hover:text-ink",
  ].join(" ");
}

function mobileLinkClass({ isActive }: { isActive: boolean }): string {
  return [
    "block rounded-md px-3 py-2 text-sm transition-colors",
    isActive
      ? "bg-accent-soft font-semibold text-accent-ink"
      : "text-ink-soft hover:bg-accent-soft hover:text-accent-ink",
  ].join(" ");
}

function MenuIcon({ open }: { open: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
      {open ? (
        <>
          <line x1="6" y1="6" x2="18" y2="18" />
          <line x1="18" y1="6" x2="6" y2="18" />
        </>
      ) : (
        <>
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </>
      )}
    </svg>
  );
}
