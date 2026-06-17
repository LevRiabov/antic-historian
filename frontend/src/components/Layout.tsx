import { NavLink, Outlet } from "react-router-dom";

import { HealthBadge } from "@/components/HealthBadge";

const NAV = [
  { to: "/", label: "Ask", end: true },
  { to: "/sources", label: "Sources", end: false },
  { to: "/evals", label: "Evals", end: false },
  { to: "/security", label: "Security", end: false },
  { to: "/how-it-works", label: "How it works", end: false },
];

export function Layout() {
  return (
    <div className="flex min-h-full flex-col">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-3">
          <span className="font-serif text-[17px] font-bold tracking-[0.2px] text-ink">
            Antic Historian
          </span>
          <nav className="flex items-center gap-1 text-sm">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  [
                    "rounded-md px-3 py-1.5 transition-colors",
                    isActive
                      ? "bg-accent-soft text-accent-ink"
                      : "text-ink-soft hover:text-ink",
                  ].join(" ")
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto">
            <HealthBadge />
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
