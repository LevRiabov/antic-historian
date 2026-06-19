import { Component, type ErrorInfo, type ReactNode } from "react";

/*
 * A render-time safety net. Without it, any thrown error in a child unmounts the
 * whole React tree and leaves a blank page. Used in two places: around the app
 * shell (main.tsx) as the last resort, and around the routed <Outlet> (Layout)
 * keyed by path so a single page crashing keeps the nav usable and recovers on
 * navigation. Errors here are unexpected (the data layer already handles fetch
 * failures), so this is a fallback, not a feature.
 */
interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}
interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Surfaced in the console for debugging; no telemetry sink wired up for the demo.
    console.error("Unhandled UI error:", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    if (this.props.fallback !== undefined) return this.props.fallback;
    return (
      <div className="mx-auto max-w-md px-6 py-16 text-center">
        <h1 className="font-serif text-2xl font-semibold text-ink">Something broke on this page</h1>
        <p className="mt-2 text-[15px] text-ink-soft">
          An unexpected error stopped this view from rendering. Reloading usually clears it.
        </p>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="mt-5 rounded-full bg-accent px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-accent-ink"
        >
          Reload
        </button>
      </div>
    );
  }
}
