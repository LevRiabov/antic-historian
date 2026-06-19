import { lazy } from "react";
import { createBrowserRouter, Navigate } from "react-router-dom";

import { Layout } from "@/components/Layout";
import { Chat } from "@/routes/Chat";

// Chat is the landing route, so it stays eager. The four data pages are split into
// their own chunks (React.lazy) — they each pull in TanStack Query + their tables,
// and most visitors land on Chat, so deferring them trims the initial bundle. The
// routes export named components, so adapt each to the default export lazy() wants.
const Sources = lazy(() => import("@/routes/Sources").then((m) => ({ default: m.Sources })));
const Evals = lazy(() => import("@/routes/Evals").then((m) => ({ default: m.Evals })));
const Security = lazy(() => import("@/routes/Security").then((m) => ({ default: m.Security })));
const HowItWorks = lazy(() =>
  import("@/routes/HowItWorks").then((m) => ({ default: m.HowItWorks })),
);

// Route table mirrors the four design mockups (chat / sources / golden-evals /
// security) plus a "how it works" explainer. Layout renders the lazy routes inside
// a Suspense + error boundary.
export const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Chat /> },
      { path: "sources", element: <Sources /> },
      { path: "evals", element: <Evals /> },
      { path: "security", element: <Security /> },
      { path: "how-it-works", element: <HowItWorks /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);
