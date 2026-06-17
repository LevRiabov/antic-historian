import { createBrowserRouter, Navigate } from "react-router-dom";

import { Layout } from "@/components/Layout";
import { Chat } from "@/routes/Chat";
import { Sources } from "@/routes/Sources";
import { Evals } from "@/routes/Evals";
import { Security } from "@/routes/Security";
import { HowItWorks } from "@/routes/HowItWorks";

// Route table mirrors the four design mockups (chat / sources / golden-evals /
// security) plus a "how it works" explainer. Pages are stubs for now — Phase 7
// fills them in against the typed API client in lib/.
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
