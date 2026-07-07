import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { Pill } from "@/components/ui/Pill";

// Balanced explainer: what this is, what it does, how it works, where it stops,
// and what's next. Deliberately static narrative — the live numbers live on the
// Sources / Evals / Security pages, and this page links out to them rather than
// duplicating figures that still move as the system is tuned.
export function HowItWorks() {
  return (
    <section className="max-w-3xl">
      <header className="mb-8">
        <h1 className="font-serif text-3xl font-semibold tracking-[0.2px] text-ink">
          How it works
        </h1>
        <p className="mt-2 text-[15.5px] leading-relaxed text-ink-soft">
          Antique Historian answers questions about Greco-Roman antiquity — the late Roman
          Republic and the world around Caesar — using only a fixed library of public-domain
          books. Every claim it makes is grounded in a passage you can open and read. It is built
          to be <b className="font-semibold text-ink">trustworthy first</b>: when the sources
          don&rsquo;t support an answer, it says so instead of guessing.
        </p>
      </header>

      {/* What you can do */}
      <Section title="What you can do">
        <ul className="space-y-3">
          <Capability label="Ask">
            Put a question to the corpus and get a cited answer. Each claim links back to the exact
            source passage, so you verify rather than trust.
          </Capability>
          <Capability label="Deep mode">
            For harder questions — comparing accounts, surfacing contradictions across authors,
            synthesising several books — a multi-step agent searches, reads, and reasons before
            answering, streaming its steps as it goes.
          </Capability>
          <Capability label="Inspect">
            See exactly what the system knows. The{" "}
            <PageLink to="/sources">Sources</PageLink> page is the complete corpus; the live{" "}
            <PageLink to="/evals">Evals</PageLink> and{" "}
            <PageLink to="/security">Security</PageLink> pages show how well it retrieves and how it
            holds up under adversarial prompts.
          </Capability>
        </ul>
      </Section>

      {/* The pipeline */}
      <Section title="The pipeline, end to end">
        <p className="mb-5 text-[15px] leading-relaxed text-ink-soft">
          A question travels through four stages. Each one was chosen by measurement, not fashion —
          techniques only ship after a golden-set eval says they help.
        </p>
        <ol className="space-y-4">
          <Step n={1} title="Ingest — parse, then chunk">
            Public-domain texts are cleaned (Gutenberg boilerplate, footnotes, OCR noise),
            structurally parsed into book / chapter / section, tagged with metadata, then chunked.
            Citations point at character spans in the cleaned text, so they survive re-chunking.
          </Step>
          <Step n={2} title="Retrieve — contextual dense search">
            Each chunk is embedded and stored in a vector database. A question is embedded the same
            way and matched against the corpus. The shipped retriever is{" "}
            <code className="font-mono text-[13px] text-ink">dense-ctx-v1</code> — contextual
            embeddings, no reranker (an ablation showed the reranker wasn&rsquo;t worth its cost over
            this embedder).
          </Step>
          <Step n={3} title="Reason — the grammar-ReAct agent">
            In deep mode a single agent runs a constrained search-read-cite loop: it searches the
            corpus, reads passages before citing them, and can abstain or be forced to finalise. The
            plain &ldquo;Ask&rdquo; path skips the loop for a fast single-shot answer.
          </Step>
          <Step n={4} title="Answer — grounded and cited">
            The model writes the answer from the retrieved passages and attaches citations. A
            defence layer screens input and output for prompt-injection, leaked secrets, and
            ungrounded or forged claims before anything reaches you.
          </Step>
        </ol>
      </Section>

      {/* The stack */}
      <Section title="The stack">
        <p className="mb-4 text-[15px] leading-relaxed text-ink-soft">
          Provider-agnostic by design — no model is hardcoded into the logic, so any piece can be
          swapped. The current lineup:
        </p>
        <div className="flex flex-wrap gap-2">
          <StackTag>LlamaIndex · RAG layer</StackTag>
          <StackTag>LangGraph · agent</StackTag>
          <StackTag>FastAPI · async + SSE</StackTag>
          <StackTag>Postgres + pgvector</StackTag>
          <StackTag>qwen3-embedding-8b · 1024d</StackTag>
          <StackTag>deepseek-v4-pro · agent</StackTag>
          <StackTag>Vite + React · this UI</StackTag>
          <StackTag>Langfuse · tracing &amp; cost</StackTag>
        </div>
        <p className="mt-4 text-[13.5px] leading-relaxed text-ink-faint">
          Every one of these was a written decision with criteria, not a default — the framework,
          embedder, vector store, frontend, and model lineup each went through a gated choice backed
          by either an ablation or a documented trade-off.
        </p>
      </Section>

      {/* Limits */}
      <Section title="What it can&rsquo;t do">
        <ul className="space-y-3">
          <Limit>
            <b className="font-semibold text-ink">It only knows its corpus.</b> Ask about something
            outside the ingested books and it will refuse rather than improvise. That refusal is the
            feature — see the out-of-scope behaviour on the{" "}
            <PageLink to="/evals">Evals</PageLink> page.
          </Limit>
          <Limit>
            <b className="font-semibold text-ink">Public-domain translations only.</b> Sources are
            older translations whose author or translator died over 70 years ago (the EU rule).
            They can read differently from modern scholarship.
          </Limit>
          <Limit>
            <b className="font-semibold text-ink">It is not a general chatbot.</b> No live web, no
            opinions of its own, no authority beyond what the books actually say.
          </Limit>
          <Limit>
            <b className="font-semibold text-ink">Models can still err.</b> A model may embellish or
            misattribute within an answer. That is exactly why every claim is cited — so a wrong one
            is checkable, not hidden.
          </Limit>
        </ul>
      </Section>

      {/* How it could go further */}
      <Section title="Making it stronger">
        <p className="mb-4 text-[15px] leading-relaxed text-ink-soft">
          The bottleneck now is on the answer side, not retrieval — the system usually finds the
          right passage, and the residual misses are a model occasionally embellishing or
          misattributing what it read. So the next levers all tighten the gap between the sources
          and the words you see. Each is a candidate, measured on the golden set before it ships.
        </p>
        <ul className="space-y-3">
          <Idea label="Stronger model">
            Swapping the answer step to a more capable frontier model attacks embellishment and
            misattribution at the root. The stack is provider-agnostic, so this is a config change,
            not a rewrite.
          </Idea>
          <Idea label="Verify-and-correct pass">
            A second agent reviews the draft against its own cited passages before you ever see it —
            confirming every claim is actually supported, then rewriting or dropping the ones that
            aren&rsquo;t. Trades a little latency and cost for a stronger trust guarantee.
          </Idea>
          <Idea label="Claim-level grounding check">
            An automated entailment check between each sentence and the span it cites, flagging
            unsupported claims instead of shipping them silently — an extension of the output audit
            that already runs.
          </Idea>
          <Idea label="Self-consistency">
            On hard questions, draft the answer more than once and keep the version best supported
            by the sources, smoothing out one-off model slips.
          </Idea>
        </ul>
      </Section>

      <footer className="mt-10 border-t border-line pt-5 text-[13px] leading-relaxed text-ink-faint">
        The numbers behind these claims are live, not screenshots: retrieval and judge scores on the{" "}
        <PageLink to="/evals">Evals</PageLink> page, the red-team results on{" "}
        <PageLink to="/security">Security</PageLink>, and the full corpus on{" "}
        <PageLink to="/sources">Sources</PageLink>. Eval figures move as the system is tuned —
        treat them as current readings, not fixed marketing.
      </footer>
    </section>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mb-9">
      <h2 className="mb-3 font-serif text-xl font-semibold text-ink">{title}</h2>
      {children}
    </div>
  );
}

function PageLink({ to, children }: { to: string; children: ReactNode }) {
  return (
    <Link
      to={to}
      className="font-medium text-accent underline decoration-line-strong underline-offset-2 transition-colors hover:text-accent-ink hover:decoration-accent"
    >
      {children}
    </Link>
  );
}

/** A capability row with a leading intent pill. */
function Capability({ label, children }: { label: string; children: ReactNode }) {
  return (
    <li className="flex flex-col gap-1.5 rounded-lg border border-line bg-surface p-4 shadow-card sm:flex-row sm:items-start sm:gap-3.5">
      <span className="flex-none pt-0.5">
        <Pill variant="accent">{label}</Pill>
      </span>
      <span className="text-[14.5px] leading-relaxed text-ink-soft">{children}</span>
    </li>
  );
}

/** A numbered pipeline step. */
function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li className="flex gap-4">
      <span
        className="flex h-7 w-7 flex-none items-center justify-center rounded-full bg-accent-soft font-serif text-sm font-semibold text-accent-ink"
        aria-hidden
      >
        {n}
      </span>
      <div className="pt-0.5">
        <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
        <p className="mt-1 text-[14.5px] leading-relaxed text-ink-soft">{children}</p>
      </div>
    </li>
  );
}

function StackTag({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border border-line bg-surface px-3 py-1 font-mono text-[12px] text-ink-soft">
      {children}
    </span>
  );
}

/** A limit row, marked with a restrained refuse-tone dash. */
function Limit({ children }: { children: ReactNode }) {
  return (
    <li className="flex gap-3 text-[14.5px] leading-relaxed text-ink-soft">
      <span className="flex-none pt-1 text-refuse" aria-hidden>
        —
      </span>
      <span>{children}</span>
    </li>
  );
}

/** An improvement-idea row with a leading neutral category pill. */
function Idea({ label, children }: { label: string; children: ReactNode }) {
  return (
    <li className="flex flex-col gap-1 sm:flex-row sm:items-start sm:gap-3.5">
      <span className="flex-none pt-0.5">
        <Pill variant="neutral">{label}</Pill>
      </span>
      <span className="text-[14.5px] leading-relaxed text-ink-soft">{children}</span>
    </li>
  );
}
