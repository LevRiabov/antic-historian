"""Production security guard (Phase 6.3) — the deterministic defense stack, in code.

These are the primitives the security LAB (ahx/evals/security.py) ablated, promoted to
a real serving module so the lab measures exactly what the server runs (single source of
truth). The measured hierarchy (eval-log / phase-6-3-security-plan.md): architecture
(grounding) + output validation beat input regex and prompt instructions.

Three deterministic layers, composed in `guarded_events`:
- **D3 input blocklist** — a regex pre-filter; if the question trips it the request is
  refused BEFORE retrieval/model run (cheap, and the response is still a well-formed SSE
  envelope — empty sources + a `blocked` done). Porous by nature (paraphrase evades) — the
  cheap tripwire, not the wall.
- **D2' output validation** — scan the final answer for a leaked prompt: the canary token
  (verbatim or base64/ROT13/hex) OR a verbatim system-prompt span (8-gram overlap, so it
  catches leaks with no canary). Redact if hit. Model/framing-independent.
- **D5 enforce-grounding** — refuse any in-scope answer that cites no source. The strongest,
  unbypassable lever for off-task/injection; OFF by default (it also refuses a legit uncited
  answer — measure its false-refusal cost on the golden set before enabling).

STREAMING CAVEAT (honest): output defenses (D2'/D5) verdict on the FINAL answer at the done
event; on the streaming path the deltas have already shipped, so a determined mid-stream leak
isn't redacted in-flight. D3 (pre-model) + the no-canary-in-prod posture mitigate; a hard guard
would buffer the stream. Documented, not hidden.
"""

import base64
import codecs
import re
from collections.abc import AsyncIterator

from pydantic import BaseModel

from ahx.config import Settings
from ahx.generation.citations import MarkerAudit
from ahx.generation.pipeline import AskEvent, DoneEvent, Retriever, SourcesEvent, StepEvent, ask
from ahx.generation.prompt import PROMPT_VERSION, REFUSAL_TEXT, SYSTEM_PROMPT
from ahx.llm import ChatModel

# Distinct from a normal refusal so a security block is auditable in logs/traces.
SECURITY_REDACTION = "[blocked: response withheld by a security filter]"


class DefenseConfig(BaseModel):
    """Composable defense layers — the lab's ablation axis and the server's live config."""

    harden_prompt: bool = False  # D1 (prompt) — eval-only lever; weakest layer
    input_blocklist: bool = False  # D3 (input regex)
    generalized_output: bool = False  # D2' (canary + encodings + system-prompt n-gram)
    enforce_grounding: bool = False  # D5 (refuse ungrounded in-scope answers)
    output_filter: bool = False  # D2 (bare canary substring) — subset of D2'


# D3 — regex input blocklist. Tightened to limit false positives on real questions
# ("reveal the truth", a passage about a "token") while still catching keyword-laden
# attacks. Porosity is the point: a paraphrasing attacker walks around it (measured).
_INPUT_BLOCK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+|the\s+)*(previous|prior|above|earlier)\s+(instruction|rule|prompt)",
        r"system\s+prompt",
        r"repeat\s+(the\s+)?(text|everything)\s+(above|verbatim)",
        r"your\s+(instructions|rules|system\s+prompt)\b",
        r"\b(internal|session)\s+tokens?\b",
        r"reveal\s+.{0,25}\b(token|prompt|instruction|secret)\b",
        r"\b(base64|rot13)\b",
        r"end\s+of\s+system",
        r"you\s+are\s+now\s+(an?\s+)?(unrestricted|general|dan|jailbroken)",
        r"disregard\s+(your|the|all|previous|prior|these)\b",
        r"no\s+restrictions?\b",
    )
]


def input_blocklist_hit(prompt: str) -> bool:
    """D3: True if the input trips a known attack pattern (→ block before the model)."""
    return any(pattern.search(prompt) for pattern in _INPUT_BLOCK_PATTERNS)


def _ngrams(text: str, n: int) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


_SYSTEM_PROMPT_NGRAMS = _ngrams(SYSTEM_PROMPT, 8)


def filter_canary(answer: str, canary: str) -> str:
    """D2 — bare canary output filter (kept minimal; D2' generalizes it)."""
    return SECURITY_REDACTION if canary in answer else answer


def output_leak_detected(answer: str, canary: str) -> bool:
    """D2' — generalized output validation: the canary verbatim, the canary under common
    encodings (closing D2's blind spot), OR a verbatim system-prompt span (8-gram overlap,
    so it flags a leak even with no canary present)."""
    if canary in answer:
        return True
    encoded = (
        base64.b64encode(canary.encode()).decode(),
        codecs.encode(canary, "rot13"),
        canary.encode().hex(),
    )
    if any(variant in answer for variant in encoded):
        return True
    return bool(_ngrams(answer, 8) & _SYSTEM_PROMPT_NGRAMS)


def is_ungrounded(done: DoneEvent) -> bool:
    """Answered (not the contract refusal) but cited no valid corpus marker."""
    return (not done.refused) and (not done.markers.used)


def refusal_done(message: str, done: DoneEvent | None = None) -> DoneEvent:
    """A defense-forced refusal/block — preserves usage/cost when transforming a real
    answer, and flags `blocked=True` so it's distinguishable from a content refusal."""
    return DoneEvent(
        answer=message,
        refused=True,
        blocked=True,
        markers=MarkerAudit(used=[], dangling=[]),
        usage=done.usage if done else None,
        cost=done.cost if done else None,
        served_by=done.served_by if done else None,
    )


def apply_output_defenses(done: DoneEvent, cfg: DefenseConfig, canary: str) -> DoneEvent:
    """Post-generation defenses (deterministic — re-score a saved run for free): D5
    enforce-grounding refuses ungrounded answers; D2'/D2 redact a leaked prompt."""
    if cfg.enforce_grounding and is_ungrounded(done):
        return refusal_done(REFUSAL_TEXT, done)
    leaked = (cfg.generalized_output and output_leak_detected(done.answer, canary)) or (
        cfg.output_filter and canary in done.answer
    )
    return refusal_done(SECURITY_REDACTION, done) if leaked else done


def guard_config_from_settings(settings: Settings) -> DefenseConfig:
    """The server's live guard, from config. Prompt-hardening is eval-only (marginal);
    output validation defaults on (low false-positive), grounding defaults off (it refuses
    uncited answers — opt in after measuring its cost)."""
    return DefenseConfig(
        input_blocklist=settings.guard_input_blocklist,
        generalized_output=settings.guard_output_validation,
        enforce_grounding=settings.guard_enforce_grounding,
    )


async def guard_stream(
    question: str,
    events: AsyncIterator[AskEvent | StepEvent],
    cfg: DefenseConfig,
    canary: str,
    prompt_version: str,
) -> AsyncIterator[AskEvent | StepEvent]:
    """Wrap ANY ask-event stream in the deterministic guard — single-shot OR the deep
    agent. A D3 input block short-circuits to a well-formed envelope WITHOUT iterating
    `events` (generators are lazy, so retrieval/model never start); otherwise each
    DoneEvent is output-validated and everything else (incl. agent StepEvents) passes
    through. `prompt_version` labels the block envelope's sources."""
    if cfg.input_blocklist and input_blocklist_hit(question):
        yield SourcesEvent(citations=[], prompt_version=prompt_version)
        yield refusal_done(SECURITY_REDACTION)
        return
    async for event in events:
        yield apply_output_defenses(event, cfg, canary) if isinstance(event, DoneEvent) else event


async def guarded_events(
    question: str,
    retriever: Retriever,
    chat: ChatModel,
    cfg: DefenseConfig,
    canary: str,
    top_k: int = 5,
) -> AsyncIterator[AskEvent | StepEvent]:
    """Single-shot served pipeline wrapped in the guard (the fast-path convenience over
    guard_stream). Deep mode wraps make_agent_streamer's output the same way in app.py."""
    async for event in guard_stream(
        question, ask(question, retriever, chat, top_k), cfg, canary, PROMPT_VERSION
    ):
        yield event
