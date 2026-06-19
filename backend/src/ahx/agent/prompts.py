"""The agent's policy — its system prompt and the per-turn message builder.

This is where the agent's *behavior* lives. The grammar (actions.py) guarantees
a well-formed move; this prompt teaches the model WHICH move to make and when —
the levers that target our weak categories (eval-log.md):

* source-isolation (search with pg_id) -> cross-book / contradiction / multi-hop
* read-before-cite (read) -> faithfulness / precision
* list_sources before answering about a named work -> source-absent OOS gap
* finalize discipline + the abstention contract -> honest refusal, bounded cost

Citation convention: the agent accumulates evidence across many searches, so
there is no fixed small source list to number. It cites by chunk id as `[c<id>]`
(e.g. [c41123]); the output adapter renumbers these to sequential [1]..[N] and
builds the standard Citation table, so the agent's answer is shape-identical to
single-shot for the eval/judge. REFUSAL_TEXT is the SAME contract as single-shot
(imported, not re-typed) so out-of-scope refusals stay mechanically measurable.

AGENT_PROMPT_VERSION is recorded in every run; prompt edits are ablations.
"""

from ahx.agent.state import Step
from ahx.generation.prompt import REFUSAL_TEXT
from ahx.llm import ChatMessage
from ahx.retrieval.dense import RetrievedChunk

# agent-v5.1: NO prompt-text change from v5 — the bump marks two loop/config fixes
# (graph.py + config.py) so the next run is distinguishable from the agent-v5 full
# run in the eval-log: (1) a malformed grammar reply now re-rolls before becoming a
# forced refusal (killed the zero-retrieval false-refusals, eval-log 2026-06-16);
# (2) decoding temperature promoted to an explicit pinned setting (chat_temperature,
# default 0.0) instead of an invisible default.
# agent-v6: quote-pinned finalize (eval-log 2026-06-16 audit). Load-bearing specifics
# (numbers, names, places, the DIRECTION of a relationship, outcomes) must be grounded in
# the passage's EXACT words — quote the key phrase verbatim with its chunk id, never restate
# it from memory — and a specific with no supporting passage must be OMITTED. Targets the two
# faithfulness-3 mechanisms the audit isolated: (a) source-detail garbling (cb-008 wrote "the
# foot retired among the horses" where Caesar says "to these the horse retired"; syn-013
# twenty->twelve) and (b) parametric supply of a true-but-unsourced fact (cb-028 stated
# Lucullus produced the meal — an outcome present in NO retrieved passage, only the model's
# memory of the famous anecdote). Clean ablation vs agent-v5.1: finalize-writing text only.
# agent-v7: REVERT the v6 quote-pinned finalize bullet — restores the v5.1 finalize text exactly
# (the only finalize-text diff between v5.1 and v6 was that one bullet; git-confirmed vs 0cae597).
# The v6 audit (eval-log 2026-06-17) found the quote-pinning was a wash-to-negative on
# faithfulness AND INDUCED quote fabrication (faith≤3-with-fabricated-quote went v5 3 -> v6 10);
# synth-005 collapsed 5->5->1 by dumping the famous memorized Thucydides Corcyra passage verbatim
# despite cit_recall=1.0. The eval-log pre-registered "a revert to the v5.1 finalize text is a $0
# option if the fabrication ever bites" — this is that revert. Anti-embellishment now rests on the
# system prompt's opening "never outside knowledge, even if you know the answer" (as in v5.1). All
# v5.1/v6 RUNTIME fixes (re-roll on malformed grammar, visible step budget, forced synthesis) live
# in graph.py and are unchanged. Clean ablation vs agent-v6: finalize-writing text only.
# agent-v8: named-vs-described query rule (search rule 1). The mh-007 trace showed the agent
# mis-decomposing an INDIRECTION question — "the land battle fought the same day as the SEA
# battle at Mycale" (=Plataea -> Pausanias) — by collapsing it to "Mycale's land forces",
# guessing "Leotychidas" at step 0, and BAKING that guess into the search query
# ("Mycale Spartan commander Leotychidas") so retrieval could only confirm the wrong guess
# (cit_recall 0.0, never searched for Plataea). The new clause makes the agent first classify
# the target as NAMED vs DESCRIBED-by-relationship; for a described target it must RESOLVE the
# description into a name with a NEUTRAL search (no guessed answer in the query) before
# searching the target fact — a described target is two hops. Scoped to indirection only, so
# direct lookups (literal/synonym) are untouched. Targets mh-007/mh-010-shaped questions;
# n=1-2, so measured on the FULL run with literal/synonym/cross-book watched for regression
# (rule #5 — cherry-picked traces don't predict the population). Clean ablation vs agent-v7:
# search rule 1 text only.
AGENT_PROMPT_VERSION = "agent-v8"

AGENT_SYSTEM_PROMPT = f"""You are a careful research assistant answering questions about \
Greco-Roman antiquity. You may ONLY use information you retrieve from the corpus with the \
tools below — never outside knowledge, even if you know the answer.

You work in a loop. Each turn you write a brief thought, mark which passages seen so far are \
relevant (keep_ids), then choose exactly ONE action:

- search(query, pg_id?, top_k?): find passages by meaning. Returns each hit's FULL text \
labelled with a chunk id like [c41]. Set `pg_id` to restrict the search to ONE work \
(source-isolation).
- read(chunk_id, pad?): re-read one chunk's full text by its id; set `pad` to also see the \
surrounding context — use this when an answer may straddle a chunk boundary, OR to pull back \
the full text of a passage that was compacted to a summary.
- list_sources(): list the works actually in the corpus (author, title, pg_id).
- finalize(answer, refused): end the loop with your answer (cite sources inline as [c<id>]).

keep_ids — your relevance filter: set it each turn to the chunk ids (from ANY search so far) \
that actually bear on the question. Ids you keep stay in FULL in your notes; ids you do NOT \
keep are compacted to a one-line summary on the next turn (you can still `read` one by id if \
you change your mind). Keep an id the moment a passage looks useful, and ALWAYS keep every \
passage you intend to cite in your final answer. This keeps your notes — and your final \
evidence — focused on what matters; do not keep passages that turned out irrelevant.

Searching well is the core skill — most failures are weak queries, not missing sources:
1. NEVER search with the question's wording. Convert it into a focused query that reads like \
the ANSWER passage: KEEP the proper nouns (Alcibiades, Thucydides), ADD the words you expect \
the passage to contain (ambition, tyranny, statesman), and DROP question words (how, what, \
sources, judge, did). One idea per query. Example: for "How do the sources judge Alcibiades?" \
search `Alcibiades character ambition tyranny statesman`, not the question itself. Do not \
write a fake answer paragraph — a short focused phrase retrieves best. \
First decide whether the question NAMES the person/place/thing you need or DESCRIBES it by a \
relationship ("the land battle fought the same day as Mycale", "the emperor who was the last \
of his dynasty", "the regent who led at Plataea"). If it NAMES the target, search for it \
directly. If it DESCRIBES the target, your FIRST search must RESOLVE that description into a \
name — search neutrally for the relationship itself and do NOT put a guessed answer into the \
query (that only retrieves confirmation of the guess) — THEN search for the fact about the \
name you found. A described target is TWO hops, not one.
2. Translations are Victorian English: if a modern word finds nothing, retry with period \
phrasing (e.g. "disgrace" -> "reproach"/"infamy").

Work in a loop, but answer as soon as you can:
3. Plan briefly: the fact(s) the answer needs, and — for "what do the sources say / compare \
them" questions — the main sources worth checking. This is a guide, not a checklist you must \
exhaust.
4. Search for what you need with a focused query (rule 1). To get one work's view, restrict the \
search to it with `pg_id` (source-isolation). For a BROAD question that spans several works \
("what do the sources say about X", "compare the accounts"), raise `top_k` (e.g. 10-15) so ONE \
search pulls in several works at once — that is faster and more complete than many narrow \
searches. Search returns each passage in FULL — read the whole text before concluding it lacks \
the answer (the key detail is often mid-passage); use `read` with `pad` only when an answer \
straddles a chunk boundary.
5. If a search comes up short, ADAPT before giving up: reword the query, try period synonyms, \
restrict by `pg_id`. A few distinct angles is enough to tell "present" from "absent".
6. ANSWER as soon as you have relevant evidence that addresses the question. A single-fact \
question is fully answered by ONE good source; a "compare the sources / who disagrees / across \
the authors" question needs two or three — but NEVER every source. Looking for more sources only \
ENRICHES the answer; it is NEVER a precondition for giving one. If you have a solid answer but \
cannot find more, ANSWER with what you have (you may note which source it rests on) — do NOT \
refuse because coverage feels incomplete. Refuse ONLY when your searches turn up nothing that \
addresses the question (or in the absent-named-work case of rule 7). Once the question is \
answered, stop searching.
7. A passage that MENTIONS, DESCRIBES, or FOOTNOTES a work is NOT that work's own text. If \
the question asks what a specific work or author SAYS, answer only from passages that ARE \
from that work; use list_sources / search to check. If the work's own text is not in the \
corpus (only secondary mentions of it), refuse — see "Refusing" below; never answer the \
question from a passage that merely talks about the work.

Writing the final answer (in finalize):
- Use ONLY the passages you retrieved. Cite the supporting source for every claim with its \
chunk id in brackets, e.g.: Caesar was stabbed twenty-three times [c41]. Use several when \
several support a claim [c41][c88]. Cite ONLY chunk ids that actually appeared in your search \
results above — never invent or guess an id.
- When sources DISAGREE, never silently pick one: state each version and name its source \
(e.g. "Suetonius reports it as rumour [c41], while Cassius Dio states it as near-certain [c88]").
- When you combine several sources, attribute the distinct contributions in prose.
- Translations are Victorian English; answer in plain modern English.
- Refusing — when the retrieved passages do not let you answer, refuse (set refused=true) \
rather than padding with loosely-related material:
  - If the corpus simply lacks the information, make `answer` exactly: "{REFUSAL_TEXT}"
  - If the question names a work or author whose OWN text is not in the corpus but is only \
mentioned by another source, refuse AND say so, naming where it is mentioned — e.g.: "The \
corpus does not contain Sappho's poetry; it is only mentioned in [c10]. I cannot report what \
it says from the sources here." Do NOT answer the question from that mention.
"""


def _render_hit(chunk: RetrievedChunk, full: bool) -> str:
    """One search hit. `full` -> the verbatim passage (latest search, or a kept id);
    otherwise the compact one-line context_note (fall back to a text snippet for the
    ~11 unenriched chunks), tagged so the model knows it can `read` the id for more."""
    loc = " > ".join(chunk.locator) if chunk.locator else ""
    head = f"[c{chunk.chunk_id}] {chunk.author}, {chunk.work_title}" + (f" > {loc}" if loc else "")
    if full:
        return f"{head}\n  {' '.join(chunk.text.split())}"
    note = (chunk.context_note or " ".join(chunk.text.split())[:200]).strip()
    return f"{head}\n  (compacted — read {chunk.chunk_id} for full text) {note}"


def _render_scratchpad(
    history: list[Step], by_id: dict[int, RetrievedChunk], kept: set[int]
) -> str:
    """Re-render the whole trace for this turn. Search hits are shown FULL when the
    chunk is kept OR belongs to the most recent search (which the model still has to
    triage); every other hit collapses to its context_note. Non-search steps show
    their stored observation verbatim."""
    latest_search = max((i for i, s in enumerate(history) if s.chunk_ids is not None), default=None)
    parts: list[str] = []
    for i, s in enumerate(history):
        if s.chunk_ids is not None:
            full = i == latest_search
            hits = [by_id[cid] for cid in s.chunk_ids if cid in by_id]
            obs = (
                "\n".join(_render_hit(c, full or c.chunk_id in kept) for c in hits)
                if hits
                else "No passages found."
            )
        else:
            obs = s.observation
        args = ", ".join(f"{k}={v!r}" for k, v in s.args.items())
        parts.append(f"Thought: {s.thought}\nAction: {s.action}({args})\nObservation: {obs}")
    return "\n\n".join(parts)


def _by_id(collected: list[RetrievedChunk]) -> dict[int, RetrievedChunk]:
    by_id: dict[int, RetrievedChunk] = {}
    for chunk in collected:  # first occurrence wins (a chunk can resurface)
        by_id.setdefault(chunk.chunk_id, chunk)
    return by_id


def build_think_messages(
    question: str,
    history: list[Step],
    collected: list[RetrievedChunk],
    kept: list[int],
    step: int,
    max_steps: int,
) -> list[ChatMessage]:
    """Messages for one think turn: system policy + question + a visible step budget
    + the scratchpad. The scratchpad carries the latest search and every KEPT passage
    in full and compacts the rest to a one-line note (agent-v5), so context stays
    bounded across turns instead of growing one full search per step.

    `step`/`max_steps` are surfaced so the model can manage its own budget: the
    eval-log's biggest in-scope false-refusal mechanism was the agent over-searching
    past the step bound and being force-ended (con-012). A model that cannot see its
    budget cannot spend it well — so we tell it, and warn hard near the limit."""
    if history:
        scratchpad = _render_scratchpad(history, _by_id(collected), set(kept))
    else:
        scratchpad = "(nothing yet — start by planning and searching)"
    searches_left = max_steps - step
    budget = f"This is step {step + 1} of {max_steps} (search steps remaining: {searches_left})."
    if searches_left <= 2:
        budget += (
            " You are near your limit: if you already have evidence that addresses the question, "
            "FINALIZE NOW — you will get no more searches. After your last step the system writes "
            "the answer from what you have already collected, so an unfinished search is wasted."
        )
    user = (
        f"Question: {question}\n\n{budget}\n\nYour work so far:\n{scratchpad}\n\n"
        "Decide your next action."
    )
    return [
        ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]


def build_synthesis_messages(
    question: str, history: list[Step], collected: list[RetrievedChunk], kept: list[int]
) -> list[ChatMessage]:
    """The forced FINAL turn, used when the step budget is spent (graph.py). No
    more searching — the model writes its best answer from evidence already
    gathered, or refuses honestly. This replaces a blind auto-refusal that used to
    throw away a complete answer the agent had already found (eval-log: con-012,
    'refused with the answer in hand'). Same compacted scratchpad as the think
    turn (kept + latest in full); the model can no longer `read`, so it must answer
    from what is shown."""
    scratchpad = (
        _render_scratchpad(history, _by_id(collected), set(kept))
        if history
        else "(no searches were made)"
    )
    user = (
        f"Question: {question}\n\nYour work so far:\n{scratchpad}\n\n"
        "You have run out of search steps and may no longer search, read, or look anything up. "
        "Using ONLY the passages already shown above, write your best final answer now and cite "
        "them as [c<id>]. If — and only if — those passages genuinely do not address the "
        f'question, refuse: set refused=true and make the answer exactly "{REFUSAL_TEXT}".'
    )
    return [
        ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]
