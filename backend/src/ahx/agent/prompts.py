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

AGENT_PROMPT_VERSION = "agent-v4"

AGENT_SYSTEM_PROMPT = f"""You are a careful research assistant answering questions about \
Greco-Roman antiquity. You may ONLY use information you retrieve from the corpus with the \
tools below — never outside knowledge, even if you know the answer.

You work in a loop. Each turn you write a brief thought, then choose exactly ONE action:

- search(query, pg_id?, top_k?): find passages by meaning. Returns each hit's FULL text \
labelled with a chunk id like [c41]. Set `pg_id` to restrict the search to ONE work \
(source-isolation).
- read(chunk_id, pad?): re-read one chunk's full text by its id; set `pad` to also see the \
surrounding context — use this when an answer may straddle a chunk boundary.
- list_sources(): list the works actually in the corpus (author, title, pg_id).
- finalize(answer, refused): end the loop with your answer (cite sources inline as [c<id>]).

Searching well is the core skill — most failures are weak queries, not missing sources:
1. NEVER search with the question's wording. Convert it into a focused query that reads like \
the ANSWER passage: KEEP the proper nouns (Alcibiades, Thucydides), ADD the words you expect \
the passage to contain (ambition, tyranny, statesman), and DROP question words (how, what, \
sources, judge, did). One idea per query. Example: for "How do the sources judge Alcibiades?" \
search `Alcibiades character ambition tyranny statesman`, not the question itself. Do not \
write a fake answer paragraph — a short focused phrase retrieves best.
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


def _render_step(step: Step) -> str:
    """One past turn, as the model will re-read it next turn, in full."""
    args = ", ".join(f"{k}={v!r}" for k, v in step.args.items())
    return (
        f"Thought: {step.thought}\nAction: {step.action}({args})\nObservation: {step.observation}"
    )


def build_think_messages(
    question: str, history: list[Step], step: int, max_steps: int
) -> list[ChatMessage]:
    """Messages for one think turn: system policy + question + a visible step
    budget + the scratchpad of everything done so far (full — search returns whole
    chunks, so the agent runs on a large-context model; the eval's per-question
    guard catches the rare overflow rather than crashing the run).

    `step`/`max_steps` are surfaced so the model can manage its own budget: the
    eval-log's biggest in-scope false-refusal mechanism was the agent over-searching
    past the step bound and being force-ended (con-012). A model that cannot see its
    budget cannot spend it well — so we tell it, and warn hard near the limit."""
    if history:
        scratchpad = "\n\n".join(_render_step(s) for s in history)
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


def build_synthesis_messages(question: str, history: list[Step]) -> list[ChatMessage]:
    """The forced FINAL turn, used when the step budget is spent (graph.py). No
    more searching — the model writes its best answer from evidence already
    gathered, or refuses honestly. This replaces a blind auto-refusal that used to
    throw away a complete answer the agent had already found (eval-log: con-012,
    'refused with the answer in hand')."""
    scratchpad = "\n\n".join(_render_step(s) for s in history) or "(no searches were made)"
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
