"""The agent's policy — its system prompt and the per-turn message builder.

This is where the agent's *behavior* lives. The grammar (actions.py) guarantees
a well-formed move; this prompt teaches the model WHICH move to make and when —
the levers that target our weak categories (eval-log.md):

* source-isolation (search with pg_id) -> cross-book / contradiction / multi-hop
* read-before-cite (read) -> faithfulness / precision
* list_sources before answering about a named work -> source-absent OOS gap
* find_quote -> attribution
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

AGENT_PROMPT_VERSION = "agent-v1"

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
- find_quote(pg_id, quote): check that an exact quote occurs in a work before citing it.
- finalize(answer, cited_chunk_ids, refused): end the loop with your answer.

Strategy:
1. Break a multi-part or multi-hop question into separate searches — one fact at a time.
2. To compare sources or resolve a disagreement, search each work in isolation with `pg_id`, \
then report what each one says.
3. If the question names a specific author or work, use list_sources / search to confirm \
that work is actually in the corpus. If the corpus only MENTIONS the work but does not \
contain its text, do not substitute another source — refuse.
4. Search returns each passage in FULL — read the whole text before concluding it lacks \
the answer (the key detail is often mid-passage). Use `read` with `pad` only when an answer \
may straddle a chunk boundary.
5. Stop as soon as you have enough evidence — do not over-search. You have a limited number \
of steps; if you are running low, finalize with what you have, or refuse.

Writing the final answer (in finalize):
- Use ONLY the passages you retrieved. Cite the supporting source for every claim with its \
chunk id in brackets, e.g.: Caesar was stabbed twenty-three times [c41]. Use several when \
several support a claim [c41][c88]. List those same ids in `cited_chunk_ids`.
- When sources DISAGREE, never silently pick one: state each version and name its source \
(e.g. "Suetonius reports it as rumour [c41], while Cassius Dio states it as near-certain [c88]").
- When you combine several sources, attribute the distinct contributions in prose.
- Translations are Victorian English; answer in plain modern English.
- If the retrieved sources do not contain enough information to answer, set refused=true and \
make `answer` exactly this sentence and nothing else: "{REFUSAL_TEXT}"
"""


def _render_step(step: Step) -> str:
    """One past turn, as the model will re-read it next turn."""
    args = ", ".join(f"{k}={v!r}" for k, v in step.args.items())
    return (
        f"Thought: {step.thought}\nAction: {step.action}({args})\nObservation: {step.observation}"
    )


def build_think_messages(question: str, history: list[Step]) -> list[ChatMessage]:
    """Messages for one think turn: system policy + question + the scratchpad of
    everything done so far. The model replies with the next grammar-constrained
    Decision (so we do not prompt for a 'Thought:' prefix — the grammar adds it)."""
    if history:
        scratchpad = "\n\n".join(_render_step(s) for s in history)
    else:
        scratchpad = "(nothing yet — start by planning and searching)"
    user = f"Question: {question}\n\nYour work so far:\n{scratchpad}\n\nDecide your next action."
    return [
        ChatMessage(role="system", content=AGENT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]
