"""The agent's action space — the grammar-constrained decision the model emits
each `think` turn.

This is the heart of grammar-ReAct. Instead of relying on a model's native
tool-calling (unreliable on a local 12B), we describe the legal moves as a
pydantic discriminated union, hand its JSON Schema to the chat endpoint as a
`response_format`, and llama.cpp compiles that schema into a GBNF grammar that
constrains token *sampling*. The model therefore CANNOT emit a malformed
decision or an unknown tool — validity is a property of the grammar, not of luck.
(pydantic here ≈ zod, but the schema also drives the decoder.)

`thought` is the first field on purpose: schema property order becomes grammar
order, so the model is forced to write its reasoning before it picks a tool —
the "Reason" half of ReAct, enforced structurally.

The tools mirror the existing corpus functions (see mcp_server.py) and map
1:1 onto the failure modes the agent targets (eval-log.md):
  search       — corpus search; `pg_id` restricts to one source (source-isolation
                 -> cross-book / multi-hop)
  read         — expand context around a hit before citing (read-before-cite
                 -> faithfulness / precision)
  list_sources — what works are actually in the corpus (-> source-absent OOS gap)
  finalize     — emit the answer (or abstain) and end the loop (forced-finalize)

(find_quote was dropped after the D5 audit: it verifies a VERBATIM span, but the
agent paraphrases into modern English and cites by chunk id, so it never touched
the citation path or any measured failure mode — invoked ~1/6 questions with no
effect. Removing it shrinks the decision space the constrained decoder samples.)
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class Search(BaseModel):
    """Retrieve passages. `pg_id` set -> search only within that source."""

    tool: Literal["search"]
    query: str
    pg_id: int | None = None  # source-isolation filter; None = whole corpus
    top_k: int = 5


class Read(BaseModel):
    """Read a chunk's FULL verbatim text by its id (the id shown in search
    results); set `pad` to also include that many chars of surrounding canonical
    context on each side — for an answer that straddles a chunk boundary."""

    tool: Literal["read"]
    chunk_id: int
    pad: int = 0


class ListSources(BaseModel):
    """List the works in the corpus (author/title/pg_id). No arguments —
    lets the model check 'is this work even here?' before answering."""

    tool: Literal["list_sources"]


class Finalize(BaseModel):
    """End the loop with an answer (or an abstention). Citations live in the
    answer prose as [c<id>] tokens (the adapter scores those); `refused=True`
    carries the abstention contract. No id list — an unbounded one let the model
    run away emitting integers until the JSON truncated (the 161-run crash)."""

    tool: Literal["finalize"]
    answer: str
    refused: bool = False


# Discriminated union: pydantic (and the grammar) select the variant by `tool`,
# so each branch carries exactly its own required args — nothing optional-by-
# accident. PEP-604 `|` union + an explicit discriminator field.
ToolCall = Annotated[
    Search | Read | ListSources | Finalize,
    Field(discriminator="tool"),
]


class Decision(BaseModel):
    """One full ReAct turn from the model: reason first, mark which passages seen
    so far are relevant (keep_ids), then act."""

    thought: str
    # agent-v5: the chunk ids (from ANY search so far) that bear on the question.
    # Kept ids stay in FULL in the scratchpad; un-kept ids are compacted to their
    # one-line context_note next turn (still re-readable by id). This is the
    # relevance filter — fewer, focused passages for the agent's own reasoning AND
    # the downstream judge (which scores only the kept-plus-cited set). Property order
    # is grammar order: keep_ids sits AFTER thought (decide relevance having
    # reasoned) and BEFORE action. Empty default = keep nothing new this turn.
    keep_ids: list[int] = Field(default_factory=list[int])
    action: ToolCall


def action_response_format() -> dict[str, Any]:
    """The `response_format` payload for llm.complete — OpenAI `json_schema`
    envelope around Decision's schema. llama.cpp turns this into a GBNF grammar."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "agent_decision", "schema": Decision.model_json_schema()},
    }


def finalize_response_format() -> dict[str, Any]:
    """Like `action_response_format`, but constrains the model to a `Finalize`
    only — no search/read/etc. Used for the forced synthesis turn at the step
    bound (graph.py), where searching is over and the only legal move is to write
    the answer (or refuse) from evidence already collected."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "agent_finalize", "schema": Finalize.model_json_schema()},
    }


def parse_decision(text: str) -> Decision:
    """Validate the model's grammar-constrained output into a typed Decision.
    With the grammar in force this should never raise; it stays as the boundary
    check (and the fallback path for any endpoint that ignores response_format)."""
    return Decision.model_validate_json(text)


def parse_finalize(text: str) -> Finalize:
    """Validate the model's output for a forced synthesis turn into a `Finalize`.
    Mirrors `parse_decision`; a ValidationError here means the synthesis call
    truncated/degenerated, and the caller falls back to an honest refusal."""
    return Finalize.model_validate_json(text)
