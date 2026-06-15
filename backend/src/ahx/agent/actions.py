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

The five tools mirror the existing corpus functions (see mcp_server.py) and map
1:1 onto the failure modes the agent targets (eval-log.md):
  search       — corpus search; `pg_id` restricts to one source (source-isolation
                 -> cross-book / multi-hop)
  read         — expand context around a hit before citing (read-before-cite
                 -> faithfulness / precision)
  list_sources — what works are actually in the corpus (-> source-absent OOS gap)
  find_quote   — verify a span is verbatim before citing (-> attribution)
  finalize     — emit the answer (or abstain) and end the loop (forced-finalize)
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


class FindQuote(BaseModel):
    """Locate an exact quote within a source — verify before citing."""

    tool: Literal["find_quote"]
    pg_id: int
    quote: str


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
    Search | Read | ListSources | FindQuote | Finalize,
    Field(discriminator="tool"),
]


class Decision(BaseModel):
    """One full ReAct turn from the model: reason first, then act."""

    thought: str
    action: ToolCall


def action_response_format() -> dict[str, Any]:
    """The `response_format` payload for llm.complete — OpenAI `json_schema`
    envelope around Decision's schema. llama.cpp turns this into a GBNF grammar."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "agent_decision", "schema": Decision.model_json_schema()},
    }


def parse_decision(text: str) -> Decision:
    """Validate the model's grammar-constrained output into a typed Decision.
    With the grammar in force this should never raise; it stays as the boundary
    check (and the fallback path for any endpoint that ignores response_format)."""
    return Decision.model_validate_json(text)
