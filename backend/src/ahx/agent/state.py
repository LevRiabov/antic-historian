"""Agent working memory — the LangGraph state and its trace entries.

The graph (graph.py) threads ONE object through every node: `AgentState`. Each
node reads it and returns a *partial* update, which LangGraph merges back in.

Two deliberate type choices, both following CLAUDE.md ("pydantic where data
crosses a boundary"):

* `AgentState` is a plain `TypedDict`, not pydantic — it is internal graph
  plumbing, never serialized across an API/LLM edge. LangGraph also needs the
  reducer annotations below, which live naturally on a TypedDict.
* The pieces *inside* it that DO cross the model boundary (`Step`, `AgentResult`,
  `RetrievedChunk`) are pydantic and validated.

Reducers (the one non-obvious LangGraph idea): a field annotated
`Annotated[list[X], operator.add]` ACCUMULATES — when two nodes each return one
item, LangGraph concatenates instead of overwriting (the Redux-reducer pattern,
reducer = list `+`). Un-annotated fields are OVERWRITTEN by each node's return.
So `history`/`collected` grow across loop iterations; `step`/`final` are replaced.
"""

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel

from ahx.agent.actions import Decision
from ahx.retrieval.dense import RetrievedChunk


class Step(BaseModel):
    """One ReAct iteration, recorded for two readers: the scratchpad we feed
    back to the model next turn, and the eval/trace forensics afterward.

    `thought`/`action`/`args` originate from the model (grammar-constrained, so
    always well-formed); `observation` is our rendering of the tool's result.
    Held loosely-typed (`args: dict`) on purpose — this is a trace record, not
    the dispatch path. The typed Action the think node switches on lives in
    actions.py; this just remembers what happened.
    """

    thought: str
    action: str
    args: dict[str, Any]
    observation: str


class AgentResult(BaseModel):
    """The agent's terminal output, set by the `finalize` action. Its presence
    in the state is what tells the router the loop is done. The output adapter
    (later) turns this + `collected` into the SAME Citation/DoneEvent the
    single-shot ask pipeline emits — so API/eval/judge stay unchanged."""

    answer: str
    refused: bool
    cited_chunk_ids: list[int]  # subset of `collected` the model chose to cite


class AgentState(TypedDict):
    """The graph's entire working memory."""

    question: str  # input, set once at invoke; nodes read but never change it
    history: Annotated[list[Step], operator.add]  # ReAct trace, appended each loop
    collected: Annotated[list[RetrievedChunk], operator.add]  # evidence from search/read
    step: int  # loop counter (overwritten each turn); guards max-steps / forced-finalize
    final: AgentResult | None  # set by finalize -> signals termination
    pending: Decision | None  # transient: the think node's latest move, awaiting `act`
    # Token usage SUMMED across every think call (additive reducer) — the agent's
    # total generation cost for the run record, vs single-shot's one call.
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
