"""The agent loop — a LangGraph StateGraph wiring think -> act -> think.

The whole ReAct loop is three nodes and one branch:

    START -> think --(route)--> act -> think -> ...   (act loops back to think)
                  '--> END                            (when think finalizes)

* think: render the scratchpad (prompts.py), call the model with the grammar
  (actions.py) so its reply is ALWAYS a valid Decision, parse it, and either
  stage a tool call (`pending`) or set `final` (a Finalize action).
* act: run the staged tool (tools.py), fold the observation + any chunks back
  into the state, and loop back to think.
* route: a conditional edge — if think produced a `final`, go to END; else go
  to act. This is how the model's choice (finalize vs tool) steers the graph;
  no native tool-calling involved (grammar-ReAct — see ADR-001 Phase 5 recheck).

Termination is guaranteed: think force-finalizes once `step >= max_steps`, so
the loop cannot run away regardless of what the model does.

The compiled graph produces a final AgentState; converting that to the streamed
SourcesEvent/DoneEvent the API + eval consume (and renumbering [c<id>] citations)
is the runner's job, deliberately kept out of here.
"""

# graph.py is the LangGraph boundary (ADR-001: framework types live only at the
# edge). LangGraph's builder generics are too partial for pyright strict, so the
# two framework-induced rules are relaxed for THIS file only; the rest of the
# package stays strict.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ahx.agent.actions import Finalize, action_response_format, parse_decision
from ahx.agent.prompts import build_think_messages
from ahx.agent.state import AgentResult, AgentState, Step
from ahx.agent.tools import Toolbox, execute
from ahx.generation.prompt import REFUSAL_TEXT
from ahx.llm import ChatModel

DEFAULT_MAX_STEPS = 8


def initial_state(question: str) -> AgentState:
    """The starting state for a run — empty memory, step 0."""
    return AgentState(
        question=question,
        history=[],
        collected=[],
        step=0,
        final=None,
        pending=None,
        prompt_tokens=0,
        completion_tokens=0,
    )


def _result_from_finalize(action: Finalize) -> AgentResult:
    return AgentResult(
        answer=action.answer, refused=action.refused, cited_chunk_ids=action.cited_chunk_ids
    )


def _forced_refusal() -> AgentResult:
    """Budget exhausted without the model finalizing: refuse honestly. A v1
    choice — a later arm could spend one more call to synthesize from `collected`."""
    return AgentResult(answer=REFUSAL_TEXT, refused=True, cited_chunk_ids=[])


def build_agent_graph(
    chat: ChatModel, toolbox: Toolbox, max_steps: int = DEFAULT_MAX_STEPS
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the agent graph. `chat`/`toolbox` are captured by the nodes
    (explicit injection, no globals — ADR-001)."""
    response_format = action_response_format()

    async def think(state: AgentState) -> dict[str, Any]:
        step = state["step"]
        if step >= max_steps:  # forced-finalize: the hard loop bound
            return {"final": _forced_refusal(), "step": step + 1}
        messages = build_think_messages(state["question"], state["history"])
        result = await chat.complete(messages, response_format=response_format)
        decision = parse_decision(result.text)  # grammar-guaranteed valid
        update: dict[str, Any] = {"pending": decision, "step": step + 1}
        if result.usage is not None:  # additive reducer sums these across calls
            update["prompt_tokens"] = result.usage.prompt_tokens
            update["completion_tokens"] = result.usage.completion_tokens
        if isinstance(decision.action, Finalize):
            update["final"] = _result_from_finalize(decision.action)
            update["history"] = [
                Step(
                    thought=decision.thought,
                    action="finalize",
                    args=decision.action.model_dump(exclude={"tool"}),
                    observation="(final answer)",
                )
            ]
        return update

    async def act(state: AgentState) -> dict[str, Any]:
        decision = state["pending"]
        assert decision is not None  # router only reaches act after a non-final think
        action = decision.action
        result = await execute(action, toolbox)
        step = Step(
            thought=decision.thought,
            action=action.tool,
            args=action.model_dump(exclude={"tool"}),
            observation=result.observation,
        )
        # reducers append: history grows by one completed Step, collected by the
        # new chunks (empty for read/list_sources/find_quote).
        return {"history": [step], "collected": result.chunks}

    def route(state: AgentState) -> str:
        return "end" if state["final"] is not None else "act"

    builder = StateGraph(AgentState)
    builder.add_node("think", think)
    builder.add_node("act", act)
    builder.set_entry_point("think")
    builder.add_conditional_edges("think", route, {"act": "act", "end": END})
    builder.add_edge("act", "think")
    return builder.compile()


async def invoke_agent(
    graph: CompiledStateGraph[AgentState, None, AgentState, AgentState],
    question: str,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentState:
    """Run the compiled graph to completion and return the final state. The
    ainvoke call lives here so every LangGraph touchpoint stays inside the
    boundary file (ADR-001); recursion_limit is a generous backstop above the
    think-node's own forced-finalize bound."""
    config: RunnableConfig = {"recursion_limit": max_steps * 2 + 5}
    return cast(AgentState, await graph.ainvoke(initial_state(question), config))
