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

Termination is guaranteed: once `step >= max_steps` the think node runs ONE
finalize-only synthesis call (answer from collected evidence, or refuse) and
ends, so the loop cannot run away regardless of what the model does.

The compiled graph produces a final AgentState; converting that to the streamed
SourcesEvent/DoneEvent the API + eval consume (and renumbering [c<id>] citations)
is the runner's job, deliberately kept out of here.
"""

# graph.py is the LangGraph boundary (ADR-001: framework types live only at the
# edge). LangGraph's builder generics are too partial for pyright strict, so the
# two framework-induced rules are relaxed for THIS file only; the rest of the
# package stays strict.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from ahx.agent.actions import (
    Finalize,
    Search,
    action_response_format,
    finalize_response_format,
    parse_decision,
    parse_finalize,
)
from ahx.agent.prompts import build_synthesis_messages, build_think_messages
from ahx.agent.state import AgentResult, AgentState, Step
from ahx.agent.tools import Toolbox, execute
from ahx.generation.prompt import REFUSAL_TEXT
from ahx.llm import ChatMessage, ChatModel, Usage

DEFAULT_MAX_STEPS = 8

# The compiled-graph type, named once so the runner adapter can annotate the graph it
# drives without re-spelling LangGraph's 4-arg generic. The framework type still lives
# only on this boundary side (graph.py + the agent adapter), never in eval/API code.
type AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]

# A grammar reply that won't parse is a TRANSIENT generation failure, not proof the
# question is unanswerable: on hosted providers `response_format` is best-effort (not
# the strict GBNF llama.cpp enforces), so a single call can return malformed/truncated
# JSON. Before agent-v5.1 that instantly became a forced refusal (eval-log 2026-06-16:
# the zero-retrieval false-refusals), so we re-roll a few times first — provider
# sampling differs run to run even at temperature 0 — and only refuse if every attempt
# fails. complete() already retries 429/5xx underneath; this retries the PARSE.
_PARSE_ATTEMPTS = 3


async def _complete_parsed[Parsed](
    chat: ChatModel,
    messages: Sequence[ChatMessage],
    response_format: dict[str, Any],
    parse: Callable[[str], Parsed],
    timeout_s: float | None = None,
) -> tuple[Parsed | None, Usage]:
    """Call the model and parse its grammar output, re-rolling up to _PARSE_ATTEMPTS
    times when the reply won't parse. Returns (parsed | None, usage summed across
    every attempt — we paid for each). None means even the retries stayed malformed.

    `timeout_s` bounds each individual model call (not the retry loop): a stalled call
    raises TimeoutError, which surfaces as the terminal error frame instead of hanging
    until the overall request budget. None = untimed (the eval path)."""
    prompt_tokens = 0
    completion_tokens = 0
    parsed: Parsed | None = None
    for _ in range(_PARSE_ATTEMPTS):
        if timeout_s is None:
            result = await chat.complete(messages, response_format=response_format)
        else:
            async with asyncio.timeout(timeout_s):
                result = await chat.complete(messages, response_format=response_format)
        if result.usage is not None:
            prompt_tokens += result.usage.prompt_tokens
            completion_tokens += result.usage.completion_tokens
        try:
            parsed = parse(result.text)
            break
        except ValidationError:
            parsed = None
    return parsed, Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def initial_state(question: str) -> AgentState:
    """The starting state for a run — empty memory, step 0."""
    return AgentState(
        question=question,
        history=[],
        collected=[],
        step=0,
        final=None,
        pending=None,
        kept=[],
        prompt_tokens=0,
        completion_tokens=0,
    )


def _result_from_finalize(action: Finalize) -> AgentResult:
    return AgentResult(answer=action.answer, refused=action.refused)


def _forced_refusal() -> AgentResult:
    """End with an honest refusal — the fallback when a generation won't parse (a
    degenerate/truncated output), including the forced synthesis turn. The step
    bound itself no longer auto-refuses: it runs one synthesis call first (`think`
    below) so an answer already in `collected` is written, not discarded."""
    return AgentResult(answer=REFUSAL_TEXT, refused=True)


def build_agent_graph(
    chat: ChatModel,
    toolbox: Toolbox,
    max_steps: int = DEFAULT_MAX_STEPS,
    step_timeout_seconds: float | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Compile the agent graph. `chat`/`toolbox` are captured by the nodes
    (explicit injection, no globals — ADR-001). `step_timeout_seconds` bounds each
    model call (live deep path); None leaves calls untimed (the eval path)."""
    response_format = action_response_format()

    async def synthesize(state: AgentState, step: int) -> dict[str, Any]:
        """Budget spent: one final grammar-constrained call that may ONLY finalize,
        so a complete answer already in `collected` gets written instead of being
        thrown away by a blind refusal (eval-log con-012). If even this won't parse,
        fall back to the honest refusal."""
        messages = build_synthesis_messages(
            state["question"], state["history"], state["collected"], state["kept"]
        )
        action, usage = await _complete_parsed(
            chat, messages, finalize_response_format(), parse_finalize, step_timeout_seconds
        )
        update: dict[str, Any] = {
            "step": step + 1,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
        if action is None:  # degenerate even after retries -> honest refusal
            update["final"] = _forced_refusal()
            return update
        update["final"] = _result_from_finalize(action)
        update["history"] = [
            Step(
                thought="(out of search steps — synthesizing from collected evidence)",
                action="finalize",
                args=action.model_dump(exclude={"tool"}),
                observation="(forced final answer at the step bound)",
            )
        ]
        return update

    async def think(state: AgentState) -> dict[str, Any]:
        step = state["step"]
        if step >= max_steps:  # budget spent -> forced synthesis (not a blind refusal)
            return await synthesize(state, step)
        messages = build_think_messages(
            state["question"], state["history"], state["collected"], state["kept"], step, max_steps
        )
        decision, usage = await _complete_parsed(
            chat, messages, response_format, parse_decision, step_timeout_seconds
        )
        if decision is None:
            # Malformed/truncated grammar on every retry (a runaway generation that
            # hit the context limit, or a hosted provider ignoring the schema). Only
            # NOW give up — an honest refusal rather than crashing the whole
            # (concurrent, expensive) run. Counts as a false refusal in the eval.
            return {
                "final": _forced_refusal(),
                "step": step + 1,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        # Fold this turn's relevance picks into the accumulated kept set (additive
        # reducer) — applies whether the action is a search or the finalize.
        update: dict[str, Any] = {
            "pending": decision,
            "step": step + 1,
            "kept": decision.keep_ids,
            # additive reducer sums these across calls (incl. this turn's retries)
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
        }
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
            # Record the returned ids for a search so the scratchpad can re-render
            # this turn's hits (full vs compacted) next turn; None for read /
            # list_sources, whose observation string is shown verbatim (agent-v5).
            chunk_ids=[c.chunk_id for c in result.chunks] if isinstance(action, Search) else None,
        )
        # reducers append: history grows by one completed Step, collected by the
        # new chunks (empty for read/list_sources).
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


async def astream_agent(
    graph: AgentGraph,
    question: str,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AsyncIterator[Step | AgentState]:
    """The streaming twin of invoke_agent (6.7 deep mode): yields each completed
    ReAct `Step` as it lands, then the final `AgentState` last. Consumes
    `graph.astream(stream_mode="values")` — each snapshot is the full merged state
    after a node — and emits whatever history entries are newly appended, so the
    LangGraph boundary stays inside this file (ADR-001). invoke_agent (the eval
    path) is untouched; this is purely additive."""
    config: RunnableConfig = {"recursion_limit": max_steps * 2 + 5}
    emitted = 0
    final_state: AgentState | None = None
    async for snapshot in graph.astream(initial_state(question), config, stream_mode="values"):
        state = cast(AgentState, snapshot)
        final_state = state
        history = state["history"]
        while emitted < len(history):
            yield history[emitted]
            emitted += 1
    assert final_state is not None  # astream always yields at least the final state
    yield final_state
