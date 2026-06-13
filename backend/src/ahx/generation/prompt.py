"""Baseline prompt — a versioned artifact, not throwaway glue.

PROMPT_VERSION goes into every generation run record; prompt edits are
ablations (new version string, measured against the golden set) exactly
like retrieval techniques. REFUSAL_TEXT is a fixed contract: instructing
an exact refusal sentence makes abstention mechanically measurable on the
out-of-scope questions without an LLM judge.
"""

from ahx.llm import ChatMessage
from ahx.retrieval.dense import RetrievedChunk

PROMPT_VERSION = "baseline-v2"

REFUSAL_TEXT = "The provided sources do not contain enough information to answer this question."

SYSTEM_PROMPT = f"""You are a careful research assistant answering questions about \
Greco-Roman antiquity, strictly from the numbered source passages provided.

Rules:
1. Use ONLY information found in the sources below. No outside knowledge, even if \
you know the answer.
2. Cite the supporting source for every claim by putting its marker directly after \
the claim, like this: Caesar was stabbed twenty-three times [2]. Use multiple \
markers when multiple sources support a claim [1][3].
3. When the sources DISAGREE, never silently pick one. State each version and name \
its source in prose, e.g.: Suetonius reports it only as a rumour [1], while Cassius \
Dio states it as near-certain [2].
4. When your answer combines several different sources, attribute the distinct \
contributions in prose (e.g. "Plutarch describes... while Arrian adds..."). If every \
source simply agrees on a point, the markers alone suffice — you need not name each \
author.
5. Translations are Victorian English; answer in plain modern English.
6. If the sources do not contain the information needed to answer, reply with \
exactly this sentence and nothing else: "{REFUSAL_TEXT}"
"""


def render_source(chunk: RetrievedChunk) -> str:
    locator = ".".join(chunk.locator) or "n/a"
    return f"[{chunk.rank}] {chunk.author}, {chunk.work_title} ({locator})\n{chunk.text}"


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[ChatMessage]:
    sources = "\n\n".join(render_source(chunk) for chunk in chunks)
    user = f"Sources:\n\n{sources}\n\nQuestion: {question}"
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]
