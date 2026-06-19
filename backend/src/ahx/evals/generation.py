"""Generation-tier evaluation: the full ask pipeline against the golden set.

Two layers (docs/golden-set.md cost policy):

1. **Mechanical** — free, runs on every change: citation span recall (gold
   spans covered by chunks the model actually CITED, midpoint rule),
   citation precision (used markers that point at a gold-covering chunk),
   refusal accuracy (out-of-scope questions are only measurable here),
   latency and token usage.
2. **Judge** — LLM-scored faithfulness (claims supported by cited sources),
   completeness (vs ideal_answer), and attribution (surfacing disagreement +
   naming sources where the policy requires it), 1-5 rubrics. Optional; runs at
   phase boundaries with a strong judge (weak judges miscalibrate — known footgun).

Run records mirror the retrieval-tier layout: aggregates first, then
per-question results with ideal_answer next to the model's actual answer.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from ahx.config import Settings
from ahx.db import create_async_db_engine
from ahx.evals.golden import (
    CATEGORIES,
    Category,
    GoldenQuestion,
    ResolutionError,
    ResolvedSpan,
    resolve_span,
)
from ahx.evals.runs import AGENT_TAG, tagged_stem
from ahx.generation.citations import Citation
from ahx.generation.pipeline import DoneEvent, SourcesEvent, ask, collect
from ahx.generation.prompt import PROMPT_VERSION
from ahx.llm import ChatMessage, ChatModel, chat_model_from_settings
from ahx.obs import init_langfuse, trace_request, traced_chat
from ahx.retrieval.embedding import EmbeddingClient


class GenQuestionResult(BaseModel):
    question_id: str
    category: Category
    question: str
    ideal_answer: str
    answer: str
    refused: bool  # mechanical: answer == the exact contract sentence
    refused_semantic: bool | None = None  # judge yes/no (judge-v3.1): accepts a
    # paraphrased abstention as a refusal; None until/unless the judge layer runs
    refusal_expected: bool  # True only for out-of-scope questions
    refusal_correct: bool  # uses refused_semantic when judged, else refused
    markers_used: list[int]
    markers_dangling: list[int]
    retrieved_chunk_ids: list[int]
    cited_chunk_ids: list[int]
    citation_span_recall: float | None  # None for out-of-scope (no gold spans)
    citation_precision: float | None  # None when no markers were used
    faithfulness: int | None = None  # 1-5, judge layer
    completeness: int | None = None  # 1-5, judge layer
    attribution: int | None = None  # 1-5, judge layer (judge-v3): surfacing
    # disagreement + naming sources where the policy requires it
    judge_notes: str = ""
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    trace_id: str | None = None  # Langfuse trace for this question (6.1); None when untraced


class GenCategoryAggregate(BaseModel):
    count: int
    refused: int
    refusal_correct: float  # fraction
    citation_span_recall: float | None
    citation_precision: float | None
    faithfulness: float | None
    completeness: float | None
    attribution: float | None = None  # judge-v3; absent on older run records
    mean_latency_ms: int


class GenAggregates(BaseModel):
    questions: int
    refusal_accuracy_oos: float | None  # refused / out-of-scope count
    false_refusal_rate: float  # refused in-scope / in-scope count
    citation_span_recall: float | None  # mean over in-scope questions
    citation_precision: float | None
    faithfulness: float | None
    completeness: float | None
    attribution: float | None = None  # judge-v3; absent on older run records
    mean_latency_ms: int
    mean_completion_tokens: float | None
    by_category: dict[str, GenCategoryAggregate]


class GenerationRun(BaseModel):
    created_at: str
    label: str
    chat_model: str
    embed_model: str
    chunking_version: str
    prompt_version: str
    top_k: int
    engine: str = "single-shot"  # "single-shot" or "agent" (Phase 5); absent on older records
    agent_max_steps: int | None = None  # set only for agent runs
    retriever: str = "dense"  # which retrieval path fed the prompt (Phase 4.2)
    rerank_model: str | None = None  # set only for rerank-* retrievers
    rerank_pool_n: int | None = None
    judge_model: str | None
    # Set only when the attribution rubric used a DIFFERENT model than judge_model
    # (the split judge); None = one judge scored all rubrics. Keeps a split run
    # from being silently compared to a single-judge one (rule #5/#6).
    attribution_judge_model: str | None = None
    judge_rubric: str | None = None  # None on pre-judge-v2 records
    aggregates: GenAggregates
    results: list[GenQuestionResult]


def _cited_covers_span(sources: SourcesEvent, markers: list[int], span: ResolvedSpan) -> bool:
    midpoint = (span.char_start + span.char_end) // 2
    for citation in sources.citations:
        if citation.marker not in markers:
            continue
        if citation.pg_id == span.pg_id and citation.char_start <= midpoint < citation.char_end:
            return True
    return False


def score_generation(
    question: GoldenQuestion,
    spans: list[ResolvedSpan],
    sources: SourcesEvent,
    done: DoneEvent,
    latency_ms: int,
) -> GenQuestionResult:
    used = done.markers.used
    by_marker = {c.marker: c for c in sources.citations}

    citation_span_recall: float | None = None
    if spans:
        covered = sum(1 for s in spans if _cited_covers_span(sources, used, s))
        citation_span_recall = covered / len(spans)

    citation_precision: float | None = None
    if used and spans:
        precise = sum(
            1 for marker in used if any(_cited_covers_span(sources, [marker], s) for s in spans)
        )
        citation_precision = precise / len(used)

    refusal_expected = question.category == "out-of-scope"
    return GenQuestionResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        ideal_answer=question.ideal_answer,
        answer=done.answer,
        refused=done.refused,
        refusal_expected=refusal_expected,
        refusal_correct=done.refused == refusal_expected,
        markers_used=used,
        markers_dangling=done.markers.dangling,
        retrieved_chunk_ids=[c.chunk_id for c in sources.citations],
        cited_chunk_ids=[by_marker[m].chunk_id for m in used],
        citation_span_recall=citation_span_recall,
        citation_precision=citation_precision,
        latency_ms=latency_ms,
        prompt_tokens=done.usage.prompt_tokens if done.usage else None,
        completion_tokens=done.usage.completion_tokens if done.usage else None,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _eff_refused(r: GenQuestionResult) -> bool:
    """Effective refusal: the semantic verdict once judged, else the mechanical flag."""
    return r.refused_semantic if r.refused_semantic is not None else r.refused


def compute_gen_aggregates(results: list[GenQuestionResult]) -> GenAggregates:
    def block(subset: list[GenQuestionResult]) -> GenCategoryAggregate:
        return GenCategoryAggregate(
            count=len(subset),
            refused=sum(1 for r in subset if _eff_refused(r)),
            refusal_correct=sum(1 for r in subset if r.refusal_correct) / len(subset),
            citation_span_recall=_mean(
                [r.citation_span_recall for r in subset if r.citation_span_recall is not None]
            ),
            citation_precision=_mean(
                [r.citation_precision for r in subset if r.citation_precision is not None]
            ),
            faithfulness=_mean([float(r.faithfulness) for r in subset if r.faithfulness]),
            completeness=_mean([float(r.completeness) for r in subset if r.completeness]),
            attribution=_mean([float(r.attribution) for r in subset if r.attribution]),
            mean_latency_ms=round(sum(r.latency_ms for r in subset) / len(subset)),
        )

    by_category = {
        category: block(subset)
        for category in CATEGORIES
        if (subset := [r for r in results if r.category == category])
    }

    oos = [r for r in results if r.refusal_expected]
    in_scope = [r for r in results if not r.refusal_expected]
    overall = block(results)
    return GenAggregates(
        questions=len(results),
        refusal_accuracy_oos=(sum(1 for r in oos if _eff_refused(r)) / len(oos)) if oos else None,
        false_refusal_rate=(sum(1 for r in in_scope if _eff_refused(r)) / len(in_scope))
        if in_scope
        else 0.0,
        citation_span_recall=overall.citation_span_recall,
        citation_precision=overall.citation_precision,
        faithfulness=overall.faithfulness,
        completeness=overall.completeness,
        attribution=overall.attribution,
        mean_latency_ms=overall.mean_latency_ms,
        mean_completion_tokens=_mean(
            [float(r.completion_tokens) for r in results if r.completion_tokens is not None]
        ),
        by_category=by_category,
    )


# --- judge layer ---


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = ""


class RefusalVerdict(BaseModel):
    """Semantic refusal call (judge-v3.2): now carries the judge's reasoning so the
    out-of-scope verdict is auditable — previously a bare yes/no left judge_notes empty
    for every OOS question, hiding miscalls like oos-013 (a false-premise correction the
    yes/no prompt misread as a substantive answer)."""

    refusal: bool
    reason: str = ""


class AttributionVerdict(BaseModel):
    """judge-v3.6: structured attribution — the judge COUNTS errors and code maps the
    counts to a graduated 1-5 (`attribution_score`). Replaces the holistic 1/3/5 rubric
    whose single misattribution -> 1 cliff (a 4-point drop) starved the middle of the
    scale AND drove the documented 1<->5 re-score flips (eval-log 2026-06-15)."""

    absent: int = Field(ge=0)  # required source-namings missing from the prose
    incorrect: int = Field(ge=0)  # claims credited to a source that does not support them
    settled: bool = False  # presents a genuinely CONTESTED point as settled (worst case)
    reason: str = ""


# Rubric history (rule #5 — judge changes are measured, see eval-log):
# judge-v1: judge saw only CITED chunks -> correct-but-miscited answers scored
#   like fabrications, double-counting what citation_precision already measures.
# judge-v2: judge sees ALL retrieved passages exactly as the answer model did;
#   misattribution of grounded content caps at 4, invention is the real failure.
# judge-v3: adds the ATTRIBUTION dimension — faithfulness rewards a grounded answer
#   even if it silently picks one side of a contradiction, so that behavior (and
#   un-attributed multi-source synthesis) needs its own axis. Pairs with the
#   baseline-v2 answer prompt, which instructs the model to surface disagreement.
# judge-v3.1: three calibration fixes after the first judge-v3 run (over-severity on
#   concise correct answers, brittle refusal match):
#   (a) semantic refusal — the mechanical refused flag is an exact match on the contract
#       sentence, so a correct-but-paraphrased abstention (esp. out-of-scope) scored as a
#       non-refusal. A yes/no judge call now accepts paraphrased refusals; refusal_correct
#       and the refusal aggregates use it. The 1-5 dimensions stay None on out-of-scope
#       (no ideal answer; binary refuse/answer only).
#   (b) completeness graded against the QUESTION's scope, not every detail of the rich
#       reference — a concise answer that fully answers what was asked scores 5 (lit-001:
#       "23 wounds" answers "how many wounds", was wrongly dinged to 3 for omitting
#       reference context the question never asked for).
#   (c) attribution scored in explicit agree/disagree steps so the all-agree case (bare
#       markers, nothing misattributed) reliably scores 5 instead of being penalised for
#       lacking prose attribution it doesn't need. Disagreement = incompatible claims (not
#       mere omission/emphasis); misattribution = citing a source that doesn't support the
#       claim (not an extra on-topic corroborating marker).
# judge-v3.2: two refusal-judge fixes (rule #5), isolated by rejudging frozen answers:
#   (a) the semantic refusal call now returns JSON {"refusal", "reason"} instead of a bare
#       yes/no — the reason is stored in judge_notes, so the out-of-scope verdict is
#       auditable (v3.1 left judge_notes empty for all 26 OOS, hiding miscalls).
#   (b) refusal definition widened to credit a FALSE-PREMISE CORRECTION: an answer that
#       denies the question's premise and does not supply the (impossible) requested fact
#       is a refusal, even when it volunteers correct adjacent facts. v3.1's narrow "states
#       the sources lack the info" wording credited oos-014 but not oos-013 (same pattern,
#       different phrasing). Substituting adjacent SECONDARY material for an absent named
#       work is NOT a refusal (the source-absent trap — oos-019/020/023/024/025 stay leaks).
# judge-v3.3: faithfulness gloss fix (rule #5, isolated by rejudging frozen answers). A
#   standard modern equivalent or definitional gloss of a term/event that IS in the sources
#   — "the sacred disease" -> epilepsy (syn-006), a named accession span -> its AD years
#   (lit-003, syn-022), a battle's conventional name/year (syn-013, mh-014) — was scored as
#   "unsupported embellishment" and capped at 3. ~20 of the 39 faithfulness sub-5s on the
#   agent-v4 run were this miscall. v3.3 names glosses as GROUNDED and reserves the 3/1 band
#   for genuinely NEW facts/quotes/outcomes that restate nothing in the sources (a fabricated
#   quote, an invented death) — so real invention (cb-020, con-018, cb-017) still scores low.
#   The 3-anchor's word "embellishment" (which the judge over-applied to glosses) is replaced
#   by that explicit definition. A gloss is grounded only if CORRECT: a date may be calculated
#   from the source's ancient reckoning (AUC / regnal year / Olympiad) plus common knowledge,
#   but a WRONG date/term/name is an error, not a free pass, and scores in the 3/1 band.
#   Faithfulness only; completeness/attribution/refusal unchanged.
# judge-v3.4: refusal-judge STABILITY fix for the source-absent template (rule #5). The
#   disciplined refusal-with-provenance pattern deepseek-pro emits — "the corpus does not contain
#   X; it is only discussed in [secondary authors] [n] ... I cannot report what X itself says" —
#   sent the judge two competing signals (explicit absence + decline vs. a relayed secondary
#   characterization) with NO precedence rule, so it flipped run-to-run: oos-023 (Sappho) scored
#   refusal-correct in the D5/v4/v5 runs but flipped to a leak in agent-v6 on a byte-near-identical
#   answer, while the six sibling answers using the IDENTICAL template (oos-019/020/022/024/025/026)
#   scored refusal-correct every run. oos-023's only difference: it briefly QUOTES Grote's
#   characterization rather than just naming him, tipping the ad-hoc balance. v3.4 adds a
#   deterministic precedence rule — an explicit "work absent + I cannot report its own content" IS
#   a refusal even when it names or briefly quotes clearly-attributed secondary commentary; it
#   becomes an ATTEMPT only when the substitute is presented AS the work's content WITHOUT that
#   decline (the gemma-era leaks — Bury's Syracuse dressed as the Republic's ideal state;
#   oos-024's recited NH book-structure — lack the decline, so they stay leaks). Refusal-judge
#   only; faithfulness/completeness/attribution unchanged. Isolate by `eval rejudge` of frozen
#   answers (expected to flip exactly oos-023 in the v6 set; all other OOS verdicts unchanged).
# judge-v3.5: faithfulness 4-anchor for the peripheral/rhetorical unsourced detail (calibration,
#   rule #5). v3.3's 3-anchor lumped three things at 3: (a) a rhetorical figure of speech or a
#   true-but-immaterial aside not in the retrieved excerpt, (b) an invented CHECKABLE specific
#   presented as sourced, (c) a factual claim contradicting the passages. (a) over-penalized a
#   fully-grounded answer — synth-001's "Antony dies in her arms" is both a rhetorical close AND
#   literally true in Plutarch (the cited author); synth-003 relayed "debt / harsh creditors", a
#   real but peripheral Livy theme absent from the excerpt. Both scored 3, the same band as
#   invention. v3.5 splits the band on ONE auditable test — does the unsourced element MISLEAD
#   about a checkable fact? No (rhetorical/immaterial) -> 4; yes (invented proper noun / number /
#   date / outcome, or a contradiction) -> 3; a fabricated quote or load-bearing invention -> 1.
#   So synth-001/003 -> 4, while synth-012 (invented "Caesarea") and synth-014 (source names two
#   dead, answer says "many of the Thirty") stay 3 and synth-005 (memorized verbatim quotes not in
#   the retrieved chunks) stays 1. Faithfulness anchors only; completeness/attribution/refusal
#   unchanged. Isolate by `eval rejudge` of frozen answers + human spot-check of EVERY moved
#   verdict (rule #6 — faithfulness is the headline trust number; only fix demonstrable
#   miscalibration, never a score that merely felt low).
# judge-v3.6: structured ATTRIBUTION — counts, not a holistic 1/3/5 (rule #5). The v3-v3.5
#   attribution rubric only defined 1/3/5 anchors and sent ANY misattribution straight to 1,
#   so in practice answers scored 5 (clean) or 1 (one wrong author) with nothing between — and
#   that 4-point cliff on a single borderline "is this misattribution?" call WAS the source of
#   the documented 1<->5 re-score flips (eval-log 2026-06-15, the attribution-noise blocker).
#   v3.6 has the judge COUNT two error types — `absent` (a required source-naming missing from
#   prose) and `incorrect` (a claim credited to a non-supporting source) — plus a `settled`
#   flag (a contested point asserted as settled); `attribution_score` maps the counts to 1-5
#   in code (absent=1 demerit; first incorrect=2, each further=1; settled=1). This makes 2 and
#   4 reachable, turns one misattribution into a one-band drop (5->3) instead of a cliff, and —
#   being a count of discrete items rather than a holistic gestalt — cuts the re-score variance.
#   The count breakdown is stored in judge_notes for audit. Attribution rubric only; the split
#   judge still routes it to the stronger attribution model. Isolate by `eval rejudge`.
JUDGE_RUBRIC_VERSION = "judge-v3.6"

FAITHFULNESS_RUBRIC = """You are grading a RAG system's answer for FAITHFULNESS: did the
model invent content, or is everything grounded in the source passages it was shown?

Below are ALL passages the model saw, numbered exactly as shown to it; the ones it
actually cited are flagged "(cited)". Score 1-5:
5 = every claim is supported by SOME passage below, even one it failed to cite;
4 = grounded EXCEPT for one minor blemish that does NOT mislead about a checkable fact —
    EITHER a claim attributed to the wrong author/source or wrong marker, OR a single
    PERIPHERAL element not in any passage that is mere rhetorical framing or a true,
    immaterial aside (a figure of speech; a corroborating remark the answer does not
    rest on);
3 = mostly grounded, but introduces a CHECKABLE specific that appears in NO passage and
    is presented as if sourced — an invented proper noun, number, date, or specific
    outcome — OR a factual claim that CONTRADICTS the passages (a reader could be misled);
1 = substantial invented content found in no passage below — a fabricated direct quote, or
    multiple / load-bearing inventions.
Wrong citation markers alone must NOT take the score below 4 — citation accuracy is
measured separately. Invented content is the failure this score exists to catch.
The 4-vs-3 line is exactly whether a reader is MISLED about a checkable fact: a rhetorical
or immaterial aside is 4; an invented or contradicted checkable specific is 3.

A standard modern equivalent or definitional gloss of a term or event that IS in the
sources counts as GROUNDED, not invented — do NOT lower the score for a CORRECT gloss.
Examples: naming the modern disease for an ancient term ("the sacred disease" ->
epilepsy); the conventional name of a battle the passage describes; converting a date
the source gives in ancient reckoning (years from the foundation of Rome / AUC, a
regnal year, an Olympiad, or a named accession or event) into its calendar year. A date
may be CALCULATED from the source's own reckoning plus common historical knowledge:
accept it when the conversion is correct. But a gloss must be CORRECT to count as
grounded — if it is WRONG (a date that contradicts common knowledge or cannot be
reconciled with the source, the wrong modern term, a misnamed battle) it is an error,
not a gloss: score it in the 3/1 band. Reserve that band for genuinely NEW facts,
quotes, names, or outcomes that appear in NO passage and are not a correct restatement
of source content (a fabricated direct quote, an invented death or result, a wrong date).

Question: {question}

Source passages:
{sources}

Answer to grade:
{answer}

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

COMPLETENESS_RUBRIC = """You are grading a RAG system's answer for COMPLETENESS: does it
cover everything the QUESTION asks for?

Judge against the QUESTION. Use the reference answer only as the gold standard for which
facts the question requires — the reference is a rich 5/5 example and often includes
context BEYOND what was asked. Do NOT penalize the answer for omitting such extra detail:
a concise answer that fully and correctly answers what the question asked scores 5.

Score 1-5:
5 = answers everything the question asks (being more concise than the reference is fine);
3 = answers the core but omits a part the question explicitly asks for;
1 = misses the point of the question.
Extra correct detail must not lower the score.

Question: {question}

Reference answer (gold standard — may exceed the question's scope):
{ideal}

Answer to grade:
{answer}

Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""

ATTRIBUTION_RUBRIC = """You are grading a RAG system's answer for ATTRIBUTION. You will
COUNT attribution errors; the numeric score is computed from your counts in code, so count
carefully and do NOT output a score yourself.

Policy: when the source passages DISAGREE, or the answer draws on SEVERAL different sources,
the answer must make clear IN PROSE which source each version or contribution comes from
(e.g. "Suetonius reports X, but Dio says Y", or "Plutarch describes... while Arrian adds...").
When the sources simply AGREE, naming each one is optional and its absence is NOT a fault.
Bare citation markers like [1][2] are NOT prose attribution: they cannot tell a reader that
two sources DISAGREE or which source a contribution came from.

Step 1 — do the passages relevant to the question AGREE or DISAGREE on the point at issue?
Treat them as DISAGREEING only when they make INCOMPATIBLE claims about the SAME point (X
cannot be true if Y is). One source merely OMITTING a detail another includes, or differing
in emphasis, wording, or which aspects it covers, is NOT a disagreement — that is agreement
plus extra detail, and needs NO disagreement-surfacing.

Step 2 — identify the attributions the answer is REQUIRED to make: each distinct CONFLICTING
version (on a disagreement) and each distinct SOURCE CONTRIBUTION (on a multi-source answer).
An agree / single-source answer usually has NO required attributions.

Step 3 — classify and COUNT (a correct attribution counts as neither):
- absent: a required attribution the answer does NOT make in prose — it surfaces the version
  or uses the contribution but does not name its source (or names it only with a bare marker).
- incorrect: a claim credited to a source that does NOT support it — naming the wrong author
  in prose (e.g. "Tacitus says X" when X comes from a different author's passage, say Gibbon,
  or from an editorial footnote). Check the author NAMED matches a passage that genuinely
  supports the claim. NOT incorrect: a correctly-credited claim that ALSO carries extra
  on-topic corroborating markers from other authors (a citation-precision matter, not
  misattribution); and do NOT invent a disagreement that isn't there.
- settled: TRUE only if the answer presents a genuinely CONTESTED point as SETTLED — asserts
  one side and does not present the conflicting version at all (the worst case; overrides the
  counts).

Below are ALL passages the model saw, numbered as shown to it; the ones it cited are
flagged "(cited)".

Question: {question}

Source passages:
{sources}

Answer to grade:
{answer}

Report ONLY this JSON: {{"absent": <int>, "incorrect": <int>, "settled": <true|false>, \
"reason": "<one sentence naming the specific errors, or noting there are none>"}}"""


REFUSAL_JUDGE = """Decide whether the following answer is a REFUSAL — it declines to give
the requested answer — or an ATTEMPT to answer the question as posed.

Count as a REFUSAL ("yes"):
- It states, in any wording, that the provided sources do not contain the information needed.
- It DENIES THE QUESTION'S PREMISE (the question assumes a false fact) and does not supply the
  impossible requested fact — EVEN IF it volunteers correct related facts or a correction.
  (Illustrative only: a question asking "which fleet did Hannibal command on the Moon" met with
  "the sources record no such event; Hannibal led no lunar campaign" is a refusal.)
- It EXPLICITLY states the named work/text is absent from the sources AND explicitly declines to
  report that work's OWN content — even if it also names or briefly quotes clearly-attributed
  secondary commentary ABOUT the author or work. The labelled secondary mention plus the explicit
  decline is honest provenance, not a substitution. (Illustrative only: "the corpus does not
  contain X; a secondary author notes Y about its author [n] ... but I cannot report what X itself
  says from the sources here" is a refusal.)

Count as an ATTEMPT ("no"):
- It supplies the requested fact/description AS the answer, including by SUBSTITUTING adjacent or
  secondary material for an absent named work and presenting that material AS the work's content —
  answering from other authors who merely discuss the work or its author, WITHOUT an explicit
  statement that the work is absent and that its own content cannot be reported, is an ATTEMPT.

Decisive test for the source-absent case: does the answer explicitly decline to report the absent
work's OWN content, with any secondary material clearly labelled as commentary about the author?
Yes -> REFUSAL. No (it presents the substitute as the requested answer) -> ATTEMPT.

These examples are illustrative of the RULE only; judge the answer below on its own.

Question: {question}

Answer:
{answer}

Reply with ONLY a JSON object: {{"refusal": <true|false>, "reason": "<one sentence>"}}"""


def _parse_yes_no(raw: str) -> bool | None:
    """First clear yes/no token wins; None if the reply commits to neither."""
    text = raw.strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    has_yes, has_no = "yes" in text, "no" in text
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def parse_verdict(raw: str) -> JudgeVerdict | None:
    """Tolerant of code fences / prose around the JSON object."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return JudgeVerdict.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValueError):
        return None


def parse_refusal_verdict(raw: str) -> RefusalVerdict | None:
    """Parse the judge-v3.2 refusal JSON; falls back to a bare yes/no token (older
    judge replies / models that ignore the JSON instruction) with an empty reason."""
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return RefusalVerdict.model_validate(json.loads(raw[start : end + 1]))
        except (json.JSONDecodeError, ValueError):
            pass
    token = _parse_yes_no(raw)
    return None if token is None else RefusalVerdict(refusal=token, reason="")


def parse_attribution_verdict(raw: str) -> AttributionVerdict | None:
    """Parse the judge-v3.6 structured attribution JSON (counts, not a score)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return AttributionVerdict.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValueError):
        return None


def attribution_score(v: AttributionVerdict) -> int:
    """Map attribution error counts to 1-5 (judge-v3.6), deterministically in code so a
    borderline call moves the score by one band, not by the old 4-point 5->1 cliff:

        no errors                          -> 5
        1 absent                           -> 4
        2 absent  OR 1 incorrect           -> 3
        3 absent  OR 2 incorrect           -> 2
        >=4 absent OR >=3 incorrect, or a contested point presented as settled -> 1

    Demerits: each `absent` = 1; the FIRST `incorrect` = 2, each further `incorrect` = 1
    more (a wrong source label is worse than a missing one). `settled` is the worst case.
    """
    if v.settled:
        return 1
    demerits = v.absent + (v.incorrect + 1 if v.incorrect >= 1 else 0)
    return max(1, 5 - demerits)


async def judge_question(
    judge: ChatModel,
    result: GenQuestionResult,
    citations: list[Citation],
    attribution_judge: ChatModel | None = None,
) -> None:
    """Mutates result with the judge-layer fields (judge-v3.1).

    First a yes/no semantic refusal check accepts a paraphrased abstention the
    exact-match `refused` flag missed, and refusal_correct is recomputed on it.
    Then — for answered, in-scope questions only — faithfulness, completeness,
    and attribution (1-5). The judge sees ALL retrieved passages (what the answer
    model saw), with cited ones flagged; see rubric history above.
    """
    refusal_note = ""
    if result.refused:
        result.refused_semantic = True  # the exact contract sentence is unambiguous
        refusal_note = "refusal: exact contract sentence (mechanical match)"
    else:
        reply = await judge.complete(
            [
                ChatMessage(
                    role="user",
                    content=REFUSAL_JUDGE.format(question=result.question, answer=result.answer),
                )
            ]
        )
        verdict = parse_refusal_verdict(reply.text)
        result.refused_semantic = verdict.refusal if verdict is not None else result.refused
        reason = verdict.reason if verdict is not None else "unparseable refusal verdict"
        refusal_note = f"refusal={result.refused_semantic}: {reason}"
    result.refusal_correct = result.refused_semantic == result.refusal_expected

    # 1-5 dimensions: answered, in-scope only (out-of-scope has no ideal answer —
    # it stays binary refuse/answer, scored above; the user's call, to keep the
    # faithfulness aggregate clean). For the refuse/OOS branch the refusal verdict's
    # reason IS the record's only judge note — store it so the verdict is auditable.
    if result.refused_semantic or result.refusal_expected:
        result.judge_notes = refusal_note
        return
    sources_text = "\n\n".join(
        f"[{c.marker}]{' (cited)' if c.marker in result.markers_used else ''} "
        f"{c.author}, {c.work_title}:\n{c.text}"
        for c in citations
    )
    # The three rubrics are independent hosted calls — fire them concurrently.
    # Attribution is routed to a (stronger) judge when configured — the one rubric a
    # flash judge can't score stably (eval-log 2026-06-15); faith/compl stay on `judge`.
    # judge-v3.6: attribution returns structured COUNTS scored deterministically in code
    # (attribution_score), so faith/compl take the {score,reason} path while attribution
    # takes its own — the count breakdown is stored in judge_notes for audit.
    attrib_judge = attribution_judge or judge
    faith_prompt = FAITHFULNESS_RUBRIC.format(
        question=result.question, sources=sources_text or "(none cited)", answer=result.answer
    )
    compl_prompt = COMPLETENESS_RUBRIC.format(
        question=result.question, ideal=result.ideal_answer, answer=result.answer
    )
    attrib_prompt = ATTRIBUTION_RUBRIC.format(
        question=result.question, sources=sources_text or "(none cited)", answer=result.answer
    )
    faith_resp, compl_resp, attrib_resp = await asyncio.gather(
        judge.complete([ChatMessage(role="user", content=faith_prompt)]),
        judge.complete([ChatMessage(role="user", content=compl_prompt)]),
        attrib_judge.complete([ChatMessage(role="user", content=attrib_prompt)]),
    )
    notes: list[str] = []
    for field, response in (("faithfulness", faith_resp), ("completeness", compl_resp)):
        verdict = parse_verdict(response.text)
        if verdict is None:
            notes.append(f"{field}: unparseable judge reply")
            continue
        setattr(result, field, verdict.score)
        notes.append(f"{field}: {verdict.reason}")
    av = parse_attribution_verdict(attrib_resp.text)
    if av is None:
        notes.append("attribution: unparseable judge reply")
    else:
        result.attribution = attribution_score(av)
        breakdown = f"absent={av.absent}, incorrect={av.incorrect}" + (
            ", settled" if av.settled else ""
        )
        notes.append(f"attribution={result.attribution} ({breakdown}): {av.reason}")
    result.judge_notes = " | ".join(notes)


# --- runner ---


# A generation engine: question text -> (sources shown, answer produced). Both the
# single-shot pipeline and the Phase-5 agent satisfy this; the eval loop is engine-
# agnostic (the agent's runner hides every LangGraph type behind this callable).
GenEngine = Callable[[str], Awaitable[tuple[SourcesEvent, DoneEvent]]]


def _error_result(
    question: GoldenQuestion, spans: list[ResolvedSpan], exc: Exception, latency_ms: int
) -> GenQuestionResult:
    """A question that errored mid-run, recorded as a failed refusal (empty answer,
    zero recall) so the run completes and the failure is visible in judge_notes
    rather than aborting everything. Used by the per-question resilience guard."""
    expected = question.category == "out-of-scope"
    return GenQuestionResult(
        question_id=question.id,
        category=question.category,
        question=question.question,
        ideal_answer=question.ideal_answer,
        answer="",
        refused=True,
        refusal_expected=expected,
        refusal_correct=expected,  # an error on an in-scope question counts as a false refusal
        markers_used=[],
        markers_dangling=[],
        retrieved_chunk_ids=[],
        cited_chunk_ids=[],
        citation_span_recall=0.0 if spans else None,
        citation_precision=None,
        latency_ms=latency_ms,
        prompt_tokens=None,
        completion_tokens=None,
        judge_notes=f"RUN ERROR: {type(exc).__name__}: {exc}",
    )


async def run_generation_eval(
    settings: Settings,
    questions: list[GoldenQuestion],
    label: str = "gen-baseline-v1",
    top_k: int = 5,
    judge: ChatModel | None = None,
    on_result: Callable[[GenQuestionResult], None] | None = None,
    retriever_name: str = "dense",
    agent: bool = False,
    max_steps: int = 8,
    concurrency: int = 1,
    attribution_judge: ChatModel | None = None,
) -> GenerationRun:
    from ahx.ingest.chunker import CHUNKING_VERSION
    from ahx.retrieval.factory import build_async_retriever, is_rerank_label

    engine = create_async_db_engine(settings.database_url)
    # A measurement uses ONE model: disable the serving fallback lineup (6.4) so a mid-run
    # 429 storm can't silently fall over to a different provider and confound the numbers
    # (deepseek's own 5x retry still absorbs transient 429s; a true outage becomes a visible
    # error_result, not a different model's answer). Tracing (6.1) is wired in here too —
    # opt-in (None unless AHX_LANGFUSE_* is set); when on, each question gets a Langfuse trace
    # whose id lands in the record (trace_id) so a failure links straight to its trace.
    chat = chat_model_from_settings(settings.model_copy(update={"chat_fallbacks": []}))
    langfuse = init_langfuse(settings)
    if langfuse is not None:
        chat = traced_chat(chat, langfuse)

    # Pick the engine. Agent and single-shot both produce (SourcesEvent, DoneEvent),
    # so everything downstream (scoring, judge, record) is identical.
    run_one: GenEngine
    if agent:
        from ahx.agent.prompts import AGENT_PROMPT_VERSION
        from ahx.agent.runner import make_agent_engine

        run_one = make_agent_engine(settings, engine, chat, retriever_name, max_steps)
        prompt_version = AGENT_PROMPT_VERSION
        engine_label = "agent"
    else:
        retriever = build_async_retriever(
            settings, engine, EmbeddingClient(settings), retriever_name
        )

        async def _single_shot(question_text: str) -> tuple[SourcesEvent, DoneEvent]:
            events = [event async for event in ask(question_text, retriever, chat, top_k)]
            return collect(events)

        run_one = _single_shot
        prompt_version = PROMPT_VERSION
        engine_label = "single-shot"

    # Resolve all gold spans up front. A golden-set bug must abort the whole run
    # (the fail-fast contract: "run `ahx eval validate` first"), not surface
    # mid-flight inside one concurrent task.
    def _resolve(question: GoldenQuestion) -> list[ResolvedSpan]:
        spans: list[ResolvedSpan] = []
        for span in question.gold_spans:
            resolved = resolve_span(span, settings.corpus_normalized_dir, question.id)
            if isinstance(resolved, ResolutionError):
                raise ValueError(
                    f"{question.id}: unresolved gold span ({resolved.problem}) — "
                    "run `ahx eval validate` first"
                )
            spans.append(resolved)
        return spans

    resolved_spans = [_resolve(question) for question in questions]

    # Run (engine + judge) per question, at most `concurrency` in flight. With
    # every stage hosted (agent/embed/judge on OpenRouter) there is no local
    # bottleneck, so concurrency scales the whole run — bounded only by provider
    # rate limits (complete() retries 429s with backoff). gather preserves input
    # order, so `results` stays question-ordered no matter the completion order.
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run_question(
        question: GoldenQuestion, spans: list[ResolvedSpan]
    ) -> GenQuestionResult:
        async with sem:
            started = time.perf_counter()
            # One trace per question (no-op when langfuse is off): the agent's per-step
            # chat calls nest under it, so a failed question's reasoning is inspectable.
            # Concurrent questions are separate asyncio tasks -> separate OTEL contexts,
            # so traces don't tangle.
            async with trace_request(
                langfuse,
                question=question.question,
                top_k=top_k,
                name=f"eval:{question.id}",
                metadata={
                    "question_id": question.id,
                    "category": question.category,
                    "label": label,
                },
            ) as trace:
                try:
                    sources, done = await run_one(question.question)
                    latency_ms = round((time.perf_counter() - started) * 1000)
                    result = score_generation(question, spans, sources, done, latency_ms)
                    if judge is not None:
                        await judge_question(judge, result, sources.citations, attribution_judge)
                    trace.finish(
                        answer=done.answer,
                        refused=done.refused,
                        usage=done.usage,
                        cost=done.cost,
                        served_by=done.served_by,
                    )
                except Exception as exc:
                    # One question's failure (HTTP/DB/parse) must not abort the whole
                    # concurrent run — record it as a failed refusal and carry on. Since
                    # this never re-raises, gather can't cascade-cancel siblings.
                    latency_ms = round((time.perf_counter() - started) * 1000)
                    result = _error_result(question, spans, exc, latency_ms)
            result.trace_id = trace.trace_id  # readable after the span closes
        if on_result is not None:
            on_result(result)  # fires on completion (out of order) — progress only
        return result

    try:
        results = list(
            await asyncio.gather(
                *(
                    _run_question(question, spans)
                    for question, spans in zip(questions, resolved_spans, strict=True)
                )
            )
        )
    finally:
        await engine.dispose()
        if langfuse is not None:
            langfuse.flush()  # drain the trace export buffer before returning

    return GenerationRun(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        label=label,
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        chunking_version=CHUNKING_VERSION,
        prompt_version=prompt_version,
        top_k=top_k,
        engine=engine_label,
        agent_max_steps=max_steps if agent else None,
        retriever=retriever_name,
        rerank_model=settings.rerank_model if is_rerank_label(retriever_name) else None,
        rerank_pool_n=settings.rerank_pool_n if is_rerank_label(retriever_name) else None,
        judge_model=judge.model_name if judge else None,
        attribution_judge_model=attribution_judge.model_name if attribution_judge else None,
        judge_rubric=JUDGE_RUBRIC_VERSION if judge else None,
        aggregates=compute_gen_aggregates(results),
        results=results,
    )


async def rejudge_run(
    settings: Settings,
    record_path: Path,
    judge: ChatModel,
    label: str,
    on_result: Callable[[GenQuestionResult], None] | None = None,
    attribution_judge: ChatModel | None = None,
    concurrency: int = 1,
) -> GenerationRun:
    """Re-score a saved run's FROZEN answers with the current judge/rubric.

    Isolates judge changes from generation nondeterminism (rule #5): answers
    stay byte-identical to the source record; only judge fields move. Chunk
    texts are refetched by id; markers are reconstructed from rank order
    (marker n == retrieved_chunk_ids[n-1], the prompt's numbering invariant).
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from ahx.db import ChunkRow, SourceRow, create_sync_engine

    run = GenerationRun.model_validate_json(record_path.read_text(encoding="utf-8"))

    needed = {cid for r in run.results for cid in r.retrieved_chunk_ids}
    engine = create_sync_engine(settings.database_url)
    with Session(engine) as session:
        rows = session.execute(
            select(ChunkRow, SourceRow.author, SourceRow.title)
            .join(SourceRow, SourceRow.pg_id == ChunkRow.pg_id)
            .where(ChunkRow.id.in_(needed))
        ).all()
    by_id = {
        chunk.id: Citation(
            marker=0,  # placeholder; set per-question from rank order below
            chunk_id=chunk.id,
            pg_id=chunk.pg_id,
            author=author,
            work_title=title,
            locator=chunk.locator,
            text=chunk.text,
            score=0.0,  # not stored in generation records; unused by the judge
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for chunk, author, title in rows
    }
    missing = needed - by_id.keys()
    if missing:
        raise ValueError(f"chunks no longer in DB (chunking changed?): {sorted(missing)[:5]}")

    # Re-score concurrently — frozen answers, all judge calls hosted, so this is
    # purely provider-rate-bound (complete() retries 429s). Order is irrelevant:
    # each task mutates its own result in place; aggregates recompute at the end.
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _rejudge_one(result: GenQuestionResult) -> None:
        async with sem:
            citations = [
                by_id[cid].model_copy(update={"marker": rank})
                for rank, cid in enumerate(result.retrieved_chunk_ids, start=1)
            ]
            result.faithfulness = None
            result.completeness = None
            result.attribution = None
            result.refused_semantic = None
            result.judge_notes = ""
            try:
                await judge_question(judge, result, citations, attribution_judge)
            except Exception as exc:
                # One question's judge failure must not abort the whole pass (it did:
                # a malformed-body crash lost a 161-question pass). Record and carry on;
                # the None scores drop out of aggregates and a variance diff.
                result.judge_notes = f"REJUDGE ERROR: {type(exc).__name__}: {exc}"
        if on_result is not None:
            on_result(result)

    await asyncio.gather(*(_rejudge_one(result) for result in run.results))

    run.created_at = datetime.now(UTC).isoformat(timespec="seconds")
    run.label = label
    run.judge_model = judge.model_name
    run.attribution_judge_model = attribution_judge.model_name if attribution_judge else None
    run.judge_rubric = JUDGE_RUBRIC_VERSION
    run.aggregates = compute_gen_aggregates(run.results)
    return run


def save_generation_run(run: GenerationRun, runs_dir: Path, tag: str = AGENT_TAG) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = run.created_at.replace(":", "-").replace("+00-00", "Z")
    path = runs_dir / f"{tagged_stem(f'{stamp}-{run.label}', tag)}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path
