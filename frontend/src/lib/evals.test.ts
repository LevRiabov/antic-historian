import { describe, expect, it } from "vitest";

import {
  answerTier,
  buildCategoryAggRows,
  buildStatTiles,
  categoryCounts,
  countOutcomes,
  fmtMrr,
  fmtPct,
  fmtScore,
  isFailure,
  mergeGolden,
  mrrTone,
  parseJudgeNotes,
  recallTone,
  scoreTone,
  selectRows,
  worksByPgId,
  type EvalView,
  type GoldenRow,
} from "@/lib/evals";
import type {
  Category,
  GenAggregates,
  GenerationRun,
  GenQuestionResult,
  RetrievalAggregates,
  RetrievalQuestionResult,
  RetrievalRun,
  SourceOut,
} from "@/lib/types";

/* ---- builders ---- */

function goldenRow(over: Partial<GoldenRow> = {}): GoldenRow {
  return {
    id: "q1",
    category: "literal" as Category,
    question: "Did Caesar cross the Rubicon?",
    ideal: "Yes.",
    answer: "Yes, he did. [1]",
    recall5: 100,
    recall20: 100,
    mrr: 1,
    faithfulness: 5,
    completeness: 5,
    attribution: null,
    refusalExpected: false,
    refusalCorrect: false,
    goldChunkIds: [10],
    goldPgIds: [1],
    citedChunkIds: [10],
    retrievedChunkIds: [10],
    judgeNotes: "",
    ...over,
  };
}

function ragResult(over: Partial<RetrievalQuestionResult> = {}): RetrievalQuestionResult {
  return {
    question_id: "q1",
    category: "literal",
    question: "Q?",
    ideal_answer: "A.",
    gold_spans: [{ pg_id: 1, char_start: 0, char_end: 5, groups: [] }],
    gold_chunk_ids: [10],
    retrieved_chunk_ids: [10, 11],
    recall: { "5": 1, "20": 1 },
    first_hit_rank: 1,
    mrr: 1,
    latency_ms: 50,
    ...over,
  };
}

function genResult(over: Partial<GenQuestionResult> = {}): GenQuestionResult {
  return {
    question_id: "q1",
    category: "literal",
    question: "Q?",
    ideal_answer: "A.",
    answer: "An answer [1]",
    refused: false,
    refusal_expected: false,
    refusal_correct: false,
    markers_used: [1],
    retrieved_chunk_ids: [10, 11],
    cited_chunk_ids: [10],
    faithfulness: 5,
    completeness: 4,
    attribution: null,
    judge_notes: "",
    latency_ms: 60,
    ...over,
  };
}

function ragRun(results: RetrievalQuestionResult[], agg?: RetrievalAggregates): RetrievalRun {
  return {
    created_at: "2026-06-19",
    retriever: "dense-ctx-v1",
    embed_model: "qwen3-embedding-8b",
    top_k: 20,
    aggregates: agg ?? { recall: { "5": 1, "20": 1 }, mrr: 1, by_category: {} },
    results,
  };
}

function genRun(results: GenQuestionResult[], agg?: GenAggregates): GenerationRun {
  return {
    created_at: "2026-06-19",
    label: "agent-v8",
    chat_model: "deepseek-v4-pro",
    embed_model: "qwen3-embedding-8b",
    prompt_version: "agent-v8",
    top_k: 20,
    engine: "agent",
    judge_model: "kimi-k2.6",
    judge_rubric: "v3.6",
    aggregates:
      agg ??
      {
        questions: results.length,
        refusal_accuracy_oos: null,
        false_refusal_rate: 0,
        faithfulness: 5,
        completeness: 4,
        attribution: null,
        mean_latency_ms: 60,
        by_category: {},
      },
    results,
  };
}

const view = (over: Partial<EvalView> = {}): EvalView => ({
  filter: "all",
  query: "",
  failuresOnly: false,
  ...over,
});

/* ---- tone + formatting ---- */

describe("tone thresholds", () => {
  it("recallTone (percentage)", () => {
    expect(recallTone(85)).toBe("good");
    expect(recallTone(84.9)).toBe("amber");
    expect(recallTone(60)).toBe("amber");
    expect(recallTone(59.9)).toBe("refuse");
  });
  it("scoreTone (1-5)", () => {
    expect(scoreTone(4.5)).toBe("good");
    expect(scoreTone(3.5)).toBe("amber");
    expect(scoreTone(3.4)).toBe("refuse");
  });
  it("mrrTone (0-1)", () => {
    expect(mrrTone(0.7)).toBe("good");
    expect(mrrTone(0.45)).toBe("amber");
    expect(mrrTone(0.44)).toBe("refuse");
  });
});

describe("formatters", () => {
  it("fmtPct rounds a fraction to a whole percentage", () => {
    expect(fmtPct(0.5)).toBe("50%");
    expect(fmtPct(0.876)).toBe("88%");
  });
  it("fmtScore and fmtMrr keep their decimals", () => {
    expect(fmtScore(4.2)).toBe("4.2");
    expect(fmtMrr(0.666)).toBe("0.67");
  });
});

/* ---- judge-note parsing ---- */

describe("parseJudgeNotes", () => {
  it("splits the labeled per-dimension notes and capitalizes the label", () => {
    const notes = parseJudgeNotes(
      "faithfulness: every claim holds | completeness: missed one date | attribution: n/a",
    );
    expect(notes).toEqual([
      { label: "Faithfulness", text: "every claim holds" },
      { label: "Completeness", text: "missed one date" },
      { label: "Attribution", text: "n/a" },
    ]);
  });
  it("returns an unlabeled note for non-dimension lines (run errors, refusal reasons)", () => {
    expect(parseJudgeNotes("RUN ERROR: timeout")).toEqual([{ label: "", text: "RUN ERROR: timeout" }]);
    expect(parseJudgeNotes("refusal=true: not in corpus")).toEqual([
      { label: "", text: "refusal=true: not in corpus" },
    ]);
  });
  it("is empty for blank notes", () => {
    expect(parseJudgeNotes("")).toEqual([]);
    expect(parseJudgeNotes("   ")).toEqual([]);
  });
});

/* ---- answer tiers ---- */

describe("answerTier", () => {
  it("out-of-scope is good iff it refused correctly", () => {
    expect(answerTier(goldenRow({ category: "out-of-scope", refusalCorrect: true }))).toBe("good");
    expect(answerTier(goldenRow({ category: "out-of-scope", refusalCorrect: false }))).toBe("failed");
  });
  it("an expected-but-empty answer fails", () => {
    expect(answerTier(goldenRow({ answer: "   " }))).toBe("failed");
  });
  it("an answered-but-unjudged row counts as good (passing)", () => {
    expect(
      answerTier(goldenRow({ faithfulness: null, completeness: null, attribution: null })),
    ).toBe("good");
  });
  it("is governed by the weakest judged dimension", () => {
    expect(answerTier(goldenRow({ faithfulness: 5, completeness: 4, attribution: 4 }))).toBe("good");
    expect(answerTier(goldenRow({ faithfulness: 5, completeness: 3, attribution: 4 }))).toBe("poor");
    expect(answerTier(goldenRow({ faithfulness: 2, completeness: 5, attribution: 5 }))).toBe("failed");
  });
  it("ignores null dimensions (attribution N/A never drags a row down)", () => {
    expect(answerTier(goldenRow({ faithfulness: 5, completeness: 5, attribution: null }))).toBe(
      "good",
    );
  });
});

describe("isFailure", () => {
  it("is true only for the failed tier", () => {
    expect(isFailure(goldenRow({ faithfulness: 2 }))).toBe(true);
    expect(isFailure(goldenRow({ faithfulness: 3 }))).toBe(false); // poor, but adequate
    expect(isFailure(goldenRow({ category: "out-of-scope", refusalCorrect: true }))).toBe(false);
  });
});

/* ---- merge + select ---- */

describe("mergeGolden", () => {
  it("joins by question_id with the generation run driving the row set", () => {
    const rag = ragRun([ragResult({ question_id: "q1", recall: { "5": 0.5, "20": 1 }, mrr: 0.5 })]);
    const gen = genRun([
      genResult({ question_id: "q1", answer: "ans", faithfulness: 4 }),
      genResult({ question_id: "oos", category: "out-of-scope", refusal_expected: true }),
    ]);
    const rows = mergeGolden(rag, gen);
    expect(rows.map((r) => r.id)).toEqual(["q1", "oos"]);

    const q1 = rows[0]!;
    expect(q1.recall5).toBe(50); // 0.5 * 100
    expect(q1.recall20).toBe(100);
    expect(q1.mrr).toBe(0.5);
    expect(q1.goldPgIds).toEqual([1]);

    const oos = rows[1]!;
    expect(oos.recall5).toBeNull(); // retrieval run skips out-of-scope
    expect(oos.mrr).toBeNull();
    expect(oos.goldChunkIds).toEqual([]);
    expect(oos.goldPgIds).toEqual([]);
  });

  it("dedupes gold pg_ids across spans", () => {
    const rag = ragRun([
      ragResult({
        gold_spans: [
          { pg_id: 1, char_start: 0, char_end: 1, groups: [] },
          { pg_id: 1, char_start: 5, char_end: 6, groups: [] },
          { pg_id: 2, char_start: 0, char_end: 1, groups: [] },
        ],
      }),
    ]);
    const rows = mergeGolden(rag, genRun([genResult()]));
    expect(rows[0]!.goldPgIds).toEqual([1, 2]);
  });
});

describe("selectRows", () => {
  const rows = [
    goldenRow({ id: "lit-1", category: "literal", question: "Rubicon?", faithfulness: 5 }),
    goldenRow({ id: "syn-1", category: "synonym", question: "Cannae?", faithfulness: 2 }),
    goldenRow({
      id: "oos-1",
      category: "out-of-scope",
      question: "What did Caesar think of the printing press?",
      refusalCorrect: true,
    }),
  ];
  it("filters by category", () => {
    expect(selectRows(rows, view({ filter: "synonym" })).map((r) => r.id)).toEqual(["syn-1"]);
  });
  it("filters by id or question text", () => {
    expect(selectRows(rows, view({ query: "rubicon" })).map((r) => r.id)).toEqual(["lit-1"]);
    expect(selectRows(rows, view({ query: "OOS" })).map((r) => r.id)).toEqual(["oos-1"]);
  });
  it("failuresOnly keeps just the failing rows", () => {
    expect(selectRows(rows, view({ failuresOnly: true })).map((r) => r.id)).toEqual(["syn-1"]);
  });
});

describe("categoryCounts", () => {
  it("counts per category and total, zero-filling absent categories", () => {
    const counts = categoryCounts([
      goldenRow({ category: "literal" }),
      goldenRow({ category: "literal" }),
      goldenRow({ category: "multi-hop" }),
    ]);
    expect(counts.all).toBe(3);
    expect(counts.literal).toBe(2);
    expect(counts["multi-hop"]).toBe(1);
    expect(counts.synthesis).toBe(0);
  });
});

describe("countOutcomes", () => {
  it("partitions rows into good / poor / failed and sums adequate", () => {
    const out = countOutcomes([
      goldenRow({ faithfulness: 5, completeness: 5, attribution: null }), // good
      goldenRow({ faithfulness: 3, completeness: 5, attribution: null }), // poor
      goldenRow({ faithfulness: 2, completeness: 5, attribution: null }), // failed
    ]);
    expect(out).toEqual({ total: 3, good: 1, poor: 1, adequate: 2, failed: 1 });
  });
});

/* ---- aggregate tiles + category table ---- */

describe("buildStatTiles", () => {
  it("derives the headline tiles from the two aggregate sets + the rows", () => {
    const rag: RetrievalAggregates = { recall: { "5": 0.9, "20": 1 }, mrr: 0.8, by_category: {} };
    const gen: GenAggregates = {
      questions: 2,
      refusal_accuracy_oos: 1,
      false_refusal_rate: 0,
      faithfulness: 4.6,
      completeness: 4.2,
      attribution: null,
      mean_latency_ms: 60,
      by_category: {},
    };
    const rows = [
      goldenRow({ faithfulness: 5, completeness: 5, attribution: null }), // good
      goldenRow({ faithfulness: 2, completeness: 5, attribution: null }), // failed
    ];
    const byLabel = Object.fromEntries(buildStatTiles(rag, gen, rows).map((t) => [t.label, t]));

    expect(byLabel.Questions!.value).toBe("2");
    expect(byLabel["Good answers"]!.value).toBe("50%"); // 1 of 2
    expect(byLabel["Adequate answers"]!.value).toBe("50%");
    expect(byLabel["Recall@5"]!.value).toBe("90%");
    expect(byLabel.MRR!.value).toBe("0.80");
    expect(byLabel.Faithfulness!.value).toBe("4.6/5");
    expect(byLabel.Attribution!.value).toBe("—"); // null -> dash
    expect(byLabel["Refusal accuracy"]!.value).toBe("100%");
  });
});

describe("buildCategoryAggRows", () => {
  it("emits a row per category plus an overall row, with out-of-scope refusal accuracy", () => {
    const rag: RetrievalAggregates = {
      recall: { "5": 0.9, "20": 1 },
      mrr: 0.8,
      by_category: { literal: { count: 1, recall: { "5": 1, "20": 1 }, mrr: 1 } },
    };
    const gen: GenAggregates = {
      questions: 2,
      refusal_accuracy_oos: 1,
      false_refusal_rate: 0,
      faithfulness: 4.6,
      completeness: 4.2,
      attribution: null,
      mean_latency_ms: 60,
      by_category: {
        literal: {
          count: 1,
          refused: 0,
          refusal_correct: 0,
          faithfulness: 5,
          completeness: 4,
          attribution: null,
          mean_latency_ms: 60,
        },
        "out-of-scope": {
          count: 1,
          refused: 1,
          refusal_correct: 1,
          faithfulness: null,
          completeness: null,
          attribution: null,
          mean_latency_ms: 40,
        },
      },
    };
    const rows = buildCategoryAggRows(rag, gen);
    const byCat = Object.fromEntries(rows.map((r) => [r.category, r]));

    expect(byCat.literal!.recall5).toBe(100);
    expect(byCat.literal!.faithfulness).toBe(5);
    expect(byCat["out-of-scope"]!.refusalAccuracy).toBe(1);
    expect(byCat["out-of-scope"]!.recall5).toBeNull();
    expect(byCat.overall!.recall5).toBe(90);
    expect(byCat.overall!.refusalAccuracy).toBeNull();
    expect(rows.at(-1)!.category).toBe("overall");
  });
});

describe("worksByPgId", () => {
  it("maps pg_id to 'Author, Title'", () => {
    const sources = [
      { pg_id: 1, author: "Caesar", title: "Gallic War" },
      { pg_id: 2, author: "Mommsen", title: "History of Rome" },
    ] as SourceOut[];
    const map = worksByPgId(sources);
    expect(map.get(1)).toBe("Caesar, Gallic War");
    expect(map.get(2)).toBe("Mommsen, History of Rome");
    expect(map.get(99)).toBeUndefined();
  });
});
