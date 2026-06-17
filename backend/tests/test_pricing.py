"""Cost-tracking tests (Phase 6.2). Pure arithmetic over a hand-built price table
(not the committed snapshot, so the test is deterministic). The three policies that
matter: hosted+priced -> computed USD; local (bare id) -> $0; hosted+unpriced -> None,
never a silent $0 (rule #6)."""

from ahx.llm import Usage
from ahx.pricing import Cost, ModelPrice, PriceTable, cost_for

TABLE = PriceTable(
    fetched_at="2026-06-16",
    source="test",
    models={"deepseek/deepseek-v4-pro": ModelPrice(input_per_m=0.435, output_per_m=0.870)},
)


def test_priced_hosted_model_computes_usd() -> None:
    # 1,000,000 input @ $0.435/M + 1,000,000 output @ $0.870/M = $1.305
    cost = cost_for(
        "deepseek/deepseek-v4-pro",
        Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000),
        TABLE,
    )
    assert cost == Cost(
        usd=1.305,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        model="deepseek/deepseek-v4-pro",
        priced=True,
    )


def test_realistic_small_request() -> None:
    cost = cost_for(
        "deepseek/deepseek-v4-pro", Usage(prompt_tokens=2473, completion_tokens=169), TABLE
    )
    assert cost.priced is True
    # 2473/1e6*0.435 + 169/1e6*0.870 = 0.001076 + 0.000147 = 0.001223
    assert cost.usd == round(2473 / 1e6 * 0.435 + 169 / 1e6 * 0.870, 6)


def test_local_model_is_free() -> None:
    # Bare id (no "/") = llama-swap local = $0, and still "priced" (a known $0).
    cost = cost_for("gemma-12b-16k", Usage(prompt_tokens=500, completion_tokens=50), TABLE)
    assert cost.usd == 0.0
    assert cost.priced is True


def test_unknown_hosted_model_is_unpriced_not_zero() -> None:
    cost = cost_for("acme/unknown-7b", Usage(prompt_tokens=10, completion_tokens=10), TABLE)
    assert cost.usd is None
    assert cost.priced is False
    assert cost.input_tokens == 10  # tokens still recorded for forensics


def test_missing_usage_is_zero_tokens() -> None:
    cost = cost_for("deepseek/deepseek-v4-pro", None, TABLE)
    assert cost.usd == 0.0
    assert (cost.input_tokens, cost.output_tokens) == (0, 0)


def test_no_table_falls_back_to_local_else_unpriced() -> None:
    assert cost_for("gemma-12b-16k", Usage(prompt_tokens=1, completion_tokens=1), None).usd == 0.0
    assert cost_for("acme/x", Usage(prompt_tokens=1, completion_tokens=1), None).priced is False
