"""Pipeline entrypoints: `uv run ahx <command>`.

Offline work (ingest, evals) runs through this CLI, not the API server —
ingest is a local batch job, eval runs cost money and happen at phase
boundaries (see docs/python-stack.md §2).
"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Antic Historian pipeline commands.", no_args_is_help=True)
ingest_app = typer.Typer(help="Corpus ingestion: download -> normalize -> chunk -> load.")
app.add_typer(ingest_app, name="ingest")
db_app = typer.Typer(help="Database management.")
app.add_typer(db_app, name="db")

console = Console()


@db_app.command(name="init")
def db_init() -> None:
    """Create the pgvector extension and all tables (idempotent)."""
    from ahx.config import get_settings
    from ahx.db import create_sync_engine, init_db

    engine = create_sync_engine(get_settings().database_url)
    init_db(engine)
    console.print("[green]Database initialized (extension + tables + indexes).[/green]")


@db_app.command(name="reset-chunks")
def db_reset_chunks(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
) -> None:
    """Drop + recreate the chunks table (D2 ablation: re-embed with a new
    model/dim). Follow with `ahx ingest load` and `ahx ingest parity --update`."""
    from ahx.config import get_settings
    from ahx.db import create_sync_engine, reset_chunks

    settings = get_settings()
    if not yes:
        console.print(
            "[red]This drops ALL embedded chunks (reload costs GPU/API time).[/red]\n"
            f"Target: {settings.database_url}\n"
            f"New embed config: {settings.embed_model} @ {settings.embed_dim}d\n"
            "Re-run with --yes to proceed."
        )
        raise typer.Exit(code=1)
    reset_chunks(create_sync_engine(settings.database_url))
    console.print(
        f"[green]Chunks table recreated for {settings.embed_model} @ "
        f"{settings.embed_dim}d. Now run `ahx ingest load`.[/green]"
    )


@ingest_app.command()
def download(force: bool = typer.Option(False, help="Re-download even if cached.")) -> None:
    """Fetch every manifest entry into corpus/raw/ (idempotent)."""
    from ahx.config import get_settings
    from ahx.ingest.download import download_all
    from ahx.ingest.manifest import parse_manifest

    settings = get_settings()
    entries = parse_manifest(settings.manifest_path)
    results = asyncio.run(download_all(entries, settings.corpus_raw_dir, force=force))

    failed = 0
    for entry, status, detail in results:
        mark = {"downloaded": "+", "cached": "=", "failed": "!"}[status]
        line = f"  {mark} pg{entry.pg_id:<6} {status:<10} {entry.title[:60]}"
        if status == "failed":
            failed += 1
            console.print(f"[red]{line}  {detail}[/red]")
        else:
            console.print(line)
    console.print(f"\n{len(results) - failed}/{len(results)} present in corpus/raw/")
    if failed:
        raise typer.Exit(code=1)


@ingest_app.command()
def normalize() -> None:
    """Clean + structure-parse raw texts into corpus/normalized/ and print the QA report."""
    from ahx.config import get_settings
    from ahx.ingest.manifest import parse_manifest
    from ahx.ingest.pipeline import normalize_work

    settings = get_settings()
    entries = parse_manifest(settings.manifest_path)

    table = Table(title="Normalization QA report")
    for column in ("pg_id", "title", "parser", "divs", "paras", "chars", "flags"):
        table.add_column(column)

    errors = 0
    for entry in entries:
        report = normalize_work(entry, settings.corpus_raw_dir, settings.corpus_normalized_dir)
        if report.error:
            errors += 1
        flags = "; ".join(report.flags)
        table.add_row(
            str(report.pg_id),
            report.title[:40],
            report.parser,
            str(report.divisions),
            str(report.paragraphs),
            f"{report.chars:,}",
            f"[yellow]{flags}[/yellow]" if flags else "[green]ok[/green]",
        )
    console.print(table)
    if errors:
        raise typer.Exit(code=1)


@ingest_app.command()
def chunk() -> None:
    """Pack normalized works into ~500-token chunks (corpus/chunks/*.jsonl) + stats."""
    from ahx.config import get_settings
    from ahx.ingest.manifest import parse_manifest
    from ahx.ingest.pipeline import chunk_one

    settings = get_settings()
    entries = parse_manifest(settings.manifest_path)
    chunks_dir = settings.corpus_dir / "chunks"

    table = Table(title="Chunking report (structural-v1, 500/50)")
    for column in ("pg_id", "title", "chunks", "mean tok", "max tok", "oversize"):
        table.add_column(column)

    errors = 0
    total_chunks = 0
    for entry in entries:
        report = chunk_one(entry, settings.corpus_normalized_dir, chunks_dir)
        if report.error:
            errors += 1
        total_chunks += report.chunks
        table.add_row(
            str(report.pg_id),
            report.title[:40],
            str(report.chunks),
            str(report.mean_tokens),
            str(report.max_tokens),
            f"[yellow]{report.oversize}[/yellow]" if report.oversize else "0",
        )
    console.print(table)
    console.print(f"\nTotal: {total_chunks:,} chunks")
    if errors:
        raise typer.Exit(code=1)


@ingest_app.command()
def load() -> None:
    """Embed all chunks (local model) and load them into Postgres."""
    import time

    from ahx.config import get_settings
    from ahx.db import create_sync_engine
    from ahx.ingest.manifest import parse_manifest
    from ahx.ingest.pipeline import load_one
    from ahx.retrieval.embedding import EmbeddingClient

    settings = get_settings()
    entries = parse_manifest(settings.manifest_path)
    engine = create_sync_engine(settings.database_url)
    embedder = EmbeddingClient(settings)
    chunks_dir = settings.corpus_dir / "chunks"

    errors = 0
    total = 0
    started = time.perf_counter()
    for entry in entries:
        t0 = time.perf_counter()
        report = load_one(entry, chunks_dir, engine, embedder)
        seconds = time.perf_counter() - t0
        total += report.chunks
        if report.status == "error":
            errors += 1
            console.print(f"[red]! pg{report.pg_id} {report.detail}[/red]")
        else:
            console.print(
                f"  {report.status:<7} pg{report.pg_id:<6} {report.chunks:>5} chunks "
                f"{seconds:>6.1f}s  {report.title[:50]}"
            )
    console.print(
        f"\n{total:,} chunks in DB across {len(entries)} works "
        f"({time.perf_counter() - started:.0f}s)"
    )
    if errors:
        raise typer.Exit(code=1)


@ingest_app.command()
def parity(
    update: bool = typer.Option(False, help="Regenerate the reference fixture."),
) -> None:
    """Embedding parity check (rule #3): live vectors vs committed fixture,
    cosine >= 0.999. Run after ANY embedding runtime/model change."""
    import json

    from ahx.config import get_settings
    from ahx.retrieval.embedding import EmbeddingClient, cosine

    fixture_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "parity.json"
    sentences = [
        "Caesar crossed the Rubicon with the thirteenth legion.",
        "The Athenians sent a fleet to Sicily under Nicias.",
        "He was stabbed twenty-three times in the senate-house.",
    ]
    queries = ["How did Caesar die?", "Why did the Sicilian expedition fail?"]

    settings = get_settings()
    embedder = EmbeddingClient(settings)
    live: dict[str, list[list[float]]] = {
        "documents": embedder.embed_documents(sentences),
        "queries": [embedder.embed_query_sync(q) for q in queries],
    }

    if update:
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(
            json.dumps({"model": settings.embed_model, **live}), encoding="utf-8"
        )
        console.print(f"[green]Fixture written: {fixture_path}[/green]")
        return

    if not fixture_path.exists():
        console.print("[red]No fixture. Run `ahx ingest parity --update` once.[/red]")
        raise typer.Exit(code=1)
    reference = json.loads(fixture_path.read_text(encoding="utf-8"))
    worst = 1.0
    for kind in ("documents", "queries"):
        for ref, cur in zip(reference[kind], live[kind], strict=True):
            worst = min(worst, cosine(ref, cur))
    threshold = 0.999
    if worst < threshold:
        console.print(f"[red]PARITY FAIL: worst cosine {worst:.6f} < {threshold}[/red]")
        console.print("Embedding runtime/model drifted — re-embed corpus or revert.")
        raise typer.Exit(code=1)
    console.print(
        f"[green]Parity OK: worst cosine {worst:.6f} (model {reference['model']})[/green]"
    )


@app.command()
def search(query: str, top_k: int = 5) -> None:
    """Debug tool: dense similarity search against the loaded corpus."""
    from ahx.config import get_settings
    from ahx.db import create_sync_engine
    from ahx.retrieval.dense import dense_retrieve
    from ahx.retrieval.embedding import EmbeddingClient

    settings = get_settings()
    engine = create_sync_engine(settings.database_url)
    hits = dense_retrieve(engine, EmbeddingClient(settings), query, top_k)
    for hit in hits:
        locator = ".".join(hit.locator)
        preview = hit.text[:160].replace("\n", " ")
        console.print(
            f"[bold]{hit.score:.3f}[/bold]  {hit.author}, {hit.work_title[:40]}  "
            f"[cyan]{locator}[/cyan]"
        )
        console.print(f"        {preview}...\n")


eval_app = typer.Typer(help="Golden set + evaluation harness.")
app.add_typer(eval_app, name="eval")


@eval_app.command()
def validate() -> None:
    """Validate the golden set: schema, unique ids, quote resolution, counts."""
    from ahx.config import get_settings
    from ahx.evals.golden import (
        CATEGORIES,
        TARGET_V20_PER_CATEGORY,
        ResolutionError,
        load_golden_set,
        resolve_span,
    )

    settings = get_settings()
    golden_dir = Path(__file__).resolve().parents[2] / "evals" / "golden"
    questions = load_golden_set(golden_dir)

    errors: list[ResolutionError] = []
    resolved_count = 0
    for question in questions:
        for span in question.gold_spans:
            result = resolve_span(span, settings.corpus_normalized_dir, question.id)
            if isinstance(result, ResolutionError):
                errors.append(result)
            else:
                resolved_count += 1

    table = Table(title=f"Golden set: {len(questions)} questions")
    for column in ("category", "total", "reviewed", "v2.0 target"):
        table.add_column(column)
    for category in CATEGORIES:
        in_category = [q for q in questions if q.category == category]
        reviewed = sum(1 for q in in_category if q.status == "reviewed")
        met = "[green]met[/green]" if len(in_category) >= TARGET_V20_PER_CATEGORY else ""
        table.add_row(
            category,
            str(len(in_category)),
            str(reviewed),
            f"{len(in_category)}/{TARGET_V20_PER_CATEGORY} {met}",
        )
    console.print(table)
    console.print(f"Gold spans resolved: {resolved_count}, failed: {len(errors)}")

    for error in errors:
        detail = f" ({error.occurrences} matches)" if error.problem == "ambiguous" else ""
        console.print(
            f"[red]  {error.question_id} pg{error.pg_id} {error.problem}{detail}: "
            f"{error.quote_preview!r}[/red]"
        )
    if errors:
        console.print(
            "\n[yellow]Fix: make quotes exact substrings of the canonical text "
            "(use the MCP find_quote tool), or lengthen ambiguous ones.[/yellow]"
        )
        raise typer.Exit(code=1)


@eval_app.command(name="run")
def eval_run(
    retriever: str = typer.Option("dense-v1", help="Retriever variant label for the run record."),
    top_k: int = typer.Option(20, help="Retrieval depth."),
) -> None:
    """Run retrieval-tier eval (recall@k, MRR) and save a versioned run record."""
    from ahx.config import get_settings
    from ahx.evals.golden import CATEGORIES, load_golden_set
    from ahx.evals.retrieval import K_VALUES, run_retrieval_eval, save_run

    settings = get_settings()
    golden_dir = Path(__file__).resolve().parents[2] / "evals" / "golden"
    questions = load_golden_set(golden_dir)

    run = run_retrieval_eval(settings, questions, retriever_name=retriever, top_k=top_k)

    table = Table(
        title=f"Retrieval eval — {run.retriever} · {run.embed_model} · {run.chunking_version}"
    )
    table.add_column("category")
    table.add_column("n")
    for k in K_VALUES:
        table.add_column(f"recall@{k}")
    table.add_column("MRR")
    for category in CATEGORIES:
        agg = run.aggregates.by_category.get(category)
        if agg is None:
            continue
        table.add_row(
            category,
            str(agg.count),
            *(f"{agg.recall[k]:.1%}" for k in K_VALUES),
            f"{agg.mrr:.3f}",
        )
    table.add_row(
        "[bold]overall[/bold]",
        str(len(run.results)),
        *(f"[bold]{run.aggregates.recall[k]:.1%}[/bold]" for k in K_VALUES),
        f"[bold]{run.aggregates.mrr:.3f}[/bold]",
    )
    console.print(table)

    runs_dir = Path(__file__).resolve().parents[2] / "evals" / "runs"
    path = save_run(run, runs_dir)
    console.print(f"Run record: {path}")


@eval_app.command()
def generate(
    label: str = typer.Option("gen-baseline-v1", help="Run label for the record filename."),
    top_k: int = typer.Option(5, help="Chunks stuffed into the prompt."),
    judge: bool = typer.Option(False, help="Run the LLM-judge layer (needs AHX_JUDGE_* set)."),
) -> None:
    """Run the generation-tier eval (full ask pipeline over the golden set)."""
    import sys

    from ahx.config import get_settings
    from ahx.evals.generation import (
        GenQuestionResult,
        run_generation_eval,
        save_generation_run,
    )
    from ahx.evals.golden import CATEGORIES, load_golden_set
    from ahx.llm import judge_model_from_settings

    settings = get_settings()
    golden_dir = Path(__file__).resolve().parents[2] / "evals" / "golden"
    questions = load_golden_set(golden_dir)

    judge_model = None
    if judge:
        judge_model = judge_model_from_settings(settings)
        if judge_model is None:
            console.print("[red]--judge needs AHX_JUDGE_BASE_URL and AHX_JUDGE_MODEL set.[/red]")
            raise typer.Exit(code=1)

    def progress(result: GenQuestionResult) -> None:
        mark = "refused" if result.refused else f"markers={result.markers_used}"
        recall = (
            f"cit-recall={result.citation_span_recall:.0%}"
            if result.citation_span_recall is not None
            else "oos"
        )
        console.print(
            f"  {result.question_id:<10} ok={result.refusal_correct!s:<5} {mark:<16} "
            f"{recall:<15} {result.latency_ms:>6}ms"
        )

    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    run = asyncio.run(
        run_generation_eval(
            settings, questions, label=label, top_k=top_k, judge=judge_model, on_result=progress
        ),
        loop_factory=loop_factory,
    )

    def pct(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "—"

    def score(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "—"

    table = Table(
        title=f"Generation eval — {run.label} · {run.chat_model} · {run.prompt_version}"
        + (f" · judge={run.judge_model}" if run.judge_model else " · no judge")
    )
    for column in (
        "category",
        "n",
        "refused",
        "refusal ok",
        "cit recall",
        "cit precision",
        "faith",
        "compl",
        "latency",
    ):
        table.add_column(column)
    aggregates = run.aggregates
    for category in CATEGORIES:
        agg = aggregates.by_category.get(category)
        if agg is None:
            continue
        table.add_row(
            category,
            str(agg.count),
            str(agg.refused),
            pct(agg.refusal_correct),
            pct(agg.citation_span_recall),
            pct(agg.citation_precision),
            score(agg.faithfulness),
            score(agg.completeness),
            f"{agg.mean_latency_ms}ms",
        )
    table.add_row(
        "[bold]overall[/bold]",
        str(aggregates.questions),
        "",
        pct(aggregates.refusal_accuracy_oos),
        f"[bold]{pct(aggregates.citation_span_recall)}[/bold]",
        f"[bold]{pct(aggregates.citation_precision)}[/bold]",
        score(aggregates.faithfulness),
        score(aggregates.completeness),
        f"{aggregates.mean_latency_ms}ms",
    )
    console.print(table)
    console.print(
        f"false refusal rate (in-scope): {aggregates.false_refusal_rate:.1%} · "
        f"mean completion tokens: {aggregates.mean_completion_tokens or 0:.0f}"
    )

    runs_dir = Path(__file__).resolve().parents[2] / "evals" / "runs"
    path = save_generation_run(run, runs_dir)
    console.print(f"Run record: {path}")


@eval_app.command()
def rejudge(
    record: Annotated[Path, typer.Argument(help="Path to a saved generation run record (JSON).")],
    label: str = typer.Option("rejudged", help="Label for the new record."),
) -> None:
    """Re-judge a saved run's frozen answers with the current rubric —
    isolates judge changes from generation nondeterminism."""
    import sys

    from ahx.config import get_settings
    from ahx.evals.generation import GenQuestionResult, rejudge_run, save_generation_run
    from ahx.llm import judge_model_from_settings

    settings = get_settings()
    judge_model = judge_model_from_settings(settings)
    if judge_model is None:
        console.print("[red]Needs AHX_JUDGE_BASE_URL and AHX_JUDGE_MODEL set.[/red]")
        raise typer.Exit(code=1)

    def progress(result: GenQuestionResult) -> None:
        if result.faithfulness is not None:
            console.print(
                f"  {result.question_id:<10} faith={result.faithfulness} "
                f"compl={result.completeness}"
            )

    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    run = asyncio.run(
        rejudge_run(settings, record, judge_model, label, on_result=progress),
        loop_factory=loop_factory,
    )

    def score(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "—"

    console.print(
        f"\nfaith={score(run.aggregates.faithfulness)} "
        f"compl={score(run.aggregates.completeness)} "
        f"(judge {run.judge_model}, rubric {run.judge_rubric})"
    )
    runs_dir = Path(__file__).resolve().parents[2] / "evals" / "runs"
    path = save_generation_run(run, runs_dir)
    console.print(f"Run record: {path}")


mcp_app = typer.Typer(help="MCP server over the corpus (golden-set authoring).")
app.add_typer(mcp_app, name="mcp")


@mcp_app.command(name="serve")
def mcp_serve() -> None:
    """Run the corpus MCP server on stdio (wired via repo-root .mcp.json)."""
    from ahx.mcp_server import run

    run()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = True) -> None:
    """Run the API server (dev mode)."""
    import sys

    import uvicorn

    if sys.platform == "win32" and not reload:
        # psycopg async cannot run on Windows' proactor loop (db.py), and
        # uvicorn 0.36+ picks its loop via an explicit factory (policy is
        # ignored): proactor for plain serving, selector for reload workers.
        # So reload mode is already fine; no-reload needs the factory forced.
        server = uvicorn.Server(uvicorn.Config("ahx.api.app:app", host=host, port=port))
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
        return
    uvicorn.run("ahx.api.app:app", host=host, port=port, reload=reload)
