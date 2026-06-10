"""Pipeline entrypoints: `uv run ahx <command>`.

Offline work (ingest, evals) runs through this CLI, not the API server —
ingest is a local batch job, eval runs cost money and happen at phase
boundaries (see docs/python-stack.md §2).
"""

import typer

app = typer.Typer(help="Antic Historian pipeline commands.", no_args_is_help=True)


@app.command()
def ingest() -> None:
    """Acquire, normalize, chunk and embed the corpus (Phase 1)."""
    typer.echo("Not implemented yet — Phase 1 (see project-plan.md).")
    raise typer.Exit(code=1)


@app.command(name="eval")
def run_eval() -> None:
    """Run the golden-set evaluation suite (Phase 2)."""
    typer.echo("Not implemented yet — Phase 2 (see project-plan.md).")
    raise typer.Exit(code=1)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = True) -> None:
    """Run the API server (dev mode)."""
    import uvicorn

    uvicorn.run("ahx.api.app:app", host=host, port=port, reload=reload)
