"""math-trainer CLI — a thin wrapper over the ingestion library core."""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from math_trainer.core.config import load_config
from math_trainer.factory import build_service

app = typer.Typer(help="math-trainer — material extraction pipeline (Docling → Neo4j).")


def _run(coro):
    return asyncio.run(coro)


@app.command("init-db")
def init_db(config: str = typer.Option(None, help="Path to config YAML.")) -> None:
    """Create constraints, indexes, and the vector index."""
    cfg = load_config(config)
    service, driver = build_service(cfg)

    async def _go() -> None:
        try:
            await service.init_db()
        finally:
            await driver.close()

    _run(_go())
    typer.echo("Schema bootstrapped.")


@app.command("ingest")
def ingest(
    path: Path = typer.Argument(..., exists=True, help="PDF or image to ingest."),
    title: str = typer.Option(None, help="Document title (defaults to filename)."),
    source_uuid: str = typer.Option(None, help="Reuse/resume a specific Source uuid."),
    from_stage: int = typer.Option(1, min=1, max=7, help="Resume from this stage."),
    config: str = typer.Option(None, help="Path to config YAML."),
) -> None:
    """Run the full pipeline (or resume) on a PDF or image."""
    cfg = load_config(config)
    service, driver = build_service(cfg)

    async def _go() -> str:
        try:
            return await service.ingest(
                path, title=title, source_uuid=source_uuid, from_stage=from_stage
            )
        finally:
            await driver.close()

    uid = _run(_go())
    typer.echo(f"Ingested. Source uuid: {uid}")


@app.command("stage")
def stage(
    number: int = typer.Argument(..., min=2, max=7, help="Stage number (2–7)."),
    source_uuid: str = typer.Option(..., help="Source uuid to run the stage on."),
    config: str = typer.Option(None, help="Path to config YAML."),
) -> None:
    """Run a single stage in isolation (debugging)."""
    cfg = load_config(config)
    service, driver = build_service(cfg)

    async def _go() -> None:
        try:
            await service.run_stage(number, source_uuid)
        finally:
            await driver.close()

    _run(_go())
    typer.echo(f"Stage {number} complete for {source_uuid}.")


if __name__ == "__main__":
    app()
