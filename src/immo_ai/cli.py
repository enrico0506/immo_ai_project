import asyncio
from pathlib import Path
from typing import Optional

import typer

from immo_ai.ingestion.immobilie1 import BUNDESLAND_SLUGS, CrawlConfig, collect_links, fetch_bodies
from immo_ai.io.paths import build_run_paths
from immo_ai.parsing.immobilie1 import parse_files
from immo_ai.utils.logging import setup_logging
from immo_ai.utils.http import RetryConfig

app = typer.Typer(add_completion=False, help="Immo AI CLI")
parse_app = typer.Typer(help="Parse HTML bodies into structured data")
ingest_app = typer.Typer(help="Ingest data from supported sources")
immobilie1_ingest_app = typer.Typer(help="Ingest immobilie1 data")


@app.callback()
def main() -> None:
    setup_logging()


@parse_app.command("immobilie1")
def parse_immobilie1(
    input_path: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_path: Optional[Path] = typer.Option(None, "--output"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    bs_parser: str = typer.Option("lxml", "--bs-parser"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    run_paths = build_run_paths(Path.cwd(), "immobilie1", run_id)
    resolved_output = output_path or (run_paths.processed_dir / "exposes.jsonl.gz")
    rejects_path = run_paths.rejects_path

    parse_files(
        input_path=input_path,
        output_path=resolved_output,
        rejects_path=rejects_path,
        run_paths=run_paths,
        source="immobilie1",
        run_id=run_paths.run_id,
        limit=limit,
        bs_parser=bs_parser,
    )


@immobilie1_ingest_app.command("collect-links")
def ingest_collect_links(
    base_url_template: str = typer.Option(
        "https://www.immobilie1.de/immobilien/{state}/wohnung/kaufen?page={page}",
        "--base-url-template",
    ),
    states: Optional[list[str]] = typer.Option(None, "--state"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    throttle_seconds: float = typer.Option(1.0, "--throttle-seconds"),
    timeout_seconds: float = typer.Option(20.0, "--timeout-seconds"),
    retries: int = typer.Option(2, "--retries"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    output_path: Optional[Path] = typer.Option(None, "--output"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    run_paths = build_run_paths(Path.cwd(), "immobilie1", run_id)
    resolved_output = output_path or (run_paths.raw_dir / "expose_links.json")
    rejects_path = run_paths.raw_dir / "rejects.jsonl.gz"
    config = CrawlConfig(
        throttle_seconds=throttle_seconds,
        retry=RetryConfig(retries=retries),
        timeout_seconds=timeout_seconds,
    )

    resolved_states = states or BUNDESLAND_SLUGS
    asyncio.run(
        collect_links(
            run_paths=run_paths,
            base_url_template=base_url_template,
            states=resolved_states,
            max_pages=max_pages,
            config=config,
            cache_dir=cache_dir,
            output_path=resolved_output,
            rejects_path=rejects_path,
        )
    )


@immobilie1_ingest_app.command("fetch-bodies")
def ingest_fetch_bodies(
    links_path: Path = typer.Option(..., "--input", exists=True, readable=True),
    output_path: Optional[Path] = typer.Option(None, "--output"),
    throttle_seconds: float = typer.Option(1.0, "--throttle-seconds"),
    timeout_seconds: float = typer.Option(20.0, "--timeout-seconds"),
    retries: int = typer.Option(2, "--retries"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
) -> None:
    run_paths = build_run_paths(Path.cwd(), "immobilie1", run_id)
    resolved_output = output_path or (run_paths.raw_dir / "expose_bodies.jsonl.gz")
    rejects_path = run_paths.raw_dir / "rejects.jsonl.gz"
    config = CrawlConfig(
        throttle_seconds=throttle_seconds,
        retry=RetryConfig(retries=retries),
        timeout_seconds=timeout_seconds,
    )

    asyncio.run(
        fetch_bodies(
            run_paths=run_paths,
            links_path=links_path,
            output_path=resolved_output,
            rejects_path=rejects_path,
            config=config,
            cache_dir=cache_dir,
            limit=limit,
        )
    )


app.add_typer(parse_app, name="parse")
app.add_typer(ingest_app, name="ingest")
ingest_app.add_typer(immobilie1_ingest_app, name="immobilie1")
