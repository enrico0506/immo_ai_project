import asyncio
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR / "src"))

from immo_ai.ingestion.immobilie1 import CrawlConfig, fetch_bodies
from immo_ai.io.paths import build_run_paths
from immo_ai.utils.http import RetryConfig
from immo_ai.utils.logging import setup_logging

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LINKS_FILE = RAW_DIR / "immobilie1_expose_links.json"
OUTPUT_BODIES_FILE = RAW_DIR / "immobilie1_expose_bodies.jsonl.gz"

THROTTLE_SECONDS = float(os.getenv("IMMOBILIE1_THROTTLE_SECONDS", "1.0"))
FETCH_TIMEOUT_SECONDS = float(os.getenv("IMMOBILIE1_FETCH_TIMEOUT_SECONDS", "20"))
RETRIES = int(os.getenv("IMMOBILIE1_FETCH_RETRIES", "2"))
LIMIT = os.getenv("IMMOBILIE1_FETCH_LIMIT")


def main() -> None:
    setup_logging()
    run_paths = build_run_paths(ROOT_DIR, "immobilie1", run_id=None)

    config = CrawlConfig(
        throttle_seconds=THROTTLE_SECONDS,
        retry=RetryConfig(retries=RETRIES),
        timeout_seconds=FETCH_TIMEOUT_SECONDS,
    )
    limit = int(LIMIT) if LIMIT else None

    asyncio.run(
        fetch_bodies(
            run_paths=run_paths,
            links_path=LINKS_FILE,
            output_path=OUTPUT_BODIES_FILE,
            rejects_path=RAW_DIR / "rejects.jsonl.gz",
            config=config,
            cache_dir=None,
            limit=limit,
        )
    )


if __name__ == "__main__":
    main()
