import asyncio
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR / "src"))

from immo_ai.ingestion.immobilie1 import BUNDESLAND_SLUGS, CrawlConfig, collect_links
from immo_ai.io.paths import build_run_paths
from immo_ai.utils.http import RetryConfig
from immo_ai.utils.logging import setup_logging

BASE_URL_TEMPLATE = "https://www.immobilie1.de/immobilien/{state}/wohnung/kaufen?page={page}"
PAGES_TO_FETCH = os.getenv("IMMOBILIE1_MAX_PAGES")
THROTTLE_SECONDS = float(os.getenv("IMMOBILIE1_THROTTLE_SECONDS", "1.0"))
RETRIES = int(os.getenv("IMMOBILIE1_FETCH_RETRIES", "2"))
TIMEOUT_SECONDS = float(os.getenv("IMMOBILIE1_FETCH_TIMEOUT_SECONDS", "20"))

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LINKS_FILE = RAW_DIR / "immobilie1_expose_links.json"


def main() -> None:
    setup_logging()
    max_pages = int(PAGES_TO_FETCH) if PAGES_TO_FETCH else None
    run_paths = build_run_paths(ROOT_DIR, "immobilie1", run_id=None)

    config = CrawlConfig(
        throttle_seconds=THROTTLE_SECONDS,
        retry=RetryConfig(retries=RETRIES),
        timeout_seconds=TIMEOUT_SECONDS,
    )

    asyncio.run(
        collect_links(
            run_paths=run_paths,
            base_url_template=BASE_URL_TEMPLATE,
            states=BUNDESLAND_SLUGS,
            max_pages=max_pages,
            config=config,
            cache_dir=None,
            output_path=OUTPUT_LINKS_FILE,
            rejects_path=RAW_DIR / "rejects.jsonl.gz",
        )
    )


if __name__ == "__main__":
    main()
