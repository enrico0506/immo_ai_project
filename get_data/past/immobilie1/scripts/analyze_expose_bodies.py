import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT_DIR / "src"))

from immo_ai.io.paths import build_run_paths
from immo_ai.parsing.immobilie1 import parse_files
from immo_ai.utils.logging import setup_logging

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR = RAW_DIR / "immobilie1_expose_bodies"
INPUT_FILE = RAW_DIR / "immobilie1_expose_bodies.json"  # legacy single-file dump
OUTPUT_FILE = PROCESSED_DIR / "immobilie1_exposes_structured.jsonl"

ANALYZE_LIMIT: int | None = None
BS_PARSER = os.getenv("ANALYZE_BS_PARSER", "lxml")


def find_input_path() -> Path | None:
    chunked = sorted(INPUT_DIR.glob("*.json")) if INPUT_DIR.exists() else []
    if chunked:
        return INPUT_DIR
    if INPUT_FILE.exists():
        return INPUT_FILE
    return None


def main() -> None:
    setup_logging()
    input_path = find_input_path()
    if not input_path:
        print(f"No input files found in {INPUT_DIR} or {INPUT_FILE}", file=sys.stderr)
        return

    run_paths = build_run_paths(ROOT_DIR, "immobilie1", run_id=None)
    parse_files(
        input_path=input_path,
        output_path=OUTPUT_FILE,
        rejects_path=run_paths.rejects_path,
        run_paths=run_paths,
        source="immobilie1",
        run_id=run_paths.run_id,
        limit=ANALYZE_LIMIT,
        bs_parser=BS_PARSER,
    )


if __name__ == "__main__":
    main()
