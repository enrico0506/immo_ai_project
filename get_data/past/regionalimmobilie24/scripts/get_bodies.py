#!/usr/bin/env python3
import argparse
import json
import random
import re
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Input: unified link dump (all regions)
DEFAULT_JSON = RAW_DIR / "regionalimmobilien24_expose_links_by_region.json"
# Output: base name for JSON containing exposé bodies (split into chunks if needed)
DEFAULT_OUTJSON = RAW_DIR / "regionalimmobilien24_expose_bodies.json"

# Fixed number of parallel requests
MAX_WORKERS = 20

# Max number of exposés per output JSON file
CHUNK_SIZE = 2000

# ID: optional "im" + digits at the end of the URL
ID_RE = re.compile(r"/((?:im)?\d+)/?$", re.IGNORECASE)
# Redirect hint: listing element only shows source/external link
REDIRECT_QUELLE_RE = re.compile(r">\s*quelle\s*:", re.IGNORECASE)
REDIRECT_CLICKOUT_MARKERS = ("itemprop=\"clickout\"", "data-tr-an=\"clickout\"")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Download Regionalimmobilien24 exposé HTML bodies into JSON file(s)."
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to link JSON (default: data/raw/regionalimmobilien24_expose_links_by_region.json).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUTJSON,
        help="Base path for JSON with bodies (will be split into chunks).",
    )
    ap.add_argument("--region", action="append", help="Region name filter (can be repeated).")
    ap.add_argument("--timeout", type=float, default=25.0, help="HTTP timeout per request.")
    ap.add_argument("--sleep-min", type=float, default=0.15, help="Min pause between requests (s).")
    ap.add_argument("--sleep-max", type=float, default=0.6, help="Max pause between requests (s).")

    # Multiple external worker processes (sharding)
    ap.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Index of this worker (0-based).",
    )
    ap.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="Total number of workers sharing the work.",
    )

    return ap.parse_args()


def load_links(data: Dict, region_filter: Optional[Set[str]]) -> List[Tuple[str, str]]:
    """Flatten region->links into a list of (region, url) pairs, deduped in order."""
    pairs: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for region, payload in data.items():
        if region_filter and region not in region_filter:
            continue
        for url in payload.get("links") or []:
            if url in seen:
                continue
            seen.add(url)
            pairs.append((region, url))
    return pairs


def extract_slug_and_id(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    slug = path_parts[0] if path_parts and path_parts[0] else "unknown"
    m = ID_RE.search(parsed.path)
    expose_id = m.group(1) if m else "unknown"
    return slug, expose_id


def is_external_redirect(html: str) -> bool:
    """True if the HTML only links out (Quelle/clickout) instead of an actual exposé."""
    if not html:
        return False
    lower_html = html.lower()
    if any(marker in lower_html for marker in REDIRECT_CLICKOUT_MARKERS):
        return True
    return bool(REDIRECT_QUELLE_RE.search(lower_html))


def fetch_one(session: requests.Session, url: str, timeout: float) -> Tuple[Optional[str], Optional[str]]:
    try:
        resp = session.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ri24-body-downloader/1.0)",
                "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            },
        )
        if resp.status_code >= 400:
            return None, f"HTTP {resp.status_code}"
        return resp.text, None
    except Exception as exc:
        return None, str(exc)


def main():
    args = parse_args()

    if args.worker_count < 1:
        raise SystemExit("--worker-count must be >= 1")
    if not (0 <= args.worker_index < args.worker_count):
        raise SystemExit("--worker-index must be in [0, worker-count)")

    with args.json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    region_filter = set(args.region) if args.region else None
    pairs = load_links(data, region_filter)

    # 1) Before download, filter out links with non-numeric IDs
    filtered_pairs: List[Tuple[str, str]] = []
    for region, url in pairs:
        _, expose_id = extract_slug_and_id(url)
        # Keep only IDs consisting solely of digits
        if not expose_id.isdigit():
            # e.g. /im9360812/ -> expose_id = "im9360812" -> will be skipped
            continue
        filtered_pairs.append((region, url))

    pairs = filtered_pairs

    # 2) Sharding across multiple worker processes (no global max limit)
    if args.worker_count > 1:
        total_before = len(pairs)
        pairs = [
            p for i, p in enumerate(pairs)
            if i % args.worker_count == args.worker_index
        ]
        print(
            f"Worker {args.worker_index + 1}/{args.worker_count}: "
            f"taking {len(pairs)} of {total_before} exposés with {MAX_WORKERS} threads."
        )
    else:
        print(f"Single worker: processing {len(pairs)} exposés with {MAX_WORKERS} threads.")

    if not pairs:
        print("Nothing to do for this worker (no matching exposés).")
        return

    session = requests.Session()
    results: List[Dict] = []
    tasks: Dict = {}

    # 3) Parallel requests with fixed number of threads
    with tqdm(total=len(pairs), desc="Downloading", unit="expose", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for region, url in pairs:
                slug, expose_id = extract_slug_and_id(url)
                fut = ex.submit(fetch_one, session, url, args.timeout)
                tasks[fut] = (region, url, slug, expose_id)

            for fut in as_completed(tasks):
                region, url, slug, expose_id = tasks[fut]
                html, err = fut.result()
                if html:
                    # Optional: skip external redirect-only exposes
                    if is_external_redirect(html):
                        # keep quiet; just drop redirect-only pages
                        pass
                    else:
                        results.append(
                            {
                                "region": region,
                                "slug": slug,
                                "id": expose_id,
                                "url": url,
                                "html": html,
                            }
                        )
                elif err:
                    pbar.write(f"[FAIL] {url} ({err})")
                # small pause between processing; requests still run in parallel in threads
                time.sleep(random.uniform(args.sleep_min, args.sleep_max))
                pbar.update(1)
                pbar.set_postfix(downloaded=len(results), refresh=False)

    # 4) Write results into one or more JSON files (max CHUNK_SIZE per file)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    if total == 0:
        print(
            f"Done (worker {args.worker_index + 1}/{args.worker_count}). "
            f"No exposés to store."
        )
        return

    base = args.out_json.stem      # e.g. "regionalimmobilien24_expose_bodies"
    suffix = args.out_json.suffix  # e.g. ".json"

    num_chunks = math.ceil(total / CHUNK_SIZE)

    for idx in range(num_chunks):
        start = idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total)
        chunk = results[start:end]

        # zero-padded counter: _0001, _0002, ...
        part_name = f"{base}_{idx + 1:04d}{suffix}"
        part_path = args.out_json.with_name(part_name)

        with part_path.open("w", encoding="utf-8") as outf:
            json.dump({"exposes": chunk}, outf, ensure_ascii=False)

        print(f"Stored {len(chunk)} bodies in {part_path}")

    print(
        f"Done (worker {args.worker_index + 1}/{args.worker_count}). "
        f"Stored {total} bodies in {num_chunks} file(s) "
        f"with up to {CHUNK_SIZE} exposés each."
    )


if __name__ == "__main__":
    main()
