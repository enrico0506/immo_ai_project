import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import os
import aiohttp
from aiohttp import ClientTimeout
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# New directory where all body JSON chunks will be stored
BODIES_DIR = DATA_DIR / "raw/immobilie1_expose_bodies"
BODIES_DIR.mkdir(parents=True, exist_ok=True)

LINKS_FILE = RAW_DIR / "immobilie1_expose_links.json"

# Number of concurrent workers (coroutines fetching in parallel)
WORKERS = max(250, int(os.getenv("IMMOBILIE1_WORKERS", "50")))
THROTTLE_SECONDS = float(os.getenv("IMMOBILIE1_THROTTLE_SECONDS", "0.0"))

FETCH_TIMEOUT_MS = int(os.getenv("IMMOBILIE1_FETCH_TIMEOUT_MS", "20000"))
RETRIES = int(os.getenv("IMMOBILIE1_FETCH_RETRIES", "2"))
DEDUPE = os.getenv("IMMOBILIE1_DEDUPE", "0") != "0"

# How many bodies per JSON file
CHUNK_SIZE = int(os.getenv("IMMOBILIE1_CHUNK_SIZE", "2000"))


def dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def is_error_html(body: str | None) -> bool:
    return body is None or (isinstance(body, str) and body.startswith("ERROR"))


async def fetch_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    timeout_seconds: float,
    retries: int,
) -> str | None:
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            async with session.get(url, timeout=timeout_seconds) as resp:
                # Don't raise on status; we still want the HTML to debug 4xx/5xx
                text = await resp.text()
                return text
        except Exception as e:
            last_error = f"ERROR: {e}"
            if attempt < retries:
                # Small backoff before retrying this URL
                await asyncio.sleep(0.5)
    return last_error


async def worker(
    name: str,
    session: aiohttp.ClientSession,
    queue: asyncio.Queue,
    results: Dict[int, str | None],
    progress: tqdm,
    timeout_seconds: float,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        idx, url = item
        html = await fetch_with_retries(session, url, timeout_seconds, RETRIES)
        results[idx] = html

        progress.update(1)

        if THROTTLE_SECONDS > 0:
            await asyncio.sleep(THROTTLE_SECONDS)

        queue.task_done()


async def main() -> None:
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_urls = payload.get("all_urls") or payload.get("urls") or []
    expose_urls = dedupe_preserve_order(raw_urls) if DEDUPE else raw_urls
    if not expose_urls:
        print(f"No URLs found in {LINKS_FILE}")
        return

    total = len(expose_urls)
    print(f"Loaded {total} expose URLs from {LINKS_FILE}")
    print(f"Running with {WORKERS} workers...", flush=True)

    timeout_seconds = FETCH_TIMEOUT_MS / 1000.0
    timeout = ClientTimeout(total=timeout_seconds)

    # Allow a lot of parallel connections; tune if needed
    connector = aiohttp.TCPConnector(limit=WORKERS * 2)

    results: Dict[int, str | None] = {}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        queue: asyncio.Queue = asyncio.Queue()

        # Enqueue all jobs
        for idx, url in enumerate(expose_urls):
            queue.put_nowait((idx, url))

        # Add sentinel None items to stop workers
        for _ in range(WORKERS):
            queue.put_nowait(None)

        progress = tqdm(total=total, desc="Bodies", unit="body", dynamic_ncols=True)

        workers = [
            asyncio.create_task(
                worker(
                    name=f"worker-{i+1}",
                    session=session,
                    queue=queue,
                    results=results,
                    progress=progress,
                    timeout_seconds=timeout_seconds,
                )
            )
            for i in range(WORKERS)
        ]

        try:
            # Wait until all tasks in the queue are processed
            await queue.join()
        finally:
            progress.close()

        # Ensure all workers have exited
        for w in workers:
            await w

    # ---- Write out results in chunks of CHUNK_SIZE ----
    errors_total = 0
    chunk_index = 0
    items_in_chunk = 0
    current_file = None
    current_tmp_path: Path | None = None
    current_final_path: Path | None = None
    first_item_in_chunk = True

    def open_new_chunk_file(idx: int) -> None:
        nonlocal current_file, current_tmp_path, current_final_path
        nonlocal first_item_in_chunk, items_in_chunk, chunk_index

        # Close previous file if open
        if current_file is not None:
            current_file.write("\n  ]\n}\n")
            current_file.close()
            current_tmp_path.replace(current_final_path)

        # New file path
        current_final_path = BODIES_DIR / f"immobilie1_expose_bodies_{chunk_index:05d}.json"
        current_tmp_path = current_final_path.with_suffix(".tmp")

        current_file = open(current_tmp_path, "w", encoding="utf-8")
        current_file.write('{\n')
        current_file.write(f'  "generated_at": "{datetime.utcnow().isoformat()}Z",\n')
        current_file.write(f'  "chunk_index": {chunk_index},\n')
        current_file.write(f'  "chunk_size": {CHUNK_SIZE},\n')
        current_file.write(f'  "total": {total},\n')
        current_file.write(f'  "source_links_file": "{LINKS_FILE}",\n')
        current_file.write('  "items": [\n')

        first_item_in_chunk = True
        items_in_chunk = 0
        chunk_index += 1

    # Iterate over all results and write them chunked
    for idx, url in enumerate(expose_urls):
        # Start a new chunk if there is no file yet or the current one is full
        if current_file is None or items_in_chunk >= CHUNK_SIZE:
            open_new_chunk_file(idx)

        html = results.get(idx)
        if is_error_html(html):
            errors_total += 1

        item = {"url": url, "html": html}
        if not first_item_in_chunk:
            current_file.write(",\n")
        json.dump(item, current_file, ensure_ascii=False)
        first_item_in_chunk = False

        items_in_chunk += 1

    # Close the last chunk file
    if current_file is not None:
        current_file.write("\n  ]\n}\n")
        current_file.close()
        current_tmp_path.replace(current_final_path)

    print(f"Saved {chunk_index} chunk file(s) in {BODIES_DIR}")
    print(f"Errors: {errors_total} / {total}")


if __name__ == "__main__":
    asyncio.run(main())
