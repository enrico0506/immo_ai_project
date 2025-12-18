import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from playwright.async_api import async_playwright, Page

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LINKS_FILE = RAW_DIR / "immoscout_exposes_links.json"

# Each JSON file will contain at most this many exposes
CHUNK_SIZE = 2000

# How many URLs to fetch per "fetch_html_bodies_in_session" call
BATCH_SIZE = 25  # smaller batches = slower, but safer

# After this many exposes, pause and let you check/solve any captchas
HUMAN_CHECK_INTERVAL = 500


async def fetch_html_bodies_in_session(page: Page, urls: List[str]) -> Dict[str, str]:
    """Fetch each URL via window.fetch inside the current browser session (keeps cookies/captcha)."""
    return await page.evaluate(
        """
        async (urls) => {
            const out = {};
            for (const url of urls) {
                try {
                    const res = await fetch(url, { method: 'GET', credentials: 'include' });
                    const text = await res.text();
                    out[url] = text;
                } catch (e) {
                    out[url] = `ERROR: ${e}`;
                }
            }
            return out;
        }
        """,
        urls,
    )


async def main() -> None:
    # Load URLs from the links file
    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)
    expose_urls: List[str] = payload.get("urls", [])
    search_url = payload.get("search_url") or "https://www.immobilienscout24.de/Suche/de/wohnung-kaufen?sorting=2"

    print(f"Loaded {len(expose_urls)} expose URLs from {LINKS_FILE}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=50,  # a bit slower, looks more "human"
        )

        # You can tune the user agent if you want to look more like a normal browser
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        )

        # Warm the session; you solve captcha/login as needed
        await page.goto(search_url)
        print("Use the browser window to log in / pass captcha if needed.")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "Press Enter when ready: ")

        current_chunk_items = []  # items to be written into the current JSON file
        file_index = 1            # expose_bodies_0001.json, expose_bodies_0002.json, ...
        total_processed = 0       # total number of exposes processed across all chunks

        def flush_chunk() -> None:
            """Write the current chunk to disk and reset the buffer."""
            nonlocal current_chunk_items, file_index

            if not current_chunk_items:
                return

            output = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "chunk_index": file_index,
                "chunk_size": len(current_chunk_items),
                "total_processed_up_to_chunk": total_processed,
                "items": current_chunk_items,
            }

            output_file = RAW_DIR / f"expose_bodies_{file_index:04d}.json"
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump(output, f_out, ensure_ascii=False, indent=2)

            print(
                f"Saved chunk {file_index} with {len(current_chunk_items)} items "
                f"to {output_file}"
            )

            file_index += 1
            current_chunk_items = []

        total_urls = len(expose_urls)

        for start in range(0, total_urls, BATCH_SIZE):
            batch = expose_urls[start : start + BATCH_SIZE]
            batch_bodies = await fetch_html_bodies_in_session(page, batch)

            # Add each item of this batch to the current chunk
            for url in batch:
                current_chunk_items.append(
                    {
                        "url": url,
                        "html": batch_bodies.get(url),
                    }
                )
                total_processed += 1

                # If we've reached the chunk size, flush to a new JSON file
                if len(current_chunk_items) >= CHUNK_SIZE:
                    flush_chunk()

                # Periodic human check to reduce chance of blocking/captcha
                if total_processed % HUMAN_CHECK_INTERVAL == 0:
                    print(
                        f"Processed {total_processed} exposes. "
                        "Check the browser for any captcha / blocks."
                    )
                    await loop.run_in_executor(
                        None,
                        input,
                        "If everything looks fine, press Enter to continue: ",
                    )

            print(
                f"Fetched {total_processed} / {total_urls} bodies "
                f"(current chunk size: {len(current_chunk_items)})"
            )

            # Random delay to look less like a bot
            await asyncio.sleep(random.uniform(1.0, 3.0))

        # Flush remaining items in the last chunk
        flush_chunk()

        print(f"Done. Total bodies fetched: {total_processed}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
