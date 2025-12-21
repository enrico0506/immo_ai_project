import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import urlparse, parse_qsl, urlencode

from playwright.async_api import async_playwright, Page
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
# How many pages of search results to scrape in total (including the first page).
# - Set to an integer (e.g., 10) to limit pages.
# - Set to None to go through *all* pages until a page has no links.
PAGES_TO_FETCH: Optional[int] = 1000  # e.g. 10, 5, or None for all pages

# Fetch and store the HTML bodies for each expose. Turn off to avoid huge runs.
FETCH_BODIES: bool = False

# If FETCH_BODIES is True, fetch in batches of this size to be gentle.
BODY_FETCH_BATCH_SIZE: int = 50

# Paths
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Where to store collected data
OUTPUT_JSON_FILE = RAW_DIR / "immoscout_exposes.json"
OUTPUT_LINKS_FILE = RAW_DIR / "immoscout_exposes_links.json"


async def fetch_html_bodies_in_session(page: Page, urls: List[str]) -> Dict[str, str]:
    """
    Runs inside an already-initialized browser session.
    Uses window.fetch() to GET each URL and returns their bodies as text.
    NOTE: These must be allowed by CORS / same-origin rules.
    """
    results = await page.evaluate(
        """
        async (urls) => {
            const out = {};
            for (const url of urls) {
                try {
                    const res = await fetch(url, {
                        method: 'GET',
                        credentials: 'include', // send cookies/session
                    });
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
    return results


async def extract_expose_links(page: Page) -> List[str]:
    """Collect expose links from listing cards on the *currently loaded* page."""
    # Scroll a bit to trigger lazy rendering of listings
    await page.mouse.wheel(0, 3000)
    await page.wait_for_timeout(1500)
    await page.wait_for_selector('[data-elementtype="hybridViewCardContainer"]', timeout=10_000)

    links = await page.evaluate(
        """
        () => {
            const hrefs = new Set();
            const containers = document.querySelectorAll('[data-elementtype="hybridViewCardContainer"]');

            for (const container of containers) {
                const anchors = container.querySelectorAll('a[href]');
                anchors.forEach((a) => {
                    const href = a.getAttribute('href') || '';
                    if (!href) return;
                    // Keep anything that clearly points to an expose (direct or via controller)
                    if (/expose/.test(href) || /exposeId=/.test(href)) {
                        const absolute = href.startsWith('http')
                            ? href
                            : new URL(href, window.location.origin).href;
                        hrefs.add(absolute);
                    }
                });
            }

            return Array.from(hrefs);
        }
        """
    )

    return links


def _make_page_url(base_url: str, page_number: int) -> str:
    """
    Take the current search URL (with all filters) and
    return the same URL but with pagenumber=<page_number>.
    """
    parsed = urlparse(base_url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["pagenumber"] = str(page_number)
    new_query = urlencode(qs, doseq=True)
    return parsed._replace(query=new_query).geturl()


def _get_current_page_number(url: str) -> int:
    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    return int(qs.get("pagenumber", "1"))  # default: page 1


async def collect_links_via_fetch(page: Page, max_pages: Optional[int] = None) -> List[str]:
    """
    Gather expose links from the current result page (DOM)
    and additional pages via `fetch` inside the browser, without page.goto.

    max_pages:
      - int N  -> scrape N pages in total (including the first page).
      - None   -> scrape until there are no more links (all pages).

    Progress:
      - tqdm bar shows completed pages
      - postfix displays total exposes collected so far
    """
    all_links: set[str] = set()

    base_url = page.url
    start_page = _get_current_page_number(base_url)

    # tqdm progress bar:
    # - if max_pages is known, show a bounded bar (0 / max_pages)
    # - if max_pages is None, show an open-ended bar
    total_for_tqdm = max_pages if max_pages is not None else None
    page_pbar = tqdm(total=total_for_tqdm, desc="Pages", unit="page")

    try:
        # --- 1) Current page via live DOM ---
        first_page_links = await extract_expose_links(page)
        all_links.update(first_page_links)
        page_pbar.update(1)
        page_pbar.set_postfix(exposes=len(all_links))

        # If caller asked for only 1 page, we're done
        if max_pages is not None and page_pbar.n >= max_pages:
            return sorted(all_links)

        # --- 2) Loop over subsequent pages one by one ---
        # Next page number is always start_page + pages_done (pages_done == page_pbar.n)
        while True:
            # Respect page limit if set
            if max_pages is not None and page_pbar.n >= max_pages:
                break

            page_num = start_page + page_pbar.n
            page_url = _make_page_url(base_url, page_num)

            page_result = await page.evaluate(
                """
                async (url) => {
                    try {
                        const res = await fetch(url, {
                            method: 'GET',
                            credentials: 'include', // keep session / captcha state
                        });

                        if (!res.ok) {
                            return { url, links: [], status: res.status, error: `HTTP ${res.status}` };
                        }

                        const html = await res.text();

                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');

                        const hrefs = new Set();
                        const containers = doc.querySelectorAll('[data-elementtype="hybridViewCardContainer"]');

                        for (const container of containers) {
                            const anchors = container.querySelectorAll('a[href]');
                            anchors.forEach((a) => {
                                const href = a.getAttribute('href') || '';
                                if (!href) return;
                                if (/expose/.test(href) || /exposeId=/.test(href)) {
                                    const absolute = href.startsWith('http')
                                        ? href
                                        : new URL(href, url).href; // base is the search page URL
                                    hrefs.add(absolute);
                                }
                            });
                        }

                        return {
                            url,
                            links: Array.from(hrefs),
                            status: res.status,
                        };
                    } catch (e) {
                        return { url, links: [], status: null, error: String(e) };
                    }
                }
                """,
                page_url,
            )

            links = page_result.get("links", []) or []
            new_links = [l for l in links if l not in all_links]
            all_links.update(new_links)

            page_pbar.update(1)
            page_pbar.set_postfix(exposes=len(all_links))

            # If no links on this page, assume we reached the end
            if not links:
                break

            # small delay so we don't hammer the site
            await asyncio.sleep(0.5)

    finally:
        page_pbar.close()

    return sorted(all_links)


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headful so you can interact
        page = await browser.new_page()

        # 1) Go to your search page (filters & sorting can be adjusted manually)
        await page.goto("https://www.immobilienscout24.de/Suche/de/wohnung-kaufen?sorting=2")

        # 2) You solve captcha / log in / adjust filters manually
        print("Use the browser window to log in / pass captcha / set filters.")
        print("When you're done, come back here and press Enter.")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "Press Enter when ready: ")

        # Capture the final search URL (with all filters applied) for metadata
        search_url = page.url

        # 3) Collect expose links from current page + more pages via fetch
        expose_links = await collect_links_via_fetch(page, max_pages=PAGES_TO_FETCH)
        print(f"\nDone. Collected {len(expose_links)} expose URLs.")

        # Save links immediately so you keep results even if body fetch fails or is disabled
        links_payload = {
            "search_url": search_url,
            "pages_requested": PAGES_TO_FETCH,
            "total_exposes": len(expose_links),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "urls": expose_links,
        }
        with open(OUTPUT_LINKS_FILE, "w", encoding="utf-8") as f:
            json.dump(links_payload, f, ensure_ascii=False, indent=2)
        print(f"Saved links to {OUTPUT_LINKS_FILE}")

        expose_html_map: Dict[str, str] = {}

        if FETCH_BODIES and expose_links:
            print(f"Fetching HTML bodies in batches of {BODY_FETCH_BATCH_SIZE} ...")
            for i in range(0, len(expose_links), BODY_FETCH_BATCH_SIZE):
                batch = expose_links[i : i + BODY_FETCH_BATCH_SIZE]
                batch_map = await fetch_html_bodies_in_session(page, batch)
                expose_html_map.update(batch_map)
                print(f"  fetched {i + len(batch)} / {len(expose_links)} bodies")
                await asyncio.sleep(0.25)  # brief pause to be polite
            print(f"Fetched HTML for {len(expose_html_map)} exposes.")
        else:
            print("Skipping body fetch (FETCH_BODIES is False).")

        # 5) Save everything to JSON (in same folder as this script)
        output = {
            "search_url": search_url,
            "pages_requested": PAGES_TO_FETCH,
            "total_exposes": len(expose_links),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "items": [
                {"url": url, "html": expose_html_map.get(url)} for url in expose_links
            ],
        }

        with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Saved results to {OUTPUT_JSON_FILE}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
