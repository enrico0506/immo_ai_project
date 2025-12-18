import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import parse_qsl, urlencode, urlparse

from playwright.async_api import Page, async_playwright
from tqdm import tqdm

BUNDESLAND_SLUGS = [
    "baden-wuerttemberg",
    "bayern",
    "berlin",
    "brandenburg",
    "bremen",
    "hamburg",
    "hessen",
    "mecklenburg-vorpommern",
    "niedersachsen",
    "nordrhein-westfalen",
    "rheinland-pfalz",
    "saarland",
    "sachsen",
    "sachsen-anhalt",
    "schleswig-holstein",
    "thueringen",
]

BASE_URL = "https://www.immobilie1.de/immobilien/{state}/wohnung/kaufen?page="
PAGES_TO_FETCH: Optional[int] = None  # e.g. 5, 10, or None for all pages
HEADLESS = True

# Paths 
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_LINKS_FILE = RAW_DIR / "immobilie1_expose_links.json"


def _get_current_page_number(url: str) -> int:
    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    return int(qs.get("page", "1"))


def _make_page_url(base_url: str, page_number: int) -> str:
    parsed = urlparse(base_url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["page"] = str(page_number)
    new_query = urlencode(qs, doseq=True)
    return parsed._replace(query=new_query).geturl()


async def accept_cookies(page: Page) -> None:
    selectors = [
        "button[data-testid='uc-accept-all-button']",
        "button:has-text('Alle akzeptieren')",
        "button:has-text('Akzeptieren')",
        "button:text('OK')",
    ]
    for sel in selectors:
        try:
            btn = await page.query_selector(sel)
            if btn:
                await btn.click()
                await asyncio.sleep(0.3)
                return
        except Exception:
            continue


def _looks_like_expose(href: str) -> bool:
    if not href:
        return False
    tail = href.rstrip("/").split("/")[-1]
    return ("-wohnung-" in href or "wohnung-" in href) and any(ch.isdigit() for ch in tail)


async def extract_expose_links_from_dom(page: Page) -> List[str]:
    await page.mouse.wheel(0, 3500)
    await page.wait_for_timeout(1200)
    links = await page.evaluate(
        """
        () => {
            const hrefs = new Set();
            const looksLikeExpose = (href) => {
                if (!href) return false;
                const tail = href.replace(/\\/$/, '').split('/').pop() || '';
                return (/-wohnung-/.test(href) || /wohnung-/.test(href)) && /\\d/.test(tail);
            };

            document.querySelectorAll('a[href]').forEach((a) => {
                const href = a.getAttribute('href') || '';
                if (!looksLikeExpose(href)) return;
                const absolute = href.startsWith('http')
                    ? href
                    : new URL(href, window.location.origin).href;
                if (!/immobilie1\\.de/.test(absolute)) return;
                hrefs.add(absolute.split('?')[0]);
            });

            return Array.from(hrefs);
        }
        """
    )
    return links


async def collect_links_via_fetch(page: Page, base_url: str, max_pages: Optional[int]) -> List[str]:
    all_links: set[str] = set()
    start_page = _get_current_page_number(base_url)
    total_for_tqdm = max_pages if max_pages is not None else None
    page_pbar = tqdm(total=total_for_tqdm, desc="Pages", unit="page", dynamic_ncols=True)

    try:
        first_page_links = await extract_expose_links_from_dom(page)
        all_links.update(first_page_links)
        page_pbar.update(1)
        page_pbar.set_postfix(exposes=len(all_links))

        if max_pages is not None and page_pbar.n >= max_pages:
            return sorted(all_links)

        while True:
            if max_pages is not None and page_pbar.n >= max_pages:
                break

            page_num = start_page + page_pbar.n
            page_url = _make_page_url(base_url, page_num)

            page_result = await page.evaluate(
                """
                async (url) => {
                    try {
                        const res = await fetch(url, { method: 'GET', credentials: 'include' });
                        if (!res.ok) {
                            return { url, links: [], status: res.status, error: `HTTP ${res.status}` };
                        }

                        const html = await res.text();
                        const parser = new DOMParser();
                        const doc = parser.parseFromString(html, 'text/html');

                        const hrefs = new Set();
                        doc.querySelectorAll('a[href]').forEach((a) => {
                            const href = a.getAttribute('href') || '';
                            if (!href) return;
                            const tail = href.replace(/\\/$/, '').split('/').pop() || '';
                            const looksLikeExpose =
                                (/-wohnung-/.test(href) || /wohnung-/.test(href)) && /\\d/.test(tail);
                            if (!looksLikeExpose) return;

                            const absolute = href.startsWith('http')
                                ? href
                                : new URL(href, url).href;
                            if (!/immobilie1\\.de/.test(absolute)) return;
                            hrefs.add(absolute.split('?')[0]);
                        });

                        return { url, links: Array.from(hrefs), status: res.status, error: null };
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

            if not links:
                break

            await asyncio.sleep(0.6)
    finally:
        page_pbar.close()

    return sorted(all_links)


async def collect_state(page: Page, state: str) -> List[str]:
    base_url = BASE_URL.format(state=state) + "1"
    await page.goto(base_url)
    await page.wait_for_timeout(1200)
    await accept_cookies(page)
    return await collect_links_via_fetch(page, base_url, max_pages=PAGES_TO_FETCH)


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await browser.new_page()

        all_links: set[str] = set()
        per_state: Dict[str, List[str]] = {}

        for state in BUNDESLAND_SLUGS:
            print(f"\n>>> Scrape {state} ...", flush=True)
            try:
                links = await collect_state(page, state)
            except Exception as e:
                print(f"[WARN] {state} failed: {e}", flush=True)
                continue

            new_links = [l for l in links if l not in all_links]
            all_links.update(links)
            per_state[state] = links
            print(f"{state}: {len(links)} Links ({len(new_links)} neu), Gesamt {len(all_links)}", flush=True)

        payload = {
            "base_url_template": BASE_URL,
            "states": {k: {"count": len(v), "urls": v} for k, v in per_state.items()},
            "pages_requested": PAGES_TO_FETCH,
            "total_exposes": len(all_links),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "all_urls": sorted(all_links),
        }

        with open(OUTPUT_LINKS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"\n✔ Fertig. Insgesamt {len(all_links)} Exposés. Gespeichert unter {OUTPUT_LINKS_FILE}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
