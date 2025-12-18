import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse

from playwright.async_api import Page, async_playwright
from tqdm import tqdm

# =========================================
# CONFIG
# =========================================

# Wie viele Ergebnisseiten pro Price-Bucket scrapen (inkl. Seite 1)
# Sei hier konservativ (z.B. 20–50). Sehr hohe Werte erhöhen das Block-Risiko.
MAX_PAGES_PER_BUCKET: Optional[int] = None  # None für "bis keine Links mehr kommen"

# Wie viele Price-Buckets pro Lauf maximal verarbeiten
# (None = alle Buckets, was sehr viele Requests sein können)
MAX_BUCKETS_PER_RUN: Optional[int] = None

# Throttling / Anti-Block
PAGE_DELAY_SECONDS = 3.0        # Pause nach jeder Seite
BUCKET_PAUSE_SECONDS = 12.0     # Pause zwischen Buckets
BACKOFF_STATUS_CODES = {403}    # bei 403 länger warten und Seite erneut versuchen
BACKOFF_SECONDS = 180.0
MAX_BACKOFF_RETRIES = 1         # wie oft dieselbe Seite nach Backoff probieren

# User-Agents für Bucket-weise Context-Rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
]

# Price-Slicing-Konfiguration
PRICE_FIRST_MAX = 100_000       # erster Bucket: alles darunter
PRICE_STEP = 10_000             # Schrittweite für priceMax
PRICE_NO_LIMIT_FROM = 800_000   # ab hier ohne Obergrenze (letzter Bucket)

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "immowelt_expose_links.json"

# Basis-Such-URL – hier kannst du Region / Filter anpassen
BASE_URL = (
    "https://www.immowelt.de/classified-search?"
    "distributionTypes=Buy,Buy_Auction,Compulsory_Auction&"
    "estateTypes=Apartment&"
    "locations=AD02DE1&"  # z.B. ein Bundesland/Region
    "projectTypes=Resale&"
    "order=DateDesc&"
    "page=1"
)


# =========================================
# HELFERFUNKTIONEN
# =========================================

def _normalize_search_url(url: str) -> str:
    """
    Erzwingt Listenansicht (classified-search) und entfernt Map-Parameter wie bbox.
    Dadurch liefert fetch HTML mit Exposé-Links statt der Karten-Variante.
    """
    parsed = urlparse(url)
    path = parsed.path.replace("classified-map", "classified-search")
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs.pop("bbox", None)
    new_query = urlencode(qs, doseq=True)
    return parsed._replace(path=path, query=new_query).geturl()


def _make_page_url(base_url: str, page_number: int) -> str:
    parsed = urlparse(_normalize_search_url(base_url))
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    qs["page"] = str(page_number)
    new_query = urlencode(qs, doseq=True)
    return parsed._replace(query=new_query).geturl()


def _get_current_page_number(url: str) -> int:
    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    return int(qs.get("page", "1"))


def _make_bucket_url(
    base_url: str,
    price_min: Optional[int],
    price_max: Optional[int],
) -> str:
    """Erzeugt eine URL mit priceMin/priceMax-Parametern für einen Price-Bucket."""
    parsed = urlparse(_normalize_search_url(base_url))
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))

    if price_min is None:
        qs.pop("priceMin", None)
    else:
        qs["priceMin"] = str(price_min)

    if price_max is None:
        qs.pop("priceMax", None)
    else:
        qs["priceMax"] = str(price_max)

    qs["page"] = "1"  # jeder Bucket startet auf Seite 1
    new_query = urlencode(qs, doseq=True)
    return parsed._replace(query=new_query).geturl()


def _generate_price_buckets(
    first_max: int,
    step: int,
    open_end_from: int,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Erzeugt Price-Buckets:
      (None, first_max), (first_max, first_max+step), ..., (open_end_from, None)
    """
    buckets: List[Tuple[Optional[int], Optional[int]]] = []
    buckets.append((None, first_max))  # unter first_max

    current_min = first_max
    while current_min < open_end_from:
        buckets.append((current_min, current_min + step))
        current_min += step

    buckets.append((open_end_from, None))  # letzter Bucket: open-ended
    return buckets


async def accept_cookies(page: Page) -> bool:
    selectors = [
        "button#uc-btn-accept-banner",
        "button[aria-label='Alle akzeptieren']",
        "button[data-testid='uc-accept-all-button']",
        "button:text('OK')",
        "button:has-text('OK')",
    ]
    for sel in selectors:
        try:
            btn = await page.query_selector(sel)
            if btn:
                await btn.click()
                await asyncio.sleep(0.3)
                return True
        except Exception:
            continue
    return False


async def close_popups(page: Page) -> None:
    selectors = [
        "button[aria-label='Schließen']",
        "button[aria-label='close']",
        ".modal-close",
        ".iw-cookie-overlay button",
        "button[data-testid='close-button']",
        "button:text('Schließen')",
    ]
    for sel in selectors:
        try:
            btn = await page.query_selector(sel)
            if btn:
                await btn.click()
                await asyncio.sleep(0.15)
        except Exception:
            continue


# =========================================
# SCRAPING PRO BUCKET (nur via fetch)
# =========================================

async def collect_links_via_fetch(
    page: Page,
    bucket_url: str,
    max_pages: Optional[int] = None,
) -> List[str]:
    """
    Sammle Exposé-Links für einen Price-Bucket via window.fetch.
    Es wird NICHT page.goto benutzt, sondern nur fetch(...) im Browser-Kontext.
    """
    all_links: set[str] = set()

    start_page = _get_current_page_number(bucket_url)
    total_for_tqdm = max_pages if max_pages is not None else None

    page_pbar = tqdm(
        total=total_for_tqdm,
        desc="Pages",
        unit="page",
        file=sys.stdout,
        dynamic_ncols=True,
    )

    current_page = start_page
    backoff_retries = 0

    while True:
        # Seitenlimit pro Bucket?
        if max_pages is not None:
            pages_done = current_page - start_page + 1
            if pages_done > max_pages:
                break

        page_url = _make_page_url(bucket_url, current_page)

        page_result = await page.evaluate(
            """
            async (url) => {
                try {
                    const res = await fetch(url, {
                        method: 'GET',
                        credentials: 'include'
                    });
                    if (!res.ok) {
                        return {
                            url,
                            links: [],
                            status: res.status,
                            error: `HTTP ${res.status}`
                        };
                    }

                    const html = await res.text();
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');

                    const hrefs = new Set();
                    doc.querySelectorAll('a[href*="/expose/"]').forEach((a) => {
                        const href = a.getAttribute('href') || '';
                        if (!href) return;
                        const absolute = href.startsWith('http')
                            ? href
                            : new URL(href, url).href;
                        hrefs.add(absolute.split('?')[0]);
                    });

                    return {
                        url,
                        links: Array.from(hrefs),
                        status: res.status,
                        error: null
                    };
                } catch (e) {
                    return {
                        url,
                        links: [],
                        status: null,
                        error: String(e)
                    };
                }
            }
            """,
            page_url,
        )

        if not isinstance(page_result, dict):
            print(f"[WARN] Unerwartetes Ergebnis für {page_url}: {page_result}", flush=True)
            break

        status = page_result.get("status")
        error = page_result.get("error")

        if status in BACKOFF_STATUS_CODES:
            if backoff_retries >= MAX_BACKOFF_RETRIES:
                print(
                    f"[WARN] Status {status} bei {page_url} – Backoff-Limit erreicht, breche Bucket ab.",
                    flush=True,
                )
                break
            backoff_retries += 1
            print(
                f"[WARN] Status {status} bei {page_url} – warte {BACKOFF_SECONDS:.0f}s und versuche erneut.",
                flush=True,
            )
            await asyncio.sleep(BACKOFF_SECONDS)
            continue
        else:
            backoff_retries = 0

        if status and status != 200:
            print(f"[WARN] Status {status} bei {page_url} – {error}", flush=True)

        links = page_result.get("links", []) or []
        new_links = [l for l in links if l not in all_links]
        all_links.update(new_links)

        page_pbar.update(1)
        page_pbar.set_postfix(exposes=len(all_links))

        # Keine Links mehr -> Ende dieses Buckets (oder Captcha/Fehlerseite)
        if not links:
            print(f">>> Keine Links mehr auf Seite {current_page}, breche Bucket ab.", flush=True)
            break

        current_page += 1

        # kleine Pause, um nicht zu hart zu wirken
        await asyncio.sleep(PAGE_DELAY_SECONDS)

    page_pbar.close()
    return sorted(all_links)


# =========================================
# MAIN
# =========================================

async def main() -> None:
    async with async_playwright() as p:
        user_agent = random.choice(USER_AGENTS)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(user_agent=user_agent)
        page = await context.new_page()

        print(">>> Lade Basisseite...", flush=True)
        await page.goto(BASE_URL)
        await page.wait_for_timeout(1200)
        await accept_cookies(page)
        await close_popups(page)

        print(
            f">>> Verwende User-Agent (ein Fenster für alle Buckets): {user_agent}",
            flush=True,
        )

        print(
            "\nBitte im Browser:"
            "\n  - ggf. Captcha lösen"
            "\n  - weitere Filter/Region nach Wunsch einstellen"
            "\nWenn die Ergebnisliste so passt, komm zurück ins Terminal.",
            flush=True,
        )

        print("\n>>> WARTE auf Enter im Terminal...", flush=True)
        await asyncio.to_thread(input, "Press Enter when ready: ")

        print(">>> Enter erhalten, starte Price-Bucket-Scraping...", flush=True)

        base_search_url_raw = page.url
        base_search_url = _normalize_search_url(base_search_url_raw)
        if base_search_url != base_search_url_raw:
            print(
                f">>> Verwende Basissuche (normalisiert auf Listenansicht): {base_search_url}",
                flush=True,
            )
        else:
            print(f">>> Verwende Basissuche: {base_search_url}", flush=True)

        # Buckets erzeugen
        price_buckets = _generate_price_buckets(
            PRICE_FIRST_MAX, PRICE_STEP, PRICE_NO_LIMIT_FROM
        )

        if MAX_BUCKETS_PER_RUN is not None:
            price_buckets = price_buckets[:MAX_BUCKETS_PER_RUN]

        all_expose_links: set[str] = set()
        bucket_meta = []

        for idx, (price_min, price_max) in enumerate(price_buckets, start=1):
            bucket_label = f"{price_min or 0}-{price_max or 'max'}"
            bucket_url = _make_bucket_url(base_search_url, price_min, price_max)

            print(
                f"\n>>> [{idx}/{len(price_buckets)}] Starte Bucket {bucket_label} "
                f"(max {MAX_PAGES_PER_BUCKET or 'alle'} Seiten)",
                flush=True,
            )
            print(f">>> Bucket-URL: {bucket_url}", flush=True)

            await page.goto(bucket_url)
            await page.wait_for_timeout(1200)
            await accept_cookies(page)
            await close_popups(page)

            bucket_links = await collect_links_via_fetch(
                page,
                bucket_url,
                max_pages=MAX_PAGES_PER_BUCKET,
            )

            new_links = [l for l in bucket_links if l not in all_expose_links]
            all_expose_links.update(bucket_links)

            bucket_meta.append(
                {
                    "priceMin": price_min,
                    "priceMax": price_max,
                    "bucket_url": bucket_url,
                    "bucket_total_exposes": len(bucket_links),
                    "bucket_new_exposes": len(new_links),
                }
            )

            print(
                f"Bucket {bucket_label}: {len(bucket_links)} Exposés "
                f"({len(new_links)} neu, gesamt {len(all_expose_links)})",
                flush=True,
            )

            # Kurze Pause zwischen Buckets
            await asyncio.sleep(BUCKET_PAUSE_SECONDS)

        payload = {
            "base_search_url": base_search_url,
            "price_buckets": bucket_meta,
            "pages_requested_per_bucket": MAX_PAGES_PER_BUCKET,
            "max_buckets_per_run": MAX_BUCKETS_PER_RUN,
            "total_exposes": len(all_expose_links),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "expose_links": sorted(all_expose_links),
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2, ensure_ascii=False)

        print(f"\n✔ Fertig! Total gefundene Exposés: {len(all_expose_links)}")
        print(f"Links gespeichert in: {OUTPUT_FILE}", flush=True)

        await context.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
