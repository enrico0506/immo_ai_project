import concurrent.futures
import json
import math
import os
import random
import re
import time
from urllib.parse import urlparse, parse_qs

import requests
from playwright.sync_api import sync_playwright, TimeoutError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    RAW_DIR, "regionalimmobilien24_expose_links_by_region.json"
)
DUMP_PREFIX = os.path.join(SCRIPT_DIR, "regionalimmobilien24_page_")

# Anzahl paralleler Worker für requests (Seiten >= 2)
WORKER_COUNT = 30

# Turn this on if you want per-page HTML dumps
DEBUG_DUMPS = False

# Alle Kauf- oder Miet-Wohnungs-Exposés (auch imXXXX IDs):
# https://www.regionalimmobilien24.de/<region-slug>/(kaufen|mieten)/wohnung/(im)?<id>/
EXPOSE_RE = re.compile(
    r"https://www\.regionalimmobilien24\.de/[a-z0-9\-]+/(?:kaufen|mieten)/wohnung/(?:im)?\d+/?",
    re.IGNORECASE,
)

# Gesamtanzahl Eigentumswohnungen, z.B.:
#   "<span>1.675 Eigentumswohnungen in Oberbayern</span>"
#   "In Oberbayern findest Du derzeit 1.675 Eigentumswohnungen."
TOTAL_RE_PATTERNS = [
    re.compile(r">\s*([\d\.]+)\s+Eigentumswohnungen in\b", re.IGNORECASE),
    re.compile(r"findest Du derzeit\s+([\d\.]+)\s+Eigentumswohnungen", re.IGNORECASE),
]

# Für jede Region: korrekte Slugs (aus JSON + /fromMap-Links abgeleitet)
REGION_SLUGS = {
    # Bayern
    "oberbayern": ["oberbayern"],
    "niederbayern": ["niederbayern"],
    "muenchen": ["muenchen"],
    "augsburg": ["augsburg"],
    "bayrisch-schwaben": ["bayerisch-schwaben"],
    "oberpfalz": ["oberpfalz"],
    "mittelfranken": ["mittelfranken"],
    "oberfranken": ["oberfranken"],
    "unterfranken": ["unterfranken"],

    # Baden-Württemberg
    "schwarzland": ["schwarzwald"],
    "bodensee": ["bodensee"],
    "oberschwaben": ["oberschwaben"],
    "schäbische alb": ["schwaebische-alb"],
    "rheintal": ["rheintal"],
    "kralsruhe": ["karlsruhe"],
    "stuttgart": ["stuttgart"],
    "hilbronn": ["heilbronn"],
    "rheinneckar": ["rheinneckar"],
    "mannheim": ["mannheim"],

    # Saarland
    "saarbrücken": ["saarbruecken"],
    "ostsaarland": ["ostsaarland"],
    "westsaarland": ["westsaarland"],
    "nordsaarland": ["nordsaarland"],

    # Rheinland-Pfalz
    "rheinpfalz": ["rheinpfalz"],
    "westpfalz": ["westpfalz"],
    "rheinhessen": ["rheinhessen"],
    "mainz": ["mainz"],
    "trier": ["trier"],
    "westeifel": ["westeifel"],
    "mittlerheim": ["mittelrhein"],
    "weserwald": ["westerwald"],

    # Hessen
    "sudhessen": ["suedhessen"],
    "frankfurt": ["frankfurt"],
    "taunus": ["taunus"],
    "wetterau": ["wetterau"],
    "mittelhessen": ["mittelhessen"],
    "kassel": ["kassel"],

    # Thüringen
    "ostthüringen": ["ostthueringen"],
    "mittelthüringen": ["mittelthueringen"],
    "südwestthürigen": ["suedwestthueringen"],
    "erfurt": ["erfurt"],
    "nordthüringen": ["nordthueringen"],

    # Sachsen
    "vogtland": ["vogtland"],
    "zwickau": ["zwickau"],
    "erzgebirge": ["erzgebirge"],
    "chemnitz": ["chemnitz"],
    "mittelsachsen": ["mittelsachsen"],
    "leipzig": ["leipzig"],
    "nordsachsen": ["nordsachsen"],
    "elbeland": ["elbeland"],
    "dresden": ["dresden"],
    "säsische schweiz osterzgebirge": ["saechsische-schweiz-osterzgebirge"],
    "oberlausitz": ["oberlausitz"],

    # Sachsen-Anhalt
    "saaleunstrut": ["saaleunstrut"],
    "halle": ["halle"],
    "anhalt": ["anhalt"],
    "salzland": ["salzland"],
    "harz": ["harz"],
    "megdeburg": ["magdeburg"],
    "jerichow": ["jerichow"],
    "altmark": ["altmark"],

    # Niedersachsen / Bremen / Hamburg
    "göttingen": ["goettingen"],
    "braunschweig": ["braunschweig"],
    "hannover": ["hannover"],
    "leineweser": ["leineweser"],
    "osnabrück": ["osnabrueck"],
    "emsland": ["emsland"],
    "weserems": ["weserems"],
    "bremen": ["bremen"],
    "lüneburg": ["lueneburg"],
    "hamburg": ["hamburg"],

    # NRW
    "aachen eifel": ["aachen-eifel"],
    "rhein siegkreis": ["rhein-siegkreis"],
    "rhein erft kreis": ["rhein-erft-kreis"],
    "köln": ["koeln"],
    "berisches land": ["bergisches-land"],
    "seigerland": ["siegerland"],
    "düsseldorf": ["duesseldorf"],
    "bergisches dreieck": ["bergisches-dreieck"],
    "sauerland": ["sauerland"],
    "dortmund essen": ["dortmund"],
    "duisburg": ["duisburg"],
    "niederrhein": ["niederrhein"],
    "ruhrgebiet": ["ruhrgebiet"],
    "muensterland": ["muensterland"],
    "teutoburger wald": ["teutoburger-wald"],

    # Brandenburg / Berlin
    "speerwald": ["spreewald"],
    "havelland": ["havelland"],
    "oberland": ["oderland"],
    "uckemark": ["uckermark"],
    "berlin": ["berlin"],
    "oberhavel": ["oberhavel"],

    # Mecklenburg-Vorpommern
    "südmecklenburg": ["suedmecklenburg"],
    "ludwigslust": ["ludwigslust"],
    "schwerin": ["schwerin"],
    "nordwestmecklenburg": ["nordwestmecklenburg"],
    "rostock": ["rostock"],
    "vorpommern": ["vorpommern"],

    # Schleswig-Holstein
    "lübeck": ["luebeck"],
    "holstein": ["holstein"],
    "kiel": ["kiel"],
    "schleswig": ["schleswig"],
}


def wait(min_s=0.1, max_s=0.3):
    time.sleep(random.uniform(min_s, max_s))


def dump_html(path: str, html: str):
    """Write HTML to disk only if DEBUG_DUMPS is enabled."""
    if not DEBUG_DUMPS:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def accept_cookies(page):
    selectors = [
        "button:has-text('Alle akzeptieren')",
        "button:has-text('Akzeptieren')",
        "button[aria-label='Alle akzeptieren']",
        "button[aria-label='Akzeptieren']",
    ]
    for s in selectors:
        try:
            btn = page.query_selector(s)
            if btn:
                btn.click()
                wait(0.1, 0.2)
                return True
        except Exception:
            pass
    return False


def auto_scroll(page):
    page.evaluate(
        """
        () => new Promise(resolve => {
            let total = 0;
            const distance = 400;
            const timer = setInterval(() => {
                window.scrollBy(0, distance);
                total += distance;
                if (total >= document.body.scrollHeight) {
                    clearInterval(timer);
                    resolve();
                }
            }, 200);
        })
        """
    )


def extract_expose_links_from_html(html: str):
    # Direkt aus dem HTML (inkl. immolist.push, shariff data-url, etc.)
    matches = EXPOSE_RE.findall(html)
    cleaned = set()
    for url in matches:
        url = url.split("#")[0].split("?")[0]
        cleaned.add(url)
    return cleaned


def extract_total_count(html: str):
    """
    Holt die Gesamtanzahl der Eigentumswohnungen aus dem HTML.
    Rückgabe: int oder None.
    """
    for pattern in TOTAL_RE_PATTERNS:
        m = pattern.search(html)
        if m:
            raw = m.group(1)
            # z.B. "1.675" -> 1675
            num = raw.replace(".", "").replace(" ", "")
            try:
                return int(num)
            except ValueError:
                pass
    return None


def detect_max_page(page, html: str):
    """
    Ermittelt die höchste Seitenzahl aus den Paging-Buttons.
    Nutzt data-p Attribute, klassische ?p= Links und Regex-Fallback auf dem HTML.
    """
    nums = set()

    # Paging-Buttons mit data-p
    try:
        nums.update(
            page.eval_on_selector_all(
                "a.paging_btn",
                "els => els.map(e => e.getAttribute('data-p')).filter(Boolean)",
            )
        )
    except Exception:
        pass

    # Fallback: klassische ?p= Links (falls vorhanden)
    try:
        nums.update(
            page.eval_on_selector_all(
                "a[href*='?p=']",
                "els => els.map(e => new URL(e.href).searchParams.get('p')).filter(Boolean)",
            )
        )
    except Exception:
        pass

    # Regex-Fallback auf HTML (falls JS nicht ausgewertet wurde)
    nums.update(re.findall(r'data-p="(\d+)"', html))

    as_ints = []
    for n in nums:
        try:
            as_ints.append(int(n))
        except Exception:
            pass
    return max(as_ints) if as_ints else 1


def wait_for_results(page, timeout=8000):
    """
    Warten bis mindestens ein Listing-Element geladen ist, damit nichts übersprungen wird.
    """
    try:
        page.wait_for_selector(".list-immoitem", timeout=timeout)
        return True
    except Exception:
        return False


def goto_with_retry(page, url, attempts=3, timeout_ms=45000, wait_until="networkidle"):
    """
    Robust navigation helper to reduce Page.goto timeouts.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return page.goto(url, wait_until=wait_until, timeout=timeout_ms)
        except TimeoutError as exc:
            last_exc = exc
            print(f"      Timeout beim Laden ({attempt}/{attempts}) -> retry")
        except Exception as exc:
            last_exc = exc
            print(f"      Fehler beim Laden ({attempt}/{attempts}): {exc}")
        wait(0.5, 1.0)
    if last_exc:
        raise last_exc


def build_page_url(base_url, page_num):
    parsed = urlparse(base_url)
    qs = parse_qs(parsed.query)
    qs["p"] = [str(page_num)]
    parts = []
    for k, vals in qs.items():
        for v in vals:
            parts.append(f"{k}={v}")
    query_str = "&".join(parts)
    return parsed._replace(query=query_str).geturl()


def build_start_url(region_slug):
    # nur kaufen/wohnung
    return (
        f"https://www.regionalimmobilien24.de/"
        f"{region_slug}/kaufen/wohnung/-/-/-/?rd=0&p=1"
    )


def get_requests_cookies_from_page(page):
    """
    Holt Cookies aus dem Playwright-Context, um sie mit requests zu verwenden.
    """
    state = page.context.storage_state()
    cookies = {}
    for c in state.get("cookies", []):
        # Nur relevante Domain mitnehmen
        if "regionalimmobilien24.de" in c.get("domain", ""):
            cookies[c["name"]] = c["value"]
    return cookies


def fetch_page_with_requests(url, cookies, headers, slug, page_num):
    """
    Holt eine Ergebnisseite (>=2) via requests.
    """
    try:
        resp = requests.get(url, cookies=cookies, headers=headers, timeout=20)
        if resp.status_code >= 400:
            print(
                f"      requests: HTTP {resp.status_code} auf Seite {page_num} ({slug}) -> überspringe."
            )
            return set()
        html = resp.text
        dump_path = f"{DUMP_PREFIX}{slug}_p{page_num}.html"
        dump_html(dump_path, html)
        links = extract_expose_links_from_html(html)
        print(f"      (requests) Seite {page_num}: +{len(links)} Links")
        return links
    except Exception as exc:
        print(f"      Fehler mit requests auf Seite {page_num}: {exc}")
        return set()


def scrape_all_regions():
    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_navigation_timeout(45000)

        # Einmal auf die Startseite, um Cookies loszuwerden
        print("Starte Browser und akzeptiere Cookies auf der Startseite ...")
        page.goto("https://www.regionalimmobilien24.de", wait_until="networkidle")
        wait(0.5, 1.0)
        accept_cookies(page)

        # User-Agent für requests übernehmen
        user_agent = page.evaluate("() => navigator.userAgent")

        for region_name, slug_candidates in REGION_SLUGS.items():
            print("=" * 80)
            print(f"Region: {region_name}")
            print(f"  Teste Slugs: {slug_candidates}")

            used_slug = None
            region_links = set()
            total_expected = None  # Gesamtanzahl aus Text

            for slug in slug_candidates:
                start_url = build_start_url(slug)
                print(f"  -> Versuche Slug '{slug}' mit URL: {start_url}")

                response = goto_with_retry(page, start_url)
                wait(0.1, 0.3)
                accept_cookies(page)
                wait_for_results(page)
                auto_scroll(page)
                wait(0.1, 0.3)

                if not response or response.status >= 400:
                    print(
                        f"    Antwort-Status {getattr(response, 'status', 'unknown')} -> überspringe."
                    )
                    continue

                # HTML von Seite 1
                html = page.content()

                dump_path = f"{DUMP_PREFIX}{slug}_p1.html"
                dump_html(dump_path, html)
                if DEBUG_DUMPS:
                    print(f"    Debug-HTML gespeichert: {dump_path}")

                first_links = extract_expose_links_from_html(html)
                print(f"    Gefundene Exposés auf Seite 1: {len(first_links)}")

                if not first_links:
                    # Keine Exposés -> probiere nächsten Slug
                    continue

                # Gesamtanzahl der Eigentumswohnungen aus HTML ziehen (Doublecheck-Basis)
                total_expected = extract_total_count(html)
                print(f"    Gesamtanzahl laut HTML-Text: {total_expected}")

                # Diese Schreibweise funktioniert -> alle Seiten für diese Region scrapen
                used_slug = slug
                region_links.update(first_links)

                per_page = max(1, len(first_links))  # echte Treffer auf Seite 1
                max_page_detected = detect_max_page(page, html)
                max_page_expected = None
                if total_expected and per_page:
                    max_page_expected = math.ceil(total_expected / per_page)
                if max_page_expected:
                    max_page = min(max_page_detected, max_page_expected)
                else:
                    max_page = max_page_detected
                print(f"    Max. Seiten für {slug}: {max_page}")

                # Cookies & Header aus Playwright holen, damit requests die gleiche Session nutzt
                cookies = get_requests_cookies_from_page(page)
                headers = {
                    "User-Agent": user_agent,
                    "Referer": start_url,
                    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
                }

                # Seiten 2..N per requests, aber jetzt PARALLEL mit mehreren Workern
                if max_page >= 2:
                    futures = []
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=WORKER_COUNT
                    ) as executor:
                        for pg in range(2, max_page + 1):
                            page_url = build_page_url(start_url, pg)
                            print(f"    -> Seite {pg} (requests): {page_url}")
                            futures.append(
                                executor.submit(
                                    fetch_page_with_requests,
                                    page_url,
                                    cookies,
                                    headers,
                                    slug,
                                    pg,
                                )
                            )

                        for fut in concurrent.futures.as_completed(futures):
                            links_pg = fut.result() or set()
                            region_links.update(links_pg)

                # Wir haben eine funktionierende Schreibweise -> nicht weiter versuchen
                break

            total_found = len(region_links)
            print(
                f"  => Region '{region_name}': {total_found} Exposés insgesamt gesammelt."
            )

            matches_total = None
            if total_expected is not None:
                matches_total = total_found == total_expected
                cmp = "==" if matches_total else "!="
                print(
                    f"  => Doublecheck: gefundene Links ({total_found}) {cmp} Gesamtanzahl ({total_expected})"
                )
            else:
                print(
                    "  => Gesamtanzahl konnte aus HTML nicht ermittelt werden (kein Match im Text)."
                )

            results[region_name] = {
                "tested_slugs": slug_candidates,
                "used_slug": used_slug,
                "total_expected": total_expected,
                "total_found": total_found,
                "matches_total": matches_total,
                "links": sorted(region_links),
            }

            if used_slug is None:
                print(f"  => KEINE Exposés gefunden für Region '{region_name}'.")
            else:
                print(
                    f"  => Region '{region_name}': {total_found} Exposés, Slug: '{used_slug}'"
                )

        browser.close()

    # Alles in eine gemeinsame JSON-Datei schreiben
    with open(OUTPUT_FILE, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

    print(f"Fertig. Ergebnisse gespeichert in {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape_all_regions()
