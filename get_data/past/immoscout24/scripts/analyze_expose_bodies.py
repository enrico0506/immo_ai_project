import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup


SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = RAW_DIR / "expose_bodies.json"
OUTPUT_FILE = PROCESSED_DIR / "immoscout_exposes_structured.json"
# Limit how many records to analyze; set to None to process all
ANALYZE_LIMIT: int | None = None


def clean(text: str | None) -> str | None:
    if text is None:
        return None
    return " ".join(text.split()).strip() or None


def slugify(label: str) -> str:
    label = unicodedata.normalize("NFKD", label)
    label = label.encode("ascii", "ignore").decode("ascii")
    label = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
    return label.lower() or "field"


def get_text(soup: BeautifulSoup, selector: str) -> str | None:
    el = soup.select_one(selector)
    if not el:
        return None
    return clean(el.get_text(" ", strip=True))


def get_energy_eff_class(soup: BeautifulSoup) -> str | None:
    dd = soup.select_one(".is24qa-energieeffizienzklasse")
    if not dd:
        return None
    img = dd.find("img")
    if img and img.get("alt"):
        return clean(img["alt"])
    return clean(dd.get_text(" ", strip=True))


def get_boolean_features(soup: BeautifulSoup) -> list[str]:
    container = soup.select_one(".boolean-listing")
    if not container:
        return []
    labels = container.select('[class*="is24qa-"][class*="-label"]')
    feats = [clean(lbl.get_text(" ", strip=True)) for lbl in labels]
    return sorted({f for f in feats if f})


def extract_dt_dd_pairs(soup: BeautifulSoup) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for dl in soup.select("dl"):
        dt = dl.find("dt")
        dd = dl.find("dd")
        if not dt or not dd:
            continue
        label = clean(dt.get_text(" ", strip=True))
        value = clean(dd.get_text(" ", strip=True))
        if not label or not value:
            continue
        key = slugify(label)
        if key not in pairs:
            pairs[key] = value
    return pairs


def parse_item(item: dict) -> dict:
    url = item.get("url", "")
    html = item.get("html", "") or ""
    expose_id = urlparse(url).path.rstrip("/").split("/")[-1] if url else None

    soup = BeautifulSoup(html, "html.parser")

    base = {
        "url": url,
        "expose_id": expose_id,
        "title": get_text(soup, "#expose-title"),
        "address": get_text(soup, '[data-qa="is24-expose-address"]'),
        "description": get_text(soup, ".is24qa-objektbeschreibung"),
        "location_text": get_text(soup, ".is24qa-lage"),
        "other": get_text(soup, ".is24qa-sonstiges"),
        "energy_efficiency_class": get_energy_eff_class(soup),
        "features": get_boolean_features(soup),
        "documents": [a["href"] for a in soup.select("#is24-ex-floorplans a[href]")] or None,
    }

    dynamic = extract_dt_dd_pairs(soup)

    # Image URL hashes for deduping; hash the URL string (no downloads)
    image_urls: list[str] = []
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        if not src:
            continue
        abs_url = urljoin(url, src)
        image_urls.append(abs_url)
    image_hashes = [hashlib.sha256(u.encode("utf-8")).hexdigest() for u in image_urls]
    base["images"] = image_urls or None
    base["image_hashes"] = image_hashes or None

    # Normalize some numeric-like fields
    for k, v in list(dynamic.items()):
        digits = re.sub(r"[^0-9.,-]", "", v)
        if digits and len(digits) >= 2:
            dynamic[k + "_raw"] = v
            dynamic[k] = digits

    combined = {**base, **dynamic}
    return combined


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("items", [])
    if ANALYZE_LIMIT is not None:
        items = items[:ANALYZE_LIMIT]
    print(f"Loaded {len(items)} records from {INPUT_FILE} (limit={ANALYZE_LIMIT})")

    parsed = [parse_item(item) for item in items]

    # Aggregate all keys encountered to show schema growth
    all_keys = set()
    for p in parsed:
        all_keys.update(p.keys())

    out = {
        "source_file": str(INPUT_FILE),
        "generated_at": payload.get("generated_at"),
        "total": len(parsed),
        "all_keys": sorted(all_keys),
        "items": parsed,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote structured data to {OUTPUT_FILE}")
    print(f"Keys seen: {len(all_keys)}")


if __name__ == "__main__":
    main()
