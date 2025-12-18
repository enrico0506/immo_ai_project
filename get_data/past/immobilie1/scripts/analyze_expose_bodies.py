import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR = RAW_DIR / "immobilie1_expose_bodies"
INPUT_FILE = RAW_DIR / "immobilie1_expose_bodies.json"  # legacy single-file dump

# Output is JSONL: one JSON object per line
OUTPUT_FILE = PROCESSED_DIR / "immobilie1_exposes_structured.jsonl"

# Set to an int while debugging; None processes all
ANALYZE_LIMIT: int | None = None

BS_PARSER = os.getenv("ANALYZE_BS_PARSER", "lxml")


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


def iter_items_from_text(text: str):
    """
    Robustly walk the items array; on decode error, jump to the next '{'.
    Yields one item (dict) at a time instead of building a full list.
    """
    match = re.search(r'"items"\s*:\s*\[', text)
    if not match:
        return

    idx = match.end()
    decoder = json.JSONDecoder()

    while idx < len(text):
        # skip whitespace/commas
        while idx < len(text) and text[idx] in " \n\r\t,":
            idx += 1
        if idx >= len(text) or text[idx] == "]":
            break

        try:
            obj, idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            nxt = text.find("{", idx + 1)
            if nxt == -1:
                break
            idx = nxt
            continue

        if isinstance(obj, dict):
            yield obj


def iter_items_from_file(path: Path, limit: int | None = None):
    """
    Stream items from an input JSON file.

    For well-formed JSON ({"items": [...]}) we use json.loads,
    otherwise we fall back to the robust text walker above.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")

    # First try normal JSON parsing
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None

    count = 0

    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        for obj in payload["items"]:
            if not isinstance(obj, dict):
                continue
            yield obj
            count += 1
            if limit is not None and count >= limit:
                return
        return

    # Fallback: walk the text to find the "items" array
    for obj in iter_items_from_text(raw):
        yield obj
        count += 1
        if limit is not None and count >= limit:
            return


def find_input_files() -> list[Path]:
    chunked = sorted(INPUT_DIR.glob("*.json")) if INPUT_DIR.exists() else []
    if chunked:
        return chunked
    return [INPUT_FILE] if INPUT_FILE.exists() else []


def parse_summary_cards(container: BeautifulSoup) -> dict[str, str]:
    grid = container.select_one("#expose-info-details")
    if not grid:
        return {}

    summary: dict[str, str] = {}
    for card in grid.find_all(recursive=False):
        divs = [d for d in card.find_all("div", recursive=False) if d.get_text(strip=True)]
        if len(divs) < 2:
            continue
        value = clean(divs[0].get_text(" ", strip=True))
        label = clean(divs[-1].get_text(" ", strip=True))
        if not label:
            continue
        key = slugify(label)
        if value is not None and key not in summary:
            summary[key] = value
    return summary


def parse_key_value_section(container: BeautifulSoup, heading_text: str) -> dict[str, str]:
    heading = container.find(
        lambda tag: tag.name in ("h2", "h3") and heading_text.lower() in tag.get_text(" ", strip=True).lower()
    )
    if not heading:
        return {}

    ul = heading.find_next("ul")
    if not ul:
        return {}

    pairs: dict[str, str] = {}
    for li in ul.find_all("li", recursive=False):
        divs = li.find_all("div")
        if len(divs) < 2:
            continue
        label = clean(divs[0].get_text(" ", strip=True))
        value = clean(divs[1].get_text(" ", strip=True))
        if not label or not value:
            continue
        key = slugify(label)
        if key not in pairs:
            pairs[key] = value
    return pairs


def extract_first_extended_text(container: BeautifulSoup) -> str | None:
    for div in container.find_all("div", id=re.compile(r"extendedText")):
        if div.get("id", "").endswith("-print"):
            continue
        text = clean(div.get_text(" ", strip=True))
        if text:
            return text
    return None


def extract_extended_text_after_heading(container: BeautifulSoup, heading_text: str) -> str | None:
    heading = container.find(
        lambda tag: tag.name in ("h2", "h3") and heading_text.lower() in tag.get_text(" ", strip=True).lower()
    )
    if not heading:
        return None
    div = heading.find_next("div", id=re.compile(r"extendedText"))
    if div and not div.get("id", "").endswith("-print"):
        return clean(div.get_text(" ", strip=True))
    return None


def extract_ids(container: BeautifulSoup) -> dict[str, str | None]:
    id_p = container.find("p", string=re.compile("immobilie1-ID", re.IGNORECASE))
    immobilie1_id = None
    anbieter_id = None
    if id_p:
        txt = clean(id_p.get_text(" ", strip=True)) or ""
        m1 = re.search(r"immobilie1-ID:\s*([0-9]+)", txt, re.IGNORECASE)
        if m1:
            immobilie1_id = m1.group(1)
        m2 = re.search(r"Anbieter-ID:\s*([A-Za-z0-9-]+)", txt, re.IGNORECASE)
        if m2:
            anbieter_id = m2.group(1)
    return {"immobilie1_id": immobilie1_id, "anbieter_id": anbieter_id}


def extract_features(container: BeautifulSoup) -> list[str] | None:
    chips = [clean(li.get_text(" ", strip=True)) for li in container.select("li.ol-element-bg")]
    chips = [c for c in chips if c]
    return sorted(set(chips)) or None


def extract_coordinates(html: str) -> dict[str, float] | None:
    lat_match = re.search(r"latitude\s*=\s*([0-9.+-]+)", html)
    lon_match = re.search(r"longitude\s*=\s*([0-9.+-]+)", html)
    if not lat_match or not lon_match:
        return None
    try:
        return {"latitude": float(lat_match.group(1)), "longitude": float(lon_match.group(1))}
    except ValueError:
        return None


def extract_images(soup: BeautifulSoup, base_url: str) -> tuple[list[str] | None, list[str] | None]:
    urls: list[str] = []
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        if not src:
            continue
        urls.append(urljoin(base_url, src))

    if not urls:
        return None, None

    unique = []
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        unique.append(u)

    hashes = [hashlib.sha256(u.encode("utf-8")).hexdigest() for u in unique]
    return unique, hashes


def parse_item(item: dict) -> dict:
    url = item.get("url") or ""
    html = item.get("html") or ""
    expose_id = None
    tail = urlparse(url).path.rstrip("/").split("-")[-1]
    if tail.isdigit():
        expose_id = tail

    try:
        soup = BeautifulSoup(html, BS_PARSER)
    except Exception:
        # Fallback if the requested parser is unavailable
        soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("div.w-full.max-w-full.lg\\:col-span-8.grow") or soup

    base = {
        "url": url,
        "expose_id": expose_id,
        "title": get_text(container, "h1"),
        "address": get_text(container, "div.flex.items-center.gap-4 p"),
        "description": extract_first_extended_text(container),
        "ausstattung_text": extract_extended_text_after_heading(container, "Ausstattung"),
        "lage_text": extract_extended_text_after_heading(container, "Lage"),
        "features": extract_features(container),
        "summary": parse_summary_cards(container),
        "costs": parse_key_value_section(container, "Kosten"),
        "details": parse_key_value_section(container, "Immobiliendetails"),
        "energy": parse_key_value_section(container, "Energie"),
        "coordinates": extract_coordinates(html),
    }

    base.update(extract_ids(container))

    images, hashes = extract_images(soup, url)
    base["images"] = images
    base["image_hashes"] = hashes

    return base


def main() -> None:
    """
    Stream-parse all input JSON files and write one JSON object per line
    to OUTPUT_FILE. Only a progress bar is shown on stderr.
    """
    input_files = find_input_files()
    if not input_files:
        print(f"No input files found in {INPUT_DIR} or {INPUT_FILE}", file=sys.stderr)
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for path in input_files:
            for item in tqdm(
                iter_items_from_file(path, ANALYZE_LIMIT),
                desc=f"Parse {path.name}",
                unit="item",
                file=sys.stderr,
            ):
                try:
                    parsed = parse_item(item)
                except Exception:
                    # Skip items that fail to parse, no logging to keep output clean
                    continue

                # Write one JSON object per line
                json.dump(parsed, out_f, ensure_ascii=False)
                out_f.write("\n")


if __name__ == "__main__":
    main()
