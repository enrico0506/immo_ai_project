import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from tqdm import tqdm

from immo_ai.io.jsonl import iter_json_items, open_jsonl_writer, write_jsonl_line
from immo_ai.io.paths import RunPaths, utc_now, write_json
from immo_ai.schemas.expose import ExposeRecord
from immo_ai.utils.logging import get_logger
from immo_ai.utils.text import clean, slugify

logger = get_logger(__name__)


def get_text(soup: BeautifulSoup, selector: str) -> str | None:
    el = soup.select_one(selector)
    if not el:
        return None
    return clean(el.get_text(" ", strip=True))


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
        lambda tag: tag.name in ("h2", "h3")
        and heading_text.lower() in tag.get_text(" ", strip=True).lower()
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
        lambda tag: tag.name in ("h2", "h3")
        and heading_text.lower() in tag.get_text(" ", strip=True).lower()
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
        match_immobilie = re.search(r"immobilie1-ID:\s*([0-9]+)", txt, re.IGNORECASE)
        if match_immobilie:
            immobilie1_id = match_immobilie.group(1)
        match_anbieter = re.search(r"Anbieter-ID:\s*([A-Za-z0-9-]+)", txt, re.IGNORECASE)
        if match_anbieter:
            anbieter_id = match_anbieter.group(1)
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
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        unique.append(url)

    hashes = [hashlib.sha256(url.encode("utf-8")).hexdigest() for url in unique]
    return unique, hashes


def parse_item(
    item: dict[str, Any],
    source: str,
    run_id: str | None,
    bs_parser: str,
) -> tuple[ExposeRecord | None, str | None]:
    url = item.get("url") or ""
    html = item.get("html") or ""
    if not html.strip():
        return None, "missing_html"

    expose_id = None
    tail = urlparse(url).path.rstrip("/").split("-")[-1]
    if tail.isdigit():
        expose_id = tail

    html_sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest() if html else None

    try:
        soup = BeautifulSoup(html, bs_parser)
    except Exception:
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

    base["source"] = source
    base["run_id"] = run_id
    base["fetched_at"] = item.get("fetched_at")
    base["html_sha256"] = html_sha256

    record = ExposeRecord(**base)
    return record, None


def iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        for candidate in sorted(path.rglob("*")):
            if candidate.suffix in {".json", ".jsonl", ".gz"}:
                yield candidate
        return
    yield path


def parse_files(
    input_path: Path,
    output_path: Path,
    rejects_path: Path,
    run_paths: RunPaths,
    source: str,
    run_id: str | None,
    limit: int | None,
    bs_parser: str,
) -> dict[str, Any]:
    start = time.monotonic()
    total = 0
    parsed = 0
    rejected = 0
    wrote_rejects = False

    with open_jsonl_writer(output_path) as output_handle, open_jsonl_writer(rejects_path) as rejects_handle:
        for file_path in iter_input_files(input_path):
            logger.info("Parsing %s", file_path)
            for item in tqdm(iter_json_items(file_path), desc=f"Parse {file_path.name}"):
                total += 1
                if limit is not None and total > limit:
                    break
                try:
                    record, reason = parse_item(item, source, run_id, bs_parser)
                except Exception as exc:
                    record = None
                    reason = f"parse_error:{exc}"
                if record is None:
                    rejected += 1
                    write_jsonl_line(rejects_handle, {"url": item.get("url"), "reason": reason})
                    wrote_rejects = True
                    continue
                write_jsonl_line(output_handle, record.model_dump())
                parsed += 1
            if limit is not None and total >= limit:
                break

    runtime = time.monotonic() - start
    if not wrote_rejects and rejects_path.exists():
        rejects_path.unlink()
    metrics = {
        "source": source,
        "run_id": run_id,
        "input": str(input_path),
        "output": str(output_path),
        "total": total,
        "parsed": parsed,
        "rejected": rejected,
        "runtime_seconds": round(runtime, 3),
        "generated_at": utc_now(),
    }
    write_json(run_paths.metrics_path, metrics)

    manifest = {
        "source": source,
        "run_id": run_id,
        "input": str(input_path),
        "output": str(output_path),
        "metrics": str(run_paths.metrics_path),
        "rejects": str(rejects_path) if wrote_rejects else None,
        "generated_at": utc_now(),
    }
    write_json(run_paths.manifest_path, manifest)

    return metrics
