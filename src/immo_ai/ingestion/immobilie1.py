import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from immo_ai.io.jsonl import iter_json_items, open_jsonl_writer, write_jsonl_line
from immo_ai.io.paths import RunPaths, utc_now, write_json
from immo_ai.utils.http import RetryConfig, backoff_sleep, is_blocked, polite_get
from immo_ai.utils.logging import get_logger

logger = get_logger(__name__)

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

EXPOSE_RE = re.compile(r"(wohnung-|\-wohnung-)\d+")


@dataclass
class CrawlConfig:
    throttle_seconds: float = 1.0
    retry: RetryConfig = RetryConfig()
    timeout_seconds: float = 20.0


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _looks_like_expose(url: str) -> bool:
    return bool(EXPOSE_RE.search(url))


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        absolute = href if href.startswith("http") else urljoin(base_url, href)
        if "immobilie1.de" not in absolute:
            continue
        if _looks_like_expose(absolute):
            links.append(absolute.split("?")[0])
    return sorted(set(links))


def _cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.html"


async def fetch_html_with_retries(
    client: httpx.AsyncClient,
    url: str,
    config: CrawlConfig,
    cache_dir: Path | None,
) -> tuple[str, int]:
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = _cache_path(cache_dir, url)
        if cached_path.exists():
            return cached_path.read_text(encoding="utf-8"), 200

    last_status = 0
    html = ""
    for attempt in range(config.retry.retries + 1):
        resp = await polite_get(client, url, config.throttle_seconds, config.retry)
        html = resp.text
        last_status = resp.status_code
        if last_status in {429, 503} and attempt < config.retry.retries:
            await backoff_sleep(config.retry, attempt)
            continue
        if cache_dir:
            _cache_path(cache_dir, url).write_text(html, encoding="utf-8")
        break
    return html, last_status


async def collect_links(
    run_paths: RunPaths,
    base_url_template: str,
    states: Iterable[str],
    max_pages: int | None,
    config: CrawlConfig,
    cache_dir: Path | None,
    output_path: Path,
    rejects_path: Path,
) -> dict[str, Any]:
    start = time.monotonic()
    total_pages = 0
    total_links = 0
    rejected = 0
    wrote_rejects = False
    per_state: dict[str, list[str]] = {}

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client, open_jsonl_writer(
        rejects_path
    ) as rejects_handle:
        for state in states:
            state_links: set[str] = set()
            page = 1
            while True:
                if max_pages is not None and page > max_pages:
                    break
                url = base_url_template.format(state=state, page=page)
                html, status_code = await fetch_html_with_retries(client, url, config, cache_dir)

                if is_blocked(status_code, html):
                    reason = f"blocked_status:{status_code}"
                    write_jsonl_line(rejects_handle, {"url": url, "reason": reason})
                    rejected += 1
                    wrote_rejects = True
                    logger.warning("Blocked while fetching %s", url)
                    break

                if status_code in {429, 503}:
                    rejected += 1
                    wrote_rejects = True
                    write_jsonl_line(
                        rejects_handle, {"url": url, "reason": f"http_status:{status_code}"}
                    )
                    break

                links = _extract_links(html, url)
                if not links:
                    break
                state_links.update(links)
                total_pages += 1
                page += 1

                if config.throttle_seconds > 0:
                    await asyncio.sleep(config.throttle_seconds)

            per_state[state] = sorted(state_links)
            total_links += len(state_links)

    payload = {
        "base_url_template": base_url_template,
        "states": {k: {"count": len(v), "urls": v} for k, v in per_state.items()},
        "pages_requested": max_pages,
        "total_exposes": total_links,
        "generated_at": _utc_ts(),
        "all_urls": sorted({u for urls in per_state.values() for u in urls}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if not wrote_rejects and rejects_path.exists():
        rejects_path.unlink()

    runtime = time.monotonic() - start
    metrics = {
        "source": run_paths.source,
        "run_id": run_paths.run_id,
        "total_pages": total_pages,
        "total_links": total_links,
        "rejected": rejected,
        "runtime_seconds": round(runtime, 3),
        "generated_at": utc_now(),
    }
    write_json(run_paths.metrics_path, metrics)

    manifest = {
        "source": run_paths.source,
        "run_id": run_paths.run_id,
        "output": str(output_path),
        "rejects": str(rejects_path) if wrote_rejects else None,
        "metrics": str(run_paths.metrics_path),
        "generated_at": utc_now(),
    }
    write_json(run_paths.manifest_path, manifest)

    return metrics


def _load_urls(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        urls = payload.get("all_urls") or payload.get("urls") or []
        if isinstance(urls, list):
            return [u for u in urls if isinstance(u, str)]
        return []
    urls = []
    for item in iter_json_items(path):
        url = item.get("url") or item.get("href")
        if url:
            urls.append(url)
    return urls


async def fetch_bodies(
    run_paths: RunPaths,
    links_path: Path,
    output_path: Path,
    rejects_path: Path,
    config: CrawlConfig,
    cache_dir: Path | None,
    limit: int | None,
) -> dict[str, Any]:
    start = time.monotonic()
    urls = _load_urls(links_path)
    if limit is not None:
        urls = urls[:limit]

    total = len(urls)
    fetched = 0
    rejected = 0
    wrote_rejects = False

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client, open_jsonl_writer(
        output_path
    ) as output_handle, open_jsonl_writer(rejects_path) as rejects_handle:
        for url in urls:
            html, status_code = await fetch_html_with_retries(client, url, config, cache_dir)

            if is_blocked(status_code, html):
                reason = f"blocked_status:{status_code}"
                write_jsonl_line(rejects_handle, {"url": url, "reason": reason})
                rejected += 1
                wrote_rejects = True
                logger.warning("Blocked while fetching %s", url)
                break

            if status_code in {429, 503}:
                write_jsonl_line(rejects_handle, {"url": url, "reason": f"http_status:{status_code}"})
                rejected += 1
                wrote_rejects = True
                continue

            if status_code >= 400:
                write_jsonl_line(rejects_handle, {"url": url, "reason": f"http_status:{status_code}"})
                rejected += 1
                wrote_rejects = True
                continue

            write_jsonl_line(output_handle, {"url": url, "html": html, "fetched_at": _utc_ts()})
            fetched += 1

            if config.throttle_seconds > 0:
                await asyncio.sleep(config.throttle_seconds)

    if not wrote_rejects and rejects_path.exists():
        rejects_path.unlink()

    runtime = time.monotonic() - start
    metrics = {
        "source": run_paths.source,
        "run_id": run_paths.run_id,
        "input": str(links_path),
        "output": str(output_path),
        "total": total,
        "fetched": fetched,
        "rejected": rejected,
        "runtime_seconds": round(runtime, 3),
        "generated_at": utc_now(),
    }
    write_json(run_paths.metrics_path, metrics)

    manifest = {
        "source": run_paths.source,
        "run_id": run_paths.run_id,
        "input": str(links_path),
        "output": str(output_path),
        "rejects": str(rejects_path) if wrote_rejects else None,
        "metrics": str(run_paths.metrics_path),
        "generated_at": utc_now(),
    }
    write_json(run_paths.manifest_path, manifest)

    return metrics
