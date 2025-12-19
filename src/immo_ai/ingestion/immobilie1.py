import asyncio
import gzip
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from immo_ai.io.jsonl import iter_json_items, open_jsonl_writer, write_jsonl_line
from immo_ai.io.paths import RunPaths, utc_now, write_json
from immo_ai.utils.http import (
    DEFAULT_BLOCK_KEYWORDS,
    HttpMetrics,
    PoliteHttpClient,
    RateLimiter,
    RETRYABLE_STATUSES,
    RetryConfig,
    RobotsCache,
)
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
    max_rpm: float | None = None
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout_seconds: float = 20.0
    respect_robots: bool = True
    user_agent: str = "immo-ai/0.0 (+contact: none)"
    block_keywords: tuple[str, ...] = DEFAULT_BLOCK_KEYWORDS


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


def _min_interval_seconds(config: CrawlConfig) -> float:
    if config.max_rpm:
        return max(60.0 / config.max_rpm, 0.0)
    return max(config.throttle_seconds, 0.0)


def _canonicalize_url(url: str) -> str:
    parsed = urlsplit(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(k, v) for k, v in query if not k.lower().startswith("utm_")]
    filtered.sort()
    new_query = urlencode(filtered, doseq=True)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, ""))


def _cache_body_path(cache_dir: Path, canonical_url: str) -> Path:
    digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.html.gz"


def _read_gzip_text(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read()


def _write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


async def _build_http_client(
    client: httpx.AsyncClient, config: CrawlConfig, metrics: HttpMetrics
) -> PoliteHttpClient:
    min_interval = _min_interval_seconds(config)
    rate_limiter = RateLimiter(min_interval=min_interval) if min_interval > 0 else None

    async def robots_fetcher(robots_url: str) -> str | None:
        host = urlsplit(robots_url).hostname or ""
        if rate_limiter is not None:
            await rate_limiter.acquire(host)
        resp = await client.get(robots_url, headers={"User-Agent": config.user_agent})
        metrics.total_requests += 1
        metrics.status_code_counts[resp.status_code] += 1
        if resp.status_code >= 400:
            return None
        return resp.text

    robots = RobotsCache(
        user_agent=config.user_agent,
        respect_robots=config.respect_robots,
        fetcher=robots_fetcher,
    )
    return PoliteHttpClient(
        client=client,
        user_agent=config.user_agent,
        retry=config.retry,
        metrics=metrics,
        rate_limiter=rate_limiter,
        robots=robots,
        block_keywords=config.block_keywords,
    )


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
    terminated_reason = "completed"
    http_metrics = HttpMetrics()

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        with open_jsonl_writer(rejects_path) as rejects_handle:
            http_client = await _build_http_client(client, config, http_metrics)
            for state in states:
                state_links: set[str] = set()
                page = 1
                while True:
                    if max_pages is not None and page > max_pages:
                        break
                    url = base_url_template.format(state=state, page=page)
                    html = ""
                    status_code: int | None = None
                    cached = False

                    if cache_dir:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        cached_path = _cache_path(cache_dir, url)
                        if cached_path.exists():
                            html = cached_path.read_text(encoding="utf-8")
                            status_code = 200
                            cached = True
                            http_metrics.cache_hits += 1
                        else:
                            http_metrics.cache_misses += 1

                    if not cached:
                        result = await http_client.get(url)
                        if result.disallowed_reason:
                            write_jsonl_line(
                                rejects_handle, {"url": url, "reason": result.disallowed_reason}
                            )
                            rejected += 1
                            wrote_rejects = True
                            break

                        if result.error:
                            write_jsonl_line(
                                rejects_handle,
                                {"url": url, "reason": f"request_error:{result.error}"},
                            )
                            rejected += 1
                            wrote_rejects = True
                            break

                        status_code = result.status_code
                        html = result.text or ""

                        if result.blocked:
                            reason = (
                                f"blocked_status:{status_code}" if status_code else "blocked_content"
                            )
                            write_jsonl_line(rejects_handle, {"url": url, "reason": reason})
                            rejected += 1
                            wrote_rejects = True
                            logger.warning("Blocked while fetching %s", url)
                            if http_metrics.block_events >= 3:
                                terminated_reason = "blocked"
                            break

                        if status_code in RETRYABLE_STATUSES:
                            rejected += 1
                            wrote_rejects = True
                            write_jsonl_line(
                                rejects_handle, {"url": url, "reason": f"http_status:{status_code}"}
                            )
                            break

                        if status_code and status_code >= 400:
                            rejected += 1
                            wrote_rejects = True
                            write_jsonl_line(
                                rejects_handle, {"url": url, "reason": f"http_status:{status_code}"}
                            )
                            break

                        if cache_dir:
                            _cache_path(cache_dir, url).write_text(html, encoding="utf-8")

                    links = _extract_links(html, url)
                    if not links:
                        break
                    state_links.update(links)
                    total_pages += 1
                    page += 1

                    if terminated_reason == "blocked":
                        break

                per_state[state] = sorted(state_links)
                total_links += len(state_links)
                if terminated_reason == "blocked":
                    break

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
    metrics.update(http_metrics.to_dict())
    write_json(run_paths.metrics_path, metrics)

    manifest = {
        "source": run_paths.source,
        "run_id": run_paths.run_id,
        "output": str(output_path),
        "rejects": str(rejects_path) if wrote_rejects else None,
        "metrics": str(run_paths.metrics_path),
        "generated_at": utc_now(),
        "terminated_reason": terminated_reason,
        "config": {
            "throttle_seconds": config.throttle_seconds,
            "max_rpm": config.max_rpm,
            "respect_robots": config.respect_robots,
            "user_agent": config.user_agent,
            "max_retries": config.retry.retries,
            "timeout_seconds": config.timeout_seconds,
        },
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
    terminated_reason = "completed"
    http_metrics = HttpMetrics()
    cache_dir = cache_dir or run_paths.cache_html_dir

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        with open_jsonl_writer(output_path) as output_handle, open_jsonl_writer(
            rejects_path
        ) as rejects_handle:
            http_client = await _build_http_client(client, config, http_metrics)
            for url in urls:
                canonical_url = _canonicalize_url(url)
                cached_path = _cache_body_path(cache_dir, canonical_url)
                if cached_path.exists():
                    html = _read_gzip_text(cached_path)
                    http_metrics.cache_hits += 1
                    write_jsonl_line(
                        output_handle,
                        {
                            "url": url,
                            "html": html,
                            "fetched_at": _utc_ts(),
                            "source": run_paths.source,
                        },
                    )
                    fetched += 1
                    continue

                http_metrics.cache_misses += 1
                result = await http_client.get(url)
                if result.disallowed_reason:
                    write_jsonl_line(rejects_handle, {"url": url, "reason": result.disallowed_reason})
                    rejected += 1
                    wrote_rejects = True
                    continue

                if result.error:
                    write_jsonl_line(
                        rejects_handle, {"url": url, "reason": f"request_error:{result.error}"}
                    )
                    rejected += 1
                    wrote_rejects = True
                    continue

                status_code = result.status_code or 0
                html = result.text or ""

                if result.blocked:
                    reason = f"blocked_status:{status_code}" if status_code else "blocked_content"
                    write_jsonl_line(rejects_handle, {"url": url, "reason": reason})
                    rejected += 1
                    wrote_rejects = True
                    logger.warning("Blocked while fetching %s", url)
                    if http_metrics.block_events >= 3:
                        terminated_reason = "blocked"
                        break
                    continue

                if status_code in RETRYABLE_STATUSES:
                    write_jsonl_line(
                        rejects_handle, {"url": url, "reason": f"http_status:{status_code}"}
                    )
                    rejected += 1
                    wrote_rejects = True
                    continue

                if status_code >= 400:
                    write_jsonl_line(
                        rejects_handle, {"url": url, "reason": f"http_status:{status_code}"}
                    )
                    rejected += 1
                    wrote_rejects = True
                    continue

                _write_gzip_text(cached_path, html)
                write_jsonl_line(
                    output_handle,
                    {"url": url, "html": html, "fetched_at": _utc_ts(), "source": run_paths.source},
                )
                fetched += 1

            if terminated_reason == "blocked":
                logger.warning("Terminated fetch-bodies due to repeated block signals.")

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
    metrics.update(http_metrics.to_dict())
    write_json(run_paths.metrics_path, metrics)

    manifest = {
        "source": run_paths.source,
        "run_id": run_paths.run_id,
        "input": str(links_path),
        "output": str(output_path),
        "rejects": str(rejects_path) if wrote_rejects else None,
        "metrics": str(run_paths.metrics_path),
        "generated_at": utc_now(),
        "terminated_reason": terminated_reason,
        "config": {
            "throttle_seconds": config.throttle_seconds,
            "max_rpm": config.max_rpm,
            "respect_robots": config.respect_robots,
            "user_agent": config.user_agent,
            "max_retries": config.retry.retries,
            "timeout_seconds": config.timeout_seconds,
        },
    }
    write_json(run_paths.manifest_path, manifest)

    return metrics
