import asyncio
import gzip
import json
from pathlib import Path
from unittest import mock

from immo_ai.ingestion.immobilie1 import (
    CrawlConfig,
    _cache_body_path,
    _canonicalize_url,
    fetch_bodies,
)
from immo_ai.io.paths import build_run_paths
from immo_ai.utils.http import HttpMetrics, PoliteHttpClient, RateLimiter, RetryConfig, RobotsCache


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get(self, url: str, headers: dict[str, str] | None = None):
        self.calls.append(url)
        raise AssertionError("network fetch should not be called")


def test_robots_disallowed_skips_request() -> None:
    metrics = HttpMetrics()

    async def robots_fetcher(_: str) -> str:
        return "User-agent: *\nDisallow: /private\n"

    robots = RobotsCache(user_agent="immo-ai-test", respect_robots=True, fetcher=robots_fetcher)
    fake_client = FakeHttpClient()
    polite = PoliteHttpClient(
        client=fake_client,  # type: ignore[arg-type]
        user_agent="immo-ai-test",
        retry=RetryConfig(retries=0),
        metrics=metrics,
        robots=robots,
        rate_limiter=None,
    )

    result = asyncio.run(polite.get("https://example.com/private/listing"))
    assert result.disallowed_reason == "robots_disallowed"
    assert metrics.robots_disallowed_count == 1
    assert fake_client.calls == []


def test_rate_limiter_respects_min_interval() -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.slept: list[float] = []

        def time(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.slept.append(seconds)
            self.now += seconds

    clock = FakeClock()
    limiter = RateLimiter(min_interval=1.0, time_func=clock.time, sleep_func=clock.sleep)

    async def run() -> None:
        await limiter.acquire("example.com")
        await limiter.acquire("example.com")

    asyncio.run(run())
    assert clock.slept == [1.0]


def test_cache_hit_skips_network(tmp_path: Path) -> None:
    run_paths = build_run_paths(tmp_path, "immobilie1", "run123")
    url = "https://example.com/expose?utm_source=ads"
    canonical = _canonicalize_url(url)
    cache_path = _cache_body_path(run_paths.cache_html_dir, canonical)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8") as handle:
        handle.write("<html>cached</html>")

    async def fake_builder(_client, _config, _metrics):
        return FakeHttpClient()

    output_path = run_paths.raw_dir / "bodies.jsonl.gz"
    rejects_path = run_paths.raw_dir / "rejects.jsonl.gz"
    links_path = run_paths.raw_dir / "links.json"
    links_path.write_text(json.dumps({"all_urls": [url]}), encoding="utf-8")

    with mock.patch(
        "immo_ai.ingestion.immobilie1._build_http_client", new=fake_builder
    ):
        metrics = asyncio.run(
            fetch_bodies(
                run_paths=run_paths,
                links_path=links_path,
                output_path=output_path,
                rejects_path=rejects_path,
                config=CrawlConfig(user_agent="immo-ai-test"),
                cache_dir=None,
                limit=None,
            )
        )

    assert metrics["cache_hits"] == 1
