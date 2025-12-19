import asyncio
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx


RETRYABLE_STATUSES = {429, 502, 503, 504}
DEFAULT_BLOCK_KEYWORDS = ("captcha", "verify you are human")


@dataclass
class RetryConfig:
    retries: int = 3
    backoff_base: float = 1.0
    backoff_factor: float = 2.0
    backoff_jitter: float = 0.25
    max_backoff: float = 60.0
    max_retry_after: float = 120.0


@dataclass
class HttpMetrics:
    total_requests: int = 0
    status_code_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    robots_disallowed_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries_attempted: int = 0
    backoff_sleep_seconds_total: float = 0.0
    block_events: int = 0

    def to_dict(self) -> dict[str, float | int | dict[int, int]]:
        return {
            "total_requests": self.total_requests,
            "status_code_counts": dict(self.status_code_counts),
            "robots_disallowed_count": self.robots_disallowed_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "retries_attempted": self.retries_attempted,
            "backoff_sleep_seconds_total": round(self.backoff_sleep_seconds_total, 3),
            "block_events": self.block_events,
        }


@dataclass
class FetchResult:
    url: str
    status_code: int | None
    text: str | None
    blocked: bool = False
    disallowed_reason: str | None = None
    error: str | None = None
    retries: int = 0


class RobotsCache:
    def __init__(
        self,
        user_agent: str,
        respect_robots: bool = True,
        fetcher: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self._fetcher = fetcher
        self._cache: dict[str, RobotFileParser] = {}

    async def can_fetch(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlsplit(url)
        host = parsed.netloc
        if not host:
            return True
        parser = self._cache.get(host)
        if parser is None:
            parser = await self._load_parser(parsed.scheme, host)
            self._cache[host] = parser
        return parser.can_fetch(self.user_agent, url)

    async def _load_parser(self, scheme: str, host: str) -> RobotFileParser:
        robots_url = urlunsplit((scheme or "https", host, "/robots.txt", "", ""))
        parser = RobotFileParser()
        parser.set_url(robots_url)
        if self._fetcher is None:
            try:
                parser.read()
                return parser
            except Exception:
                parser.parse([])
                return parser
        try:
            content = await self._fetcher(robots_url)
        except Exception:
            content = None
        if content:
            parser.parse(content.splitlines())
        else:
            parser.parse([])
        return parser


@dataclass
class RateLimiter:
    min_interval: float
    time_func: Callable[[], float] = time.monotonic
    sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep
    _last_request: dict[str, float] = field(default_factory=dict)

    async def acquire(self, host: str) -> None:
        if self.min_interval <= 0:
            return
        now = self.time_func()
        last = self._last_request.get(host)
        if last is not None:
            wait = self.min_interval - (now - last)
            if wait > 0:
                await self.sleep_func(wait)
                now = self.time_func()
        self._last_request[host] = now


def is_blocked(status_code: int | None, text: str | None, keywords: Iterable[str]) -> bool:
    if status_code in {403}:
        return True
    if text:
        lowered = text.lower()
        for keyword in keywords:
            if keyword in lowered:
                return True
    return False


class PoliteHttpClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str,
        retry: RetryConfig,
        metrics: HttpMetrics,
        rate_limiter: RateLimiter | None = None,
        robots: RobotsCache | None = None,
        block_keywords: Iterable[str] = DEFAULT_BLOCK_KEYWORDS,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._retry = retry
        self._metrics = metrics
        self._rate_limiter = rate_limiter
        self._robots = robots
        self._block_keywords = tuple(block_keywords)
        self._sleep = sleep_func

    async def get(self, url: str) -> FetchResult:
        if self._robots is not None:
            allowed = await self._robots.can_fetch(url)
            if not allowed:
                self._metrics.robots_disallowed_count += 1
                return FetchResult(url=url, status_code=None, text=None, disallowed_reason="robots_disallowed")

        host = urlsplit(url).hostname or ""
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire(host)

        headers = {"User-Agent": self._user_agent}
        last_error: str | None = None
        for attempt in range(self._retry.retries + 1):
            try:
                resp = await self._client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                if attempt < self._retry.retries:
                    delay = self._backoff_delay(attempt)
                    self._metrics.retries_attempted += 1
                    self._metrics.backoff_sleep_seconds_total += delay
                    await self._sleep(delay)
                    continue
                return FetchResult(url=url, status_code=None, text=None, error=last_error, retries=attempt)

            self._metrics.total_requests += 1
            self._metrics.status_code_counts[resp.status_code] += 1
            text = resp.text

            if resp.status_code in RETRYABLE_STATUSES and attempt < self._retry.retries:
                delay = self._retry_delay(resp, attempt)
                self._metrics.retries_attempted += 1
                self._metrics.backoff_sleep_seconds_total += delay
                await self._sleep(delay)
                continue

            blocked = is_blocked(resp.status_code, text, self._block_keywords)
            if blocked:
                self._metrics.block_events += 1

            if resp.status_code == 429 and attempt >= self._retry.retries:
                self._metrics.block_events += 1
                blocked = True

            return FetchResult(
                url=url,
                status_code=resp.status_code,
                text=text,
                blocked=blocked,
                retries=attempt,
            )

        return FetchResult(url=url, status_code=None, text=None, error=last_error or "failed")

    def _backoff_delay(self, attempt: int) -> float:
        base = self._retry.backoff_base * (self._retry.backoff_factor**attempt)
        jitter = random.uniform(0, self._retry.backoff_jitter)
        return min(base + jitter, self._retry.max_backoff)

    def _retry_delay(self, resp: httpx.Response, attempt: int) -> float:
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        if retry_after is not None:
            return min(retry_after, self._retry.max_retry_after)
        return self._backoff_delay(attempt)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds >= 0:
            return seconds
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    now = parsed.__class__.now(parsed.tzinfo)
    delta = (parsed - now).total_seconds()
    if delta < 0:
        return None
    return delta
