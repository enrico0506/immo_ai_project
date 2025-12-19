import asyncio
import random
from dataclasses import dataclass

import httpx


@dataclass
class RetryConfig:
    retries: int = 2
    backoff_base: float = 0.8
    backoff_jitter: float = 0.3


def is_blocked(status_code: int | None, text: str | None) -> bool:
    if status_code in {403}:
        return True
    if text:
        lowered = text.lower()
        if "captcha" in lowered or "challenge" in lowered or "verify you are human" in lowered:
            return True
    return False


async def polite_get(
    client: httpx.AsyncClient,
    url: str,
    throttle_seconds: float,
    retry: RetryConfig,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(retry.retries + 1):
        try:
            resp = await client.get(url)
            return resp
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < retry.retries:
                await asyncio.sleep(_backoff_delay(retry, attempt))
    if last_exc:
        raise last_exc
    raise httpx.HTTPError("Failed to fetch URL")


async def backoff_sleep(retry: RetryConfig, attempt: int) -> None:
    await asyncio.sleep(_backoff_delay(retry, attempt))


def _backoff_delay(retry: RetryConfig, attempt: int) -> float:
    base = retry.backoff_base * (2**attempt)
    jitter = random.uniform(0, retry.backoff_jitter)
    return base + jitter
