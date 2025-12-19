import re
import unicodedata
from typing import Iterable


def clean(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = " ".join(text.split()).strip()
    return cleaned or None


def slugify(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label)
    ascii_label = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_label).strip("_")
    return slug.lower() or "field"


def first_present(items: Iterable[str | None]) -> str | None:
    for item in items:
        if item:
            return item
    return None


def parse_float(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"[0-9]+(?:[\.,][0-9]+)?", text.replace(" ", ""))
    if not match:
        return None
    value = match.group(0).replace(".", "").replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(text: str | None) -> int | None:
    if not text:
        return None
    digits = re.search(r"[0-9]+", text.replace(" ", ""))
    if not digits:
        return None
    try:
        return int(digits.group(0))
    except ValueError:
        return None
