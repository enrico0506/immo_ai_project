import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin, urlparse as _urlparse, parse_qsl, urlencode, urlunparse

from bs4 import BeautifulSoup
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# OUTPUT: structured JSONL (chunked, max 1000 per file)
DEFAULT_OUT_PREFIX = "immoscout_structured_"
OUT_PREFIX = DEFAULT_OUT_PREFIX
OUT_RECORDS_PER_FILE = 1000

FAILED_JSONL = PROCESSED_DIR / "immoscout_structured_failed.jsonl"
PROGRESS_FILE = PROCESSED_DIR / "immoscout_structured_progress.json"
DONE_KEYS_FILE = PROCESSED_DIR / "immoscout_structured_done_keys.txt"

# Optional cap for testing; set to None for all
ANALYZE_LIMIT: Optional[int] = None
_ANALYZE_LIMIT_ENV = os.getenv("IS24_ANALYZE_LIMIT")
if _ANALYZE_LIMIT_ENV:
    ANALYZE_LIMIT = int(_ANALYZE_LIMIT_ENV)


def env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    s = raw.strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default


CHECKPOINT_EVERY = max(1, env_int("IS24_CHECKPOINT_EVERY", 1000))
POSTFIX_EVERY = max(1, env_int("IS24_POSTFIX_EVERY", 200))
FILE_BUFFER_SIZE = max(8_192, env_int("IS24_FILE_BUFFER_SIZE", 1_048_576))
DEFAULT_ANALYZE_MODE = os.getenv("IS24_ANALYZE_MODE", "sequential").strip().lower() or "sequential"
DEFAULT_ANALYZE_WORKERS = max(1, env_int("IS24_ANALYZE_WORKERS", 1))

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore


def json_loads(raw: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def json_dumps_jsonl(obj: Any) -> bytes:
    if orjson is not None:
        return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE)
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def sanitize_run_id(run_id: Optional[str]) -> str:
    s = (run_id or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    return s.lower()


INPUT_CHUNK_RE = re.compile(r"^(?P<prefix>.+?)(?P<idx>\d{4})\.jsonl$")


def infer_run_id_from_inputs(files: list[Path]) -> Optional[str]:
    """
    Infer a stable run id from chunked input filenames.
    Example: expose_analysis_run_20251225_205537_0001.jsonl -> run_20251225_205537
    """
    if not files:
        return None
    prefixes: list[str] = []
    for p in files:
        m = INPUT_CHUNK_RE.match(p.name)
        if not m:
            continue
        prefix = (m.group("prefix") or "").rstrip("_")
        if prefix:
            prefixes.append(prefix)
    if not prefixes:
        return None

    prefix = Counter(prefixes).most_common(1)[0][0]
    if prefix == "expose_analysis":
        return None
    if prefix.startswith("expose_analysis_"):
        prefix = prefix[len("expose_analysis_") :]
    rid = sanitize_run_id(prefix)
    return rid or None


def configure_run(run_id: Optional[str]) -> None:
    """
    Configure output prefix + state file names for a run.
    Keeps backward-compatible default names when `run_id` is empty.
    """
    global OUT_PREFIX, FAILED_JSONL, PROGRESS_FILE, DONE_KEYS_FILE
    rid = sanitize_run_id(run_id)
    OUT_PREFIX = DEFAULT_OUT_PREFIX if not rid else f"{DEFAULT_OUT_PREFIX}{rid}_"
    state_prefix = OUT_PREFIX.rstrip("_")
    FAILED_JSONL = PROCESSED_DIR / f"{state_prefix}_failed.jsonl"
    PROGRESS_FILE = PROCESSED_DIR / f"{state_prefix}_progress.json"
    DONE_KEYS_FILE = PROCESSED_DIR / f"{state_prefix}_done_keys.txt"


def backup_existing_outputs(ts: str) -> int:
    """
    Move existing progress/done/failed + output chunks aside (same directory) and return count.
    """
    moved = 0
    for p in [
        PROGRESS_FILE,
        DONE_KEYS_FILE,
        FAILED_JSONL,
        *sorted(PROCESSED_DIR.glob(f"{OUT_PREFIX}[0-9][0-9][0-9][0-9].jsonl")),
    ]:
        if not p.exists():
            continue
        dst = p.with_name(p.name + f".bak.{ts}")
        try:
            p.replace(dst)
            moved += 1
        except OSError:
            continue
    return moved


def parse_input_chunk_index(path: Path) -> Optional[int]:
    m = INPUT_CHUNK_RE.match(path.name)
    if not m:
        return None
    try:
        return int(m.group("idx"))
    except Exception:
        return None


def trim_invalid_jsonl_tail(path: Path) -> int:
    """
    If the last JSONL line is incomplete/invalid (e.g. killed mid-write), truncate it.
    Returns the number of lines removed (0 or more).
    """
    if not path.exists():
        return 0
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size <= 0:
        return 0

    removed = 0
    with open(path, "rb+") as f:
        while True:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size <= 0:
                break

            # Grow the window until we can capture the last non-empty line.
            window = min(size, 1_048_576)
            while True:
                f.seek(size - window)
                tail = f.read(window)
                if not tail:
                    return removed

                end = len(tail)
                if tail.endswith(b"\n"):
                    end -= 1
                if end <= 0:
                    # file ends with just a newline; truncate it
                    f.truncate(size - 1)
                    removed += 1
                    break

                nl = tail.rfind(b"\n", 0, end)
                start = nl + 1 if nl >= 0 else 0
                line = tail[start:end].strip()
                if line:
                    # Absolute byte offset of the line start
                    abs_line_start = (size - window) + start
                    abs_line_end = (size - window) + end
                    try:
                        json_loads(line)
                        return removed
                    except Exception:
                        # Truncate invalid last line
                        f.truncate(abs_line_start)
                        removed += 1
                        break

                # Line is empty, move to previous one; increase window if needed
                if window >= size:
                    return removed
                window = min(size, window * 2)
    return removed


def parse_record_to_output_line(
    rec: Dict[str, Any],
    in_file_name: str,
) -> Dict[str, Any]:
    url = rec.get("url") or ""
    canon = rec.get("canonical_url") or (canonicalize_url(url) if url else "")
    html = rec.get("html") or ""
    html_sha = rec.get("html_sha256") or body_hash(html)

    out: Dict[str, Any] = {
        "url": url,
        "canonical_url": canon or None,
        "parsed_at": utc_now(),
        "html_sha256": html_sha,
        "source_input_file": in_file_name,
        "source_has_html": bool(html),
        "source_worker": rec.get("worker"),
        "source_status": rec.get("status"),
        "source_final_url": rec.get("final_url"),
        "source_page_title": rec.get("page_title"),
        "source_skip_reason": rec.get("skip_reason"),
    }

    if not html:
        reason = rec.get("skip_reason") or "missing_html"
        out["skip_reason"] = reason
        analysis = rec.get("analysis") or {}
        jsonld_fields = extract_jsonld_fields(analysis.get("jsonld"))
        if jsonld_fields:
            minimal: dict[str, Any] = {
                "url": url,
                "expose_id": extract_expose_id(url) or extract_expose_id(canon),
                "title": analysis.get("expose_title") or analysis.get("h1") or analysis.get("title"),
                "page_canonical_url": analysis.get("canonical"),
                "address": analysis.get("address_text_guess"),
                "price_text_guess": analysis.get("price_text_guess"),
                "status_heading": analysis.get("status_heading"),
                "status_body": analysis.get("status_body"),
                "deactivated_notice": analysis.get("deactivated_notice"),
                "description": analysis.get("description"),
                "location_text": analysis.get("location_text"),
                "other": analysis.get("other"),
                "energy_efficiency_class": analysis.get("energy_efficiency_class"),
                "features": analysis.get("features"),
                "details": analysis.get("details"),
                "documents": analysis.get("documents"),
                **jsonld_fields,
            }
            out["data_source"] = "analysis_jsonld_only"
            out["data"] = minimal
        else:
            out["error"] = "missing_html"
        return out

    soup = make_soup(html)
    is_bad, bad_reason = detect_bad_body(soup)
    if is_bad:
        out["failed_reason"] = bad_reason or "bad_body"
        out["error"] = "bad_body"
        return out

    try:
        parsed = parse_item(url, soup)
        analysis = rec.get("analysis") or {}
        jsonld_fields = extract_jsonld_fields(analysis.get("jsonld"))
        if jsonld_fields:
            parsed.update(jsonld_fields)
        out["data"] = parsed
    except Exception as e:
        out["error"] = repr(e)

    return out


def process_one_input_file_job(
    in_path_str: str,
    out_dir_str: str,
    out_prefix: str,
    out_index: int,
    bs_parser: str,
) -> Dict[str, Any]:
    """
    Per-input-file mode:
    - output is one JSONL per input chunk index
    - resume is based on output line count (safe-trims invalid last JSON line)
    """
    global BS_PARSER
    BS_PARSER = select_bs_parser(bs_parser)

    in_path = Path(in_path_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_local = out_dir / f"{out_prefix}{out_index:04d}.jsonl"

    removed_tail = trim_invalid_jsonl_tail(out_path_local)
    already = count_lines(out_path_local)

    processed = 0
    skipped_existing = 0
    errors = 0
    t0 = time.time()

    with open(in_path, "rb") as fin, open(out_path_local, "ab", buffering=FILE_BUFFER_SIZE) as fout:
        # Skip already-processed records (1 output line == 1 input record)
        to_skip = already
        while to_skip > 0:
            raw = fin.readline()
            if not raw:
                break
            if raw.strip():
                to_skip -= 1
                skipped_existing += 1

        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json_loads(raw)
            except Exception as e:
                errors += 1
                fout.write(
                    json_dumps_jsonl(
                        {
                            "parsed_at": utc_now(),
                            "source_input_file": in_path.name,
                            "error": f"json_decode_error: {type(e).__name__}",
                        }
                    )
                )
                processed += 1
                continue

            if not isinstance(rec, dict):
                errors += 1
                fout.write(
                    json_dumps_jsonl(
                        {
                            "parsed_at": utc_now(),
                            "source_input_file": in_path.name,
                            "error": "json_record_not_object",
                        }
                    )
                )
                processed += 1
                continue

            out_line = parse_record_to_output_line(rec, in_file_name=in_path.name)
            if "error" in out_line:
                errors += 1
            fout.write(json_dumps_jsonl(out_line))
            processed += 1

    dt = time.time() - t0
    return {
        "input_file": in_path.name,
        "out_file": out_path_local.name,
        "out_index": out_index,
        "bs_parser": BS_PARSER,
        "trimmed_tail_lines": removed_tail,
        "resumed_lines": already,
        "skipped_existing": skipped_existing,
        "processed": processed,
        "errors": errors,
        "seconds": dt,
    }


def select_bs_parser(preferred: str) -> str:
    """
    Pick a BeautifulSoup parser once (avoid per-record try/except).
    """
    preferred = (preferred or "").strip() or "lxml"
    try:
        BeautifulSoup("", preferred)
        return preferred
    except Exception:
        return "html.parser"


# Preferred parser for BeautifulSoup (faster than the built-in html.parser) when installed.
BS_PARSER = select_bs_parser(os.getenv("IS24_BS_PARSER", "lxml"))


# ============================================================
# URL CANONICALIZATION (match your collector)
# ============================================================
def canonicalize_url(url: str) -> str:
    parsed = _urlparse(url)
    qs = list(parse_qsl(parsed.query, keep_blank_values=True))

    drop_prefixes = ("utm_",)
    drop_keys = {"referrer", "ref", "icmp"}

    kept = [
        (k, v)
        for (k, v) in qs
        if not (k in drop_keys or any(k.startswith(p) for p in drop_prefixes))
    ]
    new_query = urlencode(kept, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


# ============================================================
# TEXT HELPERS
# ============================================================
def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def clean(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = " ".join(text.split()).strip()
    return t or None


def slugify(label: str) -> str:
    label = unicodedata.normalize("NFKD", label)
    label = label.encode("ascii", "ignore").decode("ascii")
    label = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
    return label.lower() or "field"


def get_text(soup: BeautifulSoup, selector: str) -> Optional[str]:
    el = soup.select_one(selector)
    if not el:
        return None
    return clean(el.get_text(" ", strip=True))


def get_energy_eff_class(soup: BeautifulSoup) -> Optional[str]:
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


# ============================================================
# BAD BODY DETECTION
# ============================================================
STRONG_MARKERS = (
    "datadome",
    "perimeterx",
)
CAPTCHA_WIDGET_MARKERS = (
    "g-recaptcha",
    "hcaptcha",
    "cf-turnstile",
    "turnstile",
)
WEAK_MARKERS = (
    "captcha",
    "access denied",
    "unusual traffic",
    "verify you are human",
    "blocked",
    "forbidden",
    "robot",
    "security check",
    "challenge",
    "cloudflare",
    "incapsula",
)

SMALL_HTML_LEN = 30_000


def body_hash(html: str) -> str:
    return hashlib.sha256((html or "").encode("utf-8")).hexdigest()


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", BS_PARSER)


DATE_TOKEN_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
NUMBER_TOKEN_RE = re.compile(
    r"[-+]?\d{1,3}(?:[.\u00A0\s]\d{3})+(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?"
)


def extract_numberish_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = value.replace("\u00A0", " ")

    # Preserve full dates like "01.01.2026"
    dm = DATE_TOKEN_RE.search(text)
    if dm:
        return dm.group(0)

    matches = list(NUMBER_TOKEN_RE.finditer(text))
    if not matches:
        return None

    def is_letter(ch: str) -> bool:
        return bool(ch) and ch.isalpha()

    # Prefer tokens not directly glued to letters (e.g. ignore "24" in "ImmoScout24")
    for m in matches:
        start, end = m.span()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if is_letter(before) or is_letter(after):
            continue
        return m.group(0)

    return None


def parse_number_token(token: Optional[str]) -> Optional[int | float]:
    s = (token or "").strip()
    if not s:
        return None

    s = s.replace("\u00A0", "").replace(" ", "")

    # Skip date-like values (multiple dots, no comma) to avoid false parsing.
    if s.count(".") > 1 and "," not in s:
        return None

    # Normalize to a Python float string
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # 12.345,67 -> 12345.67
            s = s.replace(".", "").replace(",", ".")
        else:
            # 12,345.67 -> 12345.67
            s = s.replace(",", "")
    elif "," in s:
        # 12,5 -> 12.5
        s = s.replace(",", ".")
    elif "." in s:
        # Treat dots as thousands separators when they form groups of 3 digits.
        if re.match(r"^-?\d{1,3}(\.\d{3})+$", s):
            s = s.replace(".", "")

    try:
        num = float(s)
    except ValueError:
        return None

    return int(num) if num.is_integer() else num


def extract_expose_id(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    m = re.search(r"/expose/(\d+)", parsed.path)
    if m:
        return m.group(1)
    qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for k in ("exposeId", "exposeid", "expose_id"):
        v = qs.get(k)
        if v and v.isdigit():
            return v
    return None


def flatten_jsonld(jsonld: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(jsonld, dict):
        jsonld = [jsonld]
    if not isinstance(jsonld, list):
        return items

    for obj in jsonld:
        if isinstance(obj, dict) and isinstance(obj.get("@graph"), list):
            for g in obj["@graph"]:
                if isinstance(g, dict):
                    items.append(g)
            continue
        if isinstance(obj, dict):
            items.append(obj)
    return items


def extract_jsonld_fields(jsonld: Any) -> dict[str, Any]:
    """
    Extract a small, analysis-friendly subset from JSON-LD (if present).
    Uses `ld_` prefixes to avoid collisions with HTML-parsed fields.
    """
    items = flatten_jsonld(jsonld)
    if not items:
        return {}

    listing = next((x for x in items if x.get("@type") == "RealEstateListing"), None)
    webpage = next((x for x in items if x.get("@type") == "WebPage"), None)

    out: dict[str, Any] = {}

    if webpage:
        out["ld_webpage_url"] = webpage.get("url") or webpage.get("@id")
        out["ld_webpage_name"] = webpage.get("name")
        out["ld_webpage_description"] = webpage.get("description")

    if listing:
        out["ld_listing_url"] = listing.get("url")
        out["ld_listing_name"] = listing.get("name")
        out["ld_date_posted"] = listing.get("datePosted")

        provider = listing.get("provider")
        if isinstance(provider, dict):
            out["ld_provider_name"] = provider.get("name")

        offers = listing.get("offers")
        if isinstance(offers, dict):
            out["ld_offer_price"] = offers.get("price")
            out["ld_offer_currency"] = offers.get("priceCurrency")
            out["ld_offer_availability"] = offers.get("availability")
        elif isinstance(offers, list) and offers:
            first = offers[0]
            if isinstance(first, dict):
                out["ld_offer_price"] = first.get("price")
                out["ld_offer_currency"] = first.get("priceCurrency")
                out["ld_offer_availability"] = first.get("availability")

        image = listing.get("image")
        if isinstance(image, str):
            out["ld_images"] = [image]
        elif isinstance(image, list):
            out["ld_images"] = [x for x in image if isinstance(x, str)] or None

    return {k: v for k, v in out.items() if v is not None}


def detect_bad_body(soup: BeautifulSoup) -> Tuple[bool, Optional[str]]:
    # User-requested policy: only flag as failed when the page is explicitly the
    # "Ich bin kein Roboter" challenge (via title or h1).
    if soup is None:
        return False, None
    title = clean(soup.title.get_text(" ", strip=True)) if soup.title else None
    h1 = clean(soup.select_one("h1").get_text(" ", strip=True)) if soup.select_one("h1") else None

    if (title or "").strip().lower() == "ich bin kein roboter":
        return True, "robot_challenge_page"
    if (h1 or "").strip().lower() == "ich bin kein roboter":
        return True, "robot_challenge_page"

    return False, None


def load_failed_done_keys() -> Set[str]:
    if not FAILED_JSONL.exists():
        return set()
    keys: Set[str] = set()
    for rec in iter_jsonl_records(FAILED_JSONL):
        html_sha = rec.get("html_sha256") or ""
        if not html_sha:
            continue
        key_base = rec.get("canonical_url") or rec.get("url") or ""
        if not key_base:
            continue
        keys.add(make_done_key(key_base, html_sha))
    return keys


def load_success_done_keys_from_outputs() -> Set[str]:
    keys: Set[str] = set()
    for p in sorted(PROCESSED_DIR.glob(f"{OUT_PREFIX}[0-9][0-9][0-9][0-9].jsonl")):
        for rec in iter_jsonl_records(p):
            if "data" not in rec:
                continue
            html_sha = rec.get("html_sha256") or ""
            if not html_sha:
                continue
            key_base = rec.get("canonical_url") or rec.get("url") or ""
            if not key_base:
                continue
            keys.add(make_done_key(key_base, html_sha))
    return keys


# ============================================================
# PARSING
# ============================================================
def parse_item(url: str, soup: BeautifulSoup) -> dict:
    expose_id = extract_expose_id(url)

    base: Dict[str, Any] = {
        "url": url,
        "expose_id": expose_id,
        "title": get_text(soup, "#expose-title") or get_text(soup, "h1"),
        "address": get_text(soup, '[data-qa="is24-expose-address"]'),
        "description": get_text(soup, ".is24qa-objektbeschreibung"),
        "location_text": get_text(soup, ".is24qa-lage"),
        "other": get_text(soup, ".is24qa-sonstiges"),
        "energy_efficiency_class": get_energy_eff_class(soup),
        "features": get_boolean_features(soup) or None,
    }

    dynamic = extract_dt_dd_pairs(soup)

    docs_raw = [a.get("href") for a in soup.select("#is24-ex-floorplans a[href]")]
    docs = [urljoin(url, h) for h in docs_raw if h]
    base["documents"] = sorted(set(docs)) or None

    image_urls: list[str] = []
    for img in soup.find_all("img", src=True):
        src = img.get("src") or ""
        if src.startswith("data:"):
            continue
        abs_url = urljoin(url, src)
        image_urls.append(abs_url)

    unique_images: list[str] = []
    seen_images: set[str] = set()
    for u in image_urls:
        if u in seen_images:
            continue
        seen_images.add(u)
        unique_images.append(u)

    image_hashes = [hashlib.sha256(u.encode("utf-8")).hexdigest() for u in unique_images]
    base["images"] = unique_images or None
    base["image_hashes"] = image_hashes or None

    for k, v in list(dynamic.items()):
        token = extract_numberish_token(v)
        if not token:
            continue
        dynamic[k + "_raw"] = v
        dynamic[k] = token
        parsed_num = parse_number_token(token)
        if parsed_num is not None:
            dynamic[k + "_num"] = parsed_num

    return {**base, **dynamic}


# ============================================================
# JSONL IO + RESUME
# ============================================================
def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json_loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec


def load_done_keys() -> Set[str]:
    done: Set[str] = set()
    if not DONE_KEYS_FILE.exists():
        return done
    with open(DONE_KEYS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            k = line.strip()
            if k:
                done.add(k)
    return done


class DoneKeyWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = open(path, "a", encoding="utf-8", buffering=FILE_BUFFER_SIZE)
        self._pending = 0

    def write(self, key: str) -> None:
        self._fh.write(key + "\n")
        self._pending += 1

    def flush(self) -> None:
        if self._pending:
            self._fh.flush()
            self._pending = 0

    def close(self) -> None:
        self.flush()
        self._fh.close()


def load_progress() -> Dict[str, Any]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "generated_at": utc_now(),
        "total_processed": 0,
        "out_file_index": 1,
        "out_records_in_file": 0,
    }


def save_progress(progress: Dict[str, Any]) -> None:
    progress["generated_at"] = utc_now()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def out_path(file_index: int) -> Path:
    return PROCESSED_DIR / f"{OUT_PREFIX}{file_index:04d}.jsonl"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


class OutputWriter:
    def __init__(self, progress: Dict[str, Any]) -> None:
        self.file_index = int(progress.get("out_file_index", 1))
        self.records_in_file = count_lines(out_path(self.file_index))
        if self.records_in_file >= OUT_RECORDS_PER_FILE:
            self.file_index += 1
            self.records_in_file = count_lines(out_path(self.file_index))
        self._fh = open(out_path(self.file_index), "ab", buffering=FILE_BUFFER_SIZE)

    def write_one(self, obj: Dict[str, Any]) -> None:
        if self.records_in_file >= OUT_RECORDS_PER_FILE:
            self._fh.close()
            self.file_index += 1
            self.records_in_file = 0
            self._fh = open(out_path(self.file_index), "ab", buffering=FILE_BUFFER_SIZE)
        self._fh.write(json_dumps_jsonl(obj))
        self.records_in_file += 1

    def sync_to_progress(self, progress: Dict[str, Any]) -> None:
        progress["out_file_index"] = self.file_index
        progress["out_records_in_file"] = self.records_in_file

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


class FailedWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = open(path, "ab", buffering=FILE_BUFFER_SIZE)
        self._pending = 0

    def write_one(self, obj: Dict[str, Any]) -> None:
        self._fh.write(json_dumps_jsonl(obj))
        self._pending += 1

    def flush(self) -> None:
        if self._pending:
            self._fh.flush()
            self._pending = 0

    def close(self) -> None:
        self.flush()
        self._fh.close()


def make_done_key(key_base: str, html_sha: str) -> str:
    return f"{key_base}|{html_sha}"


# ============================================================
# COUNT TOTAL INPUT LINES (for progress bar)
# ============================================================
def count_total_records(files: list[Path]) -> int:
    total = 0
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Immoscout expose bodies JSONL")
    parser.add_argument(
        "--input-file",
        action="append",
        help="Path to a specific expose_analysis_*.jsonl file to process (can be provided multiple times).",
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        help="Glob (relative to raw dir) for input files, e.g. 'expose_analysis_0339.jsonl' or 'expose_analysis_03*.jsonl'.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Run identifier for output/progress files. If omitted, inferred from the first input filename "
            "(or from env IS24_RUN_ID)."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch for this run-id by moving existing outputs/state to *.bak.<timestamp>.",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "per-file"],
        default=DEFAULT_ANALYZE_MODE if DEFAULT_ANALYZE_MODE in {"sequential", "per-file"} else "sequential",
        help=(
            "Processing mode. 'sequential' uses done-keys (URL|html_sha) resume across ALL inputs. "
            "'per-file' writes one output JSONL per input chunk index and resumes by output line count."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_ANALYZE_WORKERS,
        help="Concurrency for per-file mode (default from env IS24_ANALYZE_WORKERS).",
    )
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default=os.getenv("IS24_ANALYZE_EXECUTOR", "process").strip().lower() or "process",
        help="Executor type for per-file mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many records (overrides IS24_ANALYZE_LIMIT).",
    )
    return parser.parse_args()


def resolve_input_files(args: argparse.Namespace) -> list[Path]:
    files: set[Path] = set()

    if args.input_file:
        for p in args.input_file:
            files.add(Path(p).resolve())

    if args.input_glob:
        for g in args.input_glob:
            files.update((RAW_DIR / g).glob("**/*") if any(ch in g for ch in "*?[]") else [RAW_DIR / g])

    if not files:
        files.update(RAW_DIR.glob("expose_analysis_*.jsonl"))

    return sorted({f for f in files if f.exists()})


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    args = parse_args()
    input_files = resolve_input_files(args)
    if not input_files:
        raise FileNotFoundError(f"No input JSONL files found (checked args or {RAW_DIR}/expose_analysis_*.jsonl)")

    analyze_limit = ANALYZE_LIMIT
    if args.limit is not None:
        analyze_limit = args.limit

    run_id = args.run_id or os.getenv("IS24_RUN_ID") or infer_run_id_from_inputs(input_files)
    if args.fresh and not run_id:
        run_id = datetime.utcnow().strftime("fresh_%Y%m%d_%H%M%S")
    configure_run(run_id)
    if args.fresh:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        moved = backup_existing_outputs(ts)
        print(f"Fresh run enabled: moved {moved} file(s) to *.bak.{ts}")
    print(f"Run id: {run_id or '(default)'} | output prefix: {OUT_PREFIX!r}")
    print(f"State files: {PROGRESS_FILE.name}, {DONE_KEYS_FILE.name}, {FAILED_JSONL.name}")

    if args.mode == "per-file":
        workers = max(1, int(args.workers or 1))
        executor_kind = (args.executor or "process").strip().lower()
        if executor_kind not in {"process", "thread"}:
            executor_kind = "process"

        jobs: list[tuple[str, int]] = []
        fallback_idx = 1
        for p in input_files:
            idx = parse_input_chunk_index(p) or fallback_idx
            fallback_idx += 1
            jobs.append((str(p), idx))

        ExecutorCls = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
        print(f"Mode: per-file | executor: {executor_kind} | workers: {workers} | files: {len(jobs)}")
        pbar = tqdm(total=len(jobs), desc="Analyzing files", unit="file")
        try:
            if workers == 1:
                for in_path_str, idx in jobs:
                    res = process_one_input_file_job(
                        in_path_str=in_path_str,
                        out_dir_str=str(PROCESSED_DIR),
                        out_prefix=OUT_PREFIX,
                        out_index=idx,
                        bs_parser=BS_PARSER,
                    )
                    pbar.update(1)
                    pbar.set_postfix(
                        file=res.get("input_file"),
                        idx=res.get("out_index"),
                        processed=res.get("processed"),
                        resumed=res.get("resumed_lines"),
                        errors=res.get("errors"),
                    )
            else:
                ex = ExecutorCls(max_workers=workers)
                try:
                    futs = [
                        ex.submit(
                            process_one_input_file_job,
                            in_path_str,
                            str(PROCESSED_DIR),
                            OUT_PREFIX,
                            idx,
                            BS_PARSER,
                        )
                        for (in_path_str, idx) in jobs
                    ]
                    for fut in as_completed(futs):
                        res = fut.result()
                        pbar.update(1)
                        pbar.set_postfix(
                            file=res.get("input_file"),
                            idx=res.get("out_index"),
                            processed=res.get("processed"),
                            resumed=res.get("resumed_lines"),
                            errors=res.get("errors"),
                        )
                except KeyboardInterrupt:
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    ex.shutdown(wait=True, cancel_futures=False)
        finally:
            pbar.close()
        return

    progress = load_progress()
    done = load_done_keys()
    # Useful after improving `detect_bad_body`: re-run items previously flagged as failed
    # without deleting `immoscout_structured_done_keys.txt`.
    REPROCESS_FAILED = os.getenv("IS24_REPROCESS_FAILED", "1").strip().lower() not in {"0", "false", "no"}
    if REPROCESS_FAILED:
        failed_done = load_failed_done_keys()
        if failed_done:
            success_done = load_success_done_keys_from_outputs()
            to_reprocess = failed_done - success_done
            if to_reprocess:
                done -= to_reprocess
            print(
                f"Failed done-keys: {len(failed_done)} | "
                f"already successful: {len(success_done)} | "
                f"will reprocess: {len(to_reprocess)}"
            )
    writer = OutputWriter(progress)
    done_key_writer = DoneKeyWriter(DONE_KEYS_FILE)
    failed_writer = FailedWriter(FAILED_JSONL)

    total_processed = int(progress.get("total_processed", 0))
    processed_this_run = 0
    skipped = 0
    skipped_no_html = 0
    failed = 0
    parse_errors = 0
    failed_reasons: Counter[str] = Counter()
    skipped_no_html_reasons: Counter[str] = Counter()
    parse_error_types: Counter[str] = Counter()

    print(f"Input files: {len(input_files)} (processing sequentially, no pre-count)")
    print(f"Already done keys: {len(done)} ({DONE_KEYS_FILE.name})")
    print(f"Output chunking: {OUT_PREFIX}####.jsonl (max {OUT_RECORDS_PER_FILE}/file)")
    print(
        "Note: output files are chunked by record count, not by input file. "
        "It's normal to finish many input JSONLs but only see a few output chunks."
    )
    print(f"Failed list: {FAILED_JSONL.name}")
    print(f"Continuing output at: {out_path(writer.file_index).name} ({writer.records_in_file}/{OUT_RECORDS_PER_FILE})")
    print(f"Parser: {BS_PARSER} | checkpoint every: {CHECKPOINT_EVERY} | postfix every: {POSTFIX_EVERY}")

    pbar = tqdm(total=None, desc="Analyzing bodies", unit="body")
    processed_since_checkpoint = 0

    def tick() -> None:
        pbar.update(1)
        if pbar.n % POSTFIX_EVERY == 0:
            pbar.set_postfix(
                skipped=skipped,
                skipped_no_html=skipped_no_html,
                failed=failed,
                parsed=processed_this_run,
                errors=parse_errors,
            )

    def checkpoint(force: bool = False) -> None:
        nonlocal processed_since_checkpoint
        if (not force) and processed_since_checkpoint < CHECKPOINT_EVERY:
            return
        writer.sync_to_progress(progress)
        progress["total_processed"] = total_processed
        save_progress(progress)
        writer.flush()
        done_key_writer.flush()
        failed_writer.flush()
        processed_since_checkpoint = 0

    def mark_done_key(key: str) -> None:
        nonlocal total_processed, processed_this_run, file_processed, processed_since_checkpoint
        done_key_writer.write(key)
        done.add(key)
        total_processed += 1
        processed_this_run += 1
        file_processed += 1
        processed_since_checkpoint += 1
        checkpoint(force=False)

    try:
        for in_file in input_files:
            file_seen = 0
            file_processed = 0
            file_failed = 0
            file_parse_errors = 0
            file_skipped = 0
            file_skipped_no_html = 0
            for rec in iter_jsonl_records(in_file):
                # stop for limit (if set)
                if analyze_limit is not None and processed_this_run >= analyze_limit:
                    break

                file_seen += 1

                url = rec.get("url") or ""
                canon = rec.get("canonical_url") or (canonicalize_url(url) if url else "")
                html = rec.get("html") or ""
                html_sha = rec.get("html_sha256") or body_hash(html)
                key_base = canon or url

                if not key_base:
                    # still advance bar to show activity
                    tick()
                    continue

                done_key = make_done_key(key_base, html_sha)
                if done_key in done:
                    skipped += 1
                    file_skipped += 1
                    tick()
                    continue

                if not html:
                    reason = rec.get("skip_reason") or "missing_html"
                    if reason in {"deactivated", "expose_deactivated"}:
                        skipped_no_html += 1
                        file_skipped_no_html += 1
                        skipped_no_html_reasons[reason] += 1
                        mark_done_key(done_key)
                        tick()
                        continue

                    # If no HTML was stored, still try to salvage key fields from `analysis.jsonld`
                    analysis = rec.get("analysis") or {}
                    jsonld_fields = extract_jsonld_fields(analysis.get("jsonld"))
                    if jsonld_fields:
                        minimal: dict[str, Any] = {
                            "url": url,
                            "expose_id": extract_expose_id(url) or extract_expose_id(canon),
                            "title": analysis.get("expose_title") or analysis.get("h1") or analysis.get("title"),
                            "page_canonical_url": analysis.get("canonical"),
                            "address": analysis.get("address_text_guess"),
                            "price_text_guess": analysis.get("price_text_guess"),
                            "status_heading": analysis.get("status_heading"),
                            "status_body": analysis.get("status_body"),
                            "deactivated_notice": analysis.get("deactivated_notice"),
                            "description": analysis.get("description"),
                            "location_text": analysis.get("location_text"),
                            "other": analysis.get("other"),
                            "energy_efficiency_class": analysis.get("energy_efficiency_class"),
                            "features": analysis.get("features"),
                            "details": analysis.get("details"),
                            "documents": analysis.get("documents"),
                            **jsonld_fields,
                        }
                        writer.write_one({
                            "url": url,
                            "canonical_url": canon or None,
                            "parsed_at": utc_now(),
                            "html_sha256": html_sha,
                            "source_input_file": in_file.name,
                            "source_has_html": False,
                            "source_worker": rec.get("worker"),
                            "source_status": rec.get("status"),
                            "source_final_url": rec.get("final_url"),
                            "source_page_title": rec.get("page_title"),
                            "source_skip_reason": rec.get("skip_reason"),
                            "data_source": "analysis_jsonld_only",
                            "data": minimal,
                        })
                        mark_done_key(done_key)
                        tick()
                        continue

                    failed += 1
                    file_failed += 1
                    failed_reasons[reason] += 1
                    failed_writer.write_one({
                        "url": url,
                        "canonical_url": canon or None,
                        "reason": reason,
                        "html_len": 0,
                        "html_sha256": html_sha,
                        "source_input_file": in_file.name,
                        "source_status": rec.get("status"),
                        "source_final_url": rec.get("final_url"),
                        "skip_reason": rec.get("skip_reason"),
                        "detected_at": utc_now(),
                    })
                    mark_done_key(done_key)
                    tick()
                    continue

                # Detect bad body
                soup = make_soup(html)
                is_bad, reason = detect_bad_body(soup)
                if is_bad:
                    failed += 1
                    file_failed += 1
                    failed_reasons[reason or "unknown"] += 1
                    failed_writer.write_one({
                        "url": url,
                        "canonical_url": canon or None,
                        "reason": reason,
                        "html_len": len(html) if html is not None else 0,
                        "html_sha256": html_sha,
                        "source_input_file": in_file.name,
                        "source_status": rec.get("status"),
                        "source_final_url": rec.get("final_url"),
                        "detected_at": utc_now(),
                    })
                    mark_done_key(done_key)
                    tick()
                    continue

                # Parse good body
                try:
                    parsed = parse_item(url, soup)
                    analysis = rec.get("analysis") or {}
                    jsonld_fields = extract_jsonld_fields(analysis.get("jsonld"))
                    if jsonld_fields:
                        parsed.update(jsonld_fields)
                    writer.write_one({
                        "url": url,
                        "canonical_url": canon or None,
                        "parsed_at": utc_now(),
                        "html_sha256": html_sha,
                        "source_input_file": in_file.name,
                        "source_has_html": True,
                        "source_worker": rec.get("worker"),
                        "source_status": rec.get("status"),
                        "source_final_url": rec.get("final_url"),
                        "source_page_title": rec.get("page_title"),
                        "source_skip_reason": rec.get("skip_reason"),
                        "data": parsed,
                    })
                except Exception as e:
                    parse_errors += 1
                    file_parse_errors += 1
                    parse_error_types[type(e).__name__] += 1
                    writer.write_one({
                        "url": url,
                        "canonical_url": canon or None,
                        "parsed_at": utc_now(),
                        "html_sha256": html_sha,
                        "source_input_file": in_file.name,
                        "source_has_html": True,
                        "error": repr(e),
                    })
                mark_done_key(done_key)
                tick()

            checkpoint(force=True)
            print(
                f"Finished {in_file.name}: seen={file_seen}, processed={file_processed}, "
                f"failed={file_failed}, parse_errors={file_parse_errors}, skipped={file_skipped}, skipped_no_html={file_skipped_no_html}"
            )

            if analyze_limit is not None and processed_this_run >= analyze_limit:
                break

    finally:
        try:
            checkpoint(force=True)
        except Exception:
            pass
        pbar.close()
        writer.close()
        done_key_writer.close()
        failed_writer.close()

    print("\nSummary")
    print(f"  processed this run: {processed_this_run}")
    print(f"  skipped (already done): {skipped}")
    print(f"  skipped (no html, expected e.g. deactivated): {skipped_no_html}")
    print(f"  failed bodies written: {failed} -> {FAILED_JSONL.name}")
    print(f"  parse errors written: {parse_errors}")
    if failed_reasons:
        top = ", ".join(f"{k}={v}" for k, v in failed_reasons.most_common(6))
        print(f"  failed reasons (top): {top}")
    if skipped_no_html_reasons:
        top = ", ".join(f"{k}={v}" for k, v in skipped_no_html_reasons.most_common(6))
        print(f"  skipped no-html reasons (top): {top}")
    if parse_error_types:
        top = ", ".join(f"{k}={v}" for k, v in parse_error_types.most_common(6))
        print(f"  parse error types (top): {top}")
    print(f"  total processed ever: {total_processed}")
    print(f"  output last file: {out_path(writer.file_index).name} ({writer.records_in_file}/{OUT_RECORDS_PER_FILE})")
    print(f"  progress: {PROGRESS_FILE.name}")
    print(f"  done keys: {DONE_KEYS_FILE.name}")


if __name__ == "__main__":
    main()
