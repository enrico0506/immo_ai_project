#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


ENRICHMENT_VERSION = 2


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def repo_root() -> Path:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
        return Path(out.decode().strip())
    except Exception:
        return Path(__file__).resolve().parents[3]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def human_duration(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "?"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Progress:
    def __init__(self, *, total: Optional[int], enabled: bool) -> None:
        self.total = total
        self.enabled = enabled and is_tty()
        self.start = time.time()
        self.last_draw = 0.0
        self.done = 0
        self._spinner = "|/-\\"
        self._spin_i = 0

    def update(self, *, scored: int, geocode_failed: int, overpass_failed: int) -> None:
        self.done += 1
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_draw < 0.2 and (self.total is None or self.done < self.total):
            return
        self.last_draw = now

        elapsed = now - self.start
        rate = self.done / max(0.001, elapsed)

        if self.total:
            pct = min(1.0, self.done / max(1, self.total))
            bar_w = 28
            filled = int(pct * bar_w)
            bar = "#" * filled + "-" * (bar_w - filled)
            eta = (self.total - self.done) / max(0.001, rate)
            msg = (
                f"\r[{bar}] {pct*100:5.1f}% {self.done}/{self.total} "
                f"scored={scored} geo_fail={geocode_failed} ov_fail={overpass_failed} "
                f"{rate:6.1f}/s ETA {human_duration(eta)}"
            )
        else:
            self._spin_i = (self._spin_i + 1) % len(self._spinner)
            spin = self._spinner[self._spin_i]
            msg = (
                f"\r{spin} processed={self.done} scored={scored} geo_fail={geocode_failed} "
                f"ov_fail={overpass_failed} {rate:6.1f}/s"
            )

        print(msg[:240], end="", flush=True)

    def finish(self) -> None:
        if self.enabled:
            print("", flush=True)


def http_post_form(url: str, form: dict[str, str], timeout_s: int) -> bytes:
    data = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        raise RuntimeError(f"HTTPError {getattr(e, 'code', '?')} {getattr(e, 'reason', '')} (bytes={len(body)})") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URLError: {e}") from e


_OVERPASS_SEM: Optional[threading.Semaphore] = None


def _overpass_call_guard():
    if _OVERPASS_SEM is None:
        return contextlib.nullcontext()
    return _OVERPASS_SEM


def _decode_preview(raw: bytes, limit: int = 220) -> str:
    try:
        s = raw.decode("utf-8", errors="replace")
    except Exception:
        s = repr(raw[:limit])
    s = s.replace("\n", "\\n")
    return s[:limit]


def overpass_query_json(overpass_url: str, ql: str, timeout_s: int, *, retries: int = 6) -> dict[str, Any]:
    """
    Robust Overpass query with retries/backoff.
    Under load Overpass may return HTML/plaintext errors (non-JSON); we retry those.
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        if attempt:
            sleep_s = min(30.0, (0.6 * (2 ** (attempt - 1))) + (0.05 * attempt))
            time.sleep(sleep_s)
        try:
            with _overpass_call_guard():
                raw = http_post_form(overpass_url, {"data": ql}, timeout_s=timeout_s)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as e:
                preview = _decode_preview(raw)
                raise RuntimeError(
                    f"Overpass returned non-JSON (bytes={len(raw)} preview={preview!r}): {e}"
                ) from e
        except Exception as e:
            last_err = e
            # retry on transient failures; final attempt re-raises
            if attempt >= retries:
                break
            continue
    raise RuntimeError(f"Overpass query failed after retries={retries}: {last_err}") from last_err


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def can_connect(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def ensure_local_overpass_running(repo: Path, overpass_url: str) -> bool:
    """
    Returns True if this function started Overpass and it should be stopped afterwards.
    """
    if not overpass_url.startswith("http://127.0.0.1:8080"):
        return False
    if can_connect("127.0.0.1", 8080):
        return False

    start = repo / "geodata" / "amensity" / "overpass_start.sh"
    if not start.exists():
        raise RuntimeError(f"Local Overpass is not running and start script missing: {start}")
    log(f"Local Overpass not running; starting via: {start}")
    try:
        subprocess.check_call([str(start)])
    except subprocess.CalledProcessError:
        # Common after crashes: stale unix socket file blocks dispatcher start.
        stop = repo / "geodata" / "amensity" / "overpass_stop.sh"
        if stop.exists():
            log(f"Overpass start failed; attempting cleanup via: {stop}")
            subprocess.call([str(stop)])
        subprocess.check_call([str(start)])
    time.sleep(0.25)
    if not can_connect("127.0.0.1", 8080, timeout_s=2.0):
        raise RuntimeError("Failed to start local Overpass (still cannot connect to 127.0.0.1:8080)")
    return True


def stop_local_overpass(repo: Path) -> None:
    stop = repo / "geodata" / "amensity" / "overpass_stop.sh"
    if stop.exists():
        subprocess.call([str(stop)])


def read_db_marker_hash(repo: Path) -> str:
    p = repo / "geodata" / "overpass" / "db" / ".import_complete"
    if not p.exists():
        return "no_db_marker"
    return sha256_text(p.read_text(encoding="utf-8", errors="replace"))


@dataclass(frozen=True)
class ParsedAddress:
    raw: str
    cleaned: str
    district: Optional[str]
    street: Optional[str]
    housenumber: Optional[str]
    postcode: Optional[str]
    city: Optional[str]


_RE_HIDDEN = re.compile(r"\bDie vollständige Adresse.*$", flags=re.IGNORECASE)
_RE_ZIP = re.compile(r"\b(\d{5})\b")
_RE_STREET_HN = re.compile(r"^(?P<street>.+?)\s+(?P<hn>\d+[a-zA-Z]?)$")


def parse_address(raw: str) -> ParsedAddress:
    cleaned = raw.strip()
    cleaned = _RE_HIDDEN.sub("", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    postcode = None
    m = _RE_ZIP.search(cleaned)
    if m:
        postcode = m.group(1)

    city = None
    if postcode:
        after = cleaned.split(postcode, 1)[1].strip(" ,")
        if after:
            city = after.split(",", 1)[0].strip()
            if not city:
                city = None

    district = None
    street = None
    housenumber = None

    before_zip = cleaned
    if postcode:
        before_zip = cleaned.split(postcode, 1)[0].strip(" ,")
    left = before_zip.split(",", 1)[0].strip()
    if left:
        if re.search(r"\d", left):
            mm = _RE_STREET_HN.match(left)
            if mm:
                street = mm.group("street").strip()
                housenumber = mm.group("hn").strip()
            else:
                street = left
        else:
            district = left

    return ParsedAddress(
        raw=raw,
        cleaned=cleaned,
        district=district or None,
        street=street or None,
        housenumber=housenumber or None,
        postcode=postcode or None,
        city=city or None,
    )


@dataclass
class GeoResult:
    lat: float
    lon: float
    method: str  # overpass_postcode | overpass_address | nominatim
    precision: str  # address | postcode | unknown
    meta: dict[str, Any]


def _mean_latlon(elements: list[dict[str, Any]]) -> Optional[tuple[float, float]]:
    lats: list[float] = []
    lons: list[float] = []
    for e in elements:
        if "lat" in e and "lon" in e:
            lats.append(float(e["lat"]))
            lons.append(float(e["lon"]))
        elif "center" in e and isinstance(e["center"], dict) and "lat" in e["center"] and "lon" in e["center"]:
            lats.append(float(e["center"]["lat"]))
            lons.append(float(e["center"]["lon"]))
    if not lats:
        return None
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def geocode_via_overpass(
    overpass_url: str,
    db_marker_hash: str,
    cache_dir: Path,
    addr: ParsedAddress,
    timeout_s: int,
    sample_limit: int,
    *,
    force: bool,
) -> Optional[GeoResult]:
    if not addr.postcode:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = sha256_text(json.dumps({"db": db_marker_hash, "addr": addr.cleaned}, ensure_ascii=False))
    cache_path = cache_dir / f"overpass_geocode_{cache_key}.json"
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return GeoResult(**cached)

    if addr.street and addr.housenumber:
        street_rx = re.sub(r"\s+", r"\\s+", re.escape(addr.street))
        ql = (
            f'[out:json][timeout:{timeout_s}];'
            f'(nwr["addr:postcode"="{addr.postcode}"]'
            f'["addr:street"~"^{street_rx}$",i]'
            f'["addr:housenumber"~"^{re.escape(addr.housenumber)}$",i];);'
            f"out center {1};"
        )
        try:
            data = overpass_query_json(overpass_url, ql, timeout_s=timeout_s + 30)
            elements = data.get("elements", [])
            mean = _mean_latlon(elements)
            if mean:
                res = GeoResult(
                    lat=mean[0],
                    lon=mean[1],
                    method="overpass_address",
                    precision="address",
                    meta={"matched": len(elements), "postcode": addr.postcode, "street": addr.street},
                )
                atomic_write_text(cache_path, json.dumps(res.__dict__, ensure_ascii=False))
                return res
        except Exception:
            pass

    filters = [f'["addr:postcode"="{addr.postcode}"]', '["addr:country"="DE"]']
    if addr.city:
        city_rx = re.sub(r"\s+", r"\\s+", re.escape(addr.city))
        filters.append(f'["addr:city"~"^{city_rx}$",i]')

    ql = f'[out:json][timeout:{timeout_s}];node{ "".join(filters) };out body {int(sample_limit)};'
    data = overpass_query_json(overpass_url, ql, timeout_s=timeout_s + 60)
    elements = data.get("elements", [])
    mean = _mean_latlon(elements)
    if not mean:
        return None
    res = GeoResult(
        lat=mean[0],
        lon=mean[1],
        method="overpass_postcode",
        precision="postcode",
        meta={"sampled": len(elements), "postcode": addr.postcode, "city": addr.city},
    )
    atomic_write_text(cache_path, json.dumps(res.__dict__, ensure_ascii=False))
    return res


_NOMINATIM_LAST_CALL = 0.0
_NOMINATIM_LOCK = threading.Lock()


def geocode_via_nominatim(
    cache_dir: Path,
    addr: ParsedAddress,
    timeout_s: int,
    *,
    force: bool,
    min_interval_s: float = 1.0,
) -> Optional[GeoResult]:
    global _NOMINATIM_LAST_CALL
    if not addr.cleaned:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = sha256_text(addr.cleaned)
    cache_path = cache_dir / f"nominatim_{cache_key}.json"
    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if not cached:
            return None
        return GeoResult(**cached)

    with _NOMINATIM_LOCK:
        wait = (_NOMINATIM_LAST_CALL + min_interval_s) - time.time()
        if wait > 0:
            time.sleep(wait)

        params = {
            "format": "jsonv2",
            "limit": "1",
            "countrycodes": "de",
            "q": addr.cleaned,
        }
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "immo-ai-project (geodata scoring; local research)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
        except urllib.error.URLError:
            atomic_write_text(cache_path, "{}")
            return None
        finally:
            globals()["_NOMINATIM_LAST_CALL"] = time.time()

    data = json.loads(raw.decode("utf-8"))
    if not data:
        atomic_write_text(cache_path, "{}")
        return None
    r0 = data[0]
    res = GeoResult(
        lat=float(r0["lat"]),
        lon=float(r0["lon"]),
        method="nominatim",
        precision=str(r0.get("addresstype") or r0.get("type") or "unknown"),
        meta={k: r0.get(k) for k in ("display_name", "type", "addresstype", "class", "importance", "place_id")},
    )
    atomic_write_text(cache_path, json.dumps(res.__dict__, ensure_ascii=False))
    return res


def compute_geocode_confidence(addr: ParsedAddress, geo: GeoResult) -> tuple[float, dict[str, Any]]:
    """
    Returns (confidence_0_1, detail).

    Heuristic confidence score intended for down-weighting low-quality locations:
    - address match > postcode centroid
    - more postcode samples => slightly higher centroid confidence
    - if input had street+housenumber but we fell back to postcode, penalize
    """
    has_street_hn = bool(addr.street and addr.housenumber)
    meta = geo.meta if isinstance(geo.meta, dict) else {}
    sampled = meta.get("sampled")
    matched = meta.get("matched")
    try:
        sampled_n = int(sampled) if sampled is not None else 0
    except Exception:
        sampled_n = 0
    try:
        matched_n = int(matched) if matched is not None else 0
    except Exception:
        matched_n = 0

    detail: dict[str, Any] = {
        "method": geo.method,
        "precision": geo.precision,
        "has_street_housenumber": has_street_hn,
        "postcode_samples": sampled_n,
        "address_matches": matched_n,
    }

    method = str(geo.method or "")
    precision = str(geo.precision or "")

    if method == "overpass_address" or precision == "address":
        conf = 0.95
        if matched_n > 0:
            conf = min(0.99, conf + 0.01 * min(5, matched_n))
        detail["rule"] = "address_match"
    elif method == "overpass_postcode" or precision == "postcode":
        # For postcode centroids: more samples -> more stable, but still coarse.
        sample_factor = 1.0 - math.exp(-float(max(0, sampled_n)) / 60.0)  # ~0.63 at 60 samples
        conf = 0.45 + 0.30 * sample_factor  # 0.45..0.75
        if has_street_hn:
            conf *= 0.75
            detail["penalty"] = "street+housenumber_present_but_no_address_match"
        detail["rule"] = "postcode_centroid"
    elif method == "nominatim":
        addresstype = str(meta.get("addresstype") or meta.get("type") or "").lower()
        conf = 0.65
        if addresstype in ("house", "building"):
            conf = 0.92
        elif addresstype in ("residential", "apartments"):
            conf = 0.85
        elif addresstype in ("road", "street"):
            conf = 0.75
        elif addresstype in ("postcode",):
            conf = 0.60
        elif addresstype in ("city", "town", "village", "municipality"):
            conf = 0.50

        importance = meta.get("importance")
        try:
            imp = float(importance) if importance is not None else None
        except Exception:
            imp = None
        if imp is not None and math.isfinite(imp):
            conf *= 0.80 + 0.20 * max(0.0, min(1.0, imp))

        if has_street_hn and conf < 0.75:
            conf *= 0.90
            detail["penalty"] = "street+housenumber_present_but_nominatim_not_address_level"
        detail["rule"] = "nominatim"
        detail["nominatim_addresstype"] = addresstype or None
        detail["nominatim_importance"] = imp
    else:
        conf = 0.40
        if has_street_hn:
            conf *= 0.80
        detail["rule"] = "unknown"

    conf = max(0.0, min(1.0, float(conf)))
    return conf, detail


def _count_total_from_count_result(data: dict[str, Any]) -> int:
    els = data.get("elements") or []
    if not els:
        return 0
    tags = els[0].get("tags") or {}
    total = tags.get("total")
    if total is None:
        return 0
    try:
        return int(total)
    except Exception:
        return 0


def overpass_count(overpass_url: str, timeout_s: int, ql_body: str) -> int:
    ql = f"[out:json][timeout:{timeout_s}];{ql_body}out count;"
    data = overpass_query_json(overpass_url, ql, timeout_s=timeout_s + 60)
    return _count_total_from_count_result(data)


def parse_maxspeed_kmh(v: Any) -> Optional[float]:
    if not v:
        return None
    s = str(v).strip().lower()
    m = re.search(r"(\d{1,3})", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def traffic_proxy(
    overpass_url: str,
    lat: float,
    lon: float,
    radius_m: int,
    timeout_s: int,
) -> dict[str, Any]:
    ql = (
        f'[out:json][timeout:{timeout_s}];'
        f'way(around:{radius_m},{lat:.7f},{lon:.7f})["highway"~"^(motorway|trunk|primary|secondary)$"];'
        f"out tags;"
    )
    data = overpass_query_json(overpass_url, ql, timeout_s=timeout_s + 120)
    els = data.get("elements") or []
    speeds: list[float] = []
    for e in els:
        ms = parse_maxspeed_kmh((e.get("tags") or {}).get("maxspeed"))
        if ms is not None:
            speeds.append(ms)
    return {
        "major_road_ways": int(len(els)),
        "major_road_maxspeed_kmh_avg": (sum(speeds) / len(speeds)) if speeds else None,
    }


def overpass_metrics_combined(
    overpass_url: str,
    *,
    lat: float,
    lon: float,
    timeout_s: int,
    radius_amenities_m: int,
    radius_green_m: int,
    radius_transit_m: int,
    radius_traffic_m: int,
    radius_nightlife_m: int,
) -> dict[str, Any]:
    """
    Fetch all metric counts with a single Overpass query (plus major roads for traffic proxy).
    Returns the same keys as the non-combined path.
    """
    count_queries: list[tuple[str, str, str]] = [
        ("amenity_any", "amen", f'nwr(around:{radius_amenities_m},{lat:.7f},{lon:.7f})["amenity"]'),
        ("shop_any", "shop", f'nwr(around:{radius_amenities_m},{lat:.7f},{lon:.7f})["shop"]'),
        ("healthcare_any", "health", f'nwr(around:{radius_amenities_m},{lat:.7f},{lon:.7f})["healthcare"]'),
        ("tourism_any", "tour", f'nwr(around:{radius_amenities_m},{lat:.7f},{lon:.7f})["tourism"]'),
        ("nightlife", "night", f'nwr(around:{radius_nightlife_m},{lat:.7f},{lon:.7f})["amenity"~"^(bar|pub|nightclub)$"]'),
        ("public_transport_platform", "pt", f'nwr(around:{radius_transit_m},{lat:.7f},{lon:.7f})["public_transport"="platform"]'),
        ("highway_bus_stop", "bus", f'nwr(around:{radius_transit_m},{lat:.7f},{lon:.7f})["highway"="bus_stop"]'),
        ("railway_station_or_tram_stop", "rail", f'nwr(around:{radius_transit_m},{lat:.7f},{lon:.7f})["railway"~"^(station|tram_stop)$"]'),
        ("amenity_bus_station", "busst", f'nwr(around:{radius_transit_m},{lat:.7f},{lon:.7f})["amenity"="bus_station"]'),
        ("leisure_park_or_playground", "park", f'nwr(around:{radius_green_m},{lat:.7f},{lon:.7f})["leisure"~"^(park|playground)$"]'),
        ("natural_wood", "wood", f'nwr(around:{radius_green_m},{lat:.7f},{lon:.7f})["natural"="wood"]'),
        ("landuse_green", "land", f'nwr(around:{radius_green_m},{lat:.7f},{lon:.7f})["landuse"~"^(forest|grass|meadow|recreation_ground|village_green)$"]'),
    ]

    ql = [f"[out:json][timeout:{timeout_s}];"]
    for _, setname, body in count_queries:
        ql.append(f"{body}->.{setname};")
    ql.append(
        f'way(around:{radius_traffic_m},{lat:.7f},{lon:.7f})["highway"~"^(motorway|trunk|primary|secondary)$"]->.roads;'
    )
    for _, setname, _ in count_queries:
        ql.append(f".{setname} out count;")
    ql.append(".roads out tags;")

    data = overpass_query_json(overpass_url, "".join(ql), timeout_s=timeout_s + 180)
    els = data.get("elements") or []

    counts: list[int] = []
    road_ways: list[dict[str, Any]] = []
    for e in els:
        if e.get("type") == "count":
            counts.append(_count_total_from_count_result({"elements": [e]}))
        elif e.get("type") == "way":
            road_ways.append(e)

    metrics: dict[str, Any] = {}
    for (key, _, _), value in zip(count_queries, counts):
        metrics[key] = int(value)
    for key, _, _ in count_queries:
        metrics.setdefault(key, 0)

    speeds: list[float] = []
    for e in road_ways:
        ms = parse_maxspeed_kmh((e.get("tags") or {}).get("maxspeed"))
        if ms is not None:
            speeds.append(ms)
    metrics["major_road_ways"] = int(len(road_ways))
    metrics["major_road_maxspeed_kmh_avg"] = (sum(speeds) / len(speeds)) if speeds else None
    return metrics


def saturating_score(x: float, k: float) -> float:
    if x <= 0:
        return 0.0
    return 1.0 - math.exp(-x / k)


def compute_living_standard_score(metrics: dict[str, Any]) -> dict[str, Any]:
    amenity_total = (
        metrics["amenity_any"]
        + metrics["shop_any"]
        + metrics["healthcare_any"]
        + metrics["tourism_any"]
    )
    transit_total = (
        metrics["public_transport_platform"]
        + metrics["highway_bus_stop"]
        + metrics["railway_station_or_tram_stop"]
        + metrics["amenity_bus_station"]
    )
    green_total = metrics["leisure_park_or_playground"] + metrics["natural_wood"] + metrics["landuse_green"]

    s_amen = saturating_score(float(amenity_total), k=250.0)
    s_transit = saturating_score(float(transit_total), k=60.0)
    s_green = saturating_score(float(green_total), k=40.0)
    s_night = saturating_score(float(metrics["nightlife"]), k=10.0)

    major_road_ways = float(metrics.get("major_road_ways") or 0.0)
    s_traffic = 1.0 / (1.0 + (major_road_ways / 6.0))

    overall = (
        0.35 * s_amen
        + 0.25 * s_transit
        + 0.25 * s_green
        + 0.05 * s_night
        + 0.10 * s_traffic
    )
    overall_0_100 = max(0.0, min(100.0, overall * 100.0))
    return {
        "amenities_score_0_1": s_amen,
        "transit_score_0_1": s_transit,
        "green_score_0_1": s_green,
        "nightlife_score_0_1": s_night,
        "traffic_score_0_1": s_traffic,
        "living_standard_score_0_100": overall_0_100,
        "amenity_total": int(amenity_total),
        "transit_total": int(transit_total),
        "green_total": int(green_total),
    }


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def elem_latlon(e: dict[str, Any]) -> Optional[tuple[float, float]]:
    if "lat" in e and "lon" in e:
        return (float(e["lat"]), float(e["lon"]))
    c = e.get("center")
    if isinstance(c, dict) and "lat" in c and "lon" in c:
        return (float(c["lat"]), float(c["lon"]))
    return None


def overpass_fetch_pois(
    overpass_url: str,
    *,
    lat: float,
    lon: float,
    radius_m: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    ql = (
        f"[out:json][timeout:{timeout_s}];("
        f'nwr(around:{radius_m},{lat:.7f},{lon:.7f})["shop"~"^(supermarket|convenience|bakery)$"];'
        f'nwr(around:{radius_m},{lat:.7f},{lon:.7f})["amenity"~"^(pharmacy|doctors|hospital|school|kindergarten|university|police)$"];'
        f'nwr(around:{radius_m},{lat:.7f},{lon:.7f})["healthcare"];'
        f'nwr(around:{radius_m},{lat:.7f},{lon:.7f})["leisure"~"^(park|playground)$"];'
        f'nwr(around:{radius_m},{lat:.7f},{lon:.7f})["railway"~"^(station|tram_stop)$"];'
        f");out center tags;"
    )
    data = overpass_query_json(overpass_url, ql, timeout_s=timeout_s + 120)
    els = data.get("elements") or []
    return [e for e in els if isinstance(e, dict)]


def overpass_fetch_highways(
    overpass_url: str,
    *,
    lat: float,
    lon: float,
    radius_m: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    ql = (
        f"[out:json][timeout:{timeout_s}];"
        f'way(around:{radius_m},{lat:.7f},{lon:.7f})["highway"];'
        f"out tags center;"
    )
    data = overpass_query_json(overpass_url, ql, timeout_s=timeout_s + 120)
    els = data.get("elements") or []
    return [e for e in els if isinstance(e, dict) and e.get("type") == "way"]


def overpass_fetch_rails(
    overpass_url: str,
    *,
    lat: float,
    lon: float,
    radius_m: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    ql = (
        f"[out:json][timeout:{timeout_s}];"
        f'way(around:{radius_m},{lat:.7f},{lon:.7f})["railway"="rail"];'
        f"out tags center;"
    )
    data = overpass_query_json(overpass_url, ql, timeout_s=timeout_s + 120)
    els = data.get("elements") or []
    return [e for e in els if isinstance(e, dict) and e.get("type") == "way"]


def ext_metrics_from_pois(
    *,
    lat: float,
    lon: float,
    pois: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "poi_supermarket_count": 0,
        "poi_convenience_count": 0,
        "poi_bakery_count": 0,
        "poi_pharmacy_count": 0,
        "poi_doctors_count": 0,
        "poi_hospital_count": 0,
        "poi_school_count": 0,
        "poi_kindergarten_count": 0,
        "poi_university_count": 0,
        "poi_police_count": 0,
        "poi_park_count": 0,
        "poi_playground_count": 0,
        "poi_station_count": 0,
        "nearest_supermarket_m": None,
        "nearest_park_m": None,
        "nearest_station_m": None,
        "nearest_school_m": None,
        "nearest_kindergarten_m": None,
        "nearest_hospital_m": None,
        "nearest_pharmacy_m": None,
        "nearest_doctors_m": None,
        "nearest_playground_m": None,
        "nearest_police_m": None,
    }

    def upd_min(key: str, d: float) -> None:
        cur = metrics.get(key)
        if cur is None or d < cur:
            metrics[key] = d

    for e in pois:
        tags = e.get("tags") or {}
        if not isinstance(tags, dict):
            continue
        p = elem_latlon(e)
        if not p:
            continue
        d = haversine_m(lat, lon, p[0], p[1])

        shop = tags.get("shop")
        if shop == "supermarket":
            metrics["poi_supermarket_count"] += 1
            upd_min("nearest_supermarket_m", d)
        elif shop == "convenience":
            metrics["poi_convenience_count"] += 1
        elif shop == "bakery":
            metrics["poi_bakery_count"] += 1

        amenity = tags.get("amenity")
        if amenity == "pharmacy":
            metrics["poi_pharmacy_count"] += 1
            upd_min("nearest_pharmacy_m", d)
        elif amenity == "doctors":
            metrics["poi_doctors_count"] += 1
            upd_min("nearest_doctors_m", d)
        elif amenity == "hospital":
            metrics["poi_hospital_count"] += 1
            upd_min("nearest_hospital_m", d)
        elif amenity == "school":
            metrics["poi_school_count"] += 1
            upd_min("nearest_school_m", d)
        elif amenity == "kindergarten":
            metrics["poi_kindergarten_count"] += 1
            upd_min("nearest_kindergarten_m", d)
        elif amenity == "university":
            metrics["poi_university_count"] += 1
        elif amenity == "police":
            metrics["poi_police_count"] += 1
            upd_min("nearest_police_m", d)

        leisure = tags.get("leisure")
        if leisure == "park":
            metrics["poi_park_count"] += 1
            upd_min("nearest_park_m", d)
        elif leisure == "playground":
            metrics["poi_playground_count"] += 1
            upd_min("nearest_playground_m", d)

        railway = tags.get("railway")
        if railway in ("station", "tram_stop"):
            metrics["poi_station_count"] += 1
            upd_min("nearest_station_m", d)

    return metrics


def distance_score(dist_m: Optional[float], d0: float) -> float:
    if dist_m is None:
        return 0.0
    return math.exp(-float(dist_m) / max(1.0, d0))


def compute_additional_scores(metrics: dict[str, Any]) -> dict[str, Any]:
    walk = (
        0.35 * saturating_score(float(metrics.get("poi_supermarket_count") or 0), 6.0)
        + 0.20 * saturating_score(float(metrics.get("poi_bakery_count") or 0), 8.0)
        + 0.25 * saturating_score(float(metrics.get("poi_pharmacy_count") or 0), 6.0)
        + 0.20 * saturating_score(float(metrics.get("poi_doctors_count") or 0), 6.0)
    )

    education = (
        0.55 * saturating_score(float(metrics.get("poi_school_count") or 0), 6.0)
        + 0.45 * saturating_score(float(metrics.get("poi_kindergarten_count") or 0), 6.0)
    )
    education = max(education, 0.6 * distance_score(metrics.get("nearest_school_m"), 1200.0))

    healthcare = (
        0.35 * saturating_score(float(metrics.get("poi_pharmacy_count") or 0), 6.0)
        + 0.35 * saturating_score(float(metrics.get("poi_doctors_count") or 0), 8.0)
        + 0.30 * saturating_score(float(metrics.get("poi_hospital_count") or 0), 2.0)
    )
    healthcare = max(healthcare, 0.7 * distance_score(metrics.get("nearest_hospital_m"), 5000.0))

    green_quality = (
        0.55 * saturating_score(float(metrics.get("poi_park_count") or 0), 8.0)
        + 0.45 * saturating_score(float(metrics.get("poi_playground_count") or 0), 8.0)
    )
    green_quality = max(green_quality, distance_score(metrics.get("nearest_park_m"), 1200.0))

    lit_yes = float(metrics.get("highway_lit_yes_count") or 0)
    lit_total = float(metrics.get("highway_total_count") or 0)
    lit_ratio = (lit_yes / lit_total) if lit_total > 0 else 0.0
    safety_proxy = 0.65 * lit_ratio + 0.35 * distance_score(metrics.get("nearest_police_m"), 4000.0)

    major_ways = float(metrics.get("major_road_ways") or 0)
    major_dist = metrics.get("major_road_center_dist_m_min")
    rail_dist = metrics.get("rail_center_dist_m_min")
    noise_proxy = (
        0.50 * (1.0 / (1.0 + major_ways / 6.0))
        + 0.30 * distance_score(major_dist, 1500.0)
        + 0.20 * distance_score(rail_dist, 1500.0)
    )

    family = 0.40 * education + 0.35 * green_quality + 0.25 * noise_proxy

    return {
        "walkability_score_0_1": max(0.0, min(1.0, walk)),
        "education_score_0_1": max(0.0, min(1.0, education)),
        "healthcare_score_0_1": max(0.0, min(1.0, healthcare)),
        "green_quality_score_0_1": max(0.0, min(1.0, green_quality)),
        "safety_proxy_score_0_1": max(0.0, min(1.0, safety_proxy)),
        "noise_proxy_score_0_1": max(0.0, min(1.0, noise_proxy)),
        "family_friendliness_score_0_1": max(0.0, min(1.0, family)),
    }


def open_maybe_zstd(path: Path) -> io.TextIOBase:
    if path.suffix == ".zst" or path.suffixes[-2:] == [".jsonl", ".zst"]:
        try:
            import zstandard as zstd  # type: ignore
        except Exception:
            raise RuntimeError(
                f"Input is zstd-compressed but python package missing. Install: pip install zstandard\nPath: {path}"
            )
        dctx = zstd.ZstdDecompressor()
        fh = path.open("rb")
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_inputs(paths: list[Path]) -> Iterable[tuple[str, dict[str, Any]]]:
    for p in paths:
        with open_maybe_zstd(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield (str(p), json.loads(line))


def count_total_lines(paths: list[Path]) -> Optional[int]:
    total = 0
    for p in paths:
        # Avoid pre-counting compressed inputs (would fully decompress just for counting).
        if p.suffix == ".zst" or p.suffixes[-2:] == [".jsonl", ".zst"]:
            return None
        try:
            with p.open("rb") as f:
                for _ in f:
                    total += 1
        except OSError:
            return None
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich ImmoScout exposes with Overpass-based neighborhood scores.")
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input JSONL or JSONL.zst; can be repeated. If omitted, uses default immoscout_structured_run_*.jsonl",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: get_data/past/immoscout24/data/processed/immoscout_scored_<ts>.jsonl)",
    )
    ap.add_argument("--output-csv", default=None, help="Optional CSV output path (slim).")
    ap.add_argument("--slim", action="store_true", help="Write only slim output (no original fields).")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a progress bar (default: enabled when stdout is a TTY).",
    )
    ap.add_argument("--workers", type=int, default=1, help="Number of worker threads (default: 1).")
    ap.add_argument(
        "--overpass-concurrency",
        type=int,
        default=0,
        help="Limit concurrent Overpass requests (0 = auto). Helps prevent overload when using many workers.",
    )
    ap.add_argument(
        "--combined-query",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch all metrics with a single Overpass query per expose (default: true).",
    )
    ap.add_argument("--radius-nearest-m", type=int, default=2000, help="Search radius for nearest-POI metrics.")
    ap.add_argument("--radius-lit-m", type=int, default=250, help="Search radius for street lighting proxy.")
    ap.add_argument("--radius-rail-m", type=int, default=1500, help="Search radius for rail proximity proxy.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If --output exists, append and skip already-scored expose_ids found in that output.",
    )
    ap.add_argument("--overpass-url", default=os.environ.get("OVERPASS_URL", "http://127.0.0.1:8080/api/interpreter"))
    ap.add_argument("--timeout-s", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help="Process only N records (0 = no limit)")
    ap.add_argument("--force", action="store_true", help="Ignore caches and recompute everything")
    ap.add_argument("--no-nominatim", action="store_true", help="Do not fall back to Nominatim if Overpass geocode fails")
    ap.add_argument("--postcode-sample-limit", type=int, default=200, help="How many addr:postcode nodes to sample")
    ap.add_argument("--radius-amenities-m", type=int, default=800)
    ap.add_argument("--radius-green-m", type=int, default=1000)
    ap.add_argument("--radius-transit-m", type=int, default=800)
    ap.add_argument("--radius-traffic-m", type=int, default=300)
    ap.add_argument("--radius-nightlife-m", type=int, default=800)
    args = ap.parse_args()

    repo = repo_root()
    log(f"Repo root: {repo}")

    started_overpass = ensure_local_overpass_running(repo, args.overpass_url)

    try:
        db_marker_hash = read_db_marker_hash(repo)
        cache_root = repo / "get_data" / "past" / "immoscout24" / "cache" / "overpass_scoring"
        geo_cache = cache_root / "geocode"
        metric_cache = cache_root / "metrics"

        input_paths: list[Path] = []
        if args.input:
            for p in args.input:
                if any(ch in p for ch in "*?[]"):
                    input_paths.extend(sorted(repo.glob(p)))
                else:
                    input_paths.append((repo / p).resolve())
        else:
            input_paths = sorted(
                (repo / "get_data" / "past" / "immoscout24" / "data" / "processed").glob("immoscout_structured_run_*.jsonl")
            )

        if not input_paths:
            raise RuntimeError("No input files found. Pass --input <path> (can be repeated).")

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_path = Path(args.output) if args.output else repo / "get_data" / "past" / "immoscout24" / "data" / "processed" / f"immoscout_scored_{ts}.jsonl"
        if not out_path.is_absolute():
            out_path = (repo / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Inputs: {len(input_paths)} file(s)")
        log(f"Output: {out_path}")
        if args.output_csv:
            log(f"Output (CSV): {args.output_csv}")

        done_ids: set[str] = set()
        done_urls: set[str] = set()
        done_versions: dict[str, int] = {}
        if args.resume and out_path.exists():
            log(f"Resume: scanning existing output for expose_ids: {out_path}")
            with out_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    expose_id = o.get("expose_id") or (o.get("data") or {}).get("expose_id")
                    if expose_id:
                        done_ids.add(str(expose_id))
                        v = (o.get("overpass") or {}).get("version")
                        if isinstance(v, int):
                            done_versions[str(expose_id)] = v
                    u = o.get("url") or (o.get("data") or {}).get("url")
                    if u:
                        done_urls.add(str(u))
            log(f"Resume: found {len(done_ids)} already-scored expose_ids")

        total_seen = 0
        total_scored = 0
        total_geocoded = 0
        total_failed_geocode = 0
        total_failed_overpass = 0
        total_skipped_resume = 0
        t0 = time.time()

        progress_enabled = (
            args.progress if args.progress is not None else (is_tty() and os.environ.get("NO_PROGRESS") != "1")
        )
        progress_total = args.limit if args.limit and args.limit > 0 else (count_total_lines(input_paths) if progress_enabled else None)
        if progress_enabled and progress_total is None:
            log("Progress: enabled (unknown total)")
        elif progress_enabled and progress_total is not None:
            log(f"Progress: enabled (total={progress_total})")
        prog = Progress(total=progress_total, enabled=progress_enabled)

        workers = max(1, int(args.workers))
        if workers > 1:
            log(f"Workers: {workers} (threaded)")
        if workers > 1 and not args.no_nominatim:
            log("Note: Nominatim is rate-limited globally; for max speed consider `--no-nominatim`.")

        global _OVERPASS_SEM
        overpass_conc = int(args.overpass_concurrency) if args.overpass_concurrency else min(4, workers)
        _OVERPASS_SEM = threading.Semaphore(overpass_conc) if overpass_conc > 0 else None
        log(f"Overpass concurrency: {overpass_conc}")

        csv_fh = None
        csv_writer = None
        if args.output_csv:
            csv_path = (repo / args.output_csv).resolve() if not os.path.isabs(args.output_csv) else Path(args.output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_fh = csv_path.open("w", encoding="utf-8", newline="")
            csv_writer = csv.DictWriter(
                csv_fh,
                fieldnames=[
                    "expose_id",
                    "url",
                    "address",
                    "lat",
                    "lon",
                    "geocode_method",
                    "geocode_precision",
                    "living_standard_score_0_100",
                    "amenities_score_0_1",
                    "transit_score_0_1",
                    "green_score_0_1",
                    "nightlife_score_0_1",
                    "traffic_score_0_1",
                    "amenity_total",
                    "transit_total",
                    "green_total",
                    "major_road_ways",
                    "major_road_maxspeed_kmh_avg",
                ],
            )
            csv_writer.writeheader()

        def process_one(src_path: str, obj: dict[str, Any]) -> tuple[str, dict[str, Any], Optional[dict[str, Any]]]:
            data = obj.get("data") or {}
            expose_id = data.get("expose_id") or obj.get("expose_id") or None
            url = data.get("url") or obj.get("url") or None
            address_raw = data.get("address") or ""
            addr = parse_address(str(address_raw))

            try:
                geo: Optional[GeoResult] = geocode_via_overpass(
                    args.overpass_url,
                    db_marker_hash=db_marker_hash,
                    cache_dir=geo_cache,
                    addr=addr,
                    timeout_s=args.timeout_s,
                    sample_limit=args.postcode_sample_limit,
                    force=args.force,
                )
                if geo is None and not args.no_nominatim:
                    geo = geocode_via_nominatim(geo_cache, addr, timeout_s=max(10, args.timeout_s), force=args.force)
            except Exception as e:
                enriched = {
                    "source_file": src_path,
                    "expose_id": expose_id,
                    "url": url,
                    "address": address_raw,
                    "overpass": {"geocode": None, "metrics": None, "scores": None, "error": f"overpass_failed: {e}"},
                }
                out_obj = enriched if args.slim else {**obj, **enriched}
                return "overpass_failed", out_obj, None

            if geo is None:
                enriched = {
                    "source_file": src_path,
                    "expose_id": expose_id,
                    "url": url,
                    "address": address_raw,
                    "overpass": {"geocode": None, "metrics": None, "scores": None, "error": "geocode_failed"},
                }
                out_obj = enriched if args.slim else {**obj, **enriched}
                return "geocode_failed", out_obj, None

            lat, lon = geo.lat, geo.lon

            key_obj = {
                "db": db_marker_hash,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "r": {
                    "amen": int(args.radius_amenities_m),
                    "green": int(args.radius_green_m),
                    "transit": int(args.radius_transit_m),
                    "traffic": int(args.radius_traffic_m),
                    "night": int(args.radius_nightlife_m),
                },
                "combined": bool(args.combined_query),
            }
            cache_key = sha256_text(json.dumps(key_obj, sort_keys=True))
            metric_cache.mkdir(parents=True, exist_ok=True)
            metric_path = metric_cache / f"metrics_{cache_key}.json"

            ext_key_obj = {
                "db": db_marker_hash,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "v": ENRICHMENT_VERSION,
                "nearest": int(args.radius_nearest_m),
                "lit": int(args.radius_lit_m),
                "rail": int(args.radius_rail_m),
            }
            ext_key = sha256_text(json.dumps(ext_key_obj, sort_keys=True))
            ext_metric_path = metric_cache / f"metrics_ext_v{ENRICHMENT_VERSION}_{ext_key}.json"

            try:
                base_metrics: dict[str, Any]
                if metric_path.exists() and not args.force:
                    base_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                else:
                    if args.combined_query:
                        base_metrics = overpass_metrics_combined(
                            args.overpass_url,
                            lat=lat,
                            lon=lon,
                            timeout_s=args.timeout_s,
                            radius_amenities_m=args.radius_amenities_m,
                            radius_green_m=args.radius_green_m,
                            radius_transit_m=args.radius_transit_m,
                            radius_traffic_m=args.radius_traffic_m,
                            radius_nightlife_m=args.radius_nightlife_m,
                        )
                    else:
                        base_metrics = {
                            "amenity_any": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_amenities_m},{lat:.7f},{lon:.7f})["amenity"];',
                            ),
                            "shop_any": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_amenities_m},{lat:.7f},{lon:.7f})["shop"];',
                            ),
                            "healthcare_any": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_amenities_m},{lat:.7f},{lon:.7f})["healthcare"];',
                            ),
                            "tourism_any": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_amenities_m},{lat:.7f},{lon:.7f})["tourism"];',
                            ),
                            "nightlife": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_nightlife_m},{lat:.7f},{lon:.7f})["amenity"~"^(bar|pub|nightclub)$"];',
                            ),
                            "public_transport_platform": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_transit_m},{lat:.7f},{lon:.7f})["public_transport"="platform"];',
                            ),
                            "highway_bus_stop": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_transit_m},{lat:.7f},{lon:.7f})["highway"="bus_stop"];',
                            ),
                            "railway_station_or_tram_stop": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_transit_m},{lat:.7f},{lon:.7f})["railway"~"^(station|tram_stop)$"];',
                            ),
                            "amenity_bus_station": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_transit_m},{lat:.7f},{lon:.7f})["amenity"="bus_station"];',
                            ),
                            "leisure_park_or_playground": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_green_m},{lat:.7f},{lon:.7f})["leisure"~"^(park|playground)$"];',
                            ),
                            "natural_wood": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_green_m},{lat:.7f},{lon:.7f})["natural"="wood"];',
                            ),
                            "landuse_green": overpass_count(
                                args.overpass_url,
                                args.timeout_s,
                                f'nwr(around:{args.radius_green_m},{lat:.7f},{lon:.7f})["landuse"~"^(forest|grass|meadow|recreation_ground|village_green)$"];',
                            ),
                        }
                        base_metrics.update(
                            traffic_proxy(
                                args.overpass_url,
                                lat=lat,
                                lon=lon,
                                radius_m=args.radius_traffic_m,
                                timeout_s=args.timeout_s,
                            )
                        )
                    with contextlib.suppress(Exception):
                        atomic_write_text(metric_path, json.dumps(base_metrics, ensure_ascii=False))
            except Exception as e:
                enriched = {
                    "source_file": src_path,
                    "expose_id": expose_id,
                    "url": url,
                    "address": address_raw,
                    "overpass": {
                        "version": ENRICHMENT_VERSION,
                        "geocode": geo.__dict__,
                        "metrics": None,
                        "scores": None,
                        "error": f"overpass_failed: {e}",
                    },
                }
                out_obj = enriched if args.slim else {**obj, **enriched}
                return "overpass_failed", out_obj, None

            if ext_metric_path.exists() and not args.force:
                ext_metrics = json.loads(ext_metric_path.read_text(encoding="utf-8"))
            else:
                pois = overpass_fetch_pois(
                    args.overpass_url, lat=lat, lon=lon, radius_m=args.radius_nearest_m, timeout_s=args.timeout_s
                )
                ext_metrics = ext_metrics_from_pois(lat=lat, lon=lon, pois=pois)

                highways = overpass_fetch_highways(
                    args.overpass_url, lat=lat, lon=lon, radius_m=args.radius_lit_m, timeout_s=args.timeout_s
                )
                lit_yes = 0
                lit_no = 0
                lit_unknown = 0
                for w in highways:
                    lit = ((w.get("tags") or {}).get("lit") or "").strip().lower()
                    if lit in ("yes", "true", "1"):
                        lit_yes += 1
                    elif lit in ("no", "false", "0"):
                        lit_no += 1
                    else:
                        lit_unknown += 1
                ext_metrics["highway_total_count"] = int(len(highways))
                ext_metrics["highway_lit_yes_count"] = int(lit_yes)
                ext_metrics["highway_lit_no_count"] = int(lit_no)
                ext_metrics["highway_lit_unknown_count"] = int(lit_unknown)

                rails = overpass_fetch_rails(
                    args.overpass_url, lat=lat, lon=lon, radius_m=args.radius_rail_m, timeout_s=args.timeout_s
                )
                ext_metrics["rail_ways"] = int(len(rails))
                rail_min = None
                for w in rails:
                    p = elem_latlon(w)
                    if not p:
                        continue
                    d = haversine_m(lat, lon, p[0], p[1])
                    if rail_min is None or d < rail_min:
                        rail_min = d
                ext_metrics["rail_center_dist_m_min"] = rail_min

                roads = overpass_query_json(
                    args.overpass_url,
                    f'[out:json][timeout:{args.timeout_s}];way(around:{args.radius_traffic_m},{lat:.7f},{lon:.7f})["highway"~"^(motorway|trunk|primary|secondary)$"];out center;',
                    timeout_s=args.timeout_s + 120,
                ).get("elements", [])
                major_min = None
                for w in roads or []:
                    p = elem_latlon(w)
                    if not p:
                        continue
                    d = haversine_m(lat, lon, p[0], p[1])
                    if major_min is None or d < major_min:
                        major_min = d
                ext_metrics["major_road_center_dist_m_min"] = major_min

                with contextlib.suppress(Exception):
                    atomic_write_text(ext_metric_path, json.dumps(ext_metrics, ensure_ascii=False))

            metrics: dict[str, Any] = {**base_metrics, **ext_metrics}
            scores = compute_living_standard_score(metrics)
            extra_scores = compute_additional_scores(metrics)
            geo_conf, geo_conf_detail = compute_geocode_confidence(addr, geo)
            enriched = {
                "source_file": src_path,
                "expose_id": expose_id,
                "url": url,
                "address": address_raw,
                "overpass": {
                    "version": ENRICHMENT_VERSION,
                    "geocode": {**geo.__dict__, "confidence_0_1": geo_conf, "confidence_detail": geo_conf_detail},
                    "metrics": metrics,
                    "scores": {**scores, **extra_scores, "geocode_confidence_0_1": geo_conf, "geocode_confidence": geo_conf},
                    "db_marker_hash": db_marker_hash,
                },
            }
            out_obj = enriched if args.slim else {**obj, **enriched}

            csv_row = None
            if csv_writer is not None:
                csv_row = {
                    "expose_id": expose_id,
                    "url": url,
                    "address": address_raw,
                    "lat": geo.lat,
                    "lon": geo.lon,
                    "geocode_method": geo.method,
                    "geocode_precision": geo.precision,
                    **{k: scores.get(k) for k in scores.keys()},
                    "major_road_ways": metrics.get("major_road_ways"),
                    "major_road_maxspeed_kmh_avg": metrics.get("major_road_maxspeed_kmh_avg"),
                }
            return "ok", out_obj, csv_row

        try:
            out_mode = "a" if (args.resume and out_path.exists()) else "w"
            with out_path.open(out_mode, encoding="utf-8") as out:
                if workers == 1:
                    for src_path, obj in iter_inputs(input_paths):
                        if args.limit and total_seen >= args.limit:
                            break
                        total_seen += 1
                        data = obj.get("data") or {}
                        expose_id = data.get("expose_id") or obj.get("expose_id") or None
                        url = data.get("url") or obj.get("url") or None
                        if (
                            args.resume
                            and expose_id
                            and str(expose_id) in done_ids
                            and done_versions.get(str(expose_id), 0) >= ENRICHMENT_VERSION
                        ):
                            total_skipped_resume += 1
                            prog.update(
                                scored=total_scored,
                                geocode_failed=total_failed_geocode,
                                overpass_failed=total_failed_overpass,
                            )
                            continue
                        if (
                            args.resume
                            and url
                            and str(url) in done_urls
                            and done_versions.get(str(expose_id), 0) >= ENRICHMENT_VERSION
                        ):
                            total_skipped_resume += 1
                            prog.update(
                                scored=total_scored,
                                geocode_failed=total_failed_geocode,
                                overpass_failed=total_failed_overpass,
                            )
                            continue
                        kind, out_obj, csv_row = process_one(src_path, obj)

                        if kind == "ok":
                            total_geocoded += 1
                            total_scored += 1
                        elif kind == "geocode_failed":
                            total_failed_geocode += 1
                        else:
                            total_failed_overpass += 1

                        out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                        if csv_writer is not None and csv_row is not None:
                            csv_writer.writerow(csv_row)
                        prog.update(
                            scored=total_scored,
                            geocode_failed=total_failed_geocode,
                            overpass_failed=total_failed_overpass,
                        )
                else:
                    max_in_flight = int(os.environ.get("MAX_IN_FLIGHT", str(workers * 8)))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                        it = iter_inputs(input_paths)
                        in_flight: set[concurrent.futures.Future] = set()

                        def submit_next() -> bool:
                            nonlocal total_seen, total_skipped_resume
                            try:
                                src_path, obj = next(it)
                            except StopIteration:
                                return False
                            if args.limit and total_seen >= args.limit:
                                return False
                            data = obj.get("data") or {}
                            expose_id = data.get("expose_id") or obj.get("expose_id") or None
                            url = data.get("url") or obj.get("url") or None
                            total_seen += 1
                            if (
                                args.resume
                                and expose_id
                                and str(expose_id) in done_ids
                                and done_versions.get(str(expose_id), 0) >= ENRICHMENT_VERSION
                            ):
                                total_skipped_resume += 1
                                prog.update(
                                    scored=total_scored,
                                    geocode_failed=total_failed_geocode,
                                    overpass_failed=total_failed_overpass,
                                )
                                return True
                            if (
                                args.resume
                                and url
                                and str(url) in done_urls
                                and done_versions.get(str(expose_id), 0) >= ENRICHMENT_VERSION
                            ):
                                total_skipped_resume += 1
                                prog.update(
                                    scored=total_scored,
                                    geocode_failed=total_failed_geocode,
                                    overpass_failed=total_failed_overpass,
                                )
                                return True
                            in_flight.add(ex.submit(process_one, src_path, obj))
                            return True

                        while len(in_flight) < max_in_flight and submit_next():
                            pass

                        while in_flight:
                            done, in_flight = concurrent.futures.wait(
                                in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for fut in done:
                                try:
                                    kind, out_obj, csv_row = fut.result()
                                except Exception as e:
                                    kind = "overpass_failed"
                                    out_obj = {
                                        "source_file": None,
                                        "expose_id": None,
                                        "url": None,
                                        "address": None,
                                        "overpass": {"geocode": None, "metrics": None, "scores": None, "error": f"overpass_failed: {e}"},
                                    }
                                    csv_row = None

                                if kind == "ok":
                                    total_geocoded += 1
                                    total_scored += 1
                                elif kind == "geocode_failed":
                                    total_failed_geocode += 1
                                else:
                                    total_failed_overpass += 1

                                out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                                if csv_writer is not None and csv_row is not None:
                                    csv_writer.writerow(csv_row)
                                prog.update(
                                    scored=total_scored,
                                    geocode_failed=total_failed_geocode,
                                    overpass_failed=total_failed_overpass,
                                )

                            while len(in_flight) < max_in_flight and submit_next():
                                pass
        finally:
            with contextlib.suppress(Exception):
                prog.finish()
            if csv_fh is not None:
                csv_fh.close()

        elapsed = time.time() - t0
        log(
            f"Done. seen={total_seen} scored={total_scored} geocoded={total_geocoded} "
            f"geocode_failed={total_failed_geocode} overpass_failed={total_failed_overpass} "
            f"skipped_resume={total_skipped_resume} elapsed={elapsed:.1f}s"
        )
        return 0
    finally:
        if started_overpass:
            log("Stopping local Overpass (started by script)")
            stop_local_overpass(repo)


if __name__ == "__main__":
    raise SystemExit(main())
