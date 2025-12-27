#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


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


def http_post_form(url: str, form: dict[str, str], timeout_s: int) -> bytes:
    data = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def overpass_query_json(overpass_url: str, ql: str, timeout_s: int) -> dict[str, Any]:
    raw = http_post_form(overpass_url, {"data": ql}, timeout_s=timeout_s)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Overpass returned non-JSON (bytes={len(raw)}): {e}") from e


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
                cache_path.write_text(json.dumps(res.__dict__, ensure_ascii=False), encoding="utf-8")
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
    cache_path.write_text(json.dumps(res.__dict__, ensure_ascii=False), encoding="utf-8")
    return res


_NOMINATIM_LAST_CALL = 0.0


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
        cache_path.write_text("{}", encoding="utf-8")
        return None
    finally:
        _NOMINATIM_LAST_CALL = time.time()

    data = json.loads(raw.decode("utf-8"))
    if not data:
        cache_path.write_text("{}", encoding="utf-8")
        return None
    r0 = data[0]
    res = GeoResult(
        lat=float(r0["lat"]),
        lon=float(r0["lon"]),
        method="nominatim",
        precision=str(r0.get("addresstype") or r0.get("type") or "unknown"),
        meta={k: r0.get(k) for k in ("display_name", "type", "addresstype", "class", "importance", "place_id")},
    )
    cache_path.write_text(json.dumps(res.__dict__, ensure_ascii=False), encoding="utf-8")
    return res


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
        out_path = (
            Path(args.output)
            if args.output
            else repo / "get_data" / "past" / "immoscout24" / "data" / "processed" / f"immoscout_scored_{ts}.jsonl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Inputs: {len(input_paths)} file(s)")
        log(f"Output: {out_path}")
        if args.output_csv:
            log(f"Output (CSV): {args.output_csv}")

        total_seen = 0
        total_scored = 0
        total_geocoded = 0
        total_failed_geocode = 0
        total_failed_overpass = 0
        t0 = time.time()

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

        try:
            with out_path.open("w", encoding="utf-8") as out:
                for src_path, obj in iter_inputs(input_paths):
                    if args.limit and total_seen >= args.limit:
                        break
                    total_seen += 1

                    data = obj.get("data") or {}
                    expose_id = data.get("expose_id") or obj.get("expose_id") or None
                    url = data.get("url") or obj.get("url") or None
                    address_raw = data.get("address") or ""
                    addr = parse_address(str(address_raw))

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
                        geo = geocode_via_nominatim(
                            geo_cache, addr, timeout_s=max(10, args.timeout_s), force=args.force
                        )

                    if geo is None:
                        total_failed_geocode += 1
                        enriched = {
                            "source_file": src_path,
                            "expose_id": expose_id,
                            "url": url,
                            "address": address_raw,
                            "overpass": {"geocode": None, "metrics": None, "scores": None, "error": "geocode_failed"},
                        }
                        out_obj = enriched if args.slim else {**obj, **enriched}
                        out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                        if is_tty() and total_seen % 200 == 0:
                            elapsed = time.time() - t0
                            rate = total_seen / max(0.001, elapsed)
                            log(
                                f"progress: {total_seen} seen, {total_scored} scored, "
                                f"{total_failed_geocode} geocode_failed ({rate:.1f}/s)"
                            )
                        continue

                    total_geocoded += 1
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
                    }
                    cache_key = sha256_text(json.dumps(key_obj, sort_keys=True))
                    metric_cache.mkdir(parents=True, exist_ok=True)
                    metric_path = metric_cache / f"metrics_{cache_key}.json"

                    metrics: Optional[dict[str, Any]] = None
                    if metric_path.exists() and not args.force:
                        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                    else:
                        try:
                            metrics = {
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
                            metrics.update(
                                traffic_proxy(
                                    args.overpass_url,
                                    lat=lat,
                                    lon=lon,
                                    radius_m=args.radius_traffic_m,
                                    timeout_s=args.timeout_s,
                                )
                            )
                            metric_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
                        except Exception as e:
                            total_failed_overpass += 1
                            enriched = {
                                "source_file": src_path,
                                "expose_id": expose_id,
                                "url": url,
                                "address": address_raw,
                                "overpass": {
                                    "geocode": geo.__dict__,
                                    "metrics": None,
                                    "scores": None,
                                    "error": f"overpass_failed: {e}",
                                },
                            }
                            out_obj = enriched if args.slim else {**obj, **enriched}
                            out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                            continue

                    scores = compute_living_standard_score(metrics)
                    total_scored += 1

                    enriched = {
                        "source_file": src_path,
                        "expose_id": expose_id,
                        "url": url,
                        "address": address_raw,
                        "overpass": {
                            "geocode": geo.__dict__,
                            "metrics": metrics,
                            "scores": scores,
                            "db_marker_hash": db_marker_hash,
                        },
                    }
                    out_obj = enriched if args.slim else {**obj, **enriched}
                    out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                    if csv_writer is not None:
                        csv_writer.writerow(
                            {
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
                        )

                    if is_tty() and total_seen % 200 == 0:
                        elapsed = time.time() - t0
                        rate = total_seen / max(0.001, elapsed)
                        log(
                            f"progress: {total_seen} seen, {total_scored} scored, "
                            f"{total_failed_geocode} geocode_failed, {total_failed_overpass} overpass_failed ({rate:.1f}/s)"
                        )
        finally:
            if csv_fh is not None:
                csv_fh.close()

        elapsed = time.time() - t0
        log(
            f"Done. seen={total_seen} scored={total_scored} geocoded={total_geocoded} "
            f"geocode_failed={total_failed_geocode} overpass_failed={total_failed_overpass} elapsed={elapsed:.1f}s"
        )
        return 0
    finally:
        if started_overpass:
            log("Stopping local Overpass (started by script)")
            stop_local_overpass(repo)


if __name__ == "__main__":
    raise SystemExit(main())
