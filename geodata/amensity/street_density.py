#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{ts}] {message}", file=sys.stderr, flush=True)


def die(message: str, code: int = 1) -> None:
    log(f"ERROR: {message}")
    raise SystemExit(code)


def resolve_repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(script_dir),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return Path(out).resolve()
    except Exception:
        pass
    return script_dir.parent.parent.resolve()


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("ß", "ss")
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = text.replace(" ", "-").replace("/", "-")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "city"


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float


def read_cities(path: Path) -> List[City]:
    if not path.exists():
        die(f"Cities file not found: {path}")
    cities: List[City] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                die(f"Invalid line {line_no} in {path} (expected 3 tab-separated columns)")
            name, lat_s, lon_s = parts
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except ValueError:
                die(f"Invalid lat/lon on line {line_no} in {path}")
            cities.append(City(name=name, lat=lat, lon=lon))
    if not cities:
        die(f"No cities found in: {path}")
    return cities


def m_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))


def m_per_deg_lat() -> float:
    return 111_320.0


def latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    x = (lon - lon0) * m_per_deg_lon(lat0)
    y = (lat - lat0) * m_per_deg_lat()
    return x, y


def xy_m_to_latlon(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    lat = lat0 + (y / m_per_deg_lat())
    lon = lon0 + (x / m_per_deg_lon(lat0))
    return lat, lon


def overpass_request_json(overpass_url: str, query: str, timeout_s: int) -> Dict[str, object]:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        overpass_url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s + 10) as resp:
            payload = resp.read()
            status = getattr(resp, "status", 200)
            if status != 200:
                raise urllib.error.HTTPError(overpass_url, status, "Non-200", resp.headers, None)
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        die(f"Overpass HTTP error {e.code}: {msg.strip()}")
    except Exception as e:
        die(f"Failed to query Overpass at {overpass_url}: {e}")

    try:
        return json.loads(payload)
    except Exception as e:
        snippet = payload[:400].decode("utf-8", errors="replace")
        die(f"Failed to parse Overpass response as JSON: {e}; snippet={snippet!r}")


def check_overpass_health(overpass_url: str) -> None:
    if overpass_url.endswith("/api/interpreter"):
        health_url = overpass_url[: -len("/api/interpreter")] + "/health"
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if getattr(resp, "status", 200) != 200:
                    raise RuntimeError(f"status={resp.status}")
                _ = resp.read()
            return
        except Exception:
            pass
    # Fallback tiny query
    _ = overpass_request_json(
        overpass_url,
        '[out:json][timeout:10];nwr(around:10,0.0,0.0)["amenity"];out count;',
        timeout_s=15,
    )


def sha1_short(text: str, length: int = 12) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return h[:length]


def read_import_marker(repo_root: Path) -> Tuple[Path, str, Dict[str, str]]:
    marker_path = repo_root / "geodata" / "overpass" / "db" / ".import_complete"
    if not marker_path.exists():
        return marker_path, "", {}
    try:
        content = marker_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return marker_path, "", {}
    parsed: Dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        parsed[k.strip()] = v.strip()
    return marker_path, content, parsed


class ProgressBar:
    def __init__(
        self,
        *,
        label: str,
        total: Optional[int],
        enabled: bool,
        update_every: int = 1,
    ) -> None:
        self.label = label
        self.total = total
        self.enabled = enabled
        self.update_every = max(1, update_every)
        self.start = time.time()
        self.last_render = 0.0
        self.count = 0

    def _render(self, extra: str = "") -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_render < 0.2:
            return
        self.last_render = now
        elapsed = max(0.001, now - self.start)
        rate = self.count / elapsed
        if self.total is not None and self.total > 0:
            pct = (self.count / self.total) * 100.0
            remaining = max(0, self.total - self.count)
            eta_s = int(remaining / rate) if rate > 0 else 0
            msg = (
                f"\r{self.label}: {pct:6.2f}% "
                f"({self.count}/{self.total}) "
                f"{rate:,.1f}/s ETA {eta_s}s"
            )
        else:
            msg = f"\r{self.label}: {self.count} {rate:,.1f}/s"
        if extra:
            msg += f" {extra}"
        # pad to clear leftover text
        msg = msg.ljust(120)
        sys.stderr.write(msg)
        sys.stderr.flush()

    def update(self, inc: int = 1, extra: str = "") -> None:
        self.count += inc
        if self.count % self.update_every == 0:
            self._render(extra=extra)

    def finish(self, extra: str = "") -> None:
        if not self.enabled:
            return
        self._render(extra=extra)
        sys.stderr.write("\n")
        sys.stderr.flush()


def is_local_wrapper_url(overpass_url: str) -> bool:
    try:
        u = urllib.parse.urlparse(overpass_url)
    except Exception:
        return False
    if u.scheme not in {"http", "https"}:
        return False
    host = (u.hostname or "").lower()
    port = u.port or (443 if u.scheme == "https" else 80)
    if host not in {"127.0.0.1", "localhost"}:
        return False
    if port != 8080:
        return False
    if not (u.path or "").endswith("/api/interpreter"):
        return False
    return True


def is_overpass_healthy(overpass_url: str) -> bool:
    if overpass_url.endswith("/api/interpreter"):
        health_url = overpass_url[: -len("/api/interpreter")] + "/health"
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                return getattr(resp, "status", 200) == 200
        except Exception:
            return False
    # Generic fallback: try a tiny request; treat any failure as unhealthy.
    try:
        _ = overpass_request_json(
            overpass_url,
            '[out:json][timeout:10];nwr(around:10,0.0,0.0)["amenity"];out count;',
            timeout_s=15,
        )
        return True
    except SystemExit:
        return False


def ensure_local_overpass_running(repo_root: Path, overpass_url: str) -> bool:
    """
    Returns True if this function started services (and should stop them on exit).
    """
    if not is_local_wrapper_url(overpass_url):
        check_overpass_health(overpass_url)
        return False

    if is_overpass_healthy(overpass_url):
        return False

    start_script = repo_root / "geodata" / "amensity" / "overpass_start.sh"
    stop_script = repo_root / "geodata" / "amensity" / "overpass_stop.sh"
    if not start_script.exists():
        die(f"Local Overpass not running and start script missing: {start_script}")
    if not stop_script.exists():
        die(f"Local Overpass not running and stop script missing: {stop_script}")

    log(f"Local Overpass not running; starting via: {start_script}")
    proc = subprocess.run([str(start_script)], cwd=str(repo_root))
    if proc.returncode != 0:
        die(f"Failed to start Overpass (exit {proc.returncode}). Check logs under ./geodata/overpass/logs/")

    deadline = time.time() + 30
    while time.time() < deadline:
        if is_overpass_healthy(overpass_url):
            return True
        time.sleep(0.5)

    die("Overpass start timed out; check ./geodata/overpass/logs/http.log and dispatcher.log")


def point_segment_distance_m(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 <= 0:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab_len2
    if t <= 0:
        cx, cy = ax, ay
    elif t >= 1:
        cx, cy = bx, by
    else:
        cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


@dataclass(frozen=True)
class AmenityPoint:
    lat: float
    lon: float
    x: float
    y: float
    osm_type: str
    osm_id: int


@dataclass(frozen=True)
class StreetSegment:
    street_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    minx: float
    miny: float
    maxx: float
    maxy: float


def parse_amenities(elements: Iterable[Dict[str, object]], lat0: float, lon0: float) -> List[AmenityPoint]:
    points: List[AmenityPoint] = []
    for el in elements:
        osm_type = str(el.get("type", ""))
        osm_id = int(el.get("id", 0) or 0)

        lat: Optional[float] = None
        lon: Optional[float] = None

        if "lat" in el and "lon" in el:
            try:
                lat = float(el["lat"])  # type: ignore[arg-type]
                lon = float(el["lon"])  # type: ignore[arg-type]
            except Exception:
                lat = None
                lon = None
        elif "center" in el and isinstance(el["center"], dict):
            center = el["center"]  # type: ignore[assignment]
            try:
                lat = float(center.get("lat"))  # type: ignore[arg-type]
                lon = float(center.get("lon"))  # type: ignore[arg-type]
            except Exception:
                lat = None
                lon = None

        if lat is None or lon is None:
            continue

        x, y = latlon_to_xy_m(lat, lon, lat0, lon0)
        points.append(AmenityPoint(lat=lat, lon=lon, x=x, y=y, osm_type=osm_type, osm_id=osm_id))
    return points


def parse_street_segments(
    ways: Iterable[Dict[str, object]],
    lat0: float,
    lon0: float,
) -> List[StreetSegment]:
    segments: List[StreetSegment] = []
    for el in ways:
        if str(el.get("type", "")) != "way":
            continue
        tags = el.get("tags")
        if not isinstance(tags, dict):
            continue
        name = tags.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        street_name = name.strip()

        geom = el.get("geometry")
        if not isinstance(geom, list) or len(geom) < 2:
            continue

        pts: List[Tuple[float, float]] = []
        for p in geom:
            if not isinstance(p, dict):
                continue
            try:
                lat = float(p.get("lat"))  # type: ignore[arg-type]
                lon = float(p.get("lon"))  # type: ignore[arg-type]
            except Exception:
                continue
            x, y = latlon_to_xy_m(lat, lon, lat0, lon0)
            pts.append((x, y))
        if len(pts) < 2:
            continue

        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            minx = min(x1, x2)
            maxx = max(x1, x2)
            miny = min(y1, y2)
            maxy = max(y1, y2)
            segments.append(
                StreetSegment(
                    street_name=street_name,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    minx=minx,
                    miny=miny,
                    maxx=maxx,
                    maxy=maxy,
                )
            )
    return segments


def build_segment_index(
    segments: List[StreetSegment],
    *,
    half_m: float,
    index_cell_m: float,
    buffer_m: float,
) -> Dict[Tuple[int, int], List[int]]:
    index: Dict[Tuple[int, int], List[int]] = {}

    def clamp_i(v: int, max_i: int) -> int:
        if v < 0:
            return 0
        if v > max_i:
            return max_i
        return v

    grid_n = int(math.ceil((2 * half_m) / index_cell_m))
    max_i = max(0, grid_n - 1)

    for i, seg in enumerate(segments):
        minx = seg.minx - buffer_m
        maxx = seg.maxx + buffer_m
        miny = seg.miny - buffer_m
        maxy = seg.maxy + buffer_m

        ix0 = clamp_i(int(math.floor((minx + half_m) / index_cell_m)), max_i)
        ix1 = clamp_i(int(math.floor((maxx + half_m) / index_cell_m)), max_i)
        iy0 = clamp_i(int(math.floor((miny + half_m) / index_cell_m)), max_i)
        iy1 = clamp_i(int(math.floor((maxy + half_m) / index_cell_m)), max_i)

        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                index.setdefault((ix, iy), []).append(i)

    return index


def main() -> int:
    repo_root = resolve_repo_root()

    parser = argparse.ArgumentParser(
        description="Amenity micro-density per 300m x 300m cell + per street (named highways) for big German cities."
    )
    parser.add_argument(
        "--cities-file",
        default=str(repo_root / "geodata" / "amensity" / "cities_de.tsv"),
        help="TSV file: city<TAB>lat<TAB>lon",
    )
    parser.add_argument(
        "--city",
        action="append",
        default=[],
        help="Filter (repeatable, substring match; e.g. --city Berlin)",
    )
    parser.add_argument("--limit-cities", type=int, default=0, help="Only run first N matched cities (0=all)")
    parser.add_argument(
        "--overpass-url",
        default=os.environ.get("OVERPASS_URL", "http://127.0.0.1:8080/api/interpreter"),
        help="Overpass /api/interpreter endpoint (default: local wrapper)",
    )
    parser.add_argument(
        "--bbox-half-size-km",
        type=float,
        default=10.0,
        help="Half-size of a square bbox around city center (km)",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=0.0,
        help="Use a circular radius around city center (km). When set (>0), this replaces bbox filtering.",
    )
    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=300.0,
        help="Square cell side length in meters (default: 300m => 300m x 300m)",
    )
    parser.add_argument(
        "--cell-area-m2",
        type=float,
        default=0.0,
        help="Optional: set cell area in m^2 (side = sqrt(area)); overrides --cell-size-m",
    )
    parser.add_argument(
        "--street-buffer-m",
        type=float,
        default=50.0,
        help="Assign amenities to nearest named street within this distance (meters)",
    )
    parser.add_argument(
        "--cell-street-max-dist-m",
        type=float,
        default=200.0,
        help="Also annotate each grid cell with nearest street if within this distance (meters, 0=always)",
    )
    parser.add_argument(
        "--highway-regex",
        default="^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|pedestrian)$",
        help='Regex for way["highway"~...], default filters to common street classes',
    )
    parser.add_argument("--timeout-s", type=int, default=180, help="Overpass query timeout (seconds)")
    parser.add_argument("--force", action="store_true", help="Re-fetch + recompute outputs")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show progress bars (default: on when stderr is a TTY)",
    )
    args = parser.parse_args()

    if args.radius_km and args.radius_km > 0:
        if args.bbox_half_size_km != 10.0:
            log("Note: --radius-km is set; --bbox-half-size-km is ignored.")
    else:
        if args.bbox_half_size_km <= 0:
            die("--bbox-half-size-km must be > 0")
    if args.cell_area_m2 and args.cell_area_m2 > 0:
        args.cell_size_m = math.sqrt(args.cell_area_m2)
    if args.cell_size_m <= 0:
        die("--cell-size-m must be > 0")
    if args.street_buffer_m < 0:
        die("--street-buffer-m must be >= 0")
    if args.cell_street_max_dist_m < 0:
        die("--cell-street-max-dist-m must be >= 0")

    cities = read_cities(Path(args.cities_file))
    if args.city:
        needles = [c.lower() for c in args.city]
        cities = [c for c in cities if any(n in c.name.lower() for n in needles)]
    if args.limit_cities and args.limit_cities > 0:
        cities = cities[: args.limit_cities]
    if not cities:
        die("No cities matched. Check --cities-file or --city filters.")

    out_root = repo_root / "geodata" / "amensity" / "analysis" / "street_density"
    out_root.mkdir(parents=True, exist_ok=True)

    progress_enabled = bool(sys.stderr.isatty())
    if args.progress is not None:
        progress_enabled = bool(args.progress)
    if os.environ.get("NO_PROGRESS", "").strip():
        progress_enabled = False

    log(f"Repo root: {repo_root}")
    log(f"Overpass URL: {args.overpass_url}")
    log(f"Cities: {len(cities)}")
    log(
        f"Params: "
        f"{('radius='+str(args.radius_km)+'km') if (args.radius_km and args.radius_km>0) else ('bbox_half='+str(args.bbox_half_size_km)+'km')} "
        f"cell={args.cell_size_m:.2f}m buffer={args.street_buffer_m}m"
    )
    log(f"Output: {out_root}")

    marker_path, marker_content, marker_kv = read_import_marker(repo_root)
    marker_id = sha1_short(marker_content or "no_marker")
    if marker_kv.get("extract"):
        log(f"Overpass DB import: {marker_kv.get('extract')} (marker={marker_path})")
    else:
        log(f"Overpass DB import marker: {marker_path} (missing or unreadable)")

    started_by_script = False
    try:
        started_by_script = ensure_local_overpass_running(repo_root, args.overpass_url)
    except SystemExit:
        log("Overpass not reachable/healthy. Start it with: ./geodata/amensity/overpass_start.sh")
        raise

    radius_m: Optional[float] = None
    if args.radius_km and args.radius_km > 0:
        radius_m = args.radius_km * 1000.0
        half_m = radius_m
    else:
        half_m = args.bbox_half_size_km * 1000.0
    cell_m = float(args.cell_size_m)
    cell_area_km2 = (cell_m * cell_m) / 1_000_000.0
    n_cells = int(math.ceil((2 * half_m) / cell_m))

    try:
        for idx, city in enumerate(cities, start=1):
            city_slug = slugify(city.name)
            city_dir = out_root / city_slug
            cache_dir = city_dir / "cache"
            city_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            meta_path = city_dir / "meta.json"
            cells_path = city_dir / "cells.csv"
            streets_path = city_dir / "streets.csv"

            if meta_path.exists() and cells_path.exists() and streets_path.exists() and not args.force:
                log(f"[{idx}/{len(cities)}] {city.name}: skip (outputs exist)")
                continue

            log(f"[{idx}/{len(cities)}] {city.name}: start")

            south, west = xy_m_to_latlon(-half_m, -half_m, city.lat, city.lon)
            north, east = xy_m_to_latlon(half_m, half_m, city.lat, city.lon)
            bbox = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"

            if radius_m is not None:
                log(f"[{idx}/{len(cities)}] {city.name}: fetch amenities + streets (radius_km={args.radius_km})")
            else:
                log(f"[{idx}/{len(cities)}] {city.name}: fetch amenities + streets (bbox={bbox})")

            if radius_m is not None:
                amenities_query = (
                    f'[out:json][timeout:{args.timeout_s}];'
                    f'(nwr(around:{int(radius_m)},{city.lat:.6f},{city.lon:.6f})["amenity"];);'
                    "out center;"
                )
                streets_query = (
                    f'[out:json][timeout:{args.timeout_s}];'
                    f'(way(around:{int(radius_m)},{city.lat:.6f},{city.lon:.6f})'
                    f'["highway"~"{args.highway_regex}"]["name"];);out geom;'
                )
            else:
                amenities_query = f'[out:json][timeout:{args.timeout_s}];(nwr["amenity"]({bbox}););out center;'
                streets_query = (
                    f'[out:json][timeout:{args.timeout_s}];'
                    f'(way["highway"~"{args.highway_regex}"]["name"]({bbox}););out geom;'
                )

            amenities_cache = cache_dir / f"amenities_{marker_id}_{sha1_short(amenities_query)}.json"
            streets_cache = cache_dir / f"streets_{marker_id}_{sha1_short(streets_query)}.json"

            if amenities_cache.exists() and not args.force:
                amenities_data = json.loads(amenities_cache.read_text(encoding="utf-8"))
            else:
                amenities_data = overpass_request_json(args.overpass_url, amenities_query, timeout_s=args.timeout_s)
                amenities_cache.write_text(json.dumps(amenities_data), encoding="utf-8")

            if streets_cache.exists() and not args.force:
                streets_data = json.loads(streets_cache.read_text(encoding="utf-8"))
            else:
                streets_data = overpass_request_json(args.overpass_url, streets_query, timeout_s=args.timeout_s)
                streets_cache.write_text(json.dumps(streets_data), encoding="utf-8")

            amenities_elements = amenities_data.get("elements", [])
            streets_elements = streets_data.get("elements", [])
            if not isinstance(amenities_elements, list) or not isinstance(streets_elements, list):
                die("Unexpected Overpass JSON format (missing elements array)")

            amenity_points = parse_amenities(amenities_elements, city.lat, city.lon)
            segments = parse_street_segments(streets_elements, city.lat, city.lon)

            log(f"  amenities: {len(amenity_points)} points")
            log(f"  street segments: {len(segments)}")

            # Cell counts
            cell_counts: List[int] = [0] * (n_cells * n_cells)

            def cell_index(x: float, y: float) -> Optional[int]:
                if x < -half_m or x > half_m or y < -half_m or y > half_m:
                    return None
                ix = int((x + half_m) // cell_m)
                iy = int((y + half_m) // cell_m)
                if ix < 0 or iy < 0 or ix >= n_cells or iy >= n_cells:
                    return None
                return iy * n_cells + ix

            for p in amenity_points:
                ci = cell_index(p.x, p.y)
                if ci is not None:
                    cell_counts[ci] += 1

            # Street length aggregation
            street_length_m: Dict[str, float] = {}
            for seg in segments:
                length = math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)
                street_length_m[seg.street_name] = street_length_m.get(seg.street_name, 0.0) + length

            # Assign amenities to streets using a segment spatial index
            assigned: Dict[str, int] = {}
            unassigned = 0

            seg_index: Optional[Dict[Tuple[int, int], List[int]]] = None
            index_cell_m = 0.0
            grid_n = 0

            if segments:
                buffer_for_index = max(
                    args.street_buffer_m,
                    args.cell_street_max_dist_m if args.cell_street_max_dist_m > 0 else 0.0,
                )
                index_cell_m = max(cell_m, buffer_for_index * 2.0, 200.0)
                seg_index = build_segment_index(
                    segments,
                    half_m=half_m,
                    index_cell_m=index_cell_m,
                    buffer_m=buffer_for_index,
                )
                grid_n = int(math.ceil((2 * half_m) / index_cell_m))

            if segments and seg_index is not None and args.street_buffer_m > 0:
                pbar = ProgressBar(
                    label=f"{city.name}: assign amenities->streets",
                    total=len(amenity_points),
                    enabled=progress_enabled,
                    update_every=max(1, len(amenity_points) // 200),
                )
                for p in amenity_points:
                    ix = int((p.x + half_m) // index_cell_m)
                    iy = int((p.y + half_m) // index_cell_m)
                    best_name: Optional[str] = None
                    best_dist = float("inf")

                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            cx = ix + dx
                            cy = iy + dy
                            if cx < 0 or cy < 0 or cx >= grid_n or cy >= grid_n:
                                continue
                            for seg_i in seg_index.get((cx, cy), []):
                                seg = segments[seg_i]
                                dist = point_segment_distance_m(p.x, p.y, seg.x1, seg.y1, seg.x2, seg.y2)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_name = seg.street_name

                    if best_name is not None and best_dist <= args.street_buffer_m:
                        assigned[best_name] = assigned.get(best_name, 0) + 1
                    else:
                        unassigned += 1
                    pbar.update(1)
                pbar.finish()
            else:
                unassigned = len(amenity_points)

            # Write cells.csv (full grid, including zeros)
            cells_written = 0
            total_cells = n_cells * n_cells
            cell_pbar = ProgressBar(
                label=f"{city.name}: grid cells",
                total=total_cells,
                enabled=progress_enabled,
                update_every=max(1, total_cells // 300),
            )
            with cells_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "city",
                        "ix",
                        "iy",
                        "cell_center_lat",
                        "cell_center_lon",
                        "cell_min_lat",
                        "cell_min_lon",
                        "cell_max_lat",
                        "cell_max_lon",
                        "nearest_street_name",
                        "nearest_street_distance_m",
                        "amenity_count",
                        "amenity_density_per_km2",
                    ],
                )
                writer.writeheader()
                for iy in range(n_cells):
                    for ix in range(n_cells):
                        x0 = -half_m + ix * cell_m
                        y0 = -half_m + iy * cell_m
                        x1 = x0 + cell_m
                        y1 = y0 + cell_m
                        cx = x0 + cell_m / 2.0
                        cy = y0 + cell_m / 2.0
                        if radius_m is not None:
                            if (cx * cx + cy * cy) > (radius_m * radius_m):
                                cell_pbar.update(1, extra=f"written={cells_written}")
                                continue
                        c_lat, c_lon = xy_m_to_latlon(cx, cy, city.lat, city.lon)
                        min_lat, min_lon = xy_m_to_latlon(x0, y0, city.lat, city.lon)
                        max_lat, max_lon = xy_m_to_latlon(x1, y1, city.lat, city.lon)
                        count = cell_counts[iy * n_cells + ix]

                        nearest_name = ""
                        nearest_dist_s = ""
                        if segments and seg_index is not None:
                            qx, qy = cx, cy
                            qix = int((qx + half_m) // index_cell_m)
                            qiy = int((qy + half_m) // index_cell_m)
                            best_name = None
                            best_dist = float("inf")
                            for dx in (-1, 0, 1):
                                for dy in (-1, 0, 1):
                                    sx = qix + dx
                                    sy = qiy + dy
                                    if sx < 0 or sy < 0 or sx >= grid_n or sy >= grid_n:
                                        continue
                                    for seg_i in seg_index.get((sx, sy), []):
                                        seg = segments[seg_i]
                                        dist = point_segment_distance_m(qx, qy, seg.x1, seg.y1, seg.x2, seg.y2)
                                        if dist < best_dist:
                                            best_dist = dist
                                            best_name = seg.street_name
                            if best_name is not None:
                                if args.cell_street_max_dist_m == 0 or best_dist <= args.cell_street_max_dist_m:
                                    nearest_name = best_name
                                    nearest_dist_s = f"{best_dist:.3f}"

                        writer.writerow(
                            {
                                "city": city.name,
                                "ix": ix,
                                "iy": iy,
                                "cell_center_lat": f"{c_lat:.6f}",
                                "cell_center_lon": f"{c_lon:.6f}",
                                "cell_min_lat": f"{min_lat:.6f}",
                                "cell_min_lon": f"{min_lon:.6f}",
                                "cell_max_lat": f"{max_lat:.6f}",
                                "cell_max_lon": f"{max_lon:.6f}",
                                "nearest_street_name": nearest_name,
                                "nearest_street_distance_m": nearest_dist_s,
                                "amenity_count": count,
                                "amenity_density_per_km2": f"{(count / cell_area_km2):.6f}",
                            }
                        )
                        cells_written += 1
                        cell_pbar.update(1, extra=f"written={cells_written}")
            cell_pbar.finish(extra=f"written={cells_written}")

            # Write streets.csv
            street_rows: List[Dict[str, object]] = []
            for name, length_m in sorted(street_length_m.items(), key=lambda kv: kv[0].lower()):
                count = assigned.get(name, 0)
                length_km = length_m / 1000.0
                per_km = (count / length_km) if length_km > 0 else 0.0
                street_rows.append(
                    {
                        "city": city.name,
                        "street_name": name,
                        "length_m": f"{length_m:.3f}",
                        "length_km": f"{length_km:.6f}",
                        "amenity_assigned_count": count,
                        "amenities_per_km": f"{per_km:.6f}",
                    }
                )

            with streets_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "city",
                        "street_name",
                        "length_m",
                        "length_km",
                        "amenity_assigned_count",
                        "amenities_per_km",
                    ],
                )
                writer.writeheader()
                writer.writerows(street_rows)

            # meta.json
            nonzero_counts = [c for c in cell_counts if c > 0]
            meta = {
                "city": {"name": city.name, "lat": city.lat, "lon": city.lon},
                "db_import": {
                    "marker_path": str(marker_path),
                    "marker_id": marker_id,
                    "extract": marker_kv.get("extract", ""),
                    "meta": marker_kv.get("meta", ""),
                    "imported_at": marker_kv.get("imported_at", ""),
                },
                "bbox": {
                    "half_size_km": args.bbox_half_size_km,
                    "south": south,
                    "west": west,
                    "north": north,
                    "east": east,
                },
                "radius": {"km": args.radius_km, "enabled": radius_m is not None},
                "grid": {"cell_size_m": cell_m, "cell_area_km2": cell_area_km2, "cells_per_axis": n_cells},
                "streets": {
                    "highway_regex": args.highway_regex,
                    "street_buffer_m": args.street_buffer_m,
                    "cell_street_max_dist_m": args.cell_street_max_dist_m,
                    "named_streets": len(street_length_m),
                },
                "amenities": {
                    "points": len(amenity_points),
                    "assigned_to_streets": sum(assigned.values()),
                    "unassigned": unassigned,
                },
                "cells": {
                    "nonzero_cells": len(nonzero_counts),
                    "cells_written": cells_written,
                    "count_mean_nonzero": mean(nonzero_counts) if nonzero_counts else 0.0,
                    "count_median_nonzero": median(nonzero_counts) if nonzero_counts else 0.0,
                    "count_max": max(cell_counts) if cell_counts else 0,
                },
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

            log(f"  wrote: {cells_path}")
            log(f"  wrote: {streets_path}")
            log(f"  wrote: {meta_path}")

        log("Done.")
        return 0
    finally:
        if started_by_script:
            stop_script = repo_root / "geodata" / "amensity" / "overpass_stop.sh"
            log(f"Stopping local Overpass (started by script): {stop_script}")
            subprocess.run([str(stop_script)], cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
