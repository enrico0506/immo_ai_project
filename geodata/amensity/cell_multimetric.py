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


def sha1_short(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


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


def overpass_request_json(overpass_url: str, query: str, timeout_s: int) -> Dict[str, object]:
    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        overpass_url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s + 20) as resp:
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
    return True


def ensure_local_overpass_running(repo_root: Path, overpass_url: str) -> bool:
    if not is_local_wrapper_url(overpass_url):
        return False
    if is_overpass_healthy(overpass_url):
        return False

    start_script = repo_root / "geodata" / "amensity" / "overpass_start.sh"
    stop_script = repo_root / "geodata" / "amensity" / "overpass_stop.sh"
    if not start_script.exists() or not stop_script.exists():
        die("Local Overpass scripts missing under geodata/amensity/")

    log(f"Local Overpass not running; starting via: {start_script}")
    proc = subprocess.run([str(start_script)], cwd=str(repo_root))
    if proc.returncode != 0:
        die(f"Failed to start Overpass (exit {proc.returncode}). Check ./geodata/overpass/logs/")

    deadline = time.time() + 30
    while time.time() < deadline:
        if is_overpass_healthy(overpass_url):
            return True
        time.sleep(0.5)

    die("Overpass start timed out; check ./geodata/overpass/logs/http.log and dispatcher.log")


class ProgressBar:
    def __init__(self, *, label: str, total: Optional[int], enabled: bool, update_every: int = 1) -> None:
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
            msg = f"\r{self.label}: {pct:6.2f}% ({self.count}/{self.total}) {rate:,.1f}/s ETA {eta_s}s"
        else:
            msg = f"\r{self.label}: {self.count} {rate:,.1f}/s"
        if extra:
            msg += f" {extra}"
        sys.stderr.write(msg.ljust(140))
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


def parse_points(elements: Iterable[Dict[str, object]], lat0: float, lon0: float) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for el in elements:
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
        pts.append((x, y))
    return pts


@dataclass(frozen=True)
class RoadSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    highway: str
    maxspeed: Optional[int]


def parse_major_road_segments(elements: Iterable[Dict[str, object]], lat0: float, lon0: float) -> List[RoadSegment]:
    segs: List[RoadSegment] = []
    for el in elements:
        if str(el.get("type", "")) != "way":
            continue
        tags = el.get("tags")
        if not isinstance(tags, dict):
            continue
        highway = tags.get("highway")
        if not isinstance(highway, str) or not highway:
            continue
        maxspeed: Optional[int] = None
        ms = tags.get("maxspeed")
        if isinstance(ms, str) and ms:
            m = re.search(r"(\d+)", ms)
            if m:
                try:
                    maxspeed = int(m.group(1))
                except Exception:
                    maxspeed = None

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
            segs.append(RoadSegment(x1=x1, y1=y1, x2=x2, y2=y2, highway=highway, maxspeed=maxspeed))
    return segs


def build_segment_index(
    segments: List[RoadSegment],
    *,
    half_m: float,
    index_cell_m: float,
    buffer_m: float,
) -> Dict[Tuple[int, int], List[int]]:
    index: Dict[Tuple[int, int], List[int]] = {}
    grid_n = int(math.ceil((2 * half_m) / index_cell_m))
    max_i = max(0, grid_n - 1)

    def clamp_i(v: int) -> int:
        if v < 0:
            return 0
        if v > max_i:
            return max_i
        return v

    for i, seg in enumerate(segments):
        minx = min(seg.x1, seg.x2) - buffer_m
        maxx = max(seg.x1, seg.x2) + buffer_m
        miny = min(seg.y1, seg.y2) - buffer_m
        maxy = max(seg.y1, seg.y2) + buffer_m

        ix0 = clamp_i(int(math.floor((minx + half_m) / index_cell_m)))
        ix1 = clamp_i(int(math.floor((maxx + half_m) / index_cell_m)))
        iy0 = clamp_i(int(math.floor((miny + half_m) / index_cell_m)))
        iy1 = clamp_i(int(math.floor((maxy + half_m) / index_cell_m)))

        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                index.setdefault((ix, iy), []).append(i)

    return index


def read_import_marker(repo_root: Path) -> Dict[str, str]:
    marker_path = repo_root / "geodata" / "overpass" / "db" / ".import_complete"
    if not marker_path.exists():
        return {}
    txt = marker_path.read_text(encoding="utf-8", errors="replace")
    parsed: Dict[str, str] = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        parsed[k.strip()] = v.strip()
    return parsed


def main() -> int:
    repo_root = resolve_repo_root()
    parser = argparse.ArgumentParser(
        description="Per-cell multi-metric analysis (amenities, transit, green, nightlife, traffic proxy) using local Overpass."
    )
    parser.add_argument(
        "--cities-file",
        default=str(repo_root / "geodata" / "amensity" / "cities_de.tsv"),
        help="TSV file: city<TAB>lat<TAB>lon",
    )
    parser.add_argument("--city", action="append", default=[], help="Filter (repeatable, substring match)")
    parser.add_argument("--limit-cities", type=int, default=0, help="Only run first N matched cities (0=all)")
    parser.add_argument(
        "--overpass-url",
        default=os.environ.get("OVERPASS_URL", "http://127.0.0.1:8080/api/interpreter"),
        help="Overpass /api/interpreter endpoint (default: local wrapper)",
    )
    parser.add_argument("--radius-km", type=float, default=5.0, help="Radius around city center (km)")
    parser.add_argument("--cell-size-m", type=float, default=300.0, help="Cell side length in meters")
    parser.add_argument("--timeout-s", type=int, default=300, help="Overpass query timeout (seconds)")
    parser.add_argument("--force", action="store_true", help="Re-fetch + recompute outputs")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show progress bars (default: on when stderr is a TTY)",
    )
    args = parser.parse_args()

    if args.radius_km <= 0:
        die("--radius-km must be > 0")
    if args.cell_size_m <= 0:
        die("--cell-size-m must be > 0")

    cities = read_cities(Path(args.cities_file))
    if args.city:
        needles = [c.lower() for c in args.city]
        cities = [c for c in cities if any(n in c.name.lower() for n in needles)]
    if args.limit_cities and args.limit_cities > 0:
        cities = cities[: args.limit_cities]
    if not cities:
        die("No cities matched.")

    progress_enabled = bool(sys.stderr.isatty())
    if args.progress is not None:
        progress_enabled = bool(args.progress)
    if os.environ.get("NO_PROGRESS", "").strip():
        progress_enabled = False

    out_root = repo_root / "geodata" / "amensity" / "analysis" / "cell_multimetric"
    out_root.mkdir(parents=True, exist_ok=True)

    log(f"Repo root: {repo_root}")
    log(f"Overpass URL: {args.overpass_url}")
    log(f"Cities: {len(cities)}")
    log(f"Params: radius={args.radius_km}km cell={args.cell_size_m}m")
    log(f"Output: {out_root}")

    db_marker = read_import_marker(repo_root)
    if db_marker.get("extract"):
        log(f"Overpass DB import: {db_marker.get('extract')}")

    started_by_script = False
    try:
        started_by_script = ensure_local_overpass_running(repo_root, args.overpass_url)

        radius_m = args.radius_km * 1000.0
        half_m = radius_m
        cell_m = args.cell_size_m
        cell_area_km2 = (cell_m * cell_m) / 1_000_000.0
        n_cells = int(math.ceil((2 * half_m) / cell_m))
        total_cells = n_cells * n_cells

        for idx, city in enumerate(cities, start=1):
            city_slug = slugify(city.name)
            city_dir = out_root / city_slug
            cache_dir = city_dir / "cache"
            city_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            cells_path = city_dir / "cells.csv"
            meta_path = city_dir / "meta.json"

            if cells_path.exists() and meta_path.exists() and not args.force:
                log(f"[{idx}/{len(cities)}] {city.name}: skip (outputs exist)")
                continue

            log(f"[{idx}/{len(cities)}] {city.name}: fetch data (radius_km={args.radius_km})")

            around = f"around:{int(radius_m)},{city.lat:.6f},{city.lon:.6f}"

            # Overpass queries (OSM-only metrics)
            pois_query = (
                f'[out:json][timeout:{args.timeout_s}];('
                f'nwr({around})["amenity"];'
                f'nwr({around})["shop"];'
                f'nwr({around})["healthcare"];'
                f'nwr({around})["tourism"];'
                ");out center;"
            )
            nightlife_query = (
                f'[out:json][timeout:{args.timeout_s}];('
                f'nwr({around})["amenity"~"^(bar|pub|nightclub)$"];'
                ");out center;"
            )
            transit_query = (
                f'[out:json][timeout:{args.timeout_s}];('
                f'nwr({around})["public_transport"="platform"];'
                f'nwr({around})["highway"="bus_stop"];'
                f'nwr({around})["railway"="station"];'
                f'nwr({around})["railway"="tram_stop"];'
                f'nwr({around})["amenity"="bus_station"];'
                ");out center;"
            )
            green_query = (
                f'[out:json][timeout:{args.timeout_s}];('
                f'nwr({around})["leisure"="park"];'
                f'nwr({around})["natural"="wood"];'
                f'nwr({around})["landuse"~"^(forest|grass|meadow|recreation_ground|village_green)$"];'
                f'nwr({around})["leisure"="playground"];'
                ");out center;"
            )
            major_roads_query = (
                f'[out:json][timeout:{args.timeout_s}];('
                f'way({around})["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"];'
                ");out geom;"
            )

            def cached_fetch(kind: str, query: str) -> Dict[str, object]:
                key = f"{kind}:{query}"
                path = cache_dir / f"{kind}_{sha1_short(key)}.json"
                if path.exists() and not args.force:
                    return json.loads(path.read_text(encoding="utf-8"))
                data = overpass_request_json(args.overpass_url, query, timeout_s=args.timeout_s)
                path.write_text(json.dumps(data), encoding="utf-8")
                return data

            pois = cached_fetch("pois", pois_query)
            nightlife = cached_fetch("nightlife", nightlife_query)
            transit = cached_fetch("transit", transit_query)
            green = cached_fetch("green", green_query)
            major_roads = cached_fetch("major_roads", major_roads_query)

            def elements(d: Dict[str, object]) -> List[Dict[str, object]]:
                els = d.get("elements", [])
                return els if isinstance(els, list) else []

            poi_elements = elements(pois)
            nightlife_elements = elements(nightlife)
            transit_elements = elements(transit)
            green_elements = elements(green)
            major_road_elements = elements(major_roads)

            # Categorize POIs by tags (avoid double counting within the combined query result)
            # We treat each returned element as potentially belonging to multiple categories.
            poi_tags = []
            poi_pts: List[Tuple[float, float, Dict[str, object]]] = []
            for el in poi_elements:
                tags = el.get("tags")
                if not isinstance(tags, dict):
                    continue
                pts = parse_points([el], city.lat, city.lon)
                if not pts:
                    continue
                x, y = pts[0]
                poi_pts.append((x, y, tags))

            nightlife_pts = parse_points(nightlife_elements, city.lat, city.lon)
            transit_pts = parse_points(transit_elements, city.lat, city.lon)
            green_pts = parse_points(green_elements, city.lat, city.lon)
            road_segs = parse_major_road_segments(major_road_elements, city.lat, city.lon)

            log(
                f"  fetched: pois={len(poi_pts)} nightlife={len(nightlife_pts)} transit={len(transit_pts)} "
                f"green={len(green_pts)} major_road_segs={len(road_segs)}"
            )

            # Cell grids (square over [-half, +half], then filter to circle by center)
            def cell_index(x: float, y: float) -> Optional[int]:
                if x < -half_m or x > half_m or y < -half_m or y > half_m:
                    return None
                ix = int((x + half_m) // cell_m)
                iy = int((y + half_m) // cell_m)
                if ix < 0 or iy < 0 or ix >= n_cells or iy >= n_cells:
                    return None
                return iy * n_cells + ix

            # Precompute which cells are in circle (by cell center)
            cell_in_circle: List[bool] = [False] * total_cells
            for iy in range(n_cells):
                for ix in range(n_cells):
                    x0 = -half_m + ix * cell_m
                    y0 = -half_m + iy * cell_m
                    cx = x0 + cell_m / 2.0
                    cy = y0 + cell_m / 2.0
                    if (cx * cx + cy * cy) <= (radius_m * radius_m):
                        cell_in_circle[iy * n_cells + ix] = True

            # Counters per cell (only for circle cells, but array is full for speed)
            amenity_cnt = [0] * total_cells
            shop_cnt = [0] * total_cells
            healthcare_cnt = [0] * total_cells
            tourism_cnt = [0] * total_cells
            nightlife_cnt = [0] * total_cells
            transit_cnt = [0] * total_cells
            green_cnt = [0] * total_cells
            major_road_len_m = [0.0] * total_cells
            major_road_nearest_m = [float("inf")] * total_cells
            major_road_maxspeed_avg = [0.0] * total_cells
            major_road_maxspeed_n = [0] * total_cells

            # Points -> cell counts
            pbar = ProgressBar(
                label=f"{city.name}: assign POIs",
                total=len(poi_pts),
                enabled=progress_enabled,
                update_every=max(1, len(poi_pts) // 200) if poi_pts else 1,
            )
            for x, y, tags in poi_pts:
                ci = cell_index(x, y)
                if ci is None or not cell_in_circle[ci]:
                    pbar.update(1)
                    continue
                if "amenity" in tags:
                    amenity_cnt[ci] += 1
                if "shop" in tags:
                    shop_cnt[ci] += 1
                if "healthcare" in tags:
                    healthcare_cnt[ci] += 1
                if "tourism" in tags:
                    tourism_cnt[ci] += 1
                pbar.update(1)
            pbar.finish()

            for label, pts, arr in [
                ("nightlife", nightlife_pts, nightlife_cnt),
                ("transit", transit_pts, transit_cnt),
                ("green", green_pts, green_cnt),
            ]:
                pbar = ProgressBar(
                    label=f"{city.name}: assign {label}",
                    total=len(pts),
                    enabled=progress_enabled,
                    update_every=max(1, len(pts) // 200) if pts else 1,
                )
                for x, y in pts:
                    ci = cell_index(x, y)
                    if ci is None or not cell_in_circle[ci]:
                        pbar.update(1)
                        continue
                    arr[ci] += 1
                    pbar.update(1)
                pbar.finish()

            # Road segments -> length per cell (midpoint assignment)
            pbar = ProgressBar(
                label=f"{city.name}: process major roads",
                total=len(road_segs),
                enabled=progress_enabled,
                update_every=max(1, len(road_segs) // 200) if road_segs else 1,
            )
            for seg in road_segs:
                mx = (seg.x1 + seg.x2) / 2.0
                my = (seg.y1 + seg.y2) / 2.0
                ci = cell_index(mx, my)
                if ci is None or not cell_in_circle[ci]:
                    pbar.update(1)
                    continue
                major_road_len_m[ci] += math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)
                if seg.maxspeed is not None:
                    major_road_maxspeed_avg[ci] += float(seg.maxspeed)
                    major_road_maxspeed_n[ci] += 1
                pbar.update(1)
            pbar.finish()

            # Nearest major road distance per cell (spatial index over segments)
            if road_segs:
                buffer_m = max(cell_m * 2.0, 500.0)
                index_cell_m = max(cell_m * 4.0, 1000.0)
                seg_index = build_segment_index(road_segs, half_m=half_m, index_cell_m=index_cell_m, buffer_m=buffer_m)
                grid_n = int(math.ceil((2 * half_m) / index_cell_m))

                pbar = ProgressBar(
                    label=f"{city.name}: nearest major road",
                    total=total_cells,
                    enabled=progress_enabled,
                    update_every=max(1, total_cells // 300),
                )
                for iy in range(n_cells):
                    for ix in range(n_cells):
                        ci = iy * n_cells + ix
                        pbar.update(1)
                        if not cell_in_circle[ci]:
                            continue
                        x0 = -half_m + ix * cell_m
                        y0 = -half_m + iy * cell_m
                        cx = x0 + cell_m / 2.0
                        cy = y0 + cell_m / 2.0

                        qix = int((cx + half_m) // index_cell_m)
                        qiy = int((cy + half_m) // index_cell_m)

                        best = float("inf")
                        for dx in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                sx = qix + dx
                                sy = qiy + dy
                                if sx < 0 or sy < 0 or sx >= grid_n or sy >= grid_n:
                                    continue
                                for seg_i in seg_index.get((sx, sy), []):
                                    seg = road_segs[seg_i]
                                    dist = point_segment_distance_m(cx, cy, seg.x1, seg.y1, seg.x2, seg.y2)
                                    if dist < best:
                                        best = dist
                        major_road_nearest_m[ci] = best
                pbar.finish()

            # Write output CSV
            cells_written = 0
            out_pbar = ProgressBar(
                label=f"{city.name}: write cells.csv",
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
                        "amenity_count",
                        "shop_count",
                        "healthcare_count",
                        "tourism_count",
                        "nightlife_count",
                        "transit_count",
                        "green_count",
                        "major_road_length_m",
                        "major_road_nearest_dist_m",
                        "major_road_maxspeed_avg",
                        "major_road_maxspeed_n",
                        "amenity_density_per_km2",
                        "shop_density_per_km2",
                        "transit_density_per_km2",
                        "green_density_per_km2",
                    ],
                )
                writer.writeheader()
                for iy in range(n_cells):
                    for ix in range(n_cells):
                        ci = iy * n_cells + ix
                        out_pbar.update(1, extra=f"written={cells_written}")
                        if not cell_in_circle[ci]:
                            continue
                        x0 = -half_m + ix * cell_m
                        y0 = -half_m + iy * cell_m
                        cx = x0 + cell_m / 2.0
                        cy = y0 + cell_m / 2.0
                        c_lat, c_lon = xy_m_to_latlon(cx, cy, city.lat, city.lon)

                        ms_n = major_road_maxspeed_n[ci]
                        ms_avg = (major_road_maxspeed_avg[ci] / ms_n) if ms_n else 0.0
                        nearest = major_road_nearest_m[ci]
                        nearest_s = f"{nearest:.3f}" if math.isfinite(nearest) else ""

                        writer.writerow(
                            {
                                "city": city.name,
                                "ix": ix,
                                "iy": iy,
                                "cell_center_lat": f"{c_lat:.6f}",
                                "cell_center_lon": f"{c_lon:.6f}",
                                "amenity_count": amenity_cnt[ci],
                                "shop_count": shop_cnt[ci],
                                "healthcare_count": healthcare_cnt[ci],
                                "tourism_count": tourism_cnt[ci],
                                "nightlife_count": nightlife_cnt[ci],
                                "transit_count": transit_cnt[ci],
                                "green_count": green_cnt[ci],
                                "major_road_length_m": f"{major_road_len_m[ci]:.3f}",
                                "major_road_nearest_dist_m": nearest_s,
                                "major_road_maxspeed_avg": f"{ms_avg:.3f}",
                                "major_road_maxspeed_n": ms_n,
                                "amenity_density_per_km2": f"{(amenity_cnt[ci] / cell_area_km2):.6f}",
                                "shop_density_per_km2": f"{(shop_cnt[ci] / cell_area_km2):.6f}",
                                "transit_density_per_km2": f"{(transit_cnt[ci] / cell_area_km2):.6f}",
                                "green_density_per_km2": f"{(green_cnt[ci] / cell_area_km2):.6f}",
                            }
                        )
                        cells_written += 1
            out_pbar.finish(extra=f"written={cells_written}")

            nonzero = [amenity_cnt[i] for i in range(total_cells) if cell_in_circle[i] and amenity_cnt[i] > 0]
            meta = {
                "city": {"name": city.name, "lat": city.lat, "lon": city.lon},
                "radius_km": args.radius_km,
                "cell_size_m": args.cell_size_m,
                "cells_per_axis": n_cells,
                "cells_written": cells_written,
                "overpass_db_extract": db_marker.get("extract", ""),
                "counts": {
                    "pois": len(poi_pts),
                    "nightlife": len(nightlife_pts),
                    "transit": len(transit_pts),
                    "green": len(green_pts),
                    "major_road_segments": len(road_segs),
                },
                "amenity_cells": {
                    "nonzero_cells": len(nonzero),
                    "mean_nonzero": mean(nonzero) if nonzero else 0.0,
                    "median_nonzero": median(nonzero) if nonzero else 0.0,
                    "max": max(nonzero) if nonzero else 0,
                },
                "notes": {
                    "air_quality": "Not included yet (would require external API or local sensors).",
                    "traffic_noise": "Proxy via major road length/nearest distance/maxspeed where present.",
                },
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

            log(f"[{idx}/{len(cities)}] {city.name}: wrote {cells_path}")

        log("Done.")
        return 0
    finally:
        if started_by_script:
            stop_script = repo_root / "geodata" / "amensity" / "overpass_stop.sh"
            log(f"Stopping local Overpass (started by script): {stop_script}")
            subprocess.run([str(stop_script)], cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())

