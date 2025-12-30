#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from typing import Dict, Iterable, List, Tuple


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{ts}] {message}", file=sys.stderr, flush=True)


def die(message: str, code: int = 1) -> None:
    log(f"ERROR: {message}")
    raise SystemExit(code)


def resolve_repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    # Prefer git root if available; fallback to ../../ from this script.
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
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace(" ", "-")
        .replace("/", "-")
    )
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "city"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


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


def meters_to_deg_lat(meters: float) -> float:
    return meters / 111_320.0


def meters_to_deg_lon(meters: float, lat_deg: float) -> float:
    return meters / (111_320.0 * math.cos(math.radians(lat_deg)))


def generate_grid_points(
    center_lat: float,
    center_lon: float,
    *,
    half_size_km: float,
    step_km: float,
) -> List[Tuple[float, float, float, float]]:
    half_m = half_size_km * 1000.0
    step_m = step_km * 1000.0
    if half_m <= 0 or step_m <= 0:
        die("--half-size-km and --step-km must be > 0")

    # Include both ends: [-half, ..., 0, ..., +half]
    n = int(round((2 * half_m) / step_m))
    # Ensure odd count so 0 is included even with rounding.
    if n % 2 == 1:
        n += 1
    offsets = [(-half_m + i * step_m) for i in range(n + 1)]

    points: List[Tuple[float, float, float, float]] = []
    for dy_m in offsets:
        lat = center_lat + meters_to_deg_lat(dy_m)
        for dx_m in offsets:
            lon = center_lon + meters_to_deg_lon(dx_m, center_lat)
            points.append((lat, lon, dx_m, dy_m))
    return points


def overpass_count_amenities(
    *,
    overpass_url: str,
    radius_m: int,
    lat: float,
    lon: float,
    timeout_s: int,
) -> Dict[str, int]:
    query = (
        f'[out:json][timeout:{timeout_s}];'
        f'nwr(around:{radius_m},{lat:.6f},{lon:.6f})["amenity"];'
        "out count;"
    )

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
        data = json.loads(payload)
        elements = data.get("elements", [])
        if not elements:
            return {"total": 0, "nodes": 0, "ways": 0, "relations": 0}
        first = elements[0]
        tags = first.get("tags", {})
        return {
            "total": int(tags.get("total", "0")),
            "nodes": int(tags.get("nodes", "0")),
            "ways": int(tags.get("ways", "0")),
            "relations": int(tags.get("relations", "0")),
        }
    except Exception as e:
        snippet = payload[:400].decode("utf-8", errors="replace")
        die(f"Failed to parse Overpass response as JSON: {e}; snippet={snippet!r}")


def check_overpass_health(overpass_url: str) -> None:
    # If using our local wrapper, check /health first (fast and clear error).
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
    # Fallback: try a tiny query (count amenities at 0,0; should return JSON even if empty).
    _ = overpass_count_amenities(
        overpass_url=overpass_url,
        radius_m=10,
        lat=0.0,
        lon=0.0,
        timeout_s=10,
    )


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
    try:
        check_overpass_health(overpass_url)
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


def main() -> int:
    repo_root = resolve_repo_root()

    parser = argparse.ArgumentParser(
        description="Micro amenity density analysis for big German cities (uses local Overpass /api/interpreter)."
    )
    parser.add_argument(
        "--cities-file",
        default=str(repo_root / "geodata" / "amensity" / "cities_de.tsv"),
        help="TSV file: city<TAB>lat<TAB>lon",
    )
    parser.add_argument(
        "--overpass-url",
        default=os.environ.get("OVERPASS_URL", "http://127.0.0.1:8080/api/interpreter"),
        help="Overpass /api/interpreter endpoint (default: local wrapper)",
    )
    parser.add_argument("--radius-m", type=int, default=300, help="Micro radius in meters")
    parser.add_argument("--half-size-km", type=float, default=3.0, help="Half-size of square around center (km)")
    parser.add_argument("--step-km", type=float, default=1.5, help="Grid step (km)")
    parser.add_argument("--timeout-s", type=int, default=60, help="Per-query Overpass timeout (seconds)")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between queries (ms)")
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    parser.add_argument("--limit-cities", type=int, default=0, help="Only run first N cities (0 = all)")
    args = parser.parse_args()

    if args.radius_m <= 0:
        die("--radius-m must be > 0")

    cities = read_cities(Path(args.cities_file))
    if args.limit_cities and args.limit_cities > 0:
        cities = cities[: args.limit_cities]

    out_base = repo_root / "geodata" / "amensity" / "analysis" / "micro_density"
    raw_dir = out_base / "raw"
    out_base.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    log(f"Repo root: {repo_root}")
    log(f"Overpass URL: {args.overpass_url}")
    log(f"Cities: {len(cities)} ({args.cities_file})")
    log(f"Grid: half={args.half_size_km}km step={args.step_km}km radius={args.radius_m}m")
    log(f"Output: {out_base}")

    started_by_script = False
    try:
        started_by_script = ensure_local_overpass_running(repo_root, args.overpass_url)
    except SystemExit:
        log("Overpass not reachable/healthy. Start it with: ./geodata/amensity/overpass_start.sh")
        raise

    run_info = {
        "overpass_url": args.overpass_url,
        "radius_m": args.radius_m,
        "half_size_km": args.half_size_km,
        "step_km": args.step_km,
        "timeout_s": args.timeout_s,
        "sleep_ms": args.sleep_ms,
        "cities_file": str(Path(args.cities_file).resolve()),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_base / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    circle_area_km2 = math.pi * (args.radius_m / 1000.0) ** 2

    summary_rows: List[Dict[str, object]] = []

    try:
        for idx, city in enumerate(cities, start=1):
            city_slug = slugify(city.name)
            raw_path = raw_dir / f"{city_slug}.csv"

            if raw_path.exists() and not args.force:
                log(f"[{idx}/{len(cities)}] {city.name}: skip (exists: {raw_path})")
                # Still include in summary by reading existing raw file.
                rows: List[Dict[str, str]] = []
                with raw_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            else:
                log(f"[{idx}/{len(cities)}] {city.name}: querying...")
                points = generate_grid_points(
                    city.lat,
                    city.lon,
                    half_size_km=args.half_size_km,
                    step_km=args.step_km,
                )

                rows = []
                for p_i, (lat, lon, dx_m, dy_m) in enumerate(points, start=1):
                    counts = overpass_count_amenities(
                        overpass_url=args.overpass_url,
                        radius_m=args.radius_m,
                        lat=lat,
                        lon=lon,
                        timeout_s=args.timeout_s,
                    )
                    rows.append(
                        {
                            "city": city.name,
                            "center_lat": f"{city.lat:.6f}",
                            "center_lon": f"{city.lon:.6f}",
                            "lat": f"{lat:.6f}",
                            "lon": f"{lon:.6f}",
                            "dx_m": f"{dx_m:.1f}",
                            "dy_m": f"{dy_m:.1f}",
                            "radius_m": str(args.radius_m),
                            "amenity_total": str(counts["total"]),
                            "amenity_nodes": str(counts["nodes"]),
                            "amenity_ways": str(counts["ways"]),
                            "amenity_relations": str(counts["relations"]),
                        }
                    )
                    if args.sleep_ms > 0:
                        time.sleep(args.sleep_ms / 1000.0)
                    if p_i % 10 == 0:
                        log(f"  {city.name}: {p_i}/{len(points)} points")

                with raw_path.open("w", encoding="utf-8", newline="") as f:
                    fieldnames = list(rows[0].keys()) if rows else []
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                log(f"  wrote: {raw_path}")

            totals = [float(r["amenity_total"]) for r in rows] if rows else []
            densities = [t / circle_area_km2 for t in totals] if totals else []

            summary_rows.append(
                {
                    "city": city.name,
                    "center_lat": f"{city.lat:.6f}",
                    "center_lon": f"{city.lon:.6f}",
                    "radius_m": args.radius_m,
                    "circle_area_km2": f"{circle_area_km2:.6f}",
                    "half_size_km": args.half_size_km,
                    "step_km": args.step_km,
                    "points": len(totals),
                    "amenity_total_mean": f"{mean(totals):.3f}" if totals else "",
                    "amenity_total_median": f"{median(totals):.3f}" if totals else "",
                    "amenity_total_p10": f"{percentile(totals, 10):.3f}" if totals else "",
                    "amenity_total_p90": f"{percentile(totals, 90):.3f}" if totals else "",
                    "amenity_total_min": f"{min(totals):.3f}" if totals else "",
                    "amenity_total_max": f"{max(totals):.3f}" if totals else "",
                    "amenity_density_per_km2_mean": f"{mean(densities):.3f}" if densities else "",
                    "amenity_density_per_km2_median": f"{median(densities):.3f}" if densities else "",
                }
            )

        summary_path = out_base / "summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(summary_rows[0].keys()) if summary_rows else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        log(f"Wrote summary: {summary_path}")

        return 0
    finally:
        if started_by_script:
            stop_script = repo_root / "geodata" / "amensity" / "overpass_stop.sh"
            log(f"Stopping local Overpass (started by script): {stop_script}")
            subprocess.run([str(stop_script)], cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
