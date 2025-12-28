#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def repo_root() -> Path:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
        return Path(out.decode().strip())
    except Exception:
        return Path(__file__).resolve().parents[5]


def newest_run_dir(repo: Path) -> Optional[Path]:
    d = repo / "get_data" / "past" / "immoscout24" / "db_creation" / "data" / "json"
    candidates = sorted(d.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def load_json_file_with_repair(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raw = path.read_bytes()
        repaired = bytes((b if b >= 0x20 else 0x20) for b in raw)
        try:
            obj = json.loads(repaired.decode("utf-8", errors="replace"))
        except Exception:
            raise e
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        backup = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            if not backup.exists():
                backup.write_bytes(raw)
        except Exception:
            pass
        atomic_write_json(path, obj)
        log(f"Repaired invalid JSON and rewrote: {path.name} (backup: {backup.name})")
        return obj


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


_RE_TIME = re.compile(r"^(\\d+):(\\d{2})(?::(\\d{2}))?$")


def parse_gtfs_time_to_seconds(t: str) -> Optional[int]:
    """
    GTFS allows hours >= 24 for trips after midnight.
    Returns seconds since 00:00 (may exceed 86400).
    """
    if not t:
        return None
    m = _RE_TIME.match(t.strip())
    if not m:
        return None
    h = int(m.group(1))
    mi = int(m.group(2))
    s = int(m.group(3) or "0")
    if mi >= 60 or s >= 60:
        return None
    return h * 3600 + mi * 60 + s


@dataclass
class StopStat:
    lat: float
    lon: float
    total: int
    peak_am: int
    peak_pm: int


@dataclass
class GtfsIndex:
    # Arrays by stop_idx
    lats: list[float]
    lons: list[float]
    total: list[int]
    peak_am: list[int]
    peak_pm: list[int]
    # grid: (iy, ix) -> list[stop_idx]
    grid: dict[tuple[int, int], list[int]]
    cell_deg: float
    feeds: list[str]


def gtfs_fingerprint(paths: list[Path]) -> str:
    parts: list[dict[str, Any]] = []
    for p in paths:
        st = p.stat()
        parts.append({"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    import hashlib

    h = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
    return h


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> Optional[csv.DictReader]:
    try:
        raw = zf.read(name)
    except KeyError:
        return None
    # GTFS CSVs are usually UTF-8; some feeds use BOM.
    text = raw.decode("utf-8-sig", errors="replace")
    return csv.DictReader(text.splitlines())


def build_gtfs_index(gtfs_zips: list[Path], *, cell_deg: float) -> GtfsIndex:
    stops: dict[str, StopStat] = {}
    feeds: list[str] = []

    for zip_path in gtfs_zips:
        z_id = f"{zip_path.stem}:{zip_path.stat().st_size:x}"
        feeds.append(str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            stops_reader = read_csv_from_zip(zf, "stops.txt")
            if stops_reader is None:
                raise RuntimeError(f"Missing stops.txt in GTFS feed: {zip_path}")

            feed_stop_coords: dict[str, tuple[float, float]] = {}
            for row in stops_reader:
                sid = (row.get("stop_id") or "").strip()
                slat = row.get("stop_lat")
                slon = row.get("stop_lon")
                if not sid or slat is None or slon is None:
                    continue
                try:
                    lat = float(slat)
                    lon = float(slon)
                except Exception:
                    continue
                # Uniquify stop ids across feeds
                key = f"{z_id}:{sid}"
                feed_stop_coords[key] = (lat, lon)
                stops.setdefault(key, StopStat(lat=lat, lon=lon, total=0, peak_am=0, peak_pm=0))

            stop_times_reader = read_csv_from_zip(zf, "stop_times.txt")
            if stop_times_reader is None:
                raise RuntimeError(f"Missing stop_times.txt in GTFS feed: {zip_path}")

            for row in stop_times_reader:
                sid = (row.get("stop_id") or "").strip()
                if not sid:
                    continue
                key = f"{z_id}:{sid}"
                st = stops.get(key)
                if st is None:
                    # Some feeds use parent/child stop ids; ignore unknown.
                    continue
                dep = (row.get("departure_time") or row.get("arrival_time") or "").strip()
                sec = parse_gtfs_time_to_seconds(dep)
                if sec is None:
                    continue
                sec_mod = sec % 86400
                st.total += 1
                if 7 * 3600 <= sec_mod < 9 * 3600:
                    st.peak_am += 1
                if 16 * 3600 <= sec_mod < 19 * 3600:
                    st.peak_pm += 1

    lats: list[float] = []
    lons: list[float] = []
    total: list[int] = []
    peak_am: list[int] = []
    peak_pm: list[int] = []
    grid: dict[tuple[int, int], list[int]] = {}
    for st in stops.values():
        idx = len(lats)
        lats.append(st.lat)
        lons.append(st.lon)
        total.append(int(st.total))
        peak_am.append(int(st.peak_am))
        peak_pm.append(int(st.peak_pm))
        iy = int(math.floor(st.lat / cell_deg))
        ix = int(math.floor(st.lon / cell_deg))
        grid.setdefault((iy, ix), []).append(idx)

    return GtfsIndex(lats=lats, lons=lons, total=total, peak_am=peak_am, peak_pm=peak_pm, grid=grid, cell_deg=cell_deg, feeds=feeds)


def load_or_build_index(cache_dir: Path, gtfs_zips: list[Path], *, cell_deg: float, force: bool) -> tuple[str, GtfsIndex]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = gtfs_fingerprint(gtfs_zips)
    pkl = cache_dir / f"gtfs_index_{fp}.pkl"
    meta = cache_dir / f"gtfs_index_{fp}.meta.json"
    if pkl.exists() and meta.exists() and not force:
        with pkl.open("rb") as f:
            idx = pickle.load(f)
        return fp, idx

    log(f"Building GTFS index (this can take a while): feeds={len(gtfs_zips)}")
    idx = build_gtfs_index(gtfs_zips, cell_deg=cell_deg)
    with pkl.open("wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write_json(
        meta,
        {
            "fingerprint": fp,
            "cell_deg": cell_deg,
            "feeds": [str(p) for p in gtfs_zips],
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stops": len(idx.lats),
        },
    )
    return fp, idx


def query_index(idx: GtfsIndex, lat: float, lon: float, *, radius_m: float) -> dict[str, Any]:
    # Convert radius to degrees for grid neighborhood bounds
    lat_deg = radius_m / 111000.0
    lon_deg = radius_m / max(1.0, 111000.0 * math.cos(math.radians(lat)))
    min_lat = lat - lat_deg
    max_lat = lat + lat_deg
    min_lon = lon - lon_deg
    max_lon = lon + lon_deg

    cell = idx.cell_deg
    iy0 = int(math.floor(min_lat / cell))
    iy1 = int(math.floor(max_lat / cell))
    ix0 = int(math.floor(min_lon / cell))
    ix1 = int(math.floor(max_lon / cell))

    nearest_m: Optional[float] = None
    stops_in_r = 0
    dep_total = 0
    dep_am = 0
    dep_pm = 0

    r = float(radius_m)
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            for si in idx.grid.get((iy, ix), []):
                d = haversine_m(lat, lon, idx.lats[si], idx.lons[si])
                if nearest_m is None or d < nearest_m:
                    nearest_m = d
                if d <= r:
                    stops_in_r += 1
                    dep_total += int(idx.total[si])
                    dep_am += int(idx.peak_am[si])
                    dep_pm += int(idx.peak_pm[si])

    return {
        "radius_m": int(radius_m),
        "nearest_stop_m": nearest_m,
        "stops_within_radius": int(stops_in_r),
        "departures_within_radius_total": int(dep_total),
        "departures_within_radius_peak_am": int(dep_am),
        "departures_within_radius_peak_pm": int(dep_pm),
    }


def latlon_from_record(rec: dict[str, Any], *, field: str) -> Optional[tuple[float, float]]:
    data = rec.get("data")
    if not isinstance(data, dict):
        return None
    enrich = data.get(field)
    if not isinstance(enrich, dict):
        return None
    geo = enrich.get("geocode")
    if not isinstance(geo, dict):
        return None
    try:
        return (float(geo["lat"]), float(geo["lon"]))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Attach GTFS-based transit accessibility metrics to db_creation JSON batches.")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Target db_creation run dir (default: newest run_* under db_creation/data/json/)",
    )
    ap.add_argument(
        "--field",
        default="overpass_enrichment",
        help="Where to store the enrichment (under record['data'][FIELD])",
    )
    ap.add_argument(
        "--gtfs-zip",
        action="append",
        default=[],
        help="GTFS feed zip path (repeatable). If omitted, uses geodata/amensity/gtfs/feeds/*.zip",
    )
    ap.add_argument("--radius-m", type=int, default=500, help="Radius for stop density metrics (default: 500m)")
    ap.add_argument(
        "--cell-deg",
        type=float,
        default=0.005,
        help="Grid cell size in degrees for spatial index (default: 0.005 ~ 550m)",
    )
    ap.add_argument("--force-index", action="store_true", help="Rebuild GTFS index even if cached")
    ap.add_argument("--force", action="store_true", help="Overwrite existing gtfs_transit field")
    ap.add_argument("--limit-files", type=int, default=0, help="Only process N JSON files (0 = all)")
    ap.add_argument("--limit-records", type=int, default=0, help="Only update N records total (0 = unlimited)")
    args = ap.parse_args()

    repo = repo_root()
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo)
    if not run_dir:
        raise SystemExit("No db_creation run dir found. Pass --run-dir get_data/past/immoscout24/db_creation/data/json/run_.../")
    if not run_dir.is_absolute():
        run_dir = (repo / run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    gtfs_zips: list[Path] = []
    if args.gtfs_zip:
        for p in args.gtfs_zip:
            pp = Path(p)
            if not pp.is_absolute():
                pp = (repo / pp).resolve()
            gtfs_zips.append(pp)
    else:
        gtfs_zips = sorted((repo / "geodata" / "amensity" / "gtfs" / "feeds").glob("*.zip"))

    if not gtfs_zips:
        raise SystemExit(
            "No GTFS feeds found. Put GTFS zip(s) under geodata/amensity/gtfs/feeds/ or pass --gtfs-zip <path>."
        )
    for p in gtfs_zips:
        if not p.exists():
            raise SystemExit(f"GTFS zip not found: {p}")

    cache_dir = repo / "geodata" / "amensity" / "gtfs" / "cache"
    fp, idx = load_or_build_index(cache_dir, gtfs_zips, cell_deg=float(args.cell_deg), force=bool(args.force_index))
    log(f"GTFS index: stops={len(idx.lats)} feeds={len(gtfs_zips)} fingerprint={fp[:12]}...")

    json_files = sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")
    if args.limit_files and args.limit_files > 0:
        json_files = json_files[: args.limit_files]
    if not json_files:
        raise SystemExit(f"No JSON batch files found in: {run_dir}")

    log(f"Repo root: {repo}")
    log(f"Run dir: {run_dir}")
    log(f"Field: data.{args.field}")
    log(f"Radius: {args.radius_m}m")
    log(f"Files: {len(json_files)}")

    updated_files = 0
    updated_records = 0
    skipped_existing = 0
    missing_geo = 0
    loc_cache: dict[tuple[float, float], dict[str, Any]] = {}

    for p in json_files:
        arr = load_json_file_with_repair(p)
        if not isinstance(arr, list):
            raise SystemExit(f"Expected a JSON array in {p}, got {type(arr)}")
        changed = False
        for rec in arr:
            if args.limit_records and updated_records >= args.limit_records:
                break
            if not isinstance(rec, dict):
                continue
            data = rec.get("data")
            if not isinstance(data, dict):
                continue
            enrich = data.get(args.field)
            if not isinstance(enrich, dict):
                continue

            if not args.force and isinstance(enrich.get("gtfs_transit"), dict):
                skipped_existing += 1
                continue

            latlon = latlon_from_record(rec, field=args.field)
            if not latlon:
                missing_geo += 1
                continue
            lat, lon = latlon
            k = (round(lat, 4), round(lon, 4))
            if k in loc_cache:
                res = loc_cache[k]
            else:
                res = query_index(idx, k[0], k[1], radius_m=float(args.radius_m))
                res = {**res, "index_fingerprint": fp, "feeds": idx.feeds}
                loc_cache[k] = res
            enrich["gtfs_transit"] = res
            changed = True
            updated_records += 1

        if changed:
            atomic_write_json(p, arr)
            updated_files += 1
            log(f"Updated: {p.name}")

        if args.limit_records and updated_records >= args.limit_records:
            break

    marker = run_dir / ".gtfs_transit_enriched"
    atomic_write_json(
        marker,
        {
            "field": args.field,
            "radius_m": int(args.radius_m),
            "cell_deg": float(args.cell_deg),
            "index_fingerprint": fp,
            "feeds": [str(p) for p in gtfs_zips],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_files": updated_files,
            "updated_records": updated_records,
            "skipped_existing": skipped_existing,
            "missing_geo": missing_geo,
        },
    )
    log(
        f"Done. updated_files={updated_files} updated_records={updated_records} "
        f"skipped_existing={skipped_existing} missing_geo={missing_geo} marker={marker}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

