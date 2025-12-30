#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional


UBA_BASE = "https://www.umweltbundesamt.de/api/air_data/v2"

# UBA component ids (from /components/json)
UBA_COMPONENT_CODE_TO_ID = {
    "PM10": 1,
    "CO": 2,
    "O3": 3,
    "SO2": 4,
    "NO2": 5,
    "PM2": 9,  # PM2.5 in UBA naming
}


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


def http_get_json(url: str, timeout_s: int = 30) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "immo-ai-project (air quality enrichment; local research)",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def fetch_uba_stations() -> list[dict[str, Any]]:
    """
    Returns simplified station list with fields we need.
    """
    data = http_get_json(f"{UBA_BASE}/stations/json", timeout_s=60)
    indices = data["indices"]
    idx = {name: i for i, name in enumerate(indices)}

    stations: list[dict[str, Any]] = []
    for _, row in (data.get("data") or {}).items():
        if not isinstance(row, list):
            continue
        active_to = row[idx["station active to"]]
        # Keep active stations only
        if active_to not in (None, ""):
            continue
        try:
            sid = int(row[idx["station id"]])
            code = str(row[idx["station code"]])
            name = str(row[idx["station name"]])
            city = str(row[idx["station city"]])
            lon = float(row[idx["station longitude"]])
            lat = float(row[idx["station latitude"]])
        except Exception:
            continue

        stations.append(
            {
                "id": sid,
                "code": code,
                "name": name,
                "city": city,
                "lat": lat,
                "lon": lon,
                "type": row[idx.get("station type name", 16)] if "station type name" in idx else None,
                "setting": row[idx.get("station setting name", 14)] if "station setting name" in idx else None,
                "network": row[idx.get("network name", 13)] if "network name" in idx else None,
                "zip": row[idx.get("station zip code", 19)] if "station zip code" in idx else None,
            }
        )
    return stations


def build_station_grid(stations: list[dict[str, Any]], *, cell_deg: float) -> tuple[dict[tuple[int, int], list[int]], list[float], list[float]]:
    grid: dict[tuple[int, int], list[int]] = {}
    lats: list[float] = []
    lons: list[float] = []
    for i, st in enumerate(stations):
        lat = float(st["lat"])
        lon = float(st["lon"])
        lats.append(lat)
        lons.append(lon)
        iy = int(math.floor(lat / cell_deg))
        ix = int(math.floor(lon / cell_deg))
        grid.setdefault((iy, ix), []).append(i)
    return grid, lats, lons


def nearest_station(
    *,
    stations: list[dict[str, Any]],
    grid: dict[tuple[int, int], list[int]],
    lats: list[float],
    lons: list[float],
    cell_deg: float,
    lat: float,
    lon: float,
    max_rings: int,
) -> Optional[tuple[dict[str, Any], float]]:
    iy = int(math.floor(lat / cell_deg))
    ix = int(math.floor(lon / cell_deg))
    best_i: Optional[int] = None
    best_d: Optional[float] = None

    for r in range(max_rings + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if r and abs(dy) != r and abs(dx) != r:
                    continue
                cell = (iy + dy, ix + dx)
                for si in grid.get(cell, []):
                    d = haversine_m(lat, lon, lats[si], lons[si])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_i = si
        if best_d is not None and r >= 2:
            break

    if best_i is None or best_d is None:
        return None
    return stations[best_i], float(best_d)


def load_station_cache(cache_path: Path, *, refresh: bool) -> list[dict[str, Any]]:
    if cache_path.exists() and not refresh:
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("stations"), list):
            return obj["stations"]
    # refresh
    stations = fetch_uba_stations()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        cache_path,
        {
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stations": stations,
        },
    )
    return stations


def measures_cache_key(station_id: int, component_id: int, scope_id: int, date_from: str, time_from: str, date_to: str, time_to: str) -> str:
    import hashlib

    payload = {
        "station": int(station_id),
        "component": int(component_id),
        "scope": int(scope_id),
        "date_from": date_from,
        "time_from": time_from,
        "date_to": date_to,
        "time_to": time_to,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def fetch_measures(
    *,
    station_id: int,
    component_id: int,
    scope_id: int,
    date_from: str,
    time_from: str,
    date_to: str,
    time_to: str,
    timeout_s: int,
) -> dict[str, Any]:
    params = {
        "station": str(station_id),
        "component": str(component_id),
        "scope": str(scope_id),
        "date_from": date_from,
        "time_from": time_from,
        "date_to": date_to,
        "time_to": time_to,
    }
    url = f"{UBA_BASE}/measures/json?" + urllib.parse.urlencode(params)
    data = http_get_json(url, timeout_s=timeout_s)
    series = (data.get("data") or {}).get(str(station_id)) or {}
    values: list[float] = []
    for _, row in series.items():
        if not isinstance(row, list) or len(row) < 3:
            continue
        v = row[2]
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            values.append(fv)
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None}
    values.sort()
    mean = sum(values) / len(values)
    return {"n": len(values), "mean": mean, "min": values[0], "max": values[-1]}


def get_measures_cached(
    cache_dir: Path,
    *,
    station_id: int,
    component_id: int,
    scope_id: int,
    date_from: str,
    time_from: str,
    date_to: str,
    time_to: str,
    timeout_s: int,
    force: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = measures_cache_key(station_id, component_id, scope_id, date_from, time_from, date_to, time_to)
    path = cache_dir / f"uba_measures_{key}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    obj = fetch_measures(
        station_id=station_id,
        component_id=component_id,
        scope_id=scope_id,
        date_from=date_from,
        time_from=time_from,
        date_to=date_to,
        time_to=time_to,
        timeout_s=timeout_s,
    )
    obj = {
        "station_id": int(station_id),
        "component_id": int(component_id),
        "scope_id": int(scope_id),
        "date_from": date_from,
        "time_from": time_from,
        "date_to": date_to,
        "time_to": time_to,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **obj,
    }
    atomic_write_json(path, obj)
    return obj


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
    ap = argparse.ArgumentParser(description="Attach UBA air-quality nearest-station (and optional measures) to db_creation JSON batches.")
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
    ap.add_argument("--cell-deg", type=float, default=0.2, help="Station grid cell size in degrees (default: 0.2 ~ 22km)")
    ap.add_argument("--max-rings", type=int, default=6, help="Max neighbor rings to search (default: 6)")
    ap.add_argument("--refresh-stations", action="store_true", help="Re-download station metadata")
    ap.add_argument("--force", action="store_true", help="Overwrite existing air_quality field")
    ap.add_argument("--limit-files", type=int, default=0, help="Only process N JSON files (0 = all)")
    ap.add_argument("--limit-records", type=int, default=0, help="Only update N records total (0 = unlimited)")

    ap.add_argument("--fetch-measures", action="store_true", help="Also fetch pollutant measures (can be slow)")
    ap.add_argument(
        "--components",
        default="NO2,PM10,PM2",
        help="Comma-separated UBA component codes (default: NO2,PM10,PM2)",
    )
    ap.add_argument("--scope-id", type=int, default=2, help="UBA scope id (default: 2 = hourly average)")
    ap.add_argument("--date-from", default=None, help="Date from YYYY-MM-DD (default: 7 days ago)")
    ap.add_argument("--date-to", default=None, help="Date to YYYY-MM-DD (default: today)")
    ap.add_argument("--time-from", default="01", help="Hour start (0 is ambiguous in UBA API; default: 01)")
    ap.add_argument("--time-to", default="23", help="Hour end (default: 23)")
    ap.add_argument("--timeout-s", type=int, default=30, help="HTTP timeout for UBA calls (default: 30)")
    ap.add_argument("--force-measures", action="store_true", help="Ignore cached UBA measures and refetch")
    args = ap.parse_args()

    repo = repo_root()
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo)
    if not run_dir:
        raise SystemExit("No db_creation run dir found. Pass --run-dir get_data/past/immoscout24/db_creation/data/json/run_.../")
    if not run_dir.is_absolute():
        run_dir = (repo / run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    cache_root = repo / "get_data" / "past" / "immoscout24" / "cache" / "air_quality"
    station_cache = cache_root / "uba_stations.json"
    stations = load_station_cache(station_cache, refresh=bool(args.refresh_stations))
    if not stations:
        raise SystemExit("No active UBA stations loaded.")

    grid, lats, lons = build_station_grid(stations, cell_deg=float(args.cell_deg))
    log(f"UBA stations: {len(stations)} (cached: {station_cache})")

    json_files = sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")
    if args.limit_files and args.limit_files > 0:
        json_files = json_files[: args.limit_files]
    if not json_files:
        raise SystemExit(f"No JSON batch files found in: {run_dir}")

    date_to = args.date_to or time.strftime("%Y-%m-%d", time.gmtime())
    if args.date_from:
        date_from = args.date_from
    else:
        # 7 days back
        t = time.time() - 7 * 86400
        date_from = time.strftime("%Y-%m-%d", time.gmtime(t))

    comp_codes = [c.strip().upper() for c in str(args.components).split(",") if c.strip()]
    comp_ids: list[tuple[str, int]] = []
    for c in comp_codes:
        cid = UBA_COMPONENT_CODE_TO_ID.get(c)
        if cid is None:
            raise SystemExit(f"Unknown component code {c!r}. Known: {sorted(UBA_COMPONENT_CODE_TO_ID.keys())}")
        comp_ids.append((c, int(cid)))

    log(f"Repo root: {repo}")
    log(f"Run dir: {run_dir}")
    log(f"Field: data.{args.field}")
    log(f"Files: {len(json_files)}")
    if args.fetch_measures:
        log(
            f"Measures: components={','.join([c for c,_ in comp_ids])} scope_id={args.scope_id} "
            f"range={date_from} {args.time_from}:00 .. {date_to} {args.time_to}:00"
        )

    updated_files = 0
    updated_records = 0
    skipped_existing = 0
    missing_geo = 0
    loc_cache: dict[tuple[float, float], dict[str, Any]] = {}
    measures_cache: dict[tuple[int, int], dict[str, Any]] = {}
    measures_dir = cache_root / "uba_measures"

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

            if not args.force and isinstance(enrich.get("air_quality"), dict) and isinstance(enrich["air_quality"].get("uba_station"), dict):
                skipped_existing += 1
                continue

            latlon = latlon_from_record(rec, field=args.field)
            if not latlon:
                missing_geo += 1
                continue

            lat, lon = latlon
            k = (round(lat, 4), round(lon, 4))
            if k in loc_cache:
                aq = loc_cache[k]
            else:
                ns = nearest_station(
                    stations=stations,
                    grid=grid,
                    lats=lats,
                    lons=lons,
                    cell_deg=float(args.cell_deg),
                    lat=k[0],
                    lon=k[1],
                    max_rings=int(args.max_rings),
                )
                if not ns:
                    aq = {"uba_station": None, "error": "no_station_found"}
                else:
                    st, d_m = ns
                    aq = {"uba_station": {**st, "dist_m": d_m}}
                loc_cache[k] = aq

            if args.fetch_measures and aq.get("uba_station") and isinstance(aq["uba_station"], dict):
                st_id = int(aq["uba_station"]["id"])
                measures_obj: dict[str, Any] = {}
                for code, cid in comp_ids:
                    mk = (st_id, cid)
                    if mk in measures_cache and not args.force_measures:
                        measures_obj[code] = measures_cache[mk]
                        continue
                    m = get_measures_cached(
                        measures_dir,
                        station_id=st_id,
                        component_id=cid,
                        scope_id=int(args.scope_id),
                        date_from=str(date_from),
                        time_from=str(args.time_from),
                        date_to=str(date_to),
                        time_to=str(args.time_to),
                        timeout_s=int(args.timeout_s),
                        force=bool(args.force_measures),
                    )
                    measures_cache[mk] = m
                    measures_obj[code] = m
                aq = {**aq, "uba_measures": measures_obj}

            enrich["air_quality"] = aq
            changed = True
            updated_records += 1

        if changed:
            atomic_write_json(p, arr)
            updated_files += 1
            log(f"Updated: {p.name}")

        if args.limit_records and updated_records >= args.limit_records:
            break

    marker = run_dir / ".air_quality_uba_enriched"
    atomic_write_json(
        marker,
        {
            "field": args.field,
            "cell_deg": float(args.cell_deg),
            "max_rings": int(args.max_rings),
            "fetch_measures": bool(args.fetch_measures),
            "components": comp_codes,
            "scope_id": int(args.scope_id),
            "date_from": date_from,
            "date_to": date_to,
            "time_from": str(args.time_from),
            "time_to": str(args.time_to),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_files": updated_files,
            "updated_records": updated_records,
            "skipped_existing": skipped_existing,
            "missing_geo": missing_geo,
            "station_cache": str(station_cache),
        },
    )
    log(
        f"Done. updated_files={updated_files} updated_records={updated_records} "
        f"skipped_existing={skipped_existing} missing_geo={missing_geo} marker={marker}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
