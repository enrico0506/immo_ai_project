#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
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


def require_cmd(cmd: str) -> None:
    if shutil_which(cmd) is not None:
        return
    log(f"Missing dependency: {cmd}")
    log("Install commands:")
    log("  Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y gdal-bin")
    log("  Fedora:        sudo dnf install -y gdal")
    log("  Arch:          sudo pacman -S gdal")
    raise SystemExit(2)


def shutil_which(cmd: str) -> Optional[str]:
    from shutil import which

    return which(cmd)


def gdalloc_sample_val(tif: Path, *, lat: float, lon: float, timeout_s: int = 30) -> Optional[float]:
    """
    Sample a GeoTIFF at WGS84 coordinate using gdallocationinfo.
    Returns None if NoData/empty.
    """
    try:
        out = subprocess.check_output(
            ["gdallocationinfo", "-valonly", "-wgs84", str(tif), str(lon), str(lat)],
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    s = out.decode("utf-8", errors="replace").strip()
    if not s:
        return None
    try:
        v = float(s.split()[0])
    except Exception:
        return None
    if not (v == v):  # NaN
        return None
    return v


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
    ap = argparse.ArgumentParser(description="Attach noise-map values by sampling GeoTIFF rasters at listing locations.")
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
        "--lden-tif",
        default=None,
        help="GeoTIFF for L_den (day-evening-night) noise (default: geodata/amensity/noise/lden.tif)",
    )
    ap.add_argument(
        "--lnight-tif",
        default=None,
        help="GeoTIFF for L_night noise (default: geodata/amensity/noise/lnight.tif)",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing noise_map field")
    ap.add_argument("--timeout-s", type=int, default=20, help="Timeout per sample call (default: 20s)")
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

    lden = Path(args.lden_tif) if args.lden_tif else repo / "geodata" / "amensity" / "noise" / "lden.tif"
    ln = Path(args.lnight_tif) if args.lnight_tif else repo / "geodata" / "amensity" / "noise" / "lnight.tif"
    if not lden.is_absolute():
        lden = (repo / lden).resolve()
    if not ln.is_absolute():
        ln = (repo / ln).resolve()

    if not lden.exists() and not ln.exists():
        raise SystemExit(
            "No noise GeoTIFFs found. Put files at geodata/amensity/noise/lden.tif and/or lnight.tif, or pass --lden-tif/--lnight-tif."
        )

    require_cmd("gdallocationinfo")

    json_files = sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")
    if args.limit_files and args.limit_files > 0:
        json_files = json_files[: args.limit_files]
    if not json_files:
        raise SystemExit(f"No JSON batch files found in: {run_dir}")

    log(f"Repo root: {repo}")
    log(f"Run dir: {run_dir}")
    log(f"Field: data.{args.field}")
    if lden.exists():
        log(f"L_den: {lden}")
    if ln.exists():
        log(f"L_night: {ln}")
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

            if not args.force and isinstance(enrich.get("noise_map"), dict):
                skipped_existing += 1
                continue

            latlon = latlon_from_record(rec, field=args.field)
            if not latlon:
                missing_geo += 1
                continue
            lat, lon = latlon
            k = (round(lat, 4), round(lon, 4))
            if k in loc_cache:
                nm = loc_cache[k]
            else:
                nm = {"method": "gdallocationinfo", "lat": k[0], "lon": k[1]}
                if lden.exists():
                    nm["lden_db"] = gdalloc_sample_val(lden, lat=k[0], lon=k[1], timeout_s=int(args.timeout_s))
                if ln.exists():
                    nm["lnight_db"] = gdalloc_sample_val(ln, lat=k[0], lon=k[1], timeout_s=int(args.timeout_s))
                nm["sources"] = {"lden": str(lden) if lden.exists() else None, "lnight": str(ln) if ln.exists() else None}
                loc_cache[k] = nm

            enrich["noise_map"] = nm
            changed = True
            updated_records += 1

        if changed:
            atomic_write_json(p, arr)
            updated_files += 1
            log(f"Updated: {p.name}")

        if args.limit_records and updated_records >= args.limit_records:
            break

    marker = run_dir / ".noise_raster_enriched"
    atomic_write_json(
        marker,
        {
            "field": args.field,
            "lden_tif": str(lden) if lden.exists() else None,
            "lnight_tif": str(ln) if ln.exists() else None,
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
