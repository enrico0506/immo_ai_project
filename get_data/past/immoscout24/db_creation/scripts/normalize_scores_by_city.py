#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
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
    """
    Some upstream HTML/text fields may contain raw control characters that break JSON.
    We try to repair by replacing all ASCII control bytes with spaces and re-parsing.
    """
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


_RE_ZIP = re.compile(r"\\b(\\d{5})\\b")


def extract_city_from_address(addr_raw: str) -> Optional[str]:
    """
    Heuristic: find 5-digit postcode and take the following token as city.
    Example: '... 10115 Berlin ...' => 'Berlin'
    """
    s = (addr_raw or "").strip()
    if not s:
        return None
    m = _RE_ZIP.search(s)
    if not m:
        return None
    after = s.split(m.group(1), 1)[1].strip(" ,")
    if not after:
        return None
    city = after.split(",", 1)[0].strip()
    # Some addresses have trailing boilerplate text; keep first word chunk.
    city = city.split(" Die vollständige", 1)[0].strip()
    city = city.split(" The full address", 1)[0].strip()
    return city or None


def city_key_for_record(rec: dict[str, Any], *, field: str) -> Optional[str]:
    data = rec.get("data")
    if not isinstance(data, dict):
        return None

    addr_raw = data.get("address")
    if isinstance(addr_raw, str):
        c = extract_city_from_address(addr_raw)
        if c:
            return c

    enrich = data.get(field)
    if isinstance(enrich, dict):
        geo = enrich.get("geocode")
        if isinstance(geo, dict):
            meta = geo.get("meta")
            if isinstance(meta, dict):
                c = meta.get("city")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    return None


def score_value_for_record(rec: dict[str, Any], *, field: str, score_key: str) -> Optional[float]:
    data = rec.get("data")
    if not isinstance(data, dict):
        return None
    enrich = data.get(field)
    if not isinstance(enrich, dict):
        return None
    scores = enrich.get("scores")
    if not isinstance(scores, dict):
        return None
    v = scores.get(score_key)
    try:
        fv = float(v)
    except Exception:
        return None
    if not math.isfinite(fv):
        return None
    return fv


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    m = sum(values) / len(values)
    if len(values) < 2:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return (m, math.sqrt(max(0.0, var)))


def percentile(sorted_vals: list[float], x: float) -> float:
    n = len(sorted_vals)
    if n <= 1:
        return 0.5
    lo = bisect.bisect_left(sorted_vals, x)
    hi = bisect.bisect_right(sorted_vals, x)
    mid = (lo + hi - 1) / 2.0  # 0..n-1
    return max(0.0, min(1.0, mid / float(n - 1)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize attached Overpass living-standard scores by city.")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Target db_creation run dir (default: newest run_* under db_creation/data/json/)",
    )
    ap.add_argument(
        "--field",
        default="overpass_enrichment",
        help="Where the enrichment is stored (under record['data'][FIELD])",
    )
    ap.add_argument(
        "--score-key",
        default="living_standard_score_0_100",
        help="Score key inside enrichment.scores to normalize",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing city-normalized fields")
    ap.add_argument("--limit-files", type=int, default=0, help="Only process N JSON files (0 = all)")
    args = ap.parse_args()

    repo = repo_root()
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo)
    if not run_dir:
        raise SystemExit("No db_creation run dir found. Pass --run-dir get_data/past/immoscout24/db_creation/data/json/run_.../")
    if not run_dir.is_absolute():
        run_dir = (repo / run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    json_files = sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")
    if args.limit_files and args.limit_files > 0:
        json_files = json_files[: args.limit_files]
    if not json_files:
        raise SystemExit(f"No JSON batch files found in: {run_dir}")

    log(f"Repo root: {repo}")
    log(f"Run dir: {run_dir}")
    log(f"Field: data.{args.field}")
    log(f"Score key: {args.score_key}")
    log(f"Files: {len(json_files)}")

    # First pass: collect values per city
    city_values: dict[str, list[float]] = {}
    records_seen = 0
    records_with_score = 0
    records_with_city = 0
    for p in json_files:
        arr = load_json_file_with_repair(p)
        if not isinstance(arr, list):
            raise SystemExit(f"Expected a JSON array in {p}, got {type(arr)}")
        for rec in arr:
            if not isinstance(rec, dict):
                continue
            records_seen += 1
            city = city_key_for_record(rec, field=args.field)
            if city:
                records_with_city += 1
            score = score_value_for_record(rec, field=args.field, score_key=args.score_key)
            if score is None:
                continue
            records_with_score += 1
            if not city:
                continue
            ck = city.strip().lower()
            city_values.setdefault(ck, []).append(score)

    if not city_values:
        log("No city+score pairs found; nothing to normalize.")
        return 0

    # Precompute stats per city
    city_stats: dict[str, dict[str, Any]] = {}
    city_sorted: dict[str, list[float]] = {}
    for ck, vals in city_values.items():
        vals_sorted = sorted(vals)
        m, s = mean_std(vals)
        city_stats[ck] = {"n": len(vals), "mean": m, "std": s}
        city_sorted[ck] = vals_sorted

    log(f"Cities with values: {len(city_stats)}")

    # Second pass: write fields
    updated_files = 0
    updated_records = 0
    skipped_existing = 0
    for p in json_files:
        arr = load_json_file_with_repair(p)
        if not isinstance(arr, list):
            raise SystemExit(f"Expected a JSON array in {p}, got {type(arr)}")
        changed = False
        for rec in arr:
            if not isinstance(rec, dict):
                continue
            data = rec.get("data")
            if not isinstance(data, dict):
                continue
            enrich = data.get(args.field)
            if not isinstance(enrich, dict):
                continue
            scores = enrich.get("scores")
            if not isinstance(scores, dict):
                continue
            city = city_key_for_record(rec, field=args.field)
            if not city:
                continue
            ck = city.strip().lower()
            stats = city_stats.get(ck)
            svals = city_sorted.get(ck)
            if not stats or not svals:
                continue
            score = score_value_for_record(rec, field=args.field, score_key=args.score_key)
            if score is None:
                continue

            if not args.force and ("score_percentile_in_city" in scores or "score_z_in_city" in scores):
                skipped_existing += 1
                continue

            pct = percentile(svals, score)
            std = float(stats["std"])
            mu = float(stats["mean"])
            z = 0.0 if std <= 1e-9 else (score - mu) / std

            scores["score_percentile_in_city"] = pct
            scores["score_z_in_city"] = z
            scores["score_city_norm"] = city
            scores["score_city_norm_n"] = int(stats["n"])
            changed = True
            updated_records += 1

        if changed:
            atomic_write_json(p, arr)
            updated_files += 1
            log(f"Updated: {p.name}")

    marker = run_dir / ".city_normalization"
    marker_obj = {
        "field": args.field,
        "score_key": args.score_key,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records_seen": records_seen,
        "records_with_city": records_with_city,
        "records_with_score": records_with_score,
        "updated_files": updated_files,
        "updated_records": updated_records,
        "skipped_existing": skipped_existing,
        "cities": city_stats,
    }
    atomic_write_json(marker, marker_obj)
    log(
        f"Done. updated_files={updated_files} updated_records={updated_records} skipped_existing={skipped_existing} "
        f"(marker: {marker})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

