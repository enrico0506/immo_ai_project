#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


_RE_HIDDEN = re.compile(r"\\bDie vollständige Adresse.*$", flags=re.IGNORECASE)
_RE_ZIP = re.compile(r"\\b(\\d{5})\\b")
_RE_STREET_HN = re.compile(r"^(?P<street>.+?)\\s+(?P<hn>\\d+[a-zA-Z]?)$")


def parse_address_has_street_hn(raw: str) -> bool:
    s = (raw or "").strip()
    s = _RE_HIDDEN.sub("", s).strip()
    s = re.sub(r"\\s+", " ", s)
    postcode = None
    m = _RE_ZIP.search(s)
    if m:
        postcode = m.group(1)
    before_zip = s
    if postcode:
        before_zip = s.split(postcode, 1)[0].strip(" ,")
    left = before_zip.split(",", 1)[0].strip()
    if not left:
        return False
    if not re.search(r"\\d", left):
        return False
    return bool(_RE_STREET_HN.match(left))


def compute_geocode_confidence(has_street_hn: bool, geocode: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    method = str(geocode.get("method") or "")
    precision = str(geocode.get("precision") or "")
    meta = geocode.get("meta") if isinstance(geocode.get("meta"), dict) else {}

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
        "method": method,
        "precision": precision,
        "has_street_housenumber": bool(has_street_hn),
        "postcode_samples": sampled_n,
        "address_matches": matched_n,
    }

    if method == "overpass_address" or precision == "address":
        conf = 0.95
        if matched_n > 0:
            conf = min(0.99, conf + 0.01 * min(5, matched_n))
        detail["rule"] = "address_match"
    elif method == "overpass_postcode" or precision == "postcode":
        sample_factor = 1.0 - math.exp(-float(max(0, sampled_n)) / 60.0)
        conf = 0.45 + 0.30 * sample_factor
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Add geocode_confidence to attached overpass_enrichment payloads.")
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
    ap.add_argument("--force", action="store_true", help="Overwrite existing confidence fields")
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

    json_files = sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")
    if args.limit_files and args.limit_files > 0:
        json_files = json_files[: args.limit_files]
    if not json_files:
        raise SystemExit(f"No JSON batch files found in: {run_dir}")

    log(f"Repo root: {repo}")
    log(f"Run dir: {run_dir}")
    log(f"Field: data.{args.field}")
    log(f"Files: {len(json_files)}")

    updated_files = 0
    updated_records = 0
    skipped_existing = 0
    missing_enrichment = 0

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
                missing_enrichment += 1
                continue
            geocode = enrich.get("geocode")
            if not isinstance(geocode, dict):
                continue
            scores = enrich.get("scores")
            if not isinstance(scores, dict):
                continue

            if (
                not args.force
                and isinstance(geocode.get("confidence_0_1"), (int, float))
                and isinstance(scores.get("geocode_confidence_0_1"), (int, float))
            ):
                skipped_existing += 1
                continue

            addr_raw = data.get("address")
            has_street_hn = parse_address_has_street_hn(addr_raw if isinstance(addr_raw, str) else "")
            conf, detail = compute_geocode_confidence(has_street_hn, geocode)

            geocode["confidence_0_1"] = conf
            geocode["confidence_detail"] = detail
            scores["geocode_confidence_0_1"] = conf
            scores["geocode_confidence"] = conf

            changed = True
            updated_records += 1

        if changed:
            atomic_write_json(p, arr)
            updated_files += 1
            log(f"Updated: {p.name}")

        if args.limit_records and updated_records >= args.limit_records:
            break

    marker = run_dir / ".geocode_confidence_enriched"
    atomic_write_json(
        marker,
        {
            "field": args.field,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_files": updated_files,
            "updated_records": updated_records,
            "skipped_existing": skipped_existing,
            "missing_enrichment": missing_enrichment,
        },
    )
    log(
        f"Done. updated_files={updated_files} updated_records={updated_records} "
        f"skipped_existing={skipped_existing} marker={marker}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

