#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scores(scores_jsonl: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """
    Returns (scores_sha256, mapping expose_id -> payload_to_attach).
    """
    scores_sha = sha256_file(scores_jsonl)
    mapping: dict[str, dict[str, Any]] = {}

    with scores_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            expose_id = obj.get("expose_id") or (obj.get("data") or {}).get("expose_id")
            if not expose_id:
                continue

            overpass = obj.get("overpass") or {}
            overpass_version = overpass.get("version")
            if not isinstance(overpass_version, int):
                overpass_version = 1
            payload = {
                "version": overpass_version,
                "scored_at": obj.get("scored_at") or None,
                "source_scores_file_sha256": scores_sha,
                "db_marker_hash": overpass.get("db_marker_hash"),
                "geocode": overpass.get("geocode"),
                "metrics": overpass.get("metrics"),
                "scores": overpass.get("scores"),
            }
            mapping[str(expose_id)] = payload

    return scores_sha, mapping


def newest_scored_file(repo: Path) -> Optional[Path]:
    d = repo / "get_data" / "past" / "immoscout24" / "data" / "processed"
    candidates = sorted(d.glob("immoscout_scored_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


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

        # Persist repaired canonical JSON for future tools.
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


def get_expose_id_from_record(rec: dict[str, Any]) -> Optional[str]:
    d = rec.get("data")
    if isinstance(d, dict) and d.get("expose_id"):
        return str(d["expose_id"])
    if rec.get("expose_id"):
        return str(rec["expose_id"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Attach Overpass living-standard scores into db_creation JSON files.")
    ap.add_argument(
        "--scores-jsonl",
        default=None,
        help="Scored JSONL (from score_exposes_living_standard.py). If omitted, uses newest immoscout_scored_*.jsonl",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="Target db_creation run dir (default: newest run_* under db_creation/data/json/)",
    )
    ap.add_argument(
        "--field",
        default="overpass_enrichment",
        help="Where to store the attached payload (under record['data'][FIELD])",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite even if already attached with same scores sha")
    ap.add_argument("--skip-bad-json", action="store_true", default=True, help="Skip unrecoverably broken JSON files")
    ap.add_argument("--fail-on-bad-json", action="store_true", help="Abort if a JSON batch file can't be parsed/repaired")
    ap.add_argument("--limit-files", type=int, default=0, help="Only process N JSON files (0 = all)")
    ap.add_argument("--limit-records", type=int, default=0, help="Only update N records total (0 = unlimited)")
    args = ap.parse_args()

    repo = repo_root()

    scores_path = Path(args.scores_jsonl) if args.scores_jsonl else newest_scored_file(repo)
    if not scores_path:
        raise SystemExit("No scores JSONL found. Run score_exposes_living_standard.py first or pass --scores-jsonl ...")
    if not scores_path.is_absolute():
        scores_path = (repo / scores_path).resolve()
    if not scores_path.exists():
        raise SystemExit(f"Scores JSONL not found: {scores_path}")

    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo)
    if not run_dir:
        raise SystemExit("No db_creation run dir found. Pass --run-dir get_data/past/immoscout24/db_creation/data/json/run_.../")
    if not run_dir.is_absolute():
        run_dir = (repo / run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    log(f"Repo root: {repo}")
    log(f"Scores: {scores_path}")
    log(f"Run dir: {run_dir}")
    log(f"Attach field: data.{args.field}")

    scores_sha, mapping = load_scores(scores_path)
    log(f"Loaded scores: {len(mapping)} exposes (scores_sha256={scores_sha[:12]}...)")

    json_files = sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")
    if args.limit_files and args.limit_files > 0:
        json_files = json_files[: args.limit_files]
    if not json_files:
        raise SystemExit(f"No JSON batch files found in: {run_dir}")
    log(f"Batch files: {len(json_files)}")

    total_records = 0
    total_updated = 0
    total_skipped_already = 0
    total_missing_score = 0
    total_bad_json = 0

    for p in json_files:
        try:
            arr = load_json_file_with_repair(p)
        except json.JSONDecodeError as e:
            total_bad_json += 1
            msg = f"JSON decode failed for {p}: {e}"
            if args.fail_on_bad_json:
                raise SystemExit(msg) from e
            log(f"WARNING: {msg} (skipping)")
            continue

        if not isinstance(arr, list):
            raise SystemExit(f"Expected a JSON array in {p}, got {type(arr)}")

        changed = False
        for rec in arr:
            if args.limit_records and total_updated >= args.limit_records:
                break
            if not isinstance(rec, dict):
                continue
            total_records += 1

            expose_id = get_expose_id_from_record(rec)
            if not expose_id:
                continue
            payload = mapping.get(expose_id)
            if not payload or not payload.get("scores"):
                total_missing_score += 1
                continue

            data = rec.get("data")
            if not isinstance(data, dict):
                continue

            existing = data.get(args.field)
            if (
                not args.force
                and isinstance(existing, dict)
                and existing.get("source_scores_file_sha256") == scores_sha
            ):
                total_skipped_already += 1
                continue

            # Fill scored_at if missing in the score file (older outputs)
            if payload.get("scored_at") is None:
                payload = {**payload, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

            data[args.field] = payload
            changed = True
            total_updated += 1

        if args.limit_records and total_updated >= args.limit_records:
            if changed:
                atomic_write_json(p, arr)
                log(f"Updated: {p.name} (partial, limit reached)")
            break

        if changed:
            atomic_write_json(p, arr)
            log(f"Updated: {p.name}")

    marker = run_dir / ".overpass_scores_attached"
    marker_obj = {
        "scores_jsonl": str(scores_path),
        "scores_sha256": scores_sha,
        "field": args.field,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records_seen": total_records,
        "records_updated": total_updated,
        "records_missing_score": total_missing_score,
        "records_skipped_already": total_skipped_already,
        "bad_json_files": total_bad_json,
    }
    atomic_write_json(marker, marker_obj)

    log(
        f"Done. records_seen={total_records} updated={total_updated} "
        f"missing_score={total_missing_score} skipped_already={total_skipped_already} bad_json_files={total_bad_json}"
    )
    log(f"Marker: {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
