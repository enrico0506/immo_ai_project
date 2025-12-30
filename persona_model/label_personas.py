#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


PERSONAS: list[str] = [
    "student",
    "young_professional",
    "family_with_kids",
    "senior",
    "public_transit_commuter",
    "car_commuter",
    "nightlife_lover",
    "nature_lover",
    "quiet_seeker",
    "budget_sensitive",
    "luxury",
    "remote_worker",
]


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def repo_root() -> Path:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
        return Path(out.decode().strip())
    except Exception:
        return Path(__file__).resolve().parents[1]


def newest_run_dir(repo: Path) -> Optional[Path]:
    d = repo / "get_data" / "past" / "immoscout24" / "db_creation" / "data" / "json"
    candidates = sorted(d.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def get_nested(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        fv = float(v)
        return fv if math.isfinite(fv) else None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        s = s.replace("€", "").replace("EUR", "").replace("m²", "").replace("qm", "")
        s = s.replace(" ", "").replace("\u00a0", "")
        if s.count(",") == 1 and s.count(".") >= 1:
            s = s.replace(".", "").replace(",", ".")
        elif s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        s = re.sub(r"[^0-9.+-]", "", s)
        try:
            fv = float(s)
        except Exception:
            return None
        return fv if math.isfinite(fv) else None
    return None


def iter_records(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"Expected JSON array in {path}, got {type(obj)}")
    return obj  # type: ignore[return-value]


def load_existing_labels(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            eid = obj.get("expose_id")
            if eid is None:
                continue
            seen.add(str(eid))
    return seen


def format_score(x: Any) -> str:
    fv = as_float(x)
    if fv is None:
        return "-"
    return f"{fv:.3f}" if 0.0 <= fv <= 1.0 else f"{fv:.2f}"


def print_listing_summary(data: dict[str, Any]) -> None:
    title = str(data.get("title") or "").strip()
    addr = str(data.get("address") or "").strip()
    typ = str(data.get("typ") or "").strip()
    expose_id = str(data.get("expose_id") or "")
    url = str(data.get("url") or "")
    price = data.get("kaufpreis") or data.get("kaufpreis_num")
    price_m2 = data.get("preis_m2") or data.get("preis_m2_num")
    area = data.get("wohnflache_ca") or data.get("wohnflache_ca_num")
    rooms = data.get("zimmer") or data.get("zimmer_num")
    floor = data.get("etage") or data.get("etage_num")
    year = data.get("baujahr") or data.get("baujahr_num")

    over = data.get("overpass_enrichment") or {}
    scores = over.get("scores") or {}

    print("\n" + "=" * 90)
    print(f"expose_id: {expose_id}")
    print(f"url:      {url}")
    if title:
        print(f"title:    {title}")
    if typ:
        print(f"type:     {typ}")
    if addr:
        print(f"address:  {addr}")
    print(f"price:    {price} | price_m2: {price_m2} | area_m2: {area} | rooms: {rooms} | floor: {floor} | year: {year}")

    if isinstance(scores, dict) and scores:
        keys = [
            "living_standard_score_0_100",
            "score_percentile_in_city",
            "score_z_in_city",
            "geocode_confidence_0_1",
            "walkability_score_0_1",
            "education_score_0_1",
            "healthcare_score_0_1",
            "green_quality_score_0_1",
            "noise_proxy_score_0_1",
            "transit_score_0_1",
            "nightlife_score_0_1",
            "safety_proxy_score_0_1",
            "family_friendliness_score_0_1",
        ]
        row = " | ".join(f"{k}={format_score(scores.get(k))}" for k in keys)
        print(row)

    desc = str(data.get("description") or "").strip()
    if desc:
        snippet = re.sub(r"\\s+", " ", desc)[:500]
        print(f"desc:     {snippet}{'...' if len(desc) > 500 else ''}")


def parse_persona_input(s: str) -> Optional[list[str]]:
    """
    Returns:
      - None => user wants to quit
      - [] => skip
      - [persona,...] => positives
    """
    t = (s or "").strip()
    if not t:
        return []
    t_low = t.lower()
    if t_low in ("q", "quit", "exit"):
        return None
    if t_low in ("s", "skip"):
        return []
    if t_low in ("?", "help"):
        return ["__HELP__"]

    parts = [p.strip() for p in t.split(",") if p.strip()]
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive manual labeling for persona classification.")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="db_creation run dir (default: newest run_* under get_data/past/immoscout24/db_creation/data/json/)",
    )
    ap.add_argument(
        "--labels-out",
        default=None,
        help="Output JSONL labels file (default: persona_model/data/persona_labels_<run>.jsonl)",
    )
    ap.add_argument("--sample", type=int, default=200, help="How many listings to label (default: 200)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    ap.add_argument("--force", action="store_true", help="Overwrite labels file (dangerous)")
    args = ap.parse_args()

    repo = repo_root()
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo)
    if not run_dir:
        raise SystemExit("No run dir found. Pass --run-dir .../run_YYYYMMDD_HHMMSS")
    if not run_dir.is_absolute():
        run_dir = (repo / run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    out_path = (
        Path(args.labels_out)
        if args.labels_out
        else repo / "persona_model" / "data" / f"persona_labels_{run_dir.name}.jsonl"
    )
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and args.force:
        out_path.unlink()

    existing = load_existing_labels(out_path)
    rng = random.Random(int(args.seed))

    files = iter_records(run_dir)
    if not files:
        raise SystemExit(f"No JSON batch files found in: {run_dir}")

    log(f"Repo root: {repo}")
    log(f"Run dir: {run_dir} files={len(files)}")
    log(f"Labels out: {out_path}")
    log(f"Already labeled: {len(existing)} expose_ids")
    log("Personas: " + ", ".join(PERSONAS))
    log("Input help: comma-separated personas | 's' skip | 'q' quit | '?' help")

    # Reservoir sample unlabeled items
    k = max(1, int(args.sample))
    chosen: list[dict[str, Any]] = []
    seen = 0

    for fp in files:
        arr = load_json_array(fp)
        for rec in arr:
            if not isinstance(rec, dict):
                continue
            data = rec.get("data")
            if not isinstance(data, dict):
                continue
            eid = data.get("expose_id") or rec.get("expose_id")
            if eid is None:
                continue
            eid_s = str(eid)
            if eid_s in existing:
                continue

            seen += 1
            item = {"file": fp.name, "expose_id": eid_s, "url": str(data.get("url") or ""), "data": data}
            if len(chosen) < k:
                chosen.append(item)
            else:
                j = rng.randrange(seen)
                if j < k:
                    chosen[j] = item

    if not chosen:
        log("No unlabeled records found (or sample=0). Nothing to do.")
        return 0

    # Deterministic order for session
    chosen.sort(key=lambda x: (x.get("file") or "", x.get("expose_id") or ""))
    log(f"Sampled for labeling: {len(chosen)} (unlabeled pool size={seen})")

    written = 0
    with out_path.open("a", encoding="utf-8") as out:
        for i, item in enumerate(chosen, 1):
            data = item["data"]
            print_listing_summary(data)
            while True:
                s = input(f"[{i}/{len(chosen)}] personas> ")
                parsed = parse_persona_input(s)
                if parsed is None:
                    log("Quit requested.")
                    return 0
                if parsed == ["__HELP__"]:
                    print("Valid personas:")
                    print("  " + ", ".join(PERSONAS))
                    print("Commands: s=skip, q=quit, ?=help")
                    continue
                if parsed == []:
                    # skip
                    break

                unknown = [p for p in parsed if p not in PERSONAS]
                if unknown:
                    print(f"Unknown persona(s): {', '.join(unknown)}")
                    continue

                labels = {p: (1 if p in parsed else 0) for p in PERSONAS}
                obj = {
                    "expose_id": str(item["expose_id"]),
                    "url": str(item["url"]),
                    "source_file": str(item["file"]),
                    "labels": labels,
                    "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "label_version": 1,
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
                written += 1
                existing.add(str(item["expose_id"]))
                break

    log(f"Done. labeled={written} labels_file={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

