#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def human_duration(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "?"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Progress:
    def __init__(self, *, label: str, total: Optional[int], enabled: bool) -> None:
        self.label = label
        self.total = total
        self.enabled = enabled and is_tty()
        self.start = time.time()
        self.last_draw = 0.0
        self.done = 0
        self._spinner = "|/-\\"
        self._spin_i = 0
        self._extra = ""

    def set_extra(self, extra: str) -> None:
        self._extra = extra

    def tick(self, n: int = 1) -> None:
        self.done += int(n)
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_draw < 0.2 and (self.total is None or self.done < self.total):
            return
        self.last_draw = now

        elapsed = now - self.start
        rate = self.done / max(0.001, elapsed)
        extra = f" {self._extra}" if self._extra else ""

        if self.total:
            pct = min(1.0, self.done / max(1, self.total))
            bar_w = 28
            filled = int(pct * bar_w)
            bar = "#" * filled + "-" * (bar_w - filled)
            eta = (self.total - self.done) / max(0.001, rate)
            msg = (
                f"\r[{bar}] {pct*100:5.1f}% {self.done}/{self.total} "
                f"{rate:6.1f}/s ETA {human_duration(eta)} {self.label}{extra}"
            )
        else:
            self._spin_i = (self._spin_i + 1) % len(self._spinner)
            spin = self._spinner[self._spin_i]
            msg = f"\r{spin} processed={self.done} {rate:6.1f}/s {self.label}{extra}"

        print(msg[:240], end="", flush=True)

    def finish(self) -> None:
        if self.enabled:
            print("", flush=True)


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


_RE_ZIP = re.compile(r"\\b(\\d{5})\\b")


def city_from_address(addr: str) -> Optional[str]:
    s = (addr or "").strip()
    if not s:
        return None
    m = _RE_ZIP.search(s)
    if not m:
        return None
    after = s.split(m.group(1), 1)[1].strip(" ,")
    if not after:
        return None
    city = after.split(",", 1)[0].strip()
    city = city.split(" Die vollständige", 1)[0].strip()
    return city or None


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
        # keep digits and separators
        s = s.replace("€", "").replace("EUR", "").replace("m²", "").replace("qm", "")
        s = s.replace(" ", "").replace("\u00a0", "")
        # German format: 1.234,56
        if s.count(",") == 1 and s.count(".") >= 1:
            s = s.replace(".", "").replace(",", ".")
        elif s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        else:
            # 1.234 or 1234.56
            pass
        # strip non-numeric trailing
        s = re.sub(r"[^0-9.+-]", "", s)
        try:
            fv = float(s)
        except Exception:
            return None
        return fv if math.isfinite(fv) else None
    return None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def lin_scale(x: Optional[float], x0: float, x1: float) -> float:
    if x is None:
        return 0.0
    if x1 == x0:
        return 0.0
    return clamp01((float(x) - x0) / (x1 - x0))


def keyword_score(text: str, keywords: list[str]) -> float:
    if not text:
        return 0.0
    t = text.lower()
    hits = 0
    for kw in keywords:
        if kw in t:
            hits += 1
    return clamp01(hits / max(1.0, float(min(5, len(keywords)))))


PERSONA_KEYWORDS: dict[str, list[str]] = {
    "student": ["student", "uni", "campus", "wg", "azubi", "ausbildung"],
    "family_with_kids": ["familie", "familienfreund", "kinder", "kita", "kindergarten", "schule", "spielplatz", "garten"],
    "senior": ["senior", "barrierefrei", "rollstuhl", "aufzug", "lift", "ebenerdig", "altersgerecht"],
    "remote_worker": ["homeoffice", "home office", "arbeitszimmer", "office", "schreibtisch", "glasfaser"],
    "car_commuter": ["garage", "stellplatz", "carport", "parkplatz", "tiefgarage"],
    "luxury": ["luxus", "exklusiv", "premium", "penthouse", "high-end", "neubau", "designer"],
}


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


def compute_persona_scores(
    *,
    scores: dict[str, Any],
    price_m2_percentile: Optional[float],
    area_m2: Optional[float],
    rooms: Optional[float],
    parking_count: Optional[float],
    text: str,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    # Base 0..1 features from overpass
    transit = clamp01(float(scores.get("transit_score_0_1") or 0.0))
    amenities = clamp01(float(scores.get("amenities_score_0_1") or 0.0))
    green_q = clamp01(float(scores.get("green_quality_score_0_1") or 0.0))
    nightlife = clamp01(float(scores.get("nightlife_score_0_1") or 0.0))
    walk = clamp01(float(scores.get("walkability_score_0_1") or 0.0))
    edu = clamp01(float(scores.get("education_score_0_1") or 0.0))
    health = clamp01(float(scores.get("healthcare_score_0_1") or 0.0))
    safety = clamp01(float(scores.get("safety_proxy_score_0_1") or 0.0))
    quiet = clamp01(float(scores.get("noise_proxy_score_0_1") or 0.0))
    family = clamp01(float(scores.get("family_friendliness_score_0_1") or 0.0))

    # Derived features
    if price_m2_percentile is None:
        budget = 0.5
    else:
        budget = clamp01(1.0 - float(price_m2_percentile))  # cheaper => higher budget score

    area_large = lin_scale(area_m2, 70.0, 140.0)
    area_small = 1.0 - lin_scale(area_m2, 35.0, 90.0)
    rooms_large = lin_scale(rooms, 2.0, 5.0)
    rooms_small = 1.0 - lin_scale(rooms, 1.0, 3.0)

    parking = 1.0 if (parking_count is not None and parking_count > 0) else 0.0

    kw = {p: keyword_score(text, PERSONA_KEYWORDS.get(p, [])) for p in PERSONA_KEYWORDS.keys()}

    # Helper to score + keep explanation
    explain: dict[str, list[tuple[str, float]]] = {p: [] for p in PERSONAS}

    def add(p: str, name: str, w: float, x: float) -> None:
        explain[p].append((name, w * x))

    # Personas (weights sum ~1; scores are 0..1)
    # Student: cheap + transit + nightlife + walkability + small units + keyword
    add("student", "budget", 0.30, budget)
    add("student", "transit", 0.25, transit)
    add("student", "nightlife", 0.15, nightlife)
    add("student", "walkability", 0.15, walk)
    add("student", "small_area", 0.10, area_small)
    add("student", "student_keywords", 0.05, kw.get("student", 0.0))

    # Young professional: transit + amenities + safety + walkability + (slight) nightlife
    add("young_professional", "transit", 0.25, transit)
    add("young_professional", "amenities", 0.20, amenities)
    add("young_professional", "walkability", 0.20, walk)
    add("young_professional", "safety_proxy", 0.20, safety)
    add("young_professional", "nightlife", 0.10, nightlife)
    add("young_professional", "budget", 0.05, budget)

    # Family: family friendliness + space
    add("family_with_kids", "family_friendliness", 0.60, family)
    add("family_with_kids", "large_area", 0.15, area_large)
    add("family_with_kids", "more_rooms", 0.15, rooms_large)
    add("family_with_kids", "family_keywords", 0.10, kw.get("family_with_kids", 0.0))

    # Senior: healthcare + walkability + quiet + accessibility keywords
    add("senior", "healthcare", 0.30, health)
    add("senior", "walkability", 0.25, walk)
    add("senior", "quiet", 0.20, quiet)
    add("senior", "safety_proxy", 0.15, safety)
    add("senior", "senior_keywords", 0.10, kw.get("senior", 0.0))

    # Public transport commuter: transit + walkability + (optional) budget
    add("public_transit_commuter", "transit", 0.70, transit)
    add("public_transit_commuter", "walkability", 0.25, walk)
    add("public_transit_commuter", "budget", 0.05, budget)

    # Car commuter: parking + proximity to major roads proxy (inverse quiet) + space
    add("car_commuter", "parking", 0.45, parking + 0.4 * kw.get("car_commuter", 0.0))
    add("car_commuter", "major_road_access_proxy", 0.35, 1.0 - quiet)
    add("car_commuter", "space", 0.20, area_large)

    # Nightlife lover: nightlife + transit + amenities (penalize quiet a bit)
    add("nightlife_lover", "nightlife", 0.55, nightlife)
    add("nightlife_lover", "transit", 0.25, transit)
    add("nightlife_lover", "amenities", 0.15, amenities)
    add("nightlife_lover", "not_too_quiet", 0.05, 1.0 - quiet)

    # Nature lover: green + quiet + walkability
    add("nature_lover", "green_quality", 0.60, green_q)
    add("nature_lover", "quiet", 0.20, quiet)
    add("nature_lover", "walkability", 0.20, walk)

    # Quiet seeker: quiet + green + safety, penalize nightlife
    add("quiet_seeker", "quiet", 0.65, quiet)
    add("quiet_seeker", "green_quality", 0.20, green_q)
    add("quiet_seeker", "safety_proxy", 0.15, safety)
    # apply penalty later using nightlife

    # Budget sensitive: budget + (some) amenities/transit
    add("budget_sensitive", "budget", 0.70, budget)
    add("budget_sensitive", "transit", 0.15, transit)
    add("budget_sensitive", "amenities", 0.15, amenities)

    # Luxury: expensive + large + keywords + (some) safety/quiet
    luxury_kw = kw.get("luxury", 0.0)
    expensive = 1.0 - budget
    add("luxury", "expensive_proxy", 0.35, expensive)
    add("luxury", "large_area", 0.25, area_large)
    add("luxury", "luxury_keywords", 0.20, luxury_kw)
    add("luxury", "quiet", 0.10, quiet)
    add("luxury", "safety_proxy", 0.10, safety)

    # Remote worker: quiet + space + walkability + keyword
    add("remote_worker", "quiet", 0.25, quiet)
    add("remote_worker", "space", 0.20, area_large)
    add("remote_worker", "walkability", 0.15, walk)
    add("remote_worker", "green_quality", 0.10, green_q)
    add("remote_worker", "remote_keywords", 0.30, kw.get("remote_worker", 0.0))

    # Final scores
    out: dict[str, float] = {}
    out_expl: dict[str, list[str]] = {}
    for p in PERSONAS:
        total = sum(v for _, v in explain[p])
        if p == "quiet_seeker":
            total *= 1.0 - 0.35 * nightlife
        out[p] = clamp01(total)
        top = sorted(explain[p], key=lambda x: x[1], reverse=True)[:3]
        out_expl[p] = [n for n, _ in top if _ > 0]

    return out, out_expl


def percentile(sorted_vals: list[float], x: float) -> float:
    n = len(sorted_vals)
    if n <= 1:
        return 0.5
    lo = bisect.bisect_left(sorted_vals, x)
    hi = bisect.bisect_right(sorted_vals, x)
    mid = (lo + hi - 1) / 2.0
    return max(0.0, min(1.0, mid / float(n - 1)))


def record_iter(run_dir: Path) -> list[Path]:
    files = sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")
    return files


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"Expected JSON array in {path}, got {type(obj)}")
    return obj  # type: ignore[return-value]


def extract_price_m2(data: dict[str, Any]) -> Optional[float]:
    v = data.get("preis_m2_num")
    fv = as_float(v)
    if fv is not None and fv > 0:
        return fv
    # fallback: kaufpreis / wohnfläche
    kp = as_float(data.get("kaufpreis_num"))
    area = as_float(data.get("wohnflache_ca_num"))
    if kp is not None and area is not None and kp > 0 and area > 0:
        return kp / area
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Rule-based persona classification for ImmoScout listings.")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="db_creation run dir (default: newest run_* under get_data/past/immoscout24/db_creation/data/json/)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output JSONL (default: persona_model/output/personas_<run>.jsonl)",
    )
    ap.add_argument("--top-k", type=int, default=3, help="How many top personas to include (default: 3)")
    ap.add_argument("--min-score", type=float, default=0.0, help="Only keep personas with score >= this (default: 0)")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a progress bar (default: enabled when stdout is a TTY).",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    args = ap.parse_args()

    repo = repo_root()
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo)
    if not run_dir:
        raise SystemExit("No run dir found. Pass --run-dir .../run_YYYYMMDD_HHMMSS")
    if not run_dir.is_absolute():
        run_dir = (repo / run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    run_name = run_dir.name
    out_path = Path(args.output) if args.output else repo / "persona_model" / "output" / f"personas_{run_name}.jsonl"
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        log(f"Output exists (use --force to overwrite): {out_path}")
        return 0

    files = record_iter(run_dir)
    if not files:
        raise SystemExit(f"No JSON batch files found in: {run_dir}")

    log(f"Repo root: {repo}")
    log(f"Run dir: {run_dir} files={len(files)}")
    log(f"Output: {out_path}")

    progress_enabled = args.progress if args.progress is not None else (is_tty() and os.environ.get("NO_PROGRESS") != "1")

    # Pass 1: price distributions per city
    city_prices: dict[str, list[float]] = {}
    total_seen = 0
    total_with_price = 0
    records_total = 0
    prog1 = Progress(label="pass1(prices)", total=None, enabled=progress_enabled)
    for p in files:
        prog1.set_extra(p.name)
        arr = load_json_array(p)
        records_total += len(arr)
        for rec in arr:
            if not isinstance(rec, dict):
                continue
            data = rec.get("data")
            if not isinstance(data, dict):
                continue
            city = get_nested(data, ["overpass_enrichment", "scores", "score_city_norm"])
            if not isinstance(city, str) or not city.strip():
                city = city_from_address(str(data.get("address") or ""))
            if not isinstance(city, str) or not city.strip():
                continue
            price_m2 = extract_price_m2(data)
            total_seen += 1
            if price_m2 is None:
                prog1.tick()
                continue
            total_with_price += 1
            ck = city.strip().lower()
            city_prices.setdefault(ck, []).append(float(price_m2))
            prog1.tick()
    prog1.finish()

    city_price_sorted: dict[str, list[float]] = {k: sorted(v) for k, v in city_prices.items() if v}
    log(f"Price distributions: cities={len(city_price_sorted)} records_with_price={total_with_price}/{max(1,total_seen)}")

    # Pass 2: compute persona scores and write JSONL
    counts: dict[str, int] = {p: 0 for p in PERSONAS}
    written = 0
    prog2 = Progress(label="pass2(classify)", total=records_total, enabled=progress_enabled)
    with out_path.open("w", encoding="utf-8") as out:
        for p in files:
            prog2.set_extra(p.name)
            arr = load_json_array(p)
            for rec in arr:
                if not isinstance(rec, dict):
                    prog2.tick()
                    continue
                data = rec.get("data")
                if not isinstance(data, dict):
                    prog2.tick()
                    continue
                expose_id = data.get("expose_id") or rec.get("expose_id")
                url = data.get("url") or rec.get("url")

                over = data.get("overpass_enrichment") or {}
                scores = (over.get("scores") or {}) if isinstance(over, dict) else {}
                if not isinstance(scores, dict):
                    scores = {}

                city = scores.get("score_city_norm")
                if not isinstance(city, str) or not city.strip():
                    city = city_from_address(str(data.get("address") or ""))

                price_m2 = extract_price_m2(data)
                price_pct = None
                if price_m2 is not None and isinstance(city, str) and city.strip():
                    sv = city_price_sorted.get(city.strip().lower())
                    if sv:
                        price_pct = percentile(sv, float(price_m2))

                area_m2 = as_float(data.get("wohnflache_ca_num")) or as_float(data.get("wohnflache_ca"))
                rooms = as_float(data.get("zimmer_num")) or as_float(data.get("zimmer"))
                parking_count = as_float(data.get("garage_stellplatz_num")) or as_float(data.get("garage_stellplatz"))

                text = " ".join(
                    str(x or "")
                    for x in (
                        data.get("title"),
                        data.get("description"),
                        data.get("location_text"),
                    )
                )

                persona_scores, persona_expl = compute_persona_scores(
                    scores=scores,
                    price_m2_percentile=price_pct,
                    area_m2=area_m2,
                    rooms=rooms,
                    parking_count=parking_count,
                    text=text,
                )

                # Downweight low-confidence geocodes
                geo_conf = as_float(scores.get("geocode_confidence_0_1")) or as_float(get_nested(over, ["geocode", "confidence_0_1"]))
                if geo_conf is not None:
                    w = max(0.25, clamp01(float(geo_conf)))
                    persona_scores = {k: clamp01(v * w) for k, v in persona_scores.items()}

                # top-k ranking
                items = sorted(persona_scores.items(), key=lambda kv: kv[1], reverse=True)
                items = [(k, v) for k, v in items if v >= float(args.min_score)]
                topk = items[: max(1, int(args.top_k))]
                primary_persona, primary_score = (topk[0][0], topk[0][1]) if topk else ("unknown", 0.0)

                for p_name, sc in topk:
                    if sc >= 0.55:
                        counts[p_name] += 1

                out_obj = {
                    "expose_id": str(expose_id) if expose_id is not None else None,
                    "url": str(url) if url is not None else None,
                    "city": city if isinstance(city, str) else None,
                    "price_m2": price_m2,
                    "price_m2_percentile_in_city": price_pct,
                    "primary_persona": primary_persona,
                    "primary_score": primary_score,
                    "top_personas": [{"persona": k, "score": v, "top_factors": persona_expl.get(k, [])} for k, v in topk],
                    "personas": persona_scores,
                }
                out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                written += 1
                prog2.tick()
    prog2.finish()

    top_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    log(f"Done. wrote={written} output={out_path}")
    log("Top persona counts (score>=0.55): " + ", ".join(f"{k}={v}" for k, v in top_counts[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
