#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import optimize  # type: ignore


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
        s = re_sub_non_numeric(s)
        try:
            fv = float(s)
        except Exception:
            return None
        return fv if math.isfinite(fv) else None
    return None


_RE_NON_NUM = None


def re_sub_non_numeric(s: str) -> str:
    global _RE_NON_NUM
    if _RE_NON_NUM is None:
        import re

        _RE_NON_NUM = re.compile(r"[^0-9.+-]")
    return _RE_NON_NUM.sub("", s)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def train_test_split_mask(expose_id: str, *, train_frac: float, val_frac: float, seed: str) -> str:
    """
    Deterministic split by hashing expose_id.
    Returns 'train' | 'val' | 'test'
    """
    h = hashlib.sha256((seed + ":" + expose_id).encode("utf-8")).digest()
    x = int.from_bytes(h[:2], "big") / 65535.0  # 0..1
    if x < train_frac:
        return "train"
    if x < train_frac + val_frac:
        return "val"
    return "test"


def percentile(sorted_vals: list[float], x: float) -> float:
    n = len(sorted_vals)
    if n <= 1:
        return 0.5
    lo = bisect.bisect_left(sorted_vals, x)
    hi = bisect.bisect_right(sorted_vals, x)
    mid = (lo + hi - 1) / 2.0
    return max(0.0, min(1.0, mid / float(n - 1)))


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"Expected JSON array in {path}, got {type(obj)}")
    return obj  # type: ignore[return-value]


def keyword_score(text: str, keywords: list[str]) -> float:
    if not text:
        return 0.0
    t = text.lower()
    hits = 0
    for kw in keywords:
        if kw in t:
            hits += 1
    return clamp01(hits / max(1.0, float(min(5, len(keywords)))))


# Keep this list aligned with persona_model/classify_personas.py
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


def compute_rule_persona_scores(
    *,
    scores: dict[str, Any],
    price_m2_percentile: Optional[float],
    area_m2: Optional[float],
    rooms: Optional[float],
    parking_count: Optional[float],
    text: str,
) -> dict[str, float]:
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
        budget = clamp01(1.0 - float(price_m2_percentile))

    def lin_scale(x: Optional[float], x0: float, x1: float) -> float:
        if x is None:
            return 0.0
        if x1 == x0:
            return 0.0
        return clamp01((float(x) - x0) / (x1 - x0))

    area_large = lin_scale(area_m2, 70.0, 140.0)
    area_small = 1.0 - lin_scale(area_m2, 35.0, 90.0)
    rooms_large = lin_scale(rooms, 2.0, 5.0)
    rooms_small = 1.0 - lin_scale(rooms, 1.0, 3.0)

    parking = 1.0 if (parking_count is not None and parking_count > 0) else 0.0

    kw = {p: keyword_score(text, PERSONA_KEYWORDS.get(p, [])) for p in PERSONA_KEYWORDS.keys()}

    out: dict[str, float] = {}
    out["student"] = clamp01(
        0.30 * budget + 0.25 * transit + 0.15 * nightlife + 0.15 * walk + 0.10 * area_small + 0.05 * kw.get("student", 0.0)
    )
    out["young_professional"] = clamp01(
        0.25 * transit + 0.20 * amenities + 0.20 * walk + 0.20 * safety + 0.10 * nightlife + 0.05 * budget
    )
    out["family_with_kids"] = clamp01(
        0.60 * family + 0.15 * area_large + 0.15 * rooms_large + 0.10 * kw.get("family_with_kids", 0.0)
    )
    out["senior"] = clamp01(0.30 * health + 0.25 * walk + 0.20 * quiet + 0.15 * safety + 0.10 * kw.get("senior", 0.0))
    out["public_transit_commuter"] = clamp01(0.70 * transit + 0.25 * walk + 0.05 * budget)
    out["car_commuter"] = clamp01(0.45 * (parking + 0.4 * kw.get("car_commuter", 0.0)) + 0.35 * (1.0 - quiet) + 0.20 * area_large)
    out["nightlife_lover"] = clamp01(0.55 * nightlife + 0.25 * transit + 0.15 * amenities + 0.05 * (1.0 - quiet))
    out["nature_lover"] = clamp01(0.60 * green_q + 0.20 * quiet + 0.20 * walk)
    out["quiet_seeker"] = clamp01((0.65 * quiet + 0.20 * green_q + 0.15 * safety) * (1.0 - 0.35 * nightlife))
    out["budget_sensitive"] = clamp01(0.70 * budget + 0.15 * transit + 0.15 * amenities)
    expensive = 1.0 - budget
    out["luxury"] = clamp01(0.35 * expensive + 0.25 * area_large + 0.20 * kw.get("luxury", 0.0) + 0.10 * quiet + 0.10 * safety)
    out["remote_worker"] = clamp01(
        0.25 * quiet + 0.20 * area_large + 0.15 * walk + 0.10 * green_q + 0.30 * kw.get("remote_worker", 0.0)
    )

    # Ensure all personas present
    for p in PERSONAS:
        out.setdefault(p, 0.0)
    return out


def extract_price_m2(data: dict[str, Any]) -> Optional[float]:
    v = data.get("preis_m2_num")
    fv = as_float(v)
    if fv is not None and fv > 0:
        return fv
    kp = as_float(data.get("kaufpreis_num"))
    area = as_float(data.get("wohnflache_ca_num"))
    if kp is not None and area is not None and kp > 0 and area > 0:
        return kp / area
    return None


def build_dataset(
    *,
    run_dir: Path,
    field: str,
    label_threshold: float,
    label_source: str,
    manual_labels: Optional[dict[str, list[int]]],
    train_frac: float,
    val_frac: float,
    split_seed: str,
    progress: bool,
) -> dict[str, Any]:
    json_files = sorted(p for p in run_dir.glob("*.json") if p.name != "failed.json")
    if not json_files:
        raise RuntimeError(f"No JSON batch files found in: {run_dir}")

    # First pass: build price distributions per city (for persona scoring)
    city_prices: dict[str, list[float]] = {}
    total_seen = 0
    total_with_price = 0
    rec_total = 0
    prog1 = Progress(label="pass1(prices)", total=None, enabled=progress)
    for p in json_files:
        prog1.set_extra(p.name)
        arr = load_json_array(p)
        rec_total += len(arr)
        for rec in arr:
            if not isinstance(rec, dict):
                prog1.tick()
                continue
            data = rec.get("data")
            if not isinstance(data, dict):
                prog1.tick()
                continue

            city = get_nested(data, [field, "scores", "score_city_norm"])
            if not isinstance(city, str) or not city.strip():
                # fallback: geocode meta city
                city = get_nested(data, [field, "geocode", "meta", "city"])
            if not isinstance(city, str) or not city.strip():
                prog1.tick()
                continue

            price_m2 = extract_price_m2(data)
            total_seen += 1
            if price_m2 is None:
                prog1.tick()
                continue
            total_with_price += 1
            city_prices.setdefault(city.strip().lower(), []).append(float(price_m2))
            prog1.tick()
    prog1.finish()

    city_price_sorted = {k: sorted(v) for k, v in city_prices.items() if v}
    log(f"Price distributions: cities={len(city_price_sorted)} records_with_price={total_with_price}/{max(1,total_seen)}")

    # Second pass: extract features + labels
    feat_rows: list[dict[str, float]] = []
    y_rows: list[list[int]] = []
    weights: list[float] = []
    splits: list[str] = []
    ids: list[str] = []
    urls: list[str] = []

    prog2 = Progress(label="pass2(features)", total=rec_total, enabled=progress)
    missing_overpass = 0
    for p in json_files:
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
            over = data.get(field)
            if not isinstance(over, dict):
                missing_overpass += 1
                prog2.tick()
                continue
            scores = over.get("scores")
            if not isinstance(scores, dict):
                missing_overpass += 1
                prog2.tick()
                continue

            expose_id = data.get("expose_id") or rec.get("expose_id")
            if expose_id is None:
                prog2.tick()
                continue
            expose_id_s = str(expose_id)

            url = data.get("url") or rec.get("url")
            url_s = str(url) if url is not None else ""

            city = scores.get("score_city_norm")
            if not isinstance(city, str) or not city.strip():
                city = get_nested(over, ["geocode", "meta", "city"])
            ck = str(city).strip().lower() if isinstance(city, str) and city.strip() else None

            price_m2 = extract_price_m2(data)
            price_pct = None
            if price_m2 is not None and ck and ck in city_price_sorted:
                price_pct = percentile(city_price_sorted[ck], float(price_m2))

            area_m2 = as_float(data.get("wohnflache_ca_num")) or as_float(data.get("wohnflache_ca"))
            rooms = as_float(data.get("zimmer_num")) or as_float(data.get("zimmer"))
            parking_count = as_float(data.get("garage_stellplatz_num")) or as_float(data.get("garage_stellplatz"))
            text = " ".join(str(x or "") for x in (data.get("title"), data.get("description"), data.get("location_text")))

            persona_scores = compute_rule_persona_scores(
                scores=scores,
                price_m2_percentile=price_pct,
                area_m2=area_m2,
                rooms=rooms,
                parking_count=parking_count,
                text=text,
            )

            y_manual = manual_labels.get(expose_id_s) if manual_labels is not None else None
            if y_manual is not None:
                y = list(y_manual)
            else:
                if label_source == "manual":
                    prog2.tick()
                    continue
                y = [1 if persona_scores.get(pn, 0.0) >= label_threshold else 0 for pn in PERSONAS]

            # Feature vector (sparse dict)
            feats: dict[str, float] = {}

            # Listing numerics: include *_num fields
            for k, v in data.items():
                if not isinstance(k, str):
                    continue
                if not k.endswith("_num"):
                    continue
                fv = as_float(v)
                if fv is None:
                    continue
                feats[f"listing.{k}"] = float(fv)

            # Overpass scores + metrics
            for k, v in scores.items():
                if not isinstance(k, str):
                    continue
                fv = as_float(v)
                if fv is None:
                    continue
                feats[f"osm.score.{k}"] = float(fv)

            metrics = over.get("metrics")
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if not isinstance(k, str):
                        continue
                    fv = as_float(v)
                    if fv is None:
                        continue
                    feats[f"osm.metric.{k}"] = float(fv)

            # Optional enrichments
            aq = over.get("air_quality")
            if isinstance(aq, dict):
                dist = get_nested(aq, ["uba_station", "dist_m"])
                fv = as_float(dist)
                if fv is not None:
                    feats["air.uba_station_dist_m"] = float(fv)

            gtfs = over.get("gtfs_transit")
            if isinstance(gtfs, dict):
                for k in ("nearest_stop_m", "stops_within_radius", "departures_within_radius_total", "departures_within_radius_peak_am", "departures_within_radius_peak_pm"):
                    fv = as_float(gtfs.get(k))
                    if fv is not None:
                        feats[f"gtfs.{k}"] = float(fv)

            nm = over.get("noise_map")
            if isinstance(nm, dict):
                for k in ("lden_db", "lnight_db"):
                    fv = as_float(nm.get(k))
                    if fv is not None:
                        feats[f"noise.{k}"] = float(fv)

            # Keyword features (explicit)
            for p_name, kws in PERSONA_KEYWORDS.items():
                feats[f"kw.{p_name}"] = float(keyword_score(text, kws))

            # Price percentile feature
            if price_pct is not None:
                feats["listing.price_m2_pct_in_city"] = float(price_pct)

            # Geocode confidence as both feature and sample weight
            geo_conf = as_float(scores.get("geocode_confidence_0_1")) or as_float(get_nested(over, ["geocode", "confidence_0_1"]))
            if geo_conf is None:
                geo_conf = 0.5
            geo_conf = clamp01(float(geo_conf))
            feats["geo.confidence_0_1"] = geo_conf
            sample_w = max(0.25, geo_conf)

            split = train_test_split_mask(expose_id_s, train_frac=train_frac, val_frac=val_frac, seed=split_seed)

            feat_rows.append(feats)
            y_rows.append(y)
            weights.append(sample_w)
            splits.append(split)
            ids.append(expose_id_s)
            urls.append(url_s)

            prog2.tick()

    prog2.finish()
    log(f"Records: total={len(feat_rows)} missing_overpass={missing_overpass}")

    return {
        "feature_rows": feat_rows,
        "y": np.asarray(y_rows, dtype=np.int8),
        "weights": np.asarray(weights, dtype=np.float64),
        "splits": np.asarray(splits),
        "expose_ids": np.asarray(ids),
        "urls": np.asarray(urls),
        "personas": PERSONAS,
    }


def build_matrix(feature_rows: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    feat_names = sorted({k for row in feature_rows for k in row.keys()})
    name_to_i = {n: i for i, n in enumerate(feat_names)}
    n = len(feature_rows)
    d = len(feat_names)
    X = np.zeros((n, d), dtype=np.float64)
    for i, row in enumerate(feature_rows):
        for k, v in row.items():
            j = name_to_i.get(k)
            if j is None:
                continue
            X[i, j] = float(v)
    return X, feat_names


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Xs = (X - mean) / std
    return Xs, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_logreg_binary(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    l2: float,
    max_iter: int,
) -> np.ndarray:
    """
    Weighted L2-regularized logistic regression.
    Returns weight vector including bias as last element.
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype=np.float64)])

    # Handle degenerate labels
    y_sum = int(y.sum())
    if y_sum == 0:
        # Always negative => large negative bias
        wb = np.zeros(d + 1, dtype=np.float64)
        wb[-1] = -20.0
        return wb
    if y_sum == n:
        wb = np.zeros(d + 1, dtype=np.float64)
        wb[-1] = 20.0
        return wb

    def loss_and_grad(wb: np.ndarray) -> tuple[float, np.ndarray]:
        z = Xb @ wb
        p = sigmoid(z)
        # log-loss: -[y log p + (1-y) log(1-p)]
        eps = 1e-12
        ll = -(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        L = float(np.sum(w * ll))
        # L2 on weights excluding bias
        w_no_bias = wb[:-1]
        L += 0.5 * float(l2) * float(np.dot(w_no_bias, w_no_bias))

        grad = Xb.T @ (w * (p - y))
        grad[:-1] += float(l2) * w_no_bias
        return L, grad

    wb0 = np.zeros(d + 1, dtype=np.float64)
    res = optimize.minimize(
        fun=lambda wb: loss_and_grad(wb)[0],
        x0=wb0,
        jac=lambda wb: loss_and_grad(wb)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "disp": False},
    )
    wb = np.asarray(res.x, dtype=np.float64)
    return wb


def predict_proba_binary(X: np.ndarray, wb: np.ndarray) -> np.ndarray:
    z = X @ wb[:-1] + wb[-1]
    return sigmoid(z)


def prf1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.10, 0.90, 17):
        pred = (prob >= t).astype(np.int8)
        _, _, f1 = prf1(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def eval_multilabel(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    # micro
    tp = int(np.sum((Y_true == 1) & (Y_pred == 1)))
    fp = int(np.sum((Y_true == 0) & (Y_pred == 1)))
    fn = int(np.sum((Y_true == 1) & (Y_pred == 0)))
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # macro
    f1s: list[float] = []
    for j in range(Y_true.shape[1]):
        _, _, f1 = prf1(Y_true[:, j], Y_pred[:, j])
        f1s.append(float(f1))
    macro_f1 = float(sum(f1s) / max(1, len(f1s)))
    return {"micro_f1": float(micro_f1), "macro_f1": float(macro_f1)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a persona classifier (multi-label logistic regression).")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="db_creation run dir (default: newest run_* under get_data/past/immoscout24/db_creation/data/json/)",
    )
    ap.add_argument(
        "--field",
        default="overpass_enrichment",
        help="Field under record['data'] to read enrichment from (default: overpass_enrichment)",
    )
    ap.add_argument("--label-threshold", type=float, default=0.55, help="Weak-label threshold for rules (default: 0.55)")
    ap.add_argument(
        "--label-source",
        choices=["weak", "manual", "hybrid"],
        default="weak",
        help="Which labels to train on: weak (rules), manual (labels file), hybrid (manual if present else weak).",
    )
    ap.add_argument(
        "--labels-jsonl",
        default=None,
        help="Manual labels JSONL created by persona_model/label_personas.py (optional).",
    )
    ap.add_argument("--train-frac", type=float, default=0.80, help="Train fraction (default: 0.80)")
    ap.add_argument("--val-frac", type=float, default=0.10, help="Validation fraction (default: 0.10)")
    ap.add_argument("--split-seed", default="v1", help="Deterministic split seed (default: v1)")
    ap.add_argument("--l2", type=float, default=1.0, help="L2 regularization strength (default: 1.0)")
    ap.add_argument("--max-iter", type=int, default=200, help="Max optimizer iterations per persona (default: 200)")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a progress bar (default: enabled when stdout is a TTY).",
    )
    ap.add_argument(
        "--model-out",
        default=None,
        help="Output model JSON (default: persona_model/artifacts/persona_logreg_<run>.json)",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite model file if it exists")
    args = ap.parse_args()

    repo = repo_root()
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(repo)
    if not run_dir:
        raise SystemExit("No run dir found. Pass --run-dir .../run_YYYYMMDD_HHMMSS")
    if not run_dir.is_absolute():
        run_dir = (repo / run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    field = str(args.field)
    run_name = run_dir.name
    out_path = Path(args.model_out) if args.model_out else repo / "persona_model" / "artifacts" / f"persona_logreg_{run_name}.json"
    if not out_path.is_absolute():
        out_path = (repo / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        log(f"Model exists (use --force to overwrite): {out_path}")
        return 0

    progress_enabled = args.progress if args.progress is not None else (is_tty() and os.environ.get("NO_PROGRESS") != "1")

    log(f"Repo root: {repo}")
    log(f"Run dir: {run_dir}")
    log(f"Field: data.{field}")
    log(f"Label source: {args.label_source}")
    log(f"Weak-label threshold: {args.label_threshold}")
    log(f"Split: train={args.train_frac:.2f} val={args.val_frac:.2f} test={1.0 - args.train_frac - args.val_frac:.2f} seed={args.split_seed}")

    manual_labels: Optional[dict[str, list[int]]] = None
    labels_jsonl_path: Optional[Path] = None
    if args.labels_jsonl:
        labels_path = Path(args.labels_jsonl)
        if not labels_path.is_absolute():
            labels_path = (repo / labels_path).resolve()
        if not labels_path.exists():
            raise SystemExit(f"Labels JSONL not found: {labels_path}")
        labels_jsonl_path = labels_path
        manual_labels = {}
        with labels_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                eid = obj.get("expose_id")
                if eid is None:
                    continue
                labels = obj.get("labels")
                if isinstance(labels, dict):
                    y = [int(labels.get(pn, 0) or 0) for pn in PERSONAS]
                elif isinstance(obj.get("positive_personas"), list):
                    pos = {str(p) for p in obj["positive_personas"]}
                    y = [1 if pn in pos else 0 for pn in PERSONAS]
                else:
                    continue
                manual_labels[str(eid)] = y
        log(f"Manual labels loaded: {len(manual_labels)}")

    if args.label_source in ("manual", "hybrid") and not manual_labels:
        raise SystemExit("--label-source manual|hybrid requires --labels-jsonl with at least 1 label")

    ds = build_dataset(
        run_dir=run_dir,
        field=field,
        label_threshold=float(args.label_threshold),
        label_source=str(args.label_source),
        manual_labels=manual_labels,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        split_seed=str(args.split_seed),
        progress=progress_enabled,
    )

    X, feat_names = build_matrix(ds["feature_rows"])
    Y = ds["y"].astype(np.float64)
    w = ds["weights"].astype(np.float64)
    splits = ds["splits"]
    personas = ds["personas"]

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]
    Y_train = Y[train_mask]
    Y_val = Y[val_mask]
    Y_test = Y[test_mask]
    w_train = w[train_mask]
    w_val = w[val_mask]
    w_test = w[test_mask]

    log(f"Shapes: X={X.shape} train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]} features={X.shape[1]}")

    if X_train.shape[0] < 10:
        raise SystemExit(
            f"Train split too small (n_train={X_train.shape[0]}). "
            "Label more samples or adjust --train-frac/--val-frac/--split-seed."
        )

    X_train_s, mean, std = standardize_fit(X_train)
    X_val_s = standardize_apply(X_val, mean, std)
    X_test_s = standardize_apply(X_test, mean, std)

    l2 = float(args.l2)
    max_iter = int(args.max_iter)

    model_personas: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    val_f1s: dict[str, float] = {}

    t0 = time.time()
    for j, pn in enumerate(personas):
        ytr = Y_train[:, j]
        yva = Y_val[:, j]
        wb = fit_logreg_binary(X_train_s, ytr, w_train, l2=l2, max_iter=max_iter)
        if X_val_s.shape[0] > 0:
            pva = predict_proba_binary(X_val_s, wb)
            thr, f1 = best_threshold(yva, pva)
        else:
            thr, f1 = 0.5, 0.0
        thresholds[pn] = float(thr)
        val_f1s[pn] = float(f1)
        model_personas[pn] = {"weights": wb.tolist(), "threshold": float(thr)}
        log(f"[{j+1}/{len(personas)}] {pn}: val_f1={f1:.3f} thr={thr:.2f} pos_train={int(ytr.sum())} pos_val={int(yva.sum())}")

    # Evaluate on test
    if X_test_s.shape[0] > 0:
        prob_test = np.zeros_like(Y_test, dtype=np.float64)
        pred_test = np.zeros_like(Y_test, dtype=np.int8)
        for j, pn in enumerate(personas):
            wb = np.asarray(model_personas[pn]["weights"], dtype=np.float64)
            prob_test[:, j] = predict_proba_binary(X_test_s, wb)
            pred_test[:, j] = (prob_test[:, j] >= float(model_personas[pn]["threshold"])).astype(np.int8)

        metrics = eval_multilabel(Y_test.astype(np.int8), pred_test)
        elapsed = time.time() - t0
        log(
            f"Test metrics (label_source={args.label_source}): micro_f1={metrics['micro_f1']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} elapsed={human_duration(elapsed)}"
        )
    else:
        metrics = {"micro_f1": 0.0, "macro_f1": 0.0}
        elapsed = time.time() - t0
        log(f"Test split empty; skipping test metrics. elapsed={human_duration(elapsed)}")

    if args.label_source == "weak":
        notes = [
            "This model is trained/evaluated against weak labels derived from the rule-based persona scores.",
            "To reduce real-world error, add a manually labeled file and retrain.",
        ]
    elif args.label_source == "manual":
        notes = [
            "This model is trained/evaluated against manual labels from --labels-jsonl.",
            "To reduce error further, label more samples (especially hard/ambiguous listings).",
        ]
    else:
        notes = [
            "This model is trained/evaluated against hybrid labels: manual when present, else weak labels from rules.",
            "For best quality metrics, use --label-source manual on a sufficiently sized labeled set.",
        ]

    out_obj = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "field": field,
        "label_source": str(args.label_source),
        "label_threshold": float(args.label_threshold),
        "labels_jsonl": str(labels_jsonl_path) if labels_jsonl_path is not None else None,
        "split": {"train_frac": float(args.train_frac), "val_frac": float(args.val_frac), "seed": str(args.split_seed)},
        "l2": l2,
        "max_iter": max_iter,
        "features": feat_names,
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "personas": personas,
        "models": model_personas,
        "val_f1": val_f1s,
        "test_metrics": metrics,
        "notes": notes,
    }
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out_path)
    log(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
