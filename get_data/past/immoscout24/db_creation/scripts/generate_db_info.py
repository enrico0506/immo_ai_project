#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def json_loads(raw: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def json_dump_pretty(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


CHUNK_RE = re.compile(r"^(?P<prefix>.+?)(?P<idx>\d{4})\.jsonl$", re.IGNORECASE)
CHUNK_FILE_RE = re.compile(r"^(?P<prefix>.+?)(?P<idx>\d{4})\.(?:jsonl|json)$", re.IGNORECASE)


def infer_group_name(files: list[Path], *, input_dir: Path) -> tuple[str, list[str]]:
    """
    Best-effort run/group name from either:
    - short-run JSON export layout: <input_dir>/<group>/<0001.json>
    - chunked filenames: immoscout_structured_run_..._0001.jsonl/.json -> run_...
    """
    groups: set[str] = set()
    for p in files:
        try:
            rel = p.relative_to(input_dir)
        except Exception:
            continue
        if len(rel.parts) >= 2:
            groups.add(rel.parts[0])

    if groups:
        ordered = sorted(groups)
        if len(ordered) == 1:
            return ordered[0], ordered
        return "mixed", ordered

    prefixes: list[str] = []
    for p in files:
        m = CHUNK_FILE_RE.match(p.name)
        if not m:
            continue
        prefix = (m.group("prefix") or "").rstrip("_")
        if prefix:
            prefixes.append(prefix)
    if not prefixes:
        return "unknown"

    most_common = Counter(prefixes).most_common(1)[0][0]
    # Strip the constant prefix so the group stays short.
    if most_common.startswith("immoscout_structured_"):
        most_common = most_common[len("immoscout_structured_") :]
    return most_common or "unknown", []


@dataclass
class FieldStats:
    count: int = 0
    types: Counter[str] = field(default_factory=Counter)
    max_str_len: int = 0
    max_list_len: int = 0
    min_num: Optional[float] = None
    max_num: Optional[float] = None

    def update(self, value: Any) -> None:
        self.count += 1
        tname = type(value).__name__
        self.types[tname] += 1

        if isinstance(value, str):
            self.max_str_len = max(self.max_str_len, len(value))
            return

        if isinstance(value, list):
            self.max_list_len = max(self.max_list_len, len(value))
            return

        # bool is a subclass of int; treat it separately.
        if isinstance(value, bool):
            return

        if isinstance(value, int | float):
            v = float(value)
            self.min_num = v if self.min_num is None else min(self.min_num, v)
            self.max_num = v if self.max_num is None else max(self.max_num, v)

    def to_dict(self, total_records: int) -> dict[str, Any]:
        suggested_type = suggest_db_type(self.types)
        out: dict[str, Any] = {
            "count": self.count,
            "presence_pct": (self.count / total_records) if total_records else 0.0,
            "types": dict(self.types),
            "suggested_db_type": suggested_type,
        }
        if self.max_str_len:
            out["max_str_len"] = self.max_str_len
        if self.max_list_len:
            out["max_list_len"] = self.max_list_len
        if self.min_num is not None and self.max_num is not None:
            out["min_num"] = self.min_num
            out["max_num"] = self.max_num
        return out


def suggest_db_type(type_counts: Counter[str]) -> str:
    """
    Very small heuristic to help with schema design (generic SQL-ish types).
    """
    if not type_counts:
        return "unknown"
    types = set(type_counts)

    # JSON-ish containers
    if "dict" in types or "list" in types:
        return "json"

    # bool is distinct from int for storage purposes
    if types.issubset({"bool", "NoneType"}):
        return "boolean"

    if types.issubset({"int", "float", "NoneType"}):
        return "numeric"

    if "str" in types:
        return "text"

    # Fallback
    return "unknown"


def parse_args() -> argparse.Namespace:
    scripts_dir = Path(__file__).resolve().parent
    immoscout_dir = scripts_dir.parents[1]  # .../immoscout24
    db_creation_dir = scripts_dir.parent  # .../immoscout24/db_creation

    default_input_dir = db_creation_dir / "data" / "json"

    p = argparse.ArgumentParser(description="Generate DB field/key info from structured JSON exports.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Directory with exported JSON/JSONL files (default: {default_input_dir}).",
    )
    p.add_argument(
        "--pattern",
        default="**/*.json",
        help=(
            "Glob pattern within input-dir for input files. "
            "Default scans exported JSON arrays recursively."
        ),
    )
    p.add_argument(
        "--input-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
        help="Force input format; default auto-detects by file extension.",
    )
    p.add_argument(
        "--out",
        type=Path,
        help="Output JSON path (default: <db_creation>/data/db_info/fields.json).",
    )
    p.add_argument(
        "--limit",
        type=int,
        help="Scan at most N records total (faster, but may miss rare fields).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    pattern: str = args.pattern
    input_format: str = (args.input_format or "auto").strip().lower()

    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched: {input_dir}/{pattern}")

    group, groups = infer_group_name(files, input_dir=input_dir)

    out_path = args.out
    if out_path is None:
        scripts_dir = Path(__file__).resolve().parent
        db_creation_dir = scripts_dir.parent
        out_path = db_creation_dir / "data" / "db_info" / "fields.json"

    t0 = time.time()
    limit = args.limit

    decode_errors = 0
    object_records = 0
    records_with_data = 0
    skipped_non_dict_items = 0

    json_files = 0
    jsonl_files = 0

    top_stats: dict[str, FieldStats] = {}
    data_stats: dict[str, FieldStats] = {}

    for p in files:
        if limit is not None and object_records >= limit:
            break

        is_jsonl = (p.suffix or "").lower() == ".jsonl"
        is_json = (p.suffix or "").lower() == ".json"
        if input_format == "jsonl":
            is_jsonl = True
        elif input_format == "json":
            is_jsonl = False

        if is_jsonl:
            jsonl_files += 1
            with p.open("rb") as f:
                for raw in f:
                    if limit is not None and object_records >= limit:
                        break
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json_loads(raw)
                    except Exception:
                        decode_errors += 1
                        continue
                    if not isinstance(obj, dict):
                        skipped_non_dict_items += 1
                        continue
                    _consume_record(obj, top_stats=top_stats, data_stats=data_stats)
                    object_records += 1
                    if isinstance(obj.get("data"), dict):
                        records_with_data += 1
        else:
            if not is_json and input_format == "auto":
                continue
            json_files += 1
            try:
                root = json_loads(p.read_bytes())
            except Exception:
                decode_errors += 1
                continue

            if isinstance(root, dict):
                _consume_record(root, top_stats=top_stats, data_stats=data_stats)
                object_records += 1
                if isinstance(root.get("data"), dict):
                    records_with_data += 1
            elif isinstance(root, list):
                for obj in root:
                    if limit is not None and object_records >= limit:
                        break
                    if not isinstance(obj, dict):
                        skipped_non_dict_items += 1
                        continue
                    _consume_record(obj, top_stats=top_stats, data_stats=data_stats)
                    object_records += 1
                    if isinstance(obj.get("data"), dict):
                        records_with_data += 1
            else:
                skipped_non_dict_items += 1

    if input_format != "auto":
        detected_input_format = input_format
    elif jsonl_files and not json_files:
        detected_input_format = "jsonl"
    elif json_files and not jsonl_files:
        detected_input_format = "json"
    else:
        detected_input_format = "mixed"

    dt = time.time() - t0

    out_obj = {
        "generated_at": utc_now(),
        "group": group,
        "groups": groups,
        "source": {
            "input_dir": str(input_dir),
            "pattern": pattern,
            "input_format": detected_input_format,
            "files": [str(p.relative_to(input_dir)) for p in files],
            "file_count": len(files),
            "json_file_count": json_files,
            "jsonl_file_count": jsonl_files,
        },
        "records": {
            "json_decode_errors": decode_errors,
            "json_object_records": object_records,
            "records_with_data_object": records_with_data,
            "skipped_non_dict_items": skipped_non_dict_items,
            "limit": limit,
        },
        "fields": {
            "top": {k: top_stats[k].to_dict(object_records) for k in sorted(top_stats)},
            "data": {k: data_stats[k].to_dict(records_with_data) for k in sorted(data_stats)},
        },
        "field_lists": {
            "top": sorted(top_stats),
            "data": sorted(data_stats),
            "all_paths": [*(f"top.{k}" for k in sorted(top_stats)), *(f"data.{k}" for k in sorted(data_stats))],
        },
        "timing": {"seconds": dt},
    }

    json_dump_pretty(out_obj, out_path)
    rel = out_path
    try:
        rel = out_path.relative_to(Path.cwd())
    except Exception:
        pass
    print(f"Wrote {rel} | top_fields={len(top_stats)} data_fields={len(data_stats)} records={object_records}")


def _consume_record(
    obj: dict[str, Any],
    *,
    top_stats: dict[str, FieldStats],
    data_stats: dict[str, FieldStats],
) -> None:
    for k, v in obj.items():
        st = top_stats.get(k)
        if st is None:
            st = FieldStats()
            top_stats[k] = st
        st.update(v)

    data = obj.get("data")
    if isinstance(data, dict):
        for k, v in data.items():
            st = data_stats.get(k)
            if st is None:
                st = FieldStats()
                data_stats[k] = st
            st.update(v)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        os._exit(0)
