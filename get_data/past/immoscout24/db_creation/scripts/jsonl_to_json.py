#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


def _indent_block(block: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in block.splitlines())


CHUNK_RE = re.compile(r"^(?P<prefix>.+?)(?P<idx>\d{4})\.jsonl$", re.IGNORECASE)
FAILED_RE = re.compile(r"^(?P<prefix>.+?)_failed\.jsonl$", re.IGNORECASE)
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _shorten_stem(stem: str) -> str:
    """
    Make filenames less verbose by stripping known prefixes.
    """
    s = (stem or "").strip()
    for prefix in ("immoscout_structured_", "expose_analysis_", "immoscout_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    s = SAFE_NAME_RE.sub("_", s).strip("._-")
    return s or "output"


def output_path_for_input(
    in_path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    naming: str,
) -> Path:
    """
    naming:
      - original: keep original filename (mirrors input dir layout)
      - short-flat: shorten filename but keep output_dir flat
      - short-run: create run subfolder + short filenames (preferred for chunked runs)
    """
    rel = in_path.relative_to(input_dir)

    if naming == "original":
        return (output_dir / rel).with_suffix(".json")

    name = in_path.name
    chunk_m = CHUNK_RE.match(name)
    if naming == "short-run" and chunk_m:
        group = _shorten_stem((chunk_m.group("prefix") or "").rstrip("_"))
        idx = chunk_m.group("idx")
        return output_dir / group / f"{idx}.json"

    failed_m = FAILED_RE.match(name)
    if naming == "short-run" and failed_m:
        group = _shorten_stem(failed_m.group("prefix") or "")
        return output_dir / group / "failed.json"

    if naming == "short-flat":
        return output_dir / f"{_shorten_stem(in_path.stem)}.json"

    # Fallback for short-run when the file isn't a recognized chunk/special file.
    return output_dir / f"{_shorten_stem(in_path.stem)}.json"


def convert_jsonl_to_json_array(
    in_path: Path,
    out_path: Path,
    *,
    pretty: bool,
    sort_keys: bool,
    overwrite: bool,
    max_records: int | None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return {"input": str(in_path), "output": str(out_path), "skipped": True, "reason": "exists"}

    t0 = time.time()
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    records = 0
    decode_errors = 0
    first = True

    with in_path.open("rb") as fin, tmp_path.open("w", encoding="utf-8", newline="\n") as fout:
        fout.write("[\n")
        for raw in fin:
            if max_records is not None and records >= max_records:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                decode_errors += 1
                continue

            if not first:
                fout.write(",\n")
            first = False

            if pretty:
                dumped = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=sort_keys)
                fout.write(_indent_block(dumped, "  "))
            else:
                dumped = json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys)
                fout.write("  " + dumped)
            records += 1

        fout.write("\n]\n")

    tmp_path.replace(out_path)
    dt = time.time() - t0
    return {
        "input": str(in_path),
        "output": str(out_path),
        "records": records,
        "decode_errors": decode_errors,
        "seconds": dt,
    }


def parse_args() -> argparse.Namespace:
    scripts_dir = Path(__file__).resolve().parent
    immoscout_dir = scripts_dir.parents[1]  # .../immoscout24
    db_creation_dir = scripts_dir.parent  # .../immoscout24/db_creation

    default_input_dir = immoscout_dir / "data" / "processed"
    default_output_dir = db_creation_dir / "data" / "json"

    p = argparse.ArgumentParser(description="Convert JSONL files to JSON arrays (pretty-printed).")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Directory with JSONL files (default: {default_input_dir}).",
    )
    p.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern to select JSONL files within input-dir (default: *.jsonl).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for JSON files (default: {default_output_dir}).",
    )
    p.add_argument(
        "--naming",
        choices=["short-run", "short-flat", "original"],
        default="short-run",
        help=(
            "Output naming strategy. "
            "'short-run' creates a subfolder per run/prefix and uses short filenames like 0001.json. "
            "'short-flat' shortens the filename but keeps output_dir flat. "
            "'original' keeps the original filename."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSON files.",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON (one line per element) instead of pretty printing.",
    )
    p.add_argument(
        "--sort-keys",
        action="store_true",
        help="Sort keys in output JSON objects (stable output, slightly slower).",
    )
    p.add_argument(
        "--max-records",
        type=int,
        help="Convert at most N records per file (for quick inspection).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    pattern: str = args.pattern

    if args.compact:
        pretty = False
    else:
        pretty = True

    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched: {input_dir}/{pattern}")

    total_records = 0
    total_errors = 0
    t0 = time.time()

    for in_path in files:
        out_path = output_path_for_input(
            in_path,
            input_dir=input_dir,
            output_dir=output_dir,
            naming=str(args.naming),
        )
        out_rel = out_path.relative_to(output_dir) if out_path.is_relative_to(output_dir) else out_path
        res = convert_jsonl_to_json_array(
            in_path,
            out_path,
            pretty=pretty,
            sort_keys=bool(args.sort_keys),
            overwrite=bool(args.overwrite),
            max_records=args.max_records,
        )
        if res.get("skipped"):
            print(f"SKIP  {in_path.name} -> {out_rel} ({res.get('reason')})")
            continue
        total_records += int(res.get("records", 0))
        total_errors += int(res.get("decode_errors", 0))
        print(
            f"OK    {in_path.name} -> {out_rel} | "
            f"records={res.get('records')} decode_errors={res.get('decode_errors')} time={res.get('seconds'):.2f}s"
        )

    dt = time.time() - t0
    print(f"\nDone. files={len(files)} records={total_records} decode_errors={total_errors} total_time={dt:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Allow piping to `head` / `rg` without stack traces.
        os._exit(0)
