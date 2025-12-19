import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class RunPaths:
    source: str
    run_id: str
    raw_dir: Path
    processed_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.processed_dir / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.processed_dir / "metrics.json"

    @property
    def rejects_path(self) -> Path:
        return self.processed_dir / "rejects.jsonl.gz"

    @property
    def cache_html_dir(self) -> Path:
        return self.raw_dir / "cache" / "html"


def build_run_paths(base_dir: Path, source: str, run_id: str | None) -> RunPaths:
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = base_dir / "data" / "raw" / source / resolved_run_id
    processed_dir = base_dir / "data" / "processed" / source / resolved_run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(source=source, run_id=resolved_run_id, raw_dir=raw_dir, processed_dir=processed_dir)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
