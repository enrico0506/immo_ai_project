import gzip
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

import ijson


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(path, "w") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


@contextmanager
def open_jsonl_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = _open_text(path, "w")
    try:
        yield handle
    finally:
        handle.close()


def write_jsonl_line(handle: TextIO, row: dict[str, Any]) -> None:
    json.dump(row, handle, ensure_ascii=False)
    handle.write("\n")


def iter_items_from_json(path: Path, items_key: str = "items") -> Iterator[dict[str, Any]]:
    with open(path, "rb") as handle:
        for item in ijson.items(handle, f"{items_key}.item"):
            if isinstance(item, dict):
                yield item


def iter_json_items(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffixes[-2:] == [".jsonl", ".gz"] or path.suffix == ".jsonl":
        yield from read_jsonl(path)
        return
    if path.suffix == ".json":
        yield from iter_items_from_json(path)
        return
    raise ValueError(f"Unsupported input type: {path}")
