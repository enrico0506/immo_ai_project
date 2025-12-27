ImmoScout24 past data collection lives here. Run the scripts manually in this order:

1) `python scripts/collect_expose_links.py` – capture search filters in the opened browser, then gather expose links (and optionally bodies) into `data/raw/immoscout_exposes_links.json` and `data/raw/immoscout_exposes.json`.
2) `python scripts/fetch_expose_bodies.py` – fetch HTML bodies for the collected links into `data/raw/expose_bodies.json` if they were not fetched in step 1.
3) `python scripts/analyze_expose_bodies.py` – parse the HTML bodies into structured fields saved at `data/processed/immoscout_exposes_structured.json`.
4) `python scripts/score_exposes_living_standard.py` – geocode via local Overpass DB + compute neighborhood metrics + overall score (writes JSONL to `data/processed/`).

Folder layout:
- `scripts/` holds the Playwright scraping/parsing scripts.
- `data/raw/` stores links, raw exposes, and raw HTML bodies.
- `data/processed/` stores structured output derived from the raw data.

Optional (save disk space):
- Compress raw `*.jsonl` to `*.jsonl.zst`: `get_data/past/immoscout24/scripts/compress_raw_jsonl.sh`
- Decompress one file: `zstd -d path/to/file.jsonl.zst -o path/to/file.jsonl`

Scoring example (from repo root):

```bash
python get_data/past/immoscout24/scripts/score_exposes_living_standard.py \\
  --input get_data/past/immoscout24/data/processed/immoscout_structured_run_20251225_205537_*.jsonl \\
  --limit 2000
```

Slim + CSV output:

```bash
python get_data/past/immoscout24/scripts/score_exposes_living_standard.py \\
  --slim \\
  --output-csv get_data/past/immoscout24/data/processed/immoscout_scored.csv
```
