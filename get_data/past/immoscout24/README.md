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

Faster scoring (threaded + single Overpass query per expose):

```bash
python get_data/past/immoscout24/scripts/score_exposes_living_standard.py \\
  --slim \\
  --workers 4 \\
  --combined-query
```

Resume after a crash/stop (append to a fixed output file and skip already-written exposes):

```bash
python get_data/past/immoscout24/scripts/score_exposes_living_standard.py \\
  --slim \\
  --workers 8 \\
  --output get_data/past/immoscout24/data/processed/immoscout_scored_latest.jsonl \\
  --resume
```

Extended enrichment (v2: nearest POIs, education/healthcare/walkability/family/noise proxies):

```bash
python get_data/past/immoscout24/scripts/score_exposes_living_standard.py \\
  --slim --no-nominatim \\
  --workers 6 --overpass-concurrency 4 \\
  --output get_data/past/immoscout24/data/processed/immoscout_scored_latest_v2.jsonl
```

Attach scores into db_creation JSON (so everything is in one place):

```bash
# 1) Create a scored JSONL (or reuse an existing one)
python get_data/past/immoscout24/scripts/score_exposes_living_standard.py --slim

# 2) Attach into the latest db_creation run_*/ JSON batch files
python get_data/past/immoscout24/db_creation/scripts/attach_overpass_scores.py
```

Post-enrichment (adds new fields directly into db_creation JSON batches):

```bash
# Add geocode confidence (downweight low-quality coordinates)
python get_data/past/immoscout24/db_creation/scripts/enrich_geocode_confidence.py

# Normalize the overall score within each city (adds score_percentile_in_city + score_z_in_city)
python get_data/past/immoscout24/db_creation/scripts/normalize_scores_by_city.py
```

GTFS transit (real stops + frequency proxy):

```bash
# 1) Put one or more GTFS zip feeds here (kept local / ignored by git):
mkdir -p geodata/amensity/gtfs/feeds

# 2) Enrich all records with GTFS stop distance + stop/departure counts within 500m
python get_data/past/immoscout24/db_creation/scripts/enrich_transit_gtfs.py --radius-m 500
```

Air quality (UBA):

```bash
# Nearest UBA station + distance (fast, cached station list)
python get_data/past/immoscout24/db_creation/scripts/enrich_air_quality_uba.py

# Optional: also fetch NO2/PM10/PM2 measures for the nearest station (can be slow)
python get_data/past/immoscout24/db_creation/scripts/enrich_air_quality_uba.py --fetch-measures
```

Noise maps (optional, requires local GeoTIFFs + `gdallocationinfo` from GDAL):

```bash
mkdir -p geodata/amensity/noise
# place: geodata/amensity/noise/lden.tif and/or geodata/amensity/noise/lnight.tif
python get_data/past/immoscout24/db_creation/scripts/enrich_noise_raster.py
```
