# Geodata

This folder contains all geo-related components and artifacts for this repo, including:
- project modules (e.g. `air_quality/`, `transit_accessibility/`, …)
- raw OSM extracts under `osm/`
- a repo-local Overpass database under `overpass/` (built/imported locally)
- project-specific Overpass + density tooling under `amensity/` (folder name intentionally kept as-is)

## Folder layout

- `geodata/amensity/`
  - Repo-local Overpass build + setup scripts and density analysis scripts.
  - Start here: `geodata/amensity/README.md`
- `geodata/osm/`
  - Raw downloads (e.g. `germany-latest.osm.pbf`) and converted extracts used for import.
- `geodata/overpass/`
  - `db/` Overpass database files
  - `logs/` import/build/dispatcher/http logs
  - `diffs/` optional replication diffs (currently unused)

Other directories (`air_quality/`, `green_proximity/`, …) are placeholders for additional geodata pipelines.

## Overpass (local)

This repo runs Overpass fully locally (no root, no system-wide install required).

Typical workflow (from repo root):

```bash
# 1) build Overpass in-repo
./geodata/amensity/overpass_build.sh

# 2) import an extract into ./geodata/overpass/db (long step)
./geodata/amensity/overpass_setup.sh --extract-url https://download.geofabrik.de/europe/germany-latest.osm.pbf --force-import

# 3) start local dispatcher + HTTP endpoint
./geodata/amensity/overpass_start.sh

# 4) verify
./geodata/amensity/overpass_status.sh

# 5) stop
./geodata/amensity/overpass_stop.sh
```

Local endpoint:
- `POST http://127.0.0.1:8080/api/interpreter` (form field `data` with Overpass QL)

## Density analysis

After importing Germany (or another extract) and starting Overpass, you can run:

- Street + grid density:
  - `./geodata/amensity/street_density.py --radius-km 5 --cell-size-m 300 --force`
- Micro grid sampling (quick, coarse):
  - `./geodata/amensity/micro_density.py --force`

Outputs are written under `geodata/amensity/analysis/` (ignored by git).

## Git / large files

The following are local artifacts and are ignored via `.gitignore`:
- `geodata/osm/`
- `geodata/overpass/db/`
- `geodata/overpass/logs/`
- `geodata/overpass/diffs/`
- `geodata/amensity/vendor/overpass/`
- `geodata/amensity/analysis/`
