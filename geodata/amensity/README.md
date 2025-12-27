# Repo-local Overpass API (no root, no nginx)

This folder provides fully repo-local scripts to download/build Overpass (osm-3s), import an OSM extract into an Overpass DB under `./geodata/overpass/db/`, and expose a local HTTP endpoint compatible with `/api/interpreter`.

## Layout

- Build/install: `./geodata/amensity/vendor/overpass/install/`
- Raw extracts: `./geodata/osm/`
- Overpass DB: `./geodata/overpass/db/`
- Logs: `./geodata/overpass/logs/`
- PIDs: `./geodata/overpass/dispatcher.pid`, `./geodata/overpass/http.pid`

## Quickstart

1) Build Overpass (repo-local):

```bash
./geodata/amensity/overpass_build.sh
```

2) Download + import an extract (default: first entry in `extracts.txt`):

```bash
./geodata/amensity/overpass_setup.sh
```

Or pick a URL explicitly:

```bash
./geodata/amensity/overpass_setup.sh --extract-url https://download.geofabrik.de/europe/germany/berlin-latest.osm.pbf
```

3) Start dispatcher + local HTTP wrapper:

```bash
./geodata/amensity/overpass_start.sh
```

4) Query it:

```bash
curl -sS -X POST \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'data=[out:json][timeout:25];node(around:300,52.5200,13.4050)["amenity"="cafe"];out 5;' \
  http://127.0.0.1:8080/api/interpreter | head
```

5) Status + health check:

```bash
./geodata/amensity/overpass_status.sh
```

6) Stop services:

```bash
./geodata/amensity/overpass_stop.sh
```

## Micro density analysis (big German cities)

1) Import a Germany-wide extract (recommended) and start services:

```bash
./geodata/amensity/overpass_setup.sh --extract-url https://download.geofabrik.de/europe/germany-latest.osm.pbf
./geodata/amensity/overpass_start.sh
```

2) Run micro amenity density for the default big-city list (`cities_de.tsv`):

```bash
./geodata/amensity/micro_density.py
```

Outputs go to `./geodata/amensity/analysis/micro_density/` (summary + per-city raw grids).

## Street-level + 300m x 300m grid density

This produces:
- a full grid of `--cell-size-m` squares (default 300m x 300m) with amenity counts/density
- a per-street summary for named streets (amenities assigned to nearest street within `--street-buffer-m`)

Run (example for Berlin only):

```bash
./geodata/amensity/overpass_start.sh
./geodata/amensity/street_density.py --city Berlin --radius-km 5 --cell-size-m 300
./geodata/amensity/overpass_stop.sh
```

Outputs go to `./geodata/amensity/analysis/street_density/<city>/`.

## Per-cell multi-metric features (amenities, transit, green, nightlife, traffic proxy)

This produces **per 300m x 300m cell** metrics around a city center radius (default 5km):
- amenities / shops / healthcare / tourism counts + densities
- transit stops/platforms/stations counts + density (OSM tags)
- green POIs/areas (park/wood/forest/grass/playground) counts + density (OSM tags)
- nightlife (`amenity=bar/pub/nightclub`) counts
- traffic/noise proxy: major road length + nearest major-road distance + average `maxspeed` where present

Run (all cities, 5km radius):

```bash
./geodata/amensity/cell_multimetric.py --radius-km 5 --cell-size-m 300 --force
```

Outputs go to `./geodata/amensity/analysis/cell_multimetric/<city>/cells.csv`.

## Extract selection

- Edit `./geodata/amensity/extracts.txt` to add your own extracts (format: `Label<TAB>URL`).
- Use interactive selection:

```bash
./geodata/amensity/overpass_setup.sh --interactive
```

## Flags / Idempotence

- Rebuild Overpass:

```bash
./geodata/amensity/overpass_build.sh --force-build
```

- Reimport DB (clears `./geodata/overpass/db/` contents first):

```bash
./geodata/amensity/overpass_setup.sh --force-import
```

- Import with metadata (optional; can fail if extract lacks metadata):

```bash
./geodata/amensity/overpass_setup.sh --meta=yes
```

## Troubleshooting

- Build failures: check `./geodata/overpass/logs/overpass_build.log` and install the missing build deps printed by `overpass_build.sh`.
- Import failures: check `./geodata/overpass/logs/import.log`. Ensure you use a `.osm.bz2` extract (or install `osmium-tool` to convert `.pbf`).
- HTTP returns `503`: dispatcher not running (or stale pidfile). Run `./geodata/amensity/overpass_start.sh` and check `./geodata/overpass/logs/dispatcher.log`.
