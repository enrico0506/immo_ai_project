# Immo AI Project

Refactored into a maintainable, testable Python package with a CLI for ingestion and parsing.

## Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Usage

Parse immobilie1 expose bodies:

```bash
immo-ai parse immobilie1 --input data/raw/immobilie1/<run_id>/expose_bodies.jsonl.gz --output data/processed/immobilie1/<run_id>/exposes.jsonl.gz
```

Collect links (polite crawling, stop on blocks):

```bash
immo-ai ingest immobilie1 collect-links --max-pages 5 --throttle-seconds 1.2 --respect-robots
```

Fetch bodies from links:

```bash
immo-ai ingest immobilie1 fetch-bodies --input data/raw/immobilie1/<run_id>/expose_links.json --max-rpm 30
```

## Data layout

Outputs follow:

```
data/raw/<source>/<run_id>/...
data/processed/<source>/<run_id>/...
```

Each run writes `manifest.json`, `metrics.json`, and optional `rejects.jsonl.gz`.
