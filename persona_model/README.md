# Persona classification (prototype)

This folder is for developing a **persona classifier** that maps ImmoScout listings to likely user “personas” (multi-label).

It uses the already-enriched dataset in:

- `get_data/past/immoscout24/db_creation/data/json/run_*/####.json`

and relies on fields like:

- listing attributes (`kaufpreis_num`, `preis_m2_num`, `wohnflache_ca_num`, `zimmer_num`, …)
- OSM/Overpass enrichment (`data.overpass_enrichment.scores.*`)
- optional: `data.overpass_enrichment.air_quality` (UBA) and `gtfs_transit` (GTFS) if you ran those enrichers.

## Quick start

From repo root:

```bash
python persona_model/classify_personas.py \
  --run-dir get_data/past/immoscout24/db_creation/data/json/run_20251225_205537 \
  --output persona_model/output/personas_run_20251225_205537.jsonl
```

This writes a JSONL file with:

- `primary_persona` + `primary_score`
- `top_personas` (top-k ranking)
- full `personas` score map (0..1)

The script is idempotent: it won’t overwrite the output unless you pass `--force`.

## Trainable model (baseline)

You can train a simple **multi-label logistic regression** model.

Important: without human labels, the default is to train on **weak labels** derived from the rule-based persona scores
(so “error” = disagreement with those labels). Later you can swap in a manually labeled file.

### Create manual labels (recommended)

Create a small labeled set first (e.g. 200 listings):

```bash
python persona_model/label_personas.py \
  --run-dir get_data/past/immoscout24/db_creation/data/json/run_20251225_205537 \
  --sample 200
```

This appends labels into `persona_model/data/persona_labels_<run>.jsonl` and is safe to resume.

Train + evaluate:

```bash
python persona_model/train_persona_model.py \
  --run-dir get_data/past/immoscout24/db_creation/data/json/run_20251225_205537 \
  --progress \
  --force
```

This writes:

- model artifact: `persona_model/artifacts/persona_logreg_<run>.json`
- metrics summary printed to stdout

Train using manual labels:

```bash
python persona_model/train_persona_model.py \
  --run-dir get_data/past/immoscout24/db_creation/data/json/run_20251225_205537 \
  --labels-jsonl persona_model/data/persona_labels_run_20251225_205537.jsonl \
  --label-source manual \
  --progress \
  --force
```

## Notes

- This is a **rule-based baseline** (weights + simple text keywords). It’s intended as a starting point.
- Later you can add a supervised model using:
  - user feedback (clicks/saves/requests), or
  - a manually labeled subset, or
  - weak labels generated from these rules.
