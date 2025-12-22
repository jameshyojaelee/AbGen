# Config Catalog

## Core configs

- `configs/train.yaml` — training loop defaults (`scripts/train.py`, `scripts/train_cv.py`).
- `configs/model.yaml` — model architecture knobs (transformer + Mamba + DPO).
- `configs/data.yaml` — data paths and parquet export layout (`scripts/etl.py`, `scripts/train.py`, `scripts/eval.py`).
- `configs/dist.yaml` — distributed training presets (`scripts/launch_slurm.py`).
- `configs/benchmarks.yaml` — benchmark registry configuration (`scripts/run_benchmarks.py`).
- `configs/benchmarks_local.yaml` — local override for benchmarks pointing at `data/processed/oas_real_full`.
- `configs/therapeutic_bench.yaml` — therapeutic benchmark configuration (dataset curation + eval).
- `configs/esm2_tiny.yaml` — tiny ESM2 probe config (`scripts/train_esm2_probes.py`).
- `configs/dashboard.example.json` — Streamlit dashboard config template.

## Legacy / deprecated

- `configs/legacy/bench.yaml` — legacy benchmark sweep config for `scripts/dev/bench_train.py`.
- `configs/legacy/benchmarks_tmp.yaml` — temporary benchmark config kept for reference.
