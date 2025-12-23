# Repository Layout

This document describes the canonical repo structure and where to put new work.

## Top-level structure

- `src/abprop/` — core package code (models, data, training, eval, server, utils).
- `scripts/` — CLI entrypoints (thin wrappers around `src/abprop/commands/*`); dev-only tools live in `scripts/dev/`.
- `configs/` — YAML/JSON configs; deprecated files live under `configs/legacy/`.
- `docs/` — documentation hub + topic subfolders; assets in `docs/figures/`.
- `tests/` — unit/integration tests; fixtures in `tests/fixtures/`.
- `data/` — provenance + ETL scripts; raw/interim/processed are gitignored.
- `outputs/`, `mlruns/`, `logs/` — generated artifacts (ignored by git).

## Documentation conventions

- `docs/README.md` is the navigation hub.
- Canonical pages live under:
  - `docs/training/` (training, eval, serve)
  - `docs/design/` (generation + design benchmark)
  - `docs/evaluation/` (benchmarks + results)
  - `docs/data/` (datasets + ETL)
  - `docs/reference/` (methods, limitations, reproducibility, status)
- Root-level files like `docs/METHODS.md` and `docs/RESULTS.md` are **compatibility stubs** that point to the canonical pages.

## Script conventions

- Core CLIs: `scripts/train.py`, `scripts/eval.py`, `scripts/serve.py`, `scripts/run_benchmarks.py`, `scripts/run_design_benchmark.py`.
- Dev/smoke tools: `scripts/dev/*`.
- CI guardrails: `scripts/run_ci_guardrails.py` (trains tiny models + compares against committed baselines).

## Config conventions

- `configs/model.yaml` contains all architecture knobs (transformer + Mamba).
- CI configs live in `configs/data_ci.yaml` and `configs/benchmarks_ci.yaml`.
- Legacy configs go under `configs/legacy/` with a deprecation note.

