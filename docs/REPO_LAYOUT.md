# Repository Layout

## Inventory (Top-Level)
- `src/abprop/` — package code (models, data, eval, train, server)
- `scripts/` — CLIs (core) and `scripts/dev/` (smoke/bench)
- `configs/` — YAML/JSON configs; `configs/legacy/` for deprecated
- `docs/` — docs hub with subfolders and backward-compatible stubs at old paths
- `tests/` — unit/integration tests
- `data/` — raw/interim/processed (raw ignored by git)
- `outputs/` — generated artifacts (benchmarks, checkpoints, logs)

## Docs hub
- `docs/README.md` — navigation
- `docs/training/` — training/DPO quickstarts
- `docs/design/` — methods + case studies
- `docs/evaluation/` — results/leaderboard + guardrails
- `docs/data/` — dataset pointers
- `docs/reference/` — status, reproducibility, limitations
- Stubs removed; links should target the subfolders above.

## Scripts
- Core: `scripts/train.py`, `scripts/eval.py`, `scripts/etl.py`, `scripts/run_benchmarks.py`, `scripts/run_guardrails.py`, `scripts/generate.py`, `scripts/train_dpo.py`.
- Dev/smoke: `scripts/dev/verify_backbone.py`, `scripts/dev/verify_mamba.py`, `scripts/dev/bench_*`.
- Wrappers at old paths forward to `scripts/dev/` counterparts for compatibility.

## Configs
- `configs/train.yaml`, `configs/model.yaml`, `configs/data.yaml`
- Benchmarks: `configs/benchmarks.yaml` (default), `configs/benchmarks_local.yaml` (local OAS paths)
- Model variants: `configs/model_mamba.yaml`
- Legacy: `configs/legacy/`

## Generated artifacts
- Benchmarks: `outputs/benchmarks*/summary.json`
- Guardrails: `outputs/guardrails/*.json`
- DPO: `outputs/dpo_guardrail/`
- Design benchmark: `outputs/benchmarks/design/`
