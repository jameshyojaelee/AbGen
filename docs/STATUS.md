# Status Report (2025-12-23)

## Environment snapshot

- Python: 3.10.12
- OS: Linux (container/host)
- Repo root: /gpfs/commons/home/jameslee/AbGen

## Quick repo audit

Top-level directories:
- `src/abprop/` (core package code)
- `scripts/` (CLI entrypoints; dev tools in `scripts/dev/`)
- `configs/` (model/train/data/bench configs)
- `docs/` (documentation hub + subfolders)
- `tests/` (unit/integration tests + fixtures)

Key entrypoints:
- `scripts/train.py` → `src/abprop/commands/train.py`
- `scripts/eval.py` → `src/abprop/commands/eval.py`
- `scripts/train_dpo.py` → `src/abprop/commands/dpo.py`
- `scripts/run_benchmarks.py`
- `scripts/run_design_benchmark.py`
- `scripts/run_ci_guardrails.py`

Config surfaces:
- `configs/train.yaml`, `configs/model.yaml`, `configs/data.yaml`
- `configs/benchmarks.yaml` (full) + `configs/benchmarks_ci.yaml` (fixture)
- `configs/data_ci.yaml` (fixture)

## Commands run + outcomes

- `python3 -m compileall -q src scripts`
  - Result: **PASS** (no output)
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`
  - Result: **PASS** (1 skipped: CUDA AMP not required on CPU)
- `PYTHONPATH=src python3 scripts/dev/verify_backbone.py`
  - Result: **PASS** (MLM logits + regression heads; backward OK)
- `PYTHONPATH=src python3 scripts/dev/verify_mamba.py`
  - Result: **PASS** (non-zero loss; backward OK)

## Aspirational / not-yet-wired claims

- README “Model Zoo” entries are illustrative (no tracked checkpoints for those IDs yet): `README.md:99-105`.
- Real-data benchmarks depend on local processed OAS exports (not tracked in git); CI uses fixtures instead.

## Top 10 breakages / env-dependent pitfalls

1. Missing processed dataset path halts training: `src/abprop/commands/train.py:168`.
2. Missing split raises a hard error: `src/abprop/data/dataset.py:105`.
3. Risk stratification can produce NaNs on small/imbalanced splits: `src/abprop/benchmarks/liability_benchmark.py:262-272`.
4. FastAPI server requires optional dependency: `src/abprop/server/app.py:296-297`.
5. UMAP reducer requires optional dependency: `src/abprop/viz/embeddings.py:163-167`.
6. Streamlit dashboard import requires optional dependency: `src/abprop/viz/dashboard.py:11-12`.
7. Dashboard launcher shells out to `streamlit` (fails if not installed): `scripts/launch_dashboard.py:34-48`.
8. Duplicate registry IDs raise an error: `src/abprop/registry/json_registry.py:61-62`.
9. Invalid/unknown config overrides raise errors by default: `src/abprop/utils/overrides.py:95-109`.
10. `scripts/run_benchmarks.py --strict` exits non-zero if benchmarks fail or metrics are missing: `scripts/run_benchmarks.py:510-515`.

## Next actions (if needed)

- Track real checkpoints for the README “Model Zoo” entries or remove the table.
- Expand fixture datasets if CI guardrail metrics are too noisy.

