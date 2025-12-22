# AbGen/AbProp Status Snapshot

Date: 2025-12-22

## Repository audit (quick)

**Top-level layout (high-signal):**
- `src/abprop/` — core package (models, data, eval, train, server, registry, commands)
- `scripts/` — CLI entrypoints + benchmarks + ETL helpers
- `configs/` — YAML configs (`train.yaml`, `model.yaml`, `data.yaml`, `dist.yaml`, `benchmarks*.yaml`)
- `tests/` — unit + smoke tests
- `docs/` — design docs + results + reproducibility

**Key entrypoints:**
- Console scripts in `pyproject.toml` / `setup.cfg`: `abprop-etl`, `abprop-train`, `abprop-eval`, `abprop-launch`
- CLI wiring: `src/abprop/cli.py`, `src/abprop/commands/*.py`
- Training/Eval wrappers: `scripts/train.py`, `scripts/eval.py`, `scripts/train_cv.py`
- Serving: `src/abprop/server/app.py` + `scripts/serve.py`

**Config surfaces:**
- YAML loaders via `abprop.utils.load_yaml_config` used in train/eval/server
- Primary configs: `configs/train.yaml`, `configs/model.yaml`, `configs/data.yaml`, `configs/dist.yaml`
- Benchmark configs: `configs/bench.yaml`, `configs/benchmarks.yaml`, `configs/therapeutic_bench.yaml`

## Commands executed (default validation)

```bash
python3 -m compileall -q src scripts
```
Result: OK (no output)

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
Result: 6 failures, 1 skip
- Environment-related: Parquet engine unavailable (pyarrow / fastparquet) due to `libssl.so.1.1` missing
- Deterministic: API mismatch in registry + missing required arg in dataloader helper

```bash
PYTHONPATH=src python3 scripts/verify_backbone.py
```
Result: OK
- Forward + backward runs, loss ~3.151

```bash
PYTHONPATH=src python3 scripts/verify_mamba.py
```
Result: Runs, but loss is **0.0** (likely a test objective issue)

## What currently works

- Transformer backbone forward/backward path: `PYTHONPATH=src python3 scripts/verify_backbone.py`
- Mamba forward/backward path executes: `PYTHONPATH=src python3 scripts/verify_mamba.py` (loss is zero; see breakages)
- Basic registry read/write (keyword-only register) and model loading via `abprop.commands.*` pipelines

## What is currently aspirational (README claims not yet wired)

- `--config-overrides` CLI flag referenced in README but not implemented in train/eval CLIs.
- “Linear-time Mamba” scaling (current SSM uses a Python loop over sequence length).
- DPO training / preference pipeline: README claims “Generative Alignment (DPO)” but no CLI (`abprop.commands.dpo`) or scripts exist.
- End-to-end generation API (README markets “generative design,” but there is no sampling/generation CLI).

## Top breakages (with pointers)

1) **Parquet engine missing (env)** — tests that write parquet fail because `pyarrow`/`fastparquet` cannot load (libssl missing).
   - `tests/test_dataset.py:41` (`df.to_parquet` in `_make_parquet_dataset`)
   - `tests/test_dataset.py:141` (synthetic parquet write)

2) **ETL parquet write fails (env)** — ETL uses pyarrow; fails on this environment.
   - `src/abprop/data/etl.py:196` (pandas `to_parquet` call)
   - `tests/test_etl_schema.py:116` (calls `run_etl`)

3) **Missing required arg in dataloader helper (deterministic)** — `build_oas_dataloaders` requires `mlm_probability`, tests omit it.
   - `src/abprop/commands/train.py:124` (signature)
   - `tests/test_dataset.py:188` (call site)

4) **Registry API mismatch (deterministic)** — `ModelRegistry.register` is keyword-only but tests use positional args.
   - `src/abprop/registry/json_registry.py:52`
   - `tests/test_registry.py:30`

5) **Mamba verification loss is zero (deterministic)** — current check uses labels identical to inputs; loss can be 0 depending on masking.
   - `scripts/verify_mamba.py:28`

6) **Inference server uses training collate (deterministic)** — server expects training-only fields and can `KeyError` on plain sequences.
   - `src/abprop/server/app.py:59` (uses `build_collate_fn` for inference)

7) **Mamba uses shared norm + no dropout (deterministic)** — single `norm_f` reused for all layers, no dropout, pad mask ignored.
   - `src/abprop/models/ssm.py:209` (shared norm, no dropout)
   - `src/abprop/models/ssm.py:220` (attention_mask ignored)

8) **Linear-time claim not accurate (doc/code mismatch)** — SSM uses a Python loop over sequence length.
   - `src/abprop/models/ssm.py:132` (explicit per-token loop)

9) **Config overrides claimed but unsupported (doc/code mismatch)** — README examples include `--config-overrides` but CLI args don’t exist.
   - `README.md:36`
   - `src/abprop/commands/train.py` / `src/abprop/commands/eval.py` (arg parsing)

10) **DPO training pipeline missing (doc/code mismatch)** — README markets DPO but there is no CLI/script.
    - `README.md:7`
    - `src/abprop/commands/` (no `dpo.py`)

## Next actions

- Address deterministic API mismatches (registry + dataloader defaults).
- Implement config overrides and config wiring for train/eval/server.
- Build inference-safe collate for server.
- Decide on perplexity definition and enforce consistency.
- Decide how to handle parquet in tests (required vs fallback).
