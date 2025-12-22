# AbGen Status Snapshot

Date: 2025-12-22
Scope: Post-reorg, with synthetic guardrail runs and design benchmark.

## Repository audit (quick)
- Code: `src/abprop/` (models, data, eval, train, server, registry, commands)
- Scripts: `scripts/` (core) and `scripts/dev/` (smoke/bench)
- Configs: `configs/` (`train.yaml`, `model.yaml`, `data.yaml`, `benchmarks*.yaml`, `model_mamba.yaml`, `benchmarks_local.yaml`)
- Docs: `docs/` hub with subfolders (`training/`, `design/`, `evaluation/`, `data/`, `reference/`) plus stubs for backward links
- Data: processed OAS at `data/processed/oas_real_full/`
- Outputs of latest guardrails/benchmarks under `outputs/benchmarks*`, `outputs/guardrails/`, `outputs/dpo_guardrail/`, `outputs/benchmarks/design/`

## Validation commands (latest)
```bash
python3 -m compileall -q src scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```
Results: **pass** (1 skipped: CUDA amp not required on CPU)

Smoke:
- `PYTHONPATH=src python3 scripts/dev/verify_backbone.py` ✅
- `PYTHONPATH=src python3 scripts/dev/verify_mamba.py` ✅

Guardrails (synthetic/short runs vs baseline_example): regression checks **failed** (perplexity high; baseline mismatch); smokes passed.
- Transformer summary: `outputs/guardrails/transformer.json`
- Mamba summary: `outputs/guardrails/mamba.json` (perplexity = inf from synthetic run)
- DPO summary: `outputs/guardrails/dpo.json`

Design benchmark:
- `outputs/benchmarks/design/summary.json`
  - valid_fraction 0.7083, liability_pass_fraction 0.9583, unique_fraction 1.0, mean_pairwise_identity 0.0814, mean_reward_delta 0.0423

## What works now
- Config overrides implemented and applied in train/eval/serve; stored in checkpoints.
- Inference-safe batching (`batch_encode` + MLM masking) used in server; avoids training-only fields.
- Perplexity definition standardized to MLM perplexity across repo.
- Parquet-free test fallback (CSV) allows tests to pass without pyarrow/fastparquet.
- DPO pipeline present (`scripts/train_dpo.py`, `scripts/make_preferences.py`), registry accepts positional args.
- Generation via MLM edit sampler (`scripts/generate.py`); design benchmark hooked up.

## Known issues / gaps
- Guardrail regressions are expected because checkpoints are synthetic/short runs and compared to `baseline_example` (perplexity large, Mamba inf).
- Benchmark numbers are not representative; rerun after real training to refresh `outputs/guardrails/*.json` and `docs/evaluation/RESULTS.md`.
- Mamba implementation remains reference-quality (Python scan, no fused kernels); throughput not competitive.
- Perplexity benchmarks may produce inf if model collapses; plots now skip non-finite values but metrics will still be poor until retrained.

## Next actions
1) Train real Transformer/Mamba runs on `data/processed/oas_real_full/`, regenerate benchmarks (`configs/benchmarks_local.yaml`), and rerun `scripts/run_guardrails.py`.
2) Replace `benchmarks/results/baseline_example.json` with a real baseline to make guardrail comparisons meaningful.
3) Populate docs/training and docs/design with real run logs/links after retraining; update `docs/evaluation/RESULTS.md` with new metrics.

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
