# Results Snapshot

Perplexity in this repo refers to **MLM perplexity** on masked tokens (default 15% mask rate).

## Real dataset baselines (OAS)

Source: `benchmarks/results/baseline_example.json`

| Task | Metric | Value | Dataset | Notes |
|------|--------|-------|---------|-------|
| Masked language modeling (Transformer) | Perplexity ↓ | 1.95 | real | Baseline checkpoint |
| CDR identification | Macro F1 ↑ | 0.89 | real | Token classifier on OAS hold-out |
| Liability regression | RMSE ↓ | 0.27 | real | MC-dropout (32 samples) |

## CI guardrails (fixture dataset)

Sources: `benchmarks/results/baseline_ci_transformer.json`, `baseline_ci_mamba.json`, `baseline_ci_dpo.json`

| Model | Metric | Value | Dataset | Notes |
|-------|--------|-------|---------|-------|
| Transformer (CI) | Perplexity ↓ | 26.2007 | synthetic fixture | tiny model, 10 steps |
| Mamba (CI) | Perplexity ↓ | 8.6948e+26 | synthetic fixture | tiny model, 10 steps |
| DPO-aligned (CI) | Perplexity ↓ | 25.1154 | synthetic fixture | tiny DPO, 10 steps |
| DPO-aligned (CI) | Accuracy ↑ | 0.7368 | synthetic fixture | CDR token accuracy |

## Design benchmark

See `outputs/benchmarks/design/summary.json` for the latest run and `scripts/run_design_benchmark.py` for generation settings.

