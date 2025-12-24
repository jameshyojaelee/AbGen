# Results Snapshot

Perplexity in this repo refers to **MLM perplexity** on masked tokens (default 15% mask rate).

## Real dataset baselines (OAS)

Source: `benchmarks/results/baseline_example.json`

| Task | Metric | Value | Dataset | Notes |
|------|--------|-------|---------|-------|
| Masked language modeling (Transformer) | Perplexity ↓ | 1.95 | real | Baseline checkpoint |
| CDR3 span tagging (substring-derived) | Macro F1 ↑ | 0.89 | real | Token classifier on OAS hold-out |
| Liability regression | RMSE ↓ | 0.27 | real | MC-dropout (32 samples) |

## CI guardrails (fixture dataset)

Sources: `benchmarks/results/baseline_ci_transformer.json`, `baseline_ci_mamba.json`, `baseline_ci_dpo.json`

| Model | Metric | Value | Dataset | Notes |
|-------|--------|-------|---------|-------|
| Transformer (CI) | Perplexity ↓ | 26.2007 | synthetic fixture | tiny model, 10 steps |
| Mamba (CI) | Perplexity ↓ | 8.6948e+26 | synthetic fixture | tiny model, 10 steps |
| DPO-aligned (CI) | Perplexity ↓ | 25.1154 | synthetic fixture | tiny DPO, 10 steps |
| DPO-aligned (CI) | Accuracy ↑ | 0.7368 | synthetic fixture | CDR3 span token accuracy |

## Design benchmark

Latest run summary (from `outputs/benchmarks/design/summary.json`):

| Metric | Value |
|---|---:|
| num_candidates | 24 |
| valid_fraction | 0.7083 |
| liability_pass_fraction | 0.9583 |
| unique_fraction | 1.0000 |
| mean_pairwise_identity | 0.0814 |
| mean_reward_delta | 0.0423 |

See `scripts/run_design_benchmark.py` for generation settings.
