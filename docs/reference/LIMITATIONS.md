# Limitations

## Modeling
- **Mamba implementation is reference-quality.** The SSM scan is a Python loop (no fused kernels), so throughput is not competitive with optimized backends.
- **MLM objective only.** Perplexity is reported for masked tokens; causal language modeling is not supported out of the box.
- **Generation is heuristic.** The MLM edit sampler provides a practical baseline but does not guarantee optimality or diversity.

## Data & Evaluation
- **Dataset coverage varies.** Species and germline coverage are uneven; interpret results for rare classes cautiously.
- **Synthetic evaluations are proxies.** Synthetic preference data and toy benchmarks are useful for plumbing validation, not scientific conclusions.

## Deployment
- **Inference requires careful input validation.** Sequences with invalid characters are mapped to `X`, and length limits are enforced, but domain-specific validation is still recommended.
- **Checkpoint metadata is best-effort.** Config and overrides are stored in checkpoints, but older checkpoints may lack these fields.
