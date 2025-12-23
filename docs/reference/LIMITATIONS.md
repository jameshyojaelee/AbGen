# Limitations

- Mamba encoder uses a reference Python scan (no fused kernels); throughput is not competitive.
- Generation uses a heuristic MLM edit sampler baseline (not optimized for global optimum/diversity).
- Synthetic preference runs validate plumbing only; they are not scientific conclusions.
- Real-data benchmarks require local access to processed OAS exports (not tracked in git).

