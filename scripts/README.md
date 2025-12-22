# Scripts Catalog

## Core (supported)

- `scripts/train.py` — train model via `abprop.commands.train`
- `scripts/eval.py` — evaluate checkpoints
- `scripts/etl.py` — run ETL pipeline
- `scripts/train_cv.py` — clonotype-aware cross-validation
- `scripts/train_dpo.py` — DPO alignment training
- `scripts/generate.py` — sequence generation (MLM edit sampler)
- `scripts/make_preferences.py` — build preference pairs from sequences
- `scripts/run_benchmarks.py` — run benchmark suite
- `scripts/run_design_benchmark.py` — generation-centric design benchmark
- `scripts/run_guardrails.py` — regression + smoke guardrails (transformer/mamba/DPO)
- `scripts/registry.py` — model registry CLI
- `scripts/serve.py` — FastAPI server wrapper
- `scripts/visualize_attention.py` — attention visualization
- `scripts/visualize_embeddings.py` — embedding visualization
- `scripts/launch_dashboard.py` — Streamlit dashboard launcher

## Dev / Smoke (best-effort)

- `scripts/dev/verify_backbone.py` — transformer smoke test
- `scripts/dev/verify_mamba.py` — Mamba smoke test
- `scripts/dev/verify_benchmarks.py` — benchmark smoke test
- `scripts/dev/bench_dataloader.py` — dataloader micro-benchmark
- `scripts/dev/bench_train.py` — training micro-benchmark
- `scripts/dev/bench_mamba.py` — Mamba vs transformer runtime benchmark

Legacy wrappers remain at the old paths for compatibility (e.g., `scripts/verify_mamba.py`).
