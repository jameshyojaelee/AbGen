# AbGen Playbook

This is the consolidated documentation for the AbGen codebase (package name: `abprop`): setup, training, evaluation,
generation, preferences/DPO, benchmarks, and guardrails.

## Navigation

- Training & eval: `docs/training/README.md`
- Generation & design: `docs/design/README.md`
- Benchmarks & guardrails: `docs/evaluation/README.md`
- Data & ETL: `docs/data/README.md`
- Reference docs: `docs/reference/README.md`

## Environment

```bash
# Pip
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,serve,bench,viz,dashboard]'

# Conda
conda env create -f environment.yml && conda activate abgen

# Docker
docker build -t abgen . && docker run -it --rm -v "$PWD":/workspace -w /workspace abgen bash
```

## Core workflows

### Train

Transformer:
```bash
python scripts/train.py \
  --config-path configs/train.yaml \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --config-overrides "encoder_type=transformer" \
  --output-dir outputs/transformer_run
```

Mamba (reference implementation; Python scan, not fused kernels):
```bash
python scripts/train.py \
  --config-path configs/train.yaml \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --config-overrides "encoder_type=mamba ssm_d_state=16" \
  --output-dir outputs/mamba_run
```

### Evaluate

```bash
python scripts/eval.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --splits val test \
  --uncertainty --mc-samples 32 \
  --output-dir outputs/eval
```

### Serve

```bash
python scripts/serve.py --checkpoint outputs/transformer_run/checkpoints/best.pt --model-config configs/model.yaml
```

### Generate candidates (MLM edit sampler)

```bash
python scripts/generate.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --num-samples 16 \
  --length 120 \
  --steps 6 \
  --output-dir outputs/generation
```

### Build preferences + DPO

Build preference pairs:
```bash
python scripts/make_preferences.py \
  --input tests/fixtures/toy_sequences.fa \
  --output outputs/preferences/pairs.jsonl
```

Run synthetic DPO (plumbing sanity check):
```bash
python scripts/train_dpo.py \
  --synthetic \
  --policy-checkpoint outputs/transformer_run/checkpoints/best.pt \
  --ref-checkpoint outputs/transformer_run/checkpoints/best.pt \
  --output-dir outputs/dpo_run
```

## Benchmarks and guardrails

Run standard benchmarks (uses `configs/benchmarks.yaml`):
```bash
python scripts/run_benchmarks.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --config configs/benchmarks.yaml \
  --output-dir outputs/benchmarks
```

Design benchmark (generation-centric):
```bash
python scripts/run_design_benchmark.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --seeds tests/fixtures/toy_sequences.fa \
  --output-dir outputs/benchmarks
```

Guardrails (regression + smoke):
```bash
python scripts/run_ci_guardrails.py \
  --train-steps 10 \
  --dpo-steps 10 \
  --batch-size 2 \
  --max-samples 32
```

Notes:
- Uses the tracked fixture dataset (`tests/fixtures/oas_fixture.csv`) and CI benchmark config (`configs/benchmarks_ci.yaml`).
- Compares against committed baselines: `benchmarks/results/baseline_ci_transformer.json`, `benchmarks/results/baseline_ci_mamba.json`, `benchmarks/results/baseline_ci_dpo.json`.

## Results snapshot

Perplexity in this repo refers to **MLM perplexity** computed on masked tokens (default 15% mask rate), not causal/next-token perplexity.

| Task | Metric | Validation | Test | Dataset | Notes |
|------|--------|------------|------|---------|-------|
| Masked language modeling (Transformer) | Perplexity ↓ | **1.95** | 2.01 | real | Baseline checkpoint (`benchmarks/results/baseline_example.json`) |
| Masked language modeling (Mamba) | Perplexity ↓ | inf | inf | real | Synthetic-trained Mamba guardrail run; perplexity diverged on real eval |
| Preference alignment (DPO) | DPO loss ↓ | 0.686 | 0.686 | synthetic | Last-step synthetic DPO loss from `outputs/dpo_guardrail/metrics.json` |
| CDR3 span tagging (substring-derived) | Macro F1 ↑ | **0.89** | 0.88 | real | Token classifier on OAS hold-out |
| Liability regression | RMSE ↓ | **0.27** | 0.29 | real | MC-dropout (32 samples) uncertainty < 0.05 median |

Design benchmark (`outputs/benchmarks/design/summary.json`):
- valid_fraction: 0.7083
- liability_pass_fraction: 0.9583
- unique_fraction: 1.0
- mean_pairwise_identity: 0.0814
- mean_reward_delta: 0.0423

## Methods (high-level)

Preference reward function:
- Penalizes motif liabilities (length-normalized), non-canonical residues, optional uncertainty (MC-dropout variance), and optional length-range proxy.
- Implemented in `src/abprop/rewards.py` and used by `scripts/make_preferences.py`.

DPO:
- Implemented in `src/abprop/train/dpo.py` and exposed via `scripts/train_dpo.py`.

## Reproducibility (high-signal)

- Training checkpoints store `model_config` and `config_overrides` for replayable eval/serve.
- Data: processed exports expected under `data/processed/oas_real_full/` (raw/interim/processed are ignored by git).
- Default sanity checks:
  - `python3 -m compileall -q src scripts`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`

## Limitations

- Mamba encoder uses a reference Python scan (no fused kernels) → not throughput-competitive.
- “Generation” is a heuristic MLM edit sampler baseline (not guaranteed optimal/diverse).
- Synthetic preference runs validate plumbing, not scientific conclusions.

## Repo layout (consolidated)

- `src/abprop/` — package code
- `scripts/` — CLIs (core) and `scripts/dev/` smoke/bench
- `configs/` — configs (`configs/legacy/` deprecated)
- `docs/` — hub + topic subfolders (`training/`, `design/`, `evaluation/`, `data/`, `reference/`)
- `tests/` — unit/integration tests
- `outputs/`, `mlruns/` — generated artifacts (ignored by git)
