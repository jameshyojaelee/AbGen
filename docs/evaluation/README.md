# Benchmarks & Guardrails

## Run benchmarks

```bash
python scripts/run_benchmarks.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --config configs/benchmarks.yaml \
  --output-dir outputs/benchmarks
```

## CI guardrails (fixture-based)

```bash
python scripts/run_ci_guardrails.py \
  --train-steps 10 \
  --dpo-steps 10 \
  --batch-size 2 \
  --max-samples 32
```

Baselines live under `benchmarks/results/`:
- `baseline_ci_transformer.json`
- `baseline_ci_mamba.json`
- `baseline_ci_dpo.json`

## Results

See `docs/evaluation/RESULTS.md` for the current results snapshot.

