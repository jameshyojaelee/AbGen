# Generation & Design

## Generate candidates (MLM edit sampler)

```bash
python scripts/generate.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --num-samples 16 \
  --length 120 \
  --steps 6 \
  --output-dir outputs/generation
```

## Build preferences + DPO

Build preference pairs:
```bash
python scripts/make_preferences.py \
  --input tests/fixtures/toy_sequences.fa \
  --output outputs/preferences/pairs.jsonl
```

Run DPO alignment:
```bash
python scripts/train_dpo.py \
  --preferences outputs/preferences/pairs.jsonl \
  --policy-checkpoint outputs/transformer_run/checkpoints/best.pt \
  --ref-checkpoint outputs/transformer_run/checkpoints/best.pt \
  --output-dir outputs/dpo_run
```

## Design benchmark

```bash
python scripts/run_design_benchmark.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --seeds tests/fixtures/toy_sequences.fa \
  --output-dir outputs/benchmarks
```

