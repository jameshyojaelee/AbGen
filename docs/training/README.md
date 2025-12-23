# Training & Evaluation

## Train

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

## Evaluate

```bash
python scripts/eval.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --splits val test \
  --uncertainty --mc-samples 32 \
  --output-dir outputs/eval
```

## Serve

```bash
python scripts/serve.py --checkpoint outputs/transformer_run/checkpoints/best.pt --model-config configs/model.yaml
```

