# Training

- Transformer: `python scripts/train.py --config-path configs/train.yaml --data-config configs/data.yaml --model-config configs/model.yaml --output-dir outputs/transformer_run`
- Mamba: `python scripts/train.py --config-overrides "encoder_type=mamba" --output-dir outputs/mamba_run`
- Cross-validation: `scripts/train_cv.py`
- DPO alignment: `scripts/train_dpo.py` (policy + reference checkpoints)
- Synthetic smoke: `scripts/dev/verify_backbone.py`, `scripts/dev/verify_mamba.py`
