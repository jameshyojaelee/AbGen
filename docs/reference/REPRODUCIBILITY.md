# Reproducibility

## Checkpoint metadata

Training checkpoints store the following metadata:

- `model_config`: the full model configuration used at train time
- `config_overrides`: the exact CLI overrides string passed to `scripts/train.py`

These are written by `src/abprop/commands/train.py` into the checkpoint metadata and
reloaded by `abprop.utils.extract_model_config` when evaluating or serving.

## Recommended run metadata

- Save the resolved config snapshot (train/model/data) alongside checkpoints.
- Record the git SHA and data export version.
- Fix `seed` in `configs/train.yaml` to ensure determinism.

## Data expectations

Processed data is expected under `data/processed/oas_real_full/`. Raw/interim/processed
folders are gitignored. Small, tracked fixtures live under `tests/fixtures/`.

