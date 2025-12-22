# Reproducibility Checklist

## Environment
- Python 3.10
- Install via `pip install -r requirements.txt`, `conda env create -f environment.yml`, or `docker build -t abgen .`
- GPU optional (CUDA 11.8 tested). CPU paths also supported.

## Seeds
- Training scripts default to `seed: 42` (see `configs/train.yaml`).
- Set `PYTHONHASHSEED=0` for deterministic hashing when running distributed jobs.

## Data Access
- Processed parquet splits under `data/processed/oas_real_full` (see `data/DATA_PROVENANCE.md`).
- Ensure identical snapshot across team by sharing checksums.

## One-Click Pipelines

```bash
# Full pipeline
scripts/reproduce_all.sh

# Minimal visualization sanity check
scripts/reproduce_minimal.sh
```

## Artifacts Generated
- Evaluation metrics: `outputs/eval/*.json`
- Attention and embedding visualizations: `outputs/attention/`, `docs/figures/embeddings/`
- Publication figures: `docs/figures/publication/`
- Registry snapshot: `models/registry.json`
- Training config snapshot: `outputs/config_snapshot.json`

## Checkpoint Configuration
- Training checkpoints now store `model_config` (full `TransformerConfig` fields) inside the `.pt` file.
- CLI overrides passed via `--config-overrides` are recorded as `config_overrides` in the checkpoint
  and in `outputs/config_snapshot.json`.
- Eval/serve scripts prefer checkpoint `model_config` and then apply any explicit CLI overrides.

## Continuous Checks
- GitHub Action `.github/workflows/benchmark.yml` runs scheduled regression guardrails.
- `scripts/check_regression.py` compares new benchmark results against the committed baseline.

## Troubleshooting
- **Missing checkpoints**: run `python scripts/train.py ...` or pull from the registry via `scripts/registry.py show --id <model>`.
- **Visualization skipping**: scripts emit warnings if required files are missing; populate them before rerunning.
- **Docker GPU**: add `--gpus all` when launching containers with CUDA support.
