# AbGen Playbook

This playbook is the navigation hub for day-to-day AbGen workflows: setup, data, training,
preference modeling, generation, evaluation, and deployment.

---

## Navigation

- **Training**: `docs/training/`
- **Design & Preferences**: `docs/design/`
- **Evaluation**: `docs/evaluation/`
- **Data**: `docs/data/`
- **Reference**: `docs/reference/`

Key docs:
- Results: [docs/evaluation/RESULTS.md](evaluation/RESULTS.md)
- Methods: [docs/design/METHODS.md](design/METHODS.md)
- Limitations: [docs/reference/LIMITATIONS.md](reference/LIMITATIONS.md)
- Reproducibility: [docs/reference/REPRODUCIBILITY.md](reference/REPRODUCIBILITY.md)
- Status snapshot: [docs/reference/STATUS.md](reference/STATUS.md)
- Case studies: [docs/design/CASE_STUDIES.md](design/CASE_STUDIES.md)
- Leaderboard: [docs/evaluation/LEADERBOARD.md](evaluation/LEADERBOARD.md)

---

## Environment & Tooling

```bash
# Pip
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,serve,bench,viz,dashboard]'

# Conda
conda env create -f environment.yml && conda activate abgen

# Docker
docker build -t abgen . && docker run -it --rm -v "$PWD":/workspace -w /workspace abgen bash
```

---

## Core Workflows

### 1) Train (Transformer)
```bash
python scripts/train.py \
  --config-path configs/train.yaml \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --config-overrides "encoder_type=transformer" \
  --output-dir outputs/transformer_run
```

### 2) Train (Mamba)
```bash
python scripts/train.py \
  --config-path configs/train.yaml \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --config-overrides "encoder_type=mamba ssm_d_state=16" \
  --output-dir outputs/mamba_run
```

### 3) Evaluate
```bash
python scripts/eval.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --splits val test \
  --uncertainty --mc-samples 32 \
  --output outputs/eval/val_eval.json
```

### 4) Generate Candidates
```bash
python scripts/generate.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --num-samples 16 \
  --length 120 \
  --steps 6 \
  --output-dir outputs/generation
```

### 5) Build Preferences
```bash
python scripts/make_preferences.py \
  --input examples/attention_success.fa \
  --output outputs/preferences/pairs.jsonl
```

### 6) DPO Alignment
```bash
python scripts/train_dpo.py \
  --synthetic \
  --policy-checkpoint outputs/transformer_run/checkpoints/best.pt \
  --ref-checkpoint outputs/transformer_run/checkpoints/best.pt \
  --output-dir outputs/dpo_run
```

---

## Visualization & Dashboards

- Attention: `python scripts/visualize_attention.py --checkpoint ... --sequence ... --output outputs/attention`
- Embeddings: `python scripts/visualize_embeddings.py --checkpoints ... --parquet ... --output docs/figures/embeddings`
- Streamlit: `python scripts/launch_dashboard.py --root outputs --config configs/dashboard.example.json`
- Gradio demo: `python demo/app.py`

---

## Benchmarks

```bash
python scripts/run_benchmarks.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --config configs/benchmarks.yaml
```

Design benchmark:
```bash
python scripts/run_design_benchmark.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --seeds tests/fixtures/toy_sequences.fa
```

---

## Reference

- Repo layout: [docs/REPO_LAYOUT.md](REPO_LAYOUT.md)
- Config catalog: [configs/README.md](../configs/README.md)
- Script catalog: [scripts/README.md](../scripts/README.md)
