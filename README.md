# AbGen

**State-of-the-art Generative Antibody Design**, powered by **Selective State Space Models (Mamba)** and **Direct Preference Optimization (DPO)**.

AbGen moves beyond simple property prediction to true *generative design*. It combines modern engineering (RoPE, RMSNorm) with SSM-based sequence modeling (reference implementation of Mamba) to design antibodies that meet complex developability and binding constraints.

This project is a **research prototype** for in‑silico candidate generation and triage; it does **not** replace wet‑lab validation.

---
## Key Features

- **Next-Gen Architecture**: Choose between **Mamba-S6** (reference implementation; O(L) scan in Python, not optimized) or modernized **Transformers** (RoPE, RMSNorm, SwiGLU).
- **Generative Alignment**: Align models to biophysical constraints using **Direct Preference Optimization (DPO)**—no reinforcement learning required.

---

## Quickstart

### 1. Installation

```bash
# Clone the repo
git clone https://github.com/abgen/abgen.git
cd abgen

# Install with development tools
pip install -e '.[dev,serve,bench,viz,dashboard]'
```

### Documentation Hub

For the full playbook (training, evaluation, generation, DPO, benchmarks), see:
- `docs/README.md`

### 2. Train a Mamba Model

Train a reference Mamba/SSM model on antibody sequences:

```bash
python scripts/train.py \
  --model-config configs/model.yaml \
  --config-overrides "encoder_type=mamba ssm_d_state=16" \
  --output-dir outputs/mamba_run
```

### 3. Train a Transformer Model

Train the modernized Transformer backbone:

```bash
python scripts/train.py \
  --model-config configs/model.yaml \
  --config-overrides "encoder_type=transformer nhead=8 dim_feedforward=2048" \
  --output-dir outputs/transformer_run
```

### 4. Usage Example (Python)

```python
import torch
from abprop.models import AbPropModel, TransformerConfig

# Initialize a Mamba-based Antibody Model
config = TransformerConfig(
    encoder_type="mamba",   # <--- The Novelty Pivot
    d_model=384,
    vocab_size=25,
    ssm_d_state=16
)
model = AbPropModel(config)

# Forward pass (B, L)
input_ids = torch.randint(0, 25, (1, 128))
outputs = model(input_ids, torch.ones_like(input_ids))

print(f"Logits: {outputs['mlm_logits'].shape}")
```

Note: the Python package name is `abprop` even though the project is called **AbGen**.

### 5. DPO Alignment (Synthetic Example)

Align a policy checkpoint with a synthetic preference dataset:

```bash
python scripts/train_dpo.py \
  --synthetic \
  --policy-checkpoint outputs/transformer_run/checkpoints/best.pt \
  --ref-checkpoint outputs/transformer_run/checkpoints/best.pt \
  --output-dir outputs/dpo_run \
  --model-id abgen-dpo-demo
```

---

## Showcase: design loop (5 minutes, CPU)

1) Generate candidates:
```bash
python scripts/generate.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --num-samples 8 --length 120 --steps 6 \
  --output-dir outputs/generation
```

2) Score + rank (liabilities + uncertainty):
```bash
python scripts/generate.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --num-samples 8 --length 120 --steps 6 \
  --score-liabilities --mc-samples 16 \
  --output-dir outputs/generation
```

3) Run the design benchmark:
```bash
python scripts/run_design_benchmark.py \
  --checkpoint outputs/transformer_run/checkpoints/best.pt \
  --seeds tests/fixtures/toy_sequences.fa \
  --output-dir outputs/benchmarks
```

Latest design benchmark snapshot (from `outputs/benchmarks/design/summary.json`):

| Metric | Value |
|---|---:|
| num_candidates | 24 |
| valid_fraction | 0.7083 |
| liability_pass_fraction | 0.9583 |
| unique_fraction | 1.0000 |
| mean_pairwise_identity | 0.0814 |
| mean_reward_delta | 0.0423 |

Example output (from `outputs/generation/<run>/candidates.jsonl`):
```json
{"id":"cand_0000","sequence":"EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS...","length":120,"liabilities":{"nglyc":0.0083,"deamidation":0.0000,"isomerization":0.0000,"oxidation":0.0000,"free_cysteines":0.0000},"uncertainty":{"mean":{"nglyc":0.0102,"deamidation":0.0001,"isomerization":0.0000,"oxidation":0.0000,"free_cysteines":0.0000},"std":{"nglyc":0.0019,"deamidation":0.0001,"isomerization":0.0000,"oxidation":0.0000,"free_cysteines":0.0000}}}
```

## Demo (Gradio)

```bash
cd demo
pip install -r requirements.txt
export ABPROP_DEMO_CHECKPOINT=outputs/transformer_run/checkpoints/best.pt
python app.py
```


---

## Model Zoo & Benchmarks

| ID | Backbone | Method | Metric | Description |
|----|----------|--------|--------|-------------|
| `abgen-mamba-s` | **Mamba (S6)** | DPO | **Aligned** | Generative model aligned for low immunogenicity preference |
| `abgen-base-v2` | Transformer++ | MLM | 1.85 PPL | RoPE + RMSNorm + SwiGLU baseline |
| `abgen-legacy` | BERT | MLM | 1.95 PPL | Legacy architecture (for comparison) |

Model zoo entries are illustrative; use `models/registry.json` for locally tracked runs.

Perplexity in this repo refers to **MLM perplexity** computed on masked tokens (default 15% mask rate), not causal/next-token perplexity.

---

## Project Layout

```
├── configs/                # YAML configs for Training/DPO
├── data/                   # Data provenance & ETL
├── scripts/                # CLI: train, eval, registry
├── src/abprop/            
│   ├── models/             
│   │   ├── ssm.py          # [NEW] Pure PyTorch Mamba Implementation
│   │   ├── layers.py       # [NEW] RoPE, RMSNorm, SwiGLU
│   │   └── transformer.py  # Modernized Backbone
│   ├── train/
│   │   └── dpo.py          # [NEW] Direct Preference Optimization
└── tests/                  
```
