# AbGen

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Backbone-Mamba%2FSSM-purple.svg)](src/abprop/models/ssm.py)
[![Design](https://img.shields.io/badge/Design-DPO-red.svg)](src/abprop/train/dpo.py)

**State-of-the-art Generative Antibody Design**, powered by **Selective State Space Models (Mamba)** and **Direct Preference Optimization (DPO)**.

AbGen moves beyond simple property prediction to true *generative design*. It combines modern engineering (RoPE, RMSNorm) with SSM-based sequence modeling (reference implementation of Mamba) to design antibodies that meet complex developability and binding constraints.

---
## Key Features

- **Next-Gen Architecture**: Choose between **Mamba-S6** (reference implementation; O(L) scan in Python, not optimized) or modernized **Transformers** (RoPE, RMSNorm, SwiGLU).
- **Generative Alignment**: Align models to biophysical constraints using **Direct Preference Optimization (DPO)**—no reinforcement learning required.
- **Battle-Tested Engineering**: End-to-end tooling including ETL, distributed training, benchmark registries, and Streamlit dashboards.

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
