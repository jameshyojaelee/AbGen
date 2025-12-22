# AbGen

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Backbone-Mamba%2FSSM-purple.svg)](src/abprop/models/ssm.py)
[![Design](https://img.shields.io/badge/Design-DPO-red.svg)](src/abprop/train/dpo.py)

**State-of-the-art Generative Antibody Design**, powered by **Selective State Space Models (Mamba)** and **Direct Preference Optimization (DPO)**.

AbGen moves beyond simple property prediction to true *generative design*. It combines modern engineering (RoPE, RMSNorm) with cutting-edge sequence modeling (Linear-time SSMs) to design antibodies that meet complex developability and binding constraints.

---

## Key Features

- **⚡ Next-Gen Architecture**: Choose between **Mamba-S6** (linear scaling for long complexes) or modernized **Transformers** (RoPE, RMSNorm, SwiGLU).
- **🎨 Generative Alignment**: Align models to biophysical constraints using **Direct Preference Optimization (DPO)**—no reinforcement learning required.
- **🛡️ Battle-Tested Engineering**: End-to-end tooling including ETL, distributed training, benchmark registries, and Streamlit dashboards.

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

### 2. Train a Mamba Model

Train a linearly-scaling State Space Model on antibody sequences:

```bash
python scripts/train.py \
  --model-config configs/model.yaml \
  --config-overrides "encoder_type=mamba ssm_d_state=16" \
  --output-dir outputs/mamba_run
```

### 3. Usage Example (Python)

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

---

## Model Zoo & Benchmarks

| ID | Backbone | Method | Metric | Description |
|----|----------|--------|--------|-------------|
| `abgen-mamba-s` | **Mamba (S6)** | DPO | **Aligned** | Generative model aligned for low immunogenicity preference |
| `abgen-base-v2` | Transformer++ | MLM | 1.85 PPL | RoPE + RMSNorm + SwiGLU baseline |
| `abprop-legacy` | BERT | MLM | 1.95 PPL | Legacy architecture (for comparison) |

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

---

## Contributing

We welcome contributions on:
1.  **New Rewards**: Implementing biophysical reward models for DPO alignment.
2.  **Structural Modalities**: Integrating Foldseek structural tokens into the Mamba encoder.

---

*AbGen is released under the Apache 2.0 License.*
