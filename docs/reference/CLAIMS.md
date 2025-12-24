# Claims & Evidence

This document enumerates **what the repo claims** and the **exact evidence** that supports each claim.
Anything not listed here is a **non‑claim**.

## Claims (and how to reproduce)

1) **We provide an end‑to‑end antibody sequence modeling pipeline (train → eval → generate → DPO).**
   - Evidence:
     - Train: `python scripts/train.py --config-path configs/train.yaml --data-config configs/data.yaml --model-config configs/model.yaml`
     - Eval: `python scripts/eval.py --checkpoint outputs/transformer_run/checkpoints/best.pt --data-config configs/data.yaml --model-config configs/model.yaml`
     - Generate: `python scripts/generate.py --checkpoint outputs/transformer_run/checkpoints/best.pt --num-samples 16 --length 120 --steps 6 --output-dir outputs/generation`
     - DPO: `python scripts/train_dpo.py --synthetic --policy-checkpoint outputs/transformer_run/checkpoints/best.pt --ref-checkpoint outputs/transformer_run/checkpoints/best.pt --output-dir outputs/dpo_run`

2) **We support two backbones (Transformer and reference Mamba/SSM) behind the same API.**
   - Evidence:
     - Transformer run: `--config-overrides "encoder_type=transformer"`
     - Mamba run: `--config-overrides "encoder_type=mamba ssm_d_state=16"`
     - Implementation: `src/abprop/models/transformer.py`, `src/abprop/models/ssm.py`

3) **“Perplexity” refers to MLM perplexity on masked tokens.**
   - Evidence:
     - Definition: `src/abprop/eval/perplexity.py`
     - Documentation: `docs/reference/METHODS.md`

4) **We provide a design benchmark that evaluates constraint satisfaction + diversity.**
   - Evidence:
     - Runner: `python scripts/run_design_benchmark.py --checkpoint <ckpt> --seeds tests/fixtures/toy_sequences.fa --output-dir outputs/benchmarks`
     - Implementation: `src/abprop/benchmarks/design_benchmark.py`

5) **We provide preference-based alignment (DPO) over MLM pseudo‑likelihood.**
   - Evidence:
     - Implementation: `src/abprop/train/dpo.py`
     - Pseudo‑likelihood helper: `src/abprop/eval/pseudolikelihood.py`
     - Documentation: `docs/reference/METHODS.md`

## Non‑claims (explicitly not validated)

- **Binding affinity** or **wet‑lab efficacy** improvements.
- **Clinical developability** beyond the heuristic proxy metrics in this repo.
- **True causal likelihoods** for MLM models (we use pseudo‑likelihood).
- **Generalization to unseen species** without running the zero‑shot benchmark.

If you need any of these claims, add the corresponding datasets, experiments, and evaluation
procedures, then update this document.
