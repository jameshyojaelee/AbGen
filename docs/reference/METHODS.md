# Methods

## Perplexity definition

This repo reports **MLM perplexity** (masked-token perplexity) computed on masked tokens only.
Causal/next-token loss is reported separately as `causal_ppl` or `next_token_nll` where relevant.

## Preference reward function

Preference construction uses a reward that penalizes undesirable antibody features:

- **Motif liabilities penalty**: counts known liability motifs (length-normalized).
- **Non-canonical residue penalty**: fraction of residues not in the 20 AA alphabet.
- **Uncertainty penalty** (optional): MC-dropout variance from the regression head.
- **Humanness proxy** (optional): length-range deviation penalty.

Implementation: `src/abprop/rewards.py`, used by `scripts/make_preferences.py`.

## DPO alignment

DPO training is implemented in `src/abprop/train/dpo.py` and exposed via `scripts/train_dpo.py`.
The policy and reference models are scored on MLM-masked sequences; the DPO loss is applied
with a configurable `beta` and optional label smoothing.

