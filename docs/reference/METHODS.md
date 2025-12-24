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

## Preference dataset schema

Preference data is stored as JSONL with the following fields:

- `prompt`: conditioning prefix (empty string for unconditional runs)
- `chosen`: preferred candidate sequence
- `rejected`: less preferred candidate sequence
- `metadata`: arbitrary dict (e.g., seed sequence, reward breakdowns)

During batching, the model input is formed by concatenating `prompt + chosen` (or
`prompt + rejected`) **only if** `prompt` is non-empty. For unconditional runs,
`prompt` is set to `""` so the model sees only the candidate sequence.

## DPO alignment

DPO training is implemented in `src/abprop/train/dpo.py` and exposed via `scripts/train_dpo.py`.
The policy and reference models are scored on MLM-masked sequences; the DPO loss is applied
with a configurable `beta` and optional label smoothing.
Masking is deterministic by default (seeded) and the same masked positions are used for
policy and reference scores to ensure fair comparison.

This is a **pseudo-likelihood** objective: log-probabilities are computed only on masked
positions, not a true causal sequence likelihood. It is a practical approximation for
MLM models and should be interpreted as an alignment signal rather than a full
generative likelihood.
