"""Pseudo-likelihood helpers for MLM-style sequence scoring."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from abprop.models import AbPropModel
from abprop.tokenizers import apply_mlm_mask


def build_mlm_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mlm_probability: float = 0.15,
    mask_seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked input ids and MLM labels with deterministic masking by default."""
    generator = torch.Generator(device=input_ids.device)
    seed = 0 if mask_seed is None else int(mask_seed)
    generator.manual_seed(seed)
    return apply_mlm_mask(
        input_ids,
        attention_mask,
        mlm_probability=mlm_probability,
        rng=generator,
    )


def mlm_pseudologp_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    average_log_prob: bool = True,
) -> torch.Tensor:
    """Compute per-sequence pseudo-log-prob from MLM logits and labels."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels shape mismatch.")

    loss_mask = labels != -100
    labels = labels.clone()
    labels[~loss_mask] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1),
        dim=2,
        index=labels.unsqueeze(2),
    ).squeeze(2)

    if average_log_prob:
        denom = loss_mask.sum(dim=-1).clamp_min(1)
        return (per_token_logps * loss_mask).sum(dim=-1) / denom
    return (per_token_logps * loss_mask).sum(dim=-1)


def mlm_pseudologp(
    model: "AbPropModel",
    masked_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    average_log_prob: bool = True,
) -> torch.Tensor:
    """Run the model and compute per-sequence pseudo-log-probability."""
    logits = model(masked_input_ids, attention_mask, tasks=("mlm",))["mlm_logits"]
    return mlm_pseudologp_from_logits(logits, labels, average_log_prob=average_log_prob)


__all__ = ["build_mlm_mask", "mlm_pseudologp", "mlm_pseudologp_from_logits"]
