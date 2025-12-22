"""Perplexity helpers for masked and causal language modeling."""

from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def mlm_perplexity_from_logits(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Compute per-sequence MLM perplexity using masked labels.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab)
        labels: Tensor of shape (batch, seq_len) with masked positions labeled and
            non-MLM positions set to ignore_index.
        ignore_index: Label id to ignore (default: -100).
    """
    vocab = logits.size(-1)
    per_token_loss = F.cross_entropy(
        logits.view(-1, vocab),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.size())
    mask = labels != ignore_index
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_sequence_loss = (per_token_loss * mask).sum(dim=1) / denom
    return torch.exp(per_sequence_loss)


def causal_perplexity_from_logits(
    logits: Tensor,
    labels: Tensor,
    *,
    pad_token_id: int = 0,
) -> Tensor:
    """Compute per-sequence causal (next-token) perplexity.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab)
        labels: Tensor of shape (batch, seq_len)
        pad_token_id: Padding token index to ignore.
    """
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    per_token_loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=pad_token_id,
        reduction="none",
    ).view(shifted_labels.size())
    mask = (shifted_labels != pad_token_id).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_sequence_loss = (per_token_loss * mask).sum(dim=1) / denom
    return torch.exp(per_sequence_loss)


__all__ = ["mlm_perplexity_from_logits", "causal_perplexity_from_logits"]
