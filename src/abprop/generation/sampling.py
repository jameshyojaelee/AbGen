"""Sampling utilities for generating antibody sequences with MLM models."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch

from abprop.models import AbPropModel
from abprop.tokenizers import AMINO_ACIDS, SPECIAL_TOKENS, TOKEN_TO_ID, decode


def _filter_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    invalid_token_ids: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature
    if invalid_token_ids:
        logits = logits.clone()
        logits[..., list(invalid_token_ids)] = -float("inf")

    vocab = logits.size(-1)
    if top_k is not None:
        k = max(1, min(int(top_k), vocab))
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(-1, topk_idx, topk_vals)
        logits = filtered

    if top_p is not None:
        p = float(top_p)
        if not 0.0 < p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        remove = cumprobs > p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        filtered = torch.full_like(logits, -float("inf"))
        filtered.scatter_(-1, sorted_idx, sorted_logits)
        logits = filtered

    return logits


def sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    invalid_token_ids: Optional[Sequence[int]] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    filtered = _filter_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        invalid_token_ids=invalid_token_ids,
    )
    probs = torch.softmax(filtered, dim=-1)
    vocab = probs.size(-1)
    samples = torch.multinomial(probs.view(-1, vocab), 1, generator=generator)
    return samples.view(logits.size()[:-1])


def sample_mlm_edit(
    model: AbPropModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    steps: int = 5,
    mask_rate: float = 0.15,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Iteratively edit sequences by masking and resampling tokens.

    This strategy aligns with MLM training objectives by replacing random
    positions with samples from the masked-token distribution.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= mask_rate <= 1.0:
        raise ValueError("mask_rate must be in [0, 1]")

    model.eval()
    device = input_ids.device
    generator = generator or torch.Generator(device=device)

    special_ids = {TOKEN_TO_ID[token] for token in SPECIAL_TOKENS}
    mask_id = TOKEN_TO_ID["<mask>"]

    ids = input_ids.clone()
    attn = attention_mask.bool()

    with torch.no_grad():
        for _ in range(steps):
            candidate_mask = attn & ~torch.isin(ids, torch.tensor(list(special_ids), device=device))
            rand = torch.rand(ids.shape, device=device, generator=generator)
            mask_positions = (rand < mask_rate) & candidate_mask

            # Ensure at least one position masked per sequence.
            for row in range(ids.size(0)):
                if mask_positions[row].any():
                    continue
                candidates = torch.nonzero(candidate_mask[row], as_tuple=False).view(-1)
                if candidates.numel() == 0:
                    continue
                pick = candidates[torch.randint(candidates.numel(), (1,), generator=generator)]
                mask_positions[row, pick] = True

            masked_ids = ids.clone()
            masked_ids[mask_positions] = mask_id
            logits = model(masked_ids, attn, tasks=("mlm",))["mlm_logits"]
            sampled = sample_from_logits(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                invalid_token_ids=list(special_ids),
                generator=generator,
            )
            ids[mask_positions] = sampled[mask_positions]

    return ids


def decode_sequences(token_ids: torch.Tensor) -> List[str]:
    return [decode(seq.tolist(), strip_special=True) for seq in token_ids]


def random_sequences(
    num_samples: int,
    length: int,
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate random amino-acid sequences encoded as token ids (no specials)."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if length <= 0:
        raise ValueError("length must be positive")
    device = device or torch.device("cpu")
    generator = generator or torch.Generator(device=device)

    vocab = [TOKEN_TO_ID[token] for token in AMINO_ACIDS]
    idx = torch.randint(0, len(vocab), (num_samples, length), device=device, generator=generator)
    ids = torch.tensor(vocab, device=device, dtype=torch.long)[idx]
    return ids


__all__ = [
    "sample_mlm_edit",
    "sample_from_logits",
    "decode_sequences",
    "random_sequences",
]
