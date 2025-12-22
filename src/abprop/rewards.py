"""Reward builders for antibody design preferences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

from abprop.tokenizers import batch_encode
from abprop.utils import find_motifs, normalize_by_length

CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class RewardComponents:
    motif_penalty: float
    noncanonical_penalty: float
    uncertainty_penalty: float
    humanness_penalty: float
    total_reward: float
    metadata: Dict[str, float]

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def motif_liability_penalty(sequence: str) -> Tuple[float, Dict[str, float]]:
    seq = sequence.upper()
    counts = find_motifs(seq)
    normalized = normalize_by_length(counts, len(seq))
    penalty = float(sum(normalized.values()))
    return penalty, {f"motif_{k}": float(v) for k, v in normalized.items()}


def noncanonical_residue_penalty(sequence: str) -> float:
    seq = sequence.upper()
    if not seq:
        return 0.0
    noncanonical = sum(1 for res in seq if res not in CANONICAL_AMINO_ACIDS)
    return float(noncanonical / len(seq))


def humanness_proxy_penalty(sequence: str, length_range: Tuple[int, int] = (90, 130)) -> float:
    """Lightweight proxy based on expected antibody length range."""
    seq_len = len(sequence)
    low, high = length_range
    if low <= seq_len <= high:
        return 0.0
    if seq_len < low:
        return float((low - seq_len) / max(1, low))
    return float((seq_len - high) / max(1, high))


def uncertainty_penalty(
    sequence: str,
    *,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    mc_samples: int = 0,
) -> float:
    if model is None or mc_samples <= 0:
        return 0.0
    batch = batch_encode([sequence], add_special=True, max_length=getattr(model.config, "max_position_embeddings", 1024))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    samples = model.stochastic_forward(
        input_ids,
        attention_mask,
        tasks=("reg",),
        mc_samples=mc_samples,
        enable_dropout=True,
        no_grad=True,
    )
    stacked = torch.stack([out["regression"].detach() for out in samples], dim=0)
    variance = stacked.var(dim=0, unbiased=False)
    return float(variance.mean().item())


def build_reward(
    sequence: str,
    *,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    mc_samples: int = 0,
    weights: Optional[Dict[str, float]] = None,
    length_range: Tuple[int, int] = (90, 130),
    use_humanness_proxy: bool = False,
) -> RewardComponents:
    weights = weights or {}
    motif_weight = float(weights.get("motif", 1.0))
    noncanonical_weight = float(weights.get("noncanonical", 1.0))
    uncertainty_weight = float(weights.get("uncertainty", 1.0))
    humanness_weight = float(weights.get("humanness", 0.5))

    motif_penalty, motif_meta = motif_liability_penalty(sequence)
    noncanonical = noncanonical_residue_penalty(sequence)
    uncertainty = uncertainty_penalty(sequence, model=model, device=device, mc_samples=mc_samples)
    humanness = humanness_proxy_penalty(sequence, length_range=length_range) if use_humanness_proxy else 0.0

    total_penalty = (
        motif_weight * motif_penalty
        + noncanonical_weight * noncanonical
        + uncertainty_weight * uncertainty
        + humanness_weight * humanness
    )
    reward = -float(total_penalty)
    metadata = {
        **motif_meta,
        "noncanonical_penalty": float(noncanonical),
        "uncertainty_penalty": float(uncertainty),
        "humanness_penalty": float(humanness),
    }
    return RewardComponents(
        motif_penalty=float(motif_penalty),
        noncanonical_penalty=float(noncanonical),
        uncertainty_penalty=float(uncertainty),
        humanness_penalty=float(humanness),
        total_reward=reward,
        metadata=metadata,
    )


def build_rewards(
    sequences: Iterable[str],
    *,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    mc_samples: int = 0,
    weights: Optional[Dict[str, float]] = None,
    length_range: Tuple[int, int] = (90, 130),
    use_humanness_proxy: bool = False,
) -> Sequence[RewardComponents]:
    return [
        build_reward(
            seq,
            model=model,
            device=device,
            mc_samples=mc_samples,
            weights=weights,
            length_range=length_range,
            use_humanness_proxy=use_humanness_proxy,
        )
        for seq in sequences
    ]


__all__ = [
    "RewardComponents",
    "build_reward",
    "build_rewards",
    "motif_liability_penalty",
    "noncanonical_residue_penalty",
    "uncertainty_penalty",
    "humanness_proxy_penalty",
    "CANONICAL_AMINO_ACIDS",
]
