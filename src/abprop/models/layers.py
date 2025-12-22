"""Shared architectural components for AbProp models."""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.scale * x_normed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [batch, head, seq_len, head_dim]
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len, device=q.device, dtype=q.dtype)

        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        
        return (
            self._apply_rotary_pos_emb(q, cos, sin),
            self._apply_rotary_pos_emb(k, cos, sin),
        )

    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def get_norm_fn(name: str, d_model: int, eps: float = 1e-5) -> nn.Module:
    if name == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    if name == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    raise ValueError(f"Unsupported norm '{name}'.")
