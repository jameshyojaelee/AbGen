"""
Pure PyTorch implementation of Mamba (Selective State Space Model) components.

This module provides a portable, CUDA-free implementation of the S6 block described in
'Mamba: Linear-Time Sequence Modeling with Selective State Spaces' (Gu & Dao, 2023).
It is designed for rapid prototyping and architectural exploration in AbProp.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from abprop.models.transformer import RMSNorm  # Reuse existing components


@dataclass
class MambaConfig:
    d_model: int = 384
    n_layers: int = 3
    vocab_size: int = 25
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    ssm_dt_rank: str = "auto"
    ssm_conv_bias: bool = True
    ssm_bias: bool = False
    dropout: float = 0.1
    use_rope: bool = False  # Mamba typically doesn't need PE, but we can support it.
    norm_type: str = "rmsnorm"


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, d_model)
        """
        batch, seq_len, d = x.shape
        
        # 1. Project input
        x_and_res = self.in_proj(x)  # (B, L, 2*ED)
        (x_branch, res_branch) = x_and_res.split(
            [self.d_inner, self.d_inner], dim=-1
        )
        
        # 2. Convolution (Causal)
        x_branch = x_branch.transpose(1, 2)  # (B, ED, L)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]  # Causal crop
        x_branch = x_branch.transpose(1, 2)  # (B, L, ED)
        x_branch = self.silu(x_branch)
        
        # 3. State Space Model
        y_ssm = self.ssm(x_branch)
        
        # 4. Gating
        y = y_ssm * self.silu(res_branch)
        
        # 5. Output Project
        return self.out_proj(y)

    def ssm(self, u: torch.Tensor) -> torch.Tensor:
        """
        Runs the SSM.
        u: (B, L, ED)
        Returns: (B, L, ED)
        """
        batch, seq_len, dim = u.shape
        
        # Compute delta, B, C dependent on u
        # x_proj maps u to (dt, B, C)
        # B, C are (Batch, L, d_state)
        # dt is (Batch, L, dt_rank)
        
        dt_B_C = self.x_proj(u)
        
        dt, B, C = torch.split(
            dt_B_C, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        dt = F.softplus(self.dt_proj(dt))  # (Batch, L, ED)
        
        # Discretization
        # A is (ED, d_state)
        A = -torch.exp(self.A_log.float())  # Force A to be negative
        
        # This is the sequential scan (slow but correct reference)
        # y_t = SSM(u_t)
        # For efficiency in python, we can iterate over L.
        # But this is very slow.
        # 
        # For this prototype, to keep it fast enough for small batches/demos
        # while remaining readable, we stick to sequential.
        # "Novelty" comes from the architecture, not necessarily a CUDA kernel here.
        
        # Dimensions:
        # h: (batch, dim, d_state)
        h = torch.zeros(batch, dim, self.d_state, device=u.device)
        ys = []
        
        # A: (dim, d_state)
        # B: (batch, L, d_state)
        # C: (batch, L, d_state)
        # dt: (batch, L, dim)
        
        # Precompute discretized matrices?
        # dA = exp(dt * A) -> (Batch, L, dim, d_state) -- HIGH MEMORY
        # We process step by step to save memory, or precompute if L is small.
        # Let's do step-by-step.
        
        for i in range(seq_len):
            # dt[i]: (Batch, dim)
            # A: (dim, d_state)
            # dA = exp(dt[i, :, None] * A[None, :, :]) -> (Batch, dim, d_state)
            
            dt_i = dt[:, i, :, None] # (B, D, 1)
            dA = torch.exp(dt_i * A) 
            
            # dB = dt * B
            # B[i]: (B, d_state)
            # dB = dt[i] * B[i] -> (B, D, 1) * (B, 1, S) -> (B, D, S)
            B_i = B[:, i, None, :] # (B, 1, S)
            dB = dt_i * B_i
            
            # u[i]: (B, D)
            u_i = u[:, i, :, None] # (B, D, 1)
            
            # h = dA * h + dB * u
            h = h * dA + dB * u_i
            
            # y = C * h + D * u
            # C[i]: (B, S)
            C_i = C[:, i, None, :] # (B, 1, S)
            y_i = torch.sum(h * C_i, dim=-1) # (B, D)
            
            y_i = y_i + self.D * u[:, i]
            ys.append(y_i)
            
        y = torch.stack(ys, dim=1)
        return y


class MambaEncoder(nn.Module):
    """Stack of Mamba blocks acting as an Encoder."""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=config.d_model,
                d_state=config.ssm_d_state,
                d_conv=config.ssm_d_conv,
                expand=config.ssm_expand,
                dt_rank=config.ssm_dt_rank,
                conv_bias=config.ssm_conv_bias,
                bias=config.ssm_bias
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm_f = RMSNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False # Mamba has no attention weights
    ) -> torch.Tensor | Tuple[torch.Tensor, None]:
        
        x = self.embedding(input_ids)
        
        # Mamba doesn't strictly need attention_mask for padding in the Conv/SSM 
        # in the same way Attn does (it just processes pad tokens as state updates).
        # But for correctness, we should mask the input x or output?
        # Usually we just let it run.
        
        for layer in self.layers:
            # Pre-Norm residual connection
            # Mamba architecture: x = x + Block(Norm(x))
            residual = x
            x = self.norm_f(x)
            x = layer(x)
            x = residual + x
            
        x = self.norm_f(x) # Final norm
        
        if return_attentions:
            return x, None
        return x
