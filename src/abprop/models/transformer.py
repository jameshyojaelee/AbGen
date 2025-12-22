"""Baseline Transformer model with auxiliary heads for AbProp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from abprop.tokenizers import AMINO_ACIDS, SPECIAL_TOKENS
from abprop.utils.liabilities import CANONICAL_LIABILITY_KEYS
from abprop.eval.metrics import classification_summary, regression_summary


@dataclass
class TransformerConfig:
    vocab_size: int = len(SPECIAL_TOKENS) + len(AMINO_ACIDS)
    d_model: int = 384
    nhead: int = 6
    num_layers: int = 3
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_position_embeddings: int = 1024
    liability_keys: Tuple[str, ...] = CANONICAL_LIABILITY_KEYS
    mlm_weight: float = 1.0
    cls_weight: float = 1.0
    reg_weight: float = 1.0
    # Modern Architecture Toggles
    use_rope: bool = True
    norm_type: str = "rmsnorm"  # "layernorm" or "rmsnorm"
    activation: str = "swiglu"  # "relu", "gelu", "swiglu"


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


def _get_norm_fn(name: str, d_model: int, eps: float = 1e-5) -> nn.Module:
    if name == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    if name == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    raise ValueError(f"Unsupported norm '{name}'.")


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class TransformerEncoderLayerWithAttention(nn.Module):
    """Modernized Transformer encoder layer: Pre-Norm, RoPE support, SwiGLU."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        activation: str = "relu",
        norm_type: str = "layernorm",
        layer_norm_eps: float = 1e-5,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward
        self.activation_name = activation
        if activation == "swiglu":
            # For SwiGLU, we need (gate + value), so separate project or double width?
            # Standard practice: W1 (gate), W3 (val) -> mul -> W2 (out).
            # We will use "dim_feedforward" as the hidden size.
            # So, linear1 maps d_model -> 2 * dim_feedforward
            # linear2 maps dim_feedforward -> d_model
            self.linear1 = nn.Linear(d_model, 2 * dim_feedforward, bias=False)
            self.act_fn = SwiGLU()
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            if activation == "relu":
                self.act_fn = nn.ReLU()
            elif activation == "gelu":
                self.act_fn = nn.GELU()
            else:
                raise ValueError(f"Unsupported activation {activation}")

        self.norm1 = _get_norm_fn(norm_type, d_model, layer_norm_eps)
        self.norm2 = _get_norm_fn(norm_type, d_model, layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # PRE-NORM Architecture: x = x + attn(norm(x))
        residual = src
        x = self.norm1(src)

        # Apply RoPE if present
        # nn.MultiheadAttention doesn't natively support RoPE callbacks.
        # We need to manually handle query/key projection if we want strictly correct RoPE,
        # OR we hack it by applying RoPE to 'x' before attention if x is used as Q,K.
        # However, MultiheadAttention does internal projections Q = x W_q.
        # Applying RoPE requires access to Q, K *after* projection but *before* attention.
        # Standard nn.MultiheadAttention makes this hard.
        #
        # OPTION: Re-implement MHA or use Scaled Dot Product Attention (SDPA) manually.
        # Given this is a demo of "novelty", implementing a clean manual SDPA with RoPE is better
        # than hacking nn.MHA. `F.scaled_dot_product_attention` is efficient.
        # Let's do a Manual MHA here for full control.
        
        aln_output, attn_weights = self._mha_forward(
            x, x, x, 
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            return_attentions=return_attentions
        )
        
        src = residual + self.dropout1(aln_output)
        
        # Feed Forward
        residual = src
        x = self.norm2(src)
        
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        src = residual + self.dropout2(x)

        return src, attn_weights

    def _mha_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        return_attentions: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # We will reuse the self.self_attn weights for convenience, or better,
        # just use their parameters if we want to be compatible with loading weights?
        # NOTE: Loading old weights into this new architecture will break anyway because
        # of Pre-Norm/different shapes. So we can just act freely.
        # To keep it simple, I'll allow self.self_attn to handle the projections if possible
        # but since I need to inject RoPE, I need to extract Q, K, V.
        #
        # Workaround: Use self.self_attn.in_proj_weight/bias manually.
        
        B, L, E = query.shape
        # standard MHA packs q,k,v in one weight if possible, or separate.
        # Let's assume standard usage of the existing module to keep code size low is tricky with RoPE.
        # I'll just rely on `F.multi_head_attention_forward`? No, too complex.
        #
        # Simplified Manual Attn using the linear layers from self.self_attn
        # But wait, self.self_attn is an opaque bucket.
        # Let's just define our own projections if we are replacing the architecture.
        # To be safe and cleaner: WE IGNORE self.self_attn and define new layers?
        # No, let's try to grab weights from it to not break initialization logic if reused.
        
        qkv = self.self_attn.in_proj_weight
        bias = self.self_attn.in_proj_bias
        
        if qkv is not None:
             # Combined projection
            qkv_out = F.linear(query, qkv, bias)
            q, k, v = qkv_out.chunk(3, dim=-1)
        else:
             # Separated (if bias/kdim/vdim were different, but defaults are same)
             # Fallback to separate
             q = F.linear(query, self.self_attn.q_proj_weight, self.self_attn.q_proj_bias)
             k = F.linear(key, self.self_attn.k_proj_weight, self.self_attn.k_proj_bias)
             v = F.linear(value, self.self_attn.v_proj_weight, self.self_attn.v_proj_bias)

        # Reshape to [B, H, L, D]
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)
        
        # Attention
        # PyTorch SDPA prefers [B, H, L, D]
        
        # Mask handling is slightly complex for SDPA with different versions.
        # If attn_mask is meant for bias-add (float) or bool mask.
        dropout_p = self.dropout.p if self.training else 0.0
        
        # Ensure masks are correct format
        # src_key_padding_mask is [B, L], True where padding.
        # attn_mask might be [L, L] or [B*num_heads, L, L].
        # Is causal mask used? BERT doesn't use causal usually.
        
        # For simplicity, combine masks.
        is_causal = False
        
        # SDPA handles key_padding_mask automatically in newer torch?
        # Check signature: query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False.
        # It does NOT take key_padding_mask separate. We must merge.
        
        combined_mask = attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] (True=pad). Expand to [B, 1, 1, S]
            pad_mask = key_padding_mask.view(B, 1, 1, L).expand(-1, self.nhead, -1, -1)
            # attn_mask usually [L, L] for seq mask.
            # We need a full bias mask.
            if combined_mask is None:
                combined_mask = torch.zeros((B, self.nhead, L, L), device=query.device, dtype=query.dtype)
                combined_mask.masked_fill_(pad_mask, float("-inf"))
            else:
                # Merge logic (assuming additive mask)
                combined_mask = combined_mask + pad_mask.float() * -1e9

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=combined_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        
        if return_attentions:
             # Re-compute weights for viz if requested (SDPA doesn't return them)
             # This is expensive but necessary for viz.
             # Or we fallback to manual logic.
             # Given 'viz' is a feature (AbProp introspect), we should probably support it.
             # But F.sdpa is fast.
             # Let's do standard computation if return_attentions is True.
             scale = self.head_dim ** -0.5
             attn_logits = (q @ k.transpose(-2, -1)) * scale
             if combined_mask is not None:
                 attn_logits = attn_logits + combined_mask
             attn_weights = F.softmax(attn_logits, dim=-1)
             out = attn_weights @ v
        else:
             attn_weights = None

        out = out.transpose(1, 2).contiguous().view(B, L, E)
        out = self.self_attn.out_proj(out)
        
        return out, attn_weights


class SmallEncoder(nn.Module):
    """Lightweight Transformer encoder with modernization support."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.scale = config.d_model**-0.5
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # RoPE replaces absolute embeddings
        self.use_rope = getattr(config, "use_rope", False)
        if not self.use_rope:
            self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        else:
            self.position_embedding = None
            head_dim = config.d_model // config.nhead
            self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len=config.max_position_embeddings)

        norm_type = getattr(config, "norm_type", "layernorm")
        activation = getattr(config, "activation", "relu")

        self.layers = nn.ModuleList(
            TransformerEncoderLayerWithAttention(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=activation,
                norm_type=norm_type,
                rotary_emb=self.rotary_emb if self.use_rope else None,
            )
            for _ in range(config.num_layers)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm = _get_norm_fn(norm_type, config.d_model) # Final Norm for Pre-Norm architecture
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, Optional[List[torch.Tensor]]], torch.Tensor]:
        attention_mask = attention_mask.bool()
        
        # Embeddings
        hidden = self.token_embedding(input_ids) * self.scale
        
        if not self.use_rope:
            positions = torch.arange(
                input_ids.size(1), device=input_ids.device, dtype=torch.long
            ).unsqueeze(0)
            hidden = hidden + self.position_embedding(positions)
        
        hidden = self.dropout(hidden)
        key_padding_mask = ~attention_mask
        
        attentions: Optional[List[torch.Tensor]] = [] if return_attentions else None
        
        for layer in self.layers:
            hidden, attn_weights = layer(
                hidden,
                src_key_padding_mask=key_padding_mask,
                return_attentions=return_attentions,
            )
            if return_attentions and attn_weights is not None:
                attentions.append(attn_weights)
                
        encoded = self.norm(hidden)
        
        if return_attentions:
            return encoded, attentions
        return encoded


class MLMHead(nn.Module):
    """Masked language modeling head with tied embeddings."""

    def __init__(self, hidden_size: int, vocab_size: int, embedding_weight: nn.Parameter) -> None:
        super().__init__()
        # Ensure we use LayerNorm for head anyway, or match norm_type? 
        # Usually head has its own LayerNorm.
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.decoder.weight = embedding_weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states) + self.bias
        return logits


class SeqClassifierHead(nn.Module):
    """Token-level classifier (e.g., framework vs CDR)."""

    def __init__(self, hidden_size: int, dropout: float, num_labels: int = 2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(hidden_states)
        return self.classifier(x)


class LiabilityRegHead(nn.Module):
    """Sequence-level liability regression head."""

    def __init__(self, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        return self.regressor(pooled)


class AbPropModel(nn.Module):
    """Wrapper model combining encoder with MLM, classification, and regression heads."""

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self._mc_dropout_enabled = False
        self.config = config or TransformerConfig()
        self.encoder = SmallEncoder(self.config)
        self.mlm_head = MLMHead(
            hidden_size=self.config.d_model,
            vocab_size=self.config.vocab_size,
            embedding_weight=self.encoder.token_embedding.weight,
        )
        self.classifier = SeqClassifierHead(
            hidden_size=self.config.d_model,
            dropout=self.config.dropout,
        )
        self.regressor = LiabilityRegHead(
            hidden_size=self.config.d_model,
            output_size=len(self.config.liability_keys),
            dropout=self.config.dropout,
        )

        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.reg_loss_fn = nn.MSELoss()

        # Cache dropout modules for quick toggling during MC dropout inference.
        self._dropout_modules: Tuple[nn.Dropout, ...] = tuple(
            module for module in self.modules() if isinstance(module, nn.Dropout)
        )

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mlm_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]] = None,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor] = None,
        tasks: Optional[Sequence[str]] = None,
        return_attentions: bool = False,
    ) -> Dict[str, object]:
        tasks = tuple(tasks or ("mlm", "cls", "reg"))
        attention_mask = attention_mask.bool()
        encoder_output = self.encoder(
            input_ids,
            attention_mask,
            return_attentions=return_attentions,
        )
        if return_attentions:
            hidden_states, attentions = encoder_output  # type: ignore[misc]
        else:
            hidden_states = encoder_output  # type: ignore[assignment]
            attentions = None

        outputs: Dict[str, object] = {"hidden_states": hidden_states}
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)
        loss_tracked = False

        if attentions is not None:
            outputs["attentions"] = attentions

        if "mlm" in tasks:
            mlm_logits = self.mlm_head(hidden_states)
            outputs["mlm_logits"] = mlm_logits
            if mlm_labels is not None:
                mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
                losses["mlm_loss"] = mlm_loss
                metrics["mlm_perplexity"] = torch.exp(mlm_loss.detach())
                total_loss = total_loss + self.config.mlm_weight * mlm_loss
                loss_tracked = True

        if "cls" in tasks:
            token_logits = self.classifier(hidden_states)
            outputs["cls_logits"] = token_logits
            prepared_labels = self._prepare_token_labels(token_labels, attention_mask, device=input_ids.device)
            if prepared_labels is not None:
                cls_loss = self.cls_loss_fn(
                    token_logits.view(-1, token_logits.size(-1)),
                    prepared_labels.view(-1),
                )
                losses["cls_loss"] = cls_loss
                total_loss = total_loss + self.config.cls_weight * cls_loss
                loss_tracked = True
                valid_mask = prepared_labels != -100
                if valid_mask.any():
                    valid_logits = token_logits[valid_mask]
                    valid_labels = prepared_labels[valid_mask]
                    predictions = valid_logits.argmax(dim=-1)
                    accuracy = (predictions == valid_labels).float().mean()
                    metrics["cls_accuracy"] = accuracy.detach()
                    summary = classification_summary(
                        int(((predictions == 1) & (valid_labels == 1)).sum()),
                        int(((predictions == 1) & (valid_labels == 0)).sum()),
                        int(((predictions == 0) & (valid_labels == 0)).sum()),
                        int(((predictions == 0) & (valid_labels == 1)).sum()),
                    )
                    metrics["cls_f1"] = torch.tensor(summary["f1"], device=input_ids.device)

        if "reg" in tasks:
            reg_targets = self._prepare_regression_targets(
                liability_targets,
                batch_size=input_ids.size(0),
                device=input_ids.device,
            )
            regression_logits = self.regressor(hidden_states, attention_mask)
            outputs["regression"] = regression_logits
            if reg_targets is not None:
                reg_loss = self.reg_loss_fn(regression_logits, reg_targets)
                losses["reg_loss"] = reg_loss
                total_loss = total_loss + self.config.reg_weight * reg_loss
                loss_tracked = True
                preds_det = regression_logits.detach()
                targets_det = reg_targets.detach()
                reg_summary = regression_summary(preds_det, targets_det)
                metrics["reg_mse"] = torch.tensor(reg_summary["mse"], device=input_ids.device)
                metrics["reg_r2"] = torch.tensor(reg_summary["r2"], device=input_ids.device)
                metrics["reg_spearman"] = torch.tensor(reg_summary["spearman"], device=input_ids.device)

        if loss_tracked:
            outputs["loss"] = total_loss
        else:
            outputs["loss"] = None

        outputs["losses"] = losses
        outputs["metrics"] = metrics
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mlm_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]] = None,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor] = None,
        tasks: Optional[Sequence[str]] = None,
        return_attentions: bool = False,
    ) -> Dict[str, object]:
        return self._forward_impl(
            input_ids,
            attention_mask,
            mlm_labels=mlm_labels,
            token_labels=token_labels,
            liability_targets=liability_targets,
            tasks=tasks,
            return_attentions=return_attentions,
        )

    def stochastic_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mc_samples: int,
        mlm_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]] = None,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor] = None,
        tasks: Optional[Sequence[str]] = None,
        enable_dropout: bool = True,
        no_grad: bool = True,
        return_attentions: bool = False,
    ) -> List[Dict[str, object]]:
        if mc_samples <= 0:
            raise ValueError("mc_samples must be a positive integer.")

        original_training = self.training
        dropout_states = self._snapshot_dropout_states()
        previous_flag = self._mc_dropout_enabled

        self.eval()
        if enable_dropout:
            self.set_inference_dropout(True)

        outputs: List[Dict[str, object]] = []
        grad_ctx = torch.no_grad if no_grad else torch.enable_grad  # type: ignore[assignment]
        with grad_ctx():
            for _ in range(mc_samples):
                outputs.append(
                    self._forward_impl(
                        input_ids,
                        attention_mask,
                        mlm_labels=mlm_labels,
                        token_labels=token_labels,
                        liability_targets=liability_targets,
                        tasks=tasks,
                        return_attentions=return_attentions,
                    )
                )

        if enable_dropout:
            self._restore_dropout_states(dropout_states)
            self._mc_dropout_enabled = previous_flag

        if original_training:
            self.train()

        return outputs

    def set_inference_dropout(self, enabled: bool) -> None:
        """Toggle dropout layers while keeping the rest of the model in eval mode."""
        self._mc_dropout_enabled = enabled
        for module in self._dropout_modules:
            if self.training and not enabled:
                module.train(True)
            else:
                module.train(enabled)

    def inference_dropout_enabled(self) -> bool:
        return self._mc_dropout_enabled

    def _snapshot_dropout_states(self) -> Tuple[bool, ...]:
        return tuple(module.training for module in self._dropout_modules)

    def _restore_dropout_states(self, states: Sequence[bool]) -> None:
        for module, state in zip(self._dropout_modules, states):
            module.train(state)

    def _prepare_token_labels(
        self,
        token_labels: Optional[torch.Tensor | Sequence[Sequence[int]]],
        attention_mask: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if token_labels is None:
            return None
        if isinstance(token_labels, torch.Tensor):
            return token_labels.to(device=device, dtype=torch.long)

        batch_size, seq_len = attention_mask.shape
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        for idx, sequence_labels in enumerate(token_labels):
            if sequence_labels is None:
                continue
            if isinstance(sequence_labels, torch.Tensor):
                seq_tensor = sequence_labels.to(device=device, dtype=torch.long)
            else:
                seq_tensor = torch.tensor(sequence_labels, dtype=torch.long, device=device)
            if seq_tensor.numel() == 0:
                continue
            max_copy = min(seq_tensor.size(0), seq_len - 2)
            labels[idx, 1 : 1 + max_copy] = seq_tensor[:max_copy]
        return labels

    def _prepare_regression_targets(
        self,
        liability_targets: Optional[Sequence[Dict[str, float]] | torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if liability_targets is None:
            return None
        if isinstance(liability_targets, torch.Tensor):
            return liability_targets.to(device=device, dtype=torch.float32)

        target_tensor = torch.zeros(
            (batch_size, len(self.config.liability_keys)),
            dtype=torch.float32,
            device=device,
        )
        for idx, entry in enumerate(liability_targets):
            if entry is None:
                continue
            for key_idx, key in enumerate(self.config.liability_keys):
                target_tensor[idx, key_idx] = float(entry.get(key, 0.0))
        return target_tensor


__all__ = [
    "TransformerConfig",
    "SmallEncoder",
    "MLMHead",
    "SeqClassifierHead",
    "LiabilityRegHead",
    "AbPropModel",
    "RotaryEmbedding",
    "RMSNorm",
]
