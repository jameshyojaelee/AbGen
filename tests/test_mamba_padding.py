from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for Mamba tests.")

from abprop.models import AbPropModel, TransformerConfig
from abprop.tokenizers import TOKEN_TO_ID, encode


def test_mamba_padding_invariance() -> None:
    torch.manual_seed(0)
    config = TransformerConfig(
        encoder_type="mamba",
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        ssm_d_state=8,
    )
    model = AbPropModel(config)
    model.eval()

    seq = "ACDEFG"
    ids = encode(seq, add_special=True)
    pad_id = TOKEN_TO_ID["<pad>"]

    input_ids_short = ids.unsqueeze(0)
    attn_short = torch.ones_like(input_ids_short, dtype=torch.bool)

    pad_len = 4
    padded_ids = torch.cat(
        [ids, torch.full((pad_len,), pad_id, dtype=torch.long)], dim=0
    )
    input_ids_long = padded_ids.unsqueeze(0)
    attn_long = torch.cat(
        [torch.ones(ids.size(0), dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)],
        dim=0,
    ).unsqueeze(0)

    with torch.no_grad():
        out_short = model.encoder(input_ids_short, attn_short)
        out_long = model.encoder(input_ids_long, attn_long)

    assert torch.allclose(out_short, out_long[:, : ids.size(0), :], atol=1e-5)
