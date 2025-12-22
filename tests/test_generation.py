from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for generation tests.")

from abprop.generation import decode_sequences, random_sequences, sample_mlm_edit
from abprop.models import AbPropModel, TransformerConfig
from abprop.tokenizers import AMINO_ACIDS, TOKEN_TO_ID


def _add_special_tokens(raw_ids: torch.Tensor) -> torch.Tensor:
    pad_id = TOKEN_TO_ID["<pad>"]
    bos_id = TOKEN_TO_ID["<bos>"]
    eos_id = TOKEN_TO_ID["<eos>"]
    batch_size, length = raw_ids.shape
    input_ids = torch.full((batch_size, length + 2), pad_id, dtype=torch.long)
    input_ids[:, 0] = bos_id
    input_ids[:, -1] = eos_id
    input_ids[:, 1:-1] = raw_ids
    return input_ids


def test_mlm_sampling_preserves_length_and_vocab() -> None:
    torch.manual_seed(0)
    config = TransformerConfig(
        encoder_type="transformer",
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_position_embeddings=32,
    )
    model = AbPropModel(config)
    model.eval()

    generator = torch.Generator().manual_seed(123)
    raw_ids = random_sequences(2, 6, generator=generator)
    input_ids = _add_special_tokens(raw_ids)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    sampled_ids = sample_mlm_edit(
        model,
        input_ids,
        attention_mask,
        steps=2,
        mask_rate=0.3,
        temperature=1.0,
        generator=generator,
    )
    sequences = decode_sequences(sampled_ids)

    assert all(len(seq) == 6 for seq in sequences)
    valid = set(AMINO_ACIDS)
    assert all(all(res in valid for res in seq) for seq in sequences)
