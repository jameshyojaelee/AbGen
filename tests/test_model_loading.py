from __future__ import annotations

from pathlib import Path

import torch

from abprop.models import AbPropModel, TransformerConfig
from abprop.models.loading import load_model_from_checkpoint, resolve_model_config


def test_model_config_precedence_checkpoint_over_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "model.yaml"
    yaml_path.write_text("encoder_type: transformer\nd_model: 32\n", encoding="utf-8")

    ckpt_config = TransformerConfig(
        encoder_type="mamba",
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        max_position_embeddings=32,
    )
    model = AbPropModel(ckpt_config)
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {"model_state": model.state_dict(), "model_config": dict(ckpt_config.__dict__)},
        ckpt_path,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    resolved = resolve_model_config(yaml_path, state)
    assert resolved.encoder_type == "mamba"
    assert resolved.d_model == 16

    loaded_model, loaded_cfg, _ = load_model_from_checkpoint(ckpt_path, yaml_path, device="cpu")
    assert loaded_cfg.encoder_type == "mamba"
    assert loaded_model.config.encoder_type == "mamba"
