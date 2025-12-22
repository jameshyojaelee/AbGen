from __future__ import annotations

from pathlib import Path

import torch

from abprop.models import AbPropModel, TransformerConfig
from abprop.server.app import ModelWrapper


def test_modelwrapper_score_perplexity_runs(tmp_path: Path) -> None:
    config = TransformerConfig(
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        max_position_embeddings=32,
    )
    model = AbPropModel(config)

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {"model_state": model.state_dict(), "model_config": dict(config.__dict__)},
        ckpt_path,
    )

    cfg_path = tmp_path / "model.yaml"
    cfg_path.write_text("{}", encoding="utf-8")

    wrapper = ModelWrapper(ckpt_path, cfg_path, device="cpu")
    scores = wrapper.score_perplexity(["ACD", "WQX"])

    assert len(scores) == 2
