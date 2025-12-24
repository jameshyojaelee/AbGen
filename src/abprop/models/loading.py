"""Shared helpers for loading AbProp models and configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from abprop.utils import extract_model_config, load_yaml_config

from .transformer import AbPropModel, TransformerConfig


def resolve_model_config(model_config: Path | str, checkpoint_state: Dict[str, Any]) -> TransformerConfig:
    cfg = load_yaml_config(Path(model_config))
    if isinstance(cfg, dict) and isinstance(cfg.get("model"), dict):
        cfg = cfg["model"]

    if cfg is None:
        cfg = {}

    checkpoint_cfg = extract_model_config(checkpoint_state)
    if checkpoint_cfg:
        cfg = {**cfg, **checkpoint_cfg}

    allowed = set(TransformerConfig.__dataclass_fields__.keys())
    cfg = {key: value for key, value in cfg.items() if key in allowed}
    return TransformerConfig(**cfg) if cfg else TransformerConfig()


def load_model_from_checkpoint(
    checkpoint: Path | str,
    model_config: Path | str,
    device: torch.device | str,
    *,
    state: Dict[str, Any] | None = None,
    strict: bool = False,
) -> Tuple[AbPropModel, TransformerConfig, Dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    state = state or torch.load(checkpoint_path, map_location="cpu")
    config = resolve_model_config(model_config, state)
    model = AbPropModel(config).to(device)
    model_state = state.get("model_state_dict", state.get("model_state", state))
    model.load_state_dict(model_state, strict=strict)
    model.eval()
    return model, config, state


__all__ = ["load_model_from_checkpoint", "resolve_model_config"]
