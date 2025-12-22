"""Checkpoint helpers for extracting stored configuration."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def extract_model_config(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return model configuration dict from a checkpoint state if present."""
    for key in ("model_config", "config", "transformer_config"):
        maybe_cfg = state.get(key)
        if isinstance(maybe_cfg, dict):
            return dict(maybe_cfg)
    return {}


__all__ = ["extract_model_config"]
