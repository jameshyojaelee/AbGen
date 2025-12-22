from __future__ import annotations

import pytest

from abprop.utils.overrides import (
    ConfigOverrideError,
    apply_config_overrides,
    parse_config_overrides,
)


def test_parse_and_apply_overrides_flat_and_nested() -> None:
    cfg = {
        "encoder_type": "transformer",
        "nhead": 6,
        "task_weights": {"mlm": 1.0},
    }

    overrides = parse_config_overrides(
        "encoder_type=mamba task_weights.mlm=0.5 model.nhead=8"
    )
    apply_config_overrides({"model": cfg}, overrides)

    assert cfg["encoder_type"] == "mamba"
    assert cfg["nhead"] == 8
    assert cfg["task_weights"]["mlm"] == 0.5


def test_unknown_overrides_raise_by_default() -> None:
    cfg = {"encoder_type": "transformer"}
    overrides = parse_config_overrides("unknown_key=1")

    with pytest.raises(ConfigOverrideError):
        apply_config_overrides({"model": cfg}, overrides)


def test_allow_unknown_overrides_adds_keys() -> None:
    cfg = {"encoder_type": "transformer"}
    overrides = parse_config_overrides("new_key=1")
    apply_config_overrides({"model": cfg}, overrides, allow_unknown=True)

    assert cfg["new_key"] == 1
