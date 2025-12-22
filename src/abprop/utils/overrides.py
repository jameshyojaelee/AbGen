"""Helpers for parsing and applying CLI config overrides."""

from __future__ import annotations

import shlex
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import yaml


class ConfigOverrideError(ValueError):
    """Raised when config overrides cannot be parsed or applied."""


OverridePath = List[str]
ParsedOverride = Tuple[OverridePath, Any]


def parse_config_overrides(overrides: str | Iterable[str] | None) -> List[ParsedOverride]:
    """Parse space-separated key=value overrides into path/value pairs.

    Supports nested keys via dot notation (e.g., task_weights.mlm=0.5).
    """
    if not overrides:
        return []

    tokens: List[str] = []
    if isinstance(overrides, str):
        tokens.extend(shlex.split(overrides))
    else:
        for item in overrides:
            tokens.extend(shlex.split(str(item)))

    parsed: List[ParsedOverride] = []
    for token in tokens:
        if "=" not in token:
            raise ConfigOverrideError(f"Override '{token}' must be in key=value form.")
        key, raw_value = token.split("=", 1)
        if not key:
            raise ConfigOverrideError("Override key cannot be empty.")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigOverrideError(f"Invalid override value for '{key}': {exc}") from exc
        parsed.append((key.split("."), value))

    return parsed


def apply_config_overrides(
    targets: Mapping[str, Dict[str, Any]],
    overrides: Iterable[ParsedOverride],
    *,
    allow_unknown: bool = False,
    default_namespace: str = "model",
) -> None:
    """Apply parsed overrides to the provided config dictionaries.

    If the first path element matches a namespace in targets, it is used
    to select the target config. Otherwise the default namespace is used.
    """
    for path, value in overrides:
        if not path:
            raise ConfigOverrideError("Override path cannot be empty.")

        namespace = default_namespace
        key_path = path
        if path[0] in targets:
            namespace = path[0]
            key_path = path[1:]

        if not key_path:
            raise ConfigOverrideError(
                f"Override '{'.'.join(path)}' must include a key after the namespace."
            )

        if namespace not in targets:
            raise ConfigOverrideError(
                f"Unknown override namespace '{namespace}'. Valid: {', '.join(targets.keys())}"
            )

        _apply_to_dict(targets[namespace], key_path, value, allow_unknown)


def _apply_to_dict(
    target: Dict[str, Any],
    path: OverridePath,
    value: Any,
    allow_unknown: bool,
) -> None:
    cursor: Dict[str, Any] = target
    for key in path[:-1]:
        if key not in cursor:
            if not allow_unknown:
                raise ConfigOverrideError(
                    f"Unknown override key '{'.'.join(path)}' (missing '{key}')."
                )
            cursor[key] = {}
        if not isinstance(cursor[key], dict):
            if not allow_unknown:
                raise ConfigOverrideError(
                    f"Cannot set nested override '{'.'.join(path)}'; '{key}' is not a mapping."
                )
            cursor[key] = {}
        cursor = cursor[key]

    leaf = path[-1]
    if leaf not in cursor and not allow_unknown:
        raise ConfigOverrideError(f"Unknown override key '{'.'.join(path)}'.")
    cursor[leaf] = value


__all__ = ["ConfigOverrideError", "parse_config_overrides", "apply_config_overrides"]
