"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file cannot be used."""


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping, returning an empty mapping for an empty file."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load configuration {config_path}: {exc}") from exc

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    return value


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into a copied base configuration."""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged
