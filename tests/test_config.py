from pathlib import Path

import pytest

from src.utils.config import ConfigError, load_yaml_config, merge_config


def test_load_and_merge_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("project:\n  results_dir: old\nmock:\n  seed: 42\n", encoding="utf-8")
    loaded = load_yaml_config(path)
    merged = merge_config(loaded, {"project": {"results_dir": "new"}})
    assert merged["project"]["results_dir"] == "new"
    assert merged["mock"]["seed"] == 42
    assert loaded["project"]["results_dir"] == "old"


def test_config_root_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_yaml_config(path)
