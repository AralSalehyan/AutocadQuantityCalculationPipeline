from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_pipeline_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    config = load_config(DEFAULT_CONFIG_PATH)
    for extra_path in (Path("configs/detectors.yaml"), Path("configs/segmenters.yaml")):
        if extra_path.exists():
            config = merge_dicts(config, load_config(extra_path))
    if config_path != DEFAULT_CONFIG_PATH:
        config = merge_dicts(config, load_config(config_path))
    return config


def merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def deep_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
