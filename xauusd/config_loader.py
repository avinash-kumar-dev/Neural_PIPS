import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
