# ppedcrf/utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import copy
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base and return a new dict.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    Example:
      set_by_dotted_key(cfg, "train.epochs", 10)
    """
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def maybe_override(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """
    Only apply override if value is not None.
    """
    if value is None:
        return
    set_by_dotted_key(cfg, dotted_key, value)