"""YAML config loading with shallow `base:` inheritance and deep merge."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load a config, resolving an optional `base:` path (relative to file)."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base_rel = cfg.pop("base", None)
    if base_rel:
        base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_rel))
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg
