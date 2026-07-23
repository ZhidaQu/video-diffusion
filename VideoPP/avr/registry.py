"""Name->class registries so components are selectable from config."""
from __future__ import annotations

from typing import Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._m: Dict[str, type] = {}

    def register(self, key: str) -> Callable[[type], type]:
        def deco(cls: type) -> type:
            if key in self._m:
                raise KeyError(f"'{key}' already registered in {self.kind}")
            self._m[key] = cls
            return cls
        return deco

    def get(self, key: str) -> type:
        if key not in self._m:
            raise KeyError(
                f"Unknown {self.kind} '{key}'. Available: {sorted(self._m)}"
            )
        return self._m[key]

    def build(self, cfg: dict):
        cfg = dict(cfg)
        name = cfg.pop("name")
        return self.get(name)(**cfg)

    def available(self):
        return sorted(self._m)


DETECTORS = Registry("detector")
INPAINTERS = Registry("inpainter")
SCORERS = Registry("scorer")
ANOMALIES = Registry("anomaly")
