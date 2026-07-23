"""Build and load a benchmark of (anom, clean, mask) triples on disk."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..registry import ANOMALIES


@dataclass
class Clip:
    clip_id: str
    anom: np.ndarray
    clean: np.ndarray
    mask: np.ndarray
    meta: dict


DEFAULT_ANOMALIES = [
    {"name": "static_object"},
    {"name": "moving_object"},
    {"name": "local_appearance"},
]


def build_benchmark(root, clips, anomaly_types=None, seed=0) -> str:
    """Inject anomalies into (name, clean_clip) pairs and write triples to disk.

    `clips` is any iterable of (name, uint8[T,H,W,3]); anomaly types are assigned
    round-robin so every type is represented.
    """
    anomaly_types = anomaly_types or DEFAULT_ANOMALIES
    injectors = [(a["name"], ANOMALIES.build(a)) for a in anomaly_types]
    os.makedirs(root, exist_ok=True)
    manifest = []
    for idx, (name, clean) in enumerate(clips):
        a_name, injector = injectors[idx % len(injectors)]
        s = seed + idx
        anom, clean, mask = injector.inject(clean, seed=s)
        clip_id = f"{name}_{a_name}"
        d = os.path.join(root, clip_id)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "anom.npy"), anom)
        np.save(os.path.join(d, "clean.npy"), clean)
        np.save(os.path.join(d, "mask.npy"), mask)
        T, H, W = clean.shape[:3]
        meta = {"clip_id": clip_id, "anomaly_type": a_name, "seed": s,
                "T": T, "H": H, "W": W}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"clips": manifest}, f, indent=2)
    return root


def load_dataset(root: str) -> Iterator[Clip]:
    with open(os.path.join(root, "manifest.json")) as f:
        manifest = json.load(f)
    for meta in manifest["clips"]:
        d = os.path.join(root, meta["clip_id"])
        yield Clip(
            clip_id=meta["clip_id"],
            anom=np.load(os.path.join(d, "anom.npy")),
            clean=np.load(os.path.join(d, "clean.npy")),
            mask=np.load(os.path.join(d, "mask.npy")),
            meta=meta,
        )
