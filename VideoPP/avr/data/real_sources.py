"""Iterators over normal clips from UCSD Ped2 and CUHK Avenue training splits."""
from __future__ import annotations

import os
from typing import Iterator, Optional, Tuple

import numpy as np

NamedClip = Tuple[str, np.ndarray]


def _find_dir(root: str, name: str) -> str:
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == name:
            return dirpath
        for d in dirnames:
            if d == name:
                return os.path.join(dirpath, d)
    raise FileNotFoundError(f"'{name}' not found under {root}")


def _resize(clip: np.ndarray, size: int) -> np.ndarray:
    import cv2
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
                     for f in clip], axis=0)


def _chunks(frames: np.ndarray, T: int, stride: int) -> Iterator[np.ndarray]:
    for s in range(0, len(frames) - T + 1, stride):
        yield frames[s:s + T]


def ped2_normal_clips(raw_root: str, T: int = 16, size: int = 256,
                      stride: Optional[int] = None,
                      limit: Optional[int] = None) -> Iterator[NamedClip]:
    import imageio.v2 as imageio
    train = os.path.join(_find_dir(raw_root, "UCSDped2"), "Train")
    stride = stride or T
    folders = sorted(d for d in os.listdir(train)
                     if os.path.isdir(os.path.join(train, d)))
    n = 0
    for folder in folders:
        fdir = os.path.join(train, folder)
        files = sorted(f for f in os.listdir(fdir) if f.lower().endswith(".tif"))
        if len(files) < T:
            continue
        frames = np.stack([np.asarray(imageio.imread(os.path.join(fdir, f)))
                           for f in files], axis=0)
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        for i, clip in enumerate(_chunks(frames, T, stride)):
            yield f"ped2_{folder}_{i:02d}", _resize(clip, size)
            n += 1
            if limit and n >= limit:
                return


def avenue_normal_clips(raw_root: str, T: int = 16, size: int = 256,
                        stride: Optional[int] = None,
                        limit: Optional[int] = None) -> Iterator[NamedClip]:
    import imageio.v2 as imageio
    vdir = _find_dir(raw_root, "training_videos")
    stride = stride or T
    videos = sorted(f for f in os.listdir(vdir) if f.lower().endswith(".avi"))
    n = 0
    for vid in videos:
        reader = imageio.get_reader(os.path.join(vdir, vid))
        frames = np.stack([np.asarray(fr)[..., :3] for fr in reader], axis=0)
        reader.close()
        for i, clip in enumerate(_chunks(frames, T, stride)):
            yield f"avenue_{os.path.splitext(vid)[0]}_{i:02d}", _resize(clip, size)
            n += 1
            if limit and n >= limit:
                return


def shanghaitech_normal_clips(raw_root: str, T: int = 16, size: int = 256,
                              stride: Optional[int] = None,
                              limit: Optional[int] = None) -> Iterator[NamedClip]:
    import imageio.v2 as imageio
    frames_dir = _find_dir(raw_root, "frames")
    folders = sorted(d for d in os.listdir(frames_dir)
                     if os.path.isdir(os.path.join(frames_dir, d)))
    n = 0
    for folder in folders:
        fdir = os.path.join(frames_dir, folder)
        files = sorted(f for f in os.listdir(fdir) if f.lower().endswith(".jpg"))
        if len(files) < T:
            continue
        clip = np.stack([np.asarray(imageio.imread(os.path.join(fdir, files[i])))[..., :3]
                         for i in range(T)], axis=0)
        yield f"sht_{folder}", _resize(clip, size)
        n += 1
        if limit and n >= limit:
            return


SOURCES = {"ped2": ped2_normal_clips, "avenue": avenue_normal_clips,
           "shanghaitech": shanghaitech_normal_clips}
