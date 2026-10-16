"""Tile pictures the agent has learned to name, so it can recognise things without a vision model.

Every time a right-click menu tells the agent what's on a tile, the tile's picture
is stored under that name. Later, a tile that looks the same gets the same name.
Nothing is shipped up front, so it works with any server's sprites: on a new
server the agent probes a lot at first and less as the library fills up.

Pictures are compared as small, brightness-normalised thumbnails, which copes with
lighting changes but not with rotation or animation; those just get learned as
extra pictures for the same name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .geometry import Geometry

THUMB = 12              # thumbnail side in pixels
PER_LABEL = 8           # pictures kept per name


def thumbnail(crop: np.ndarray) -> np.ndarray | None:
    """A tile crop as a unit-length, mean-removed vector (THUMB x THUMB x RGB)."""
    if crop.ndim != 3 or crop.shape[0] < THUMB or crop.shape[1] < THUMB:
        return None
    h, w, _ = crop.shape
    ys = np.linspace(0, h, THUMB + 1).astype(int)
    xs = np.linspace(0, w, THUMB + 1).astype(int)
    small = np.empty((THUMB, THUMB, 3), dtype=np.float32)
    for i in range(THUMB):
        for j in range(THUMB):
            small[i, j] = crop[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].reshape(-1, 3).mean(axis=0)
    v = small.reshape(-1)
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    if n < 1e-3:  # a flat colour: nothing to recognise
        return None
    return v / n


def tile_crop(frame: np.ndarray, geo: Geometry, dx: float, dy: float) -> np.ndarray:
    """The pixels of the tile at (dx, dy) from the player, without its blurry border."""
    t = geo.tile_px
    pad = max(1, int(t * 0.12))
    cx, cy = geo.tile_to_px(dx, dy)
    x0, y0 = int(cx - t / 2) + pad, int(cy - t / 2) + pad
    x1, y1 = int(cx + t / 2) - pad, int(cy + t / 2) - pad
    h, w, _ = frame.shape
    return frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]


def clean_label(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,:;|").lower()


class TileLibrary:
    def __init__(self, path: Path | None = None, threshold: float = 0.95) -> None:
        self.path = path
        self.threshold = threshold
        self.labels: list[str] = []
        self.vecs = np.zeros((0, THUMB * THUMB * 3), dtype=np.float32)
        self.dirty = False
        if path is not None and path.with_suffix(".npz").exists():
            self.load()

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def names(self) -> list[str]:
        return sorted(set(self.labels))

    def learn(self, crop: np.ndarray, label: str) -> bool:
        vec = thumbnail(crop)
        label = clean_label(label)
        if vec is None or not label:
            return False
        if len(self.labels):
            sims = self.vecs @ vec
            same = [i for i, lb in enumerate(self.labels) if lb == label]
            if same and float(sims[same].max()) > 0.97:
                return False  # already have a picture just like it
            # A near-identical picture under another name was a misread; the newer probe wins.
            wrong = [i for i, lb in enumerate(self.labels) if lb != label and sims[i] > 0.97]
            if wrong:
                self._drop(wrong)
            same = [i for i, lb in enumerate(self.labels) if lb == label]
            if len(same) >= PER_LABEL:
                self._drop(same[:1])
        self.labels.append(label)
        self.vecs = np.vstack([self.vecs, vec[None, :]])
        self.dirty = True
        return True

    def _drop(self, idx: list[int]) -> None:
        keep = [i for i in range(len(self.labels)) if i not in set(idx)]
        self.labels = [self.labels[i] for i in keep]
        self.vecs = self.vecs[keep]

    def match(self, crop: np.ndarray) -> tuple[str, float] | None:
        """Best-matching name and its similarity (0-1), or None below the threshold."""
        if not len(self.labels):
            return None
        vec = thumbnail(crop)
        if vec is None:
            return None
        sims = self.vecs @ vec
        i = int(np.argmax(sims))
        score = float(sims[i])
        return (self.labels[i], score) if score >= self.threshold else None

    # ---- persistence -----------------------------------------------------

    def save(self) -> None:
        if self.path is None or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path.with_suffix(".npz"), vecs=self.vecs)
        self.path.with_suffix(".json").write_text(json.dumps({"labels": self.labels}, indent=0))
        self.dirty = False

    def load(self) -> None:
        npz = self.path.with_suffix(".npz")
        meta = self.path.with_suffix(".json")
        try:
            vecs = np.load(npz)["vecs"].astype(np.float32)
            labels = json.loads(meta.read_text())["labels"]
        except (OSError, KeyError, ValueError):
            return
        if len(labels) == len(vecs) and vecs.shape[1:] == (THUMB * THUMB * 3,):
            self.labels, self.vecs = list(labels), vecs
