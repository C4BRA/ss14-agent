"""Cheap per-tile analysis of the game viewport: space detection and object salience.

This is deliberately simple. It finds tiles that look like empty space (and so
can't be walked on) and tiles that contain "something interesting" compared to
the surrounding floor. A trained detector can replace `find_markers` later
without touching the rest of the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Geometry


@dataclass
class TileStats:
    dx: int
    dy: int
    mean: np.ndarray      # mean RGB
    dark_frac: float      # share of near-black pixels
    detail: float         # mean gradient magnitude


@dataclass
class Marker:
    id: int
    dx: int               # tile offset from the player at capture time
    dy: int
    px: float             # tile centre in capture pixels
    py: float
    score: float
    world: tuple[float, float] = (0.0, 0.0)   # tile position in odometry coordinates

    @property
    def dist(self) -> float:
        return float(np.hypot(self.dx, self.dy))


def tile_stats(rgb: np.ndarray, geo: Geometry) -> list[TileStats]:
    hx, hy = geo.half_tiles
    t = geo.tile_px
    pad = max(1, int(t * 0.12))  # ignore tile borders, which blend into neighbours
    out: list[TileStats] = []
    for dy in range(-hy, hy + 1):
        for dx in range(-hx, hx + 1):
            cx, cy = geo.tile_to_px(dx, dy)
            x0, y0 = int(cx - t / 2) + pad, int(cy - t / 2) + pad
            x1, y1 = int(cx + t / 2) - pad, int(cy + t / 2) - pad
            if not (geo.in_view(x0, y0) and geo.in_view(x1, y1)) or geo.masked(cx, cy):
                continue
            crop = rgb[y0:y1, x0:x1].astype(np.float32)
            if crop.size == 0:
                continue
            lum = crop @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            gx = np.abs(np.diff(lum, axis=1)).mean() if lum.shape[1] > 1 else 0.0
            gy = np.abs(np.diff(lum, axis=0)).mean() if lum.shape[0] > 1 else 0.0
            out.append(
                TileStats(dx, dy, crop.reshape(-1, 3).mean(axis=0), float((lum < 22).mean()), float(gx + gy))
            )
    return out


def space_tiles(stats: list[TileStats], dark_threshold: float = 0.92) -> set[tuple[int, int]]:
    """Tiles that are almost entirely black: open space (or unlit areas). Treated as unwalkable."""
    return {(s.dx, s.dy) for s in stats if s.dark_frac >= dark_threshold}


def find_markers(
    stats: list[TileStats],
    geo: Geometry,
    max_markers: int = 12,
    min_score: float = 1.0,
) -> list[Marker]:
    """Pick the tiles that stand out most from the typical floor around the player."""
    usable = [s for s in stats if s.dark_frac < 0.92 and (s.dx, s.dy) != (0, 0)]
    if len(usable) < 4:
        return []
    means = np.stack([s.mean for s in usable])
    details = np.array([s.detail for s in usable])
    floor_color = np.median(means, axis=0)
    color_dev = np.linalg.norm(means - floor_color, axis=1)
    color_z = (color_dev - np.median(color_dev)) / (np.std(color_dev) + 1e-6)
    detail_z = (details - np.median(details)) / (np.std(details) + 1e-6)
    scores = 0.6 * detail_z + 0.4 * color_z
    # Slight preference for nearby tiles: they're what the character can act on.
    dists = np.array([np.hypot(s.dx, s.dy) for s in usable])
    scores = scores - 0.04 * dists

    order = np.argsort(-scores)
    picked: list[Marker] = []
    taken: set[tuple[int, int]] = set()
    for i in order:
        if scores[i] < min_score or len(picked) >= max_markers:
            break
        s = usable[i]
        # Non-maximum suppression: one marker per 2x2 neighbourhood keeps the image readable.
        if any(abs(s.dx - a) <= 1 and abs(s.dy - b) <= 1 for a, b in taken):
            continue
        taken.add((s.dx, s.dy))
        px, py = geo.tile_to_px(s.dx, s.dy)
        picked.append(Marker(id=0, dx=s.dx, dy=s.dy, px=px, py=py, score=float(scores[i])))
    # Number markers by distance so "1" is always the closest.
    picked.sort(key=lambda m: (m.dist, m.dy, m.dx))
    for n, m in enumerate(picked, start=1):
        m.id = n
    return picked
