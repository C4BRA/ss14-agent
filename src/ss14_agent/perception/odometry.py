"""Visual odometry: how far the player moved, measured from how the world shifted on screen.

Because the camera follows the player, walking right makes the whole scene
slide left. Phase correlation between consecutive frames measures that slide
to sub-pixel accuracy. The player sprite itself sits still at the centre, so
it's masked out.
"""

from __future__ import annotations

import numpy as np

from .geometry import Geometry


def to_gray_small(rgb: np.ndarray, factor: int) -> np.ndarray:
    g = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if factor > 1:
        h, w = g.shape[0] // factor * factor, g.shape[1] // factor * factor
        g = g[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return g


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return (dx, dy, peak) such that b ≈ a translated by (dx, dy). peak is a 0-1 quality score."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    corr = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[iy, ix])

    def refine(c_m, c_0, c_p):
        denom = c_m - 2 * c_0 + c_p
        return 0.0 if abs(denom) < 1e-12 else 0.5 * (c_m - c_p) / denom

    h, w = corr.shape
    sx = ix + refine(corr[iy, (ix - 1) % w], corr[iy, ix], corr[iy, (ix + 1) % w])
    sy = iy + refine(corr[(iy - 1) % h, ix], corr[iy, ix], corr[(iy + 1) % h, ix])
    if sx > w / 2:
        sx -= w
    if sy > h / 2:
        sy -= h
    return float(sx), float(sy), peak


class Odometry:
    """Accumulates the player's position in tiles since start. +x right, +y down (screen axes)."""

    def __init__(self, target_tile_px: float = 8.0, min_peak: float = 0.05) -> None:
        self.x = 0.0
        self.y = 0.0
        self._prev: np.ndarray | None = None
        self._factor = 1
        self._target = target_tile_px
        self._min_peak = min_peak
        self.last_quality = 0.0
        self.last_step = (0.0, 0.0)

    def reset(self) -> None:
        self.x = self.y = 0.0
        self._prev = None

    def _view_gray(self, rgb: np.ndarray, geo: Geometry) -> np.ndarray:
        self._factor = max(1, int(round(geo.tile_px / self._target)))
        x0, y0 = int(geo.view_x), int(geo.view_y)
        crop = rgb[y0:y0 + int(geo.view_h), x0:x0 + int(geo.view_w)]
        g = to_gray_small(crop, self._factor)
        # Blank out UI overlays (chat scrolling would look like movement).
        f = self._factor
        for mx, my, mw, mh in geo.masks:
            ys, xs = int((my - y0) / f), int((mx - x0) / f)
            g[max(0, ys):max(0, ys + int(mh / f) + 1), max(0, xs):max(0, xs + int(mw / f) + 1)] = g.mean()
        # Mask the player sprite (a 2x2 tile box around the centre) with the mean.
        t = geo.tile_px / self._factor
        cy, cx = g.shape[0] / 2, g.shape[1] / 2
        g[int(cy - t):int(cy + t), int(cx - t):int(cx + t)] = g.mean()
        return g

    def update(self, rgb: np.ndarray, geo: Geometry) -> tuple[float, float]:
        g = self._view_gray(rgb, geo)
        if self._prev is None or self._prev.shape != g.shape:
            self._prev = g
            self.last_step = (0.0, 0.0)
            return self.x, self.y
        sx, sy, peak = phase_shift(self._prev, g)
        self._prev = g
        self.last_quality = peak
        if peak < self._min_peak:
            self.last_step = (0.0, 0.0)
            return self.x, self.y
        tile_small = geo.tile_px / self._factor
        step = (-sx / tile_small, -sy / tile_small)
        # Ignore implausible jumps (teleports, UI flashes): walking is < 1 tile per frame.
        if abs(step[0]) > 1.5 or abs(step[1]) > 1.5:
            self.last_step = (0.0, 0.0)
            return self.x, self.y
        # Dead zone so lighting flicker doesn't make a standing player drift.
        step = tuple(0.0 if abs(s) < 0.02 else s for s in step)
        self.last_step = step
        self.x += step[0]
        self.y += step[1]
        return self.x, self.y
