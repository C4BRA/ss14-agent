"""Pixel buffer conversion, kept free of macOS imports so it can be unit-tested anywhere."""

from __future__ import annotations

import numpy as np


def buffer_to_rgb(buf: bytes, width: int, height: int, bytes_per_row: int, order: str) -> np.ndarray:
    """Convert a 32-bit-per-pixel buffer to an RGB uint8 array (H, W, 3).

    `order` is the byte order of each pixel in memory: "BGRA", "ARGB" or "RGBA".
    Rows may be padded (bytes_per_row > width * 4) and the buffer may carry
    trailing bytes past the last row; both are ignored.
    """
    need = height * bytes_per_row
    if len(buf) < need:
        raise ValueError(f"Pixel buffer too small: {len(buf)} bytes for {height} rows of {bytes_per_row}")
    rows = np.frombuffer(buf, dtype=np.uint8, count=need).reshape(height, bytes_per_row)
    px = rows[:, : width * 4].reshape(height, width, 4)
    idx = {"BGRA": [2, 1, 0], "ARGB": [1, 2, 3], "RGBA": [0, 1, 2]}[order]
    return np.ascontiguousarray(px[..., idx])
