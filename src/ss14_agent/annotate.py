"""Draw numbered markers on a frame ("set-of-marks") and encode it for the decision model."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .perception.geometry import Geometry
from .perception.tiles import Marker

MARKER_COLOR = (255, 230, 0)
PLAYER_COLOR = (0, 220, 255)


def _font(size: int):
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(
    frame: np.ndarray,
    geo: Geometry,
    markers: list[Marker],
    width: int | None = None,
    draw_layout: bool = False,
) -> Image.Image:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    t = geo.tile_px
    lw = max(2, int(t / 14))
    font = _font(max(12, int(t * 0.45)))

    if draw_layout:
        d.rectangle([geo.view_x, geo.view_y, geo.view_x + geo.view_w, geo.view_y + geo.view_h], outline=(0, 255, 0), width=lw)
        cx, cy, cw, ch = geo.chat_rect
        d.rectangle([cx, cy, cx + cw, cy + ch], outline=(255, 0, 255), width=lw)
        for mx, my, mw, mh in geo.masks[1:]:
            d.rectangle([mx, my, mx + mw, my + mh], outline=(255, 120, 0), width=lw)
        hx, hy = geo.half_tiles
        for gx in range(-hx, hx + 1):
            x, _ = geo.tile_to_px(gx - 0.5, 0)
            d.line([x, geo.view_y, x, geo.view_y + geo.view_h], fill=(80, 80, 80), width=1)
        for gy in range(-hy, hy + 1):
            _, y = geo.tile_to_px(0, gy - 0.5)
            d.line([geo.view_x, y, geo.view_x + geo.view_w, y], fill=(80, 80, 80), width=1)

    px, py = geo.center
    d.rectangle([px - t / 2, py - t / 2, px + t / 2, py + t / 2], outline=PLAYER_COLOR, width=lw)

    for m in markers:
        if not geo.in_view(m.px, m.py):
            continue
        d.rectangle([m.px - t / 2, m.py - t / 2, m.px + t / 2, m.py + t / 2], outline=MARKER_COLOR, width=lw)
        label = str(m.id)
        lx, ly = m.px - t / 2, m.py - t / 2
        bbox = d.textbbox((lx, ly), label, font=font)
        d.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(0, 0, 0))
        d.text((lx, ly), label, fill=MARKER_COLOR, font=font)

    if width and img.width > width:
        img = img.resize((width, int(img.height * width / img.width)), Image.LANCZOS)
    return img


def to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_jpeg_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
