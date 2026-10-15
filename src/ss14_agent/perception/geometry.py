"""Mapping between window pixels, screen points and tiles around the player.

SS14's camera is locked to the player, so the player is always at the centre
of the rendered viewport, and the world is a grid of square tiles.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LayoutCfg
from ..platform import WindowInfo

DEFAULT_TITLEBAR_PT = 28.0


@dataclass(frozen=True)
class Geometry:
    win: WindowInfo
    scale: float          # device pixels per point
    content_top: float    # pixels of title bar at the top of the capture
    content_w: float
    content_h: float
    view_x: float         # rendered game area inside the capture, in pixels
    view_y: float
    view_w: float
    view_h: float
    tile_px: float
    chat_rect: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        return self.view_x + self.view_w / 2, self.view_y + self.view_h / 2

    @property
    def half_tiles(self) -> tuple[int, int]:
        """How many whole tiles are visible either side of the player."""
        return int(self.view_w / self.tile_px / 2), int(self.view_h / self.tile_px / 2)

    def tile_to_px(self, dx: float, dy: float) -> tuple[float, float]:
        """Centre of the tile (dx, dy) relative to the player, in capture pixels. +y is down."""
        cx, cy = self.center
        return cx + dx * self.tile_px, cy + dy * self.tile_px

    def px_to_tile(self, x: float, y: float) -> tuple[float, float]:
        cx, cy = self.center
        return (x - cx) / self.tile_px, (y - cy) / self.tile_px

    def px_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Capture pixel to global screen point, for mouse events."""
        return self.win.x + x / self.scale, self.win.y + y / self.scale

    def in_view(self, x: float, y: float, margin: float = 0) -> bool:
        return (
            self.view_x + margin <= x <= self.view_x + self.view_w - margin
            and self.view_y + margin <= y <= self.view_y + self.view_h - margin
        )


def compute_geometry(
    win: WindowInfo,
    image_w: int,
    image_h: int,
    layout: LayoutCfg,
    screen_size: tuple[float, float] | None = None,
) -> Geometry:
    scale = image_w / win.width if win.width else 1.0
    titlebar_pt = layout.titlebar_pt
    if titlebar_pt < 0:
        fullscreen = screen_size is not None and (win.width, win.height) == tuple(screen_size)
        titlebar_pt = 0.0 if fullscreen else DEFAULT_TITLEBAR_PT
    top = titlebar_pt * scale
    cw, ch = float(image_w), float(image_h) - top

    vx, vy, vw, vh = layout.viewport
    area_x, area_y, area_w, area_h = vx * cw, top + vy * ch, vw * cw, vh * ch

    if layout.tile_px > 0:
        tile = float(layout.tile_px)
    else:
        tile = min(area_w / layout.viewport_tiles_wide, area_h / layout.viewport_tiles_high)
    # The game is scaled to fit and centred inside its area (letterboxed).
    rw = min(area_w, tile * layout.viewport_tiles_wide)
    rh = min(area_h, tile * layout.viewport_tiles_high)
    rx = area_x + (area_w - rw) / 2
    ry = area_y + (area_h - rh) / 2

    cx, cy, cwf, chf = layout.chat
    chat = (cx * cw, top + cy * ch, cwf * cw, chf * ch)
    return Geometry(win, scale, top, cw, ch, rx, ry, rw, rh, tile, chat)
