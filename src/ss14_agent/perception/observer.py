"""Turns raw frames into an Observation the skills and the decision layer can use."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..config import Config
from ..platform import OcrEngine, WindowInfo
from .chat import ChatReader
from .geometry import Geometry, compute_geometry
from .odometry import Odometry
from .tiles import Marker, find_markers, space_tiles, tile_stats


@dataclass
class Observation:
    t: float
    frame: np.ndarray
    geo: Geometry
    pos: tuple[float, float]                     # player position, tiles, odometry frame
    space: set[tuple[int, int]]                  # world tiles that look like open space
    markers: list[Marker] = field(default_factory=list)
    new_chat: list[str] = field(default_factory=list)
    recent_chat: list[str] = field(default_factory=list)
    moved: tuple[float, float] = (0.0, 0.0)      # odometry step since last frame

    def marker(self, mid: int) -> Marker | None:
        return next((m for m in self.markers if m.id == mid), None)


class Observer:
    def __init__(self, cfg: Config, ocr: OcrEngine, screen_size: tuple[float, float] | None = None) -> None:
        self.cfg = cfg
        self.odometry = Odometry()
        self.chat = ChatReader(ocr, cfg.ocr.chat_interval_s)
        self._screen = screen_size
        self.markers_every = 3   # recompute markers every N frames; it's the slowest step
        self._n = 0
        self._markers: list[Marker] = []
        self._space: set[tuple[int, int]] = set()

    def observe(self, frame: np.ndarray, win: WindowInfo, read_chat: bool = True) -> Observation:
        geo = compute_geometry(win, frame.shape[1], frame.shape[0], self.cfg.layout, self._screen)
        x, y = self.odometry.update(frame, geo)
        px, py = round(x), round(y)
        if self._n % self.markers_every == 0:
            stats = tile_stats(frame, geo)
            self._space = {(dx + px, dy + py) for dx, dy in space_tiles(stats)}
            self._markers = find_markers(stats, geo, self.cfg.agent.max_markers)
            for m in self._markers:
                m.world = (m.dx + x, m.dy + y)
        else:
            # Keep marker screen positions in step with the camera between recomputes.
            for m in self._markers:
                m.px, m.py = geo.tile_to_px(m.world[0] - x, m.world[1] - y)
                m.dx, m.dy = round(m.world[0] - x), round(m.world[1] - y)
        self._n += 1
        new_chat = self.chat.poll(frame, geo) if read_chat else []
        return Observation(
            t=time.time(), frame=frame, geo=geo, pos=(x, y), space=set(self._space),
            markers=[replace(m) for m in self._markers], new_chat=new_chat,
            recent_chat=list(self.chat.recent), moved=self.odometry.last_step,
        )
