"""What the agent remembers between decisions.

A decision model sees one screenshot at a time, so without this it forgets
where it has been and what it already tried. The memory keeps:

- a map of every tile that has been on screen, so exploring heads for areas
  it hasn't seen instead of wandering back and forth;
- things it has identified (inspected), with where they are, so it can walk
  back to them after they leave the screen;
- what it tried at each spot and how that went, so it doesn't repeat failures.

Everything is in odometry coordinates (tiles from where the agent started).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

from .perception.observer import Observation

Tile = tuple[int, int]
DIRECTIONS: dict[str, Tile] = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}


@dataclass
class Landmark:
    world: tuple[float, float]
    name: str
    t: float


def compass(dx: float, dy: float) -> str:
    """'12 tiles north-east' style description of an offset in tiles (+y is south)."""
    d = math.hypot(dx, dy)
    if d < 1.5:
        return "right here"
    ns = "north" if dy < 0 else "south"
    ew = "west" if dx < 0 else "east"
    if abs(dy) > 2.4 * abs(dx):
        where = ns
    elif abs(dx) > 2.4 * abs(dy):
        where = ew
    else:
        where = f"{ns}-{ew}"
    return f"{d:.0f} tiles {where}"


class WorldMemory:
    def __init__(self, explore_reach: int = 14) -> None:
        self.seen: set[Tile] = set()
        self.space: set[Tile] = set()
        self.visits: dict[Tile, int] = {}
        self.landmarks: list[Landmark] = []
        self.attempts: dict[tuple[str, Tile], deque[str]] = {}
        self.explore_reach = explore_reach
        self._last_tile: Tile | None = None
        self._half: Tile = (10, 7)

    # ---- updates ---------------------------------------------------------

    def update(self, obs: Observation) -> None:
        tile = (round(obs.pos[0]), round(obs.pos[1]))
        self.space |= obs.space
        if tile == self._last_tile:
            return
        self._last_tile = tile
        self.visits[tile] = self.visits.get(tile, 0) + 1
        hx, hy = obs.geo.half_tiles
        self._half = (hx, hy)
        for y in range(tile[1] - hy, tile[1] + hy + 1):
            for x in range(tile[0] - hx, tile[0] + hx + 1):
                self.seen.add((x, y))

    def add_landmark(self, x: float, y: float, name: str) -> None:
        # One entry per spot: a later inspection of the same tile replaces the earlier one.
        self.landmarks = [lm for lm in self.landmarks if math.hypot(lm.world[0] - x, lm.world[1] - y) >= 0.7]
        self.landmarks.append(Landmark((x, y), name, time.time()))
        self.landmarks = self.landmarks[-40:]

    def record(self, skill: str, world: tuple[float, float] | None, outcome: str) -> None:
        if world is None:
            return
        key = (skill, (round(world[0]), round(world[1])))
        self.attempts.setdefault(key, deque(maxlen=4)).append(outcome)

    # ---- queries ---------------------------------------------------------

    def label_at(self, world: tuple[float, float]) -> str | None:
        best = min(((math.hypot(lm.world[0] - world[0], lm.world[1] - world[1]), lm.name) for lm in self.landmarks),
                   default=(9e9, ""))
        return best[1] if best[0] < 0.7 else None

    def attempts_at(self, skill: str, world: tuple[float, float]) -> list[str]:
        return list(self.attempts.get((skill, (round(world[0]), round(world[1]))), ()))

    def _cone(self, pos: tuple[float, float], direction: Tile):
        """Walkable-looking tiles ahead of pos in a 90-degree cone, nearest first."""
        px, py = round(pos[0]), round(pos[1])
        dx, dy = direction
        out = []
        for f in range(1, self.explore_reach + 1):
            for s in range(-f, f + 1):
                t = (px + dx * f - dy * s, py + dy * f + dx * s)
                if t not in self.space:
                    out.append((f, abs(s), t))
        return out

    def unexplored(self, pos: tuple[float, float], direction: Tile) -> float:
        """Share of the area ahead in this direction that has never been on screen (0-1)."""
        cone = self._cone(pos, direction)
        if not cone:
            return 0.0
        return sum(t not in self.seen for _, _, t in cone) / len(cone)

    def explore_target(self, pos: tuple[float, float], direction: Tile) -> tuple[float, float] | None:
        """Nearest never-seen tile ahead, pushed a few tiles further so the walk reveals new ground."""
        for f, _, t in sorted(self._cone(pos, direction)):
            if t not in self.seen:
                push = min(4, self.explore_reach - f)
                return (t[0] + direction[0] * push, t[1] + direction[1] * push)
        return None

    def times_walked_through(self, pos: tuple[float, float], direction: Tile, depth: int = 8) -> int:
        px, py = round(pos[0]), round(pos[1])
        return sum(self.visits.get((px + direction[0] * f, py + direction[1] * f), 0) for f in range(1, depth + 1))

    def summary(self, pos: tuple[float, float]) -> dict:
        """The memory as plain text for the decision model."""
        explored = {}
        for name, vec in DIRECTIONS.items():
            u = self.unexplored(pos, vec)
            if u > 0.6:
                explored[name] = "mostly unexplored"
            elif u > 0.2:
                explored[name] = "partly explored"
            else:
                explored[name] = "already explored"
        places = sorted(self.landmarks, key=lambda lm: math.hypot(lm.world[0] - pos[0], lm.world[1] - pos[1]))
        return {
            "where_you_are": "at your starting point" if math.hypot(*pos) < 1.5
            else f"{compass(pos[0], pos[1])} of where you started",
            "area_explored_so_far": f"{len(self.seen)} tiles seen",
            "exploration_by_direction": explored,
            "things_you_have_identified": [
                f"{lm.name}: {compass(lm.world[0] - pos[0], lm.world[1] - pos[1])}" for lm in places[:10]
            ],
        }
