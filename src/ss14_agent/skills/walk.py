"""Walking: A* over tiles in odometry coordinates, steered with held WASD keys.

The map is learned as the agent goes: tiles that look like open space are
avoided, and tiles it bumps into are remembered as blocked.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque

from ..perception.observer import Observation
from .base import Skill, SkillContext, Status, dist_to, failed, ok, running

Tile = tuple[int, int]
MOVE_KEYS = ("w", "a", "s", "d")
_NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def astar(start: Tile, goal: Tile, blocked: set[Tile], margin: int = 8, max_nodes: int = 6000) -> list[Tile] | None:
    """8-connected A* inside a box around start and goal. No diagonal corner cutting."""
    if start == goal:
        return [start]
    x0, x1 = min(start[0], goal[0]) - margin, max(start[0], goal[0]) + margin
    y0, y1 = min(start[1], goal[1]) - margin, max(start[1], goal[1]) + margin

    def h(t: Tile) -> float:
        dx, dy = abs(t[0] - goal[0]), abs(t[1] - goal[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    open_heap: list[tuple[float, float, Tile]] = [(h(start), 0.0, start)]
    came: dict[Tile, Tile] = {}
    best: dict[Tile, float] = {start: 0.0}
    expanded = 0
    while open_heap and expanded < max_nodes:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if g > best.get(cur, math.inf):
            continue
        expanded += 1
        for dx, dy in _NEIGHBOURS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (x0 <= nxt[0] <= x1 and y0 <= nxt[1] <= y1) or (nxt in blocked and nxt != goal):
                continue
            if dx and dy and ((cur[0] + dx, cur[1]) in blocked or (cur[0], cur[1] + dy) in blocked):
                continue
            ng = g + (math.sqrt(2) if dx and dy else 1.0)
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def keys_for(vx: float, vy: float) -> set[str]:
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return set()
    vx, vy = vx / n, vy / n
    keys = set()
    if vx > 0.38:
        keys.add("d")
    elif vx < -0.38:
        keys.add("a")
    if vy > 0.38:
        keys.add("s")
    elif vy < -0.38:
        keys.add("w")
    return keys


class WalkTo(Skill):
    name = "walk_to"

    def __init__(self, world: tuple[float, float], stop_within: float = 0.35, timeout_s: float | None = None) -> None:
        self.target = (float(world[0]), float(world[1]))
        self.stop_within = stop_within
        self.timeout_s = timeout_s
        self._t0 = 0.0
        self._hist: deque[tuple[float, float, float]] = deque()
        self._stuck = 0
        self.path: list[Tile] = []

    def describe(self) -> str:
        return f"walk to ({self.target[0]:.0f}, {self.target[1]:.0f})"

    def start(self, ctx: SkillContext, obs: Observation) -> None:
        self._t0 = time.monotonic()
        d = dist_to(obs, self.target)
        if self.timeout_s is None:
            self.timeout_s = 4.0 + 1.5 * d

    def _set_keys(self, ctx: SkillContext, want: set[str]) -> None:
        held = ctx.input.held()
        for k in MOVE_KEYS:
            if k in want and k not in held:
                ctx.input.key_down(k)
            elif k not in want and k in held:
                ctx.input.key_up(k)

    def tick(self, ctx: SkillContext, obs: Observation) -> Status:
        now = time.monotonic()
        d = dist_to(obs, self.target)
        if d <= self.stop_within:
            self._set_keys(ctx, set())
            return ok(f"arrived ({d:.1f} tiles away)")
        if now - self._t0 > (self.timeout_s or 10):
            self._set_keys(ctx, set())
            return failed(f"timed out {d:.1f} tiles short")

        start = (round(obs.pos[0]), round(obs.pos[1]))
        goal = (round(self.target[0]), round(self.target[1]))
        blocked = (ctx.blocked | obs.space) - {start}
        path = astar(start, goal, blocked)
        if path is None:
            self._set_keys(ctx, set())
            return failed("no walkable path found")
        self.path = path
        waypoint = path[1] if len(path) > 1 else goal
        if len(path) > 2 and math.hypot(waypoint[0] - obs.pos[0], waypoint[1] - obs.pos[1]) < 0.35:
            waypoint = path[2]
        if len(path) <= 2 and self.stop_within >= 1.0:
            # Last step into an occupied tile: aim at the target itself and stop at range.
            waypoint = self.target
        self._set_keys(ctx, keys_for(waypoint[0] - obs.pos[0], waypoint[1] - obs.pos[1]))

        # Stuck detection: keys held but barely moving over the last ~0.7 s.
        self._hist.append((now, obs.pos[0], obs.pos[1]))
        while self._hist and now - self._hist[0][0] > 0.8:
            self._hist.popleft()
        t, x, y = self._hist[0]
        if now - t >= 0.6 and math.hypot(obs.pos[0] - x, obs.pos[1] - y) < 0.1:
            self._stuck += 1
            wp = (round(waypoint[0]), round(waypoint[1]))
            if wp != goal:
                ctx.blocked.add(wp)
            self._hist.clear()
            if self._stuck >= 5:
                self._set_keys(ctx, set())
                return failed(f"blocked, {d:.1f} tiles short")
        return running()
