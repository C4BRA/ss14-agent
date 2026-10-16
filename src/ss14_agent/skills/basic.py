from __future__ import annotations

import math
import time

from ..perception.observer import Observation
from .base import Skill, SkillContext, Status, dist_to, failed, ok, running, world_to_px
from .walk import WalkTo


class _Timed(Skill):
    """Helper: a skill that finishes after `duration` seconds."""

    duration = 0.25

    def start(self, ctx: SkillContext, obs: Observation) -> None:
        self._t0 = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self._t0


class _Approach(Skill):
    """Walk into reach of a world tile, then do something there."""

    def __init__(self, world: tuple[float, float], reach: float) -> None:
        self.world = (float(world[0]), float(world[1]))
        self.reach = reach
        self._walk: WalkTo | None = None
        self._acted_at: float | None = None

    def start(self, ctx: SkillContext, obs: Observation) -> None:
        if dist_to(obs, self.world) > self.reach:
            self._walk = WalkTo(self.world, stop_within=max(0.6, self.reach - 0.25))
            self._walk.start(ctx, obs)

    def act(self, ctx: SkillContext, obs: Observation, sx: float, sy: float) -> None:
        raise NotImplementedError

    def tick(self, ctx: SkillContext, obs: Observation) -> Status:
        if self._walk is not None:
            st = self._walk.tick(ctx, obs)
            if not st.done:
                return st
            self._walk = None
            if st.state == "failed" and dist_to(obs, self.world) > self.reach + 0.4:
                return failed(f"couldn't get in reach: {st.detail}")
        if self._acted_at is None:
            x, y = world_to_px(obs, self.world)
            if not obs.geo.in_view(x, y):
                return failed("target is no longer on screen")
            sx, sy = obs.geo.px_to_screen(x, y)
            self.act(ctx, obs, sx, sy)
            self._acted_at = time.monotonic()
            return running()
        if time.monotonic() - self._acted_at >= 0.3:
            return ok("done")
        return running()


class Click(_Approach):
    name = "click"

    def describe(self) -> str:
        return f"click ({self.world[0]:.0f}, {self.world[1]:.0f})"

    def act(self, ctx: SkillContext, obs: Observation, sx: float, sy: float) -> None:
        ctx.input.click(sx, sy, "left")


class Activate(_Approach):
    name = "activate"

    def describe(self) -> str:
        return f"activate ({self.world[0]:.0f}, {self.world[1]:.0f})"

    def act(self, ctx: SkillContext, obs: Observation, sx: float, sy: float) -> None:
        ctx.input.mouse_move(sx, sy)
        time.sleep(0.05)
        ctx.input.tap("e")


class Inspect(Skill):
    """Right-click opens SS14's entity menu, which lists what's under the cursor by name."""

    name = "inspect"

    def __init__(self, world: tuple[float, float]) -> None:
        self.world = (float(world[0]), float(world[1]))
        self._t0: float | None = None
        self._click_px = (0.0, 0.0)

    def describe(self) -> str:
        return f"inspect ({self.world[0]:.0f}, {self.world[1]:.0f})"

    def tick(self, ctx: SkillContext, obs: Observation) -> Status:
        if self._t0 is None:
            x, y = world_to_px(obs, self.world)
            if not obs.geo.in_view(x, y):
                return failed("target is not on screen")
            ctx.input.click(*obs.geo.px_to_screen(x, y), button="right")
            self._click_px = (x, y)
            self._t0 = time.monotonic()
            return running()
        if time.monotonic() - self._t0 < 0.4:
            return running()
        t = obs.geo.tile_px
        x, y = self._click_px
        h, w, _ = obs.frame.shape
        x0, y0 = int(max(0, x - t)), int(max(0, y - t))
        x1, y1 = int(min(w, x + 9 * t)), int(min(h, y + 7 * t))
        lines = [ln.text.strip() for ln in ctx.ocr.read(obs.frame[y0:y1, x0:x1]) if ln.confidence >= 0.4]
        lines = [ln for ln in lines if len(ln) > 1][:5]
        ctx.input.tap("escape")
        if not lines:
            return failed("no menu text read")
        ctx.on_label(self.world[0], self.world[1], lines[0])
        return ok("it's: " + "; ".join(lines))


class KeyTap(_Timed):
    name = "key"

    def __init__(self, key: str) -> None:
        self.key = key

    def describe(self) -> str:
        return f"press {self.key.upper()}"

    def start(self, ctx: SkillContext, obs: Observation) -> None:
        super().start(ctx, obs)
        ctx.input.tap(self.key)

    def tick(self, ctx: SkillContext, obs: Observation) -> Status:
        return ok() if self.elapsed() >= self.duration else running()


class Say(_Timed):
    name = "say"
    duration = 0.3

    def __init__(self, text: str) -> None:
        self.text = text

    def describe(self) -> str:
        return f'say "{self.text}"'

    def start(self, ctx: SkillContext, obs: Observation) -> None:
        super().start(ctx, obs)
        ctx.input.release_all()
        ctx.input.tap("t")          # focus the chat input
        time.sleep(0.15)
        ctx.input.type_text(self.text)
        ctx.input.tap("return")

    def tick(self, ctx: SkillContext, obs: Observation) -> Status:
        return ok() if self.elapsed() >= self.duration else running()


class Wait(_Timed):
    name = "wait"

    def __init__(self, seconds: float = 1.5) -> None:
        self.duration = seconds

    def describe(self) -> str:
        return f"wait {self.duration:.1f}s"

    def tick(self, ctx: SkillContext, obs: Observation) -> Status:
        return ok() if self.elapsed() >= self.duration else running()


class Explore(Skill):
    name = "explore"
    distance = 6

    def __init__(self, direction: tuple[int, int], target: tuple[float, float] | None = None) -> None:
        self.direction = direction
        self.target = target       # nearest unseen area that way, from WorldMemory
        self._walk: WalkTo | None = None
        self._start = (0.0, 0.0)

    def describe(self) -> str:
        names = {(0, -1): "north", (1, 0): "east", (0, 1): "south", (-1, 0): "west"}
        return f"explore {names.get(tuple(self.direction), self.direction)}"

    def start(self, ctx: SkillContext, obs: Observation) -> None:
        self._start = obs.pos
        if self.target is not None:
            tx, ty = self.target
        else:
            tx = obs.pos[0] + self.direction[0] * self.distance
            ty = obs.pos[1] + self.direction[1] * self.distance
        self._walk = WalkTo((tx, ty), stop_within=1.0)
        self._walk.start(ctx, obs)

    def tick(self, ctx: SkillContext, obs: Observation) -> Status:
        st = self._walk.tick(ctx, obs)
        if not st.done:
            return st
        moved = math.hypot(obs.pos[0] - self._start[0], obs.pos[1] - self._start[1])
        if st.state == "failed" and moved >= 2:
            return ok(f"walked {moved:.0f} tiles, then {st.detail}")
        return st if st.state == "failed" else ok(f"walked {moved:.0f} tiles")
