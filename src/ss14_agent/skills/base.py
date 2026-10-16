from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from ..perception.observer import Observation
from ..platform import InputDriver, OcrEngine

RUNNING = "running"


@dataclass
class Status:
    state: str            # "running", "ok" or "failed"
    detail: str = ""

    @property
    def done(self) -> bool:
        return self.state != RUNNING


def running() -> Status:
    return Status(RUNNING)


def ok(detail: str = "") -> Status:
    return Status("ok", detail)


def failed(detail: str) -> Status:
    return Status("failed", detail)


@dataclass
class SkillContext:
    input: InputDriver
    ocr: OcrEngine
    # Tiles the agent bumped into, in odometry coordinates. Shared across skills.
    # They expire: the bump may have been a person or a door, and odometry drifts.
    blocked: set[tuple[int, int]] = field(default_factory=set)
    blocked_at: dict[tuple[int, int], float] = field(default_factory=dict)
    blocked_ttl_s: float = 45.0
    # Called with (world_x, world_y, name, tile_pixels) when a skill learns what something is.
    on_label: Callable[..., None] = lambda x, y, name, crop=None: None
    # Called with (world_x, world_y, lines) when a skill reads text about something (examine).
    on_text: Callable[[float, float, list[str]], None] = lambda x, y, lines: None
    # Grab a fresh frame + observation on demand (used by skills that wait for UI).
    observe_now: Callable[[], Observation] | None = None


    def block(self, tile: tuple[int, int]) -> None:
        self.blocked.add(tile)
        self.blocked_at[tile] = time.monotonic()

    def prune_blocked(self) -> None:
        cutoff = time.monotonic() - self.blocked_ttl_s
        for tile in [t for t, at in self.blocked_at.items() if at < cutoff]:
            self.blocked.discard(tile)
            del self.blocked_at[tile]
        for tile in [t for t in self.blocked if t not in self.blocked_at]:  # added directly
            self.blocked_at[tile] = time.monotonic()

    def unblock_near(self, tile: tuple[int, int], radius: int = 2) -> None:
        for t in [t for t in self.blocked if max(abs(t[0] - tile[0]), abs(t[1] - tile[1])) <= radius]:
            self.blocked.discard(t)
            self.blocked_at.pop(t, None)


class Skill:
    name = "skill"

    def describe(self) -> str:
        return self.name

    def start(self, ctx: SkillContext, obs: Observation) -> None:
        pass

    def tick(self, ctx: SkillContext, obs: Observation) -> Status:
        raise NotImplementedError

    def stop(self, ctx: SkillContext) -> None:
        """Called when the skill ends or is interrupted. Must release any held input."""
        ctx.input.release_all()


def world_to_px(obs: Observation, world: tuple[float, float]) -> tuple[float, float]:
    return obs.geo.tile_to_px(world[0] - obs.pos[0], world[1] - obs.pos[1])


def dist_to(obs: Observation, world: tuple[float, float]) -> float:
    return ((world[0] - obs.pos[0]) ** 2 + (world[1] - obs.pos[1]) ** 2) ** 0.5
