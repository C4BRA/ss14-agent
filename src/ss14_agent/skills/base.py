from __future__ import annotations

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
    blocked: set[tuple[int, int]] = field(default_factory=set)
    # Called with (world_x, world_y, name) when a skill learns what something is.
    on_label: Callable[[float, float, str], None] = lambda x, y, name: None
    # Grab a fresh frame + observation on demand (used by skills that wait for UI).
    observe_now: Callable[[], Observation] | None = None


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
