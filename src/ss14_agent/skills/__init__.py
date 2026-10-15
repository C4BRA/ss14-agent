"""Closed-loop motor skills. The decision model picks one; the fast loop runs it frame by frame."""

from __future__ import annotations

from .base import Skill, SkillContext, Status
from .basic import Activate, Click, Explore, Inspect, KeyTap, Say, Wait
from .walk import WalkTo, astar

__all__ = [
    "Skill", "SkillContext", "Status", "WalkTo", "astar",
    "Click", "Activate", "Inspect", "KeyTap", "Say", "Wait", "Explore", "make_skill",
]


def make_skill(name: str, args: dict, reach: float) -> Skill:
    if name == "walk_to":
        return WalkTo(args["world"], stop_within=1.0)
    if name == "click":
        return Click(args["world"], reach)
    if name == "activate":
        return Activate(args["world"], reach)
    if name == "inspect":
        return Inspect(args["world"])
    if name == "explore":
        return Explore(args["dir"])
    if name == "key":
        return KeyTap(args["key"])
    if name == "say":
        return Say(args["text"])
    if name == "wait":
        return Wait(args.get("seconds", 1.5))
    raise ValueError(f"Unknown skill '{name}'")
