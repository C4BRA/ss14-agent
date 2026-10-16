"""The action menu offered to the decision model each tick, and the questions asked about it."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..memory import DIRECTIONS, WorldMemory, compass
from ..perception.observer import Observation
from ..perception.scene import TEXT_GUIDE, Scene
from ..perception.tiles import Marker
from ..playbook import Playbook

SCREEN_GUIDE = (
    "The image is the Space Station 14 game window, seen from above with north at the top. "
    "Your character is always in the cyan box at the centre. Yellow boxes with numbers mark "
    "things on screen you can target; actions refer to them by number. One tile is one square "
    "of floor. The chat panel shows what people are saying."
)


@dataclass
class ActionSpec:
    id: str                      # option key sent to the model, e.g. "walk_to:3"
    skill: str                   # skill name in ss14_agent.skills
    description: str
    args: dict[str, Any] = field(default_factory=dict)


def _offset_text(m: Marker) -> str:
    parts = []
    if m.dy:
        parts.append(f"{abs(m.dy)} {'north' if m.dy < 0 else 'south'}")
    if m.dx:
        parts.append(f"{abs(m.dx)} {'west' if m.dx < 0 else 'east'}")
    return (", ".join(parts) or "here") + f" ({m.dist:.1f} tiles away)"


def build_actions(
    obs: Observation,
    playbook: Playbook,
    reach: float,
    can_chat: bool,
    known: dict[int, str] | None = None,
    memory: WorldMemory | None = None,
    text_mode: bool = False,
    window_open: bool | None = None,
) -> list[ActionSpec]:
    """window_open: whether a window or menu is on screen, when known. Escape with nothing open
    brings up the game menu, so close_window is only offered when one might be open."""
    known = known or {}
    noun = "thing" if text_mode else "marker"
    acts: list[ActionSpec] = []

    def tried(skill: str, world: tuple[float, float]) -> str:
        if memory is None:
            return ""
        past = memory.attempts_at(skill, world)
        if not past:
            return ""
        return f" You already tried this here {len(past)} time(s); last result: {past[-1]}."

    for m in obs.markers:
        name = f"{noun} {m.id}" + (f" ({known[m.id]})" if m.id in known else "")
        where = _offset_text(m)
        tgt = {"world": m.world}
        acts.append(ActionSpec(
            f"walk_to:{m.id}", "walk_to", f"Walk next to {name}, {where}." + tried("walk_to", m.world), tgt,
        ))
        acts.append(ActionSpec(
            f"click:{m.id}", "click",
            f"Left-click {name}: picks it up with an empty hand, or uses the held item on it. Walks into reach first."
            + tried("click", m.world),
            tgt,
        ))
        acts.append(ActionSpec(
            f"activate:{m.id}", "activate",
            f"Activate {name} with E: opens, toggles or operates it (doors, machines, containers). Walks into reach first."
            + tried("activate", m.world),
            tgt,
        ))
        if m.id not in known:
            acts.append(ActionSpec(f"inspect:{m.id}", "inspect", f"Right-click {name} to read what it is.", tgt))
        acts.append(ActionSpec(
            f"examine:{m.id}", "examine",
            f"Examine {name} (shift-click) to read its description and state, e.g. whether a plant needs "
            "water or a machine is on." + tried("examine", m.world),
            tgt,
        ))
    for d, vec in DIRECTIONS.items():
        if memory is None:
            acts.append(ActionSpec(f"explore:{d}", "explore", f"Walk about 6 tiles {d} to explore.", {"dir": vec}))
            continue
        target = memory.explore_target(obs.pos, vec)
        if target is not None:
            share = memory.unexplored(obs.pos, vec)
            desc = (f"Explore {d}: walk to the nearest area {d} of you that you haven't seen yet, "
                    f"{compass(target[0] - obs.pos[0], target[1] - obs.pos[1])} ({share:.0%} of that way is unexplored).")
        else:
            desc = f"Walk about 6 tiles {d}. Everything that way is already explored, so this finds nothing new."
        acts.append(ActionSpec(f"explore:{d}", "explore", desc, {"dir": vec, "target": target}))
    if memory is not None:
        on_screen = {(round(m.world[0]), round(m.world[1])) for m in obs.markers}
        far = [lm for lm in memory.landmarks
               if (round(lm.world[0]), round(lm.world[1])) not in on_screen
               and math.hypot(lm.world[0] - obs.pos[0], lm.world[1] - obs.pos[1]) > 2.5]
        far.sort(key=lambda lm: math.hypot(lm.world[0] - obs.pos[0], lm.world[1] - obs.pos[1]))
        for i, lm in enumerate(far[:6]):
            acts.append(ActionSpec(
                f"go_back:{i}", "walk_to",
                f"Walk back to the {lm.name} you found earlier, "
                f"{compass(lm.world[0] - obs.pos[0], lm.world[1] - obs.pos[1])}.",
                {"world": lm.world},
            ))
    acts += [
        ActionSpec("use_held", "key", "Use the item in your active hand (Z), e.g. turn on a tool.", {"key": "z"}),
        ActionSpec("drop", "key", "Drop the item in your active hand (Q).", {"key": "q"}),
        ActionSpec("swap_hands", "key", "Switch to your other hand (X).", {"key": "x"}),
    ]
    if window_open is not False:
        acts.append(ActionSpec(
            "close_window", "key",
            "Press Escape once to close an open window or menu. Only if a window or menu is open.",
            {"key": "escape"},
        ))
    acts += [
        ActionSpec("wait", "wait", "Wait and watch for a moment.", {"seconds": 1.5}),
    ]
    if can_chat:
        for i, phrase in enumerate(playbook.phrases):
            acts.append(ActionSpec(f"say:{i}", "say", f'Say in local chat: "{phrase}"', {"text": phrase}))
    return acts


def build_state(
    obs: Observation,
    playbook: Playbook,
    current_duty: str | None,
    history: list[dict[str, str]],
    notes: list[str],
    operator_note: str = "",
    known: dict[int, str] | None = None,
    memory: WorldMemory | None = None,
    duty_info: dict[str, Any] | None = None,
    scene: Scene | None = None,
) -> dict[str, Any]:
    """The decision model's view of the situation. With a scene, the screen is described as text."""
    known = known or {}
    duty = playbook.duty(current_duty)
    state: dict[str, Any] = {
        "game": "Space Station 14, a multiplayer space station role-playing game.",
        "how_to_read_the_screen": TEXT_GUIDE if scene is not None else SCREEN_GUIDE,
        "your_role": f"{playbook.name}. {playbook.summary}",
        "current_duty": (
            {"id": duty.id, "goal": duty.description, "steps": duty.steps} if duty else "none chosen yet"
        ),
    }
    if scene is not None:
        state["what_you_see"] = scene.for_state()
    else:
        state["markers_on_screen"] = [
            {"number": m.id, "where": _offset_text(m), **({"identified_as": known[m.id]} if m.id in known else {})}
            for m in obs.markers
        ]
    state |= {
        "new_chat_lines": obs.new_chat[-8:],
        "recent_chat": obs.recent_chat[-10:],
        "your_recent_actions": history[-10:],
        "notes": notes[-10:],
    }
    if duty_info:
        state["progress_on_current_duty"] = duty_info
    if memory is not None:
        state["memory"] = memory.summary(obs.pos)
    if operator_note:
        state["instruction_from_your_operator"] = operator_note
    return state


def build_questions(playbook: Playbook, actions: list[ActionSpec], text_mode: bool = False) -> dict[str, dict[str, Any]]:
    targets = (
        "Things are numbered in what_you_see; unknown ones can be probed (inspect) to learn what they are. "
        if text_mode else "Use the numbered markers in the image to choose targets. "
    )
    return {
        "duty": {
            "type": "choice",
            "instructions": (
                "Which of your duties should you work on right now, given what's on screen, the chat, "
                "and what you've just been doing? Stick with the current duty unless something more "
                "important is happening or it's finished."
            ),
            "criteria": {d.id: f"{d.description} When: {d.cue}" for d in playbook.duties},
        },
        "action": {
            "type": "choice",
            "instructions": (
                "Pick the single next action that best moves the current duty forward, or answers "
                "someone who just spoke to you. If the state has a plan_from_your_orchestrator, pick "
                "the action that carries out its do_now step. " + targets +
                "Don't repeat an action that just failed in the same way. When you need to find "
                "something, explore toward unexplored areas instead of going back where you've been, "
                "and use go_back to return to things you identified earlier."
            ),
            "criteria": {a.id: a.description for a in actions},
        },
        "spoken_to": {
            "type": "noul",
            "instructions": "Did someone in the new chat lines speak to you or ask for your help?",
            "criteria": {"true": "Someone addressed you or asked for your job's help.", "false": "No one addressed you."},
        },
        "danger": {
            "type": "noul",
            "instructions": "Is your character being attacked, badly hurt, or in immediate danger?",
            "criteria": {"true": "Yes, in danger now.", "false": "No immediate danger."},
        },
    }
