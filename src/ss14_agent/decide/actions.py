"""The action menu offered to the decision model each tick, and the questions asked about it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..perception.observer import Observation
from ..perception.tiles import Marker
from ..playbook import Playbook

DIRECTIONS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}

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
) -> list[ActionSpec]:
    known = known or {}
    acts: list[ActionSpec] = []
    for m in obs.markers:
        name = f"marker {m.id}" + (f" ({known[m.id]})" if m.id in known else "")
        where = _offset_text(m)
        tgt = {"world": m.world}
        acts.append(ActionSpec(f"walk_to:{m.id}", "walk_to", f"Walk next to {name}, {where}.", tgt))
        acts.append(ActionSpec(
            f"click:{m.id}", "click",
            f"Left-click {name}: picks it up with an empty hand, or uses the held item on it. Walks into reach first.",
            tgt,
        ))
        acts.append(ActionSpec(
            f"activate:{m.id}", "activate",
            f"Activate {name} with E: opens, toggles or operates it (doors, machines, containers). Walks into reach first.",
            tgt,
        ))
        if m.id not in known:
            acts.append(ActionSpec(f"inspect:{m.id}", "inspect", f"Right-click {name} to read what it is.", tgt))
    for d, vec in DIRECTIONS.items():
        acts.append(ActionSpec(f"explore:{d}", "explore", f"Walk about 6 tiles {d} to explore.", {"dir": vec}))
    acts += [
        ActionSpec("use_held", "key", "Use the item in your active hand (Z), e.g. turn on a tool.", {"key": "z"}),
        ActionSpec("drop", "key", "Drop the item in your active hand (Q).", {"key": "q"}),
        ActionSpec("swap_hands", "key", "Switch to your other hand (X).", {"key": "x"}),
        ActionSpec(
            "close_window", "key",
            "Press Escape once to close an open window or menu. Only if a window or menu is open.",
            {"key": "escape"},
        ),
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
) -> dict[str, Any]:
    known = known or {}
    duty = playbook.duty(current_duty)
    state: dict[str, Any] = {
        "game": "Space Station 14, a multiplayer space station role-playing game.",
        "how_to_read_the_screen": SCREEN_GUIDE,
        "your_role": f"{playbook.name}. {playbook.summary}",
        "current_duty": (
            {"id": duty.id, "goal": duty.description, "steps": duty.steps} if duty else "none chosen yet"
        ),
        "markers_on_screen": [
            {"number": m.id, "where": _offset_text(m), **({"identified_as": known[m.id]} if m.id in known else {})}
            for m in obs.markers
        ],
        "new_chat_lines": obs.new_chat[-8:],
        "recent_chat": obs.recent_chat[-10:],
        "your_recent_actions": history[-8:],
        "notes": notes[-10:],
    }
    if operator_note:
        state["instruction_from_your_operator"] = operator_note
    return state


def build_questions(playbook: Playbook, actions: list[ActionSpec]) -> dict[str, dict[str, Any]]:
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
                "someone who just spoke to you. Use the numbered markers in the image to choose targets. "
                "Don't repeat an action that just failed in the same way."
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
