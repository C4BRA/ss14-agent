"""The screen as text: a tile map, the things on it with names, and what's been read.

This is what the decision model gets instead of a screenshot when perception
mode is "text", and what the control window's Model view shows.

Names come from two places:
- probed: a right-click menu named it (stored in WorldMemory as a landmark);
- learned: its picture matches one the TileLibrary learned from an earlier probe.
Anything else is "unknown" until it's probed.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any

from ..memory import WorldMemory
from .library import TileLibrary, tile_crop
from .observer import Observation

MARKER_CHARS = string.digits[1:] + string.ascii_uppercase

TEXT_GUIDE = (
    "You can't see the screen; you get it as text. `what_you_see.map` is the area around you, "
    "north at the top, one character per floor tile: @ is you, . is floor, ~ is open space or "
    "darkness (you can't walk there), x is somewhere you bumped into, a blank is hidden behind "
    "the game's UI, and digits or capital letters are things you can target. `what_you_see.things` "
    "says what each one is and how sure that is: probed (read from the game's right-click menu), "
    "learned (looks like something probed before) or unknown. Actions refer to things by number. "
    "`text_on_screen_now` is any text visible over the game view, such as an open window or menu."
)


def marker_char(mid: int) -> str:
    return MARKER_CHARS[mid - 1] if 0 < mid <= len(MARKER_CHARS) else "*"


@dataclass
class Thing:
    id: int
    char: str
    dx: int
    dy: int
    world: tuple[float, float]
    label: str | None
    source: str            # probed | learned | unknown
    score: float = 0.0

    @property
    def where(self) -> str:
        parts = []
        if self.dy:
            parts.append(f"{abs(self.dy)}{'N' if self.dy < 0 else 'S'}")
        if self.dx:
            parts.append(f"{abs(self.dx)}{'W' if self.dx < 0 else 'E'}")
        return " ".join(parts) or "here"


@dataclass
class Scene:
    rows: list[str]
    things: list[Thing]
    chat: list[str] = field(default_factory=list)
    new_chat: list[str] = field(default_factory=list)
    read_text: list[str] = field(default_factory=list)   # menu and examine text read recently
    screen_text: list[str] | None = None                 # text on the game view now (open windows, popups)

    def labels(self) -> dict[int, str]:
        return {t.id: t.label for t in self.things if t.label}

    def unknown(self) -> list[Thing]:
        return [t for t in self.things if t.source == "unknown"]

    def for_state(self) -> dict[str, Any]:
        things = []
        for t in self.things:
            item: dict[str, Any] = {"number": t.id, "where": t.where}
            item["what"] = t.label or "unknown (probe it to find out)"
            item["how_sure"] = t.source if t.source != "learned" else f"learned ({t.score:.0%} match)"
            things.append(item)
        out: dict[str, Any] = {"map": self.rows, "things": things}
        if self.screen_text is not None:
            out["text_on_screen_now"] = self.screen_text or "none: no window or menu is open"
        if self.read_text:
            out["text_read_from_the_game"] = self.read_text[-6:]
        return out

    def as_text(self) -> str:
        """Plain-text rendering for the control window."""
        lines = ["MAP (north up, @ = you)", *self.rows, "", "THINGS"]
        for t in self.things:
            sure = {"probed": "probed", "unknown": "unknown"}.get(t.source, f"learned {t.score:.0%}")
            lines.append(f"  {t.char}  {t.label or '?':<28} {t.where:<8} [{sure}]")
        if self.screen_text:
            lines += ["", "TEXT ON SCREEN NOW", *(f"  {s}" for s in self.screen_text)]
        if self.read_text:
            lines += ["", "TEXT READ FROM THE GAME", *(f"  {s}" for s in self.read_text[-6:])]
        if self.chat:
            lines += ["", "CHAT", *(f"  {s}" for s in self.chat[-8:])]
        return "\n".join(lines)


def read_screen_text(obs: Observation, ocr, max_lines: int = 20) -> list[str]:
    """OCR the game view (not the chat or masked UI) for windows, menus and popups."""
    geo = obs.geo
    x0, y0 = int(geo.view_x), int(geo.view_y)
    view = obs.frame[y0:int(geo.view_y + geo.view_h):2, x0:int(geo.view_x + geo.view_w):2]
    out = []
    for ln in ocr.read(view):
        text = ln.text.strip()
        bx, by, bw, bh = ln.box
        cx, cy = x0 + 2 * (bx + bw / 2), y0 + 2 * (by + bh / 2)
        if ln.confidence >= 0.4 and len(text) > 1 and not geo.masked(cx, cy):
            out.append(text)
    return out[:max_lines]


def build_scene(
    obs: Observation,
    memory: WorldMemory | None = None,
    library: TileLibrary | None = None,
    blocked: set[tuple[int, int]] | None = None,
    read_text: list[str] | None = None,
) -> Scene:
    geo = obs.geo
    hx, hy = geo.half_tiles
    px, py = round(obs.pos[0]), round(obs.pos[1])
    blocked = blocked or set()

    things: list[Thing] = []
    for m in obs.markers:
        label, source, score = None, "unknown", 0.0
        probed = memory.label_at(m.world) if memory is not None else None
        if probed:
            label, source, score = probed, "probed", 1.0
        elif library is not None:
            hit = library.match(tile_crop(obs.frame, geo, m.dx, m.dy))
            if hit:
                label, score = hit
                source = "learned"
        things.append(Thing(m.id, marker_char(m.id), m.dx, m.dy, m.world, label, source, score))
    at = {(t.dx, t.dy): t.char for t in things}

    rows = []
    for dy in range(-hy, hy + 1):
        row = []
        for dx in range(-hx, hx + 1):
            cx, cy = geo.tile_to_px(dx, dy)
            if (dx, dy) == (0, 0):
                ch = "@"
            elif (dx, dy) in at:
                ch = at[(dx, dy)]
            elif not geo.in_view(cx, cy) or geo.masked(cx, cy):
                ch = " "
            elif (dx + px, dy + py) in obs.space:
                ch = "~"
            elif (dx + px, dy + py) in blocked:
                ch = "x"
            else:
                ch = "."
            row.append(ch)
        rows.append("".join(row))
    return Scene(rows, things, list(obs.recent_chat), list(obs.new_chat), list(read_text or []))
