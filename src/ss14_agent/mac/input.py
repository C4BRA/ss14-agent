"""Synthetic keyboard and mouse input through Quartz event posting.

Needs Accessibility permission for the terminal (or app) running the agent.
Every event is tagged with AGENT_EVENT_TAG so the takeover watcher can tell
the agent's input apart from the user's.
"""

from __future__ import annotations

import threading
import time

import Quartz

from .keys import keycode

AGENT_EVENT_TAG = 0x55A9E7

_FLAG = {
    "shift": Quartz.kCGEventFlagMaskShift,
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "control": Quartz.kCGEventFlagMaskControl,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "cmd": Quartz.kCGEventFlagMaskCommand,
}

_MOUSE = {
    "left": (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp, Quartz.kCGMouseButtonLeft),
    "right": (Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp, Quartz.kCGMouseButtonRight),
}


class QuartzInput:
    def __init__(self, click_hold_s: float = 0.04) -> None:
        self._src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._click_hold_s = click_hold_s

    def _post(self, ev, flags: int = 0) -> None:
        Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGEventSourceUserData, AGENT_EVENT_TAG)
        if flags:
            Quartz.CGEventSetFlags(ev, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

    @staticmethod
    def _flags(mods) -> int:
        f = 0
        for m in mods:
            f |= _FLAG[m.lower()]
        return f

    def key_down(self, key: str) -> None:
        with self._lock:
            if key in self._held:
                return
            self._held.add(key)
        self._post(Quartz.CGEventCreateKeyboardEvent(self._src, keycode(key), True))

    def key_up(self, key: str) -> None:
        with self._lock:
            self._held.discard(key)
        self._post(Quartz.CGEventCreateKeyboardEvent(self._src, keycode(key), False))

    def tap(self, key: str, mods: tuple[str, ...] = ()) -> None:
        flags = self._flags(mods)
        for m in mods:
            self._post(Quartz.CGEventCreateKeyboardEvent(self._src, keycode(m), True), flags)
        self._post(Quartz.CGEventCreateKeyboardEvent(self._src, keycode(key), True), flags)
        time.sleep(0.03)
        self._post(Quartz.CGEventCreateKeyboardEvent(self._src, keycode(key), False), flags)
        for m in reversed(mods):
            self._post(Quartz.CGEventCreateKeyboardEvent(self._src, keycode(m), False))

    def type_text(self, text: str) -> None:
        for ch in text:
            down = Quartz.CGEventCreateKeyboardEvent(self._src, 0, True)
            Quartz.CGEventKeyboardSetUnicodeString(down, len(ch), ch)
            up = Quartz.CGEventCreateKeyboardEvent(self._src, 0, False)
            Quartz.CGEventKeyboardSetUnicodeString(up, len(ch), ch)
            self._post(down)
            self._post(up)
            time.sleep(0.012)

    def mouse_move(self, x: float, y: float) -> None:
        ev = Quartz.CGEventCreateMouseEvent(self._src, Quartz.kCGEventMouseMoved, (x, y), Quartz.kCGMouseButtonLeft)
        self._post(ev)

    def click(self, x: float, y: float, button: str = "left", mods: tuple[str, ...] = ()) -> None:
        down_t, up_t, btn = _MOUSE[button]
        flags = self._flags(mods)
        self.mouse_move(x, y)
        time.sleep(0.02)
        for m in mods:
            self._post(Quartz.CGEventCreateKeyboardEvent(self._src, keycode(m), True), flags)
        self._post(Quartz.CGEventCreateMouseEvent(self._src, down_t, (x, y), btn), flags)
        time.sleep(self._click_hold_s)
        self._post(Quartz.CGEventCreateMouseEvent(self._src, up_t, (x, y), btn), flags)
        for m in reversed(mods):
            self._post(Quartz.CGEventCreateKeyboardEvent(self._src, keycode(m), False))

    def release_all(self) -> None:
        with self._lock:
            held = list(self._held)
        for k in held:
            self.key_up(k)

    def held(self) -> set[str]:
        with self._lock:
            return set(self._held)
