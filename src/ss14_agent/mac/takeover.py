"""Watch the user's real keyboard and mouse with a listen-only Quartz event tap.

Any key press or click that didn't come from the agent (see AGENT_EVENT_TAG)
counts as the user taking over. F8 toggles pause. Needs Input Monitoring
permission; if the tap can't be created the agent runs without takeover and
logs a warning.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

import Quartz
from AppKit import NSWorkspace

from .input import AGENT_EVENT_TAG
from .keys import keycode

log = logging.getLogger(__name__)

HOTKEY = "f8"


class QuartzInputWatcher:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop = None
        self._tap = None

    def start(self, on_user_input: Callable[[], None], on_hotkey: Callable[[], None]) -> bool:
        hotkey_code = keycode(HOTKEY)
        own_pid = os.getpid()
        ready = threading.Event()
        ok = {"value": False}

        def callback(proxy, etype, event, refcon):
            if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(self._tap, True)
                return event
            if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGEventSourceUserData) == AGENT_EVENT_TAG:
                return event
            if etype == Quartz.kCGEventKeyDown:
                code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                if code == hotkey_code:
                    on_hotkey()
                    return event
            front = NSWorkspace.sharedWorkspace().frontmostApplication()
            if front is not None and front.processIdentifier() == own_pid:
                return event  # clicks on our own control window aren't a takeover
            on_user_input()
            return event

        mask = (
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
        )

        def run():
            self._tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionListenOnly, mask, callback, None,
            )
            if self._tap is None:
                ready.set()
                return
            source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
            self._loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(self._tap, True)
            ok["value"] = True
            ready.set()
            Quartz.CFRunLoopRun()

        self._thread = threading.Thread(target=run, name="input-watcher", daemon=True)
        self._thread.start()
        ready.wait(2.0)
        if not ok["value"]:
            log.warning(
                "Couldn't watch the keyboard/mouse, so takeover and the F8 hotkey are off. "
                "Grant Input Monitoring to your terminal in System Settings > Privacy & Security."
            )
        return ok["value"]

    def stop(self) -> None:
        if self._loop is not None:
            Quartz.CFRunLoopStop(self._loop)
            self._loop = None
