"""Check (and prompt for) the macOS privacy permissions the agent needs."""

from __future__ import annotations

import Quartz


def check(prompt: bool = False) -> dict[str, bool]:
    """Returns {permission: granted}. With prompt=True, asks macOS to show the dialogs."""
    screen = bool(Quartz.CGPreflightScreenCaptureAccess())
    post = bool(Quartz.CGPreflightPostEventAccess())
    listen = bool(Quartz.CGPreflightListenEventAccess())
    if prompt:
        if not screen:
            Quartz.CGRequestScreenCaptureAccess()
        if not post:
            Quartz.CGRequestPostEventAccess()
        if not listen:
            Quartz.CGRequestListenEventAccess()
    return {
        "Screen Recording (capture the game window)": screen,
        "Accessibility (send keys and clicks)": post,
        "Input Monitoring (detect when you take over, F8 hotkey)": listen,
    }
