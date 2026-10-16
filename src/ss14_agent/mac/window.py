"""Find and focus the SS14 window using Quartz window lists."""

from __future__ import annotations

import Quartz
from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication, NSScreen

from ..platform import WindowInfo
from .pool import pooled


class QuartzWindows:
    @pooled
    def list_windows(self) -> list[WindowInfo]:
        opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        infos = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
        out: list[WindowInfo] = []
        for w in infos:
            if w.get("kCGWindowLayer", 0) != 0:  # skip menu bar, dock, overlays
                continue
            b = w.get("kCGWindowBounds", {})
            out.append(
                WindowInfo(
                    window_id=int(w["kCGWindowNumber"]),
                    pid=int(w.get("kCGWindowOwnerPID", 0)),
                    owner=str(w.get("kCGWindowOwnerName", "") or ""),
                    # Titles are empty unless the terminal has Screen Recording permission.
                    title=str(w.get("kCGWindowName", "") or ""),
                    x=float(b.get("X", 0)),
                    y=float(b.get("Y", 0)),
                    width=float(b.get("Width", 0)),
                    height=float(b.get("Height", 0)),
                )
            )
        return out

    def find(self, patterns: list[str]) -> WindowInfo | None:
        pats = [p.lower() for p in patterns]
        candidates = [
            w for w in self.list_windows()
            if any(p in w.title.lower() or p in w.owner.lower() for p in pats)
        ]
        if not candidates:
            return None
        # The biggest matching window is the game, not e.g. the launcher's popups.
        return max(candidates, key=lambda w: w.width * w.height)

    @pooled
    def activate(self, win: WindowInfo) -> None:
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(win.pid)
        if app is not None:
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)

    @pooled
    def screen_size(self) -> tuple[float, float]:
        frame = NSScreen.mainScreen().frame()
        return float(frame.size.width), float(frame.size.height)
