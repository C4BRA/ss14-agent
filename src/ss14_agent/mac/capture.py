"""Window capture: ScreenCaptureKit (macOS 14+) with a CGWindowListCreateImage fallback."""

from __future__ import annotations

import logging
import threading

import numpy as np
import Quartz

from ..platform import WindowInfo

log = logging.getLogger(__name__)


def cgimage_to_rgb(img) -> np.ndarray:
    """Convert a CGImage (32-bit BGRA, the format window captures use) to RGB uint8."""
    w = Quartz.CGImageGetWidth(img)
    h = Quartz.CGImageGetHeight(img)
    bpr = Quartz.CGImageGetBytesPerRow(img)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(h, bpr // 4, 4)[:, :w, :]
    info = Quartz.CGImageGetBitmapInfo(img)
    little = (info & Quartz.kCGBitmapByteOrderMask) == Quartz.kCGBitmapByteOrder32Little
    if little:  # memory order B, G, R, A
        return np.ascontiguousarray(arr[..., [2, 1, 0]])
    return np.ascontiguousarray(arr[..., [1, 2, 3]])  # A, R, G, B


class CGWindowCapturer:
    name = "cgwindow"

    def grab(self, win: WindowInfo) -> np.ndarray:
        img = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            win.window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming | Quartz.kCGWindowImageBestResolution,
        )
        if img is None:
            raise RuntimeError(
                "Window capture returned nothing. Grant Screen Recording permission to your "
                "terminal app in System Settings > Privacy & Security, then restart it."
            )
        return cgimage_to_rgb(img)


class ScreenCaptureKitCapturer:
    """Uses SCScreenshotManager (macOS 14+). Completion handlers arrive on a GCD queue."""

    name = "screencapturekit"

    def __init__(self) -> None:
        import ScreenCaptureKit as SCK  # noqa: N814

        self.SCK = SCK
        if not hasattr(SCK, "SCScreenshotManager"):
            raise RuntimeError("SCScreenshotManager needs macOS 14 or newer")
        self._sc_window = None
        self._sc_window_id = None

    def _wait(self, starter, timeout: float = 3.0):
        done = threading.Event()
        box: dict = {}

        def handler(result, error):
            box["result"], box["error"] = result, error
            done.set()

        starter(handler)
        if not done.wait(timeout):
            raise TimeoutError("ScreenCaptureKit did not respond")
        if box.get("error") is not None:
            raise RuntimeError(f"ScreenCaptureKit error: {box['error']}")
        return box["result"]

    def _window(self, win: WindowInfo):
        if self._sc_window_id != win.window_id:
            sc = self.SCK.SCShareableContent
            get_content = sc.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_
            content = self._wait(lambda h: get_content(True, True, h))
            match = [w for w in content.windows() if int(w.windowID()) == win.window_id]
            if not match:
                raise RuntimeError("ScreenCaptureKit can't see the SS14 window (Screen Recording permission?)")
            self._sc_window, self._sc_window_id = match[0], win.window_id
        return self._sc_window

    def grab(self, win: WindowInfo) -> np.ndarray:
        SCK = self.SCK
        scw = self._window(win)
        filt = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(scw)
        cfg = SCK.SCStreamConfiguration.alloc().init()
        scale = float(filt.pointPixelScale()) if hasattr(filt, "pointPixelScale") else 2.0
        frame = scw.frame()
        cfg.setWidth_(int(frame.size.width * scale))
        cfg.setHeight_(int(frame.size.height * scale))
        cfg.setShowsCursor_(False)
        img = self._wait(
            lambda h: SCK.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(filt, cfg, h)
        )
        return cgimage_to_rgb(img)


class AutoCapturer:
    """Tries ScreenCaptureKit, and switches to CGWindowListCreateImage for good on the first failure."""

    def __init__(self) -> None:
        self._cg = CGWindowCapturer()
        try:
            self._active = ScreenCaptureKitCapturer()
        except Exception as exc:  # depends on macOS version
            log.info("ScreenCaptureKit unavailable (%s); using CGWindowListCreateImage", exc)
            self._active = self._cg

    @property
    def name(self) -> str:
        return self._active.name

    def grab(self, win: WindowInfo) -> np.ndarray:
        try:
            return self._active.grab(win)
        except Exception as exc:
            if self._active is self._cg:
                raise
            log.warning("ScreenCaptureKit capture failed (%s); falling back to CGWindowListCreateImage", exc)
            self._active = self._cg
            return self._cg.grab(win)


def make_capturer(backend: str = "auto"):
    if backend == "screencapturekit":
        return ScreenCaptureKitCapturer()
    if backend == "cgwindow":
        return CGWindowCapturer()
    return AutoCapturer()
