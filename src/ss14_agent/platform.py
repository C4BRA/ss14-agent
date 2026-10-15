"""Interfaces the agent needs from the operating system.

The agent core only talks to these protocols, so it can be unit-tested with
fakes on any OS. The macOS implementations live in ss14_agent.mac.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass
class WindowInfo:
    window_id: int
    pid: int
    owner: str
    title: str
    # Window frame in global screen points (origin top-left of main display).
    x: float
    y: float
    width: float
    height: float


@dataclass
class OcrLine:
    text: str
    confidence: float
    # Bounding box in pixels of the image that was OCR'd: x, y, w, h (top-left origin).
    box: tuple[float, float, float, float]


class WindowFinder(Protocol):
    def list_windows(self) -> list[WindowInfo]: ...
    def find(self, patterns: list[str]) -> WindowInfo | None: ...
    def activate(self, win: WindowInfo) -> None: ...
    def screen_size(self) -> tuple[float, float]: ...


class Capturer(Protocol):
    name: str

    def grab(self, win: WindowInfo) -> np.ndarray:
        """Return the window as an RGB uint8 array (H, W, 3) in device pixels."""
        ...


class InputDriver(Protocol):
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def tap(self, key: str, mods: tuple[str, ...] = ()) -> None: ...
    def type_text(self, text: str) -> None: ...
    def mouse_move(self, x: float, y: float) -> None: ...
    def click(self, x: float, y: float, button: str = "left", mods: tuple[str, ...] = ()) -> None: ...
    def release_all(self) -> None: ...
    def held(self) -> set[str]: ...


class OcrEngine(Protocol):
    def read(self, rgb: np.ndarray) -> list[OcrLine]: ...


class InputWatcher(Protocol):
    """Watches the user's real keyboard/mouse for takeover and the pause hotkey."""

    def start(self, on_user_input: Callable[[], None], on_hotkey: Callable[[], None]) -> bool: ...
    def stop(self) -> None: ...


@dataclass
class Platform:
    windows: WindowFinder
    capturer: Capturer
    input: InputDriver
    ocr: OcrEngine
    watcher: InputWatcher


def get_platform(capture_backend: str = "auto") -> Platform:
    import sys

    if sys.platform != "darwin":
        raise RuntimeError(
            "ss14-agent's screen capture and input layers are macOS-only for now. "
            "The agent core and tests run anywhere."
        )
    from .mac import make_platform

    return make_platform(capture_backend)
