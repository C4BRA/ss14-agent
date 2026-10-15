"""In-memory stand-ins for the macOS layer, so the agent core can be tested anywhere."""

from __future__ import annotations

import numpy as np

from ss14_agent.platform import OcrLine, Platform, WindowInfo

WIN = WindowInfo(window_id=1, pid=1, owner="Space Station 14", title="Space Station 14", x=0, y=0, width=700, height=508)
TILE = 32  # device pixels per tile in the fake world
VIEW_W, VIEW_H = 21 * TILE, 15 * TILE


def make_world(size: int = 80, seed: int = 0) -> np.ndarray:
    """A textured floor with some bright 'objects' and a wall line, at TILE px per tile."""
    rng = np.random.default_rng(seed)
    floor = rng.integers(70, 95, size=(size * TILE, size * TILE, 3), dtype=np.uint8)
    for _ in range(60):
        tx, ty = rng.integers(2, size - 2, size=2)
        color = rng.integers(150, 255, size=3, dtype=np.uint8)
        y0, x0 = ty * TILE + 8, tx * TILE + 8
        floor[y0:y0 + 16, x0:x0 + 16] = color
        floor[y0 + 6:y0 + 10, x0:x0 + 16] = 30
    return floor


class FakeWorld:
    """Renders a window-sized frame centred on the player, who moves when WASD are held."""

    def __init__(self, speed_tiles_per_frame: float = 0.25) -> None:
        self.world = make_world()
        self.px = 40.0 * TILE
        self.py = 40.0 * TILE
        self.speed = speed_tiles_per_frame * TILE
        self.held: set[str] = set()

    def step(self) -> None:
        vx = ("d" in self.held) - ("a" in self.held)
        vy = ("s" in self.held) - ("w" in self.held)
        self.px += vx * self.speed
        self.py += vy * self.speed

    def frame(self) -> np.ndarray:
        # Window: title bar (28 pt * scale 1) + content where the left 70% is the viewport.
        h, w = int(WIN.height), int(WIN.width)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:28] = 200
        x0 = int(round(self.px - VIEW_W / 2))
        y0 = int(round(self.py - VIEW_H / 2))
        view = self.world[y0:y0 + VIEW_H, x0:x0 + VIEW_W]
        # Tests use a layout (fake_layout) whose viewport exactly fits this 21x15 tile view.
        img[28:28 + VIEW_H, 0:VIEW_W] = view
        return img


class FakeWindows:
    def list_windows(self):
        return [WIN]

    def find(self, patterns):
        return WIN

    def activate(self, win):
        pass

    def screen_size(self):
        return (1920.0, 1080.0)


class FakeCapturer:
    name = "fake"

    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def grab(self, win):
        self.world.step()
        return self.world.frame()


class FakeInput:
    def __init__(self, world: FakeWorld | None = None) -> None:
        self.world = world
        self.events: list[tuple] = []
        self._held: set[str] = set()

    def _sync(self):
        if self.world is not None:
            self.world.held = set(self._held)

    def key_down(self, key):
        self._held.add(key)
        self.events.append(("down", key))
        self._sync()

    def key_up(self, key):
        self._held.discard(key)
        self.events.append(("up", key))
        self._sync()

    def tap(self, key, mods=()):
        self.events.append(("tap", key, tuple(mods)))

    def type_text(self, text):
        self.events.append(("type", text))

    def mouse_move(self, x, y):
        self.events.append(("move", round(x), round(y)))

    def click(self, x, y, button="left", mods=()):
        self.events.append(("click", button, round(x), round(y)))

    def release_all(self):
        for k in list(self._held):
            self.key_up(k)

    def held(self):
        return set(self._held)


class FakeOcr:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines or []

    def read(self, rgb):
        return [OcrLine(t, 0.9, (0, i * 10, 100, 10)) for i, t in enumerate(self.lines)]


class FakeWatcher:
    def start(self, on_user_input, on_hotkey):
        self.on_user_input, self.on_hotkey = on_user_input, on_hotkey
        return True

    def stop(self):
        pass


def fake_platform(world: FakeWorld | None = None, ocr_lines: list[str] | None = None) -> Platform:
    world = world or FakeWorld()
    return Platform(FakeWindows(), FakeCapturer(world), FakeInput(world), FakeOcr(ocr_lines), FakeWatcher())


def fake_layout(cfg):
    """Layout matching FakeWorld.frame(): viewport = left 672 px of a 700 px wide window, below a 28 px title bar."""
    content_h = WIN.height - 28
    cfg.layout.viewport = (0.0, 0.0, VIEW_W / WIN.width, VIEW_H / content_h)
    cfg.layout.chat = (0.97, 0.0, 0.03, 0.5)
    cfg.layout.titlebar_pt = 28
    return cfg
