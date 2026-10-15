"""Read new chat lines from the chat panel with OCR."""

from __future__ import annotations

import re
import time
from collections import deque

import numpy as np

from ..platform import OcrEngine
from .geometry import Geometry


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class ChatReader:
    def __init__(self, ocr: OcrEngine, interval_s: float = 1.0, history: int = 30) -> None:
        self._ocr = ocr
        self._interval = interval_s
        self._last = 0.0
        self._seen: deque[str] = deque(maxlen=400)
        self._seen_set: set[str] = set()
        self.recent: deque[str] = deque(maxlen=history)
        self._primed = False

    def _remember(self, key: str) -> None:
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(key)
        self._seen_set.add(key)

    def poll(self, rgb: np.ndarray, geo: Geometry, force: bool = False) -> list[str]:
        """Return chat lines that weren't on screen before. Rate limited to interval_s."""
        now = time.monotonic()
        if not force and now - self._last < self._interval:
            return []
        self._last = now
        x, y, w, h = (int(v) for v in geo.chat_rect)
        crop = rgb[max(0, y):y + h, max(0, x):x + w]
        lines = [ln.text.strip() for ln in self._ocr.read(crop) if ln.confidence >= 0.3 and ln.text.strip()]
        new: list[str] = []
        for text in lines:
            key = _norm(text)
            if len(key) < 2 or key in self._seen_set:
                continue
            self._remember(key)
            new.append(text)
        if not self._primed:
            # Whatever is already in the chat box when we start isn't "new".
            self._primed = True
            self.recent.extend(new)
            return []
        self.recent.extend(new)
        return new
