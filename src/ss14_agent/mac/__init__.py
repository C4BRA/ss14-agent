"""macOS implementations of the platform interfaces (Quartz, ScreenCaptureKit, Vision)."""

from __future__ import annotations

from ..platform import Platform


def make_platform(capture_backend: str = "auto") -> Platform:
    from .capture import make_capturer
    from .input import QuartzInput
    from .ocr import VisionOcr
    from .takeover import QuartzInputWatcher
    from .window import QuartzWindows

    return Platform(
        windows=QuartzWindows(),
        capturer=make_capturer(capture_backend),
        input=QuartzInput(),
        ocr=VisionOcr(),
        watcher=QuartzInputWatcher(),
    )
