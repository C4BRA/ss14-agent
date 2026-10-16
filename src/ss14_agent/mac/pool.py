"""Autorelease pools for Objective-C calls made from Python threads.

Cocoa objects returned by framework calls (captured images, OCR results, window
lists, input events) are "autoreleased": freed when the thread's autorelease pool
drains. Python's worker threads have no pool, so without one every frame's
screenshot stayed in memory for good. At several full-resolution frames a second
that's gigabytes within minutes, enough to bring a Mac to a halt or restart it.
Every public method of the macOS layer runs inside its own pool via @pooled.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

import objc

F = TypeVar("F", bound=Callable)


def pooled(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with objc.autorelease_pool():
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
