"""OCR with Apple's Vision framework (VNRecognizeTextRequest). No extra install needed."""

from __future__ import annotations

import numpy as np
import Quartz
import Vision
from Foundation import NSData

from ..platform import OcrLine
from .pool import pooled


def rgb_to_cgimage(rgb: np.ndarray):
    h, w, _ = rgb.shape
    rgbx = np.empty((h, w, 4), dtype=np.uint8)
    rgbx[..., :3] = rgb
    rgbx[..., 3] = 255
    data = NSData.dataWithBytes_length_(rgbx.tobytes(), rgbx.nbytes)
    provider = Quartz.CGDataProviderCreateWithCFData(data)
    return Quartz.CGImageCreate(
        w, h, 8, 32, w * 4,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrderDefault,
        provider, None, False, Quartz.kCGRenderingIntentDefault,
    )


class VisionOcr:
    def __init__(self, accurate: bool = True) -> None:
        self._level = (
            Vision.VNRequestTextRecognitionLevelAccurate if accurate else Vision.VNRequestTextRecognitionLevelFast
        )

    @pooled
    def read(self, rgb: np.ndarray) -> list[OcrLine]:
        if rgb.size == 0:
            return []
        h, w, _ = rgb.shape
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(self._level)
        req.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(rgb_to_cgimage(rgb), None)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            raise RuntimeError(f"Vision OCR failed: {err}")
        lines: list[OcrLine] = []
        for obs in req.results() or []:
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            c = cands[0]
            bb = obs.boundingBox()  # normalized, origin bottom-left
            x, y = bb.origin.x * w, (1 - bb.origin.y - bb.size.height) * h
            lines.append(OcrLine(str(c.string()), float(c.confidence()), (x, y, bb.size.width * w, bb.size.height * h)))
        lines.sort(key=lambda ln: (ln.box[1], ln.box[0]))
        return lines
