import numpy as np
import pytest

from ss14_agent.imageconv import buffer_to_rgb


def _bgra(h, w, pad_bytes=0, trailing=0):
    rgb = np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    bgra = np.zeros((h, w * 4 + pad_bytes), dtype=np.uint8)
    px = bgra[:, : w * 4].reshape(h, w, 4)
    px[..., 0], px[..., 1], px[..., 2], px[..., 3] = rgb[..., 2], rgb[..., 1], rgb[..., 0], 255
    return rgb, bgra.tobytes() + b"\0" * trailing, w * 4 + pad_bytes


def test_trailing_bytes_are_ignored():
    # The real failure: ScreenCaptureKit returned 1024 bytes more than height * bytes_per_row.
    rgb, buf, bpr = _bgra(6, 8, trailing=1024)
    assert np.array_equal(buffer_to_rgb(buf, 8, 6, bpr, "BGRA"), rgb)


def test_padded_rows():
    rgb, buf, bpr = _bgra(5, 7, pad_bytes=36)
    assert np.array_equal(buffer_to_rgb(buf, 7, 5, bpr, "BGRA"), rgb)


def test_other_orders_and_short_buffer():
    px = np.array([[[10, 20, 30, 40]]], dtype=np.uint8).tobytes()
    assert buffer_to_rgb(px, 1, 1, 4, "RGBA").tolist() == [[[10, 20, 30]]]
    assert buffer_to_rgb(px, 1, 1, 4, "ARGB").tolist() == [[[20, 30, 40]]]
    with pytest.raises(ValueError):
        buffer_to_rgb(px, 2, 1, 8, "BGRA")
