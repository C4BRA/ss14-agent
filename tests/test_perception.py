import numpy as np

from fakes import TILE, WIN, FakeWorld, fake_layout
from ss14_agent.config import load_config
from ss14_agent.perception.geometry import compute_geometry
from ss14_agent.perception.odometry import Odometry, phase_shift
from ss14_agent.perception.tiles import find_markers, space_tiles, tile_stats


def _geo(cfg=None):
    cfg = fake_layout(cfg or load_config())
    return compute_geometry(WIN, int(WIN.width), int(WIN.height), cfg.layout)


def test_config_loads_defaults():
    cfg = load_config()
    assert cfg.decision.model == "inception/mercury-decide:free"
    assert cfg.layout.viewport_tiles_high == 15
    assert cfg.agent.playbook == "janitor"


def test_geometry_tile_size_and_centre():
    g = _geo()
    assert abs(g.tile_px - TILE) < 1e-6
    cx, cy = g.center
    assert abs(cx - 21 * TILE / 2) < 1e-6 and abs(cy - (28 + 15 * TILE / 2)) < 1e-6
    x, y = g.tile_to_px(2, -1)
    assert g.px_to_tile(x, y) == (2.0, -1.0)
    assert g.half_tiles == (10, 7)


def test_geometry_letterboxes_when_area_is_wide():
    cfg = load_config()
    cfg.layout.titlebar_pt = 0
    cfg.layout.viewport = (0, 0, 1, 1)
    win = WIN.__class__(1, 1, "x", "x", 0, 0, 1000, 300)
    g = compute_geometry(win, 2000, 600, cfg.layout)
    assert g.scale == 2
    assert abs(g.tile_px - 40) < 1e-6        # 600 px / 15 tiles
    assert abs(g.view_w - 21 * 40) < 1e-6
    assert abs(g.view_x - (2000 - 840) / 2) < 1e-6


def test_phase_shift_recovers_translation():
    rng = np.random.default_rng(1)
    a = rng.random((96, 128)).astype(np.float32)
    b = np.roll(a, shift=(3, -5), axis=(0, 1))
    dx, dy, peak = phase_shift(a, b)
    assert abs(dx - (-5)) < 0.3 and abs(dy - 3) < 0.3 and peak > 0.2


def test_odometry_tracks_walking():
    world = FakeWorld(speed_tiles_per_frame=0.25)
    g = _geo()
    od = Odometry()
    od.update(world.frame(), g)
    world.held = {"d"}
    for _ in range(12):
        world.step()
        od.update(world.frame(), g)
    world.held = {"s"}
    for _ in range(8):
        world.step()
        od.update(world.frame(), g)
    assert abs(od.x - 3.0) < 0.3, od.x
    assert abs(od.y - 2.0) < 0.3, od.y


def test_markers_find_objects_and_space():
    world = FakeWorld()
    frame = world.frame()
    g = _geo()
    # Paint the far-left column of tiles black: open space.
    x_left, _ = g.tile_to_px(-10, 0)
    frame[28:, : int(x_left + TILE / 2)] = 0
    stats = tile_stats(frame, g)
    space = space_tiles(stats)
    assert any(dx == -10 for dx, _ in space)
    markers = find_markers(stats, g, max_markers=8)
    assert markers, "expected some salient tiles"
    assert [m.id for m in markers] == list(range(1, len(markers) + 1))
    assert all((m.dx, m.dy) != (0, 0) for m in markers)
    assert markers == sorted(markers, key=lambda m: m.dist)
