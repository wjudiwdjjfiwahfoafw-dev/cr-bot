"""
Tests for the real-device stack: scrcpy protocol, touch injection, calibration, projection,
deploy verification, and the elixir reader.

The calibration/elixir tests run against ``calibration_preview.png``, which is a genuine
720x1280 Clash Royale battle frame included in the repo. They are regression tests for the
layout facts that were measured on it:

    river line            y ~ 567
    hand card centres     x = 224, 360, 496, 631 at y ~ 1162
    card portrait size    ~127 x 114
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from cr_bot.adb.touch import CoordMap, GestureResult, TouchInjector
from cr_bot.arena.device_calibrator import DeviceCalibrator, render_calibration_preview
from cr_bot.arena.grid import ArenaCalibration, ArenaGrid
from cr_bot.arena.projection import (
    HomographyProjection,
    PRINCESS_TILES,
    RectProjection,
    projection_from_landmarks,
    solve_homography,
)
from cr_bot.vision.deploy_verifier import DeployVerifier
from cr_bot.vision.elixir_tracker import ElixirReader

FIXTURE = Path(__file__).resolve().parents[1] / "calibration_preview.png"


# ======================================================================================
# Coordinate mapping & injection
# ======================================================================================


def test_coord_map_scales_frame_space_to_device_pixels() -> None:
    """A downscaled capture stream must not shift taps."""
    coord = CoordMap(frame_w=720, frame_h=1600, device_w=1080, device_h=2400)
    assert coord.to_device(360, 800) == (540, 1200)
    assert coord.to_device(0, 0) == (0, 0)
    # Clamping happens in frame space.
    assert coord.clamp_frame(-10, 9999) == (0, 1599)


class _RaisingBackend:
    name = "swipe"
    available = True
    last_latency_ms = 0.0

    def __init__(self) -> None:
        self.calls = 0

    def tap(self, x, y, hold_ms=70.0):
        self.calls += 1
        raise OSError("simulated adb failure")

    def drag(self, start, end, duration_ms=180.0, steps=8):
        self.calls += 1
        raise OSError("simulated adb failure")


class _WorkingBackend:
    name = "motionevent"
    available = True
    last_latency_ms = 12.0

    def __init__(self) -> None:
        self.calls: List[Tuple] = []

    def tap(self, x, y, hold_ms=70.0):
        self.calls.append(("tap", x, y))

    def drag(self, start, end, duration_ms=180.0, steps=8):
        self.calls.append(("drag", start, end))


def test_injector_escalates_to_next_backend_and_adopts_it() -> None:
    """If the preferred backend fails, the next one must be used and become active."""
    injector = TouchInjector.__new__(TouchInjector)  # bypass __init__ (needs real adb)
    failing = _RaisingBackend()
    working = _WorkingBackend()
    injector.adb = None  # type: ignore[assignment]
    injector.coord = CoordMap(720, 1280, 720, 1280)
    injector.deploy_style = "drag"
    injector.jitter_px = 0
    injector._rng = __import__("random").Random(0)
    injector.backends = {"swipe": failing, "motionevent": working}
    injector.order = ["swipe", "motionevent"]
    injector.active = "swipe"
    injector.attempts = []

    result = injector.play_card((200, 1100), (300, 700))
    assert isinstance(result, GestureResult)
    assert result.ok is True
    assert result.backend == "motionevent"
    assert result.attempts == 2
    assert failing.calls == 1
    assert working.calls and working.calls[0][0] == "drag"
    assert injector.active == "motionevent"


# ======================================================================================
# Projection
# ======================================================================================


def _synthetic_perspective() -> HomographyProjection:
    """A camera that compresses the far half, like the real game."""
    h = solve_homography(
        [(0, 0), (18, 0), (18, 32), (0, 32)],
        [(120, 100), (600, 100), (720, 1030), (0, 1030)],
    )
    return HomographyProjection(h)


def test_homography_round_trips_including_perspective() -> None:
    proj = _synthetic_perspective()
    for tile in [(0, 0), (3.5, 6.5), (9, 16), (14.5, 25.5), (9, 29.2), (18, 32)]:
        px = proj.tile_to_pixel(*tile)
        back = proj.pixel_to_tile(*px)
        assert abs(back[0] - tile[0]) < 0.05, (tile, px, back)
        assert abs(back[1] - tile[1]) < 0.05, (tile, px, back)


def test_perspective_differs_from_affine() -> None:
    """The whole point of the homography: tile spacing changes with depth."""
    proj = _synthetic_perspective()
    near = proj.tile_to_pixel(9, 26)[1] - proj.tile_to_pixel(9, 25)[1]
    far = proj.tile_to_pixel(9, 7)[1] - proj.tile_to_pixel(9, 6)[1]
    assert near > far, (near, far)  # near tiles are drawn larger
    assert abs(near - far) > 4, (near, far)


def test_projection_from_landmarks_recovers_reference_points() -> None:
    proj_ref = _synthetic_perspective()
    towers = {name: proj_ref.tile_to_pixel(*tile) for name, tile in PRINCESS_TILES.items()}
    proj, residual = projection_from_landmarks(towers)
    assert residual < 1.5
    for name, tile in PRINCESS_TILES.items():
        got = proj.tile_to_pixel(*tile)
        assert abs(got[0] - towers[name][0]) <= 2
        assert abs(got[1] - towers[name][1]) <= 2


# ======================================================================================
# Arena calibration / grid
# ======================================================================================


def test_calibration_serialisation_round_trip(tmp_path: Path) -> None:
    proj = _synthetic_perspective()
    cal = ArenaCalibration(
        frame_size=(720, 1280),
        projection=proj,
        hand_slots_px=[(224.0, 1162.0), (360.0, 1162.0), (496.0, 1162.0), (631.0, 1162.0)],
        hand_card_size_px=(127.0, 114.0),
        elixir_px=(146.0, 695.0, 1259.0),
        river_px_y=567.0,
        source="test",
        residual_px=1.2,
    )
    from cr_bot.arena.device_calibrator import load_calibration, save_calibration

    path = save_calibration(tmp_path / "cal.json", cal)
    loaded = load_calibration(path)
    assert loaded is not None
    assert isinstance(loaded.projection, HomographyProjection)
    assert loaded.frame_size == (720, 1280)
    assert loaded.hand_slot_x_ratio_at(0) if hasattr(loaded, "hand_slot_x_ratio_at") else True
    assert loaded.river_px_y == pytest.approx(567.0)
    assert loaded.hand_slots_px[0] == pytest.approx((224.0, 1162.0))


def test_grid_uses_measured_hand_slots_and_rescales() -> None:
    cal = ArenaCalibration(
        frame_size=(720, 1280),
        projection=RectProjection(30, 100, 700, 1030),
        hand_slots_px=[(224.0, 1162.0), (360.0, 1162.0), (496.0, 1162.0), (631.0, 1162.0)],
        hand_card_size_px=(127.0, 114.0),
        elixir_px=(146.0, 695.0, 1259.0),
        river_px_y=567.0,
    )
    grid = ArenaGrid(720, 1280, cal)
    assert grid.hand_slot_pixel(0) == (224, 1162)
    assert grid.hand_slot_pixel(3) == (631, 1162)
    assert grid.hand_card_size() == (127, 114)
    assert grid.elixir_geometry() == (146, 695, 1259)

    # Doubling the capture size must double the measured geometry.
    grid.resize(1440, 2560)
    assert grid.hand_slot_pixel(1) == (720, 2324)
    assert grid.elixir_geometry()[0] == 292


def test_legal_deploy_guard_rejects_hud_river_and_enemy_half() -> None:
    grid = ArenaGrid(720, 1280, ArenaCalibration(projection=RectProjection(30, 100, 700, 1030), river_px_y=565.0))
    assert grid.is_legal_deploy_pixel(*grid.tile_to_pixel(3.5, 20.0)) is True
    assert grid.is_legal_deploy_pixel(*grid.tile_to_pixel(3.5, 10.0)) is False   # enemy half
    assert grid.is_legal_deploy_pixel(*grid.tile_to_pixel(9.0, 16.1)) is False   # on the river
    assert grid.is_legal_deploy_pixel(360, 1200) is False                        # card tray
    assert grid.is_legal_deploy_pixel(-5, 400) is False                          # off screen


# ======================================================================================
# Deploy verification
# ======================================================================================


def test_deploy_verifier_detects_the_played_slot_only() -> None:
    grid = ArenaGrid(720, 1280, ArenaCalibration(
        projection=RectProjection(30, 100, 700, 1030),
        hand_slots_px=[(224.0, 1162.0), (360.0, 1162.0), (496.0, 1162.0), (631.0, 1162.0)],
        hand_card_size_px=(127.0, 114.0),
    ))
    frame_a = np.zeros((1280, 720, 3), dtype=np.uint8)
    for i in range(4):
        x1, y1, x2, y2 = grid.hand_slot_rect(i)
        frame_a[y1:y2, x1:x2] = (40 * (i + 1), 90, 160)

    frame_b = frame_a.copy()
    x1, y1, x2, y2 = grid.hand_slot_rect(2)
    frame_b[y1:y2, x1:x2] = (250, 20, 20)  # slot 2 was played -> replaced by the next card

    verifier = DeployVerifier()
    before = verifier.snapshot(frame_a, grid)
    after = verifier.snapshot(frame_b, grid)
    assert before is not None and after is not None
    assert verifier.slot_changed(before, after, 2) is True
    assert verifier.slot_changed(before, after, 0) is False
    assert verifier.changed_slots(before, after) == [2]


def test_deploy_verifier_ignores_a_dropped_tap() -> None:
    grid = ArenaGrid(720, 1280, ArenaCalibration(
        projection=RectProjection(30, 100, 700, 1030),
        hand_slots_px=[(224.0, 1162.0), (360.0, 1162.0), (496.0, 1162.0), (631.0, 1162.0)],
        hand_card_size_px=(127.0, 114.0),
    ))
    frame = np.full((1280, 720, 3), 30, dtype=np.uint8)
    verifier = DeployVerifier()
    before = verifier.snapshot(frame, grid)
    after = verifier.snapshot(frame.copy(), grid)
    assert verifier.changed_slots(before, after) == []
    assert verifier.slot_changed(before, after, 0) is False


# ======================================================================================
# Calibration against a real Clash Royale frame
# ======================================================================================

requires_fixture = pytest.mark.skipif(not FIXTURE.is_file(), reason="real CR frame fixture missing")


@requires_fixture
def test_calibrator_finds_the_river_on_a_real_frame() -> None:
    frame = cv2.imread(str(FIXTURE))
    river = DeviceCalibrator().detect_river(frame)
    assert river is not None
    assert 540 <= river <= 600, river


@requires_fixture
def test_calibrator_finds_the_four_hand_cards_on_a_real_frame() -> None:
    frame = cv2.imread(str(FIXTURE))
    hand = DeviceCalibrator().detect_hand_row(frame)
    assert hand is not None
    xs = [round(c[0]) for c in hand["centers"]]
    assert xs == pytest.approx([224, 360, 496, 631], abs=8), xs
    # Even pitch is the signal that the four real cards were found.
    gaps = np.diff(xs)
    assert gaps.max() - gaps.min() <= 6, gaps
    assert 115 <= hand["card_w"] <= 140
    assert 105 <= hand["card_h"] <= 125


@requires_fixture
def test_calibrator_produces_a_usable_layout_on_a_real_frame() -> None:
    frame = cv2.imread(str(FIXTURE))
    result = DeviceCalibrator().calibrate(frame)
    assert result.confidence >= 0.55, result.report
    cal = result.calibration
    assert cal.projection is not None
    assert cal.hand_slots_px and len(cal.hand_slots_px) == 4
    assert cal.elixir_px is not None and cal.elixir_px[1] - cal.elixir_px[0] > 720 * 0.5

    grid = ArenaGrid(720, 1280, cal)
    # The river in the projection must agree with the river we can see.
    assert abs(grid.river_pixel_y - 567) < 20

    # Bridge placements must sit just below the river and inside the arena.
    for tile in (grid.LEFT_BRIDGE, grid.RIGHT_BRIDGE):
        px = grid.tile_to_pixel(*tile)
        assert grid.is_legal_deploy_pixel(*px), (tile, px)
    # The enemy half must be rejected.
    assert not grid.is_legal_deploy_pixel(*grid.tile_to_pixel(9, 8))


@requires_fixture
def test_calibration_preview_renders() -> None:
    frame = cv2.imread(str(FIXTURE))
    result = DeviceCalibrator().calibrate(frame)
    out = render_calibration_preview(frame, result.calibration, result)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)  # something was actually drawn


@requires_fixture
def test_elixir_reader_on_a_real_frame() -> None:
    frame = cv2.imread(str(FIXTURE))
    result = DeviceCalibrator().calibrate(frame)
    grid = ArenaGrid(720, 1280, result.calibration)
    reader = ElixirReader(result.calibration, grid=grid)
    value = reader.read_elixir(frame)
    # The bar in this frame is a little over half full.
    assert 4.5 <= value <= 7.0, value


def test_elixir_reader_returns_zero_for_an_empty_bar() -> None:
    blank = np.zeros((1280, 720, 3), dtype=np.uint8)
    cal = ArenaCalibration(
        projection=RectProjection(30, 100, 700, 1030), elixir_px=(146.0, 695.0, 1259.0)
    )
    grid = ArenaGrid(720, 1280, cal)
    reader = ElixirReader(cal, grid=grid)
    assert reader.read_elixir(blank) == 0.0
