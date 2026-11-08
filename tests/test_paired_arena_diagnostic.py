"""Transparent still-pair diagnostics and narrow regression tests; not a live benchmark."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cr_bot.arena.device_calibrator import DeviceCalibrator
from cr_bot.arena.grid import ArenaGrid
from cr_bot.models import Side
from cr_bot.vision.unit_detector import (ArenaStructureSuppressor, BlobUnit, BlobUnitDetector,
                                         HealthBar, TeamVote, pixels_per_tile)
from cr_bot.models import WeightClass
from tools.paired_arena_check import match_detections, run

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/paired_battle"


def test_pair_diagnostic_reports_misses_sides_and_extras_together():
    report = run(experimental=False)
    assert report["experimental_overlay_normalization"] is False
    assert report["summary"]["labelled_units"] == 8
    by_name = {c["name"]: c for c in report["cases"]}
    assert by_name["same_selected_frame_with_emote"]["extras_vs_labels"] == 0
    one = by_name["one_unit_with_selection_overlay"]
    assert one["matched_locations"] == 1 and one["correct_sides"] == 1
    assert one["extras_vs_labels"] <= 8
    # Do not silently score suppressed outputs as successes: misses remain visible.
    assert all(c["labelled_units"] == c["matched_locations"] + c["misses"]
               for c in report["cases"])


def test_experimental_pair_diagnostic_requires_real_locations_as_well_as_extras():
    report = run(experimental=True)
    three = next(c for c in report["cases"] if c["name"] == "three_units_without_selection_overlay")
    assert three["matched_locations"] == 3 and three["correct_sides"] == 3
    assert three["extras_vs_labels"] <= 6
    # More recall is not a reliable result if it also creates many extra detections.
    assert report["summary"]["misses"] > 0
    assert report["summary"]["extras_vs_labels"] > 0


def test_matching_is_one_to_one_and_does_not_ignore_wrong_sides():
    det = [SimpleNamespace(pixel_bbox=(0, 0, 20, 20), side=Side.ALLY, name="unknown_medium", source="x")]
    labels = [
        {"name": "a", "side": "enemy", "bbox": [0, 0, 18, 18]},
        {"name": "b", "side": "enemy", "bbox": [2, 2, 20, 20]},
    ]
    matches, misses, extras = match_detections(det, labels, .15)
    assert len(matches) == 1 and matches[0]["side_correct"] is False
    assert len(misses) == 1 and not extras


def test_motion_side_classifier_gets_roi_local_coordinates(monkeypatch):
    detector = BlobUnitDetector()
    grid = ArenaGrid(400, 800)
    roi = np.zeros((200, 200, 3), np.uint8)
    motion = np.zeros((100, 100), np.uint8)
    motion[20:35, 20:35] = 255
    seen = []
    def classify(image, bbox, *args):
        seen.append((image.shape, bbox))
        return TeamVote(Side.ENEMY, 0.8, {})
    monkeypatch.setattr(detector.team, "classify", classify)
    out = detector._add_motion_units([], motion, roi, roi, grid, 20, (50, 100), 0.5)
    assert len(out) == 1 and len(seen) == 1
    _, local = seen[0]
    actual = out[0].bbox
    assert local == (actual[0] - 50, actual[1] - 100, actual[2] - 50, actual[3] - 100)


@pytest.mark.parametrize("x,y", [(40, 325), (525, 325), (235, 105), (235, 745)])
def test_fallback_roi_retains_these_synthetic_edge_sprites(x, y):
    # This verifies the crop boundaries only. A composited library sprite is not a substitute
    # for actual gameplay at the edges of the map.
    base = cv2.imread(str(FIX / "1JWPNVHeVkxZ7Gq5SnqM.jpg"))
    grid = ArenaGrid(608, 1080, DeviceCalibrator().calibrate(base).calibration)
    paths = sorted((ROOT / "assets/unit_sprites/musketeer").glob("*.png"))
    sprite = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    height = 75
    width = round(sprite.shape[1] * height / sprite.shape[0])
    sprite = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
    rgb = np.clip(sprite[:, :, :3].astype(np.float32) * (0.55, 0.45, 1.0) + (30, 10, 70), 0, 255)
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255
    scene = base.copy()
    patch = scene[y:y + height, x:x + width]
    patch[:] = (rgb * alpha + patch * (1 - alpha)).astype(np.uint8)
    detector = BlobUnitDetector()
    for _ in range(6):
        detector.detect(base, grid)
    units = []
    for _ in range(3):
        units = detector.detect(scene, grid)
    assert any(max(0, min(u.bbox[2], x + width) - max(u.bbox[0], x)) *
               max(0, min(u.bbox[3], y + height) - max(u.bbox[1], y)) > .35 * width * height
               for u in units)


def test_wide_tower_bar_does_not_turn_the_tower_into_a_troop():
    grid = ArenaGrid(720, 1280)
    tx, ty = grid.TOWER_TILES["enemy_left"]
    ppt = pixels_per_tile(grid)
    bx, by = grid.tile_to_pixel(tx, ty - .5)
    def unit(bar):
        return BlobUnit(bbox=(bx-20, by, bx+20, by+40), area_px=1000, side=Side.ENEMY,
                        side_confidence=.8, height_tiles=1.5, width_tiles=1.0,
                        tile_x=tx, tile_y=ty, weight_class=WeightClass.MEDIUM,
                        fallback_label="unknown_medium", has_bar=True, bars=[bar])
    wide = HealthBar((int(bx-.95*ppt), int(by-2), int(bx+.95*ppt), int(by+2)),
                     Side.ENEMY, 1.0, int(1.9*ppt), int(1.9*ppt), True, 0.0)
    narrow = HealthBar((int(bx-.4*ppt), int(by-2), int(bx+.4*ppt), int(by+2)),
                       Side.ENEMY, 1.0, int(.8*ppt), int(.8*ppt), True, 0.0)
    kept = ArenaStructureSuppressor().suppress([unit(wide), unit(narrow)], grid)
    assert len(kept) == 1 and kept[0].bars[0] is narrow
