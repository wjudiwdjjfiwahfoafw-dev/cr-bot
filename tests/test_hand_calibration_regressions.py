"""Hand/calibration regressions using attributed stills, not live-accuracy claims."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cr_bot.arena.device_calibrator import (
    AUTO_CALIBRATION_REVISION, DeviceCalibrator, load_calibration, save_calibration,
)
from cr_bot.arena.grid import ArenaCalibration, ArenaGrid
from cr_bot.bot_runner import ClashRoyaleBotRunner
from cr_bot.config.loader import BotConfig, load_config
from cr_bot.models import BattleState, GamePhase
from cr_bot.vision.hand_detector import HandCardDetector
from cr_bot.vision.unit_detector import (
    BG_BACKGROUND, BG_FOREGROUND, BG_UNKNOWN, BlobUnitDetector,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/paired_battle"
HOG = ["hog_rider", "skeletons", "cannon", "ice_spirit", "musketeer", "ice_golem", "fireball", "the_log"]
BAIT = ["mega_knight", "miner", "inferno_dragon", "skeleton_barrel", "zap", "bats", "spear_goblins", "goblin_gang"]
TEMPLATES = ROOT / "assets/card_templates"


def frame(name):
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None
    return image


def hand(frame_bgr, deck=HOG, cal=None, elixir=10):
    h, w = frame_bgr.shape[:2]
    cal = cal or DeviceCalibrator().calibrate(frame_bgr).calibration
    return HandCardDetector(ArenaGrid(w, h, cal), deck, TEMPLATES).detect_hand(frame_bgr, elixir)


@pytest.mark.parametrize("name,expected", [
    ("0RqoBtTgewOXP37lNwMI.jpg", ["fireball", "skeletons", "ice_golem", "hog_rider"]),
    ("04FED5slLByee0jMFkCR.jpg", ["ice_spirit", "skeletons", "cannon", "musketeer"]),
    ("0J09bneS0qzybsMz6ypX.jpg", ["ice_spirit", "skeletons", "cannon", "musketeer"]),
    ("0JPlKsDPop5jfLGdqEya.jpg", ["fireball", "skeletons", "cannon", "musketeer"]),
    ("12xkBm52RVK8zKqZyGRS.jpg", ["ice_spirit", "skeletons", "cannon", "musketeer"]),
    ("1EiRDae0M8B3GTXWGvFD.jpg", ["ice_spirit", "skeletons", "cannon", "musketeer"]),
])
def test_labelled_hands_including_raised_cards(name, expected):
    slots = hand(frame(name))
    assert [s.card_name for s in slots] == expected
    assert all(s.is_ready for s in slots)


def test_empty_slot_is_unknown_and_never_ready():
    slots = hand(frame("0G5tHKUczboymR0A6Vdu.jpg"))
    assert [s.card_name for s in slots] == ["ice_spirit", "unknown", "cannon", "musketeer"]
    assert not slots[1].is_ready


def test_grey_cards_do_not_become_ready_and_can_abstain():
    calibration = DeviceCalibrator().calibrate(frame("0RqoBtTgewOXP37lNwMI.jpg")).calibration
    grey = frame("17fmmbFkR9sklKs725OJ.jpg")
    assert DeviceCalibrator().detect_hand_row(grey) is None  # only two visible pink badges
    slots = hand(grey, cal=calibration, elixir=10)
    assert [s.card_name for s in slots] == ["ice_spirit", "skeletons", "cannon", "unknown"]
    assert not slots[2].is_ready and not slots[3].is_ready


def test_off_deck_art_is_not_forced_into_the_deck():
    image = frame("0JP5Zs9NiQk4rwMjXHGh.jpg")  # Archers / Fireball / Knight / Skeleton Army
    assert [s.card_name for s in hand(image, HOG)] == ["unknown", "fireball", "unknown", "unknown"]
    slots = hand(image, BAIT)
    assert all(s.card_name == "unknown" and not s.is_ready for s in slots)


def test_template_prefixes_cannot_load_an_off_deck_card_as_giant(tmp_path):
    img = np.full((100, 80, 3), (10, 40, 80), np.uint8)
    for name in ("giant.png", "giant_skeleton.png", "giant_skeleton_extra.png"):
        assert cv2.imwrite(str(tmp_path / name), img)
    only_giant = HandCardDetector(ArenaGrid(), ["giant"], tmp_path)
    assert len(only_giant._templates["giant"]) == 1
    both = HandCardDetector(ArenaGrid(), ["giant", "giant_skeleton"], tmp_path)
    assert len(both._templates["giant"]) == 1
    assert len(both._templates["giant_skeleton"]) == 2


def test_three_badges_fit_and_next_card_is_not_first_slot():
    image = frame("0G5tHKUczboymR0A6Vdu.jpg")
    row = DeviceCalibrator().detect_hand_row(image)
    assert row is not None and row["observed_badges"] == 3
    assert [x for x, _ in row["centers"]] == pytest.approx([190, 304, 419, 533], abs=4)
    other = DeviceCalibrator().detect_hand_row(frame("0JP5Zs9NiQk4rwMjXHGh.jpg"))
    assert other is not None
    assert [x for x, _ in other["centers"]] == pytest.approx([190, 305, 419, 534], abs=4)


@pytest.mark.parametrize("scale", [.33, .5, 1.0, 1.5, 2.0])
def test_hand_badge_size_scales_with_capture_resolution(scale):
    image = frame("0RqoBtTgewOXP37lNwMI.jpg")
    scaled = cv2.resize(image, None, fx=scale, fy=scale)
    row = DeviceCalibrator().detect_hand_row(scaled)
    assert row is not None
    assert [x / scale for x, _ in row["centers"]] == pytest.approx([191, 305, 419, 533], abs=5)


@pytest.mark.parametrize("hue", [100, 150, 18])
def test_river_accepts_narrow_blue_magenta_or_lava_band(hue):
    hsv = np.full((800, 500, 3), (60, 100, 120), np.uint8)
    hsv[315:338, 40:460] = (hue, 240, 230)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert DeviceCalibrator().detect_river(image) == pytest.approx(326, abs=3)


def test_river_rejects_a_broad_coloured_floor():
    hsv = np.full((800, 500, 3), (100, 190, 200), np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    assert DeviceCalibrator().detect_river(image) is None


def test_floodfill_handles_foreground_in_top_left_corner_without_filling_board():
    state = np.full((30, 30), BG_BACKGROUND, np.uint8)
    state[0, 0] = BG_FOREGROUND
    state[8:19, 8:19] = BG_FOREGROUND
    state[11:16, 11:16] = BG_UNKNOWN  # enclosed hole
    state[24:26, 24:26] = BG_UNKNOWN  # outside the object
    mask = BlobUnitDetector()._detection_mask(state, np.zeros_like(state))
    assert mask[0, 0] == 255 and mask[13, 13] == 255
    assert mask[24, 24] == 0 and mask[3, 20] == 0
    assert np.count_nonzero(mask) < 150


def test_roi_only_clips_fallback_segmentation_not_projection():
    image = frame("0RqoBtTgewOXP37lNwMI.jpg")
    cal = DeviceCalibrator().calibrate(image).calibration
    grid = ArenaGrid(image.shape[1], image.shape[0], cal)
    before = grid.arena_pixel_bounds
    roi = BlobUnitDetector._arena_roi_bounds(image, grid)
    assert roi != before and grid.arena_pixel_bounds == before
    assert roi[0] <= 40 and roi[2] >= 565
    # Manual geometry must not be silently clipped by the fallback heuristic.
    cal.source = "manual"
    assert BlobUnitDetector._arena_roi_bounds(image, grid) == before


def test_old_auto_cache_is_remeasured_and_manual_cache_is_preserved(tmp_path):
    image = frame("0RqoBtTgewOXP37lNwMI.jpg")
    cfg = BotConfig(calibration_dir=str(tmp_path), identity_dir=str(tmp_path / "id"),
                    yolo_weights_path=None, enable_unit_recognition=False,
                    templates_dir=str(TEMPLATES))
    bot = ClashRoyaleBotRunner(cfg, show_hud=False, dry_run=True, manual_input=False)
    bot.adb.frame_size = (image.shape[1], image.shape[0])
    old = DeviceCalibrator().calibrate(image).calibration
    old.auto_revision = 0
    old.hand_slots_px = [(10., 10.)] * 4
    save_calibration(bot._calibration_file(), old)
    assert bot.ensure_calibration(image)
    assert bot.calibration.auto_revision == AUTO_CALIBRATION_REVISION
    assert bot.calibration.hand_slots_px[0][0] > 100
    assert load_calibration(bot._calibration_file()).auto_revision == AUTO_CALIBRATION_REVISION

    manual = ArenaCalibration.from_dict(bot.calibration.to_dict())
    manual.source, manual.auto_revision = "manual", 0
    manual.hand_slots_px[0] = (201., 960.)
    save_calibration(bot._calibration_file(), manual)
    bot._calibrated_this_match = False
    bot._last_calibration_attempt_ts = -999.0
    assert bot.ensure_calibration(image)
    assert bot.calibration.source == "manual" and bot.calibration.hand_slots_px[0] == (201., 960.)


def test_incomplete_auto_calibration_is_not_saved_and_is_retried(tmp_path):
    image = frame("17fmmbFkR9sklKs725OJ.jpg")
    cfg = BotConfig(calibration_dir=str(tmp_path), identity_dir=str(tmp_path / "id"),
                    yolo_weights_path=None, enable_unit_recognition=False,
                    templates_dir=str(TEMPLATES))
    bot = ClashRoyaleBotRunner(cfg, show_hud=False, dry_run=True, manual_input=False)
    bot.adb.frame_size = (image.shape[1], image.shape[0])
    assert not bot.ensure_calibration(image)
    assert not bot._calibrated_this_match
    assert not bot._calibration_file().exists()
    bot._last_calibration_attempt_ts = -999.0  # next eligible attempt
    assert bot.ensure_calibration(frame("0RqoBtTgewOXP37lNwMI.jpg"))
    assert bot._calibration_file().exists()


def test_calibration_revision_round_trips_and_scales():
    cal = ArenaCalibration(frame_size=(100, 200), source="auto", auto_revision=7,
                           hand_slots_px=[(10, 20)] * 4)
    restored = ArenaCalibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert restored.auto_revision == 7 and restored.scaled(200, 400).auto_revision == 7
    assert ArenaCalibration.from_dict({"source": "auto"}).auto_revision == 0


def test_no_brain_decision_on_missing_or_just_rebuilt_calibration(tmp_path, monkeypatch):
    cfg = BotConfig(calibration_dir=str(tmp_path), identity_dir=str(tmp_path / "id"),
                    yolo_weights_path=None, enable_unit_recognition=False,
                    templates_dir=str(TEMPLATES), loop_hz=10000)
    bot = ClashRoyaleBotRunner(cfg, show_hud=False, dry_run=True, manual_input=False)
    image = np.zeros((120, 80, 3), dtype=np.uint8)
    bot.adb.frame_size = (80, 120)
    monkeypatch.setattr(bot.adb, "connect", lambda: None)
    monkeypatch.setattr(bot.adb, "close", lambda: None)
    monkeypatch.setattr(bot, "_start_deploy_worker", lambda: None)
    index = {"value": 0}
    def capture(timeout=1.0):
        index["value"] += 1
        if index["value"] > 3:
            raise KeyboardInterrupt
        return image
    monkeypatch.setattr(bot.adb, "capture_frame", capture)
    monkeypatch.setattr(bot, "process_single_frame", lambda image, ts: BattleState(
        timestamp=float(index["value"]), phase=GamePhase.BATTLE_1X, elixir=5, hand=[]))
    def ensure(image):
        if index["value"] == 1:
            return False
        bot._calibrated_this_match = True
        return True
    monkeypatch.setattr(bot, "ensure_calibration", ensure)
    decided = []
    monkeypatch.setattr(bot.brain, "decide", lambda state: decided.append(state.timestamp))
    bot.run_forever()
    assert decided == [3.0]  # neither the uncalibrated frame nor the rebuild frame


def test_experimental_overlay_is_opt_in(tmp_path):
    assert BotConfig().experimental_overlay_normalization is False
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text("vision:\n  experimental_overlay_normalization: true\n")
    cfg = load_config(cfgfile)
    assert cfg.experimental_overlay_normalization is True
    assert BlobUnitDetector().experimental_overlay_normalization is False
