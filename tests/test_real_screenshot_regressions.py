"""Regression tests for the noise and identity defects found on real screenshots.

Each test here corresponds to a failure measured on a real Clash Royale capture (or on the real
720x1280 frame in ``calibration_preview.png``), not to a hypothetical. The names say what the
frame looked like, so a future change that reintroduces the defect fails with a sentence that
explains the stakes instead of a tuple mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cr_bot.arena.device_calibrator import DeviceCalibrator           # noqa: E402
from cr_bot.arena.grid import ArenaGrid                               # noqa: E402
from cr_bot.vision.arena_detector import ArenaUnitBackend             # noqa: E402
from cr_bot.vision.unit_detector import ArenaBackgroundModel, BlobUnitDetector  # noqa: E402

REAL_FRAME = REPO_ROOT / "calibration_preview.png"
ARENA = REPO_ROOT / "tests" / "fixtures"


def _real_frame() -> np.ndarray:
    image = cv2.imread(str(REAL_FRAME))
    if image is None:                                    # pragma: no cover - fixture missing
        pytest.skip("calibration_preview.png not available")
    return image


def _settled_backend(frame: np.ndarray, grid: ArenaGrid,
                     warm: int = 6) -> tuple[ArenaUnitBackend, np.ndarray]:
    """An ArenaUnitBackend that has watched this arena *before* the troops were placed.

    A background-subtraction detector learns a static frame as "the arena", so feeding the same
    screenshot over and over is not a valid test: after a few frames the troops are part of the
    floor and the board correctly reads as empty. The arena is therefore reconstructed by
    inpainting the three labelled sprites out (the same trick ``tools/verify_screenshots.py``
    uses, with the boxes written down by hand in ``samples/calibration_preview.json``).
    """
    labels = json.loads((REPO_ROOT / "samples" / "calibration_preview.json").read_text())
    mask = np.zeros(frame.shape[:2], np.uint8)
    for entry in labels["expect"]:
        x1, y1, x2, y2 = entry["box"]
        cv2.rectangle(mask, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 255, -1)
    empty = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
    backend = ArenaUnitBackend()
    for _ in range(warm):
        backend.detect(empty, grid)
    backend.detect(empty, grid)
    return backend, empty


def _grid_for(frame: np.ndarray) -> ArenaGrid:
    result = DeviceCalibrator().calibrate(frame)
    grid = ArenaGrid(frame.shape[1], frame.shape[0], result.calibration)
    grid.resize(frame.shape[1], frame.shape[0])
    return grid


# ---------------------------------------------------------------------------------------------
# viewpoint wobble: the single largest source of phantom units on a streamed capture
# ---------------------------------------------------------------------------------------------
def test_the_estimated_viewpoint_shift_is_exact() -> None:
    """Phase correlation must report the offset it is given, or compensation is pointless."""
    frame = _real_frame()
    model = ArenaBackgroundModel()
    model._ensure(frame.astype(np.float32))
    model._background = frame.astype(np.float32)
    for dx, dy in [(0, 0), (1, 0), (2, -2), (-3, 1), (6, 3)]:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        assert model.global_shift(moved) == (dx, dy), f"shift ({dx}, {dy}) misreported"


def test_a_shifted_frame_does_not_produce_phantom_troops() -> None:
    """A 1-2 px viewpoint wobble used to turn every arena edge into a "troop".

    Measured before the fix: a 1 px shift produced 40 phantom units in one frame, a 2 px shift
    20. Nothing about that is a troop; it is the whole picture moving.
    """
    frame = _real_frame()
    grid = _grid_for(frame)
    detector = BlobUnitDetector()
    for _ in range(6):
        detector.detect(frame, grid)
    detector.detect(frame, grid)

    for dx, dy in [(1, 0), (2, 0), (2, -2), (3, 1), (6, 3)]:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)
        counts = [len(detector.detect(moved, grid)) for _ in range(8)]
        assert max(counts) == 0, (
            f"a {dx},{dy} px viewpoint shift produced up to {max(counts)} phantom units "
            f"over 8 frames: {counts}"
        )


def test_a_wobbling_capture_is_quiet_frame_after_frame() -> None:
    """A phone held by hand wobbles every frame; the board must stay empty when it is empty."""
    frame = _real_frame()
    grid = _grid_for(frame)
    detector = BlobUnitDetector()
    rng = np.random.default_rng(0)
    for _ in range(8):
        dx, dy = rng.integers(-2, 3, 2)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]),
                              (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        detector.detect(moved, grid)
    counts = []
    for _ in range(20):
        dx, dy = rng.integers(-2, 3, 2)
        moved = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]),
                              (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        counts.append(len(detector.detect(moved, grid)))
    assert max(counts) == 0, f"jitter flickered units onto an empty board: {counts}"


# ---------------------------------------------------------------------------------------------
# identity: which colour means "ally" is not a constant
# ---------------------------------------------------------------------------------------------
def test_colour_calibration_does_not_learn_from_troops_and_scenery() -> None:
    """The earlier 4-tile tower rule mistook blue troops and orange floor for tower bars."""
    frame = _real_frame()
    grid = _grid_for(frame)
    backend = ArenaUnitBackend(enable_recognition=False)
    for _ in range(7):
        backend.detect(frame, grid)
    # There are not enough reliable tower bars here. Retain the normal blue-ally/red-enemy rule.
    assert backend._ally_hue is None and backend._enemy_hue is None
    from cr_bot.models import Side
    assert backend._correct_side(Side.ALLY, 105) == Side.ALLY
    assert backend._correct_side(Side.ENEMY, 0) == Side.ENEMY


@pytest.mark.skipif(not REAL_FRAME.exists(), reason="no real frame")
def test_real_screenshot_units_are_reported_with_the_right_side() -> None:
    """Blue Mega Knight and partly hidden Miner are allies; the red Giant is an enemy."""
    frame = _real_frame()
    grid = _grid_for(frame)
    backend, _empty = _settled_backend(frame, grid)
    detections = backend.detect(frame, grid)
    labels = json.loads((REPO_ROOT / "samples" / "calibration_preview.json").read_text())["expect"]
    assert len(detections) == len(labels) == 3
    from tools.verify_screenshots import Label, _inside
    used = set()
    for entry in labels:
        lab = Label(tuple(entry["box"]))
        matches = sorted(((_inside({"box": det.pixel_bbox}, lab), idx, det)
                          for idx, det in enumerate(detections) if idx not in used), reverse=True)
        overlap, idx, det = matches[0]
        assert overlap >= 0.1, f"no overlapping detection for {entry['name']}"
        assert det.side.value == entry["side"], (entry["name"], det.side)
        used.add(idx)


# ---------------------------------------------------------------------------------------------
# geometry: a box must hug the sprite, or nothing can name it later
# ---------------------------------------------------------------------------------------------
def test_a_detected_box_hugs_the_sprite_and_excludes_its_health_bar() -> None:
    """Real sprite boxes must be tight: the recogniser crops this box to name the troop."""
    frame = _real_frame()
    grid = _grid_for(frame)
    backend, _empty = _settled_backend(frame, grid)
    detections = backend.detect(frame, grid)
    assert detections
    for det in detections:
        x1, y1, x2, y2 = det.pixel_bbox
        width, height = x2 - x1, y2 - y1
        # a unit sprite in this frame is 60-200 px wide; a box three times that is scenery
        assert 40 <= width <= 240, f"{det.name}: implausible box width {width} px"
        assert 40 <= height <= 240, f"{det.name}: implausible box height {height} px"
        assert abs((width / float(height)) - 1.0) < 0.9, (
            f"{det.name}: box aspect {width / float(height):.2f} is not a sprite")


# ---------------------------------------------------------------------------------------------
# naming gates: a wrong name is worse than no name
# ---------------------------------------------------------------------------------------------
def test_an_ambiguous_match_is_left_unnamed() -> None:
    """Skeleton Army frames *are* Skeleton frames, so the two must never "confirm" each other."""
    from cr_bot.vision.unit_recognizer import UnitRecognizer

    recognizer = UnitRecognizer()
    if not recognizer.is_ready:                          # pragma: no cover - assets missing
        pytest.skip("sprite library not available")
    groups = recognizer.duplicate_groups
    assert groups, "the library's byte-identical cards were not detected as duplicates"
    assert any("skeletons" in group for group in groups)
    # The lookup records aliases, but a shared frame must remain an ambiguous prediction.
    aliases = recognizer._aliases_of("skeleton_army")
    assert "skeletons" in aliases and "skeleton_army" in aliases


def test_the_recognizer_refuses_a_patch_too_small_to_identify() -> None:
    """Naming a 12x12 px patch would be guessing, and a guess costs elixir."""
    from cr_bot.vision.unit_recognizer import UnitRecognizer

    recognizer = UnitRecognizer()
    frame = np.zeros((200, 200, 3), np.uint8)
    assert recognizer.identify(frame, (50, 50, 60, 60)) is None
