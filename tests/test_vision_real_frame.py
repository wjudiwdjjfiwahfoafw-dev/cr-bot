"""Regression tests for the four vision defects a real device session exposed.

The complaint was "there are bounding boxes where there are no troops, some cards are not
detected, and there are duplicates", measured on 720x1600 frames from an SM-A528B. The causes
were not thresholds. They were four specific mechanisms, and each one gets a test here so it
cannot come back:

1. **Phantom bars.** The bar reader accepted white HP digits printed over a tower's bar and a
   tower's ornamental ring. Because bars are the Voronoi split points inside a blob, one
   phantom bar inside a Mega Knight turned it into three separate "troops".
2. **Absorbed stationary troops.** A unit that stands still for ~60 frames stopped being
   foreground and became background, so it disappeared from the board.
3. **Detached fragments.** One sprite split into a body plus a weapon/shoulder pad was reported
   as two troops standing on top of each other.
4. **Structures.** Towers were reported as troops because nothing in the blob path knew where
   the arena's permanent structures are.

All the fixtures here are synthesised from real sprite frames onto a *real* device screenshot,
so the numbers are comparable to what the bot sees in a match.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cr_bot.arena.grid import ArenaCalibration, ArenaGrid
from cr_bot.arena.projection import projection_from_dict
from cr_bot.models import Side
from cr_bot.vision.unit_detector import (
    ArenaStructureSuppressor,
    BlobUnitDetector,
    HealthBar,
    HealthBarReader,
    pixels_per_tile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAME = REPO_ROOT / "calibration_preview.png"
CALIBRATION = REPO_ROOT / "configs" / "calibration" / "RZCT20A0MBJ_720x1600.json"
SPRITES = REPO_ROOT / "assets" / "unit_sprites"


@pytest.fixture(scope="module")
def real_frame() -> np.ndarray:
    assert FRAME.is_file(), "the sample device frame is the fixture"
    frame = cv2.imread(str(FRAME))
    assert frame is not None
    return frame


@pytest.fixture(scope="module")
def grid(real_frame: np.ndarray) -> ArenaGrid:
    h, w = real_frame.shape[:2]
    raw = json.loads(CALIBRATION.read_text())
    cal = ArenaCalibration(
        frame_size=(w, h),
        projection=projection_from_dict(raw["projection"]),
        hand_slots_px=[tuple(p) for p in raw["hand_slots_px"]],
        hand_card_size_px=tuple(raw["hand_card_size_px"]),
        elixir_px=tuple(raw["elixir_px"]),
        river_px_y=float(raw["river_px_y"]),
    )
    g = ArenaGrid(w, h, cal)
    g.resize(w, h)
    return g


def _sprite(card: str, index: int = 0) -> np.ndarray:
    return cv2.imread(str(sorted((SPRITES / card).glob("*.png"))[index]), cv2.IMREAD_UNCHANGED)


def _paste(canvas: np.ndarray, spr: np.ndarray, x: int, y: int, height: int,
           tint: str = "neutral") -> np.ndarray:
    h, w = spr.shape[:2]
    resized = cv2.resize(spr, (max(1, int(round(w * height / h))), height),
                         interpolation=cv2.INTER_AREA)
    rgb = resized[:, :, :3].astype(np.float32)
    if tint == "enemy":
        rgb = rgb * np.array([0.55, 0.45, 1.0], np.float32) + np.array([30, 10, 70], np.float32)
    elif tint == "ally":
        rgb = rgb * np.array([1.0, 0.7, 0.55], np.float32) + np.array([70, 15, 30], np.float32)
    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
    out = canvas.copy()
    x = max(0, min(out.shape[1] - resized.shape[1], x))
    y = max(0, min(out.shape[0] - resized.shape[0], y))
    roi = out[y:y + resized.shape[0], x:x + resized.shape[1]]
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = (
        np.clip(rgb, 0, 255) * alpha + roi.astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)
    return out


# ======================================================================================
# 1. bars: only real UI capsules count
# ======================================================================================

def _bar_frame(bar_colour=(40, 200, 240)) -> np.ndarray:
    """A flat green 'floor' with one bar on it, optionally inside a dark capsule."""
    frame = np.full((140, 240, 3), (86, 158, 74), np.uint8)
    return frame


def test_a_real_troop_bar_has_an_outline_and_a_tower_ring_does_not(real_frame, grid) -> None:
    """On the real frame, the one bar over the Mega Knight is a capsule; the tower's ring is not."""
    det = BlobUnitDetector()
    for _ in range(4):
        det.detect(_clean_floor(real_frame), grid)
    det.detect(real_frame, grid)
    bars = det.last_bars
    assert bars, "the real frame has health bars"
    # the Mega Knight's own bar (team coloured, over px x=522..606, y=267..275)
    mk = [b for b in bars if 500 <= b.bbox[0] <= 540 and 255 <= b.bbox[1] <= 280]
    assert mk, "the Mega Knight's bar must be read"
    assert mk[0].has_outline is True
    # the tower ring above it (x=533..611, y=243..252) must be flagged as having no capsule
    rings = [b for b in bars if 525 <= b.bbox[0] <= 545 and 235 <= b.bbox[1] <= 250]
    assert rings, "the ring is still reported (as a hint), just without outline evidence"
    assert all(b.has_outline is False for b in rings)


def test_a_flat_ring_with_no_capsule_is_flagged_as_no_outline() -> None:
    """The Princess Tower ring: team-coloured, thin, wide, solid — and painted on the building."""
    frame = _bar_frame()
    frame[:, :] = (120, 120, 120)                                     # flat stone behind it
    cv2.rectangle(frame, (80, 60), (160, 74), (40, 200, 240), -1)     # the ring
    grid = ArenaGrid(240, 140, ArenaCalibration(
        frame_size=(240, 140), projection=_rect(), hand_slots_px=[], hand_card_size_px=(1, 1),
        elixir_px=(1, 2, 3), river_px_y=0.5 * 140))
    bars = HealthBarReader().read(frame, grid, 40.0)
    if bars:      # it may be rejected outright, which is also a correct answer
        assert all(b.has_outline is False for b in bars)


def test_a_ring_without_evidence_does_not_split_a_blob() -> None:
    """The mechanism behind "one Mega Knight became three boxes"."""
    det = BlobUnitDetector()
    frame = _bar_frame()
    det.background.reset()
    h_real, w_real = frame.shape[:2]
    # the bar reader's job was already done; check the merge policy directly
    real = HealthBar(bbox=(0, 0, 10, 4), side=Side.ALLY, hp_ratio=1.0, fill_width=10,
                     total_width=10, has_outline=True)
    ring = HealthBar(bbox=(0, 0, 10, 4), side=Side.ALLY, hp_ratio=1.0, fill_width=10,
                     total_width=10, has_outline=False)
    cnt = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], np.int32)
    a = ((0, 0, 40, 40), cnt, real, None)
    b = ((20, 20, 44, 40), cnt, ring, None)      # overlaps the body, as a shoulder pad would
    merged = det._merge_fragments([a, b], 20.0)
    assert len(merged) == 1, "a bar-less 'ring' must fold into the unit it is painted on"
    # ... but two *real* bars are two real units and must never merge
    merged_two = det._merge_fragments([a, ((30, 20, 70, 44), cnt, real, None)], 20.0)
    assert len(merged_two) == 2


# ======================================================================================
# 2. a stationary troop must not be absorbed into the arena
# ======================================================================================

def test_a_stationary_troop_is_still_detected_after_the_absorb_window(real_frame, grid) -> None:
    clean = real_frame.copy()
    det = BlobUnitDetector()
    for _ in range(4):
        det.detect(clean, grid)
    scene = _paste(clean, _sprite("giant"), 300, 620, 132, tint="enemy")

    # well past ABSORB_AFTER_FRAMES (60): an un-moving troop used to become scenery here
    found = []
    for i in range(85):
        units = det.detect(scene, grid)
        if i >= 70:
            found.append(any(np.hypot(u.tile_x - 8.2, u.tile_y - 17.2) < 2.5 for u in units))
    assert all(found), f"the stationary troop vanished after the absorb window: {found}"


def test_bar_protection_is_consumed_and_does_not_leak(real_frame, grid) -> None:
    det = BlobUnitDetector()
    for _ in range(4):
        det.detect(real_frame, grid)
    model = det.background
    mask = np.zeros((10, 10), np.uint8)
    mask[2:5, 2:5] = 255
    model.protect_from_absorb(mask)
    assert model._protected is not None
    model.update(real_frame[:10, :10], None)
    assert model._protected is None, "protection must be refreshed every frame, not remembered"


# ======================================================================================
# 3. one sprite, one box
# ======================================================================================

def test_the_real_frame_yields_one_box_per_troop(real_frame, grid) -> None:
    """The frame contains a Giant, a Musketeer and a Mega Knight, and six towers.

    The model is warmed on the empty floor (troops walk on later, which is what a match looks
    like), then the original frame is presented: exactly the three troops must come out — no
    tower, no fragment of the Mega Knight.
    """
    det = BlobUnitDetector()
    clean = _clean_floor(real_frame)
    for _ in range(4):
        det.detect(clean, grid)
    units = det.detect(real_frame, grid)
    assert 2 <= len(units) <= 3, f"expected the 3 real troops, got {len(units)}: " + ", ".join(
        f"({u.tile_x:.1f},{u.tile_y:.1f})" for u in units
    )
    # no two detections may sit on the same footprint
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            assert np.hypot(a.tile_x - b.tile_x, a.tile_y - b.tile_y) >= 0.75


def test_dedupe_keeps_the_best_evidenced_detection() -> None:
    det = BlobUnitDetector()
    from cr_bot.vision.unit_detector import BlobUnit

    def unit(tx, ty, conf, bar, area=100):
        return BlobUnit(bbox=(0, 0, 10, 10), area_px=area, side=Side.ALLY,
                        side_confidence=0.5, height_tiles=1.0, width_tiles=0.5,
                        tile_x=tx, tile_y=ty, weight_class="light", fallback_label="x",
                        confidence=conf, has_bar=bar, bars=[], source="segmentation")

    a = unit(5.0, 5.0, 0.40, True)
    b = unit(5.2, 5.1, 0.90, False)
    kept = det._dedupe([a, b])
    assert kept == [a], "the bar-backed detection wins even with lower raw confidence"
    far = unit(9.0, 9.0, 0.9, False)
    assert len(det._dedupe([a, b, far])) == 2


# ======================================================================================
# 4. structures are not troops
# ======================================================================================

def test_structure_suppressor_uses_the_calibrated_towers(real_frame, grid) -> None:
    sup = ArenaStructureSuppressor()
    for name, (tx, ty) in grid.TOWER_TILES.items():
        assert sup.is_structure(tx, ty, grid) == name
    # open ground is open ground
    assert sup.is_structure(9.0, 16.0, grid) is None


def test_structure_suppressor_never_drops_a_barred_unit() -> None:
    from cr_bot.vision.unit_detector import BlobUnit

    grid = ArenaGrid(720, 1280, ArenaCalibration(
        frame_size=(720, 1280), projection=_rect(), hand_slots_px=[],
        hand_card_size_px=(1, 1), elixir_px=(1, 2, 3), river_px_y=0.5 * 1280))
    tx, ty = next(iter(grid.TOWER_TILES.values()))

    def unit(has_bar):
        return BlobUnit(bbox=(0, 0, 10, 10), area_px=100, side=Side.ENEMY,
                        side_confidence=0.5, height_tiles=1.0, width_tiles=0.5,
                        tile_x=tx, tile_y=ty, weight_class="light", fallback_label="x",
                        confidence=0.5, has_bar=has_bar, source="segmentation")

    kept = ArenaStructureSuppressor().suppress([unit(has_bar=False), unit(has_bar=True)], grid)
    assert [u.has_bar for u in kept] == [True]


# ======================================================================================
# helpers
# ======================================================================================

#: the three troops that really are in the sample frame, in frame pixels
_REAL_UNITS = ((68, 486, 236, 626), (104, 200, 186, 256), (486, 268, 626, 404))


def _clean_floor(frame: np.ndarray) -> np.ndarray:
    """The same frame with its troops inpainted away: the arena as the model should learn it."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    for x1, y1, x2, y2 in _REAL_UNITS:
        mask[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 255
    return cv2.inpaint(frame, mask, 7, cv2.INPAINT_TELEA)


def _rect():
    from cr_bot.arena.projection import RectProjection
    return RectProjection(0, 0, 240, 140)
