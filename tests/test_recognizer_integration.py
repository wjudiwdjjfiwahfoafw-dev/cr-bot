"""Integration and safety checks for optional, conservative sprite naming."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cr_bot.arena.grid import ArenaGrid
from cr_bot.models import MoveType, Side, WeightClass
from cr_bot.vision.arena_detector import ArenaUnitBackend, HybridArenaDetector, RawDetection
from cr_bot.vision.sprite_matcher import SpriteHit, SpriteTemplate
from cr_bot.vision.unit_detector import BlobUnit, HealthBar, pixels_per_tile
from cr_bot.vision.unit_recognizer import Recognition, UnitRecognizer

ROOT = Path(__file__).resolve().parents[1]


def raw(name="unknown_medium", source="segmentation", side=Side.ENEMY, x=6.0,
        box=(40, 40, 100, 120), confidence=0.68):
    return RawDetection(name, side, x, 10.0, box, confidence, 1.0,
                        MoveType.GROUND, WeightClass.MEDIUM, source)


def blob(box, confidence=0.68):
    return BlobUnit(bbox=box, area_px=(box[2]-box[0])*(box[3]-box[1]), side=Side.ENEMY,
                    side_confidence=0.9, height_tiles=2.0, width_tiles=1.5,
                    tile_x=6.0, tile_y=10.0, weight_class=WeightClass.MEDIUM,
                    fallback_label="unknown_medium", has_bar=True, confidence=confidence)


class FakeRecognizer:
    is_ready = True

    def __init__(self, card=None, raises=False):
        self.card = card
        self.raises = raises
        self.calls = []

    def identify(self, frame, box, has_bar=True):
        self.calls.append((box, has_bar))
        if self.raises:
            raise RuntimeError("broken recognizer")
        return Recognition(self.card, 0.95, 0.60, 0.35, box) if self.card else None


def install_blob(monkeypatch, detector, item):
    backend = detector.cv_fallback if isinstance(detector, HybridArenaDetector) else detector
    monkeypatch.setattr(backend.segmenter, "detect", lambda frame, grid: [item])
    monkeypatch.setattr(backend, "_detect_bars_only", lambda frame, grid: [])


def test_injected_recognizer_names_a_detected_blob_and_retains_detection_confidence(monkeypatch):
    recognizer = FakeRecognizer("baby_dragon")
    detector = HybridArenaDetector(ArenaGrid(240, 240), recognizer=recognizer)
    box = (40, 40, 100, 120)
    install_blob(monkeypatch, detector, blob(box))
    units = detector.detect_and_track(np.zeros((240, 240, 3), np.uint8), 1.0)
    assert recognizer.calls == [(box, False)]  # no bar overlaps the measured sprite crop
    assert len(units) == 1
    assert units[0].name == "baby_dragon"
    assert units[0].source == "segmentation+sprite"
    assert units[0].move_type == MoveType.AIR
    assert units[0].confidence == pytest.approx(0.68)  # classification score isn't detection confidence


def test_recognition_can_be_disabled_end_to_end(monkeypatch):
    recognizer = FakeRecognizer("baby_dragon")
    detector = HybridArenaDetector(ArenaGrid(240, 240), recognizer=recognizer,
                                   enable_recognition=False)
    install_blob(monkeypatch, detector, blob((40, 40, 100, 120)))
    units = detector.detect_and_track(np.zeros((240, 240, 3), np.uint8), 1.0)
    assert units[0].name == "unknown_medium"
    assert not recognizer.calls


def test_real_sprite_asset_can_reach_the_tracker_as_a_name(monkeypatch):
    """Positive synthetic smoke test: a library sprite pasted onto a real arena, NOT live recall."""
    sprite_path = sorted((ROOT / "assets/unit_sprites/mega_knight").glob("*.png"))[0]
    sprite = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
    background = cv2.imread(str(ROOT / "tests/fixtures/real_frames/real_solo_battle_empty.png"))
    assert sprite is not None and background is not None
    height = 100
    sprite = cv2.resize(sprite, (round(sprite.shape[1] * height / sprite.shape[0]), height))
    h, w = sprite.shape[:2]
    patch = background[350:350+h, 350:350+w].copy()
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
    patch = (sprite[:, :, :3] * alpha + patch * (1.0 - alpha)).astype(np.uint8)
    detector = HybridArenaDetector(ArenaGrid(w, h), recognizer=UnitRecognizer())
    install_blob(monkeypatch, detector, blob((0, 0, w, h)))
    units = detector.detect_and_track(patch, 1.0)
    assert len(units) == 1
    assert units[0].name == "mega_knight"
    assert units[0].source == "segmentation+sprite"


def test_naming_rotates_across_candidates_and_skips_estimated_bar_boxes():
    recognizer = FakeRecognizer()
    backend = ArenaUnitBackend(recognizer=recognizer)
    frame = np.zeros((250, 250, 3), np.uint8)
    boxes = [(10, 40, 60, 100), (70, 40, 120, 100), (130, 40, 180, 100)]
    for _ in range(3):
        items = [raw(box=b) for b in boxes] + [raw(source="bars", box=(180, 40, 240, 100))]
        backend._name_troops(items, frame)
    assert [call[0] for call in recognizer.calls] == boxes
    assert all(not has_bar for _, has_bar in recognizer.calls)


def test_naming_failure_leaves_detection_generic_and_does_not_retry():
    recognizer = FakeRecognizer(raises=True)
    backend = ArenaUnitBackend(recognizer=recognizer)
    unit = raw()
    frame = np.zeros((150, 150, 3), np.uint8)
    backend._name_troops([unit], frame)
    backend._name_troops([unit], frame)
    assert unit.name == "unknown_medium"
    assert len(recognizer.calls) == 1
    assert backend._recognizer_failed


class FakeLibrary:
    cards = ["knight", "golem"]

    def __init__(self, shared=False):
        rng = np.random.default_rng(42)
        first = rng.random((20, 20), dtype=np.float32)
        second = first.copy() if shared else rng.random((20, 20), dtype=np.float32)
        mask = np.ones((20, 20), np.float32)
        self.templates = {"knight": [SpriteTemplate("knight", first, mask, 0)],
                          "golem": [SpriteTemplate("golem", second, mask, 0)]}

    def frames_for(self, card):
        return self.templates[card]


class FakeMatcher:
    def __init__(self, hits, shared=False):
        self.library = FakeLibrary(shared)
        self.hits = hits
        self.requested_topk = None

    def identify_window(self, frame, box, candidates, topk):
        self.requested_topk = topk
        return self.hits


def hit(card, score, box=(0, 0, 64, 64)):
    return SpriteHit(card=card, score=score, bbox=box, height_px=box[3]-box[1], template=0)


def run_mocked_matches(hits, shared=False):
    matcher = FakeMatcher(hits, shared)
    rec = UnitRecognizer(matcher=matcher)
    rec._coarse_score = lambda card, image, bh, bw: 0.5
    frame = np.zeros((64, 64, 3), np.uint8)
    found = rec.identify(frame, (0, 0, 64, 64), has_bar=False, topk=1)
    return rec, matcher, found


def test_injected_matcher_is_prepared_and_margin_uses_independent_competitor():
    rec, matcher, found = run_mocked_matches([hit("knight", .95), hit("golem", .72)])
    assert rec._prepared
    assert found is not None and found.card == "knight"
    assert found.runner_up == .72
    assert matcher.requested_topk == 2  # display topk=1 must not hide the runner-up


@pytest.mark.parametrize("hits,shared", [
    ([hit("knight", .95, (0, 0, 30, 30)), hit("golem", .60)], False),  # tiny partial match
    ([hit("knight", .95), hit("golem", .93)], False),  # no meaningful margin
    ([hit("knight", .70), hit("golem", .30)], False),  # below score floor
    ([hit("knight", .95)], False),                      # no independent runner-up
    ([hit("knight", .95), hit("golem", .60)], True),   # identical art across cards
])
def test_recognizer_abstains_on_weak_or_ambiguous_evidence(hits, shared):
    _rec, _matcher, found = run_mocked_matches(hits, shared=shared)
    assert found is None


def test_missing_sprite_library_fails_open_to_generic(tmp_path):
    rec = UnitRecognizer(sprites_dir=tmp_path)
    assert rec.identify(np.zeros((64, 64, 3), np.uint8), (0, 0, 64, 64), has_bar=False) is None
    assert not rec.is_ready


def _tower_bar(grid, tower, hue, outline=True):
    tx, ty = grid.TOWER_TILES[tower]
    x, y = grid.tile_to_pixel(tx, ty - .15)
    ppt = pixels_per_tile(grid)
    width, height = int(1.1 * ppt), max(4, int(.18 * ppt))
    return HealthBar((x-width//2, y-height//2, x+width//2, y+height//2),
                     Side.ENEMY if hue == 0 else Side.ALLY, 1.0, width, width,
                     has_outline=outline, hue=hue)


def test_colour_swap_requires_multiple_towers_and_consistent_frames():
    grid = ArenaGrid(720, 1280)
    backend = ArenaUnitBackend(enable_recognition=False)
    bars = [_tower_bar(grid, name, 0 if name.startswith("ally") else 105)
            for name in ("ally_left", "ally_right", "enemy_left", "enemy_right")]
    backend.last_bars = bars[:3]
    backend._learn_team_colours(grid)
    assert backend._ally_hue is None
    backend.last_bars = bars
    backend._learn_team_colours(grid)
    assert backend._ally_hue is None  # one frame is not enough
    backend._learn_team_colours(grid)
    assert backend._ally_hue == 0 and backend._enemy_hue == 105
    assert backend._correct_side(Side.ENEMY, 0) == Side.ALLY
    assert backend._correct_side(Side.ALLY, 105) == Side.ENEMY
    backend.reset()
    assert backend._ally_hue is None and backend._enemy_hue is None


def test_unoutlined_tower_neighbours_cannot_calibrate_colours():
    grid = ArenaGrid(720, 1280)
    backend = ArenaUnitBackend(enable_recognition=False)
    backend.last_bars = [_tower_bar(grid, name, 0 if name.startswith("ally") else 105, False)
                         for name in ("ally_left", "ally_right", "enemy_left", "enemy_right")]
    for _ in range(3):
        backend._learn_team_colours(grid)
    assert backend._ally_hue is None and backend._enemy_hue is None


def test_bar_only_tracks_need_confirmation_and_expire_early():
    detector = HybridArenaDetector(ArenaGrid(720, 1280), enable_opencv_fallback=False)
    item = raw(source="bars")
    assert detector._update_tracks([item], 1.0) == []
    assert detector._update_tracks([], 1.1) == []
    confirmed = detector._update_tracks([item], 1.2)
    assert len(confirmed) == 1
    assert detector._update_tracks([], 2.0) == []
    assert detector._update_tracks([raw()], 3.0)  # silhouette-backed detections are immediate


def test_fusion_keeps_specific_cv_name_when_yolo_class_is_generic():
    yolo = raw(source="yolo")
    cv = raw(name="baby_dragon", source="segmentation+sprite", x=6.1)
    cv.move_type = MoveType.AIR
    merged = HybridArenaDetector._fuse_detections([yolo], [cv])
    assert len(merged) == 1
    assert merged[0].name == "baby_dragon"
    assert merged[0].move_type == MoveType.AIR
    assert merged[0].source == "fused+segmentation+sprite"


def test_tracking_preserves_names_and_requires_two_conflicting_cv_votes():
    detector = HybridArenaDetector(ArenaGrid(720, 1280), enable_opencv_fallback=False)
    first = raw(name="mega_knight", source="segmentation+sprite")
    track = detector._update_tracks([first], 1.0)[0]
    tid = track.track_id
    conflict = raw(name="giant", source="segmentation+sprite")
    assert detector._update_tracks([conflict], 1.1)[0].name == "mega_knight"
    assert detector._update_tracks([raw()], 1.2)[0].name == "mega_knight"
    track = detector._update_tracks([conflict], 1.3)[0]
    assert track.track_id == tid and track.name == "giant"
    # A specific model prediction is authoritative over later isolated CV guesses.
    assert detector._update_tracks([raw(name="knight", source="yolo")], 1.4)[0].name == "knight"
    for timestamp in (1.5, 1.6, 1.7):
        assert detector._update_tracks([conflict], timestamp)[0].name == "knight"
