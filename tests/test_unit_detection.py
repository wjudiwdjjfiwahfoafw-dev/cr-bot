"""Tests for real-sprite troop detection: segmentation, sizing, identity learning, hand reading.

These tests exist because the previous vision stack was validated against *synthetic* health bars
and emulator pixel constants, which is exactly why it did not work on a real device.  Everything
here is either driven by the real 720x1280 device frame (``calibration_preview.png``) or by real
in-game sprite frames composited onto a static arena, with the ground truth known.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

from cr_bot.arena.device_calibrator import ArenaCalibration
from cr_bot.arena.grid import ArenaGrid
from cr_bot.arena.projection import RectProjection, projection_from_landmarks
from cr_bot.models import Side, WeightClass
from cr_bot.vision.arena_detector import ArenaUnitBackend, HybridArenaDetector
from cr_bot.vision.hand_detector import HandCardDetector
from cr_bot.vision.sprite_matcher import (
    SpriteLibrary,
    masked_zncc,
    masked_zncc_at,
)
from cr_bot.vision.unit_detector import (
    ArenaBackgroundModel,
    BlobUnitDetector,
    HealthBarReader,
    classify_size,
    pixels_per_tile,
)
from cr_bot.vision.unit_identity import UnitIdentityMemory, normalise_patch, patch_similarity

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_FRAME = REPO_ROOT / "calibration_preview.png"
SPRITES = REPO_ROOT / "assets" / "unit_sprites"
TEMPLATES = REPO_ROOT / "assets" / "card_templates"
ART = REPO_ROOT / "assets" / "card_art"

DECK = ["mega_knight", "miner", "inferno_dragon", "skeleton_barrel",
        "zap", "bats", "spear_goblins", "goblin_gang"]


def _flat_calibration(width: int = 720, height: int = 1280) -> ArenaCalibration:
    return ArenaCalibration(
        frame_size=(width, height),
        projection=RectProjection(30, 100, width - 20, height - 250),
        hand_slots_px=[(0.31 * width, 0.908 * height), (0.50 * width, 0.908 * height),
                       (0.69 * width, 0.908 * height), (0.875 * width, 0.908 * height)],
        hand_card_size_px=(0.16 * width, 0.09 * height),
        elixir_px=(0.20 * width, 0.965 * width, 0.983 * height),
        river_px_y=0.44 * height,
    )


def _device_calibration() -> ArenaCalibration:
    """Calibration for the real frame, built the way the calibrator does it (princess towers)."""
    proj, _residual = projection_from_landmarks(
        {
            "enemy_left": (145, 325), "enemy_right": (578, 325),
            "ally_left": (145, 762), "ally_right": (578, 762),
            "enemy_king": (355, 190), "ally_king": (350, 895),
        }
    )
    return ArenaCalibration(
        frame_size=(720, 1280),
        projection=proj,
        hand_slots_px=[(224.0, 1162.0), (360.0, 1162.0), (496.0, 1162.0), (631.0, 1162.0)],
        hand_card_size_px=(127.0, 114.0),
        elixir_px=(146.0, 695.0, 1259.0),
        river_px_y=567.0,
    )


def _sprite(card: str, index: int = 0) -> np.ndarray:
    path = sorted((SPRITES / card).glob("*.png"))[index]
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _arena_background(width: int = 720, height: int = 400) -> np.ndarray:
    """A calm, static 'arena floor' with a little texture (a flat colour would be unrealistic)."""
    rng = np.random.default_rng(7)
    bg = np.full((height, width, 3), (86, 158, 74), np.uint8)
    noise = rng.integers(-8, 8, (height, width, 1), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(0, height, 24):
        cv2.line(bg, (0, y), (width, y), (78, 146, 66), 1)
    return bg


def _paste(canvas: np.ndarray, sprite: np.ndarray, x: int, y: int, height: int,
           tint: str = "neutral") -> np.ndarray:
    h, w = sprite.shape[:2]
    scale = height / float(h)
    resized = cv2.resize(sprite, (max(1, int(round(w * scale))), height),
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
    blended = (np.clip(rgb, 0, 255) * alpha + roi.astype(np.float32) * (1 - alpha))
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = blended.astype(np.uint8)
    return out


# ======================================================================================
# geometry: no emulator pixel constants anywhere
# ======================================================================================

def test_pixels_per_tile_follows_the_resolution() -> None:
    small = ArenaGrid(720, 1280, _flat_calibration())
    large = ArenaGrid(1440, 2560, _flat_calibration(1440, 2560))
    p_small = pixels_per_tile(small)
    p_large = pixels_per_tile(large)
    assert 20.0 < p_small < 45.0
    assert p_large == pytest.approx(2.0 * p_small, rel=0.05)


def test_size_buckets_are_in_tiles() -> None:
    assert classify_size(0.6)[0] is WeightClass.SWARM
    assert classify_size(1.5)[0] is WeightClass.LIGHT
    assert classify_size(2.4)[0] is WeightClass.MEDIUM
    assert classify_size(4.1)[0] is WeightClass.HEAVY_TANK
    for h in (0.3, 1.0, 2.0, 6.0):
        assert classify_size(h)[1].startswith("unknown_")


# ======================================================================================
# background model
# ======================================================================================

def test_background_model_reports_foreground_only_where_something_moved() -> None:
    model = ArenaBackgroundModel(diff_threshold=20.0, warmup_frames=2)
    base = _arena_background()
    for _ in range(3):
        model.observe(base)
    assert model.is_ready

    sprite = _sprite("giant")
    moved = _paste(base, sprite, 200, 120, 150)
    mask = model.foreground(moved)
    ys, xs = np.nonzero(mask)
    assert ys.size > 500, "the pasted sprite must show up as foreground"
    assert 140 < float(np.median(xs)) < 420
    assert model.foreground(base).sum() == 0, "an unchanged frame must produce no foreground"


def test_selective_update_does_not_swallow_a_stationary_unit() -> None:
    """A troop that stops moving must stay detectable (a naive running median would absorb it)."""
    model = ArenaBackgroundModel(diff_threshold=20.0, warmup_frames=2, learning_rate=0.35)
    base = _arena_background()
    sprite = _sprite("musketeer")
    scene = _paste(base, sprite, 250, 140, 120)
    for _ in range(3):
        model.observe(base)
    for _ in range(12):
        mask = model.observe(scene)
    assert int((mask > 0).sum()) > 300


# ======================================================================================
# blob detection on a real sprite
# ======================================================================================

def test_blob_detector_locates_a_real_sprite_and_reads_its_side() -> None:
    grid = ArenaGrid(720, 1280, _flat_calibration())
    base = _arena_background()
    ppt = pixels_per_tile(grid)

    det = BlobUnitDetector()
    for _ in range(3):
        det.detect(base, grid)          # warm the background model

    # an enemy-tinted Giant, ~3 tiles tall, dropped well inside the arena
    scene = _paste(base, _sprite("giant"), 300, 150, int(2.6 * ppt), tint="enemy")
    units = det.detect(scene, grid)
    assert units, "the pasted troop must be detected"

    biggest = max(units, key=lambda u: u.area_px)
    assert biggest.side is Side.ENEMY
    assert biggest.height_tiles == pytest.approx(2.6, abs=0.8)
    assert biggest.weight_class in (WeightClass.MEDIUM, WeightClass.HEAVY_TANK)
    # the reported tile position must land inside the arena, near where the sprite was pasted
    assert 0.5 < biggest.tile_x < 17.5
    assert 0.0 < biggest.tile_y < 32.0


def test_blob_detector_learns_the_arena_without_being_told() -> None:
    """No calibration of colours: three plain frames of the floor are enough to start detecting."""
    grid = ArenaGrid(720, 1280, _flat_calibration())
    base = _arena_background()
    det = BlobUnitDetector()
    assert det.background.is_ready is False
    for _ in range(4):
        det.detect(base, grid)
    assert det.background.is_ready is True


def test_health_bar_reader_is_tile_relative_not_pixel_relative() -> None:
    """The same unit must be read identically at 720p and 1440p."""
    results = []
    for scale in (1, 2):
        width, height = 720 * scale, 1280 * scale
        grid = ArenaGrid(width, height, _flat_calibration(width, height))
        ppt = pixels_per_tile(grid)
        frame = _arena_background(width, height)
        cx = int(width * 0.45)
        cy = int(height * 0.30)
        bar_w = int(0.9 * ppt)
        bar_h = max(3, int(0.09 * ppt))
        cv2.rectangle(frame, (cx, cy), (cx + bar_w, cy + bar_h), (30, 30, 240), -1)
        reader = HealthBarReader()
        bars = reader.read(frame, grid, ppt)
        results.append((len(bars), bar_w / ppt))
        assert bars, f"no bar found at {width}x{height}"
    assert results[0][0] == results[1][0]
    assert results[0][1] == pytest.approx(results[1][1], rel=0.12)


# ======================================================================================
# identity: learned from the bot's own deploys, no training
# ======================================================================================

def test_masked_zncc_is_one_for_identical_content() -> None:
    rng = np.random.default_rng(0)
    template = cv2.GaussianBlur(rng.random((40, 30)).astype(np.float32), (7, 7), 0)
    mask = np.zeros((40, 30), np.float32)
    mask[3:-3, 3:-3] = 1.0
    image = np.zeros((120, 110), np.float32)
    image[50:90, 40:70] = template
    scores = masked_zncc(image, template, mask)
    _min, max_val, _min_loc, _max_loc = cv2.minMaxLoc(scores)
    assert max_val == pytest.approx(1.0, abs=1e-4)
    assert masked_zncc_at(template, template, np.ones_like(mask)) == pytest.approx(1.0, abs=1e-5)


def test_identity_memory_learns_then_recognises_a_card(tmp_path) -> None:
    memory = UnitIdentityMemory(directory=tmp_path, namespace="test", max_samples_per_card=4)
    assert memory.match(np.zeros((40, 40, 3), np.uint8)) is None, "empty memory matches nothing"

    base = _arena_background()
    giant = _paste(base, _sprite("giant"), 220, 120, 150, tint="ally")
    musketeer = _paste(base, _sprite("musketeer"), 220, 120, 150, tint="ally")
    crop = lambda img: img[110:290, 200:400]  # noqa: E731

    assert memory.add_sample("giant", crop(giant)) is True
    assert memory.add_sample("musketeer", crop(musketeer)) is True
    assert memory.sample_count == 2
    assert memory.add_sample("giant", crop(giant)) is False, "duplicates must not inflate memory"

    match = memory.match(crop(giant))
    assert match is not None and match.card == "giant"
    assert match.score > 0.9

    # persistence round-trip
    path = memory.save()
    assert path.is_file()
    reloaded = UnitIdentityMemory(directory=tmp_path, namespace="test")
    assert reloaded.sample_count == 2
    again = reloaded.match(crop(musketeer))
    assert again is not None and again.card == "musketeer"


def test_identity_memory_is_namespaced_per_resolution(tmp_path) -> None:
    a = UnitIdentityMemory(directory=tmp_path, namespace="720x1280")
    b = UnitIdentityMemory(directory=tmp_path, namespace="1080x1920")
    a.add_sample("giant", _arena_background()[100:300, 100:300])
    a.save()
    assert b.sample_count == 0
    assert UnitIdentityMemory(directory=tmp_path, namespace="720x1280").sample_count == 1


def test_identity_is_tint_tolerant_enough_to_match_enemy_to_ally_samples(tmp_path) -> None:
    """The opponent's copy of a troop is the same model, tinted red (the bot only ever labels its own)."""
    memory = UnitIdentityMemory(directory=tmp_path, namespace="memory")
    base = _arena_background()
    ally = _paste(base, _sprite("mega_knight"), 200, 110, 140, tint="ally")
    enemy = _paste(base, _sprite("mega_knight"), 200, 110, 140, tint="enemy")
    other = _paste(base, _sprite("giant"), 200, 110, 140, tint="enemy")
    crop = lambda img: img[100:260, 180:380]  # noqa: E731
    memory.add_sample("mega_knight", crop(ally))
    match = memory.match(crop(enemy))
    assert match is not None, "a red-tinted copy of a known troop must still be recognised"
    assert match.card == "mega_knight"
    assert patch_similarity(normalise_patch(crop(ally)), normalise_patch(crop(other))) < \
        patch_similarity(normalise_patch(crop(ally)), normalise_patch(crop(enemy)))


# ======================================================================================
# the backend that the bot actually runs
# ======================================================================================

def test_unit_backend_falls_back_to_bars_while_the_background_warms_up() -> None:
    grid = ArenaGrid(720, 1280, _flat_calibration())
    backend = ArenaUnitBackend()
    frame = _arena_background(720, 1280)
    ppt = pixels_per_tile(grid)
    cx, cy = 360, 500
    cv2.rectangle(frame, (cx, cy), (cx + int(0.9 * ppt), cy + max(3, int(0.09 * ppt))),
                  (40, 40, 235), -1)
    detections = backend.detect(frame, grid)
    assert detections, "bars must be used before the background model is ready"
    assert all(d.source in ("bars", "segmentation", "segmentation+identity") for d in detections)


def test_unit_backend_names_a_troop_once_it_has_learned_it(tmp_path) -> None:
    grid = ArenaGrid(720, 1280, _flat_calibration())
    base = _arena_background()
    sprite = _sprite("musketeer")
    scene = _paste(base, sprite, 250, 120, 140, tint="ally")

    memory = UnitIdentityMemory(directory=tmp_path, namespace="t")
    backend = ArenaUnitBackend(identity_memory=memory)
    for _ in range(4):
        backend.detect(base, grid)

    unnamed = backend.detect(scene, grid)
    assert unnamed, "the troop must be detected even with an empty identity memory"
    assert all(d.name.startswith("unknown_") for d in unnamed)

    blob = max(unnamed, key=lambda d: d.pixel_bbox[2] - d.pixel_bbox[0])
    memory.add_from_frame("musketeer", scene, blob.pixel_bbox)
    named = backend.detect(scene, grid)
    assert any(d.name == "musketeer" for d in named), [d.name for d in named]


def test_hybrid_detector_accepts_an_identity_memory(tmp_path) -> None:
    grid = ArenaGrid(720, 1280, _flat_calibration())
    memory = UnitIdentityMemory(directory=tmp_path, namespace="t")
    detector = HybridArenaDetector(grid, yolo_weights_path=None, identity_memory=memory)
    assert detector.cv_fallback is not None
    assert detector.cv_fallback.identity is memory


# ======================================================================================
# the real device frame: hand templates + sprite library
# ======================================================================================

def _composited_art(card: str) -> np.ndarray:
    """The portrait exactly as ``scripts/fetch_card_assets.build_hand_templates`` renders it."""
    img = cv2.imread(str(ART / f"{card}.png"), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.shape[2] == 4
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    flat = img[:, :, :3].astype(np.float32) * alpha + np.array([28, 28, 34], np.float32) * (1 - alpha)
    return cv2.cvtColor(flat.astype(np.uint8), cv2.COLOR_BGR2GRAY)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - float(a.mean())
    b = b.astype(np.float32) - float(b.mean())
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / den) if den > 1e-6 else -2.0


def test_hand_templates_are_cut_from_clean_art() -> None:
    """Regression guard: the shipped templates used to be blurry crops of an emulator frame."""
    shipped = sorted(p.stem for p in TEMPLATES.glob("*.png"))
    assert len(shipped) >= 80, "the template folder should carry the whole card database"
    for card in DECK:
        assert card in shipped, f"missing template for deck card {card}"

    templates = {c: cv2.imread(str(TEMPLATES / f"{c}.png"), cv2.IMREAD_GRAYSCALE) for c in DECK}
    arts = {p.stem: _composited_art(p.stem) for p in sorted(ART.glob("*.png"))}

    for card, tpl in templates.items():
        assert tpl is not None
        own = cv2.resize(arts[card], (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_AREA)
        assert _ncc(own, tpl) > 0.93, f"{card} template is not a faithful downscale of its portrait"

        # ...and it must be closer to its own portrait than to every other card's portrait
        best, best_card = -2.0, ""
        for name, art in arts.items():
            ref = cv2.resize(art, (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_AREA)
            score = _ncc(ref, tpl)
            if score > best:
                best, best_card = score, name
        assert best_card == card, f"template for {card} looks most like {best_card} ({best:.3f})"


def test_hand_detection_reads_the_real_frame_correctly() -> None:
    """Ground truth read off the sample device frame: Barrel, Bats, Spear Goblins, Zap."""
    assert REAL_FRAME.is_file(), "the sample device frame is the fixture for this test"
    frame = cv2.imread(str(REAL_FRAME))
    grid = ArenaGrid(720, 1280, _device_calibration())
    detector = HandCardDetector(grid, DECK, templates_dir=str(TEMPLATES))
    assert detector.has_templates

    slots = detector.detect_hand(frame, current_elixir=6.0)
    found = [s.card_name for s in slots]
    assert found == ["skeleton_barrel", "bats", "spear_goblins", "zap"], found
    assert all(s.is_ready for s in slots), "6 elixir must afford all four cards"


def test_sprite_library_ships_real_game_frames() -> None:
    library = SpriteLibrary(SPRITES)
    cards = library.cards
    assert len(cards) >= 40, f"expected a broad sprite library, got {len(cards)}"
    for card in ("mega_knight", "giant", "musketeer", "hog_rider", "bats", "valkyrie"):
        frames = library.frames_for(card)
        assert len(frames) >= 2, f"{card} should ship several poses"
        for frame in frames:
            assert frame.mask.sum() > 80
            assert frame.gray.shape == frame.mask.shape
            assert 0.2 < frame.aspect < 5.0, "sprite frames must be character-shaped"
