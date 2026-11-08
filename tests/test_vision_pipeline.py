"""Tests for ElixirReader, HandCardDetector, OpponentTracker, and HybridArenaDetector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.models import DetectedUnit, GamePhase, Side
from cr_bot.vision.arena_detector import HybridArenaDetector
from cr_bot.vision.elixir_tracker import ElixirReader, OpponentTracker
from cr_bot.vision.hand_detector import HandCardDetector


def _make_blank_battle_frame(grid: ArenaGrid, elixir_pips: int) -> np.ndarray:
    frame = np.full((grid.height, grid.width, 3), (55, 115, 60), dtype=np.uint8)
    # River band
    ax1, _, ax2, ay2 = grid.arena_pixel_bounds
    _, ry = grid.tile_to_pixel(0, 16.0)
    cv2.rectangle(frame, (ax1, ry - 12), (ax2, ry + 12), (210, 165, 50), -1)

    # Bottom HUD tray & 10-pip elixir bar
    cv2.rectangle(frame, (0, ay2 + 10), (grid.width, grid.height), (35, 30, 28), -1)
    x_start = int(round(grid.cal.elixir_x_start_ratio * grid.width))
    x_end = int(round(grid.cal.elixir_x_end_ratio * grid.width))
    y_c = int(round(grid.cal.elixir_y_ratio * grid.height))
    seg_w = (x_end - x_start) / 10.0

    for i in range(elixir_pips):
        sx1 = int(round(x_start + i * seg_w))
        sx2 = int(round(x_start + (i + 1) * seg_w)) - 1
        # Bright purple/magenta in BGR -> HSV H~150, S~210, V~230
        cv2.rectangle(frame, (sx1, y_c - 6), (sx2, y_c + 6), (225, 45, 225), -1)

    return frame


def test_elixir_reader_accuracy() -> None:
    grid = ArenaGrid(720, 1280)
    for target in (0, 3, 7, 10):
        reader = ElixirReader(grid.cal)
        frame = _make_blank_battle_frame(grid, elixir_pips=target)
        detected = reader.read_elixir(frame)
        assert abs(detected - float(target)) <= 0.25, f"Expected {target}, got {detected}"


def test_hand_card_unique_matching_and_evo_glow(tmp_path: Path) -> None:
    grid = ArenaGrid(720, 1280)
    deck = [
        "mega_knight",
        "miner",
        "inferno_dragon",
        "skeleton_barrel",
        "zap",
        "bats",
        "spear_goblins",
        "goblin_gang",
    ]

    # Create distinct synthetic card portraits for all 8 cards
    rng = np.random.default_rng(7)
    card_art = {}
    for idx, cname in enumerate(deck):
        art = np.zeros((116, 96, 3), dtype=np.uint8)
        base_col = [int((idx * 53 + 40) % 255), int((idx * 97 + 80) % 255), int((idx * 149 + 110) % 255)]
        art[:, :] = base_col
        cv2.circle(art, (24 + idx * 5, 55), 18, (250, 240, 90), -1)
        noise = rng.integers(0, 25, size=art.shape, dtype=np.uint8)
        art = cv2.add(art, noise)
        card_art[cname] = art
        cv2.imwrite(str(tmp_path / f"{cname}.png"), art)

    detector = HandCardDetector(grid, deck_cards=deck, templates_dir=tmp_path)
    frame = _make_blank_battle_frame(grid, elixir_pips=8)

    expected_hand = ["mega_knight", "bats", "miner", "zap"]
    for slot_i, cname in enumerate(expected_hand):
        cx, cy = grid.hand_slot_pixel(slot_i)
        frame[cy - 58 : cy + 58, cx - 48 : cx + 48] = card_art[cname]

    # Paint active Evolution magenta diamond above slot 0 (mega_knight)
    s0x, s0y = grid.hand_slot_pixel(0)
    cv2.rectangle(frame, (s0x - 16, s0y - 68), (s0x + 16, s0y - 48), (230, 40, 240), -1)

    detected_slots = detector.detect_hand(frame, current_elixir=8.0)
    assert [s.card_name for s in detected_slots] == expected_hand
    assert detected_slots[0].is_evolved is True
    assert detected_slots[1].is_evolved is False
    assert all(s.is_ready for s in detected_slots)


def test_opencv_healthbar_detector_and_velocity_tracking() -> None:
    grid = ArenaGrid(720, 1280)
    detector = HybridArenaDetector(grid, yolo_weights_path=None, enable_opencv_fallback=True)

    # Frame 1: Enemy medium unit at tile (3.5, 17.5) -> draw bright red health bar 26px above foot
    f1 = _make_blank_battle_frame(grid, elixir_pips=6)
    px1, py1 = grid.tile_to_pixel(3.5, 17.5)
    # a bar as the game draws it: bright fill inside a dark capsule
    cv2.rectangle(f1, (px1 - 20, py1 - 31), (px1 + 20, py1 - 21), (32, 32, 36), -1)
    cv2.rectangle(f1, (px1 - 18, py1 - 29), (px1 + 18, py1 - 23), (20, 20, 245), -1)
    units_t1 = detector.detect_and_track(f1, now_ts=1.0)
    assert len(units_t1) == 1
    assert units_t1[0].side == Side.ENEMY
    assert abs(units_t1[0].tile_x - 3.5) < 0.4
    assert abs(units_t1[0].tile_y - 17.5) < 0.5

    # Frame 2 (0.5s later): Same unit moved down to (3.5, 19.0)
    f2 = _make_blank_battle_frame(grid, elixir_pips=6)
    px2, py2 = grid.tile_to_pixel(3.5, 19.0)
    cv2.rectangle(f2, (px2 - 20, py2 - 31), (px2 + 20, py2 - 21), (32, 32, 36), -1)
    cv2.rectangle(f2, (px2 - 18, py2 - 29), (px2 + 18, py2 - 23), (20, 20, 245), -1)
    units_t2 = detector.detect_and_track(f2, now_ts=1.5)
    assert len(units_t2) == 1
    assert units_t2[0].track_id == units_t1[0].track_id
    assert units_t2[0].vy > 0.8  # Moving downward toward our Princess Tower!


def test_opponent_small_spell_bait_cycle_tracking() -> None:
    tracker = OpponentTracker(bait_window_seconds=11.0)
    tracker.reset(now_ts=0.0)

    # Opponent plays The Log at t=2.0
    log_unit = DetectedUnit(track_id=1, name="the_log", side=Side.ENEMY, tile_x=3.5, tile_y=18.0)
    st = tracker.update(2.0, GamePhase.BATTLE_1X, [log_unit])
    assert st.small_spell_baited is True
    assert st.cards_since_small_spell == 0

    # Opponent plays 4 more non-spell cards -> Log is back in hand!
    for idx, card in enumerate(["knight", "ice_spirit", "cannon", "musketeer"], start=2):
        u = DetectedUnit(track_id=idx, name=card, side=Side.ENEMY, tile_x=14.0, tile_y=10.0)
        st = tracker.update(2.0 + idx, GamePhase.BATTLE_1X, [u])

    assert st.cards_since_small_spell == 4
    assert st.small_spell_baited is False
