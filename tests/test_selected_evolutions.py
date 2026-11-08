"""The MK Zap Bait deck has exactly the two requested evolution slots."""
from pathlib import Path

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.bot_runner import ClashRoyaleBotRunner
from cr_bot.config.loader import BotConfig, load_config
from cr_bot.sim.engine import ClashRoyaleSimEngine
from cr_bot.vision.hand_detector import HandCardDetector

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["mega_knight", "skeleton_barrel"]
DECK = ["mega_knight", "miner", "inferno_dragon", "skeleton_barrel", "zap", "bats", "spear_goblins", "goblin_gang"]


def test_default_and_yaml_use_mega_knight_and_skeleton_barrel(tmp_path):
    assert BotConfig().evolutions == EXPECTED
    cfg = load_config(ROOT / "configs/mk_zap_bait.yaml")
    assert cfg.evolutions == EXPECTED
    empty = tmp_path / "default.yaml"
    empty.write_text("{}\n")
    assert load_config(empty).evolutions == EXPECTED
    cfg.identity_dir = str(tmp_path)
    cfg.yolo_weights_path = None
    bot = ClashRoyaleBotRunner(cfg, show_hud=False, dry_run=True, manual_input=False)
    assert bot.hand_detector.evolutions == set(EXPECTED)


def test_only_configured_cards_can_be_reported_as_evolved(tmp_path, monkeypatch):
    grid = ArenaGrid(720, 1280)
    names = ["mega_knight", "skeleton_barrel", "bats", "zap"]
    canvas = np.zeros((1280, 720, 3), np.uint8)
    rng = np.random.default_rng(200)
    for i, name in enumerate(names):
        x1, y1, x2, y2 = grid.hand_slot_rect(i)
        art = rng.integers(0, 256, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / (name + ".png")), art)
        canvas[y1:y2, x1:x2] = art
    detector = HandCardDetector(grid, names, tmp_path, evolutions=EXPECTED)
    # Exercise the configured-card gate independently of the unvalidated purple-glow heuristic.
    monkeypatch.setattr(detector, "_has_evo_glow", lambda frame, slot: True)
    slots = detector.detect_hand(canvas, current_elixir=10)
    assert [s.card_name for s in slots] == names
    assert [s.is_evolved for s in slots] == [True, True, False, False]


def test_sim_default_cycles_use_requested_evos_and_empty_override_is_honoured():
    engine = ClashRoyaleSimEngine(DECK, DECK)
    assert engine.ally.evolutions == EXPECTED
    engine.ally.play_counts.update({"mega_knight": 1, "skeleton_barrel": 2, "bats": 2})
    assert engine.is_card_evolved_in_hand(engine.ally, "mega_knight")
    assert engine.is_card_evolved_in_hand(engine.ally, "skeleton_barrel")
    assert not engine.is_card_evolved_in_hand(engine.ally, "bats")
    no_evos = ClashRoyaleSimEngine(DECK, DECK, ally_evos=[])
    assert no_evos.ally.evolutions == []
