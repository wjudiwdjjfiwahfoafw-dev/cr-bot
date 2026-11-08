"""Unit tests for 18x32 arena coordinate geometry, pocket rules, and MK Zap Bait card aliases."""

from __future__ import annotations

import random

from cr_bot.arena.grid import ArenaGrid
from cr_bot.config.cards_db import get_card_info, normalize_card_name
from cr_bot.config.loader import load_config
from cr_bot.models import Lane, Side, TowerState


def test_user_eight_card_aliases_and_average_elixir() -> None:
    raw_user_list = [
        "MK",
        "miner",
        "inferno dragon",
        "skeleton barrel",
        "zap",
        "bats",
        "speer goblins",
        "goblin army",
    ]
    normalized = [normalize_card_name(c) for c in raw_user_list]
    assert normalized == [
        "mega_knight",
        "miner",
        "inferno_dragon",
        "skeleton_barrel",
        "zap",
        "bats",
        "spear_goblins",
        "goblin_gang",
    ]
    total_elixir = sum(get_card_info(c).elixir for c in normalized)
    assert total_elixir == 26  # 3.25 average elixir


def test_yaml_config_loads_cleanly() -> None:
    cfg = load_config("configs/mk_zap_bait.yaml")
    assert len(cfg.normalized_deck()) == 8
    assert abs(cfg.average_elixir() - 3.25) < 1e-6


def test_pixel_tile_roundtrip_accuracy() -> None:
    grid = ArenaGrid(720, 1280)
    for tx in (0.8, 3.5, 9.0, 14.5, 17.2):
        for ty in (2.8, 6.5, 16.0, 20.5, 29.8):
            px, py = grid.tile_to_pixel(tx, ty)
            rtx, rty = grid.pixel_to_tile(px, py)
            assert abs(tx - rtx) < 0.06
            assert abs(ty - rty) < 0.06


def test_deploy_clamping_and_pocket_unlocking() -> None:
    grid = ArenaGrid(720, 1280)
    towers = {
        "enemy_left": TowerState("enemy_left", Side.ENEMY, Lane.LEFT, alive=True),
        "enemy_right": TowerState("enemy_right", Side.ENEMY, Lane.RIGHT, alive=True),
    }

    # Normal troop cannot be placed on enemy half when both towers are up
    _, clamped_y = grid.clamp_deploy_tile(3.5, 13.5, deploy_anywhere=False, towers=towers)
    assert clamped_y >= 16.9

    # Miner (deploy_anywhere=True) CAN be placed on enemy tower
    _, miner_y = grid.clamp_deploy_tile(2.0, 7.6, deploy_anywhere=True, towers=towers)
    assert abs(miner_y - 7.6) < 0.05

    # Destroy enemy left tower -> left pocket (y=13.5) becomes legal!
    towers["enemy_left"].alive = False
    _, pocket_y = grid.clamp_deploy_tile(6.8, 13.5, deploy_anywhere=False, towers=towers)
    assert abs(pocket_y - 13.5) < 0.05


def test_anti_tornado_miner_spots_never_expose_king_or_repeat() -> None:
    grid = ArenaGrid(720, 1280)
    rng = random.Random(123)
    prev = None
    for _ in range(12):
        spot = grid.pick_safe_miner_spot(Lane.LEFT, enemy_king_activated=False, rng=rng)
        # Left tower is at x=3.5; inner King-side is x > 3.5. Safe outer spots must have x <= 3.0!
        assert spot[0] <= 3.0
        assert spot != prev
        prev = spot
