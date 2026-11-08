#!/usr/bin/env python3
"""
Generates visual HUD snapshots demonstrating how the MK Zap Bait Tactical Brain
evaluates and reacts to classic Clash Royale scenarios.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr_bot.arena.grid import ArenaGrid
from cr_bot.brain.mk_zap_bait_brain import MKZapBaitBrain
from cr_bot.config.cards_db import get_card_info
from cr_bot.config.loader import BotConfig
from cr_bot.models import (
    BattleState,
    DetectedUnit,
    GamePhase,
    HandSlot,
    Lane,
    OpponentCycleState,
    Side,
    TowerState,
)
from cr_bot.ui.debug_hud import TacticalDebugHUD


def render_synthetic_arena(grid: ArenaGrid, elixir: float) -> np.ndarray:
    """Draw a clean 720x1280 synthetic Clash Royale arena & bottom card tray for HUD visualization."""
    frame = np.full((grid.height, grid.width, 3), (38, 44, 36), dtype=np.uint8)
    ax1, ay1, ax2, ay2 = grid.arena_pixel_bounds

    # Checkerboard grass
    for ty in range(32):
        for tx in range(18):
            p1 = grid.tile_to_pixel(tx, ty)
            p2 = grid.tile_to_pixel(tx + 1, ty + 1)
            shade = (68, 136, 76) if (tx + ty) % 2 == 0 else (60, 124, 68)
            cv2.rectangle(frame, p1, p2, shade, -1)

    # River & Bridges
    _, ry1 = grid.tile_to_pixel(0, 15.4)
    _, ry2 = grid.tile_to_pixel(0, 16.6)
    cv2.rectangle(frame, (ax1, ry1), (ax2, ry2), (205, 155, 55), -1)
    for bx in (3.5, 14.5):
        bp1 = grid.tile_to_pixel(bx - 1.0, 15.2)
        bp2 = grid.tile_to_pixel(bx + 1.0, 16.8)
        cv2.rectangle(frame, bp1, bp2, (95, 125, 165), -1)

    # Towers
    for name, (tx, ty) in grid.TOWER_TILES.items():
        px, py = grid.tile_to_pixel(tx, ty)
        col = (65, 70, 210) if name.startswith("enemy") else (215, 145, 55)
        sz = 26 if "king" in name else 20
        cv2.rectangle(frame, (px - sz, py - sz), (px + sz, py + sz), col, -1)
        cv2.rectangle(frame, (px - sz, py - sz), (px + sz, py + sz), (230, 230, 230), 2)

    # Bottom Card Tray & Elixir Bar
    cv2.rectangle(frame, (0, ay2 + 12), (grid.width, grid.height), (42, 34, 28), -1)
    ex1 = int(round(grid.cal.elixir_x_start_ratio * grid.width))
    ex2 = int(round(grid.cal.elixir_x_end_ratio * grid.width))
    ey = int(round(grid.cal.elixir_y_ratio * grid.height))
    cv2.rectangle(frame, (ex1, ey - 8), (ex2, ey + 8), (45, 25, 50), -1)
    fill_x = ex1 + int(round((elixir / 10.0) * (ex2 - ex1)))
    cv2.rectangle(frame, (ex1, ey - 8), (fill_x, ey + 8), (215, 55, 235), -1)
    return frame


def build_default_towers(grid: ArenaGrid) -> Dict[str, TowerState]:
    out: Dict[str, TowerState] = {}
    for name, (tx, ty) in grid.TOWER_TILES.items():
        side = Side.ENEMY if name.startswith("enemy") else Side.ALLY
        lane = Lane.CENTER if "king" in name else (Lane.LEFT if "left" in name else Lane.RIGHT)
        max_hp = 4824 if "king" in name else 3052
        out[name] = TowerState(name=name, side=side, lane=lane, alive=True, hp_ratio=1.0, max_hp=max_hp, tile_x=tx, tile_y=ty)
    return out


def make_hand(grid: ArenaGrid, cards: List[Tuple[str, bool]], elixir: float) -> List[HandSlot]:
    hand: List[HandSlot] = []
    for idx, (cname, evo) in enumerate(cards):
        info = get_card_info(cname)
        hand.append(
            HandSlot(
                slot_index=idx,
                card_name=cname,
                elixir_cost=info.elixir,
                is_ready=(elixir >= info.elixir),
                is_evolved=evo,
                confidence=0.98,
                tap_xy=grid.hand_slot_pixel(idx),
            )
        )
    return hand


def make_unit(
    grid: ArenaGrid,
    tid: int,
    name: str,
    side: Side,
    tx: float,
    ty: float,
    vy: float = 0.0,
    first_ts: float = 10.0,
    now_ts: float = 12.5,
) -> DetectedUnit:
    info = get_card_info(name)
    px, py = grid.tile_to_pixel(tx, ty)
    return DetectedUnit(
        track_id=tid,
        name=name,
        side=side,
        tile_x=tx,
        tile_y=ty,
        pixel_bbox=(px - 22, py - 28, px + 22, py + 12),
        confidence=0.94,
        hp_ratio=0.95,
        move_type=info.move_type,
        weight_class=info.weight_class,
        vx=0.0,
        vy=vy,
        first_seen_ts=first_ts,
        last_seen_ts=now_ts,
        source="fused",
    )


def main() -> int:
    cfg = BotConfig()
    grid = ArenaGrid(720, 1280)
    out_dir = ROOT / "assets" / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        # Scenario 1: Defending P.E.K.K.A (Refuses MK blunder; uses Flanking Inferno Dragon + Kite!)
        (
            "01_pekka_defense_no_mk_blunder.png",
            8.0,
            [("mega_knight", True), ("inferno_dragon", False), ("spear_goblins", False), ("zap", False)],
            [make_unit(grid, 1, "pekka", Side.ENEMY, 3.5, 18.2, vy=1.1)],
            OpponentCycleState(estimated_elixir=3.5, known_enemy_cards={"pekka"}),
        ),
        # Scenario 2: Evo Mega Knight Crush on Hog Rider + Musketeer Bridge Push
        (
            "02_evo_mk_crush_hog_musketeer.png",
            7.8,
            [("mega_knight", True), ("miner", False), ("bats", False), ("zap", False)],
            [
                make_unit(grid, 1, "hog_rider", Side.ENEMY, 14.5, 18.8, vy=2.4),
                make_unit(grid, 2, "musketeer", Side.ENEMY, 14.3, 17.5, vy=1.3),
            ],
            OpponentCycleState(estimated_elixir=2.2, known_enemy_cards={"hog_rider", "musketeer"}),
        ),
        # Scenario 3: Small Spell Baited -> Skeleton Barrel + Queued Anti-Tornado Miner Combo
        (
            "03_spell_baited_skarrel_miner_combo.png",
            7.2,
            [("skeleton_barrel", False), ("miner", False), ("goblin_gang", False), ("inferno_dragon", False)],
            [],
            OpponentCycleState(
                estimated_elixir=4.5,
                small_spell_baited=True,
                last_small_spell_ts=9.0,
                cards_since_small_spell=1,
                known_enemy_cards={"the_log", "knight"},
            ),
        ),
        # Scenario 4: Timed Zap Reset on Enemy Inferno Tower Melting Our Counterpush Mega Knight
        (
            "04_timed_zap_reset_inferno_tower.png",
            4.5,
            [("zap", False), ("spear_goblins", False), ("bats", True), ("goblin_gang", False)],
            [
                make_unit(grid, 1, "mega_knight", Side.ALLY, 3.8, 12.2, vy=-1.2, first_ts=6.0, now_ts=12.5),
                make_unit(grid, 2, "inferno_tower", Side.ENEMY, 7.2, 10.5, vy=0.0, first_ts=10.1, now_ts=12.5),
            ],
            OpponentCycleState(
                estimated_elixir=4.0,
                has_inferno_or_sparky=True,
                known_enemy_cards={"inferno_tower", "goblin_barrel"},
            ),
        ),
    ]

    for filename, elixir, hand_spec, units, opp in scenarios:
        brain = MKZapBaitBrain(grid, config=cfg, seed=42)
        hud = TacticalDebugHUD(grid, vision_mode_label="YOLO(onnx)+CV")
        towers = build_default_towers(grid)
        towers["enemy_left"].hp_ratio = 0.62
        state = BattleState(
            timestamp=12.5,
            phase=GamePhase.BATTLE_1X,
            elixir=elixir,
            hand=make_hand(grid, hand_spec, elixir),
            units=units,
            towers=towers,
            opponent=opp,
        )
        action = brain.decide(state)
        if action:
            hud.record_action(action, state.timestamp)
        frame = render_synthetic_arena(grid, elixir)
        rendered = hud.render(frame, state, brain.last_top_candidates, fps=12.0)
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), rendered)
        print(f"Rendered {filename}: chosen={action.card_name if action else None} | {action.reason if action else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
