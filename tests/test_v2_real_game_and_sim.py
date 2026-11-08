"""
v2.0 Real-Game Reliability & Full-Match Simulation Tests:
- PendingDeployTracker anti-overcommit & dropped-ADB-tap auto-recovery
- Raw Android RGBA `screencap` binary buffer decoding
- Bridge Princess long-range sniper interception
- Non-catchable spell immunity (never drops troops to "catch" a Fireball!)
- 10-HP zapped Goblin Barrel cleanup discipline
- End-to-end full Clash Royale match victories under 180ms ADB lag & strict Fog of War
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from cr_bot.adb.controller import ADBController
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
    Side,
    TowerState,
)
from cr_bot.sim.engine import ClashRoyaleSimEngine
from cr_bot.sim.opponents import META_DECKS, MetaOpponentAI
from cr_bot.vision.elixir_tracker import OpponentTracker


def _towers(grid: ArenaGrid) -> dict[str, TowerState]:
    out = {}
    for name, (tx, ty) in grid.TOWER_TILES.items():
        side = Side.ENEMY if name.startswith("enemy") else Side.ALLY
        lane = Lane.CENTER if "king" in name else (Lane.LEFT if "left" in name else Lane.RIGHT)
        max_hp = 4824 if "king" in name else 3052
        out[name] = TowerState(name=name, side=side, lane=lane, alive=True, hp_ratio=1.0, max_hp=max_hp, tile_x=tx, tile_y=ty)
    return out


def _hand(grid: ArenaGrid, names: list[str], elixir: float) -> list[HandSlot]:
    return [
        HandSlot(
            slot_index=i,
            card_name=cname,
            elixir_cost=get_card_info(cname).elixir,
            is_ready=(elixir >= get_card_info(cname).elixir),
            tap_xy=grid.hand_slot_pixel(i),
        )
        for i, cname in enumerate(names)
    ]


def test_raw_rgba_screencap_decoder() -> None:
    """The raw RGBA screencap decoder still works (now living in ScreencapSession)."""
    from cr_bot.adb.adb_utils import Adb
    from cr_bot.adb.scrcpy_multiclient import ScreencapSession

    w, h = 720, 1280
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 200  # R
    rgba[:, :, 1] = 100  # G
    rgba[:, :, 2] = 50   # B
    rgba[:, :, 3] = 255  # A
    header = w.to_bytes(4, "little") + h.to_bytes(4, "little") + (1).to_bytes(4, "little")
    payload = header + rgba.tobytes()

    session = ScreencapSession(Adb("adb"), target_width=720)
    with patch.object(session.adb, "exec_out", return_value=payload):
        bgr = session._grab_raw()
    assert bgr is not None
    assert bgr.shape == (1280, 720, 3)
    # OpenCV BGR order check: B=50, G=100, R=200
    assert tuple(int(v) for v in bgr[100, 100]) == (50, 100, 200)


def test_pending_tracker_prevents_double_commit_and_recovers_dropped_tap() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=1)
    knight = DetectedUnit(track_id=1, name="knight", side=Side.ENEMY, tile_x=3.5, tile_y=19.5, hp_ratio=1.0)

    # t=10.0: Bot answers lone Knight with center-pull Spear Goblins
    s1 = BattleState(
        timestamp=10.0,
        phase=GamePhase.BATTLE_1X,
        elixir=6.0,
        hand=_hand(grid, ["spear_goblins", "goblin_gang", "bats", "zap"], 6.0),
        units=[knight],
        towers=_towers(grid),
    )
    a1 = brain.decide(s1)
    assert a1 is not None
    assert a1.card_name in ("spear_goblins", "goblin_gang")

    # t=10.7: Suppose hand rotated (server accepted tap), but unit hasn't finished 1.0s deploy yet!
    # PendingDeployTracker prevents dumping a 2nd card onto the already-covered 3-elixir Knight!
    s2 = BattleState(
        timestamp=10.7,
        phase=GamePhase.BATTLE_1X,
        elixir=4.2,
        hand=_hand(grid, ["miner", "goblin_gang", "bats", "zap"], 4.2),
        units=[knight],
        towers=_towers(grid),
    )
    a2 = brain.decide(s2)
    assert a2 is None, f"Should not double-commit onto lone Knight during spawn clock, got {a2}"

    # Now test dropped ADB tap detection: reset brain, play at t=20.0, and at t=21.0 hand & elixir haven't changed!
    brain.reset()
    s3 = BattleState(
        timestamp=20.0,
        phase=GamePhase.BATTLE_1X,
        elixir=6.0,
        hand=_hand(grid, ["spear_goblins", "goblin_gang", "bats", "zap"], 6.0),
        units=[knight],
        towers=_towers(grid),
    )
    first_play = brain.decide(s3)
    assert first_play is not None

    s4_dropped = BattleState(
        timestamp=20.95,
        phase=GamePhase.BATTLE_1X,
        elixir=6.3,  # Elixir went UP instead of dropping -> ADB tap was dropped!
        hand=_hand(grid, ["spear_goblins", "goblin_gang", "bats", "zap"], 6.3),
        units=[knight],
        towers=_towers(grid),
    )
    retry_play = brain.decide(s4_dropped)
    assert brain.pending.dropped_tap_retries == 1
    assert retry_play is not None


def test_intercepts_bridge_princess_immediately() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=2)
    # Princess standing at enemy side of right bridge (14.5, 14.8) outranging our Princess Tower!
    princess = DetectedUnit(track_id=5, name="princess", side=Side.ENEMY, tile_x=14.5, tile_y=14.8)
    state = BattleState(
        timestamp=12.0,
        phase=GamePhase.BATTLE_1X,
        elixir=6.0,
        hand=_hand(grid, ["goblin_gang", "miner", "skeleton_barrel", "zap"], 6.0),
        units=[princess],
        towers=_towers(grid),
    )
    act = brain.decide(state)
    assert act is not None
    assert act.card_name in ("goblin_gang", "miner")
    assert act.tile_x > 12.0


def test_never_drops_troops_to_catch_incoming_fireball() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=3)
    fireball = DetectedUnit(track_id=9, name="fireball", side=Side.ENEMY, tile_x=14.5, tile_y=25.5)
    state = BattleState(
        timestamp=12.0,
        phase=GamePhase.BATTLE_1X,
        elixir=5.0,
        hand=_hand(grid, ["spear_goblins", "bats", "goblin_gang", "zap"], 5.0),
        units=[fireball],
        towers=_towers(grid),
    )
    assert brain.decide(state) is None


def test_ignores_already_zapped_10hp_goblin_barrel_remnants() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=4)
    # 10 HP / 202 HP = 0.05 hp_ratio -> Princess Tower finishes in 1 shot!
    zapped_goblin = DetectedUnit(
        track_id=11, name="goblin_barrel", side=Side.ENEMY, tile_x=3.5, tile_y=25.2, hp_ratio=0.05
    )
    state = BattleState(
        timestamp=15.0,
        phase=GamePhase.BATTLE_1X,
        elixir=5.0,
        hand=_hand(grid, ["goblin_gang", "bats", "spear_goblins", "zap"], 5.0),
        units=[zapped_goblin],
        towers=_towers(grid),
    )
    assert brain.decide(state) is None


def test_full_match_victories_under_180ms_adb_lag() -> None:
    cfg = BotConfig()
    deck = cfg.normalized_deck()
    grid = ArenaGrid()

    for arch in ("hog_26", "pekka_bridge_spam"):
        sim = ClashRoyaleSimEngine(deck, META_DECKS[arch], seed=1000)
        opp_ai = MetaOpponentAI(arch, seed=2000)
        brain = MKZapBaitBrain(grid, cfg, seed=0, adb_latency_sec=0.18)
        opp_tr = OpponentTracker(bait_window_seconds=cfg.bait_window_seconds)
        opp_tr.reset(0.0)
        queued_adb = None

        while sim.step():
            opp_ai.step_ai(sim)
            if queued_adb is not None and sim.time >= queued_adb[0]:
                _, slot_idx, px, py, reason = queued_adb
                queued_adb = None
                tx, ty = grid.pixel_to_tile(px, py)
                sim.play_card(Side.ALLY, sim.ally.hand[slot_idx], tx, ty, reason)

            st = sim.export_fog_of_war_state(opp_tr.state)
            st.opponent = opp_tr.update(sim.time, sim.phase, st.units)
            if queued_adb is None:
                act = brain.decide(st)
                if act is not None:
                    ax, ay = grid.tile_to_pixel(act.tile_x, act.tile_y)
                    queued_adb = (sim.time + 0.18, act.slot_index, ax, ay, act.reason)

        assert sim.winner == Side.ALLY, f"Expected victory vs {arch}, got {sim.end_reason}"
        assert sim.ally.total_elixir_leaked < 0.25
