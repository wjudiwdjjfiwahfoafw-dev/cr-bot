"""Tactical scenario tests verifying pro-level MK Zap Bait decisions & anti-blunder rules."""

from __future__ import annotations

from typing import Dict, List

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


def _towers(grid: ArenaGrid) -> Dict[str, TowerState]:
    out: Dict[str, TowerState] = {}
    for name, (tx, ty) in grid.TOWER_TILES.items():
        side = Side.ENEMY if name.startswith("enemy") else Side.ALLY
        lane = Lane.CENTER if "king" in name else (Lane.LEFT if "left" in name else Lane.RIGHT)
        max_hp = 4824 if "king" in name else 3052
        out[name] = TowerState(name=name, side=side, lane=lane, alive=True, hp_ratio=1.0, max_hp=max_hp, tile_x=tx, tile_y=ty)
    return out


def _hand(grid: ArenaGrid, names: List[str], elixir: float, evo_first: bool = False) -> List[HandSlot]:
    slots: List[HandSlot] = []
    for i, cname in enumerate(names):
        info = get_card_info(cname)
        slots.append(
            HandSlot(
                slot_index=i,
                card_name=cname,
                elixir_cost=info.elixir,
                is_ready=(elixir >= info.elixir),
                is_evolved=(evo_first and i == 0),
                tap_xy=grid.hand_slot_pixel(i),
            )
        )
    return slots


def _unit(
    tid: int,
    name: str,
    side: Side,
    tx: float,
    ty: float,
    vy: float = 1.2,
    first_ts: float = 5.0,
    now_ts: float = 8.0,
) -> DetectedUnit:
    info = get_card_info(name)
    return DetectedUnit(
        track_id=tid,
        name=name,
        side=side,
        tile_x=tx,
        tile_y=ty,
        hp_ratio=0.95,
        move_type=info.move_type,
        weight_class=info.weight_class,
        vy=vy,
        first_seen_ts=first_ts,
        last_seen_ts=now_ts,
    )


def test_never_drops_mega_knight_into_enemy_pekka() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=1)
    state = BattleState(
        timestamp=10.0,
        phase=GamePhase.BATTLE_1X,
        elixir=8.5,
        hand=_hand(grid, ["mega_knight", "inferno_dragon", "spear_goblins", "zap"], elixir=8.5, evo_first=True),
        units=[_unit(1, "pekka", Side.ENEMY, 3.5, 18.4)],
        towers=_towers(grid),
    )
    action = brain.decide(state)
    assert action is not None
    assert action.card_name in ("inferno_dragon", "spear_goblins")
    assert action.card_name != "mega_knight"


def test_never_wastes_mega_knight_on_air_balloon() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=1)
    state = BattleState(
        timestamp=10.0,
        phase=GamePhase.BATTLE_1X,
        elixir=8.0,
        hand=_hand(grid, ["mega_knight", "bats", "miner", "zap"], elixir=8.0),
        units=[_unit(1, "balloon", Side.ENEMY, 14.5, 18.0, vy=1.5)],
        towers=_towers(grid),
    )
    action = brain.decide(state)
    assert action is not None
    assert action.card_name == "bats"


def test_crushes_hog_plus_musketeer_push_with_mega_knight() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=1)
    state = BattleState(
        timestamp=10.0,
        phase=GamePhase.BATTLE_1X,
        elixir=7.5,
        hand=_hand(grid, ["mega_knight", "miner", "spear_goblins", "zap"], elixir=7.5, evo_first=True),
        units=[
            _unit(1, "hog_rider", Side.ENEMY, 14.5, 18.5, vy=2.2),
            _unit(2, "musketeer", Side.ENEMY, 14.3, 17.6, vy=1.2),
        ],
        towers=_towers(grid),
    )
    action = brain.decide(state)
    assert action is not None
    assert action.card_name == "mega_knight"
    assert action.tile_x > 12.5
    assert action.tile_y >= 17.3  # Strictly on our side of the river!


def test_skeleton_barrel_queues_delayed_anti_tornado_miner_combo() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=7)
    opp = OpponentCycleState(estimated_elixir=4.0, small_spell_baited=True, cards_since_small_spell=1)
    towers = _towers(grid)
    towers["enemy_left"].hp_ratio = 0.60  # Left tower is weaker -> should attack left!

    state_t1 = BattleState(
        timestamp=10.0,
        phase=GamePhase.BATTLE_1X,
        elixir=6.5,
        hand=_hand(grid, ["skeleton_barrel", "miner", "bats", "zap"], elixir=6.5),
        units=[],
        towers=towers,
        opponent=opp,
    )
    act1 = brain.decide(state_t1)
    assert act1 is not None
    assert act1.card_name == "skeleton_barrel"
    assert act1.tile_x < 6.0  # Left bridge!
    assert act1.followup is not None
    assert act1.followup.card_name == "miner"

    # 1.0s later (t=11.0): before 1.8s delay -> should hold elixir for Miner!
    state_hold = BattleState(
        timestamp=11.0,
        phase=GamePhase.BATTLE_1X,
        elixir=3.9,
        hand=_hand(grid, ["goblin_gang", "miner", "bats", "zap"], elixir=3.9),
        units=[_unit(10, "skeleton_barrel", Side.ALLY, 3.5, 13.0, vy=-1.8)],
        towers=towers,
        opponent=opp,
    )
    assert brain.decide(state_hold) is None

    # At t=11.9s: queued Miner fires onto safe outer left corner (x <= 3.0)!
    state_t2 = BattleState(
        timestamp=11.9,
        phase=GamePhase.BATTLE_1X,
        elixir=4.2,
        hand=_hand(grid, ["goblin_gang", "miner", "bats", "zap"], elixir=4.2),
        units=[_unit(10, "skeleton_barrel", Side.ALLY, 3.5, 9.5, vy=-1.8)],
        towers=towers,
        opponent=opp,
    )
    act2 = brain.decide(state_t2)
    assert act2 is not None
    assert act2.card_name == "miner"
    assert act2.tile_x <= 3.0
    assert "[COMBO STEP 2]" in act2.reason


def test_emergency_goblin_barrel_aborts_queued_combo_and_zaps() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=7)
    opp = OpponentCycleState(estimated_elixir=4.0, small_spell_baited=True, cards_since_small_spell=1)
    towers = _towers(grid)

    # Trigger combo step 1
    s1 = BattleState(
        timestamp=10.0,
        phase=GamePhase.BATTLE_1X,
        elixir=6.5,
        hand=_hand(grid, ["skeleton_barrel", "miner", "bats", "zap"], elixir=6.5),
        units=[],
        towers=towers,
        opponent=opp,
    )
    assert brain.decide(s1) is not None

    # Opponent throws Goblin Barrel directly on our right Princess Tower (14.5, 25.5) at t=11.0!
    s_emerg = BattleState(
        timestamp=11.0,
        phase=GamePhase.BATTLE_1X,
        elixir=4.0,
        hand=_hand(grid, ["goblin_gang", "miner", "bats", "zap"], elixir=4.0),
        units=[_unit(99, "goblin_barrel", Side.ENEMY, 14.5, 25.4, vy=0.0)],
        towers=towers,
        opponent=opp,
    )
    emerg_act = brain.decide(s_emerg)
    assert emerg_act is not None
    assert emerg_act.card_name == "zap"
    assert abs(emerg_act.tile_x - 14.5) < 1.0


def test_timed_zap_resets_inferno_tower_not_too_early() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=3)
    towers = _towers(grid)

    # Inferno Tower just spawned 0.4s ago -> too early to reset!
    early_state = BattleState(
        timestamp=10.4,
        phase=GamePhase.BATTLE_1X,
        elixir=4.0,
        hand=_hand(grid, ["zap", "spear_goblins", "bats", "goblin_gang"], elixir=4.0),
        units=[
            _unit(1, "mega_knight", Side.ALLY, 3.8, 12.0, vy=-1.0, first_ts=5.0, now_ts=10.4),
            _unit(2, "inferno_tower", Side.ENEMY, 7.2, 10.5, vy=0.0, first_ts=10.0, now_ts=10.4),
        ],
        towers=towers,
    )
    early_act = brain.decide(early_state)
    assert early_act is None or early_act.card_name != "zap"

    # At t=12.1 (2.1s locked onto our Mega Knight) -> Zap resets the beam!
    ripe_state = BattleState(
        timestamp=12.1,
        phase=GamePhase.BATTLE_1X,
        elixir=4.5,
        hand=_hand(grid, ["zap", "spear_goblins", "bats", "goblin_gang"], elixir=4.5),
        units=[
            _unit(1, "mega_knight", Side.ALLY, 4.5, 11.2, vy=-0.5, first_ts=5.0, now_ts=12.1),
            _unit(2, "inferno_tower", Side.ENEMY, 7.2, 10.5, vy=0.0, first_ts=10.0, now_ts=12.1),
        ],
        towers=towers,
    )
    ripe_act = brain.decide(ripe_state)
    assert ripe_act is not None
    assert ripe_act.card_name == "zap"
    assert "reset" in ripe_act.reason.lower()


def test_punishes_opposite_lane_when_opponent_drops_golem_in_back() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=5)
    state = BattleState(
        timestamp=15.0,
        phase=GamePhase.BATTLE_1X,
        elixir=6.5,
        hand=_hand(grid, ["skeleton_barrel", "miner", "inferno_dragon", "zap"], elixir=6.5),
        units=[_unit(1, "golem", Side.ENEMY, 3.5, 3.8, vy=0.6)],  # Left backline Golem!
        towers=_towers(grid),
    )
    action = brain.decide(state)
    assert action is not None
    assert action.card_name == "skeleton_barrel"
    assert action.tile_x > 13.0  # Punishes RIGHT bridge!


def test_never_cycles_naked_mega_knight_in_single_elixir() -> None:
    grid = ArenaGrid()
    brain = MKZapBaitBrain(grid, BotConfig(), seed=9)
    state = BattleState(
        timestamp=20.0,
        phase=GamePhase.BATTLE_1X,
        elixir=9.6,
        hand=_hand(grid, ["mega_knight", "zap", "inferno_dragon", "spear_goblins"], elixir=9.6),
        units=[],
        towers=_towers(grid),
    )
    action = brain.decide(state)
    assert action is not None
    assert action.card_name == "spear_goblins"
    assert abs(action.tile_x - 9.0) < 0.5  # Splits behind King!
