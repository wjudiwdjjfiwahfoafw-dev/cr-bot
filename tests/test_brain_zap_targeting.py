"""End-to-end checks on what the brain is allowed to do with a damage spell.

Unit tests cover :func:`cr_bot.brain.spell_targeting.aim_is_wasted`; these drive the real brain
through :meth:`MKZapBaitBrain.decide` so the scoring paths, the hand cycle, the elixir gate and
the last-mile guard are all exercised together. The reported bug — Zap cast into an empty corner
of the arena — is a *system* property, so it needs a system-level test.
"""

from __future__ import annotations

import pytest

from cr_bot.arena.device_calibrator import ArenaCalibration
from cr_bot.arena.grid import ArenaGrid
from cr_bot.arena.projection import RectProjection
from cr_bot.brain.mk_zap_bait_brain import MKZapBaitBrain
from cr_bot.config.cards_db import get_card_info
from cr_bot.config.loader import BotConfig
from cr_bot.models import (
    BattleState,
    DetectedUnit,
    GamePhase,
    HandSlot,
    OpponentCycleState,
    Side,
    WeightClass,
)

WIDTH, HEIGHT = 720, 1600


def _grid() -> ArenaGrid:
    cal = ArenaCalibration(
        frame_size=(WIDTH, HEIGHT),
        projection=RectProjection(25, 124, WIDTH, 1263),
        hand_slots_px=[(224.0, 1466.0), (360.0, 1466.0), (496.0, 1466.0), (630.0, 1466.0)],
        hand_card_size_px=(127.0, 114.0),
        elixir_px=(146.0, 693.0, 1574.0),
        river_px_y=710.5,
    )
    return ArenaGrid(WIDTH, HEIGHT, cal)


def _brain() -> MKZapBaitBrain:
    return MKZapBaitBrain(_grid(), BotConfig())


def _state(units, elixir=10.0, cards=("zap", "bats", "goblin_gang", "miner"),
           timestamp=10.0) -> BattleState:
    hand = [HandSlot(i, card, get_card_info(card).elixir) for i, card in enumerate(cards)]
    return BattleState(
        timestamp=timestamp, phase=GamePhase.BATTLE_1X, elixir=elixir,
        hand=hand, units=units, towers={}, opponent=OpponentCycleState(),
    )


def _swarm(tile_x=6.0, tile_y=24.0, name="bats") -> DetectedUnit:
    return DetectedUnit(
        track_id=1, name=name, side=Side.ENEMY, tile_x=tile_x, tile_y=tile_y,
        weight_class=WeightClass.SWARM, hp_ratio=1.0, vy=0.2, source="detector",
        first_seen_ts=8.0, last_seen_ts=10.0,
    )


def _decide_over(brain, state, frames: int = 10, step: float = 0.5):
    """Let the brain run for a while; returns the list of chosen actions."""
    actions = []
    for i in range(frames):
        state.timestamp = 10.0 + i * step
        action = brain.decide(state)
        if action is not None:
            actions.append(action)
    return actions


def test_zap_is_never_cast_into_an_empty_arena() -> None:
    """The headline bug: full elixir, empty board, Zap in hand. It must find something else."""
    brain = _brain()
    actions = _decide_over(brain, _state([], elixir=10.0))
    zaps = [a for a in actions if a.card_name == "zap"]
    assert not zaps, f"Zap was cast with nothing on the board: {zaps}"
    assert actions, "the bot must still spend elixir rather than sit frozen at 10"


def test_zap_is_still_cast_when_a_swarm_reaches_our_tower() -> None:
    """The guard must not make the bot passive: a swarm on our tower is a real Zap target."""
    brain = _brain()
    actions = _decide_over(brain, _state([_swarm()], elixir=6.0))
    zaps = [a for a in actions if a.card_name == "zap"]
    assert zaps, "a Bats swarm on our princess tower should be Zapped"
    aimed = zaps[0]
    assert abs(aimed.tile_x - 6.0) < 2.0 and abs(aimed.tile_y - 24.0) < 2.5, (
        f"Zap aimed at ({aimed.tile_x}, {aimed.tile_y}), far from the Bats at (6.0, 24.0)")


def test_zap_is_not_spent_on_a_unit_at_the_far_end_of_the_arena() -> None:
    """A lone Giant walking in at the back is not a Zap target; it is not even in range."""
    brain = _brain()
    giant = DetectedUnit(
        track_id=2, name="giant", side=Side.ENEMY, tile_x=9.0, tile_y=2.0,
        weight_class=WeightClass.HEAVY_TANK, hp_ratio=1.0, source="detector",
        first_seen_ts=9.0, last_seen_ts=10.0,
    )
    actions = _decide_over(brain, _state([giant], elixir=10.0))
    assert not [a for a in actions if a.card_name == "zap"]


def test_every_spell_the_brain_plays_lands_on_something() -> None:
    """Cross-check the guard over a mixed board: no action may be a wasted cast."""
    from cr_bot.brain.spell_targeting import aim_is_wasted, is_damage_spell

    brain = _brain()
    board = [_swarm(5.0, 22.0), _swarm(13.0, 27.0, name="skeletons"),
             DetectedUnit(track_id=3, name="inferno_dragon", side=Side.ENEMY,
                          tile_x=11.0, tile_y=20.0, weight_class=WeightClass.MEDIUM,
                          hp_ratio=1.0, source="detector",
                          first_seen_ts=0.0, last_seen_ts=10.0)]
    state = _state(board, elixir=8.0)
    for action in _decide_over(brain, state, frames=20):
        if not is_damage_spell(action.card_name):
            continue
        wasted, why = aim_is_wasted(action.card_name, action.tile_x, action.tile_y,
                                    state.enemy_units)
        assert not wasted, f"brain produced a wasted cast: {why}"
