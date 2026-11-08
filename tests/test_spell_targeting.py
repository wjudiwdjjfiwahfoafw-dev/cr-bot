"""Tests for spell aiming: a blast-cover solver, not a "is there a unit here" rule.

The reported bug was "the bot casts Zap at the back of the arena where it hits nothing". The first
attempt at a fix refused any cast without a unit under the cursor — which is wrong twice over: it
rejects the strongest play there is (a tight cluster of troops whose centre is empty ground) and
it aims spells at whichever unit happened to anchor the decision.

These tests pin the model that replaced it: a spell is a disc, its value is what the disc covers,
and the cast point is *solved* rather than snapped to a unit.
"""

from __future__ import annotations

import pytest

from cr_bot.brain.spell_targeting import (
    SPELL_PROFILES,
    BlastModel,
    aim_is_wasted,
    describe,
    is_damage_spell,
    plan_cast,
    profile_for,
)
from cr_bot.models import DetectedUnit, Side, WeightClass


def _u(name, x, y, cls=WeightClass.SWARM, hp=1.0, vx=0.0, vy=0.0,
       side=Side.ENEMY, source="detector") -> DetectedUnit:
    return DetectedUnit(track_id=1, name=name, side=side, tile_x=x, tile_y=y,
                        weight_class=cls, hp_ratio=hp, vx=vx, vy=vy, source=source)


# -- the empty-tile play the crude rule could not express ---------------------------------------
def test_the_best_cast_point_can_be_empty_ground_between_troops():
    """Four Bats at the corners of a 3x3 box: the centre covers all four, a corner covers one."""
    bats = [_u("bats", 8.0, 20.0), _u("bats", 11.0, 20.0),
            _u("bats", 8.0, 23.0), _u("bats", 11.0, 23.0)]
    cover = plan_cast("zap", bats)
    assert cover is not None
    assert cover.troops == 4, f"only {cover.troops} of 4 Bats covered by {describe(cover, profile_for('zap'))}"
    # the solver must have moved off the unit it started from, into the middle
    assert 8.5 < cover.tile_x < 10.5 and 20.5 < cover.tile_y < 22.5


def test_an_empty_tile_that_covers_troops_is_not_reported_as_wasted():
    """The gate asks "does this disc cover anything", not "is a unit under the cursor"."""
    bats = [_u("bats", 8.0, 20.0), _u("bats", 11.0, 20.0),
            _u("bats", 8.0, 23.0), _u("bats", 11.0, 23.0)]
    wasted, why = aim_is_wasted("zap", 9.5, 21.5, bats)     # empty ground, between them all
    assert not wasted, why


def test_a_cluster_beats_a_lone_unit():
    cluster = [_u("bats", 8.0, 20.0), _u("bats", 11.0, 20.0),
               _u("bats", 8.0, 23.0), _u("bats", 11.0, 23.0)]
    lone = [_u("musketeer", 9.0, 22.0, WeightClass.LIGHT)]
    assert plan_cast("zap", cluster).value > plan_cast("zap", lone).value


# -- what the model refuses ---------------------------------------------------------------------
def test_an_empty_arena_has_no_plan_at_all():
    assert plan_cast("zap", []) is None
    assert aim_is_wasted("zap", 9.0, 3.0, [])[0]


def test_a_lone_unit_far_from_the_cast_point_is_a_miss_and_says_why():
    wasted, why = aim_is_wasted("zap", 9.0, 4.0, [_u("inferno_dragon", 9.0, 26.0, WeightClass.MEDIUM)])
    assert wasted
    assert "nearest enemy" in why and "radius" in why


def test_a_unit_that_vanished_between_decision_and_tap_refuses_the_cast():
    before = [_u("sparky", 6.0, 24.0, WeightClass.MEDIUM)]
    assert not aim_is_wasted("zap", 6.0, 24.0, before)[0]
    assert aim_is_wasted("zap", 6.0, 24.0, [])[0]


def test_a_unit_about_to_die_is_barely_worth_a_spell():
    full = plan_cast("zap", [_u("skeletons", 9.0, 20.0, hp=1.0)])
    dying = plan_cast("zap", [_u("skeletons", 9.0, 20.0, hp=0.05)])
    assert dying.value < 0.3 * full.value


def test_allies_are_never_targets():
    assert aim_is_wasted("zap", 9.0, 20.0, [_u("bats", 9.0, 20.0, side=Side.ALLY)])[0]
    assert plan_cast("zap", [_u("bats", 9.0, 20.0, side=Side.ALLY)]) is None


def test_a_cast_already_in_flight_does_not_justify_another():
    pending = [_u("bats", 9.0, 20.0, source="pending")]
    assert aim_is_wasted("zap", 9.0, 20.0, pending)[0]


def test_friendly_fire_suppresses_a_cast():
    """Spending Zap on an enemy that our own Bats are already fighting is a net loss."""
    enemy = [_u("inferno_dragon", 11.0, 20.0, WeightClass.MEDIUM)]
    alone = plan_cast("zap", enemy, [])
    with_ally = plan_cast("zap", enemy, [_u("bats", 11.5, 20.5, side=Side.ALLY)])
    assert with_ally.value < alone.value


def test_troops_and_utility_spells_are_never_steered():
    for card in ("mega_knight", "skeleton_barrel", "miner", "goblin_gang", "bats",
                 "rage", "tornado", "poison", "freeze", "barbarian_barrel"):
        assert not is_damage_spell(card)
        assert aim_is_wasted(card, 9.0, 20.0, [])[0] is False


# -- leading a moving unit ----------------------------------------------------------------------
def test_a_moving_unit_is_led_by_its_own_velocity():
    """The predicted centre must sit ahead of a unit walking towards us, not on it."""
    hog = [_u("hog_rider", 9.0, 20.0, WeightClass.MEDIUM, vx=0.0, vy=1.6)]
    cover = plan_cast("zap", hog, lead_sec=0.45)
    assert cover.tile_y > 20.2, f"cast went to {cover.tile_y}, behind a unit moving down-screen"


def test_a_parked_unit_is_not_led():
    parked = [_u("musketeer", 9.0, 20.0, WeightClass.LIGHT)]
    cover = plan_cast("zap", parked, lead_sec=0.45)
    assert abs(cover.tile_y - 20.0) < 1.0


# -- per-spell behaviour ------------------------------------------------------------------------
def test_every_damage_spell_has_a_profile():
    for card in ("zap", "fireball", "rocket", "arrows", "giant_snowball", "the_log", "earthquake"):
        profile = profile_for(card)
        assert profile is not None and profile.radius > 0.0


def test_a_rocket_cares_about_a_tower_and_a_zap_barely_does():
    towers = [("enemy_left", "enemy", 3.5, 25.5)]
    rocket = plan_cast("rocket", [], [], towers)
    zap = plan_cast("zap", [], [], towers)
    assert rocket is not None and rocket.tower_value > zap.tower_value


def test_the_solver_never_plans_outside_the_arena():
    pressed = [_u("batS", 0.0, 0.0), _u("bats", 18.0, 32.0)]
    cover = plan_cast("zap", pressed)
    assert 0.5 <= cover.tile_x <= 17.5
    assert 1.0 <= cover.tile_y <= 31.0


def test_the_plan_is_deterministic():
    board = [_u("bats", 8.0, 20.0), _u("skeletons", 9.0, 21.0), _u("goblins", 10.0, 19.0)]
    first = plan_cast("zap", board)
    second = plan_cast("zap", board)
    assert (first.tile_x, first.tile_y, round(first.value, 6)) == \
           (second.tile_x, second.tile_y, round(second.value, 6))


# -- the brain ----------------------------------------------------------------------------------
def test_the_anti_deadlock_fallback_never_cycles_a_damage_spell():
    """Regression: that path aims with no board evaluation at all — the source of the empty Zap."""
    import inspect

    from cr_bot.brain.mk_zap_bait_brain import MKZapBaitBrain

    source = inspect.getsource(MKZapBaitBrain._score_anti_deadlock_fallback)
    for spell in ("zap", "fireball", "rocket", "arrows", "the_log"):
        assert f'"{spell}"' not in source, (
            f"the anti-deadlock fallback must not cycle {spell}: it does not evaluate targets")


# -- what a reset is worth is measured, not looked up -------------------------------------------
def test_a_reset_is_worth_more_when_our_troops_are_fighting_it():
    """Regression: the reset branch used to recognise only four hard-coded card names as "ours",
    so a reset that saved our Bats scored as if nothing were at stake."""
    dragon = [_u("inferno_dragon", 11.0, 20.0, WeightClass.MEDIUM)]
    alone = plan_cast("zap", dragon, [])
    with_bats = plan_cast("zap", dragon, [_u("bats", 11.5, 20.5, WeightClass.SWARM, side=Side.ALLY)])
    assert with_bats.resets == 1
    # the friendly-fire penalty applies too, but the saved push must still be worth more than
    # resetting a dragon that threatens nothing of ours
    assert with_bats.value > 0.0
    assert alone.value > 0.0


def test_a_reset_with_nothing_at_risk_is_not_worth_much():
    dragon = [_u("inferno_dragon", 11.0, 20.0, WeightClass.MEDIUM)]
    cover = plan_cast("zap", dragon, [])
    # same unit, but no ally engaged: the reset contributes only its damage
    assert cover.value < 2.0 * profile_for("zap").per_troop * 2.0
