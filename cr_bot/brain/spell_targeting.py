"""Where should a damage spell be cast?  A blast-cover solver, not a target rule.

The bug that started this
-------------------------
The bot kept casting Zap at the back of the arena where it hit nothing.  The first fix was a rule
— "refuse the cast unless an enemy is within N tiles of the tap" — and that rule was wrong in two
directions, which is why it is not what this module does:

* it **refused good plays**: a tight cluster of troops whose centre is empty ground (a
  Skeleton Army spread around a tile, two Bats flanking a spot) has huge spell value and no unit
  standing on the perfect cast point;
* it **papered over bad aiming**: pointing the spell at a unit's own tile is only correct by
  coincidence.  The unit you clicked is not necessarily the one worth covering.

What a spell actually is
------------------------
A disc of a known radius.  Its value is the *sum of what that disc covers*: how many troops, how
much health, whether any of them is a unit whose ability must be reset (Inferno Tower / Dragon,
Sparky, Mighty Miner), and whether it also lands on our own troops — friendly fire counts against
it.  So the right question is not "is there a unit at this point" but:

    for every candidate cast point, what does this disc cover, and is that worth the elixir?

That is a small, cheap optimisation, and it answers the empty-tile case by construction: an empty
tile *between* four troops is the best cast point there is, and it wins on value.

Design
------
``plan_cast`` enumerates candidate centres (each enemy's predicted position, the centroid of each
group of mutually close enemies, and the local maxima of a coarse coverage surface), scores every
one with :meth:`BlastModel.value`, and returns the best plan with a legible reason.  Nothing here
knows the name "zap" except to look up a radius in ``SPELL_PROFILES``, so the same solver serves
Snowball, Arrows, Fireball and Rocket, and a new spell only needs a profile.

The solver is pure geometry and arithmetic — no training, no learned weights — and it is used by
the brain for both the *aim* and the *decision*: a plan whose value does not justify its elixir is
simply not returned, which replaces the old hard-coded "min targets" rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cr_bot.config.cards_db import get_card_info
from cr_bot.models import DetectedUnit, Side

#: how far a unit is assumed to keep walking between the decision and the spell landing.  Uses the
#: unit's own reported velocity, so a fast Hog Rider is led further than a parked Musketeer.
DEFAULT_LEAD_SEC = 0.45
MAX_LEAD_SEC = 1.35

#: units that are not worth spending a spell on: a cast already in flight, a spawn marker
IGNORED_SOURCES = frozenset({"pending"})

#: An enemy counts as "worth hitting" at this fraction of full health — a spell on a unit that is
#: about to die anyway is elixir burned for nothing.
MIN_HP_TO_BE_WORTH_HITTING = 0.18


@dataclass(frozen=True)
class SpellProfile:
    """The blast geometry and the valuation weights of one damage spell."""

    name: str
    radius: float
    #: tower damage as a share of a princess tower's HP (Rocket ~0.6, Fireball ~0.25, Zap ~0.05)
    tower_damage: float
    #: how much one full-health light troop is worth to hit, in "value points"
    per_troop: float
    #: multiplier on the value of resetting an Inferno/Sparky-style ability
    reset_bonus: float
    #: multiplier on tower chip value
    tower_weight: float
    #: true when the spell's whole purpose is the damage (i.e. it needs something to hit)
    needs_target: bool = True


SPELL_PROFILES: Dict[str, SpellProfile] = {
    # radii are the real in-game blast radii, in tiles
    "zap": SpellProfile("zap", 2.5, tower_damage=0.05, per_troop=16.0,
                        reset_bonus=1.9, tower_weight=0.25),
    "giant_snowball": SpellProfile("giant_snowball", 2.5, tower_damage=0.08, per_troop=14.0,
                                   reset_bonus=1.2, tower_weight=0.3),
    "the_log": SpellProfile("the_log", 1.6, tower_damage=0.10, per_troop=18.0,
                            reset_bonus=1.0, tower_weight=0.35),
    "arrows": SpellProfile("arrows", 3.5, tower_damage=0.08, per_troop=13.0,
                           reset_bonus=1.0, tower_weight=0.3),
    "fireball": SpellProfile("fireball", 2.5, tower_damage=0.26, per_troop=24.0,
                             reset_bonus=1.0, tower_weight=0.7),
    "rocket": SpellProfile("rocket", 2.0, tower_damage=0.62, per_troop=40.0,
                           reset_bonus=1.0, tower_weight=1.0),
    "earthquake": SpellProfile("earthquake", 3.0, tower_damage=0.18, per_troop=15.0,
                               reset_bonus=0.5, tower_weight=0.4),
}

#: cards that are *not* damage spells and must never be steered by this module
NOT_A_DAMAGE_SPELL = frozenset({
    "rage", "tornado", "poison", "freeze", "graveyard", "mirror", "clone", "barbarian_barrel",
})


def profile_for(card_name: str) -> Optional[SpellProfile]:
    return SPELL_PROFILES.get(card_name)


def is_damage_spell(card_name: str) -> bool:
    return card_name in SPELL_PROFILES


# ---------------------------------------------------------------------------------------------
# the value model
# ---------------------------------------------------------------------------------------------

@dataclass
class Coverage:
    """What one candidate blast actually covers."""

    tile_x: float
    tile_y: float
    value: float = 0.0
    troops: int = 0
    resets: int = 0
    hp_covered: float = 0.0
    friendly: int = 0
    tower_value: float = 0.0
    hits: List[DetectedUnit] = field(default_factory=list)

    @property
    def is_useful(self) -> bool:
        return self.troops > 0 or self.tower_value > 0.0


def _unit_weight(unit: DetectedUnit) -> float:
    """How much health a single unit contributes to the value of hitting it.

    Normalised so that a full-health light troop counts 1.0.  This is what makes a Zap on four
    Bats better than a Zap on one Musketeer even though both are "a unit in the radius".
    """
    base = 1.0
    weight_class = getattr(unit, "weight_class", None)
    name = str(getattr(weight_class, "value", weight_class))
    base *= {"swarm": 0.55, "light": 1.0, "medium": 2.0, "heavy_tank": 3.2}.get(name, 1.0)
    # a big unit that is at 30% HP is worth much less than a full-health one
    return base * max(MIN_HP_TO_BE_WORTH_HITTING, float(unit.hp_ratio))


#: how far from a resettable enemy our unit still counts as "fighting it" (tiles)
ENGAGEMENT_TILES = 3.0


def _stake_in_reset(unit: DetectedUnit, allies: Sequence[DetectedUnit]) -> float:
    """What a reset of ``unit`` would actually save, in units of a light troop.

    Resetting an Inferno Dragon that is burning our Mega Knight is worth much more than resetting
    one that is walking alone.  This used to be a hard-coded list of four card names ("our Mega
    Knight / Miner / Skeleton Barrel / Inferno Dragon"), which quietly ignored everything else the
    deck might have on the board — our Bats fighting a Sparky scored as if nothing were at stake.
    Measuring what is actually engaged generalises to any deck and needs no list.
    """
    stake = 0.0
    for ally in allies:
        if ally.side is not Side.ALLY or ally.source in IGNORED_SOURCES:
            continue
        if math.hypot(ally.tile_x - unit.tile_x, ally.tile_y - unit.tile_y) <= ENGAGEMENT_TILES:
            stake += _unit_weight(ally)
    return stake


def _is_resettable(unit: DetectedUnit) -> bool:
    try:
        return bool(get_card_info(unit.name).is_resettable)
    except Exception:
        return False


class BlastModel:
    """Scores a candidate cast point against the live board."""

    def __init__(self, profile: SpellProfile) -> None:
        self.profile = profile

    def value(
        self,
        tile_x: float,
        tile_y: float,
        enemies: Sequence[DetectedUnit],
        allies: Sequence[DetectedUnit] = (),
        towers: Optional[Sequence[Tuple[str, int, float, float]]] = None,
    ) -> Coverage:
        """Sum what a blast centred here covers.

        ``towers`` are ``(name, side, tile_x, tile_y)`` tuples; only enemy towers score, and only
        with the spell's own tower-damage weight (Zap is a chip spell, Rocket is a win condition).
        """
        cover = Coverage(tile_x=tile_x, tile_y=tile_y)
        radius = self.profile.radius
        for unit in enemies:
            if unit.side is not Side.ENEMY or unit.source in IGNORED_SOURCES:
                continue
            if math.hypot(unit.tile_x - tile_x, unit.tile_y - tile_y) > radius:
                continue
            cover.hits.append(unit)
            cover.troops += 1
            weight = _unit_weight(unit)
            cover.hp_covered += weight
            cover.value += self.profile.per_troop * weight
            if _is_resettable(unit) and self.profile.reset_bonus > 1.0:
                cover.resets += 1
                stake = _stake_in_reset(unit, allies)
                # a reset with nothing of ours at risk is just damage; with a push behind it, it
                # is one of the best plays in the game
                scale = 0.45 + 0.55 * min(1.0, stake / 2.0)
                cover.value += (self.profile.per_troop * weight
                                * (self.profile.reset_bonus - 1.0) * scale)

        # friendly fire: our own troops inside the same disc are a real cost, not a rounding error
        for ally in allies:
            if ally.side is not Side.ALLY or ally.source in IGNORED_SOURCES:
                continue
            if math.hypot(ally.tile_x - tile_x, ally.tile_y - tile_y) <= radius:
                cover.friendly += 1
                cover.value -= self.profile.per_troop * 0.8 * _unit_weight(ally)

        for _name, side, tower_x, tower_y in (towers or ()):
            if side != "enemy":
                continue
            if math.hypot(tower_x - tile_x, tower_y - tile_y) > radius:
                continue
            cover.tower_value += self.profile.tower_damage * 100.0 * self.profile.tower_weight
        cover.value += cover.tower_value
        return cover

    # -- candidate generation --------------------------------------------------------------
    def plan(
        self,
        enemies: Sequence[DetectedUnit],
        allies: Sequence[DetectedUnit] = (),
        towers: Optional[Sequence[Tuple[str, int, float, float]]] = None,
        lead_sec: float = DEFAULT_LEAD_SEC,
        bounds: Tuple[float, float, float, float] = (0.7, 1.2, 17.3, 31.0),
    ) -> Optional[Coverage]:
        """Best cast point available, or ``None`` when nothing here is worth the elixir.

        Candidate centres are the interesting places, not a grid sweep over the arena:

        * each enemy, at its predicted position (leading by its own velocity),
        * each enemy again *un-led*, because a stationary cluster's best point is not its units,
        * the centroid of every set of enemies that fit inside one blast,
        * the midpoint of every mutually-close pair (this is what finds the empty tile between
          four troops, which is the case a "hit a unit" rule cannot express).

        The best few are then refined by local hill-climbing, which fixes the last few percent
        without paying for a fine sweep.
        """
        live = [e for e in enemies
                if e.side is Side.ENEMY and e.source not in IGNORED_SOURCES]
        enemy_towers = [t for t in (towers or ()) if t[1] == "enemy"]
        if not live and not enemy_towers:
            return None
        if not live:
            # nothing to hit but a tower is in play: the best point is the tower itself
            cx, cy = enemy_towers[0][2], enemy_towers[0][3]
            cover = self.value(cx, cy, [], allies, towers)
            return cover if cover.is_useful else None

        lead = max(0.0, min(MAX_LEAD_SEC, lead_sec))
        predicted = [(e.tile_x + e.vx * lead, e.tile_y + e.vy * lead) for e in live]
        candidates: List[Tuple[float, float]] = list(predicted) + [(e.tile_x, e.tile_y) for e in live]

        # centroids of groups that fit inside one blast
        radius = self.profile.radius
        for i, (ax, ay) in enumerate(predicted):
            group = [predicted[i]]
            for j, (bx, by) in enumerate(predicted):
                if i != j and math.hypot(bx - ax, by - ay) <= 2.0 * radius:
                    group.append((bx, by))
            if len(group) > 1:
                candidates.append((sum(x for x, _ in group) / len(group),
                                   sum(y for _, y in group) / len(group)))
            for j in range(i + 1, len(predicted)):
                bx, by = predicted[j]
                if math.hypot(bx - ax, by - ay) <= 2.0 * radius:
                    candidates.append((0.5 * (ax + bx), 0.5 * (ay + by)))

        x1, y1, x2, y2 = bounds
        scored: List[Coverage] = []
        for cx, cy in candidates:
            cx = max(x1, min(x2, cx))
            cy = max(y1, min(y2, cy))
            scored.append(self.value(cx, cy, live, allies, towers))

        if not scored:
            return None
        scored.sort(key=lambda c: c.value, reverse=True)
        best = self._climb(scored[0], live, allies, towers, bounds)
        return best if best.is_useful else None

    def _climb(
        self,
        start: Coverage,
        enemies: Sequence[DetectedUnit],
        allies: Sequence[DetectedUnit],
        towers,
        bounds: Tuple[float, float, float, float],
        step: float = 0.5,
        rounds: int = 4,
    ) -> Coverage:
        """Local search around a candidate: cheap, and it converges on the true cluster centre."""
        x1, y1, x2, y2 = bounds
        best = start
        for _ in range(rounds):
            improved = False
            for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step),
                           (step, step), (step, -step), (-step, step), (-step, -step)):
                cx = max(x1, min(x2, best.tile_x + dx))
                cy = max(y1, min(y2, best.tile_y + dy))
                candidate = self.value(cx, cy, enemies, allies, towers)
                if candidate.value > best.value + 1e-6:
                    best = candidate
                    improved = True
            if not improved:
                step *= 0.5
        return best


def describe(cover: Optional[Coverage], profile: SpellProfile) -> str:
    """A one-line, human-readable justification for the log and the HUD."""
    if cover is None:
        return f"{profile.name}: no cast worth making"
    if cover.troops == 0 and cover.tower_value <= 0:
        return f"{profile.name}: nothing worth hitting"
    parts = []
    if cover.troops:
        parts.append(f"{cover.troops} unit{'s' if cover.troops != 1 else ''}")
    if cover.resets:
        parts.append(f"{cover.resets} reset")
    if cover.tower_value > 0:
        parts.append("tower chip")
    if cover.friendly:
        parts.append(f"{cover.friendly} friendly inside")
    names = ", ".join(sorted({u.name for u in cover.hits})[:3])
    return (f"{profile.name} covering {' + '.join(parts)}"
            f"{f' ({names})' if names else ''} "
            f"@({cover.tile_x:.1f},{cover.tile_y:.1f}) value {cover.value:.0f}")


# ---------------------------------------------------------------------------------------------
# compatibility helpers: the brain and the executor both ask these
# ---------------------------------------------------------------------------------------------

def aim_is_wasted(
    card_name: str,
    tile_x: float,
    tile_y: float,
    enemies: Sequence[DetectedUnit],
    min_targets: int = 1,
) -> Tuple[bool, str]:
    """Would a cast at this exact point be worth nothing?

    This is the last-mile gate.  It is *not* a "is a unit under the cursor" test — that is what
    the old rule did and why it refused good empty-tile casts.  It asks the same question the
    planner asks: does this disc cover anything of value?  A point between four troops passes; a
    point in an empty corner does not.
    """
    profile = profile_for(card_name)
    if profile is None or not profile.needs_target:
        return False, ""
    cover = BlastModel(profile).value(tile_x, tile_y, enemies)
    if cover.troops >= max(1, min_targets) or cover.tower_value > 0:
        return False, ""
    nearest = _nearest(enemy for enemy in enemies if enemy.side is Side.ENEMY)
    if nearest is None:
        return True, f"{card_name} aimed at empty ground ({tile_x:.1f},{tile_y:.1f})"
    dist = math.hypot(nearest.tile_x - tile_x, nearest.tile_y - tile_y)
    return True, (
        f"{card_name} would hit nothing: the disc covers no troops "
        f"(nearest enemy {dist:.1f} tiles away, radius {profile.radius:.1f})"
    )


def _nearest(enemies: Iterable[DetectedUnit]) -> Optional[DetectedUnit]:
    best: Optional[DetectedUnit] = None
    best_d = float("inf")
    for unit in enemies:
        if unit.source in IGNORED_SOURCES:
            continue
        d = math.hypot(unit.tile_x, unit.tile_y)
        if d < best_d:
            best_d, best = d, unit
    return best


def plan_cast(
    card_name: str,
    enemies: Sequence[DetectedUnit],
    allies: Sequence[DetectedUnit] = (),
    towers: Optional[Sequence[Tuple[str, int, float, float]]] = None,
    lead_sec: float = DEFAULT_LEAD_SEC,
    bounds: Tuple[float, float, float, float] = (0.7, 1.2, 17.3, 31.0),
) -> Optional[Coverage]:
    """Convenience wrapper: plan the best cast of ``card_name`` for this board."""
    profile = profile_for(card_name)
    if profile is None:
        return None
    return BlastModel(profile).plan(enemies, allies, towers, lead_sec=lead_sec, bounds=bounds)
