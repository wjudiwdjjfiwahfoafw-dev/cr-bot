"""
v2.0 Tactical Board & Lane Threat Analyzer:
- Kinematic lead-time intercept prediction (compensates ADB + server deploy/cast latency)
- Net lane defensive coverage accounting (including in-flight PendingDeployTracker units!)
- Non-catchable spell filtering + exact Goblin Barrel flight-age Zap timing
- Long-range bridge sniper detection (Princess / Magic Archer / Dart Goblin / Siege)
- Splash-danger & Electro-Wizard resetter isolation
- Dynamic King-Tower & Pocket protection when a Princess Tower falls
- Enemy defensive building & Inferno Tower distraction tracking
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cr_bot.arena.grid import ArenaGrid
from cr_bot.brain.spell_targeting import (
    DEFAULT_LEAD_SEC,
    describe,
    is_damage_spell,
    plan_cast,
    profile_for,
)
from cr_bot.config.cards_db import get_card_info
from cr_bot.models import BattleState, DetectedUnit, Lane, MoveType, Side, WeightClass


NON_CATCHABLE_SPELLS = {
    "fireball",
    "rocket",
    "the_log",
    "zap",
    "arrows",
    "giant_snowball",
    "barbarian_barrel",
    "lightning",
    "poison",
    "earthquake",
    "tornado",
    "freeze",
    "rage",
}

BRIDGE_SNIPER_MIN_Y = {
    "princess": 13.6,
    "x_bow": 13.8,
    "mortar": 13.8,
    "magic_archer": 14.6,
    "dart_goblin": 14.8,
    "royal_giant": 14.8,
}

KITEABLE_MELEE_SLUGS = {
    "pekka",
    "mini_pekka",
    "prince",
    "dark_prince",
    "elite_barbarians",
    "mega_knight",
    "knight",
    "valkyrie",
    "bandit",
    "royal_ghost",
    "lumberjack",
    "giant_skeleton",
    "golden_knight",
    "skeleton_king",
    "unknown_medium",
}

TANK_SLUGS = {
    "pekka",
    "golem",
    "giant",
    "electro_giant",
    "royal_giant",
    "goblin_giant",
    "lava_hound",
    "mega_knight",
    "giant_skeleton",
    "bowler",
    "sparky",
    "unknown_tank",
}

SNIPEABLE_SUPPORT_SLUGS = {
    "princess",
    "firecracker",
    "magic_archer",
    "dart_goblin",
    "electro_wizard",
    "ice_wizard",
    "musketeer",
    "wizard",
    "witch",
}

SPLASH_ANTI_SWARM_SLUGS = {
    "baby_dragon",
    "wizard",
    "witch",
    "executioner",
    "princess",
    "electro_wizard",
    "magic_archer",
    "firecracker",
    "ice_spirit",
    "electro_spirit",
    "fire_spirit",
    "valkyrie",
    "dark_prince",
    "royal_ghost",
}

DEFENSIVE_BUILDING_SLUGS = {
    "cannon",
    "tesla",
    "inferno_tower",
    "bomb_tower",
    "goblin_cage",
    "tombstone",
}


@dataclass
class MKStompCluster:
    tile_x: float
    tile_y: float
    elixir_value: float
    unit_count: int
    has_win_condition: bool
    near_pekka_danger: bool


@dataclass
class ZapOpportunity:
    tile_x: float
    tile_y: float
    value_score: float
    is_reset: bool
    reason: str


@dataclass
class LaneAssessment:
    lane: Lane
    def_anchor_xy: Tuple[float, float] = (3.5, 25.5)
    ally_princess_alive: bool = True
    enemy_units: List[DetectedUnit] = field(default_factory=list)
    ally_units: List[DetectedUnit] = field(default_factory=list)
    ground_threat_our_side: float = 0.0
    air_threat_our_side: float = 0.0
    ally_defense_coverage: float = 0.0
    unanswered_threat: float = 0.0
    urgency_score: float = 0.0
    has_splash_anti_swarm: bool = False
    best_mk_stomp: Optional[MKStompCluster] = None
    primary_tank: Optional[DetectedUnit] = None
    enemy_resetter_or_support: Optional[DetectedUnit] = None
    kite_target: Optional[DetectedUnit] = None
    tower_emergency_unit: Optional[DetectedUnit] = None
    barrel_ready_for_zap: bool = True
    bridge_sniper: Optional[DetectedUnit] = None
    bridge_rusher: Optional[DetectedUnit] = None
    counterpush_lead: Optional[DetectedUnit] = None
    ally_miner_on_enemy_tower: bool = False
    enemy_heavy_backline: Optional[DetectedUnit] = None
    enemy_approaching_tank: Optional[DetectedUnit] = None
    enemy_active_building: Optional[DetectedUnit] = None
    enemy_inferno_vs_mk: Optional[DetectedUnit] = None


@dataclass
class BoardAssessment:
    lanes: Dict[Lane, LaneAssessment]
    best_zap: Optional[ZapOpportunity]
    weaker_enemy_lane: Lane
    lowest_enemy_tower_hp: int


class ThreatAnalyzer:
    """Transforms BattleState + pending ally deploys into calibrated lane features for MKZapBaitBrain."""

    MK_STOMP_RADIUS = 2.25
    ZAP_RADIUS = 2.5

    def __init__(self, grid: ArenaGrid, adb_latency_sec: float = 0.12) -> None:
        self.grid = grid
        self.adb_latency_sec = adb_latency_sec

    def predict_intercept_tile(
        self,
        unit: DetectedUnit,
        card_delay_sec: float,
    ) -> Tuple[float, float]:
        """Predict where a moving unit will be when our spell/troop actually lands."""
        lead_t = max(0.0, min(1.35, self.adb_latency_sec + card_delay_sec))
        px = max(0.8, min(17.2, unit.tile_x + unit.vx * lead_t))
        py = max(1.2, min(30.5, unit.tile_y + unit.vy * lead_t))
        return round(px, 2), round(py, 2)

    def analyze(
        self,
        state: BattleState,
        pending_allies: Sequence[DetectedUnit] = (),
    ) -> BoardAssessment:
        all_allies = list(state.ally_units) + list(pending_allies)
        left_eval = self._analyze_lane(Lane.LEFT, state, all_allies)
        right_eval = self._analyze_lane(Lane.RIGHT, state, all_allies)

        weaker_lane, lowest_hp = self._find_weaker_enemy_lane(state)
        best_zap = self._find_best_zap(state, all_allies)

        return BoardAssessment(
            lanes={Lane.LEFT: left_eval, Lane.RIGHT: right_eval},
            best_zap=best_zap,
            weaker_enemy_lane=weaker_lane,
            lowest_enemy_tower_hp=lowest_hp,
        )

    def _analyze_lane(
        self,
        lane: Lane,
        state: BattleState,
        all_allies: List[DetectedUnit],
    ) -> LaneAssessment:
        enemies = [
            u for u in state.enemy_units
            if self._belongs_to_lane(u, lane) and u.name not in NON_CATCHABLE_SPELLS
        ]
        # Include center-pull defenders (7.2 <= x <= 10.8) in lane defensive accounting when enemies are present!
        allies = [
            u for u in all_allies
            if self._belongs_to_lane(u, lane) or (enemies and 7.2 <= u.tile_x <= 10.8 and u.tile_y >= 17.5)
        ]

        ally_tower_key = "ally_left" if lane == Lane.LEFT else "ally_right"
        ally_tw = state.towers.get(ally_tower_key)
        princess_alive = True if ally_tw is None else ally_tw.alive

        # Dynamic King-Tower shift if this lane's Princess Tower has fallen!
        if princess_alive:
            def_x, def_y = self.grid.TOWER_TILES[ally_tower_key]
        else:
            def_x, def_y = self.grid.TOWER_TILES["ally_king"]

        eval_out = LaneAssessment(
            lane=lane,
            def_anchor_xy=(def_x, def_y),
            ally_princess_alive=princess_alive,
            enemy_units=enemies,
            ally_units=allies,
        )

        enemy_tower_key = "enemy_left" if lane == Lane.LEFT else "enemy_right"
        enemy_tw_x, enemy_tw_y = self.grid.TOWER_TILES[enemy_tower_key]

        # 1. Evaluate friendly units & compute defensive coverage already committed to this lane
        best_cp_weight = 0.0
        ally_mk_pushing: Optional[DetectedUnit] = None
        for a in allies:
            a_info = get_card_info(a.name)
            if a.name == "miner" and math.hypot(a.tile_x - enemy_tw_x, a.tile_y - enemy_tw_y) <= 3.6:
                eval_out.ally_miner_on_enemy_tower = True

            if a.name == "mega_knight" and a.tile_y <= 18.5 and a.hp_ratio >= 0.30:
                ally_mk_pushing = a

            # Defensive value of allies on our half (or Miner sniping bridge)
            if a.tile_y >= 14.5:
                if a.name == "inferno_dragon":
                    eval_out.ally_defense_coverage += 6.2 * max(0.5, a.hp_ratio)
                elif a.name == "mega_knight":
                    eval_out.ally_defense_coverage += 7.5 * max(0.5, a.hp_ratio)
                else:
                    eval_out.ally_defense_coverage += (a_info.elixir * 1.15) * max(0.5, a.hp_ratio)

            # Surviving counterpush candidates (exclude Skeleton Barrel while combo queue handles its timing)
            if (
                a.hp_ratio >= 0.35
                and 13.0 <= a.tile_y <= 21.5
                and a.vy <= 0.35
                and a.name not in ("miner", "zap")
                and a.source != "pending"
            ):
                w = a_info.elixir * a.hp_ratio
                if w > best_cp_weight:
                    best_cp_weight = w
                    eval_out.counterpush_lead = a

        # 2. Check enemy defensive buildings (even slightly centered in opposite lane can pull!)
        for e_any in state.enemy_units:
            if e_any.name in DEFENSIVE_BUILDING_SLUGS and abs(e_any.tile_x - enemy_tw_x) <= 6.5:
                eval_out.enemy_active_building = e_any
                if e_any.name == "inferno_tower" and ally_mk_pushing is not None:
                    eval_out.enemy_inferno_vs_mk = e_any

        # 3. Evaluate enemy threats in this lane
        best_tank_score = 0.0
        raw_threat_elixir = 0.0

        for e in enemies:
            info = get_card_info(e.name)
            weighted_cost = info.elixir * max(0.45, e.hp_ratio)

            # Deep backline heavy tank investment (Golem / Lava / PEKKA / MK at y <= 10.5)
            if e.tile_y <= 10.5 and info.elixir >= 6:
                eval_out.enemy_heavy_backline = e
            if e.tile_y < 15.8 and (e.name in TANK_SLUGS or info.elixir >= 6):
                eval_out.enemy_approaching_tank = e

            # Long-range Bridge Sniper check (Princess at y=14.8 outranges our Princess Tower!)
            min_sniper_y = BRIDGE_SNIPER_MIN_Y.get(e.name)
            if min_sniper_y is not None and e.tile_y >= min_sniper_y:
                already_sniped = any(
                    math.hypot(a.tile_x - e.tile_x, a.tile_y - e.tile_y) <= 2.8 for a in allies
                )
                if not already_sniped:
                    eval_out.bridge_sniper = e
                    eval_out.urgency_score += 48.0
                    raw_threat_elixir += weighted_cost

            # Enemy splash / stun support accompanying a push
            if e.name in SPLASH_ANTI_SWARM_SLUGS and e.tile_y >= 11.5:
                eval_out.has_splash_anti_swarm = True
            if e.name in SNIPEABLE_SUPPORT_SLUGS and 10.0 <= e.tile_y <= 21.0:
                if eval_out.enemy_resetter_or_support is None or e.name == "electro_wizard":
                    eval_out.enemy_resetter_or_support = e

            # Threats crossing onto our side (y >= 15.8)
            if e.tile_y >= 15.8:
                # Ignore 1-shot zapped swarm remnants (<= 22% HP) on a living Princess Tower so tower cleans up free!
                if (
                    princess_alive
                    and (e.weight_class == WeightClass.SWARM or info.weight_class == WeightClass.SWARM)
                    and e.hp_ratio <= 0.22
                ):
                    continue

                raw_threat_elixir += weighted_cost
                proximity_mult = 1.0 + max(0.0, (e.tile_y - 16.0) / 7.5)
                if not princess_alive:
                    proximity_mult *= 1.35  # Protect King Tower aggressively when Princess is down!
                if info.targets_buildings_only:
                    proximity_mult *= 1.45

                if e.move_type == MoveType.AIR and e.name != "goblin_barrel":
                    eval_out.air_threat_our_side += weighted_cost
                elif e.name != "goblin_barrel":
                    eval_out.ground_threat_our_side += weighted_cost

                eval_out.urgency_score += weighted_cost * proximity_mult * 9.5

                # Direct Tower / Goblin Barrel / Miner emergency
                if math.hypot(e.tile_x - def_x, e.tile_y - def_y) <= 4.2 or e.name == "goblin_barrel":
                    if eval_out.tower_emergency_unit is None or e.name == "goblin_barrel" or info.elixir >= 3:
                        eval_out.tower_emergency_unit = e
                        eval_out.urgency_score += 38.0
                        if e.name == "goblin_barrel" and e.weight_class == WeightClass.SWARM:
                            # Wait ~0.50s of barrel flight so 0.50s Zap cast delay impacts right as Goblins pop!
                            age = state.timestamp - e.first_seen_ts
                            eval_out.barrel_ready_for_zap = (age >= 0.48) or (e.source != "sim")

                # Fast bridge rushers (including fast `vy >= 2.0` OpenCV-only tracks when running without YOLO!)
                if (
                    (info.targets_buildings_only and e.name != "goblin_barrel")
                    or e.name in ("elite_barbarians", "bandit", "royal_ghost", "prince", "mini_pekka", "lumberjack")
                    or (e.name.startswith("unknown_") and e.vy >= 2.0)
                ) and e.tile_y >= 16.0:
                    if eval_out.bridge_rusher is None:
                        eval_out.bridge_rusher = e

            # Heavy Tanks (for Inferno Dragon)
            if (e.name in TANK_SLUGS or e.weight_class == WeightClass.HEAVY_TANK) and e.tile_y >= 13.5:
                t_score = weighted_cost * (1.25 if e.tile_y >= 16.5 else 0.85)
                if t_score > best_tank_score:
                    best_tank_score = t_score
                    eval_out.primary_tank = e

            # Kiteable melee threat on our side without an existing ground distraction (including cross-centerline kites!)
            if (
                e.name in KITEABLE_MELEE_SLUGS
                and not info.targets_buildings_only
                and 16.6 <= e.tile_y <= 22.4
            ):
                already_engaged = any(
                    a.move_type == MoveType.GROUND and math.hypot(a.tile_x - e.tile_x, a.tile_y - e.tile_y) <= 6.8
                    for a in all_allies
                )
                if not already_engaged and eval_out.kite_target is None:
                    eval_out.kite_target = e

        # Subtract small passive tower contribution against minor non-tank chip so we don't overreact to 1hp troops
        passive_tower_help = 1.4 if (princess_alive and eval_out.primary_tank is None and eval_out.bridge_sniper is None) else 0.0
        eval_out.unanswered_threat = round(
            max(0.0, raw_threat_elixir - eval_out.ally_defense_coverage - passive_tower_help), 2
        )
        # Scale down urgency if our already-deployed/pending troops comfortably cover this lane!
        if raw_threat_elixir > 0 and eval_out.unanswered_threat <= 0.6 and eval_out.tower_emergency_unit is None:
            eval_out.urgency_score *= 0.25

        # Only flag counterpush_lead once this lane's active enemy threat is actually cleared!
        if raw_threat_elixir > 1.5 or eval_out.enemy_approaching_tank is not None:
            eval_out.counterpush_lead = None

        eval_out.best_mk_stomp = self._compute_best_mk_stomp(enemies)
        return eval_out

    def _compute_best_mk_stomp(self, enemies: List[DetectedUnit]) -> Optional[MKStompCluster]:
        """Compute optimal kinematic intercept point for Mega Knight's 1.0s deploy stomp on our half."""
        ground_targets = [
            e
            for e in enemies
            if e.move_type == MoveType.GROUND
            and e.name not in NON_CATCHABLE_SPELLS
            and e.weight_class != WeightClass.BUILDING
            and e.tile_y >= 16.8
        ]
        if not ground_targets:
            return None

        best_cluster: Optional[MKStompCluster] = None
        for anchor in ground_targets:
            # Lead by (adb_latency + 0.85s) so MK lands right in front of moving Hog/Support!
            ix, iy = self.predict_intercept_tile(anchor, card_delay_sec=0.85)
            drop_x = ix
            drop_y = min(25.0, max(17.3, iy))

            val = 0.0
            cnt = 0
            has_wincon = False
            near_pekka = False

            for other in ground_targets:
                ox, oy = self.predict_intercept_tile(other, card_delay_sec=0.85)
                dist = math.hypot(ox - drop_x, oy - drop_y)
                if dist <= 3.4 and other.name == "pekka" and other.hp_ratio >= 0.40:
                    near_pekka = True
                if dist <= self.MK_STOMP_RADIUS:
                    info = get_card_info(other.name)
                    if info.stomp_vulnerable:
                        val += info.elixir * max(0.55, other.hp_ratio)
                        cnt += 1
                    elif other.name in ("golem", "giant", "goblin_giant", "royal_giant"):
                        # Beatdown tanks: MK tanks enemy air support + Evo MK uppercuts Golem backward!
                        val += 2.6 * max(0.5, other.hp_ratio)
                        cnt += 1
                    elif other.name not in ("pekka", "electro_giant"):
                        val += 0.45 * info.elixir * max(0.5, other.hp_ratio)
                        cnt += 1
                    if info.targets_buildings_only or other.name in ("elite_barbarians", "royal_hogs"):
                        has_wincon = True

            if cnt == 0:
                continue

            candidate = MKStompCluster(
                tile_x=round(drop_x, 2),
                tile_y=round(drop_y, 2),
                elixir_value=round(val, 2),
                unit_count=cnt,
                has_win_condition=has_wincon,
                near_pekka_danger=near_pekka,
            )
            if best_cluster is None or (
                candidate.elixir_value - (5.0 if candidate.near_pekka_danger else 0.0)
                > best_cluster.elixir_value - (5.0 if best_cluster.near_pekka_danger else 0.0)
            ):
                best_cluster = candidate

        return best_cluster

    def _find_best_zap(
        self,
        state: BattleState,
        all_allies: List[DetectedUnit],
    ) -> Optional[ZapOpportunity]:
        """Evaluate high-IQ Zap plays with kinematic movement lead & Goblin Barrel flight-age check."""
        best: Optional[ZapOpportunity] = None

        # 1. Timed reset on enemy Inferno Tower / Inferno Dragon / Sparky / Mighty Miner
        for e in state.enemy_units:
            info = get_card_info(e.name)
            if not info.is_resettable:
                continue
            age = state.timestamp - e.first_seen_ts
            min_age = 2.6 if e.name == "sparky" else 1.65
            if age < min_age and e.name in ("inferno_tower", "inferno_dragon", "sparky"):
                continue

            # Is anything of ours actually at risk from it? Measured from the board rather than
            # checked against a list of card names, so a push made of anything at all counts.
            engaged_ally = any(
                math.hypot(a.tile_x - e.tile_x, a.tile_y - e.tile_y) <= 6.4
                and a.source != "pending"
                for a in all_allies
            )
            if engaged_ally:
                zx, zy = self.predict_intercept_tile(e, card_delay_sec=0.45)
                cand = ZapOpportunity(
                    tile_x=zx,
                    tile_y=zy,
                    value_score=92.0,
                    is_reset=True,
                    reason=f"Zap reset on {info.display_name} to save push",
                )
                if best is None or cand.value_score > best.value_score:
                    best = cand

        # 2. Swarms and clusters: ask the solver where the *disc* should go.
        #
        # This used to pick one anchor unit and cast on it, which threw away the empty-tile
        # play (a Skeleton Army spread around a tile is best Zapped *between* its members) and
        # punished tight clusters by only ever covering the anchor plus what happened to fall
        # inside the radius from that point. The solver scores candidate centres by what the
        # blast covers, so the empty tile wins when it really is the better cast, and a moving
        # unit is led by its own velocity rather than by a fixed constant.
        defenders = [a for a in all_allies if a.source != "pending"]
        towers = [
            (tower.name, "enemy" if tower.side is Side.ENEMY else "ally", tower.tile_x, tower.tile_y)
            for tower in state.towers.values()
            if tower.alive
        ]
        zap_profile = profile_for("zap")
        if zap_profile is not None:
            cover = plan_cast(
                "zap",
                state.enemy_units,
                defenders,
                towers,
                lead_sec=DEFAULT_LEAD_SEC + self.adb_latency_sec,
            )
            if cover is not None:
                # distance from our half: a clear on our side of the river is worth more than a
                # chip on theirs, and a defence that also saves a push is worth more again
                threat = max(0.0, min(1.0, (cover.tile_y - 14.0) / 12.0))
                engaged = any(
                    math.hypot(a.tile_x - cover.tile_x, a.tile_y - cover.tile_y) <= 3.8
                    for a in defenders
                )
                value = cover.value * (0.75 + 0.55 * threat) * (1.15 if engaged else 1.0)
                if value > 0.0 and (best is None or value > best.value_score):
                    best = ZapOpportunity(
                        tile_x=cover.tile_x,
                        tile_y=cover.tile_y,
                        value_score=value,
                        is_reset=False,
                        reason=describe(cover, zap_profile),
                    )

        return best

    @staticmethod
    def _belongs_to_lane(unit: DetectedUnit, lane: Lane) -> bool:
        if lane == Lane.LEFT:
            return unit.tile_x <= 9.0
        return unit.tile_x > 9.0

    @staticmethod
    def _find_weaker_enemy_lane(state: BattleState) -> Tuple[Lane, int]:
        el = state.towers.get("enemy_left")
        er = state.towers.get("enemy_right")
        if el and er:
            if not el.alive and er.alive:
                return Lane.RIGHT, er.estimated_hp
            if not er.alive and el.alive:
                return Lane.LEFT, el.estimated_hp
            if el.alive and er.alive:
                if el.estimated_hp <= er.estimated_hp:
                    return Lane.LEFT, el.estimated_hp
                return Lane.RIGHT, el.estimated_hp
        return Lane.LEFT, 3052
