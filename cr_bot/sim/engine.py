"""
Approximate fixed-timestep Clash Royale arena and combat simulator.

Models the 18x32 tile grid, bridges, river, 6 towers, King activation, 8-card FIFO cycle,
Evolutions, 1x/2x/3x elixir + Sudden Death Overtime + Tiebreaker, deploy/cast delays,
Mega Knight spawn stomp & jump, Inferno 3-stage beam ramp & stun resets, Skeleton Barrel
death drop, Golem death split, Lumberjack death rage, Battle Ram charge & Barbarian spawn,
Log/BarbBarrel rolling projectiles, and strict Fog-of-War BattleState export + 720x1280 pixel rendering.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.config.cards_db import get_card_info, normalize_card_name
from cr_bot.models import (
    BattleState,
    DetectedUnit,
    GamePhase,
    HandSlot,
    Lane,
    MoveType,
    OpponentCycleState,
    Side,
    TowerState,
    WeightClass,
)


@dataclass(frozen=True)
class UnitArchetype:
    slug: str
    hp: int
    dmg: int
    hit_speed: float
    range_tiles: float
    speed_tps: float  # tiles per second (0 for buildings)
    move_type: MoveType = MoveType.GROUND
    targets_air: bool = False
    buildings_only: bool = False
    splash_radius: float = 0.0
    crown_tower_dmg: Optional[int] = None
    deploy_delay: float = 1.0
    lifetime: float = 0.0  # >0 for decaying buildings
    sight_range: float = 5.5
    is_inferno: bool = False
    inferno_tiers: Tuple[int, int, int] = (43, 145, 424)
    stuns_sec: float = 0.0
    weight_class: WeightClass = WeightClass.MEDIUM


UNIT_STATS: Dict[str, UnitArchetype] = {
    # MK Zap Bait units
    "mega_knight": UnitArchetype(
        "mega_knight", 3993, 268, 1.7, 1.2, 1.85, splash_radius=1.3, weight_class=WeightClass.HEAVY_TANK
    ),
    "miner": UnitArchetype(
        "miner", 1210, 193, 1.2, 1.2, 2.3, crown_tower_dmg=58, deploy_delay=0.65, weight_class=WeightClass.MEDIUM
    ),
    "inferno_dragon": UnitArchetype(
        "inferno_dragon", 1295, 43, 0.4, 3.5, 1.85, MoveType.AIR, targets_air=True,
        is_inferno=True, inferno_tiers=(43, 145, 424), weight_class=WeightClass.MEDIUM
    ),
    "skeleton_barrel": UnitArchetype(
        "skeleton_barrel", 532, 133, 99.0, 0.8, 2.45, MoveType.AIR, buildings_only=True,
        splash_radius=2.0, weight_class=WeightClass.LIGHT
    ),
    "bat": UnitArchetype(
        "bats", 81, 81, 1.3, 1.1, 3.15, MoveType.AIR, targets_air=True, weight_class=WeightClass.SWARM
    ),
    "skeleton": UnitArchetype(
        "skeletons", 81, 81, 1.0, 1.0, 2.55, weight_class=WeightClass.SWARM
    ),
    "stab_goblin": UnitArchetype(
        "goblin_gang", 202, 120, 1.1, 1.0, 3.15, weight_class=WeightClass.SWARM
    ),
    "barrel_goblin": UnitArchetype(
        "goblin_barrel", 202, 120, 1.1, 1.0, 3.15, deploy_delay=0.9, weight_class=WeightClass.SWARM
    ),
    "spear_goblin": UnitArchetype(
        "spear_goblins", 133, 81, 1.7, 5.0, 3.15, targets_air=True, weight_class=WeightClass.SWARM
    ),

    # Hog 2.6 / Log Bait / PEKKA BS / Golem Beatdown units
    "hog_rider": UnitArchetype(
        "hog_rider", 1696, 318, 1.6, 1.1, 3.25, buildings_only=True, weight_class=WeightClass.MEDIUM
    ),
    "musketeer": UnitArchetype(
        "musketeer", 720, 218, 1.0, 6.0, 1.85, targets_air=True, weight_class=WeightClass.MEDIUM
    ),
    "cannon": UnitArchetype(
        "cannon", 824, 212, 0.9, 5.5, 0.0, lifetime=30.0, weight_class=WeightClass.BUILDING
    ),
    "ice_golem": UnitArchetype(
        "ice_golem", 1197, 84, 2.5, 1.0, 1.25, buildings_only=True, weight_class=WeightClass.MEDIUM
    ),
    "ice_spirit": UnitArchetype(
        "ice_spirit", 230, 110, 99.0, 2.5, 3.15, targets_air=True, splash_radius=1.5,
        stuns_sec=1.2, weight_class=WeightClass.LIGHT
    ),
    "princess": UnitArchetype(
        "princess", 261, 169, 3.0, 9.0, 1.85, targets_air=True, splash_radius=1.8, weight_class=WeightClass.LIGHT
    ),
    "knight": UnitArchetype(
        "knight", 1766, 202, 1.2, 1.1, 1.85, weight_class=WeightClass.MEDIUM
    ),
    "inferno_tower": UnitArchetype(
        "inferno_tower", 1749, 43, 0.4, 6.0, 0.0, targets_air=True, lifetime=30.0,
        is_inferno=True, inferno_tiers=(43, 163, 864), weight_class=WeightClass.BUILDING
    ),
    "pekka": UnitArchetype(
        "pekka", 3760, 816, 1.8, 1.2, 1.30, weight_class=WeightClass.HEAVY_TANK
    ),
    "battle_ram": UnitArchetype(
        "battle_ram", 966, 286, 99.0, 1.0, 2.35, buildings_only=True, weight_class=WeightClass.MEDIUM
    ),
    "barbarian": UnitArchetype(
        "barbarians", 670, 192, 1.3, 1.1, 1.85, weight_class=WeightClass.MEDIUM
    ),
    "bandit": UnitArchetype(
        "bandit", 907, 193, 1.0, 1.1, 2.45, weight_class=WeightClass.MEDIUM
    ),
    "royal_ghost": UnitArchetype(
        "royal_ghost", 1210, 261, 1.8, 1.1, 2.45, splash_radius=1.1, weight_class=WeightClass.MEDIUM
    ),
    "electro_wizard": UnitArchetype(
        "electro_wizard", 713, 192, 1.8, 5.0, 2.40, targets_air=True, stuns_sec=0.5, weight_class=WeightClass.MEDIUM
    ),
    "magic_archer": UnitArchetype(
        "magic_archer", 532, 133, 1.1, 7.0, 1.85, targets_air=True, splash_radius=0.7, weight_class=WeightClass.LIGHT
    ),
    "golem": UnitArchetype(
        "golem", 5120, 312, 2.5, 1.1, 1.15, buildings_only=True, deploy_delay=3.0, weight_class=WeightClass.HEAVY_TANK
    ),
    "golemite": UnitArchetype(
        "golem", 1039, 64, 2.5, 1.0, 1.15, buildings_only=True, deploy_delay=0.5, weight_class=WeightClass.MEDIUM
    ),
    "baby_dragon": UnitArchetype(
        "baby_dragon", 1152, 161, 1.5, 3.5, 2.35, MoveType.AIR, targets_air=True,
        splash_radius=1.5, weight_class=WeightClass.MEDIUM
    ),
    "mega_minion": UnitArchetype(
        "mega_minion", 837, 311, 1.5, 2.0, 1.85, MoveType.AIR, targets_air=True, weight_class=WeightClass.MEDIUM
    ),
    "mini_pekka": UnitArchetype(
        "mini_pekka", 1361, 720, 1.6, 1.1, 2.45, weight_class=WeightClass.MEDIUM
    ),
    "lumberjack": UnitArchetype(
        "lumberjack", 1282, 242, 0.8, 1.1, 3.15, weight_class=WeightClass.MEDIUM
    ),
}


@dataclass
class SimEntity:
    eid: int
    arch: UnitArchetype
    card_slug: str
    side: Side
    x: float
    y: float
    hp: float
    max_hp: float
    spawn_ts: float
    active_at_ts: float
    is_evolved: bool = False
    stun_until_ts: float = 0.0
    slow_until_ts: float = 0.0
    rage_until_ts: float = 0.0
    next_attack_ts: float = 0.0
    locked_target_id: Optional[int] = None
    lock_start_ts: float = 0.0
    last_jump_ts: float = -99.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class SimTower:
    tid: int
    key: str
    side: Side
    lane: Lane
    x: float
    y: float
    hp: float
    max_hp: float
    dmg: int
    hit_speed: float
    range_tiles: float
    active: bool = True
    alive: bool = True
    next_attack_ts: float = 0.0


@dataclass
class PendingSpell:
    card_slug: str
    side: Side
    x: float
    y: float
    impact_ts: float
    is_evolved: bool = False
    visible_unit_id: Optional[int] = None


@dataclass
class RageZone:
    side: Side
    x: float
    y: float
    expire_ts: float
    radius: float = 3.0


@dataclass
class PlayerDeckState:
    side: Side
    elixir: float
    hand: List[str]
    queue: List[str]
    play_counts: Dict[str, int] = field(default_factory=dict)
    evolutions: List[str] = field(default_factory=list)
    total_elixir_spent: float = 0.0
    total_elixir_leaked: float = 0.0
    cards_played: int = 0


@dataclass
class MatchResult:
    winner: Optional[Side]
    ally_crowns: int
    enemy_crowns: int
    duration_sec: float
    ally_tower_damage_dealt: float
    enemy_tower_damage_dealt: float
    ally_elixir_spent: float
    ally_elixir_leaked: float
    ally_cards_played: int
    reason: str
    replay_frames: List[Dict[str, Any]] = field(default_factory=list)


class ClashRoyaleSimEngine:
    """Self-contained Tournament-Standard Clash Royale match engine."""

    DT = 0.10
    REGULATION_SEC = 180.0
    MAX_MATCH_SEC = 300.0

    EVO_CYCLES_REQUIRED = {
        "mega_knight": 1,  # Every 2nd play is Evolved
        "bats": 2,         # Every 3rd play is Evolved
        "zap": 2,
        "skeleton_barrel": 2,
    }

    def __init__(
        self,
        ally_deck: List[str],
        enemy_deck: List[str],
        ally_evos: Optional[List[str]] = None,
        enemy_evos: Optional[List[str]] = None,
        seed: int = 42,
        record_replay: bool = False,
    ) -> None:
        self.rng = random.Random(seed)
        self.grid = ArenaGrid(720, 1280)
        self.record_replay = record_replay

        self.time: float = 0.0
        self._next_id: int = 10

        # None means the configured default; an explicit empty list means no evolutions.
        self.ally = self._init_deck_state(
            Side.ALLY, ally_deck, ["mega_knight", "skeleton_barrel"] if ally_evos is None else ally_evos)
        self.enemy = self._init_deck_state(Side.ENEMY, enemy_deck, [] if enemy_evos is None else enemy_evos)

        self.towers: Dict[str, SimTower] = self._init_towers()
        self.entities: List[SimEntity] = []
        self.spells: List[PendingSpell] = []
        self.rages: List[RageZone] = []
        self.replay_frames: List[Dict[str, Any]] = []
        self.action_events: List[Dict[str, Any]] = []

        self.winner: Optional[Side] = None
        self.end_reason: str = ""

    def _init_deck_state(self, side: Side, deck: List[str], evos: List[str]) -> PlayerDeckState:
        cards = [normalize_card_name(c) for c in deck]
        self.rng.shuffle(cards)
        return PlayerDeckState(
            side=side,
            elixir=5.0,
            hand=cards[:4],
            queue=cards[4:],
            evolutions=[normalize_card_name(e) for e in evos],
        )

    def _init_towers(self) -> Dict[str, SimTower]:
        specs = [
            (1, "enemy_left", Side.ENEMY, Lane.LEFT, 3.5, 6.5, 3052, 109, 0.8, 7.5, True),
            (2, "enemy_right", Side.ENEMY, Lane.RIGHT, 14.5, 6.5, 3052, 109, 0.8, 7.5, True),
            (3, "enemy_king", Side.ENEMY, Lane.CENTER, 9.0, 2.8, 4824, 109, 1.0, 7.0, False),
            (4, "ally_left", Side.ALLY, Lane.LEFT, 3.5, 25.5, 3052, 109, 0.8, 7.5, True),
            (5, "ally_right", Side.ALLY, Lane.RIGHT, 14.5, 25.5, 3052, 109, 0.8, 7.5, True),
            (6, "ally_king", Side.ALLY, Lane.CENTER, 9.0, 29.2, 4824, 109, 1.0, 7.0, False),
        ]
        return {
            key: SimTower(
                tid=tid, key=key, side=side, lane=lane, x=x, y=y,
                hp=float(hp), max_hp=float(hp), dmg=dmg, hit_speed=hs,
                range_tiles=rng, active=act
            )
            for tid, key, side, lane, x, y, hp, dmg, hs, rng, act in specs
        }

    @property
    def phase(self) -> GamePhase:
        if self.winner is not None or self.time >= self.MAX_MATCH_SEC:
            return GamePhase.MATCH_OVER
        if self.time >= 240.0:
            return GamePhase.BATTLE_3X
        if self.time >= 120.0:
            return GamePhase.BATTLE_2X
        return GamePhase.BATTLE_1X

    def crowns(self, side: Side) -> int:
        opp_prefix = "enemy" if side == Side.ALLY else "ally"
        if not self.towers[f"{opp_prefix}_king"].alive:
            return 3
        c = 0
        if not self.towers[f"{opp_prefix}_left"].alive:
            c += 1
        if not self.towers[f"{opp_prefix}_right"].alive:
            c += 1
        return c

    def is_card_evolved_in_hand(self, pstate: PlayerDeckState, card_slug: str) -> bool:
        if card_slug not in pstate.evolutions:
            return False
        req = self.EVO_CYCLES_REQUIRED.get(card_slug, 2)
        cnt = pstate.play_counts.get(card_slug, 0)
        return (cnt % (req + 1)) == req

    # -------------------------------------------------------------------------
    # Card Deployment API (Works identically for ALLY and ENEMY)
    # -------------------------------------------------------------------------

    def play_card(self, side: Side, card_name: str, tile_x: float, tile_y: float, reason: str = "") -> bool:
        slug = normalize_card_name(card_name)
        pstate = self.ally if side == Side.ALLY else self.enemy
        if slug not in pstate.hand:
            return False

        info = get_card_info(slug)
        if pstate.elixir + 1e-4 < info.elixir:
            return False

        tx, ty = self._clamp_side_deploy(side, slug, tile_x, tile_y)
        is_evo = self.is_card_evolved_in_hand(pstate, slug)

        # Spend elixir & rotate 4-card FIFO cycle
        pstate.elixir -= float(info.elixir)
        pstate.total_elixir_spent += float(info.elixir)
        pstate.cards_played += 1
        pstate.play_counts[slug] = pstate.play_counts.get(slug, 0) + 1

        slot_idx = pstate.hand.index(slug)
        next_in = pstate.queue.pop(0)
        pstate.hand[slot_idx] = next_in
        pstate.queue.append(slug)

        self._spawn_card_effect(side, slug, tx, ty, is_evo)
        self.action_events.append(
            {
                "t": round(self.time, 2),
                "side": side.value,
                "card": slug,
                "evo": is_evo,
                "x": round(tx, 2),
                "y": round(ty, 2),
                "reason": reason,
            }
        )
        return True

    def _clamp_side_deploy(self, side: Side, slug: str, x: float, y: float) -> Tuple[float, float]:
        info = get_card_info(slug)
        tx = max(0.8, min(17.2, x))
        if info.deploy_anywhere:
            return tx, max(1.2, min(30.8, y))

        if side == Side.ALLY:
            min_y = 16.9
            if not self.towers["enemy_left"].alive and tx <= 8.8:
                min_y = 12.6
            elif not self.towers["enemy_right"].alive and tx >= 9.2:
                min_y = 12.6
            return tx, max(min_y, min(30.8, y))
        else:
            max_y = 15.1
            if not self.towers["ally_left"].alive and tx <= 8.8:
                max_y = 19.4
            elif not self.towers["ally_right"].alive and tx >= 9.2:
                max_y = 19.4
            return tx, max(1.2, min(max_y, y))

    def _spawn_card_effect(self, side: Side, slug: str, x: float, y: float, is_evo: bool) -> None:
        # Spells & flying barrels
        if slug in ("zap", "fireball", "rocket", "the_log", "barbarian_barrel", "goblin_barrel"):
            delay_map = {
                "zap": 0.5,
                "the_log": 0.45,
                "barbarian_barrel": 0.45,
                "goblin_barrel": 1.1,
                "fireball": 1.0,
                "rocket": 1.8,
            }
            vis_id = self._alloc_id()
            self.spells.append(
                PendingSpell(
                    card_slug=slug,
                    side=side,
                    x=x,
                    y=y,
                    impact_ts=self.time + delay_map[slug],
                    is_evolved=is_evo,
                    visible_unit_id=vis_id,
                )
            )
            return

        if slug == "miner":
            start_y = 29.0 if side == Side.ALLY else 3.0
            travel_t = math.hypot(x - 9.0, y - start_y) / 11.5
            self._add_entity("miner", "miner", side, x, y, deploy_delay=travel_t + 0.55, is_evo=is_evo)
            return

        if slug == "bats":
            offsets = [(-0.5, -0.4), (0.5, -0.4), (0.0, 0.0), (-0.45, 0.45), (0.45, 0.45)]
            for ox, oy in offsets:
                ent = self._add_entity("bat", "bats", side, x + ox, y + oy, is_evo=is_evo)
                if is_evo:
                    ent.hp = 162.0
                    ent.max_hp = 162.0
            return

        if slug == "spear_goblins":
            for ox, oy in [(-0.55, 0.0), (0.55, 0.0), (0.0, 0.45)]:
                self._add_entity("spear_goblin", "spear_goblins", side, x + ox, y + oy)
            return

        if slug == "goblin_gang":
            dir_sign = -1.0 if side == Side.ALLY else 1.0
            for ox in (-0.55, 0.0, 0.55):
                self._add_entity("stab_goblin", "goblin_gang", side, x + ox, y + 0.4 * dir_sign)
            for ox in (-0.55, 0.0, 0.55):
                self._add_entity("spear_goblin", "goblin_gang", side, x + ox, y - 0.5 * dir_sign)
            return

        if slug == "skeletons":
            for ox, oy in [(-0.45, 0.0), (0.45, 0.0), (0.0, 0.45)]:
                self._add_entity("skeleton", "skeletons", side, x + ox, y + oy)
            return

        if slug == "skeleton_army":
            for i in range(12):
                ang = (2 * math.pi * i) / 12.0
                self._add_entity("skeleton", "skeleton_army", side, x + 0.9 * math.cos(ang), y + 0.9 * math.sin(ang))
            return

        # Single-unit cards
        self._add_entity(slug, slug, side, x, y, is_evo=is_evo)

    def _add_entity(
        self,
        arch_key: str,
        card_slug: str,
        side: Side,
        x: float,
        y: float,
        deploy_delay: Optional[float] = None,
        is_evo: bool = False,
    ) -> SimEntity:
        arch = UNIT_STATS[arch_key]
        d_delay = arch.deploy_delay if deploy_delay is None else deploy_delay
        eid = self._alloc_id()
        ent = SimEntity(
            eid=eid,
            arch=arch,
            card_slug=card_slug,
            side=side,
            x=max(0.5, min(17.5, x)),
            y=max(0.8, min(31.2, y)),
            hp=float(arch.hp),
            max_hp=float(arch.hp),
            spawn_ts=self.time,
            active_at_ts=self.time + d_delay,
            is_evolved=is_evo,
            next_attack_ts=self.time + d_delay + 0.25,
        )
        self.entities.append(ent)
        return ent

    def _alloc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # -------------------------------------------------------------------------
    # Physics & Combat Step
    # -------------------------------------------------------------------------

    def step(self) -> bool:
        """Advance simulation by DT (0.10s). Returns False when match has concluded."""
        if self.winner is not None:
            return False

        crowns_before_ally = self.crowns(Side.ALLY)
        crowns_before_enemy = self.crowns(Side.ENEMY)

        self.time = round(self.time + self.DT, 3)

        # 1. Elixir regeneration
        mult = 3.0 if self.time >= 240.0 else (2.0 if self.time >= 120.0 else 1.0)
        regen = self.DT * (1.0 / 2.8) * mult
        for pstate in (self.ally, self.enemy):
            if pstate.elixir + regen > 10.0:
                pstate.total_elixir_leaked += (pstate.elixir + regen) - 10.0
                pstate.elixir = 10.0
            else:
                pstate.elixir += regen

        # 2. Resolve spells & spawn stomps
        self._process_spells()
        self._process_spawn_effects()

        # 3. Expire rage zones & apply building lifetime decay
        self.rages = [rz for rz in self.rages if rz.expire_ts > self.time]
        for ent in self.entities:
            if ent.arch.lifetime > 0:
                ent.hp -= (ent.max_hp / ent.arch.lifetime) * self.DT

        # 4. Update active entities (movement, jumps, attacks)
        for ent in list(self.entities):
            if ent.hp <= 0 or self.time < ent.active_at_ts:
                continue
            if self.time < ent.stun_until_ts:
                ent.locked_target_id = None
                continue
            self._update_entity(ent)

        # 5. Update towers shooting
        for tw in self.towers.values():
            if tw.alive and tw.active:
                self._update_tower(tw)

        # 6. Process deaths & death spawns (Skeleton Barrel, Golem, Battle Ram, Ice Golem, Lumberjack)
        self._cleanup_dead_entities()
        self._check_tower_destructions()

        # 7. Record sparse replay snapshot every 0.4s if enabled
        if self.record_replay and int(round(self.time * 10)) % 4 == 0:
            self._record_snapshot()

        # 8. Check win / overtime / tiebreaker conditions
        self._evaluate_match_end(crowns_before_ally, crowns_before_enemy)
        return self.winner is None

    def _process_spawn_effects(self) -> None:
        """Trigger Mega Knight spawn stomp & Electro Wizard spawn zap right when deploy timer finishes."""
        for ent in self.entities:
            if abs(self.time - ent.active_at_ts) < (0.5 * self.DT + 1e-5):
                if ent.arch.slug == "mega_knight":
                    self._deal_area_damage(
                        attacker_side=ent.side,
                        cx=ent.x,
                        cy=ent.y,
                        radius=2.25,
                        dmg=444,
                        tower_dmg=0,
                        hits_air=False,
                        knockback=1.4,
                    )
                elif ent.arch.slug == "electro_wizard":
                    self._deal_area_damage(
                        attacker_side=ent.side,
                        cx=ent.x,
                        cy=ent.y,
                        radius=2.4,
                        dmg=192,
                        tower_dmg=0,
                        hits_air=True,
                        stun_sec=0.5,
                    )

    def _process_spells(self) -> None:
        remaining: List[PendingSpell] = []
        for sp in self.spells:
            if self.time + 1e-6 < sp.impact_ts:
                remaining.append(sp)
                continue

            if sp.card_slug == "zap":
                self._deal_area_damage(
                    sp.side, sp.x, sp.y, radius=2.5, dmg=192, tower_dmg=58, hits_air=True, stun_sec=0.5
                )
                if sp.is_evolved:
                    # Evo Zap second pulse 0.6s later!
                    remaining.append(
                        PendingSpell("zap", sp.side, sp.x, sp.y, self.time + 0.6, is_evolved=False)
                    )
            elif sp.card_slug == "fireball":
                self._deal_area_damage(
                    sp.side, sp.x, sp.y, radius=2.5, dmg=689, tower_dmg=207, hits_air=True, knockback=1.2
                )
            elif sp.card_slug == "rocket":
                self._deal_area_damage(
                    sp.side, sp.x, sp.y, radius=2.0, dmg=1484, tower_dmg=371, hits_air=True, knockback=1.5
                )
            elif sp.card_slug == "the_log":
                # Rolls 9.6 tiles forward along y from cast point
                dir_y = -1.0 if sp.side == Side.ALLY else 1.0
                y_end = sp.y + dir_y * 9.6
                y_min, y_max = min(sp.y, y_end), max(sp.y, y_end)
                self._deal_rect_ground_damage(sp.side, sp.x, y_min, y_max, half_w=1.95, dmg=290, tower_dmg=58, knockback=1.2)
            elif sp.card_slug == "barbarian_barrel":
                dir_y = -1.0 if sp.side == Side.ALLY else 1.0
                y_end = sp.y + dir_y * 4.5
                y_min, y_max = min(sp.y, y_end), max(sp.y, y_end)
                self._deal_rect_ground_damage(sp.side, sp.x, y_min, y_max, half_w=1.4, dmg=240, tower_dmg=0, knockback=0.0)
                self._add_entity("barbarian", "barbarian_barrel", sp.side, sp.x, y_end, deploy_delay=0.35)
            elif sp.card_slug == "goblin_barrel":
                for ang in (0.0, 2.094, 4.188):
                    gx = sp.x + 1.05 * math.cos(ang)
                    gy = sp.y + 1.05 * math.sin(ang)
                    self._add_entity("barrel_goblin", "goblin_barrel", sp.side, gx, gy, deploy_delay=0.65)
        self.spells = remaining

    def _update_entity(self, ent: SimEntity) -> None:
        target_obj, dist = self._acquire_target(ent)
        if target_obj is None:
            ent.vx, ent.vy = 0.0, 0.0
            return

        tx, ty = target_obj.x, target_obj.y
        tid = target_obj.eid if isinstance(target_obj, SimEntity) else target_obj.tid

        in_rage = any(
            rz.side == ent.side and math.hypot(ent.x - rz.x, ent.y - rz.y) <= rz.radius
            for rz in self.rages
        )
        speed_mult = 1.35 if in_rage else 1.0
        if self.time < ent.slow_until_ts:
            speed_mult *= 0.65

        # Special Jump / Dash abilities (Mega Knight & Bandit)
        if (
            ent.arch.slug in ("mega_knight", "bandit")
            and 3.4 <= dist <= 5.2
            and (self.time - ent.last_jump_ts) >= 2.2
            and not (ent.y > 16.5 and ty < 15.5 or ent.y < 15.5 and ty > 16.5)  # Don't jump across wide river away from bridge
        ):
            # Instant leap/dash closure!
            ang = math.atan2(ty - ent.y, tx - ent.x)
            ent.x = tx - 0.85 * math.cos(ang)
            ent.y = ty - 0.85 * math.sin(ang)
            ent.last_jump_ts = self.time
            if ent.arch.slug == "mega_knight":
                self._deal_area_damage(
                    ent.side, ent.x, ent.y, radius=2.2, dmg=537, tower_dmg=537, hits_air=False, knockback=1.2
                )
            else:
                self._apply_single_damage(ent.side, target_obj, 386)
            ent.next_attack_ts = self.time + ent.arch.hit_speed
            return

        # Attack if within range (accounting for tower footprint radius ~1.1 tiles)
        eff_range = ent.arch.range_tiles + (1.15 if isinstance(target_obj, SimTower) else 0.35)
        if dist <= eff_range:
            ent.vx, ent.vy = 0.0, 0.0
            if ent.locked_target_id != tid:
                ent.locked_target_id = tid
                ent.lock_start_ts = self.time

            # Suicide impact units (Skeleton Barrel, Battle Ram, Ice Spirit)
            if ent.arch.slug == "skeleton_barrel":
                ent.hp = 0.0
                return
            if ent.arch.slug == "battle_ram":
                self._apply_single_damage(ent.side, target_obj, 480)
                ent.hp = 0.0
                return
            if ent.arch.slug == "ice_spirit":
                self._deal_area_damage(
                    ent.side, tx, ty, radius=1.6, dmg=110, tower_dmg=110, hits_air=True, stun_sec=1.2
                )
                ent.hp = 0.0
                return

            if self.time >= ent.next_attack_ts:
                self._perform_entity_attack(ent, target_obj, in_rage)
            return

        # Otherwise move toward target (ground units route via bridges when crossing river)
        ent.locked_target_id = None
        if ent.arch.speed_tps <= 0.0:
            return

        goal_x, goal_y = tx, ty
        if ent.arch.move_type == MoveType.GROUND and ent.arch.slug != "hog_rider":
            crosses_river = (ent.y > 16.3 and ty < 15.7) or (ent.y < 15.7 and ty > 16.3)
            if crosses_river:
                bridge_x = 3.5 if abs(ent.x - 3.5) <= abs(ent.x - 14.5) else 14.5
                if abs(ent.x - bridge_x) > 0.9:
                    goal_x = bridge_x
                    goal_y = 16.6 if ent.y > 16.0 else 15.4

        dx, dy = goal_x - ent.x, goal_y - ent.y
        norm = max(1e-4, math.hypot(dx, dy))
        step_dist = ent.arch.speed_tps * speed_mult * self.DT
        ent.vx = (dx / norm) * ent.arch.speed_tps * speed_mult
        ent.vy = (dy / norm) * ent.arch.speed_tps * speed_mult
        ent.x += (dx / norm) * min(norm, step_dist)
        ent.y += (dy / norm) * min(norm, step_dist)

    def _perform_entity_attack(self, ent: SimEntity, target_obj: Any, in_rage: bool) -> None:
        hs_div = 1.35 if in_rage else 1.0
        ent.next_attack_ts = self.time + (ent.arch.hit_speed / hs_div)

        if ent.arch.is_inferno:
            lock_dur = self.time - ent.lock_start_ts
            t0, t1, t2 = ent.arch.inferno_tiers
            dmg = t2 if lock_dur >= 4.0 else (t1 if lock_dur >= 2.0 else t0)
            self._apply_single_damage(ent.side, target_obj, dmg)
            return

        base_dmg = ent.arch.dmg
        tw_dmg = ent.arch.crown_tower_dmg if ent.arch.crown_tower_dmg is not None else base_dmg

        if ent.arch.splash_radius > 0:
            self._deal_area_damage(
                ent.side,
                target_obj.x,
                target_obj.y,
                radius=ent.arch.splash_radius,
                dmg=base_dmg,
                tower_dmg=tw_dmg,
                hits_air=ent.arch.targets_air,
                stun_sec=ent.arch.stuns_sec,
            )
        else:
            applied = tw_dmg if isinstance(target_obj, SimTower) else base_dmg
            self._apply_single_damage(ent.side, target_obj, applied, stun_sec=ent.arch.stuns_sec)
            # Evo Mega Knight uppercut knocks enemy tanks/troops backward toward their own side!
            if ent.arch.slug == "mega_knight" and ent.is_evolved and isinstance(target_obj, SimEntity):
                kb_dir = -2.2 if ent.side == Side.ALLY else 2.2
                target_obj.y = max(1.0, min(31.0, target_obj.y + kb_dir))
                target_obj.locked_target_id = None

    def _acquire_target(self, ent: SimEntity) -> Tuple[Optional[Any], float]:
        best_obj: Optional[Any] = None
        best_dist = 999.0

        # Check enemy units within sight range first (unless building-only win condition)
        for other in self.entities:
            if other.side == ent.side or other.hp <= 0 or self.time < other.active_at_ts:
                continue
            if ent.arch.buildings_only and other.arch.weight_class != WeightClass.BUILDING:
                continue
            if other.arch.move_type == MoveType.AIR and not ent.arch.targets_air:
                continue

            d = math.hypot(other.x - ent.x, other.y - ent.y)
            max_pull = max(ent.arch.sight_range, ent.arch.range_tiles + 0.5)
            # Center-placed buildings pull building-targeting wincons from slightly wider sight (7.2 tiles)
            if ent.arch.buildings_only and other.arch.weight_class == WeightClass.BUILDING:
                max_pull = 7.4

            if d <= max_pull and d < best_dist:
                best_dist = d
                best_obj = other

        # Check enemy towers (always visible global objectives, or closer than pulled unit)
        for tw in self.towers.values():
            if tw.side == ent.side or not tw.alive:
                continue
            d = math.hypot(tw.x - ent.x, tw.y - ent.y)
            if d < best_dist:
                best_dist = d
                best_obj = tw

        return best_obj, best_dist

    def _update_tower(self, tw: SimTower) -> None:
        if self.time < tw.next_attack_ts:
            return
        best_e: Optional[SimEntity] = None
        best_d = tw.range_tiles
        for ent in self.entities:
            if ent.side == tw.side or ent.hp <= 0 or self.time < ent.active_at_ts:
                continue
            d = math.hypot(ent.x - tw.x, ent.y - tw.y)
            if d <= best_d:
                best_d = d
                best_e = ent
        if best_e is not None:
            best_e.hp -= tw.dmg
            tw.next_attack_ts = self.time + tw.hit_speed

    def _deal_area_damage(
        self,
        attacker_side: Side,
        cx: float,
        cy: float,
        radius: float,
        dmg: int,
        tower_dmg: int,
        hits_air: bool,
        stun_sec: float = 0.0,
        knockback: float = 0.0,
    ) -> None:
        for ent in self.entities:
            if ent.side == attacker_side or ent.hp <= 0 or self.time < ent.active_at_ts:
                continue
            if ent.arch.move_type == MoveType.AIR and not hits_air:
                continue
            d = math.hypot(ent.x - cx, ent.y - cy)
            if d <= radius:
                ent.hp -= dmg
                if stun_sec > 0:
                    ent.stun_until_ts = max(ent.stun_until_ts, self.time + stun_sec)
                    ent.locked_target_id = None
                if knockback > 0 and ent.arch.weight_class not in (WeightClass.HEAVY_TANK, WeightClass.BUILDING):
                    ang = math.atan2(ent.y - cy, ent.x - cx) if d > 0.05 else (-1.57 if attacker_side == Side.ALLY else 1.57)
                    ent.x = max(0.6, min(17.4, ent.x + knockback * math.cos(ang)))
                    ent.y = max(1.0, min(31.0, ent.y + knockback * math.sin(ang)))

        if tower_dmg > 0:
            for tw in self.towers.values():
                if tw.side == attacker_side or not tw.alive:
                    continue
                if math.hypot(tw.x - cx, tw.y - cy) <= radius + 1.1:
                    self._damage_tower(tw, tower_dmg)

    def _deal_rect_ground_damage(
        self,
        attacker_side: Side,
        cx: float,
        y_min: float,
        y_max: float,
        half_w: float,
        dmg: int,
        tower_dmg: int,
        knockback: float,
    ) -> None:
        kb_y = -knockback if attacker_side == Side.ALLY else knockback
        for ent in self.entities:
            if ent.side == attacker_side or ent.hp <= 0 or self.time < ent.active_at_ts:
                continue
            if ent.arch.move_type == MoveType.AIR:
                continue
            if abs(ent.x - cx) <= half_w and y_min <= ent.y <= y_max:
                ent.hp -= dmg
                if knockback > 0 and ent.arch.weight_class != WeightClass.BUILDING:
                    ent.y = max(1.0, min(31.0, ent.y + kb_y))
                    ent.locked_target_id = None  # Log knockback retargets ground units!
        if tower_dmg > 0:
            for tw in self.towers.values():
                if tw.side == attacker_side or not tw.alive:
                    continue
                if abs(tw.x - cx) <= half_w + 1.0 and (y_min - 1.0) <= tw.y <= (y_max + 1.0):
                    self._damage_tower(tw, tower_dmg)

    def _apply_single_damage(self, attacker_side: Side, obj: Any, dmg: int, stun_sec: float = 0.0) -> None:
        if isinstance(obj, SimTower):
            self._damage_tower(obj, dmg)
        elif isinstance(obj, SimEntity):
            obj.hp -= dmg
            if stun_sec > 0:
                obj.stun_until_ts = max(obj.stun_until_ts, self.time + stun_sec)
                obj.locked_target_id = None

    @staticmethod
    def _damage_tower(tw: SimTower, dmg: int) -> None:
        tw.hp = max(0.0, tw.hp - dmg)
        if "king" in tw.key and dmg > 0:
            tw.active = True

    def _cleanup_dead_entities(self) -> None:
        survivors: List[SimEntity] = []
        spawns_after: List[Tuple[str, str, Side, float, float, float]] = []

        for ent in self.entities:
            if ent.hp > 0:
                survivors.append(ent)
                continue

            # Death effects
            if ent.arch.slug == "skeleton_barrel":
                self._deal_area_damage(
                    ent.side, ent.x, ent.y, radius=2.0, dmg=133, tower_dmg=133, hits_air=True
                )
                for i in range(7):
                    ang = (2 * math.pi * i) / 7.0
                    spawns_after.append(
                        ("skeleton", "skeleton_barrel", ent.side, ent.x + 0.75 * math.cos(ang), ent.y + 0.75 * math.sin(ang), 0.35)
                    )
            elif ent.arch.slug == "battle_ram":
                spawns_after.append(("barbarian", "battle_ram", ent.side, ent.x - 0.45, ent.y, 0.35))
                spawns_after.append(("barbarian", "battle_ram", ent.side, ent.x + 0.45, ent.y, 0.35))
            elif ent.arch.slug == "golem" and ent.max_hp > 2000:
                self._deal_area_damage(
                    ent.side, ent.x, ent.y, radius=2.0, dmg=224, tower_dmg=224, hits_air=True, knockback=1.0
                )
                spawns_after.append(("golemite", "golem", ent.side, ent.x - 0.6, ent.y, 0.45))
                spawns_after.append(("golemite", "golem", ent.side, ent.x + 0.6, ent.y, 0.45))
            elif ent.arch.slug == "ice_golem":
                self._deal_area_damage(ent.side, ent.x, ent.y, radius=2.0, dmg=84, tower_dmg=84, hits_air=True)
                for other in self.entities:
                    if other.side != ent.side and math.hypot(other.x - ent.x, other.y - ent.y) <= 2.0:
                        other.slow_until_ts = max(other.slow_until_ts, self.time + 2.0)
            elif ent.arch.slug == "lumberjack":
                self.rages.append(RageZone(side=ent.side, x=ent.x, y=ent.y, expire_ts=self.time + 5.5))

        self.entities = survivors
        for arch_k, c_slug, side, sx, sy, delay in spawns_after:
            self._add_entity(arch_k, c_slug, side, sx, sy, deploy_delay=delay)

    def _check_tower_destructions(self) -> None:
        for tw in self.towers.values():
            if tw.alive and tw.hp <= 0:
                tw.alive = False
                tw.hp = 0.0
                # Losing any Princess Tower wakes up that side's King Tower immediately!
                king_key = "ally_king" if tw.side == Side.ALLY else "enemy_king"
                self.towers[king_key].active = True

    def _evaluate_match_end(self, prev_ally_c: int, prev_enemy_c: int) -> None:
        ca = self.crowns(Side.ALLY)
        ce = self.crowns(Side.ENEMY)

        if not self.towers["enemy_king"].alive:
            self.winner = Side.ALLY
            self.end_reason = "3-Crown King Tower Destruction"
            return
        if not self.towers["ally_king"].alive:
            self.winner = Side.ENEMY
            self.end_reason = "3-Crown Loss"
            return

        # End of 3:00 Regulation
        if abs(self.time - self.REGULATION_SEC) < (0.5 * self.DT + 1e-5) and ca != ce:
            self.winner = Side.ALLY if ca > ce else Side.ENEMY
            self.end_reason = f"Regulation Time ({ca}-{ce} Crowns)"
            return

        # Sudden Death Overtime (180s .. 300s): first new crown wins immediately!
        if self.time > self.REGULATION_SEC and (ca > prev_ally_c or ce > prev_enemy_c):
            if ca != ce:
                self.winner = Side.ALLY if ca > ce else Side.ENEMY
                self.end_reason = f"Overtime Sudden Death ({ca}-{ce} Crowns)"
                return

        # 5:00 Tiebreaker: lowest HP alive tower loses!
        if self.time >= self.MAX_MATCH_SEC:
            if ca != ce:
                self.winner = Side.ALLY if ca > ce else Side.ENEMY
                self.end_reason = f"Overtime End ({ca}-{ce} Crowns)"
                return
            min_ally_hp = min(tw.hp for tw in self.towers.values() if tw.side == Side.ALLY and tw.alive)
            min_enemy_hp = min(tw.hp for tw in self.towers.values() if tw.side == Side.ENEMY and tw.alive)
            if min_ally_hp >= min_enemy_hp:
                self.winner = Side.ALLY
                self.end_reason = f"Tiebreaker (Lowest Tower HP: Ours {int(min_ally_hp)} vs Enemy {int(min_enemy_hp)})"
            else:
                self.winner = Side.ENEMY
                self.end_reason = f"Tiebreaker Loss (Lowest Tower HP: Ours {int(min_ally_hp)} vs Enemy {int(min_enemy_hp)})"

    # -------------------------------------------------------------------------
    # Strict Fog-of-War Export & 720x1280 Pixel Renderer (For Vision-in-the-Loop!)
    # -------------------------------------------------------------------------

    def export_fog_of_war_state(
        self,
        opp_tracker_state: OpponentCycleState,
    ) -> BattleState:
        """
        Export BattleState strictly obeying Clash Royale Fog of War:
        - NEVER exposes enemy hand, enemy queue, or true enemy elixir!
        - Only exposes visible units on the arena + tower health + ALLY hand & ALLY elixir.
        """
        vis_units: List[DetectedUnit] = []
        for ent in self.entities:
            #Miner underground isn't targetable, but Clash Royale shows dirt trail; standard troops show deploy clock
            px, py = self.grid.tile_to_pixel(ent.x, ent.y)
            vis_units.append(
                DetectedUnit(
                    track_id=ent.eid,
                    name=ent.card_slug,
                    side=ent.side,
                    tile_x=round(ent.x, 2),
                    tile_y=round(ent.y, 2),
                    pixel_bbox=(px - 18, py - 24, px + 18, py + 10),
                    confidence=0.98,
                    hp_ratio=round(max(0.05, ent.hp / ent.max_hp), 3),
                    move_type=ent.arch.move_type,
                    weight_class=ent.arch.weight_class,
                    vx=round(ent.vx, 2),
                    vy=round(ent.vy, 2),
                    first_seen_ts=ent.spawn_ts,
                    last_seen_ts=self.time,
                    source="sim",
                )
            )

        # Flying Goblin Barrels & visible rolling Logs/BarbBarrels in flight!
        for sp in self.spells:
            if sp.card_slug in ("goblin_barrel", "the_log", "barbarian_barrel", "zap", "fireball", "rocket"):
                info = get_card_info(sp.card_slug)
                px, py = self.grid.tile_to_pixel(sp.x, sp.y)
                vis_units.append(
                    DetectedUnit(
                        track_id=sp.visible_unit_id or 9999,
                        name=sp.card_slug,
                        side=sp.side,
                        tile_x=round(sp.x, 2),
                        tile_y=round(sp.y, 2),
                        pixel_bbox=(px - 16, py - 16, px + 16, py + 16),
                        confidence=0.95,
                        hp_ratio=1.0,
                        move_type=MoveType.AIR if sp.card_slug == "goblin_barrel" else MoveType.GROUND,
                        weight_class=info.weight_class,
                        vx=0.0,
                        vy=0.0,
                        first_seen_ts=sp.impact_ts - 0.6,
                        last_seen_ts=self.time,
                        source="sim",
                    )
                )

        hand_slots: List[HandSlot] = []
        for idx, cname in enumerate(self.ally.hand):
            info = get_card_info(cname)
            hand_slots.append(
                HandSlot(
                    slot_index=idx,
                    card_name=cname,
                    elixir_cost=info.elixir,
                    is_ready=(self.ally.elixir + 1e-4 >= info.elixir),
                    is_evolved=self.is_card_evolved_in_hand(self.ally, cname),
                    confidence=1.0,
                    tap_xy=self.grid.hand_slot_pixel(idx),
                )
            )

        towers_out: Dict[str, TowerState] = {}
        for key, tw in self.towers.items():
            towers_out[key] = TowerState(
                name=key,
                side=tw.side,
                lane=tw.lane,
                alive=tw.alive,
                hp_ratio=round(tw.hp / tw.max_hp, 3) if tw.alive else 0.0,
                max_hp=int(tw.max_hp),
                king_activated=tw.active if "king" in key else False,
                tile_x=tw.x,
                tile_y=tw.y,
            )

        return BattleState(
            timestamp=self.time,
            phase=self.phase,
            elixir=round(self.ally.elixir, 2),
            hand=hand_slots,
            next_card=self.ally.queue[0],
            units=vis_units,
            towers=towers_out,
            opponent=opp_tracker_state,
        )

    def render_pixel_frame(
        self,
        card_templates: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Render an authentic 720x1280 BGR game frame (with 10-pip elixir bar, 4 card portraits,
        evo glow, 6 towers + HP bars, and floating unit red/blue health bars) so the ENTIRE
        Computer Vision stack (`ElixirReader`, `HandCardDetector`, `GameStateDetector`,
        `OpenCVHealthBarBackend`) can play matches purely from raw pixels!
        """
        frame = np.full((self.grid.height, self.grid.width, 3), (52, 112, 58), dtype=np.uint8)
        ax1, ay1, ax2, ay2 = self.grid.arena_pixel_bounds

        # River & Bridges
        _, ry = self.grid.tile_to_pixel(0, 16.0)
        cv2.rectangle(frame, (ax1, ry - 12), (ax2, ry + 12), (210, 165, 50), -1)
        for bx in (3.5, 14.5):
            bp1 = self.grid.tile_to_pixel(bx - 1.0, 15.2)
            bp2 = self.grid.tile_to_pixel(bx + 1.0, 16.8)
            cv2.rectangle(frame, bp1, bp2, (95, 125, 165), -1)

        # 2x / 3x top-right purple badge
        if self.phase in (GamePhase.BATTLE_2X, GamePhase.BATTLE_3X):
            cv2.rectangle(
                frame,
                (int(0.85 * self.grid.width), int(0.055 * self.grid.height)),
                (int(0.96 * self.grid.width), int(0.11 * self.grid.height)),
                (225, 45, 225),
                -1,
            )

        # Towers & Tower HP bars
        for tw in self.towers.values():
            px, py = self.grid.tile_to_pixel(tw.x, tw.y)
            if not tw.alive:
                cv2.rectangle(frame, (px - 22, py - 20), (px + 22, py + 20), (75, 78, 82), -1)
                continue
            roof_col = (25, 25, 235) if tw.side == Side.ENEMY else (235, 175, 25)
            cv2.rectangle(frame, (px - 22, py - 20), (px + 22, py + 20), roof_col, -1)
            if tw.hp < tw.max_hp or ("king" in tw.key and tw.active):
                bar_w = int(round(52 * (tw.hp / tw.max_hp)))
                bar_col = (20, 20, 245) if tw.side == Side.ENEMY else (245, 200, 25)
                cv2.rectangle(frame, (px - 26, py - 36), (px - 26 + max(6, bar_w), py - 30), bar_col, -1)

        # Units & Health bars (scaled by weight class so OpenCVHealthBarBackend classifies weight!)
        width_by_weight = {
            WeightClass.HEAVY_TANK: 56,
            WeightClass.MEDIUM: 34,
            WeightClass.LIGHT: 22,
            WeightClass.SWARM: 14,
            WeightClass.BUILDING: 42,
        }
        for ent in self.entities:
            if ent.hp <= 0 or self.time < ent.active_at_ts:
                continue
            px, py = self.grid.tile_to_pixel(ent.x, ent.y)
            full_w = width_by_weight.get(ent.arch.weight_class, 30)
            lit_w = max(12, int(round(full_w * max(0.35, ent.hp / ent.max_hp))))
            col = (18, 18, 245) if ent.side == Side.ENEMY else (245, 205, 25)
            bx = px - full_w // 2
            by = py - 26
            cv2.rectangle(frame, (bx, by), (bx + lit_w, by + 5), col, -1)

        # Bottom HUD Tray, 4 Hand Portraits & 10-pip Elixir Bar
        cv2.rectangle(frame, (0, ay2 + 10), (self.grid.width, self.grid.height), (35, 30, 28), -1)
        for slot_i, cname in enumerate(self.ally.hand):
            cx, cy = self.grid.hand_slot_pixel(slot_i)
            art = card_templates.get(cname)
            if art is not None:
                info = get_card_info(cname)
                if self.ally.elixir + 1e-4 < info.elixir:
                    # Desaturate unaffordable cards just like real Clash Royale!
                    gray = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY)
                    art = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                frame[cy - 58 : cy + 58, cx - 48 : cx + 48] = art
            if self.is_card_evolved_in_hand(self.ally, cname):
                cv2.rectangle(frame, (cx - 16, cy - 68), (cx + 16, cy - 48), (230, 40, 240), -1)

        x_start = int(round(self.grid.cal.elixir_x_start_ratio * self.grid.width))
        x_end = int(round(self.grid.cal.elixir_x_end_ratio * self.grid.width))
        y_c = int(round(self.grid.cal.elixir_y_ratio * self.grid.height))
        seg_w = (x_end - x_start) / 10.0
        full_pips = int(math.floor(self.ally.elixir + 1e-4))
        frac_pip = self.ally.elixir - full_pips
        for i in range(full_pips):
            sx1 = int(round(x_start + i * seg_w))
            sx2 = int(round(x_start + (i + 1) * seg_w)) - 1
            cv2.rectangle(frame, (sx1, y_c - 6), (sx2, y_c + 6), (225, 45, 225), -1)
        if full_pips < 10 and frac_pip >= 0.25:
            sx1 = int(round(x_start + full_pips * seg_w))
            sx2 = sx1 + int(round(frac_pip * seg_w))
            cv2.rectangle(frame, (sx1, y_c - 6), (sx2, y_c + 6), (210, 55, 210), -1)

        return frame

    def build_result(self) -> MatchResult:
        ally_tw_dmg = sum(tw.max_hp - tw.hp for tw in self.towers.values() if tw.side == Side.ENEMY)
        enemy_tw_dmg = sum(tw.max_hp - tw.hp for tw in self.towers.values() if tw.side == Side.ALLY)
        return MatchResult(
            winner=self.winner,
            ally_crowns=self.crowns(Side.ALLY),
            enemy_crowns=self.crowns(Side.ENEMY),
            duration_sec=self.time,
            ally_tower_damage_dealt=round(ally_tw_dmg, 1),
            enemy_tower_damage_dealt=round(enemy_tw_dmg, 1),
            ally_elixir_spent=round(self.ally.total_elixir_spent, 1),
            ally_elixir_leaked=round(self.ally.total_elixir_leaked, 2),
            ally_cards_played=self.ally.cards_played,
            reason=self.end_reason,
            replay_frames=self.replay_frames,
        )

    def _record_snapshot(self) -> None:
        self.replay_frames.append(
            {
                "t": round(self.time, 1),
                "phase": self.phase.value,
                "ally_e": round(self.ally.elixir, 1),
                "enemy_e": round(self.enemy.elixir, 1),
                "ally_hand": list(self.ally.hand),
                "towers": {
                    k: {"hp": int(round(tw.hp)), "max": int(tw.max_hp), "alive": tw.alive}
                    for k, tw in self.towers.items()
                },
                "units": [
                    {
                        "id": e.eid,
                        "c": e.card_slug,
                        "s": 0 if e.side == Side.ALLY else 1,
                        "x": round(e.x, 2),
                        "y": round(e.y, 2),
                        "hp": round(e.hp / e.max_hp, 2),
                        "evo": e.is_evolved,
                    }
                    for e in self.entities
                    if e.hp > 0
                ],
                "spells": [
                    {"c": sp.card_slug, "s": 0 if sp.side == Side.ALLY else 1, "x": round(sp.x, 1), "y": round(sp.y, 1)}
                    for sp in self.spells
                ],
            }
        )
