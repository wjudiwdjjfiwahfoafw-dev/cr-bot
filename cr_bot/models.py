"""Core data models shared across ADB, Vision, State Tracking, and Tactical Brain."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class GamePhase(str, Enum):
    UNKNOWN = "unknown"
    MAIN_MENU = "main_menu"
    MATCH_LOADING = "match_loading"
    BATTLE_1X = "battle_1x"
    BATTLE_2X = "battle_2x"
    BATTLE_3X = "battle_3x"
    MATCH_OVER = "match_over"

    @property
    def is_in_battle(self) -> bool:
        return self in (GamePhase.BATTLE_1X, GamePhase.BATTLE_2X, GamePhase.BATTLE_3X)

    @property
    def elixir_multiplier(self) -> float:
        if self == GamePhase.BATTLE_3X:
            return 3.0
        if self == GamePhase.BATTLE_2X:
            return 2.0
        return 1.0


class Side(str, Enum):
    ALLY = "ally"
    ENEMY = "enemy"


class Lane(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"


class MoveType(str, Enum):
    GROUND = "ground"
    AIR = "air"


class WeightClass(str, Enum):
    SWARM = "swarm"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY_TANK = "heavy_tank"
    BUILDING = "building"
    SPELL_PROJECTILE = "spell_projectile"


@dataclass
class DetectedUnit:
    """A friendly or enemy troop, building, or projectile tracked on the 18x32 arena grid."""

    track_id: int
    name: str
    side: Side
    tile_x: float
    tile_y: float
    pixel_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.8
    hp_ratio: float = 1.0
    move_type: MoveType = MoveType.GROUND
    weight_class: WeightClass = WeightClass.MEDIUM
    vx: float = 0.0
    vy: float = 0.0
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    source: str = "opencv"

    @property
    def lane(self) -> Lane:
        if self.tile_x < 8.5:
            return Lane.LEFT
        if self.tile_x > 9.5:
            return Lane.RIGHT
        return Lane.CENTER

    @property
    def on_our_side(self) -> bool:
        """River sits at y=15.5..16.5; y >= 16.5 is our territory."""
        return self.tile_y >= 16.5


@dataclass
class TowerState:
    """Health and activation status of one of the 6 arena towers."""

    name: str
    side: Side
    lane: Lane
    alive: bool = True
    hp_ratio: float = 1.0
    max_hp: int = 3052
    king_activated: bool = False
    tile_x: float = 3.5
    tile_y: float = 25.5

    @property
    def estimated_hp(self) -> int:
        if not self.alive:
            return 0
        return max(1, int(round(self.hp_ratio * self.max_hp)))


@dataclass
class HandSlot:
    """One of the 4 playable card slots at the bottom of the screen."""

    slot_index: int
    card_name: str
    elixir_cost: int = 3
    is_ready: bool = True
    is_evolved: bool = False
    confidence: float = 1.0
    tap_xy: Tuple[int, int] = (0, 0)


@dataclass
class OpponentCycleState:
    """Tracked estimate of the opponent's elixir, 4-card FIFO cycle, and bait/counter windows."""

    estimated_elixir: float = 5.0
    small_spell_baited: bool = False
    mk_counter_out_of_cycle: bool = False
    last_small_spell_ts: float = -999.0
    cards_since_small_spell: int = 4
    cards_since_card: Dict[str, int] = field(default_factory=dict)
    known_enemy_cards: Set[str] = field(default_factory=set)
    has_inferno_or_sparky: bool = False
    has_goblin_barrel: bool = False
    has_heavy_tank: bool = False
    recent_plays: List[str] = field(default_factory=list)

    def is_card_out_of_hand(self, card_slug: str) -> bool:
        """In Clash Royale's 8-card FIFO queue, a played card requires 4 other plays before returning."""
        if card_slug not in self.cards_since_card:
            return False
        return self.cards_since_card[card_slug] < 4


@dataclass
class QueuedAction:
    """A delayed second step of a multi-card tactical combo (e.g. Miner 1.8s after Skeleton Barrel)."""

    card_name: str
    tile_x: float
    tile_y: float
    execute_at_ts: float
    max_expire_ts: float
    reason: str
    required_min_elixir: float = 0.0


@dataclass
class BotAction:
    """A scored decision chosen by the Tactical Brain."""

    card_name: str
    slot_index: int
    tile_x: float
    tile_y: float
    score: float
    reason: str
    followup: Optional[QueuedAction] = None


@dataclass
class BattleState:
    """Complete snapshot of a Clash Royale frame fed into the Tactical Brain."""

    timestamp: float
    phase: GamePhase
    elixir: float
    hand: List[HandSlot]
    next_card: str = "unknown"
    units: List[DetectedUnit] = field(default_factory=list)
    towers: Dict[str, TowerState] = field(default_factory=dict)
    opponent: OpponentCycleState = field(default_factory=OpponentCycleState)

    @property
    def ally_units(self) -> List[DetectedUnit]:
        return [u for u in self.units if u.side == Side.ALLY]

    @property
    def enemy_units(self) -> List[DetectedUnit]:
        return [u for u in self.units if u.side == Side.ENEMY]

    def find_in_hand(self, card_name: str) -> Optional[HandSlot]:
        for slot in self.hand:
            if slot.card_name == card_name:
                return slot
        return None
