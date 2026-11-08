"""18x32 Clash Royale Arena Grid, Pixel<->Tile Projection, Pocket Legality & Tactical Anchors."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cr_bot.models import Lane, TowerState


@dataclass(frozen=True)
class ArenaCalibration:
    """Normalized screen bounding box of the 18x32 arena on a 9:16 portrait screen (e.g. 720x1280)."""

    arena_left_ratio: float = 48.0 / 720.0     # 0.0667
    arena_right_ratio: float = 672.0 / 720.0   # 0.9333
    arena_top_ratio: float = 116.0 / 1280.0    # 0.0906
    arena_bottom_ratio: float = 988.0 / 1280.0 # 0.7719

    # 4 card slots in hand (normalized x, y centers on 720x1280)
    hand_slot_x_ratios: Tuple[float, float, float, float] = (
        210.0 / 720.0,
        338.0 / 720.0,
        466.0 / 720.0,
        594.0 / 720.0,
    )
    hand_slot_y_ratio: float = 1135.0 / 1280.0

    # 10-pip elixir bar geometry on 720x1280
    elixir_x_start_ratio: float = 198.0 / 720.0
    elixir_x_end_ratio: float = 688.0 / 720.0
    elixir_y_ratio: float = 1244.0 / 1280.0


class ArenaGrid:
    """Projects between screen pixels and Clash Royale's canonical 18x32 tile grid."""

    TILES_X: float = 18.0
    TILES_Y: float = 32.0
    RIVER_Y: float = 16.0

    # Canonical tower centers on 18x32 grid
    TOWER_TILES: Dict[str, Tuple[float, float]] = {
        "enemy_king": (9.0, 2.8),
        "enemy_left": (3.5, 6.5),
        "enemy_right": (14.5, 6.5),
        "ally_left": (3.5, 25.5),
        "ally_right": (14.5, 25.5),
        "ally_king": (9.0, 29.2),
    }

    # Tactical placement anchors
    LEFT_BRIDGE: Tuple[float, float] = (3.5, 16.9)
    RIGHT_BRIDGE: Tuple[float, float] = (14.5, 16.9)
    LEFT_POCKET: Tuple[float, float] = (6.8, 13.5)
    RIGHT_POCKET: Tuple[float, float] = (11.2, 13.5)

    # Split / passive cycle anchors
    SPLIT_BACK: Tuple[float, float] = (9.0, 29.8)
    SPLIT_MID: Tuple[float, float] = (9.0, 22.0)
    BACK_LEFT_CORNER: Tuple[float, float] = (1.5, 30.5)
    BACK_RIGHT_CORNER: Tuple[float, float] = (16.5, 30.5)

    # Center kiting tiles (pulls melee units from one lane slightly into opposite tower's range!)
    KITE_LEFT_THREAT_TO_CENTER: Tuple[float, float] = (9.6, 20.6)
    KITE_RIGHT_THREAT_TO_CENTER: Tuple[float, float] = (8.4, 20.6)

    # Safe Inferno Dragon flanking tiles (avoids splash + stays out of E-Wiz/Lightning cluster)
    IDRAG_OUTER_LEFT: Tuple[float, float] = (1.4, 20.5)
    IDRAG_OUTER_RIGHT: Tuple[float, float] = (16.6, 20.5)
    IDRAG_CENTER_LEFT: Tuple[float, float] = (7.2, 20.2)
    IDRAG_CENTER_RIGHT: Tuple[float, float] = (10.8, 20.2)

    # Anti-Tornado Miner tower placements (outer edges cannot be Tornado-pulled to King Tower)
    MINER_SAFE_LEFT: Tuple[Tuple[float, float], ...] = (
        (2.0, 7.6),   # Outer front-left
        (1.9, 6.4),   # Outer mid-left
        (2.2, 5.1),   # Outer back-left
        (2.9, 8.0),   # Front-left
    )
    MINER_SAFE_RIGHT: Tuple[Tuple[float, float], ...] = (
        (16.0, 7.6),  # Outer front-right
        (16.1, 6.4),  # Outer mid-right
        (15.8, 5.1),  # Outer back-right
        (15.1, 8.0),  # Front-right
    )

    def __init__(
        self,
        screen_width: int = 720,
        screen_height: int = 1280,
        calibration: Optional[ArenaCalibration] = None,
    ) -> None:
        self.width = screen_width
        self.height = screen_height
        self.cal = calibration or ArenaCalibration()
        self._last_miner_spot: Optional[Tuple[float, float]] = None

    @property
    def arena_pixel_bounds(self) -> Tuple[int, int, int, int]:
        x1 = int(round(self.cal.arena_left_ratio * self.width))
        x2 = int(round(self.cal.arena_right_ratio * self.width))
        y1 = int(round(self.cal.arena_top_ratio * self.height))
        y2 = int(round(self.cal.arena_bottom_ratio * self.height))
        return x1, y1, x2, y2

    def tile_to_pixel(self, tile_x: float, tile_y: float) -> Tuple[int, int]:
        """Convert (tile_x in [0..18], tile_y in [0..32]) to screen pixel coordinates."""
        x1, y1, x2, y2 = self.arena_pixel_bounds
        px = x1 + (tile_x / self.TILES_X) * (x2 - x1)
        py = y1 + (tile_y / self.TILES_Y) * (y2 - y1)
        return int(round(px)), int(round(py))

    def pixel_to_tile(self, px: float, py: float) -> Tuple[float, float]:
        """Convert screen pixel coordinates to (tile_x, tile_y) on the 18x32 arena grid."""
        x1, y1, x2, y2 = self.arena_pixel_bounds
        tx = ((px - x1) / max(1.0, float(x2 - x1))) * self.TILES_X
        ty = ((py - y1) / max(1.0, float(y2 - y1))) * self.TILES_Y
        return max(0.0, min(self.TILES_X, tx)), max(0.0, min(self.TILES_Y, ty))

    def hand_slot_pixel(self, slot_index: int) -> Tuple[int, int]:
        """Screen pixel center for tapping card slot 0, 1, 2, or 3."""
        idx = max(0, min(3, slot_index))
        px = int(round(self.cal.hand_slot_x_ratios[idx] * self.width))
        py = int(round(self.cal.hand_slot_y_ratio * self.height))
        return px, py

    def clamp_deploy_tile(
        self,
        tile_x: float,
        tile_y: float,
        deploy_anywhere: bool = False,
        towers: Optional[Dict[str, TowerState]] = None,
    ) -> Tuple[float, float]:
        """Ensure a requested tile placement is strictly inside legal Clash Royale deployment zones."""
        tx = max(0.7, min(17.3, tile_x))

        if deploy_anywhere:
            ty = max(1.2, min(31.0, tile_y))
            return round(tx, 2), round(ty, 2)

        min_legal_y = 16.9
        if towers:
            enemy_left = towers.get("enemy_left")
            enemy_right = towers.get("enemy_right")
            if enemy_left and not enemy_left.alive and tx <= 8.8:
                min_legal_y = 12.6
            elif enemy_right and not enemy_right.alive and tx >= 9.2:
                min_legal_y = 12.6

        ty = max(min_legal_y, min(31.0, tile_y))

        # Avoid dropping directly on our living tower centers (prevents unpredictable game nudge)
        for tower_key in ("ally_left", "ally_right", "ally_king"):
            if towers and tower_key in towers and not towers[tower_key].alive:
                continue
            tw_x, tw_y = self.TOWER_TILES[tower_key]
            if abs(tx - tw_x) < 1.3 and abs(ty - tw_y) < 1.3:
                ty = tw_y - 1.6  # Step cleanly in front of tower

        return round(tx, 2), round(ty, 2)

    def pick_safe_miner_spot(
        self,
        lane: Lane,
        enemy_king_activated: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Tuple[float, float]:
        """Select a non-repeating, anti-Tornado Miner placement around the target Princess Tower."""
        chooser = rng or random
        candidates: List[Tuple[float, float]] = list(
            self.MINER_SAFE_LEFT if lane == Lane.LEFT else self.MINER_SAFE_RIGHT
        )
        if enemy_king_activated:
            # Once King is already awake, add back-inner mixups too to become even less predictable
            extra = (4.8, 5.4) if lane == Lane.LEFT else (13.2, 5.4)
            candidates.append(extra)

        filtered = [c for c in candidates if c != self._last_miner_spot] or candidates
        chosen = chooser.choice(filtered)
        self._last_miner_spot = chosen
        return chosen

    @staticmethod
    def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
