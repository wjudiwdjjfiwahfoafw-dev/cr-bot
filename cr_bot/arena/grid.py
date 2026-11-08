"""
18x32 Clash Royale Arena Grid, Pixel<->Tile Projection, Pocket Legality & Tactical Anchors.

Screen geometry comes from a :class:`~cr_bot.arena.projection.Projection`, which is either
a perspective-correct homography calibrated from the four Princess Towers (preferred, and
what real devices need) or a simple arena rectangle (legacy / emulator default).

The tactical constants below are expressed in *tile* space, so they stay correct no matter
what the device resolution or camera projection is.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cr_bot.arena.projection import (
    HomographyProjection,
    Projection,
    RectProjection,
    TILES_X,
    TILES_Y,
    clamp_rect_to_frame,
    projection_from_dict,
)
from cr_bot.models import Lane, TowerState


@dataclass
class ArenaCalibration:
    """Where the arena and HUD live on screen.

    Ratio fields are kept for backwards compatibility with the original 9:16 emulator
    tuning; the ``*_px`` fields (or ``projection``) take precedence when present because
    they are measured on a real frame and survive odd aspect ratios and camera perspective.
    """

    # --- legacy ratio geometry (relative to a 720x1280 portrait frame) ---------------
    arena_left_ratio: float = 48.0 / 720.0
    arena_right_ratio: float = 672.0 / 720.0
    arena_top_ratio: float = 116.0 / 1280.0
    arena_bottom_ratio: float = 988.0 / 1280.0

    hand_slot_x_ratios: Tuple[float, float, float, float] = (
        210.0 / 720.0,
        338.0 / 720.0,
        466.0 / 720.0,
        594.0 / 720.0,
    )
    hand_slot_y_ratio: float = 1135.0 / 1280.0

    elixir_x_start_ratio: float = 198.0 / 720.0
    elixir_x_end_ratio: float = 688.0 / 720.0
    elixir_y_ratio: float = 1244.0 / 1280.0

    # --- measured geometry (preferred) ----------------------------------------------
    frame_size: Optional[Tuple[int, int]] = None
    projection: Optional[Projection] = None
    hand_slots_px: Optional[List[Tuple[float, float]]] = None
    hand_card_size_px: Optional[Tuple[float, float]] = None
    elixir_px: Optional[Tuple[float, float, float]] = None
    river_px_y: Optional[float] = None
    # Distinct from the JSON schema version: old automatic measurements used different
    # anchors and should be re-measured. Manually supplied calibrations remain untouched.
    auto_revision: int = 0
    source: str = "default"
    residual_px: float = 0.0
    notes: str = ""

    # -------------------------------------------------------------------------------
    def scaled(self, width: int, height: int) -> "ArenaCalibration":
        """Return a copy whose pixel geometry matches a ``width`` x ``height`` frame."""
        if self.frame_size is None:
            return self
        fw, fh = self.frame_size
        if (fw, fh) == (width, height) or fw <= 0 or fh <= 0:
            return self
        sx, sy = width / float(fw), height / float(fh)

        proj = self.projection
        if isinstance(proj, HomographyProjection):
            proj = proj.scale(sx, sy)

        return ArenaCalibration(
            arena_left_ratio=self.arena_left_ratio,
            arena_right_ratio=self.arena_right_ratio,
            arena_top_ratio=self.arena_top_ratio,
            arena_bottom_ratio=self.arena_bottom_ratio,
            hand_slot_x_ratios=self.hand_slot_x_ratios,
            hand_slot_y_ratio=self.hand_slot_y_ratio,
            elixir_x_start_ratio=self.elixir_x_start_ratio,
            elixir_x_end_ratio=self.elixir_x_end_ratio,
            elixir_y_ratio=self.elixir_y_ratio,
            frame_size=(width, height),
            projection=proj,
            hand_slots_px=[(x * sx, y * sy) for x, y in self.hand_slots_px] if self.hand_slots_px else None,
            hand_card_size_px=(
                (self.hand_card_size_px[0] * sx, self.hand_card_size_px[1] * sy)
                if self.hand_card_size_px
                else None
            ),
            elixir_px=(
                (self.elixir_px[0] * sx, self.elixir_px[1] * sx, self.elixir_px[2] * sy)
                if self.elixir_px
                else None
            ),
            river_px_y=self.river_px_y * sy if self.river_px_y is not None else None,
            auto_revision=self.auto_revision,
            source=self.source,
            residual_px=self.residual_px,
            notes=self.notes,
        )

    # -- serialization ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 2,
            "auto_revision": self.auto_revision,
            "source": self.source,
            "residual_px": self.residual_px,
            "notes": self.notes,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "projection": self.projection.to_dict() if self.projection else None,
            "hand_slots_px": [list(p) for p in self.hand_slots_px] if self.hand_slots_px else None,
            "hand_card_size_px": list(self.hand_card_size_px) if self.hand_card_size_px else None,
            "elixir_px": list(self.elixir_px) if self.elixir_px else None,
            "river_px_y": self.river_px_y,
            "ratios": {
                "arena": [
                    self.arena_left_ratio,
                    self.arena_top_ratio,
                    self.arena_right_ratio,
                    self.arena_bottom_ratio,
                ],
                "hand_x": list(self.hand_slot_x_ratios),
                "hand_y": self.hand_slot_y_ratio,
                "elixir": [
                    self.elixir_x_start_ratio,
                    self.elixir_x_end_ratio,
                    self.elixir_y_ratio,
                ],
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArenaCalibration":
        ratios = data.get("ratios", {}) or {}
        arena = ratios.get("arena")
        cal = cls()
        if arena:
            cal.arena_left_ratio, cal.arena_top_ratio, cal.arena_right_ratio, cal.arena_bottom_ratio = arena
        if ratios.get("hand_x"):
            cal.hand_slot_x_ratios = tuple(ratios["hand_x"])  # type: ignore[assignment]
        if "hand_y" in ratios:
            cal.hand_slot_y_ratio = float(ratios["hand_y"])
        if ratios.get("elixir"):
            e = ratios["elixir"]
            cal.elixir_x_start_ratio, cal.elixir_x_end_ratio, cal.elixir_y_ratio = e

        if data.get("frame_size"):
            cal.frame_size = tuple(data["frame_size"])  # type: ignore[assignment]
        if data.get("projection"):
            cal.projection = projection_from_dict(data["projection"])
        if data.get("hand_slots_px"):
            cal.hand_slots_px = [tuple(p) for p in data["hand_slots_px"]]  # type: ignore[misc]
        if data.get("hand_card_size_px"):
            cal.hand_card_size_px = tuple(data["hand_card_size_px"])  # type: ignore[assignment]
        if data.get("elixir_px"):
            cal.elixir_px = tuple(data["elixir_px"])  # type: ignore[assignment]
        cal.river_px_y = data.get("river_px_y")
        cal.auto_revision = int(data.get("auto_revision", 0) or 0)
        cal.source = data.get("source", "file")
        cal.residual_px = float(data.get("residual_px", 0.0) or 0.0)
        cal.notes = data.get("notes", "")
        return cal


class ArenaGrid:
    """Projects between screen pixels and Clash Royale's canonical 18x32 tile grid."""

    TILES_X: float = TILES_X
    TILES_Y: float = TILES_Y
    RIVER_Y: float = 16.0

    # Canonical tower centres on the 18x32 grid
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

    SPLIT_BACK: Tuple[float, float] = (9.0, 29.8)
    SPLIT_MID: Tuple[float, float] = (9.0, 22.0)
    BACK_LEFT_CORNER: Tuple[float, float] = (1.5, 30.5)
    BACK_RIGHT_CORNER: Tuple[float, float] = (16.5, 30.5)

    KITE_LEFT_THREAT_TO_CENTER: Tuple[float, float] = (9.6, 20.6)
    KITE_RIGHT_THREAT_TO_CENTER: Tuple[float, float] = (8.4, 20.6)

    IDRAG_OUTER_LEFT: Tuple[float, float] = (1.4, 20.5)
    IDRAG_OUTER_RIGHT: Tuple[float, float] = (16.6, 20.5)
    IDRAG_CENTER_LEFT: Tuple[float, float] = (7.2, 20.2)
    IDRAG_CENTER_RIGHT: Tuple[float, float] = (10.8, 20.2)

    MINER_SAFE_LEFT: Tuple[Tuple[float, float], ...] = (
        (2.0, 7.6),
        (1.9, 6.4),
        (2.2, 5.1),
        (2.9, 8.0),
    )
    MINER_SAFE_RIGHT: Tuple[Tuple[float, float], ...] = (
        (16.0, 7.6),
        (16.1, 6.4),
        (15.8, 5.1),
        (15.1, 8.0),
    )

    def __init__(
        self,
        screen_width: int = 720,
        screen_height: int = 1280,
        calibration: Optional[ArenaCalibration] = None,
    ) -> None:
        self.width = int(screen_width)
        self.height = int(screen_height)
        base = calibration or ArenaCalibration()
        self.cal = base.scaled(self.width, self.height)
        self._last_miner_spot: Optional[Tuple[float, float]] = None
        if self.cal.projection is None:
            rect = (
                self.cal.arena_left_ratio * self.width,
                self.cal.arena_top_ratio * self.height,
                self.cal.arena_right_ratio * self.width,
                self.cal.arena_bottom_ratio * self.height,
            )
            self.cal.projection = RectProjection(*clamp_rect_to_frame(rect, self.width, self.height))

    # -- geometry -------------------------------------------------------------------
    def resize(self, width: int, height: int) -> None:
        """Re-fit the grid when the capture resolution changes (e.g. rotation/scaling)."""
        if (int(width), int(height)) == (self.width, self.height):
            return
        self.cal = self.cal.scaled(int(width), int(height))
        self.width, self.height = int(width), int(height)

    @property
    def projection(self) -> Projection:
        assert self.cal.projection is not None
        return self.cal.projection

    @property
    def arena_pixel_bounds(self) -> Tuple[int, int, int, int]:
        """Axis-aligned bounding box of the arena in pixels."""
        proj = self.projection
        if isinstance(proj, RectProjection):
            x1, y1, x2, y2 = proj.rect
        else:
            pts = [
                proj.tile_to_pixel(0, 0),
                proj.tile_to_pixel(self.TILES_X, 0),
                proj.tile_to_pixel(self.TILES_X, self.TILES_Y),
                proj.tile_to_pixel(0, self.TILES_Y),
                proj.tile_to_pixel(self.TILES_X / 2, 0),
                proj.tile_to_pixel(self.TILES_X / 2, self.TILES_Y),
            ]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        return (
            int(round(max(0, x1))),
            int(round(max(0, y1))),
            int(round(min(self.width, x2))),
            int(round(min(self.height, y2))),
        )

    @property
    def river_pixel_y(self) -> float:
        """Screen y of the river centre line (the deploy-zone boundary on our side)."""
        if self.cal.river_px_y is not None:
            return float(self.cal.river_px_y)
        return float(self.projection.tile_to_pixel(self.TILES_X / 2, self.RIVER_Y)[1])

    def tile_to_pixel(self, tile_x: float, tile_y: float) -> Tuple[int, int]:
        """Convert (tile_x in [0..18], tile_y in [0..32]) to screen pixel coordinates."""
        px, py = self.projection.tile_to_pixel(tile_x, tile_y)
        return int(px), int(py)

    def pixel_to_tile(self, px: float, py: float) -> Tuple[float, float]:
        """Convert screen pixel coordinates to (tile_x, tile_y) on the 18x32 arena grid."""
        return self.projection.pixel_to_tile(px, py)

    # -- hand / elixir HUD geometry -------------------------------------------------
    def hand_slot_pixel(self, slot_index: int) -> Tuple[int, int]:
        """Screen pixel centre for card slot 0, 1, 2, or 3."""
        idx = max(0, min(3, int(slot_index)))
        if self.cal.hand_slots_px and len(self.cal.hand_slots_px) >= 4:
            x, y = self.cal.hand_slots_px[idx]
            return int(round(x)), int(round(y))
        px = int(round(self.cal.hand_slot_x_ratios[idx] * self.width))
        py = int(round(self.cal.hand_slot_y_ratio * self.height))
        return px, py

    def hand_card_size(self) -> Tuple[int, int]:
        """Pixel size of one card portrait, used to crop the slot for recognition."""
        if self.cal.hand_card_size_px:
            w, h = self.cal.hand_card_size_px
            return int(round(w)), int(round(h))
        return int(round(96.0 / 720.0 * self.width)), int(round(116.0 / 1280.0 * self.height))

    def hand_slot_rect(self, slot_index: int) -> Tuple[int, int, int, int]:
        """(x1, y1, x2, y2) crop box for a hand slot."""
        cx, cy = self.hand_slot_pixel(slot_index)
        half_w = self.hand_card_size()[0] // 2
        half_h = self.hand_card_size()[1] // 2
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(self.width, cx + half_w)
        y2 = min(self.height, cy + half_h)
        return x1, y1, x2, y2

    def elixir_geometry(self) -> Tuple[int, int, int]:
        """(x_start, x_end, y_centre) of the 10-pip elixir bar."""
        if self.cal.elixir_px:
            x1, x2, y = self.cal.elixir_px
            return int(round(x1)), int(round(x2)), int(round(y))
        return (
            int(round(self.cal.elixir_x_start_ratio * self.width)),
            int(round(self.cal.elixir_x_end_ratio * self.width)),
            int(round(self.cal.elixir_y_ratio * self.height)),
        )

    # -- legality -------------------------------------------------------------------
    def clamp_deploy_tile(
        self,
        tile_x: float,
        tile_y: float,
        deploy_anywhere: bool = False,
        towers: Optional[Dict[str, TowerState]] = None,
    ) -> Tuple[float, float]:
        """Ensure a requested tile placement is strictly inside legal deployment zones."""
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

        for tower_key in ("ally_left", "ally_right", "ally_king"):
            if towers and tower_key in towers and not towers[tower_key].alive:
                continue
            tw_x, tw_y = self.TOWER_TILES[tower_key]
            if abs(tx - tw_x) < 1.3 and abs(ty - tw_y) < 1.3:
                ty = tw_y - 1.6

        return round(tx, 2), round(ty, 2)

    def is_legal_deploy_pixel(self, px: float, py: float, margin_px: float = 6.0) -> bool:
        """Cheap pixel-space guard: never tap the river, the enemy half or the card tray."""
        tx, ty = self.pixel_to_tile(px, py)
        if not (0.4 <= tx <= self.TILES_X - 0.4):
            return False
        if ty < self.RIVER_Y + 0.55:
            return False
        if py < self.river_pixel_y + margin_px:
            return False
        x1, y1, x2, y2 = self.arena_pixel_bounds
        if not (y1 <= py <= y2):
            return False
        # Keep clear of the hand tray at the bottom of the screen.
        if py > self.height * 0.86:
            return False
        return True

    def clamp_deploy_pixel(self, px: float, py: float, towers: Optional[Dict[str, TowerState]] = None) -> Tuple[int, int]:
        """Project-safe deploy point: re-derive from tile space so the tap is legal."""
        tx, ty = self.pixel_to_tile(px, py)
        tx, ty = self.clamp_deploy_tile(tx, ty, towers=towers)
        return self.tile_to_pixel(tx, ty)

    def pick_safe_miner_spot(
        self,
        lane: Lane,
        enemy_king_activated: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Tuple[float, float]:
        """Select a non-repeating, anti-Tornado Miner placement around the target tower."""
        chooser = rng or random
        candidates: List[Tuple[float, float]] = list(
            self.MINER_SAFE_LEFT if lane == Lane.LEFT else self.MINER_SAFE_RIGHT
        )
        if enemy_king_activated:
            extra = (4.8, 5.4) if lane == Lane.LEFT else (13.2, 5.4)
            candidates.append(extra)

        filtered = [c for c in candidates if c != self._last_miner_spot] or candidates
        chosen = chooser.choice(filtered)
        self._last_miner_spot = chosen
        return chosen

    @staticmethod
    def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
