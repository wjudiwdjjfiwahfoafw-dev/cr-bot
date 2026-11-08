"""Screen phase classifier (Menu / 1x / 2x / 3x / Match Over) and 6-Tower HP reader."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.models import GamePhase, Lane, Side, TowerState

# Princess & King Tower HP by Tournament/Ladder Level (Levels 9..15)
PRINCESS_HP_BY_LEVEL = {
    9: 2534,
    10: 2786,
    11: 3052,
    12: 3346,
    13: 3668,
    14: 4032,
    15: 4424,
}
KING_HP_BY_LEVEL = {
    9: 4008,
    10: 4392,
    11: 4824,
    12: 5304,
    13: 5832,
    14: 6408,
    15: 7032,
}


class GameStateDetector:
    """Detects current screen state, elixir speed phase, and the health of all 6 towers."""

    def __init__(self, grid: ArenaGrid, tower_level: int = 11) -> None:
        self.grid = grid
        self.tower_level = tower_level
        self._battle_start_ts: Optional[float] = None

    def reset_match(self) -> None:
        self._battle_start_ts = None

    def detect_phase(self, frame_bgr: np.ndarray, now_ts: float) -> GamePhase:
        h, w = frame_bgr.shape[:2]

        # 1. Check Match Over "OK" blue button near bottom center (x ~ 0.42..0.58, y ~ 0.85..0.92)
        ok_roi = frame_bgr[int(0.855 * h) : int(0.915 * h), int(0.41 * w) : int(0.59 * w)]
        if self._has_dominant_hsv(ok_roi, (95, 130, 140), (118, 255, 255), min_ratio=0.42):
            # Confirm elixir bar is absent so a blue card in hand isn't mistaken for OK button
            if not self._has_battle_hud(frame_bgr):
                self._battle_start_ts = None
                return GamePhase.MATCH_OVER

        # 2. Check active Battle HUD (bottom dark elixir tray + green/brown arena grass)
        if self._has_battle_hud(frame_bgr):
            if self._battle_start_ts is None:
                self._battle_start_ts = now_ts
            elapsed = now_ts - self._battle_start_ts

            # Visual check for purple 2x / 3x badge in top-right corner (x ~ 0.84..0.98, y ~ 0.05..0.12)
            mult_roi = frame_bgr[int(0.045 * h) : int(0.125 * h), int(0.83 * w) : int(0.98 * w)]
            has_purple_badge = self._has_dominant_hsv(
                mult_roi, (132, 95, 120), (172, 255, 255), min_ratio=0.12
            )

            if elapsed >= 240.0:
                return GamePhase.BATTLE_3X
            if elapsed >= 120.0 or has_purple_badge:
                return GamePhase.BATTLE_2X
            return GamePhase.BATTLE_1X

        # 3. Check Main Menu yellow/amber "Battle" button (center x ~ 0.34..0.66, y ~ 0.65..0.82)
        menu_roi = frame_bgr[int(0.65 * h) : int(0.82 * h), int(0.34 * w) : int(0.66 * w)]
        if self._has_dominant_hsv(menu_roi, (14, 130, 155), (32, 255, 255), min_ratio=0.22):
            self._battle_start_ts = None
            return GamePhase.MAIN_MENU

        return GamePhase.UNKNOWN

    def read_towers(self, frame_bgr: np.ndarray) -> Dict[str, TowerState]:
        """Inspect all 6 tower positions for ruins vs intact structures and read HP bar ratios."""
        princess_max = PRINCESS_HP_BY_LEVEL.get(self.tower_level, 3052)
        king_max = KING_HP_BY_LEVEL.get(self.tower_level, 4824)

        specs = [
            ("enemy_left", Side.ENEMY, Lane.LEFT, princess_max),
            ("enemy_right", Side.ENEMY, Lane.RIGHT, princess_max),
            ("enemy_king", Side.ENEMY, Lane.CENTER, king_max),
            ("ally_left", Side.ALLY, Lane.LEFT, princess_max),
            ("ally_right", Side.ALLY, Lane.RIGHT, princess_max),
            ("ally_king", Side.ALLY, Lane.CENTER, king_max),
        ]

        towers: Dict[str, TowerState] = {}
        for name, side, lane, max_hp in specs:
            tx, ty = self.grid.TOWER_TILES[name]
            px, py = self.grid.tile_to_pixel(tx, ty)
            alive, hp_ratio, king_awake = self._inspect_single_tower(
                frame_bgr, px, py, side, is_king=("king" in name)
            )
            towers[name] = TowerState(
                name=name,
                side=side,
                lane=lane,
                alive=alive,
                hp_ratio=hp_ratio,
                max_hp=max_hp,
                king_activated=king_awake,
                tile_x=tx,
                tile_y=ty,
            )
        return towers

    def _inspect_single_tower(
        self,
        frame_bgr: np.ndarray,
        px: int,
        py: int,
        side: Side,
        is_king: bool,
    ) -> Tuple[bool, float, bool]:
        h, w = frame_bgr.shape[:2]
        # Tower body patch (destroyed towers turn into flat low-saturation brown/grey rubble)
        bx1, bx2 = max(0, px - 26), min(w, px + 26)
        by1, by2 = max(0, py - 24), min(h, py + 24)
        body = frame_bgr[by1:by2, bx1:bx2]
        if body.size == 0:
            return True, 1.0, False

        body_hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        # Intact towers have strong team roof colors (red/magenta for enemy, blue/cyan for ally) or stone highlights
        if side == Side.ENEMY:
            team_mask = cv2.inRange(body_hsv, np.array([0, 85, 85]), np.array([14, 255, 255])) | cv2.inRange(
                body_hsv, np.array([165, 85, 85]), np.array([179, 255, 255])
            )
        else:
            team_mask = cv2.inRange(body_hsv, np.array([92, 85, 85]), np.array([124, 255, 255]))

        team_ratio = float(np.count_nonzero(team_mask)) / float(team_mask.size)
        alive = team_ratio >= 0.035

        if not alive:
            return False, 0.0, False

        # Inspect HP bar area just above tower (or slightly below for ally king)
        bar_y1 = max(0, py - 48)
        bar_y2 = max(1, py - 18)
        bar_x1, bar_x2 = max(0, px - 36), min(w, px + 36)
        bar_roi = frame_bgr[bar_y1:bar_y2, bar_x1:bar_x2]
        hp_ratio, bar_visible = self._measure_hp_bar(bar_roi, side)

        king_activated = bar_visible if is_king else False
        return True, hp_ratio, king_activated

    @staticmethod
    def _measure_hp_bar(bar_roi: np.ndarray, side: Side) -> Tuple[float, bool]:
        if bar_roi.size == 0:
            return 1.0, False
        hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
        if side == Side.ENEMY:
            mask = cv2.inRange(hsv, np.array([0, 110, 110]), np.array([10, 255, 255])) | cv2.inRange(
                hsv, np.array([168, 110, 110]), np.array([179, 255, 255])
            )
        else:
            mask = cv2.inRange(hsv, np.array([92, 100, 110]), np.array([118, 255, 255]))

        col_hits = np.max(mask > 0, axis=0)
        lit_cols = int(np.count_nonzero(col_hits))
        if lit_cols < 5:
            # Undamaged towers hide their health bar until hit!
            return 1.0, False

        full_bar_px = max(36.0, bar_roi.shape[1] * 0.72)
        ratio = max(0.05, min(1.0, lit_cols / full_bar_px))
        return round(ratio, 3), True

    def _has_battle_hud(self, frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        # Check arena grass/floor presence + bottom card deck wooden/blue tray
        arena_patch = frame_bgr[int(0.45 * h) : int(0.55 * h), int(0.20 * w) : int(0.80 * w)]
        tray_patch = frame_bgr[int(0.955 * h) : int(0.99 * h), int(0.27 * w) : int(0.95 * w)]
        if arena_patch.size == 0 or tray_patch.size == 0:
            return False

        # Elixir bar either has purple pips OR dark purple/black pip background track
        tray_hsv = cv2.cvtColor(tray_patch, cv2.COLOR_BGR2HSV)
        purple_mask = cv2.inRange(tray_hsv, np.array([125, 65, 70]), np.array([175, 255, 255]))
        if (np.count_nonzero(purple_mask) / float(purple_mask.size)) >= 0.06:
            return True

        # Even at 0 elixir, check river cyan water band around y = 16/32 inside arena
        rx1, ry1, rx2, ry2 = self.grid.arena_pixel_bounds
        river_y = int(round(ry1 + 0.50 * (ry2 - ry1)))
        river_strip = frame_bgr[max(0, river_y - 10) : min(h, river_y + 10), rx1:rx2]
        return self._has_dominant_hsv(river_strip, (85, 55, 80), (118, 255, 255), min_ratio=0.15)

    @staticmethod
    def _has_dominant_hsv(
        roi_bgr: np.ndarray,
        lower: Tuple[int, int, int],
        upper: Tuple[int, int, int],
        min_ratio: float,
    ) -> bool:
        if roi_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return (float(np.count_nonzero(mask)) / float(mask.size)) >= min_ratio

    @staticmethod
    def menu_battle_button_xy(width: int = 720, height: int = 1280) -> Tuple[int, int]:
        return int(0.50 * width), int(0.73 * height)

    @staticmethod
    def match_over_ok_button_xy(width: int = 720, height: int = 1280) -> Tuple[int, int]:
        return int(0.50 * width), int(0.885 * height)
