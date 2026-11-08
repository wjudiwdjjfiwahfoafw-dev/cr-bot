"""Own elixir HSV bar reader + Opponent elixir & small-spell cycle tracker."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Set

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaCalibration
from cr_bot.config.cards_db import get_card_info
from cr_bot.models import DetectedUnit, GamePhase, OpponentCycleState, Side


class ElixirReader:
    """Reads player elixir (0.0 to 10.0) from the 10-pip purple elixir bar at the bottom of the screen."""

    def __init__(self, calibration: ArenaCalibration | None = None) -> None:
        self.cal = calibration or ArenaCalibration()
        self._history: Deque[float] = deque(maxlen=3)

    def reset(self) -> None:
        self._history.clear()

    def read_elixir(self, frame_bgr: np.ndarray) -> float:
        h, w = frame_bgr.shape[:2]
        x_start = int(round(self.cal.elixir_x_start_ratio * w))
        x_end = int(round(self.cal.elixir_x_end_ratio * w))
        y_center = int(round(self.cal.elixir_y_ratio * h))

        y1 = max(0, y_center - 4)
        y2 = min(h, y_center + 5)
        strip = frame_bgr[y1:y2, x_start:x_end]
        if strip.size == 0:
            return 0.0

        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        # Clash Royale elixir magenta/purple glow: OpenCV Hue ~ 132..172, high Saturation & Value
        mask = cv2.inRange(hsv, np.array([128, 85, 105]), np.array([174, 255, 255]))

        total_w = max(10, x_end - x_start)
        seg_w = total_w / 10.0
        elixir_val = 0.0

        for seg_idx in range(10):
            sx1 = int(round(seg_idx * seg_w)) + 2
            sx2 = int(round((seg_idx + 1) * seg_w)) - 2
            if sx2 <= sx1:
                continue
            seg_mask = mask[:, sx1:sx2]
            fill_ratio = float(np.count_nonzero(seg_mask)) / float(seg_mask.size)
            if fill_ratio >= 0.52:
                elixir_val = float(seg_idx + 1)
            elif fill_ratio >= 0.18 and elixir_val == float(seg_idx):
                # Partial pip currently charging
                elixir_val = float(seg_idx) + min(0.9, fill_ratio / 0.55)
                break
            else:
                break

        self._history.append(round(min(10.0, max(0.0, elixir_val)), 2))
        # Temporal median filter prevents 1-frame particle flashes from spiking elixir readings
        sorted_hist = sorted(self._history)
        return sorted_hist[len(sorted_hist) // 2]


class OpponentTracker:
    """Estimates opponent elixir, tracks revealed deck cards, and flags Zap/Log bait windows."""

    BASE_REGEN_PER_SEC = 1.0 / 2.8  # +1 elixir every 2.8s in 1x

    def __init__(self, bait_window_seconds: float = 11.0) -> None:
        self.bait_window_seconds = bait_window_seconds
        self.reset()

    def reset(self, now_ts: float = 0.0) -> None:
        self._last_ts: float = now_ts
        self._counted_track_ids: Set[int] = set()
        self._last_spawn_dedup: Deque[tuple[str, float, float, float]] = deque(maxlen=16)
        self.state = OpponentCycleState(
            estimated_elixir=5.0,
            small_spell_baited=False,
            last_small_spell_ts=-999.0,
            cards_since_small_spell=4,
        )

    def update(
        self,
        now_ts: float,
        phase: GamePhase,
        units: List[DetectedUnit],
    ) -> OpponentCycleState:
        if self._last_ts <= 0.0:
            self._last_ts = now_ts

        dt = max(0.0, min(2.0, now_ts - self._last_ts))
        self._last_ts = now_ts

        if phase.is_in_battle:
            regen = dt * self.BASE_REGEN_PER_SEC * phase.elixir_multiplier
            self.state.estimated_elixir = min(10.0, self.state.estimated_elixir + regen)

        for unit in units:
            if unit.side != Side.ENEMY or unit.track_id in self._counted_track_ids:
                continue

            self._counted_track_ids.add(unit.track_id)

            # Deduplicate swarm members spawning simultaneously at nearly identical coordinates
            if self._is_swarm_sibling(unit.name, unit.tile_x, unit.tile_y, now_ts):
                continue

            self._last_spawn_dedup.append((unit.name, unit.tile_x, unit.tile_y, now_ts))
            self.register_enemy_card(unit.name, now_ts)

        # Recompute whether opponent's small spell & heavy MK counters are currently out of cycle
        time_since_spell = now_ts - self.state.last_small_spell_ts
        self.state.small_spell_baited = (
            self.state.cards_since_small_spell < 4
            and time_since_spell <= self.bait_window_seconds
        )
        mk_counters = ("pekka", "inferno_tower", "mini_pekka")
        known_mk_counters = [c for c in mk_counters if c in self.state.known_enemy_cards]
        self.state.mk_counter_out_of_cycle = bool(
            known_mk_counters and all(self.state.is_card_out_of_hand(c) for c in known_mk_counters)
        )
        return self.state

    def register_enemy_card(self, card_name: str, now_ts: float) -> None:
        info = get_card_info(card_name)
        # Subtract estimated cost (clamped at 0.0 so missed elixir ramp never goes negative)
        self.state.estimated_elixir = max(0.0, self.state.estimated_elixir - float(info.elixir))

        if not card_name.startswith("unknown_"):
            for k in list(self.state.cards_since_card.keys()):
                self.state.cards_since_card[k] += 1
            self.state.cards_since_card[info.name] = 0
            self.state.known_enemy_cards.add(info.name)
            self.state.recent_plays.append(info.name)
            if len(self.state.recent_plays) > 12:
                self.state.recent_plays.pop(0)

        if info.is_small_spell:
            self.state.last_small_spell_ts = now_ts
            self.state.cards_since_small_spell = 0
        else:
            self.state.cards_since_small_spell += 1

        if info.name in ("inferno_tower", "inferno_dragon", "sparky", "mighty_miner"):
            self.state.has_inferno_or_sparky = True
        if info.name == "goblin_barrel":
            self.state.has_goblin_barrel = True
        if info.elixir >= 7 or info.name in ("giant", "royal_giant", "goblin_giant"):
            self.state.has_heavy_tank = True

    def _is_swarm_sibling(self, name: str, tx: float, ty: float, now_ts: float) -> bool:
        for prev_name, px, py, pts in self._last_spawn_dedup:
            if (now_ts - pts) < 0.55 and prev_name == name:
                if abs(tx - px) < 2.8 and abs(ty - py) < 2.8:
                    return True
        return False
