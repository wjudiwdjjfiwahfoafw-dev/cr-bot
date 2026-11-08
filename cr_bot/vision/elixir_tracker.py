"""Own elixir HSV bar reader + Opponent elixir & small-spell cycle tracker."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Set, Tuple

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaCalibration
from cr_bot.config.cards_db import get_card_info
from cr_bot.models import DetectedUnit, GamePhase, OpponentCycleState, Side


class ElixirReader:
    """Reads player elixir (0.0 to 10.0) from the purple elixir bar at the bottom of the screen.

    Uses a *continuous* fill measurement rather than ten discrete pips. Per-pip thresholds
    break as soon as the bar geometry is a few percent off, which is guaranteed on a real
    device whose capture is downscaled and whose HUD is not at the emulator's aspect ratio;
    measuring where the magenta fill ends and dividing by the measured bar width degrades
    gracefully instead.
    """

    def __init__(self, calibration: ArenaCalibration | None = None, grid=None) -> None:
        self.cal = calibration or ArenaCalibration()
        self.grid = grid
        self._history: Deque[float] = deque(maxlen=5)
        self.last_fill_ratio: float = 0.0
        self.last_confidence: float = 0.0

    def reset(self) -> None:
        self._history.clear()

    def _geometry(self, frame_bgr: np.ndarray) -> Tuple[int, int, int, int, int]:
        h, w = frame_bgr.shape[:2]
        if self.grid is not None:
            self.grid.resize(w, h)
            x1, x2, y = self.grid.elixir_geometry()
        else:
            x1 = int(round(self.cal.elixir_x_start_ratio * w))
            x2 = int(round(self.cal.elixir_x_end_ratio * w))
            y = int(round(self.cal.elixir_y_ratio * h))
        bar_h = max(6, int(round(h * 0.012)))
        return x1, x2, y - bar_h // 2, bar_h, y

    def read_elixir(self, frame_bgr: np.ndarray) -> float:
        if frame_bgr is None:
            return 0.0
        h, w = frame_bgr.shape[:2]
        x1, x2, y1, bar_h, _y = self._geometry(frame_bgr)
        x1 = max(0, min(w - 2, x1))
        x2 = max(x1 + 8, min(w, x2))
        y1 = max(0, min(h - 2, y1))
        y2 = min(h, y1 + bar_h)
        strip = frame_bgr[y1:y2, x1:x2]
        if strip.size == 0:
            return 0.0

        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([128, 80, 100]), np.array([176, 255, 255]))
        coverage = (mask > 0).mean(axis=0)  # per-column fill fraction

        width = float(max(1, coverage.size))
        # Bridge occlusion by the "Max: 10" label and the drop icon, then take the rightmost
        # column that is still lit. Interior gaps must not truncate the reading.
        smoothed = np.convolve(coverage, np.ones(9) / 9.0, mode="same")
        lit = np.where(smoothed >= 0.28)[0]
        if lit.size == 0:
            self._history.append(0.0)
            self.last_fill_ratio = 0.0
            return 0.0

        last = int(lit.max()) + 1
        ratio = max(0.0, min(1.0, (last + width * 0.012) / width))
        value = ratio * 10.0

        self.last_fill_ratio = ratio
        self.last_confidence = 1.0 if 0.02 < ratio < 0.995 else 0.6

        # Elixir never decreases except when we spend, so a median filter over recent frames
        # removes particle/ability flashes without lagging real gains.
        self._history.append(round(min(10.0, max(0.0, value)), 2))
        ordered = sorted(self._history)
        return ordered[len(ordered) // 2]


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
