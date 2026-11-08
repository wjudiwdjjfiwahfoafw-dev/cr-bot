"""Real-time OpenCV Tactical HUD: 18x32 Grid, Unit Tracking, Elixir Telemetry & Brain Reasoning."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.models import BattleState, BotAction, Side


class TacticalDebugHUD:
    """Renders an annotated arena view + right-hand telemetry sidebar for real-time monitoring."""

    PANEL_WIDTH = 440

    def __init__(self, grid: ArenaGrid, vision_mode_label: str = "Hybrid") -> None:
        self.grid = grid
        self.vision_mode_label = vision_mode_label
        self._action_log: Deque[str] = deque(maxlen=7)
        self._last_action: Optional[BotAction] = None
        self._last_action_ts: float = -999.0

    def record_action(self, action: BotAction, ts: float) -> None:
        self._last_action = action
        self._last_action_ts = ts
        entry = f"[{ts:5.1f}s] {action.card_name} -> ({action.tile_x:.1f},{action.tile_y:.1f}) [{action.score:.0f}]"
        self._action_log.appendleft(entry)

    def render(
        self,
        frame_bgr: np.ndarray,
        state: BattleState,
        top_candidates: List[BotAction],
        fps: float = 0.0,
    ) -> np.ndarray:
        canvas_img = frame_bgr.copy()
        self._draw_arena_grid(canvas_img, state)
        self._draw_units(canvas_img, state)
        self._draw_hand_annotations(canvas_img, state)
        self._draw_active_reticle(canvas_img, state.timestamp)

        h, w = canvas_img.shape[:2]
        panel = np.full((h, self.PANEL_WIDTH, 3), (24, 26, 34), dtype=np.uint8)
        self._draw_sidebar(panel, state, top_candidates, fps)

        combined = np.hstack([canvas_img, panel])
        return combined

    def _draw_arena_grid(self, img: np.ndarray, state: BattleState) -> None:
        ax1, ay1, ax2, ay2 = self.grid.arena_pixel_bounds
        overlay = img.copy()

        # Subtle tile grid every 2 tiles
        for tx in range(0, 19, 2):
            px, _ = self.grid.tile_to_pixel(float(tx), 0.0)
            cv2.line(overlay, (px, ay1), (px, ay2), (90, 110, 90), 1)
        for ty in range(0, 33, 2):
            _, py = self.grid.tile_to_pixel(0.0, float(ty))
            cv2.line(overlay, (ax1, py), (ax2, py), (90, 110, 90), 1)

        # Highlight River line (y = 16.0)
        _, river_py = self.grid.tile_to_pixel(0.0, 16.0)
        cv2.line(overlay, (ax1, river_py), (ax2, river_py), (235, 190, 50), 2)

        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)
        cv2.rectangle(img, (ax1, ay1), (ax2, ay2), (120, 200, 120), 1)

        # Mark Safe Miner Anti-Tornado anchors subtly
        for mx, my in self.grid.MINER_SAFE_LEFT + self.grid.MINER_SAFE_RIGHT:
            px, py = self.grid.tile_to_pixel(mx, my)
            cv2.circle(img, (px, py), 3, (80, 220, 255), -1)

        # Annotate tower HP percentages
        for tname, tw in state.towers.items():
            px, py = self.grid.tile_to_pixel(tw.tile_x, tw.tile_y)
            color = (100, 220, 100) if tw.alive else (90, 90, 90)
            label = f"{tw.estimated_hp} HP" if tw.alive else "DOWN"
            cv2.putText(
                img, label, (px - 28, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 15, 15), 3, cv2.LINE_AA
            )
            cv2.putText(
                img, label, (px - 28, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
            )

    def _draw_units(self, img: np.ndarray, state: BattleState) -> None:
        for u in state.units:
            color = (70, 75, 250) if u.side == Side.ENEMY else (250, 205, 65)
            px, py = self.grid.tile_to_pixel(u.tile_x, u.tile_y)
            x1, y1, x2, y2 = u.pixel_bbox
            if x2 > x1 and y2 > y1:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.circle(img, (px, py), 14, color, 2)

            # Velocity vector arrow
            if abs(u.vx) > 0.15 or abs(u.vy) > 0.15:
                vx_px, vy_py = self.grid.tile_to_pixel(u.tile_x + u.vx * 0.8, u.tile_y + u.vy * 0.8)
                cv2.arrowedLine(img, (px, py), (vx_px, vy_py), color, 2, tipLength=0.3)

            caption = f"{u.name} ({u.tile_x:.1f},{u.tile_y:.1f})"
            cv2.putText(
                img, caption, (px - 36, max(16, py - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (10, 10, 10), 3, cv2.LINE_AA
            )
            cv2.putText(
                img, caption, (px - 36, max(16, py - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA
            )

    def _draw_hand_annotations(self, img: np.ndarray, state: BattleState) -> None:
        for slot in state.hand:
            sx, sy = slot.tap_xy
            border_col = (235, 90, 255) if slot.is_evolved else ((80, 235, 120) if slot.is_ready else (110, 110, 110))
            cv2.rectangle(img, (sx - 46, sy - 56), (sx + 46, sy + 56), border_col, 2)
            tag = f"{'[EVO] ' if slot.is_evolved else ''}{slot.card_name[:10]}"
            cv2.putText(
                img, tag, (sx - 44, sy - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                img, tag, (sx - 44, sy - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, border_col, 1, cv2.LINE_AA
            )

    def _draw_active_reticle(self, img: np.ndarray, now_ts: float) -> None:
        if self._last_action is None or (now_ts - self._last_action_ts) > 2.4:
            return
        act = self._last_action
        sx, sy = self.grid.hand_slot_pixel(act.slot_index)
        tx, ty = self.grid.tile_to_pixel(act.tile_x, act.tile_y)

        cv2.line(img, (sx, sy), (tx, ty), (60, 255, 180), 2, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 18, (60, 255, 180), 2, cv2.LINE_AA)
        cv2.circle(img, (tx, ty), 4, (60, 255, 180), -1)
        label_y = min(img.shape[0] - 20, ty + 34)
        cv2.putText(
            img,
            f"PLAY: {act.card_name.upper()}",
            (max(12, min(img.shape[1] - 210, tx - 65)), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"PLAY: {act.card_name.upper()}",
            (max(12, min(img.shape[1] - 210, tx - 65)), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (60, 255, 180),
            1,
            cv2.LINE_AA,
        )

    def _draw_sidebar(
        self,
        panel: np.ndarray,
        state: BattleState,
        top_candidates: List[BotAction],
        fps: float,
    ) -> None:
        y = 38
        cv2.putText(
            panel, "MK ZAP BAIT TACTICAL AI", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 250), 2, cv2.LINE_AA
        )
        y += 28
        cv2.putText(
            panel,
            f"Phase: {state.phase.value.upper()}   |   {fps:.1f} FPS   |   {self.vision_mode_label}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (175, 185, 205),
            1,
            cv2.LINE_AA,
        )

        # Elixir gauges
        y += 38
        self._draw_bar(panel, 18, y, 395, 22, state.elixir / 10.0, (220, 75, 235), f"OUR ELIXIR: {state.elixir:.1f} / 10")
        y += 36
        self._draw_bar(
            panel,
            18,
            y,
            395,
            22,
            state.opponent.estimated_elixir / 10.0,
            (75, 95, 235),
            f"OPP ELIXIR (EST): {state.opponent.estimated_elixir:.1f} / 10",
        )

        # Small-Spell & Counter Cycle Status Banner
        y += 44
        baited = state.opponent.small_spell_baited
        mk_free = state.opponent.mk_counter_out_of_cycle
        banner_col = (45, 145, 75) if (baited or mk_free) else (55, 60, 82)
        cv2.rectangle(panel, (18, y), (413, y + 34), banner_col, -1)
        if baited and mk_free:
            bait_txt = "CYCLE WINDOW: SPELL + MK COUNTER BAITED!"
        elif baited:
            bait_txt = "ZAP/LOG WINDOW: BAITED! (GO AGGRESSIVE)"
        elif mk_free:
            bait_txt = "TANK COUNTER OUT OF HAND! (MK WINDOW)"
        else:
            bait_txt = f"ZAP/LOG WINDOW: HOLD/BAIT ({state.opponent.cards_since_small_spell} cards ago)"
        cv2.putText(
            panel, bait_txt, (26, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 250, 245), 1, cv2.LINE_AA
        )

        # Hand Status
        y += 58
        cv2.putText(panel, "HAND SLOTS:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 200, 255), 1, cv2.LINE_AA)
        for slot in state.hand:
            y += 24
            status = "READY" if slot.is_ready else "WAIT"
            evo = " [EVO]" if slot.is_evolved else ""
            col = (110, 240, 140) if slot.is_ready else (140, 140, 150)
            cv2.putText(
                panel,
                f"  [{slot.slot_index}] {slot.card_name:<17} ({slot.elixir_cost}e) - {status}{evo}",
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                col,
                1,
                cv2.LINE_AA,
            )

        # Top Scored Candidates
        y += 42
        cv2.putText(
            panel, "BRAIN CANDIDATE SCORES:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 200, 255), 1, cv2.LINE_AA
        )
        if not top_candidates:
            y += 26
            cv2.putText(
                panel,
                "  Holding / Banking Elixir...",
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (160, 165, 175),
                1,
                cv2.LINE_AA,
            )
        else:
            for rank, cand in enumerate(top_candidates, start=1):
                y += 28
                col = (90, 250, 175) if rank == 1 else (195, 200, 215)
                cv2.putText(
                    panel,
                    f"  #{rank} [{cand.score:4.1f}] {cand.card_name} @ ({cand.tile_x:.1f}, {cand.tile_y:.1f})",
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    col,
                    1,
                    cv2.LINE_AA,
                )
                reason_txt = cand.reason
                y += 20
                cv2.putText(
                    panel,
                    f"      {reason_txt[:56]}",
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (155, 165, 185),
                    1,
                    cv2.LINE_AA,
                )
                if len(reason_txt) > 56:
                    y += 17
                    cv2.putText(
                        panel,
                        f"      {reason_txt[56:112]}",
                        (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (155, 165, 185),
                        1,
                        cv2.LINE_AA,
                    )

        # Action History Log
        y += 46
        cv2.putText(panel, "RECENT PLAY LOG:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (130, 200, 255), 1, cv2.LINE_AA)
        for line in self._action_log:
            y += 24
            cv2.putText(
                panel, f"  {line}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (205, 210, 220), 1, cv2.LINE_AA
            )

        # Revealed Opponent Cards
        y += 44
        known = ", ".join(sorted(state.opponent.known_enemy_cards)) or "None tracked yet"
        cv2.putText(
            panel, "TRACKED OPPONENT DECK:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 200, 255), 1, cv2.LINE_AA
        )
        y += 24
        cv2.putText(
            panel, f"  {known[:52]}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 195, 205), 1, cv2.LINE_AA
        )
        if len(known) > 52:
            y += 20
            cv2.putText(
                panel, f"  {known[52:104]}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 195, 205), 1, cv2.LINE_AA
            )

    @staticmethod
    def _draw_bar(
        panel: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        ratio: float,
        fill_bgr: tuple[int, int, int],
        label: str,
    ) -> None:
        cv2.rectangle(panel, (x, y), (x + w, y + h), (48, 52, 65), -1)
        fill_w = int(round(max(0.0, min(1.0, ratio)) * w))
        if fill_w > 0:
            cv2.rectangle(panel, (x, y), (x + fill_w, y + h), fill_bgr, -1)
        cv2.rectangle(panel, (x, y), (x + w, y + h), (110, 115, 130), 1)
        cv2.putText(
            panel, label, (x + 10, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (250, 250, 252), 1, cv2.LINE_AA
        )
