"""Main real-time control loop tying ADB -> Hybrid Vision -> MK Zap Bait Brain -> Tap Execution."""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from cr_bot.adb.controller import ADBController
from cr_bot.arena.grid import ArenaGrid
from cr_bot.brain.mk_zap_bait_brain import MKZapBaitBrain
from cr_bot.config.loader import BotConfig
from cr_bot.models import BattleState, GamePhase
from cr_bot.ui.debug_hud import TacticalDebugHUD
from cr_bot.vision.arena_detector import HybridArenaDetector
from cr_bot.vision.elixir_tracker import ElixirReader, OpponentTracker
from cr_bot.vision.game_state import GameStateDetector
from cr_bot.vision.hand_detector import HandCardDetector

logger = logging.getLogger(__name__)


class ClashRoyaleBotRunner:
    """Coordinates ADB capture, vision state estimation, tactical decision-making, and live HUD."""

    def __init__(
        self,
        config: BotConfig,
        show_hud: bool = True,
        dry_run: bool = False,
    ) -> None:
        self.cfg = config
        self.show_hud = show_hud
        self.dry_run = dry_run

        self.grid = ArenaGrid(
            screen_width=config.target_width,
            screen_height=config.target_height,
        )
        self.adb = ADBController(
            adb_path=config.adb_path,
            serial=config.adb_serial,
            target_width=config.target_width,
            target_height=config.target_height,
            jitter_px=config.human_jitter_px,
            dry_run=dry_run,
        )
        self.phase_detector = GameStateDetector(self.grid, tower_level=config.tower_level)
        self.elixir_reader = ElixirReader(self.grid.cal)
        self.opp_tracker = OpponentTracker(bait_window_seconds=config.bait_window_seconds)
        self.hand_detector = HandCardDetector(
            self.grid,
            deck_cards=config.normalized_deck(),
            templates_dir=config.templates_dir,
        )
        self.arena_detector = HybridArenaDetector(
            self.grid,
            yolo_weights_path=config.yolo_weights_path,
            yolo_conf=config.yolo_confidence,
            enable_opencv_fallback=config.enable_opencv_fallback,
        )
        self.brain = MKZapBaitBrain(self.grid, config=config, adb_latency_sec=0.16)

        vision_label = (
            f"YOLO({self.arena_detector.yolo.mode})+CV"
            if self.arena_detector.yolo.is_active
            else "OpenCV-Fallback"
        )
        self.hud = TacticalDebugHUD(self.grid, vision_mode_label=vision_label)

        self._was_in_battle = False
        self._last_menu_click_ts = 0.0
        self.matches_played = 0

    def reset_match_state(self, now_ts: float) -> None:
        self.elixir_reader.reset()
        self.opp_tracker.reset(now_ts)
        self.arena_detector.reset()
        self.brain.reset()
        self.phase_detector.reset_match()

    def process_single_frame(self, frame_bgr: np.ndarray, now_ts: float) -> BattleState:
        """Run full vision + state tracking on one BGR frame and return the structured BattleState."""
        phase = self.phase_detector.detect_phase(frame_bgr, now_ts)

        if phase.is_in_battle and not self._was_in_battle:
            logger.info("=== Match started! Resetting elixir & opponent cycle trackers ===")
            self.reset_match_state(now_ts)
            self._was_in_battle = True
        elif not phase.is_in_battle and self._was_in_battle:
            logger.info("=== Match ended (phase=%s) ===", phase.value)
            self._was_in_battle = False
            self.matches_played += 1

        if not phase.is_in_battle:
            return BattleState(timestamp=now_ts, phase=phase, elixir=0.0, hand=[])

        elixir = self.elixir_reader.read_elixir(frame_bgr)
        hand = self.hand_detector.detect_hand(frame_bgr, elixir)
        towers = self.phase_detector.read_towers(frame_bgr)
        units = self.arena_detector.detect_and_track(frame_bgr, now_ts)
        opp_state = self.opp_tracker.update(now_ts, phase, units)

        return BattleState(
            timestamp=now_ts,
            phase=phase,
            elixir=elixir,
            hand=hand,
            units=units,
            towers=towers,
            opponent=opp_state,
        )

    def run_forever(self, max_matches: Optional[int] = None) -> None:
        """Connect to the emulator via ADB and run the continuous battle/menu loop."""
        self.adb.connect()
        if not self.hand_detector._templates:
            logger.warning(
                "No card templates found in '%s'! Run `python scripts/capture_templates.py` inside a match "
                "so the bot can identify your 4 hand cards accurately.",
                self.cfg.templates_dir,
            )

        target_period = 1.0 / max(1.0, self.cfg.loop_hz)
        start_wall = time.monotonic()
        fps_ema = self.cfg.loop_hz

        logger.info(
            "Bot running at %.1f Hz (Deck=%s, AvgElixir=%.2f, DryRun=%s). Press 'q' in HUD or Ctrl+C to stop.",
            self.cfg.loop_hz,
            ", ".join(self.cfg.normalized_deck()),
            self.cfg.average_elixir(),
            self.dry_run,
        )

        try:
            while True:
                if max_matches is not None and self.matches_played >= max_matches:
                    logger.info("Completed requested %d match(es). Stopping.", max_matches)
                    break

                t0 = time.monotonic()
                rel_ts = t0 - start_wall

                frame = self.adb.capture_frame()
                if frame is None:
                    logger.warning("Empty frame from ADB; retrying in 0.5s...")
                    time.sleep(0.5)
                    continue

                state = self.process_single_frame(frame, rel_ts)

                if state.phase.is_in_battle:
                    action = self.brain.decide(state)
                    if action is not None:
                        slot_xy = self.grid.hand_slot_pixel(action.slot_index)
                        arena_xy = self.grid.tile_to_pixel(action.tile_x, action.tile_y)
                        logger.info(
                            "[%.1fs | Elixir %.1f vs %.1f] PLAY %s (slot %d) -> tile(%.1f, %.1f) | score=%.1f | %s",
                            rel_ts,
                            state.elixir,
                            state.opponent.estimated_elixir,
                            action.card_name,
                            action.slot_index,
                            action.tile_x,
                            action.tile_y,
                            action.score,
                            action.reason,
                        )
                        self.adb.deploy_card(slot_xy, arena_xy)
                        self.hud.record_action(action, rel_ts)
                else:
                    self._handle_non_battle_screen(state.phase, rel_ts)

                dt_proc = max(1e-3, time.monotonic() - t0)
                fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / dt_proc)

                if self.show_hud:
                    hud_img = self.hud.render(frame, state, self.brain.last_top_candidates, fps=fps_ema)
                    # Scale cleanly to 75% height (960px tall) so it fits comfortably on 1080p monitors
                    disp = cv2.resize(hud_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
                    cv2.imshow("Clash Royale Tactical AI - MK Zap Bait", disp)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        logger.info("Quit requested from HUD window.")
                        break

                sleep_rem = target_period - (time.monotonic() - t0)
                if sleep_rem > 0:
                    time.sleep(sleep_rem)

        except KeyboardInterrupt:
            logger.info("Stopped by user (Ctrl+C).")
        finally:
            if self.show_hud:
                cv2.destroyAllWindows()

    def _handle_non_battle_screen(self, phase: GamePhase, now_ts: float) -> None:
        if not self.cfg.auto_queue_battle:
            return
        if (now_ts - self._last_menu_click_ts) < 2.8:
            return

        if phase == GamePhase.MATCH_OVER:
            ox, oy = self.phase_detector.match_over_ok_button_xy(
                self.cfg.target_width, self.cfg.target_height
            )
            logger.info("Match over screen detected -> clicking OK (%d, %d)", ox, oy)
            self.adb.tap(ox, oy)
            self._last_menu_click_ts = now_ts
        elif phase == GamePhase.MAIN_MENU:
            bx, by = self.phase_detector.menu_battle_button_xy(
                self.cfg.target_width, self.cfg.target_height
            )
            logger.info("Main Menu detected -> queueing Battle (%d, %d)", bx, by)
            self.adb.tap(bx, by)
            self._last_menu_click_ts = now_ts
