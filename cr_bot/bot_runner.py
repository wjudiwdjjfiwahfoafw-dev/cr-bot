"""
Real-time control loop: MultiClient capture -> vision -> MK Zap Bait brain -> verified deploy.

Differences from the original loop, all of them aimed at actually playing on a real phone:

* **Frames are never stale.** Capture streams via scrcpy and the loop reads the newest frame
  instead of waiting on a ``screencap`` round trip.
* **Calibration is measured, not assumed.** On the first battle frame the layout is detected
  (river, hand cards, elixir bar, towers), saved per device, and reused. This is what keeps
  deployments landing where the brain intends.
* **Every deploy is verified and retried.** After a gesture the hand is re-inspected; if the
  slot did not change the play is retried with the next input backend, and the brain is told
  so it does not desynchronise. Deployments run on a worker thread so vision keeps running.
* **Deploy points are legality-checked in tile space** before any touch is injected.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from cr_bot.adb.controller import ADBController
from cr_bot.arena.device_calibrator import (
    CalibrationResult,
    DeviceCalibrator,
    calibration_path,
    load_calibration,
    render_calibration_preview,
    save_calibration,
)
from cr_bot.arena.grid import ArenaCalibration, ArenaGrid
from cr_bot.brain.mk_zap_bait_brain import MKZapBaitBrain
from cr_bot.brain.spell_targeting import aim_is_wasted, is_damage_spell, plan_cast
from cr_bot.config.loader import BotConfig
from cr_bot.models import BattleState, BotAction, GamePhase
from cr_bot.perf import BRAIN, CAPTURE, HUD, SLEEP, VISION, FrameProfiler
from cr_bot.ui.click_to_tap import ClickToTapController, ManualGesture, map_click_to_frame
from cr_bot.ui.debug_hud import TacticalDebugHUD
from cr_bot.ui_window import PreviewWindow
from cr_bot.vision.arena_detector import HybridArenaDetector
from cr_bot.vision.deploy_verifier import DeployVerifier, SlotSnapshot
from cr_bot.vision.elixir_tracker import ElixirReader, OpponentTracker
from cr_bot.vision.game_state import GameStateDetector
from cr_bot.vision.gamescreen import LobbyNavigator, PhaseStabilizer, absolute
from cr_bot.vision.hand_detector import HandCardDetector
from cr_bot.vision.unit_identity import UnitIdentityMemory

logger = logging.getLogger(__name__)

#: title of the live gameplay viewer window
HUD_WINDOW_TITLE = "Clash Royale Bot - MK Zap Bait"


@dataclass
class DeployOutcome:
    """Result of one verified deployment attempt."""

    action: BotAction
    ok: bool
    backend: str
    attempts: int
    elapsed_ms: float
    note: str = ""


class ClashRoyaleBotRunner:
    """Coordinates capture, vision, decision-making and verified touch injection."""

    def __init__(
        self,
        config: BotConfig,
        show_hud: bool = True,
        dry_run: bool = False,
        auto_calibrate: bool = True,
        force_recalibrate: bool = False,
        deploy_retries: int = 2,
        verify_deploys: bool = True,
        manual_input: bool = True,
    ) -> None:
        self.cfg = config
        self.show_hud = show_hud
        #: created lazily on the first HUD frame; sizes itself to the desktop
        self._preview = None
        self.dry_run = dry_run
        self.auto_calibrate = auto_calibrate
        self.force_recalibrate = force_recalibrate
        self.deploy_retries = max(0, int(deploy_retries))
        self.verify_deploys = bool(verify_deploys) and not dry_run

        self.grid = ArenaGrid(config.target_width, config.target_height)
        self.calibration: ArenaCalibration = self.grid.cal
        self.calibration_result: Optional[CalibrationResult] = None

        self.adb = ADBController(
            adb_path=config.adb_path,
            serial=config.adb_serial,
            target_width=config.capture_width,
            target_height=config.target_height,
            jitter_px=config.human_jitter_px,
            dry_run=dry_run,
            capture_backend=config.capture_backend,
            input_backend=config.input_backend,
            deploy_style=config.deploy_style,
            capture_bitrate=config.capture_bitrate,
            capture_max_fps=config.capture_max_fps,
            keep_awake=config.keep_screen_on,
            lock_portrait=config.lock_portrait,
            input_order=config.input_backend_order(),
        )

        self._build_vision_stack(self.grid)
        self.verifier = DeployVerifier(change_threshold=config.deploy_verify_threshold)

        self._was_in_battle = False
        self._last_menu_click_ts = 0.0
        self.matches_played = 0
        self._calibrated_this_match = False

        # Async deploy plumbing
        self._deploy_lock = threading.Lock()
        self._pending_deploy: Optional[Tuple[BotAction, Tuple[int, int], Tuple[int, int]]] = None
        self._deploy_worker: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._busy = False
        self.last_outcomes: List[DeployOutcome] = []
        self.deploys_ok = 0
        self.deploys_failed = 0

        self._frames_seen = 0
        self._loop_times: List[float] = []
        #: per-stage frame timing; the first thing to read when the loop feels slow
        self.profiler = FrameProfiler(enabled=True)
        self._last_perf_log_ts = -999.0

        # Interactive viewer: a click in the preview window is a real touch on the phone.
        self.manual_input = bool(manual_input) and not dry_run
        self.clicker: Optional[ClickToTapController] = None
        self._click_binding_frame: Optional[tuple] = None
        self._hud_shown_size = (0, 0)

    # -- vision stack ---------------------------------------------------------------
    def _build_vision_stack(self, grid: ArenaGrid) -> None:
        cfg = self.cfg
        self.phase_detector = GameStateDetector(grid, tower_level=cfg.tower_level)
        #: one odd frame must not end a match (see cr_bot/vision/gamescreen.py)
        self.phase_stabilizer = PhaseStabilizer(
            battle_exit_seconds=float(getattr(cfg, "battle_exit_confirm_s", 1.0))
        )
        #: --auto-queue walks a ladder of lobby taps instead of tapping Battle and hoping
        self.lobby = LobbyNavigator()
        self.elixir_reader = ElixirReader(grid.cal, grid=grid)
        self.opp_tracker = OpponentTracker(bait_window_seconds=cfg.bait_window_seconds)
        self.hand_detector = HandCardDetector(
            grid, deck_cards=cfg.normalized_deck(), templates_dir=cfg.templates_dir
        )
        self.identity = UnitIdentityMemory(
            directory=cfg.identity_dir,
            namespace=f"{cfg.target_width}x{cfg.target_height}",
        )
        self.arena_detector = HybridArenaDetector(
            grid,
            yolo_weights_path=cfg.yolo_weights_path,
            yolo_conf=cfg.yolo_confidence,
            enable_opencv_fallback=cfg.enable_opencv_fallback,
            identity_memory=self.identity if cfg.enable_identity_learning else None,
        )
        self.brain = MKZapBaitBrain(grid, config=cfg, adb_latency_sec=cfg.adb_latency_sec)
        label = (
            f"YOLO({self.arena_detector.yolo.mode})+CV"
            if self.arena_detector.yolo.is_active
            else "OpenCV-Fallback"
        )
        self.hud = TacticalDebugHUD(grid, vision_mode_label=label)

    def apply_calibration(self, cal: ArenaCalibration) -> None:
        """Adopt a calibration and rebuild everything that depends on screen geometry."""
        self.calibration = cal
        w, h = self.adb.frame_size if self.adb.frame_size[0] else (self.cfg.target_width, self.cfg.target_height)
        self.grid = ArenaGrid(w, h, cal)
        self._build_vision_stack(self.grid)
        logger.info("Applied calibration (%s) for %dx%d", cal.source, w, h)

    # -- per-frame state ------------------------------------------------------------
    def reset_match_state(self, now_ts: float) -> None:
        self.elixir_reader.reset()
        self.opp_tracker.reset(now_ts)
        self.arena_detector.reset()
        self.brain.reset()
        self.phase_detector.reset_match()

    def process_single_frame(self, frame_bgr: np.ndarray, now_ts: float) -> BattleState:
        """Run vision + tracking on one frame and return the structured state."""
        if frame_bgr is None:
            return BattleState(timestamp=now_ts, phase=GamePhase.UNKNOWN, elixir=0.0, hand=[])
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self.grid.width, self.grid.height):
            self.grid.resize(w, h)

        raw_phase = self.phase_detector.detect_phase(frame_bgr, now_ts)
        phase = self.phase_stabilizer.update(raw_phase, now_ts)

        if phase.is_in_battle and not self._was_in_battle:
            logger.info("=== Match started: resetting trackers ===")
            self.reset_match_state(now_ts)
            self._was_in_battle = True
        elif not phase.is_in_battle and self._was_in_battle:
            logger.info("=== Match ended (%s) ===", raw_phase.value)
            self._was_in_battle = False
            self.matches_played += 1
            self._calibrated_this_match = False

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

    # -- calibration ----------------------------------------------------------------
    def _calibration_file(self) -> Path:
        w, h = self.adb.frame_size
        return calibration_path(self.cfg.calibration_dir, self.adb.serial, w, h)

    def ensure_calibration(self, frame: np.ndarray, interactive_hint: bool = True) -> None:
        """Load a saved calibration for this device, or measure one from ``frame``."""
        if self._calibrated_this_match:
            return
        self._calibrated_this_match = True

        if not self.force_recalibrate:
            saved = load_calibration(self._calibration_file())
            if saved is not None and saved.frame_size is not None:
                self.apply_calibration(saved)
                logger.info("Loaded saved calibration from %s", self._calibration_file())
                return
            portable = load_calibration(Path(self.cfg.calibration_dir) / "default.json")
            if saved is None and portable is not None:
                if portable.frame_size and portable.frame_size != self.adb.frame_size:
                    portable = portable.scaled(*self.adb.frame_size)
                self.apply_calibration(portable)
                logger.info("Loaded portable calibration (configs/calibration/default.json)")
                return

        if not self.auto_calibrate:
            logger.warning(
                "No calibration for this device and auto-calibration is disabled. "
                "Run `python scripts/calibrate_device.py` first."
            )
            return

        result = DeviceCalibrator().calibrate(frame)
        self.calibration_result = result
        logger.info(
            "Auto-calibration confidence %.2f [%s]\n  %s",
            result.confidence,
            result.stage,
            "\n  ".join(result.report),
        )
        self.apply_calibration(result.calibration)

        out_path = self._calibration_file()
        save_calibration(out_path, result.calibration)
        preview_path = out_path.with_suffix(".preview.png")
        try:
            cv2.imwrite(str(preview_path), render_calibration_preview(frame, result.calibration, result))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not write calibration preview: %s", exc)

        if not result.ok:
            logger.warning(
                "Calibration confidence is low (%.2f). The bot will still play, but placements "
                "may be a tile off. Check %s, or run:\n"
                "  python scripts/calibrate_device.py --manual",
                result.confidence,
                preview_path if interactive_hint else "the preview image",
            )
        else:
            logger.info("Calibration saved to %s (preview: %s)", out_path, preview_path)

    # -- deploy execution ------------------------------------------------------------
    def _start_deploy_worker(self) -> None:
        if self._deploy_worker is not None and self._deploy_worker.is_alive():
            return
        self._worker_stop.clear()
        self._deploy_worker = threading.Thread(target=self._deploy_loop, name="deploy-worker", daemon=True)
        self._deploy_worker.start()

    def _deploy_loop(self) -> None:
        while not self._worker_stop.is_set():
            with self._deploy_lock:
                job = self._pending_deploy
                self._pending_deploy = None
            if job is None:
                time.sleep(0.01)
                continue
            action, slot_xy, arena_xy = job
            try:
                outcome = self._execute_deploy(action, slot_xy, arena_xy)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Deploy worker error: %s", exc)
                outcome = DeployOutcome(action, False, "error", 0, 0.0, str(exc))
            self.last_outcomes.append(outcome)
            self.last_outcomes = self.last_outcomes[-20:]
            if outcome.ok:
                self.deploys_ok += 1
            else:
                self.deploys_failed += 1
            self._busy = False

    def _execute_deploy(
        self, action: BotAction, slot_xy: Tuple[int, int], arena_xy: Tuple[int, int]
    ) -> DeployOutcome:
        t0 = time.monotonic()
        before = self._hand_snapshot()
        backends = [self.adb.input_backend] + [b for b in self.cfg.input_backend_order() if b != self.adb.input_backend]
        attempts = 0
        last_note = ""

        for backend in backends:
            attempts += 1
            self.adb.set_input_backend(backend)
            result = self.adb.deploy_card(slot_xy, arena_xy)
            last_note = result.note

            if not self.verify_deploys or before is None:
                elapsed = (time.monotonic() - t0) * 1000.0
                return DeployOutcome(action, result.ok, result.backend, attempts, elapsed, last_note)

            if self._wait_for_hand_change(action.slot_index, before):
                elapsed = (time.monotonic() - t0) * 1000.0
                logger.info(
                    "  verified %s via %s in %.0f ms", action.card_name, result.backend, elapsed
                )
                if self.cfg.enable_identity_learning:
                    self._learn_deployed_unit(action.card_name, arena_xy)
                return DeployOutcome(action, True, result.backend, attempts, elapsed, last_note)

            logger.warning(
                "  deploy of %s via %s did NOT register (hand unchanged); retrying", action.card_name, backend
            )
            if attempts > self.deploy_retries:
                break

        elapsed = (time.monotonic() - t0) * 1000.0
        return DeployOutcome(action, False, self.adb.input_backend, attempts, elapsed, last_note)

    def _learn_deployed_unit(self, card_name: str, arena_xy: Tuple[int, int], budget_s: float = 1.4) -> bool:
        """File a real sprite sample for a troop the bot *knows* it just played.

        ``DeployVerifier`` has already proved the card left the hand, so whatever appears at the
        drop point is this card. That is exact supervision with no human labelling and no model
        training: it just records what this device draws, and later lets the detector name both
        our own and the opponent's copies of that troop.
        """
        learn = getattr(self, "identity", None)
        if learn is None:
            return False
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            frame = self.adb.capture_frame(timeout=0.2)
            if frame is None:
                time.sleep(0.05)
                continue
            self.grid.resize(frame.shape[1], frame.shape[0])
            detections = self.arena_detector.cv_fallback.detect(frame, self.grid) \
                if self.arena_detector.cv_fallback is not None else []
            best = None
            best_d = 1e9
            for det in detections:
                cx = 0.5 * (det.pixel_bbox[0] + det.pixel_bbox[2])
                cy = 0.5 * (det.pixel_bbox[1] + det.pixel_bbox[3])
                dist = ((cx - arena_xy[0]) ** 2 + (cy - arena_xy[1]) ** 2) ** 0.5
                if dist < best_d:
                    best, best_d = det, dist
            if best is not None and best_d <= 2.2 * self._px_per_tile():
                if learn.add_from_frame(card_name, frame, best.pixel_bbox):
                    logger.info("  learned a sprite sample for %s (%d total)",
                                card_name, learn.sample_count)
                return True
            time.sleep(0.08)
        return False

    def _px_per_tile(self) -> float:
        from cr_bot.vision.unit_detector import pixels_per_tile

        try:
            return pixels_per_tile(self.grid)
        except Exception:  # pragma: no cover - defensive
            return 32.0

    def _hand_snapshot(self) -> Optional[SlotSnapshot]:
        frame = self.adb.capture_frame(timeout=0.2)
        if frame is None:
            return None
        self.grid.resize(frame.shape[1], frame.shape[0])
        return self.verifier.snapshot(frame, self.grid)

    def _wait_for_hand_change(self, slot: int, before: SlotSnapshot, timeout: float = 1.1) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self.adb.capture_frame(timeout=0.2)
            if frame is None:
                time.sleep(0.02)
                continue
            self.grid.resize(frame.shape[1], frame.shape[0])
            after = self.verifier.snapshot(frame, self.grid)
            if after is not None and self.verifier.slot_changed(before, after, slot):
                return True
            time.sleep(0.03)
        return False

    def submit_action(self, action: BotAction, state: Optional[BattleState] = None) -> bool:
        """Queue a verified deployment (non-blocking)."""
        if self._busy:
            return False
        slot_xy = self.grid.hand_slot_pixel(action.slot_index)
        arena_xy = self.grid.tile_to_pixel(action.tile_x, action.tile_y)

        if not self.grid.is_legal_deploy_pixel(*arena_xy):
            clamped = self.grid.clamp_deploy_pixel(*arena_xy)
            logger.debug(
                "Deploy point %s rejected as illegal; clamped to %s", arena_xy, clamped
            )
            arena_xy = clamped

        # Last gate for a damage spell, against the frame this action was decided from: if the
        # point we are about to tap no longer covers anything of value (the unit died, the swarm
        # walked on, clamping moved the aim), drop the cast instead of feeding 2 elixir to an
        # empty tile. The brain re-solves the aim before this; this is the belt-and-braces check
        # for any other path that submits an action. Troops are untouched.
        if state is not None and is_damage_spell(action.card_name):
            tile = self.grid.pixel_to_tile(*arena_xy)
            wasted, why = aim_is_wasted(action.card_name, tile[0], tile[1], state.enemy_units)
            if wasted:
                logger.info("Skipped wasted %s: %s", action.card_name, why)
                return False

        with self._deploy_lock:
            self._pending_deploy = (action, slot_xy, arena_xy)
        self._busy = True
        self._start_deploy_worker()
        return True

    # -- interactive viewer (click -> real tap) ---------------------------------------
    def _hand_slot_at(self, point: Tuple[int, int]) -> Optional[int]:
        """Which hand slot does a frame-space point fall on (or ``None``)."""
        x, y = int(point[0]), int(point[1])
        for slot in range(4):
            x1, y1, x2, y2 = self.grid.hand_slot_rect(slot)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return slot
        return None

    def _ensure_clicker(self, frame: np.ndarray) -> ClickToTapController:
        if self.clicker is None:
            self.clicker = ClickToTapController(
                frame_size=(frame.shape[1], frame.shape[0]),
                perform=self.perform_manual_gesture,
                hand_slot_of=self._hand_slot_at,
                enabled=self.manual_input,
            ).start()
        self.clicker.set_frame_size(frame.shape[1], frame.shape[0])
        return self.clicker

    def _bind_mouse(self, preview: PreviewWindow, clicker: ClickToTapController,
                    canvas_size: Tuple[int, int]) -> None:
        """Install the OpenCV mouse callback once per window geometry.

        The window shows the HUD *canvas* — the annotated capture frame with the telemetry
        sidebar stacked on the right — so a click is first mapped from display pixels back to
        canvas pixels, and only clicks that land on the capture-frame part of the canvas are
        turned into gestures. A click on the sidebar is ignored instead of being teleported
        into the right-hand edge of the arena.
        """
        key = (preview.name, canvas_size, clicker.frame_size)
        if self._click_binding_frame == key:
            return
        self._click_binding_frame = key

        def _on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
            if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK,
                             cv2.EVENT_RBUTTONDOWN):
                return
            rect = preview.image_rect()
            canvas_point = map_click_to_frame(
                x, y, preview.shown_size or canvas_size, canvas_size, rect
            )
            if canvas_point is None:
                return
            if event == cv2.EVENT_RBUTTONDOWN:
                clicker.cancel()
                return
            if canvas_point[0] >= clicker.frame_size[0]:
                return  # telemetry sidebar, not the game
            clicker.handle_click(canvas_point[0], canvas_point[1], button="left")

        try:
            cv2.setMouseCallback(preview.name, _on_mouse)
            logger.info(
                "Interactive viewer ready: click the window to tap the phone, click a card "
                "then the arena to deploy it, right-click cancels."
            )
        except Exception as exc:  # noqa: BLE001 - headless/odd backends must not be fatal
            logger.warning("Could not bind clicks in the preview window: %s", exc)

    def perform_manual_gesture(self, gesture: ManualGesture) -> bool:
        """Inject a human-requested gesture (runs on the click worker thread)."""
        result = None
        if gesture.kind == "deploy" and gesture.start is not None:
            result = self.adb.deploy_card(gesture.start, gesture.point)
            logger.info(
                "[manual] deploy %s -> %s via %s (%.0f ms)",
                gesture.start,
                gesture.point,
                result.backend,
                result.latency_ms,
            )
            return bool(result.ok)
        result = self.adb.tap(*gesture.point, hold_ms=gesture.hold_ms)
        logger.info("[manual] tap %s via %s (%.0f ms)", gesture.point, result.backend, result.latency_ms)
        return bool(result.ok)

    # -- main loop -------------------------------------------------------------------
    def run_forever(self, max_matches: Optional[int] = None) -> None:
        self.adb.connect()
        if not self.hand_detector.has_templates:
            logger.warning(
                "No card templates found in '%s'. Run `python scripts/capture_templates.py` "
                "inside a match so the bot can identify your hand.",
                self.cfg.templates_dir,
            )
        logger.info(self.adb.status_line())

        target_period = 1.0 / max(1.0, self.cfg.loop_hz)
        start_wall = time.monotonic()
        fps_ema = self.cfg.loop_hz
        self._start_deploy_worker()

        try:
            while True:
                if max_matches is not None and self.matches_played >= max_matches:
                    logger.info("Completed requested %d match(es).", max_matches)
                    break

                t0 = time.monotonic()
                rel_ts = t0 - start_wall
                self.profiler.begin_frame()

                with self.profiler.stage(CAPTURE):
                    frame = self.adb.capture_frame(timeout=1.0)
                if frame is None:
                    logger.warning("No frame from capture; retrying...")
                    time.sleep(0.05)
                    self.profiler.end_frame()
                    continue
                self._frames_seen += 1

                with self.profiler.stage(VISION):
                    state = self.process_single_frame(frame, rel_ts)

                if state.phase.is_in_battle:
                    self.ensure_calibration(frame)

                    if self.grid.width != frame.shape[1] or self.grid.height != frame.shape[0]:
                        self.grid.resize(frame.shape[1], frame.shape[0])

                    if not self._busy:
                        with self.profiler.stage(BRAIN):
                            action = self.brain.decide(state)
                        if action is not None:
                            self._log_action(state, action, rel_ts)
                            if self.submit_action(action, state):
                                self.hud.record_action(action, rel_ts)
                else:
                    self._handle_non_battle_screen(state.phase, rel_ts)

                self._loop_times.append(time.monotonic() - t0)
                self._loop_times = self._loop_times[-60:]
                dt_proc = max(1e-3, time.monotonic() - t0)
                fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / dt_proc)

                if self.show_hud:
                    hud_started = time.perf_counter()
                    hud_img = self.hud.render(frame, state, self.brain.last_top_candidates, fps=fps_ema)
                    if self._preview is None:
                        self._preview = PreviewWindow(HUD_WINDOW_TITLE)
                    canvas_size = (hud_img.shape[1], hud_img.shape[0])
                    if self.manual_input:
                        clicker = self._ensure_clicker(frame)
                        clicker.render_overlay(hud_img)
                        self._bind_mouse(self._preview, clicker, canvas_size)
                    self._preview.show(hud_img)  # scaled to fit the desktop, never upscaled
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit requested from HUD.")
                        break
                    if key == 27 and self.clicker is not None:  # Esc
                        self.clicker.cancel()
                    # HUD cost spans render + imshow + waitKey (which pumps the GUI event queue,
                    # and therefore the mouse callback), so it is timed as one block.
                    self.profiler.add(HUD, time.perf_counter() - hud_started)

                sleep_rem = target_period - (time.monotonic() - t0)
                if sleep_rem > 0:
                    with self.profiler.stage(SLEEP):
                        time.sleep(sleep_rem)
                self.profiler.end_frame()
                self._log_performance(rel_ts)

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            self._worker_stop.set()
            if self.clicker is not None:
                self.clicker.stop()
            if self.show_hud:
                if self._preview is not None:
                    self._preview.close()
                cv2.destroyAllWindows()
            logger.info(
                "Session summary: %d matches, %d deploys verified, %d failed, capture %s",
                self.matches_played,
                self.deploys_ok,
                self.deploys_failed,
                f"{self.adb.fps:.1f} fps",
            )
            self.adb.close()

    #: how often the frame-cost breakdown is logged (seconds)
    PERF_LOG_INTERVAL_S = 30.0

    def _log_performance(self, rel_ts: float) -> None:
        """Log where the frame time went, once every ``PERF_LOG_INTERVAL_S``.

        Without this, "the fps is low" is unanswerable: capture, vision, the HUD and the configured
        loop rate all look the same from the outside. The line names the bottleneck explicitly so
        the next optimisation is a measurement rather than a guess.
        """
        if rel_ts - self._last_perf_log_ts < self.PERF_LOG_INTERVAL_S:
            return
        if self.profiler.frames < 10:
            return
        self._last_perf_log_ts = rel_ts
        summary = self.profiler.summary()
        if not summary:
            return
        worst = summary[0]
        logger.info("Frame cost: %s", self.profiler.report())
        logger.info(
            "  bottleneck: %s (%.1f ms mean, %.1f ms p95, %.0f%% of the frame) — "
            "loop target is %.0f Hz",
            worst[0], worst[1], worst[2], worst[3] * 100.0, self.cfg.loop_hz,
        )

    def _log_action(self, state: BattleState, action: BotAction, rel_ts: float) -> None:
        logger.info(
            "[%.1fs | E %.1f vs %.1f] %s (slot %d) -> tile(%.1f, %.1f) pixel%s | score %.0f | %s",
            rel_ts,
            state.elixir,
            state.opponent.estimated_elixir,
            action.card_name,
            action.slot_index,
            action.tile_x,
            action.tile_y,
            self.grid.tile_to_pixel(action.tile_x, action.tile_y),
            action.score,
            action.reason,
        )

    def _handle_non_battle_screen(self, phase: GamePhase, now_ts: float) -> None:
        """Drive the game back into a match when ``--auto-queue`` is on.

        Two stages, because the lobby is not a single button:

        1. ``MATCH_OVER`` — dismiss the result screen (OK / Return home).
        2. anything else outside a battle — walk :class:`LobbyNavigator`'s ladder one tap at a
           time (Battle → mode tile → Play). The old version tapped Battle and gave up, which
           stalled on the mode-selection screen that sits between the two.

        Every tap is rate limited, and the ladder is only reset when a match actually starts.
        """
        if not self.cfg.auto_queue_battle:
            return
        if (now_ts - self._last_menu_click_ts) < 3.0:
            return

        if phase == GamePhase.MATCH_OVER:
            ox, oy = self.phase_detector.match_over_ok_button_xy(self.grid.width, self.grid.height)
            logger.info("Match over -> OK (%d, %d)", ox, oy)
            self.adb.tap(ox, oy)
            self._last_menu_click_ts = now_ts
            self.lobby.reset()
            return

        if phase.is_in_battle:
            self.lobby.reset()
            return

        # UNKNOWN counts too: it is the normal reading between two lobby screens, and refusing to
        # act on it is what made the queue wait for something else to happen.
        tap = self.lobby.next_tap(now_ts)
        if tap is None:
            if self.lobby.finished:
                logger.debug("auto-queue: ladder exhausted, waiting for the lobby to settle")
                self.lobby.reset()
            return
        step, rel_xy = tap
        x, y = absolute(rel_xy, self.grid.width, self.grid.height)
        logger.info("auto-queue -> %s (%d, %d)", step, x, y)
        self.adb.tap(x, y)
        self._last_menu_click_ts = now_ts
