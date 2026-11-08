"""
ADB controller facade: scrcpy capture + low-latency touch injection for Clash Royale.

This replaces the original implementation, which had two problems that together explain
"screen capture works but cards never get placed":

* **Capture** used ``adb exec-out screencap``. On a real phone that is a 150-400 ms round
  trip, so the bot was always acting on a stale frame. It now streams over scrcpy through
  :class:`~cr_bot.adb.scrcpy_multiclient.MultiClient` and always reads the newest frame.
* **Input** used two ``input tap`` calls with a nominal 60 ms gap. Each ``input`` invocation
  spawns a JVM on the device, and a zero-length tap is routinely dropped by the game. Cards
  are now deployed with a real drag gesture, injected through the scrcpy control socket when
  available (sub-10 ms, persistent connection, genuine press durations), with ``input swipe``
  and ``input motionevent`` as fallbacks.

Coordinates are handled in **frame space** and converted to device pixels exactly once, in
:class:`~cr_bot.adb.touch.CoordMap`, so a downscaled capture stream can never shift taps.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from cr_bot.adb.adb_utils import Adb, ADBError, COMMON_ADB_PATHS, COMMON_EMULATOR_PORTS  # noqa: F401
from cr_bot.adb.scrcpy_multiclient import MultiClient
from cr_bot.adb.touch import CoordMap, DEFAULT_ORDER, GestureResult, TouchInjector

logger = logging.getLogger(__name__)

__all__ = ["ADBController", "ADBError", "COMMON_ADB_PATHS", "COMMON_EMULATOR_PORTS"]


class ADBController:
    """Owns the ADB connection, the capture stream and the touch injector."""

    def __init__(
        self,
        adb_path: str = "adb",
        serial: Optional[str] = None,
        target_width: int = 720,
        target_height: int = 1280,
        jitter_px: int = 2,
        dry_run: bool = False,
        capture_backend: str = "auto",
        input_backend: str = "auto",
        deploy_style: str = "drag",
        capture_bitrate: int = 12_000_000,
        scrcpy_jar: Optional[str] = None,
        capture_max_fps: int = 30,
        keep_awake: bool = True,
        lock_portrait: bool = True,
        input_order: Sequence[str] = DEFAULT_ORDER,
    ) -> None:
        self.adb_bin = adb_path
        self.serial = serial
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.jitter_px = int(jitter_px)
        self.dry_run = bool(dry_run)
        self.capture_backend = capture_backend
        self._input_backend_requested = input_backend
        self.deploy_style = deploy_style
        self.capture_bitrate = int(capture_bitrate)
        self.scrcpy_jar = scrcpy_jar
        self.capture_max_fps = int(capture_max_fps)
        self.keep_awake = bool(keep_awake)
        self.lock_portrait = bool(lock_portrait)
        self.input_order = tuple(input_order)

        self.adb: Optional[Adb] = None
        self.client: Optional[MultiClient] = None
        self.touch: Optional[TouchInjector] = None
        self.coord: Optional[CoordMap] = None
        self.device_size: Tuple[int, int] = (0, 0)
        self.frame_size: Tuple[int, int] = (0, 0)
        self.backend_name = "none"
        self._last_frame: Optional[np.ndarray] = None
        self._last_seq = -1

    # -- lifecycle -----------------------------------------------------------------
    def connect(self) -> str:
        self.adb = Adb(self.adb_bin, self.serial)
        serial = self.adb.connect(self.serial)
        self.serial = serial

        if self.keep_awake:
            self.adb.ensure_ready_for_play()
        if self.lock_portrait:
            self.adb.lock_portrait()

        info = self.adb.display_info()
        self.device_size = (info.width, info.height)
        if info.width > info.height:
            logger.warning(
                "Device reports a landscape display (%dx%d). Clash Royale must be in portrait "
                "for coordinate mapping to be valid.",
                info.width,
                info.height,
            )

        self.client = MultiClient(
            adb_path=self.adb_bin,
            serial=serial,
            target_width=self.target_width,
            bitrate=self.capture_bitrate,
            max_fps=self.capture_max_fps,
            backend=self.capture_backend,
            scrcpy_jar=self.scrcpy_jar,
        )
        self.client.start()
        session = self.client.session()
        self.backend_name = getattr(self.client, "backend_used", "") or type(session).__name__
        if getattr(self.client, "fallback_reason", ""):
            logger.warning(
                "Capture downgraded to %s (live scrcpy unavailable). Deploy timing still works, "
                "but reactions lag by one frame.", self.backend_name
            )

        frame = session.latest_frame(timeout=5.0)
        if frame is None:
            raise ADBError("Capture started but produced no frames")
        self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        self._last_frame = frame

        self.coord = CoordMap(
            frame_w=self.frame_size[0],
            frame_h=self.frame_size[1],
            device_w=self.device_size[0],
            device_h=self.device_size[1],
        )
        self.touch = TouchInjector(
            adb=self.adb,
            coord=self.coord,
            control=getattr(session, "control", None),
            order=self.input_order,
            deploy_style=self.deploy_style,
            jitter_px=self.jitter_px,
        )
        if self._input_backend_requested not in ("auto", "default", None):
            self.touch.set_active(self._input_backend_requested)

        logger.info(
            "Connected to %s | capture=%s %dx%d | device=%dx%d | %s",
            serial,
            self.backend_name,
            self.frame_size[0],
            self.frame_size[1],
            self.device_size[0],
            self.device_size[1],
            self.touch.describe(),
        )
        return serial

    def close(self) -> None:
        if self.client is not None:
            self.client.stop()
            self.client = None

    def __enter__(self) -> "ADBController":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- capture -------------------------------------------------------------------
    def capture_frame(self, min_seq: int = 0, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Newest frame from the live stream (no screencap round trip)."""
        if self.client is None:
            return None
        frame = self.client.latest_frame(min_seq=min_seq, timeout=timeout)
        if frame is not None:
            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                if self.coord is not None:
                    self.coord.frame_w, self.coord.frame_h = self.frame_size
            self._last_frame = frame
        return frame

    @property
    def frame_seq(self) -> int:
        return 0 if self.client is None else self.client.frame_seq()

    @property
    def fps(self) -> float:
        return 0.0 if self.client is None else self.client.fps

    @property
    def frame_age(self) -> float:
        return float("inf") if self.client is None else self.client.frame_age

    def wait_new_frame(self, last_seq: int, timeout: float = 0.4) -> Optional[np.ndarray]:
        """Block until the stream advances past ``last_seq`` (or time out)."""
        if self.client is None:
            return None
        session = self.client.session()
        waited = time.monotonic()
        while time.monotonic() - waited < timeout:
            if session.frame_seq > last_seq:
                return session.latest_frame()
            time.sleep(0.004)
        return session.latest_frame()

    # -- input ---------------------------------------------------------------------
    def tap(self, x: int, y: int, hold_ms: float = 70.0) -> GestureResult:
        assert self.touch is not None, "connect() first"
        if self.dry_run:
            logger.info("[DRY-RUN] tap(%d, %d)", x, y)
            return GestureResult(ok=True, backend="dry-run", kind="tap", start=(x, y), end=(x, y))
        return self.touch.tap_abs((x, y), hold_ms=hold_ms)

    def deploy_card(
        self,
        slot_xy: Tuple[int, int],
        arena_xy: Tuple[int, int],
        style: Optional[str] = None,
        force_backend: Optional[str] = None,
    ) -> GestureResult:
        """Drag the card at ``slot_xy`` onto ``arena_xy`` (frame-space pixels)."""
        assert self.touch is not None, "connect() first"
        if self.dry_run:
            logger.info("[DRY-RUN] deploy slot=%s -> arena=%s", slot_xy, arena_xy)
            return GestureResult(
                ok=True, backend="dry-run", kind="drag", start=slot_xy, end=arena_xy
            )
        return self.touch.play_card(
            slot_xy, arena_xy, style=style, force_backend=force_backend
        )

    def set_input_backend(self, name: str) -> bool:
        return bool(self.touch and self.touch.set_active(name))

    @property
    def input_backend(self) -> str:
        return self.touch.active if self.touch else "none"

    # -- diagnostics ---------------------------------------------------------------
    def status_line(self) -> str:
        return (
            f"{self.serial} | capture={self.backend_name} {self.frame_size[0]}x{self.frame_size[1]} "
            f"@{self.fps:.1f}fps (age {self.frame_age * 1000:.0f}ms) | input={self.input_backend} | "
            f"{self.touch.stats() if self.touch else 'no gestures'}"
        )
