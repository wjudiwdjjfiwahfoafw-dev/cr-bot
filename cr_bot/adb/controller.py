"""Fast ADB controller for BlueStacks, LDPlayer, MEmu, MuMu, and Android devices."""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COMMON_ADB_PATHS = [
    r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
    r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
    r"C:\LDPlayer\LDPlayer9\adb.exe",
    r"C:\LDPlayer\LDPlayer4.0\adb.exe",
    r"C:\Program Files\Microvirt\MEmu\adb.exe",
    r"C:\Program Files\Netease\MuMuPlayerGlobal-12.0\shell\adb.exe",
    os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
    os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
    "/usr/bin/adb",
    "/usr/local/bin/adb",
]

COMMON_EMULATOR_PORTS = [
    "127.0.0.1:5555",
    "127.0.0.1:5556",
    "127.0.0.1:5565",
    "127.0.0.1:5554",
    "127.0.0.1:21503",  # MEmu
    "127.0.0.1:16384",  # MuMu 12
    "127.0.0.1:7555",   # MuMu 6
    "127.0.0.1:62001",  # Nox
]


class ADBError(RuntimeError):
    """Raised when ADB communication or device discovery fails."""


class ADBController:
    """Handles low-latency screenshot streaming and humanized touch input over ADB."""

    def __init__(
        self,
        adb_path: str = "adb",
        serial: Optional[str] = None,
        target_width: int = 720,
        target_height: int = 1280,
        jitter_px: int = 4,
        dry_run: bool = False,
    ) -> None:
        self.adb_bin = self._resolve_adb_binary(adb_path)
        self.serial = serial
        self.target_width = target_width
        self.target_height = target_height
        self.device_width = target_width
        self.device_height = target_height
        self.jitter_px = jitter_px
        self.dry_run = dry_run
        self._rng = random.Random()

    @staticmethod
    def _resolve_adb_binary(preferred: str) -> str:
        found = shutil.which(preferred)
        if found:
            return found
        if Path(preferred).is_file():
            return str(Path(preferred))
        for candidate in COMMON_ADB_PATHS:
            if Path(candidate).is_file():
                logger.info("Auto-detected emulator ADB binary at: %s", candidate)
                return candidate
        return preferred

    def _base_cmd(self) -> List[str]:
        cmd = [self.adb_bin]
        if self.serial:
            cmd.extend(["-s", self.serial])
        return cmd

    def _run(self, args: List[str], timeout: float = 5.0) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                self._base_cmd() + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ADBError(
                f"ADB executable '{self.adb_bin}' not found. Install Android Platform Tools or "
                "enable ADB in BlueStacks/LDPlayer settings and set `adb.adb_path` in configs/mk_zap_bait.yaml."
            ) from exc

    def list_devices(self) -> List[str]:
        res = subprocess.run(
            [self.adb_bin, "devices"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=False,
        )
        lines = res.stdout.decode("utf-8", errors="ignore").splitlines()
        devices: List[str] = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    def connect(self) -> str:
        """Connect to the configured or auto-discovered Android emulator and query its screen size."""
        if self.serial and ":" in self.serial:
            subprocess.run(
                [self.adb_bin, "connect", self.serial],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=4.0,
                check=False,
            )

        devices = self.list_devices()
        if not devices:
            logger.info("No active ADB devices listed; probing common emulator ports...")
            for host_port in COMMON_EMULATOR_PORTS:
                subprocess.run(
                    [self.adb_bin, "connect", host_port],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=2.0,
                    check=False,
                )
                devices = self.list_devices()
                if devices:
                    break

        if not devices:
            raise ADBError(
                "No Android emulator detected over ADB. Make sure BlueStacks/LDPlayer is open "
                "with 'Android Debug Bridge (ADB)' enabled in Emulator Settings -> Advanced."
            )

        if not self.serial or self.serial not in devices:
            self.serial = devices[0]

        self._detect_device_resolution()
        logger.info(
            "Connected to emulator %s (physical=%dx%d -> normalized=%dx%d)",
            self.serial,
            self.device_width,
            self.device_height,
            self.target_width,
            self.target_height,
        )
        return self.serial

    def _detect_device_resolution(self) -> None:
        res = self._run(["shell", "wm", "size"], timeout=3.0)
        text = res.stdout.decode("utf-8", errors="ignore")
        matches = re.findall(r"(\d+)x(\d+)", text)
        if matches:
            # Prefer Override size if present (last match)
            w, h = int(matches[-1][0]), int(matches[-1][1])
            # Ensure portrait orientation assumption matches Clash Royale
            if w > h:
                w, h = h, w
            self.device_width, self.device_height = w, h

    def capture_frame(self, prefer_raw: bool = True) -> Optional[np.ndarray]:
        """
        Capture a BGR frame over ADB normalized to target resolution.
        Tries zero-compression raw RGBA `screencap` first (~25ms vs ~70ms PNG), with automatic
        fallback to `screencap -p` if the emulator uses a non-RGBA framebuffer format.
        """
        frame: Optional[np.ndarray] = None
        if prefer_raw:
            frame = self._capture_raw_rgba()
        if frame is None:
            frame = self._capture_png()
        if frame is None:
            return None

        h, w = frame.shape[:2]
        self.device_width, self.device_height = w, h
        if (w, h) != (self.target_width, self.target_height):
            frame = cv2.resize(
                frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA
            )
        return frame

    def _capture_raw_rgba(self) -> Optional[np.ndarray]:
        res = self._run(["exec-out", "screencap"], timeout=2.5)
        raw = res.stdout
        if not raw or len(raw) < 16:
            return None
        # Android raw screencap header: uint32 width, uint32 height, uint32 format (1 = RGBA_8888)
        # Android 12+ may prepend an extra 4-byte colorSpace field (16-byte header)
        w, h, fmt = int.from_bytes(raw[0:4], "little"), int.from_bytes(raw[4:8], "little"), int.from_bytes(raw[8:12], "little")
        if fmt != 1 or w <= 0 or h <= 0 or w > 4096 or h > 4096:
            return None
        expected = w * h * 4
        if len(raw) == expected + 12:
            offset = 12
        elif len(raw) == expected + 16:
            offset = 16
        else:
            return None
        rgba = np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape((h, w, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def _capture_png(self) -> Optional[np.ndarray]:
        res = self._run(["exec-out", "screencap", "-p"], timeout=3.0)
        raw = res.stdout
        if not raw or len(raw) < 100:
            return None

        # Handle rare older Windows ADB builds that convert LF -> CRLF
        if raw[:8] != b"\x89PNG\r\n\x1a\n" and b"\x89PNG\r\r\n\x1a\r\n" in raw[:16]:
            raw = raw.replace(b"\r\n", b"\n")

        buf = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _to_device_xy(self, norm_x: int, norm_y: int, apply_jitter: bool = True) -> Tuple[int, int]:
        jx = int(round(self._rng.gauss(0, self.jitter_px * 0.5))) if apply_jitter else 0
        jy = int(round(self._rng.gauss(0, self.jitter_px * 0.5))) if apply_jitter else 0
        jx = max(-self.jitter_px, min(self.jitter_px, jx))
        jy = max(-self.jitter_px, min(self.jitter_px, jy))

        x = max(2, min(self.target_width - 2, norm_x + jx))
        y = max(2, min(self.target_height - 2, norm_y + jy))

        dev_x = int(round(x * (self.device_width / float(self.target_width))))
        dev_y = int(round(y * (self.device_height / float(self.target_height))))
        return dev_x, dev_y

    def tap(self, x: int, y: int, apply_jitter: bool = True) -> None:
        dev_x, dev_y = self._to_device_xy(x, y, apply_jitter=apply_jitter)
        if self.dry_run:
            logger.info("[DRY-RUN] tap(%d, %d)", dev_x, dev_y)
            return
        self._run(["shell", "input", "tap", str(dev_x), str(dev_y)], timeout=2.0)

    def deploy_card(self, slot_xy: Tuple[int, int], arena_xy: Tuple[int, int]) -> None:
        """Select a hand card slot then tap the target arena coordinate with human-like timing."""
        sx, sy = self._to_device_xy(slot_xy[0], slot_xy[1], apply_jitter=True)
        ax, ay = self._to_device_xy(arena_xy[0], arena_xy[1], apply_jitter=True)

        if self.dry_run:
            logger.info("[DRY-RUN] deploy_card slot=(%d,%d) -> arena=(%d,%d)", sx, sy, ax, ay)
            return

        # Single shell invocation avoids double ADB process spawn overhead (~60ms faster!)
        delay_sec = round(self._rng.uniform(0.055, 0.088), 3)
        shell_script = f"input tap {sx} {sy}; sleep {delay_sec}; input tap {ax} {ay}"
        self._run(["shell", shell_script], timeout=2.5)
        time.sleep(0.02)
