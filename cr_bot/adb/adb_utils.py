"""
Low-level ADB helpers shared by the capture and input layers.

Everything here is deliberately defensive: real phones and real ADB installs differ a lot
(multiple devices, "unauthorized" states, `wm size` reporting an override, a locked
keyguard, the screen asleep, a 20:9 display). Each of those silently breaks a bot that
assumes a happy path.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
    "/opt/homebrew/bin/adb",
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


def find_adb(preferred: str = "adb") -> str:
    """Resolve an adb binary from PATH, an explicit path, or the usual emulator installs."""
    found = shutil.which(preferred)
    if found:
        return found
    if Path(preferred).is_file():
        return str(Path(preferred))
    for candidate in COMMON_ADB_PATHS:
        if Path(candidate).is_file():
            logger.info("Auto-detected ADB binary at: %s", candidate)
            return candidate
    return preferred


@dataclass
class DisplayInfo:
    """Physical display facts that every coordinate conversion depends on."""

    width: int
    height: int
    rotation: int = 0
    density: int = 0
    override_applied: bool = False

    @property
    def is_portrait(self) -> bool:
        return self.height >= self.width


class Adb:
    """Small typed wrapper around the adb CLI for one (optional) device serial."""

    def __init__(self, adb_path: str = "adb", serial: Optional[str] = None) -> None:
        self.adb_bin = find_adb(adb_path)
        self.serial = serial

    # -- process plumbing -------------------------------------------------------------
    def base_cmd(self, include_serial: bool = True) -> List[str]:
        cmd = [self.adb_bin]
        if include_serial and self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def run(
        self,
        args: Sequence[str],
        timeout: float = 10.0,
        include_serial: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        try:
            res = subprocess.run(
                self.base_cmd(include_serial) + list(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ADBError(
                f"ADB executable '{self.adb_bin}' not found. Install Android platform-tools, "
                "or set `adb.adb_path` in configs/mk_zap_bait.yaml."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ADBError(f"adb {' '.join(args)} timed out after {timeout}s") from exc
        if check and res.returncode != 0:
            raise ADBError(f"adb {' '.join(args)} failed: {res.stderr.decode('utf-8', 'replace').strip()}")
        return res

    def shell(self, command, timeout: float = 10.0) -> str:
        """Run a shell command. Accepts a string or an argv list."""
        if isinstance(command, (list, tuple)):
            args = ["shell"] + [str(c) for c in command]
        else:
            args = ["shell", str(command)]
        return self.run(args, timeout=timeout).stdout.decode("utf-8", "replace")

    def exec_out(self, args: Sequence[str], timeout: float = 10.0) -> bytes:
        return self.run(["exec-out"] + list(args), timeout=timeout).stdout

    def push(self, local: str | Path, remote_dir: str = "/data/local/tmp/", timeout: float = 60.0) -> None:
        res = self.run(["push", str(local), remote_dir], timeout=timeout)
        if res.returncode != 0:
            raise ADBError(f"adb push failed: {res.stderr.decode('utf-8', 'replace').strip()}")

    def forward(self, local_port: int, remote: str = "localabstract:scrcpy") -> None:
        res = self.run(["forward", f"tcp:{local_port}", remote], timeout=10.0, include_serial=True)
        if res.returncode != 0:
            raise ADBError(f"adb forward failed: {res.stderr.decode('utf-8', 'replace').strip()}")

    def forward_list(self) -> str:
        return self.run(["forward", "--list"], include_serial=False, timeout=10.0).stdout.decode("utf-8", "replace")

    # -- discovery --------------------------------------------------------------------
    def list_devices(self) -> List[str]:
        res = self.run(["devices"], include_serial=False, timeout=10.0)
        devices: List[str] = []
        for line in res.stdout.decode("utf-8", "replace").splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    def connect(self, serial: Optional[str] = None, autoprobe_emulators: bool = True) -> str:
        """Attach to a device: honour an explicit serial, else discover one."""
        target = serial or self.serial
        if target and _is_network_serial(target):
            self.run(["connect", target], include_serial=False, timeout=8.0)

        devices = self.list_devices()
        if not devices and autoprobe_emulators and (target is None or _is_network_serial(target)):
            logger.info("No ADB device attached; probing common emulator ports...")
            for host_port in COMMON_EMULATOR_PORTS:
                self.run(["connect", host_port], include_serial=False, timeout=3.0)
                devices = self.list_devices()
                if devices:
                    break

        if not devices:
            raise ADBError(
                "No Android device found over ADB.\n"
                "  - Enable USB debugging / 'Wireless debugging' on the phone and accept the RSA prompt.\n"
                "  - Or set `adb.serial` in configs/mk_zap_bait.yaml (see `adb devices`)."
            )

        if target and target in devices:
            self.serial = target
        elif target and _is_network_serial(target) and target not in devices:
            raise ADBError(f"Requested device '{target}' is not available. Attached: {devices}")
        else:
            self.serial = devices[0]
            if len(devices) > 1:
                logger.warning(
                    "Multiple devices attached %s; using '%s'. Set `adb.serial` to be explicit.",
                    devices,
                    self.serial,
                )
        return self.serial

    # -- display ----------------------------------------------------------------------
    def display_info(self) -> DisplayInfo:
        """Parse `wm size` + `wm density`+ rotation, preferring the *override* size."""
        text = self.shell("wm size", timeout=8.0)
        override = None
        physical = None
        for line in text.splitlines():
            m = re.search(r"(\d+)\s*x\s*(\d+)", line)
            if not m:
                continue
            w, h = int(m.group(1)), int(m.group(2))
            if "override" in line.lower():
                override = (w, h)
            elif "physical" in line.lower():
                physical = (w, h)
            elif physical is None and override is None:
                physical = (w, h)

        chosen = override or physical
        if chosen is None:
            raise ADBError("Could not determine device resolution from `wm size`.")
        w, h = chosen

        density = 0
        m = re.search(r"(\d+)", self.shell("wm density", timeout=8.0))
        if m:
            density = int(m.group(1))

        rotation = self.rotation()
        # `wm size` reports the natural orientation; if the device is rotated a quarter turn,
        # the *visible* frame is transposed. Report the size as currently displayed.
        if rotation in (1, 3):
            w, h = h, w

        return DisplayInfo(
            width=w,
            height=h,
            rotation=rotation,
            density=density,
            override_applied=override is not None,
        )

    def rotation(self) -> int:
        """0/1/2/3 rotation as reported by the window manager (best effort)."""
        try:
            out = self.shell("dumpsys input | grep -m1 SurfaceOrientation", timeout=8.0)
            m = re.search(r"SurfaceOrientation:\s*(\d)", out)
            if m:
                return int(m.group(1))
        except ADBError:
            pass
        try:
            out = self.shell("dumpsys window displays | grep -m1 cur=", timeout=8.0)
            m = re.search(r"cur=(\d+)x(\d+)", out)
            if m:
                # A transposed `cur` means we are in a landscape rotation.
                return 1 if int(m.group(1)) > int(m.group(2)) else 0
        except ADBError:
            pass
        return 0

    # -- device hygiene ---------------------------------------------------------------
    def ensure_ready_for_play(self, keep_awake: bool = True) -> None:
        """Wake the screen, dismiss the keyguard and stop the display from sleeping."""
        try:
            self.shell("input keyevent KEYCODE_WAKEUP", timeout=8.0)
            awake = self.shell("dumpsys power | grep -m1 'mWakefulness='", timeout=8.0)
            if "Awake" not in awake:
                self.shell("input keyevent 224", timeout=8.0)  # KEYCODE_WAKEUP
            # Swipe up to clear a swipe-style keyguard, then dismiss any PIN-less lock.
            self.shell("wm dismiss-keyguard", timeout=8.0)
            if keep_awake:
                self.shell("svc power stayon true", timeout=8.0)
                self.shell("settings put system screen_off_timeout 2147483647", timeout=8.0)
        except ADBError as exc:
            logger.debug("Device hygiene step failed (non fatal): %s", exc)

    def lock_portrait(self) -> None:
        """Pin the display to natural (portrait) orientation so touch maths stays valid."""
        try:
            self.shell("settings put system accelerometer_rotation 0", timeout=8.0)
            self.shell("settings put system user_rotation 0", timeout=8.0)
        except ADBError as exc:
            logger.debug("Could not lock orientation (non fatal): %s", exc)

    #: name of the resumed-activity line, newest Android first
    _FOCUS_PROBES = (
        "dumpsys activity activities | grep -m2 topResumedActivity",
        "dumpsys activity activities | grep -m2 mResumedActivity",
        "dumpsys window windows | grep -m3 -e mCurrentFocus -e mFocusedApp",
        "dumpsys window | grep -m3 -e mCurrentFocus -e mFocusedApp",
    )

    def foreground_package(self) -> str:
        """Package currently in front.

        Android 12+ renamed the dumped field (``mResumedActivity`` → ``topResumedActivity``) and
        Android 14 moved it again, so a single grep reports nothing on modern phones - which is why
        ``--stats`` used to print ``foreground: ?`` and ``clash royale: NO`` on a phone that was
        demonstrably showing the game. Probe newest-first and take the first hit.
        """
        for probe in self._FOCUS_PROBES:
            try:
                out = self.shell(probe, timeout=8.0)
            except ADBError:
                return ""
            if not out or not out.strip():
                continue
            for pattern in (r"([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.]+", r"u0\s+([a-zA-Z0-9_.]+)/"):
                m = re.search(pattern, out)
                if m:
                    pkg = m.group(1)
                    if pkg and not pkg.startswith("android."):
                        return pkg
        return ""

    def is_clash_royale_foreground(
        self,
        packages: Optional[Sequence[str]] = None,
        hints: Optional[Sequence[str]] = None,
    ) -> bool:
        """True when *any* configured build of the game is in front.

        Custom APKs ("Nulls Royale", modded clients, private servers) are the same game with a
        different application id, so the package name must never be hard-coded. ``packages`` are
        accepted as-is; ``hints`` match any substring case-insensitively.
        """
        return is_game_package(self.foreground_package(), packages, hints)

    def find_game_package(
        self,
        packages: Optional[Sequence[str]] = None,
        hints: Optional[Sequence[str]] = None,
    ) -> str:
        """Look through the installed packages for a build of the game.

        Used by ``--stats`` when the game is not in the foreground: instead of "open the game", we
        can name the exact package that is installed and offer to launch it.
        """
        try:
            listing = self.shell("pm list packages -3", timeout=12.0)
        except ADBError:
            return ""
        installed = [
            line.split("package:", 1)[1].strip()
            for line in listing.splitlines()
            if line.strip().startswith("package:")
        ]
        for pkg in installed:
            if is_game_package(pkg, packages, hints):
                return pkg
        return ""

    def launch_package(self, package: str, timeout: float = 12.0) -> bool:
        """Bring an app to the foreground (works for any package, including custom APKs)."""
        if not package:
            return False
        try:
            out = self.shell(
                f"monkey -p {package} -c android.intent.category.LAUNCHER 1", timeout=timeout
            )
        except ADBError:
            return False
        return "No activities found" not in out and "aborted" not in out.lower()


#: Stock application id of the game.
GAME_PACKAGE = "com.supercell.clashroyale"

#: Known custom/private-server builds. Matched exactly.
GAME_PACKAGES: Tuple[str, ...] = (GAME_PACKAGE, "nullsroyale.rel.free")

#: Substrings that identify a build of this game regardless of its application id.
GAME_PACKAGE_HINTS: Tuple[str, ...] = (
    "clashroyale",
    "clash_royale",
    "nullsroyale",
    "clash.royale",
)


def is_game_package(
    package: str,
    packages: Optional[Sequence[str]] = None,
    hints: Optional[Sequence[str]] = None,
) -> bool:
    """Does ``package`` look like a build of Clash Royale?

    ``packages`` (exact ids) and ``hints`` (substrings) default to the built-in lists above, so a
    custom APK such as ``nullsroyale.rel.free`` is recognised without any configuration.
    """
    if not package:
        return False
    pkg = package.strip()
    exact = tuple(packages) if packages is not None else GAME_PACKAGES
    for candidate in exact:
        if candidate and (pkg == candidate or pkg.startswith(candidate + ".")):
            return True
    for hint in (tuple(hints) if hints is not None else GAME_PACKAGE_HINTS):
        if hint and hint.lower() in pkg.lower():
            return True
    return False


def _is_network_serial(serial: str) -> bool:
    """True for `host:port` serials (emulators + wireless ADB)."""
    if ":" not in serial:
        return True  # unknown form: still worth a `connect` attempt (adb ignores valid USB serials)
    host = serial.rsplit(":", 1)[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def parse_wm_size(text: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Return (physical, override) parsed from `wm size` output. Exposed for tests."""
    override = physical = None
    for line in text.splitlines():
        m = re.search(r"(\d+)\s*x\s*(\d+)", line)
        if not m:
            continue
        pair = (int(m.group(1)), int(m.group(2)))
        if "override" in line.lower():
            override = pair
        elif "physical" in line.lower():
            physical = pair
    return physical, override
