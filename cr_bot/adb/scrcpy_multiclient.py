"""
Scrcpy-backed screen capture for one or many Android devices ("MultiClient").

Why this exists
---------------
The original bot captured with ``adb exec-out screencap``. On an emulator that is tolerable;
on a **real phone** a full-resolution screencap round-trip costs roughly 150-400 ms. A Clash
Royale bot that reasons about a 300 ms old frame is always a beat behind: it reacts to
troops that have already moved, mis-reads elixir, and "feels" broken even when its tactics
are fine.

This module keeps a live H.264 stream open through the scrcpy server and exposes an
always-latest-frame buffer at 30-60 fps, so the vision pipeline sees the game as it is now.
The same connection also gives us a low-latency control channel for touch injection (see
``cr_bot.adb.touch``).

Backends
--------
``auto``        native protocol implementation (default, no third-party wrapper needed)
``scrcpy_py``   use the ``scrcpy`` package (py-scrcpy-client) if installed
``screencap``   legacy ``adb exec-out screencap`` polling

The native backend implements the scrcpy 1.24 wire protocol directly. It deliberately does
*not* depend on ``scrcpy-client`` because that package pins ``av<10`` and ``adbutils<0.15``
and therefore cannot be installed on current Python, and because its touch packets are
malformed (31 bytes where the server expects 27) — which is one of the classic reasons a
scrcpy-driven bot can mirror the screen but never actually tap.
"""

from __future__ import annotations

import collections
import logging
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cr_bot.adb import scrcpy_protocol as proto
from cr_bot.adb.adb_utils import Adb, ADBError, DisplayInfo

logger = logging.getLogger(__name__)

#: Fallback name, used when nothing newer is bundled.
JAR_NAME = "scrcpy-server-v1.24.jar"
#: glob for every bundled server; the newest version wins
JAR_GLOB = "scrcpy-server-v*"


def _third_party_dirs() -> List[Path]:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    dirs = [repo_root / "cr_bot" / "third_party", here.parent / "third_party", here]
    try:  # reuse the jar shipped with the pip package when it happens to be installed
        import scrcpy as _scrcpy_pkg  # noqa: F401

        dirs.append(Path(_scrcpy_pkg.__file__).parent)
    except Exception:  # noqa: BLE001
        pass
    return dirs


def available_jars() -> List[Path]:
    """Every bundled server jar, oldest first."""
    found: Dict[str, Path] = {}
    for directory in _third_party_dirs():
        if not directory.is_dir():
            continue
        for path in directory.glob(JAR_GLOB):
            if path.is_file():
                found[str(path.resolve())] = path
        legacy = directory / JAR_NAME
        if legacy.is_file():
            found[str(legacy.resolve())] = legacy
    return sorted(found.values(), key=lambda p: proto.parse_server_version(p.name))


def find_scrcpy_jar(prefer: Optional[str] = None) -> Path:
    """Newest bundled server jar, or an explicit one.

    The version matters: a 1.x server cannot capture on Android 13+, where the display-capture
    APIs it uses were changed. Bundling several lets the bot pick the newest and still fall back
    to an older one if a device prefers it.
    """
    if prefer:
        explicit = Path(prefer)
        if explicit.is_file():
            return explicit
        for directory in _third_party_dirs():
            candidate = directory / explicit
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"capture.scrcpy_jar points at a file that does not exist: {prefer}")

    jars = available_jars()
    if jars:
        return jars[-1]
    raise FileNotFoundError(
        f"Could not locate any {JAR_GLOB} in cr_bot/third_party/. Download one from "
        "https://github.com/Genymobile/scrcpy/releases (file `scrcpy-server-v<version>`) and "
        f"save it there as {JAR_NAME}."
    )


#: Known scrcpy server failures, mapped to what the operator should actually do.
_SERVER_FAILURES: Tuple[Tuple[str, str], ...] = (
    (
        "could not listen",
        "A scrcpy server is already running on this device and is holding the socket "
        "(`localabstract:scrcpy`). The bot now kills stale servers automatically; if this keeps "
        "happening another tool (Android Studio, scrcpy GUI, Vysor, a previous run) is using it. "
        "Close it, or run `adb shell pkill -f com.genymobile.scrcpy`.",
    ),
    (
        "classnotfound",
        "The server jar on the device is missing or truncated, so `com.genymobile.scrcpy.Server` "
        "cannot be loaded. Re-push it: `adb push cr_bot/third_party/scrcpy-server-v1.24.jar "
        "/data/local/tmp/` and check the reported size matches the local file.",
    ),
    (
        "permission denied",
        "The jar in /data/local/tmp is not executable/readable by the shell user. Delete it and "
        "let the bot push it again: `adb shell rm /data/local/tmp/scrcpy-server-v1.24.jar`.",
    ),
    (
        "mediaCodec",
        "The device encoder rejected the requested stream configuration. Lower "
        "`capture.bitrate` (e.g. 8000000) and/or `capture.max_fps` (e.g. 20), then retry.",
    ),
    (
        "encoder",
        "No usable H.264 encoder for this resolution/bitrate. Lower `capture.bitrate` / "
        "`capture.max_fps`, or set `capture.width` smaller (e.g. 540).",
    ),
    (
        "java.lang.illegalstateexception",
        "The scrcpy server could not initialise on this Android version. The bundled server is "
        "v1.24; on Android 13+ some devices need a newer server jar. Run with "
        "`--capture screencap` to play in the meantime.",
    ),
    (
        "nosuchmethoderror",
        "The scrcpy server is too old for this Android version. Run with `--capture screencap`, "
        "or supply a newer scrcpy-server jar.",
    ),
)


def diagnose_server_output(text: str) -> str:
    """Turn raw scrcpy server output into one actionable sentence (or an empty string)."""
    low = (text or "").lower()
    for needle, advice in _SERVER_FAILURES:
        if needle.lower() in low:
            return advice
    return ""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ======================================================================================
# Control channel
# ======================================================================================


class ControlChannel:
    """Serialized writer for scrcpy control packets (touch / keycode).

    The payload layout depends on the *server* version that is running, so this class carries
    the version of the jar that was actually pushed and asks
    :mod:`cr_bot.adb.scrcpy_protocol` for the right serialisation. Getting that wrong is
    invisible from the outside: the video keeps flowing, the taps just stop existing.
    """

    def __init__(
        self,
        sock: socket.socket,
        screen_w: int,
        screen_h: int,
        version: Tuple[int, int, int] = (1, 24, 0),
    ) -> None:
        self._sock = sock
        self._lock = threading.Lock()
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.version = tuple(version)  # type: ignore[assignment]
        self.packets_sent = 0
        self.last_error: Optional[str] = None
        #: total wall time spent writing touch packets, for the HUD/latency report
        self.touch_seconds = 0.0

    # -- geometry ------------------------------------------------------------------
    @property
    def screen_size(self) -> Tuple[int, int]:
        return (self.screen_w, self.screen_h)

    def set_screen_size(self, width: int, height: int) -> None:
        """Update the coordinate space to the size of the frames we actually receive.

        The server scales these values to the physical display, so telling it the *streamed*
        size is what lets the bot tap in frame coordinates on a downscaled stream.
        """
        self.screen_w, self.screen_h = int(width), int(height)

    # -- primitives ----------------------------------------------------------------
    def send(self, payload: bytes) -> bool:
        with self._lock:
            try:
                self._sock.sendall(payload)
                self.packets_sent += 1
                return True
            except OSError as exc:
                self.last_error = str(exc)
                logger.debug("Control channel send failed: %s", exc)
                return False

    def touch(
        self,
        x: int,
        y: int,
        action: int,
        pointer_id: int = proto.POINTER_ID_FINGER,
        pressure: Optional[float] = None,
    ) -> bool:
        return self.send(
            proto.build_touch(
                x,
                y,
                action,
                self.screen_w,
                self.screen_h,
                pointer_id=pointer_id,
                pressure=pressure,
                # a finger touch must leave both button fields at 0, or the server injects a
                # MOUSE drag and Clash Royale ignores it
                buttons=0,
                action_button=0,
                version=self.version,
            )
        )

    def touch_down(self, x: int, y: int, pointer_id: int = proto.POINTER_ID_FINGER) -> bool:
        return self.touch(x, y, proto.ACTION_DOWN, pointer_id, pressure=1.0)

    def touch_move(self, x: int, y: int, pointer_id: int = proto.POINTER_ID_FINGER) -> bool:
        return self.touch(x, y, proto.ACTION_MOVE, pointer_id, pressure=1.0)

    def touch_up(self, x: int, y: int, pointer_id: int = proto.POINTER_ID_FINGER) -> bool:
        return self.touch(x, y, proto.ACTION_UP, pointer_id, pressure=0.0)

    def tap(self, x: int, y: int, hold_ms: float = 70.0) -> bool:
        """A press with a real duration.

        A zero-length press is what ``input tap`` effectively produces, and Unity/Engine games
        routinely drop it. Any deliberate hold of 50-120 ms is far more reliable.
        """
        t0 = time.perf_counter()
        ok = self.touch_down(x, y)
        if hold_ms > 0:
            time.sleep(hold_ms / 1000.0)
        ok = self.touch_up(x, y) and ok
        self.touch_seconds += time.perf_counter() - t0
        return ok

    def drag(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        duration_ms: float = 180.0,
        steps: int = 10,
        settle_ms: float = 45.0,
    ) -> bool:
        """Press, glide to the target, hold briefly, release — a native drag-and-drop.

        This is the gesture Clash Royale is designed around: the card follows the finger and
        the deploy preview follows the pointer, so the game samples a real position at
        release instead of guessing from a select-then-tap sequence.
        """
        t0 = time.perf_counter()
        sx, sy = int(start[0]), int(start[1])
        ex, ey = int(end[0]), int(end[1])
        ok = self.touch_down(sx, sy)
        duration_ms = max(0.0, duration_ms)
        step_sleep = (duration_ms / 1000.0) / max(1, steps)
        for i in range(1, steps + 1):
            t = i / steps
            ok = self.touch_move(int(round(sx + (ex - sx) * t)), int(round(sy + (ey - sy) * t))) and ok
            if step_sleep > 0:
                time.sleep(step_sleep)
        if settle_ms > 0:
            # Hovering on the target before release is what the game samples for placement.
            time.sleep(settle_ms / 1000.0)
        ok = self.touch_up(ex, ey) and ok
        self.touch_seconds += time.perf_counter() - t0
        return ok

    def keycode(self, keycode: int, action: int = proto.ACTION_DOWN) -> bool:
        return self.send(proto.build_keycode(keycode, action, version=self.version))

    def back(self) -> bool:
        ok = self.keycode(4, proto.ACTION_DOWN)
        return self.keycode(4, proto.ACTION_UP) and ok

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


# ======================================================================================
# Native scrcpy session
# ======================================================================================


@dataclass
class SessionStats:
    frames: int = 0
    started_at: float = field(default_factory=time.monotonic)
    last_frame_at: float = 0.0
    reconnects: int = 0

    @property
    def fps(self) -> float:
        elapsed = max(1e-3, time.monotonic() - self.started_at)
        return self.frames / elapsed


class ScrcpySession:
    """One device: H.264 video stream + control socket, native protocol."""

    def __init__(
        self,
        adb: Adb,
        target_width: int = 720,
        bitrate: int = 12_000_000,
        max_fps: int = 30,
        lock_orientation: int = -1,
        stay_awake: bool = True,
        connect_timeout: float = 12.0,
        jar_path: Optional[str] = None,
        video_codec: str = "h264",
    ) -> None:
        self.adb = adb
        self.jar_path = jar_path
        self.video_codec = video_codec
        self.target_width = int(target_width)
        self.bitrate = int(bitrate)
        self.max_fps = int(max_fps)
        self.lock_orientation = int(lock_orientation)
        self.stay_awake = bool(stay_awake)
        self.connect_timeout = float(connect_timeout)

        self.serial: Optional[str] = None
        self.display: Optional[DisplayInfo] = None
        self.device_name = ""
        self.resolution: Tuple[int, int] = (0, 0)
        self._modern_stream_meta = False

        self.frames = proto.FrameBuffer()
        self.stats = SessionStats()
        self.control: Optional[ControlChannel] = None

        self._video_sock: Optional[socket.socket] = None
        self._control_sock: Optional[socket.socket] = None
        self._server_proc: Optional[subprocess.Popen] = None
        self._forward_port: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None
        self.jar_version: Tuple[int, int, int] = (1, 24, 0)
        self._server_log: "collections.deque[str]" = collections.deque(maxlen=250)
        self._log_thread: Optional[threading.Thread] = None
        self._stale_servers_killed = 0

    # -- lifecycle -----------------------------------------------------------------
    def _attempts(self) -> List[Tuple[str, Dict[str, str]]]:
        """Server configurations tried in order before giving up.

        Encoders differ wildly between vendors, and a 720p30 H.264 stream at 12 Mbit/s is a hard
        ask for some SoCs while a smaller/slower one is fine. Rather than guess, try the default
        and then progressively humbler settings - each is a fresh server process.
        """
        attempts: List[Tuple[str, Dict[str, str]]] = [("default", {})]
        if proto.is_modern_version(self.jar_version):
            attempts.append(("software-friendly H.264", {"video_encoder": "OMX.google.h264.encoder"}))
            attempts.append(("hardware H.264 (avc)", {"video_encoder": "OMX.qcom.video.encoder.avc"}))
            attempts.append(("lower bitrate + no fps cap",
                             {"video_bit_rate": min(self.bitrate, 4_000_000), "max_fps": 0}))
            attempts.append(("H.265", {"video_codec": "h265", "video_bit_rate": min(self.bitrate, 6_000_000)}))
            # A clean socket id, last. On Android 14+ a scrcpy server left over from *anything*
            # (a crash, another tool, a previous run of this bot) keeps listening on
            # ``localabstract:scrcpy``. The new server then fails to create its socket, and the
            # symptom is precisely the one seen on the SM-A528B: the server logs its device line
            # and then nothing, and no frame ever arrives. A distinct ``scid`` moves the server
            # to its own abstract socket name, which sidesteps a stale or half-dead predecessor
            # that ``adb shell`` housekeeping cannot see.
            attempts.append(("fresh socket id", {"scid": "2b3a7f"}))
        else:
            attempts.append(("no fps cap", {"max_fps": 0}))
            attempts.append(("lower bitrate", {"bit_rate": min(self.bitrate, 4_000_000)}))
            attempts.append(("google encoder", {"encoder_name": "OMX.google.h264.encoder"}))
        return attempts

    def start(self) -> "ScrcpySession":
        self.serial = self.adb.connect(self.adb.serial)
        self.display = self.adb.display_info()
        self._kill_stale_servers()

        failures: List[Tuple[str, str]] = []
        seen: set = set()
        for label, overrides in self._attempts():
            try:
                self._attempt_stream(overrides, label)
                return self
            except Exception as exc:  # noqa: BLE001 - socket errors must not escape the ladder
                detail = exc if isinstance(exc, ADBError) else ADBError(
                    f"{type(exc).__name__}: {exc}"
                )
                text = str(detail)
                failures.append((label, text))
                # If a different encoder/bitrate produces the *same* failure, the problem is not
                # the configuration (a missing jar, a permission error, a broken display service).
                # Stop instead of relaunching the server four times for nothing.
                key = self._failure_signature(text)
                if key in seen:
                    logger.warning(
                        "scrcpy attempt (%s) failed identically; the configuration is not the "
                        "problem, stopping here", label)
                    break
                seen.add(key)
                logger.warning("scrcpy attempt (%s) failed (%s); trying the next configuration",
                               label, type(exc).__name__)
                self._teardown_connection()
                time.sleep(0.3)
                self._kill_stale_servers()

        if len(failures) == 1:
            raise ADBError(failures[0][1])
        unique: List[str] = []
        for label, text in failures:
            if text not in unique:
                unique.append(text)
        raise ADBError(
            f"scrcpy capture could not be started ({len(failures)} configurations tried, "
            f"tried: {', '.join(label for label, _ in failures)}).\n\n" + "\n\n".join(unique)
        )

    @staticmethod
    def _failure_signature(text: str) -> str:
        """What makes two attempts 'the same failure': the diagnosis and the server's own words."""
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        interesting = [
            ln for ln in lines
            if not ln.startswith("Killed ") and "trying the next" not in ln
        ]
        return " | ".join(interesting[:4])

    #: handshake budget for the retries, in seconds. The first attempt uses ``connect_timeout``
    #: because a healthy server answers in well under a second; later attempts get more room,
    #: because a cold hardware encoder that has just been torn down really can take a while to
    #: emit its first frame, and giving up on it was throwing away a working configuration.
    RETRY_HANDSHAKE_TIMEOUT = 25.0

    def _attempt_stream(self, overrides: Dict[str, str], label: str) -> None:
        """One full launch + handshake + first-frame cycle."""
        self._server_log.clear()
        self._start_error = None
        self._ready.clear()
        self.stats.frames = 0
        self._modern_stream_meta = overrides.get("send_stream_meta", "false") == "true"

        self._server_proc = self._launch_server(overrides)
        self._open_sockets()
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._read_loop, name=f"scrcpy-{self.serial}", daemon=True
        )
        self._thread.start()

        timeout = self.connect_timeout if label == "default" else max(
            self.connect_timeout, self.RETRY_HANDSHAKE_TIMEOUT
        )
        if not self._ready.wait(timeout):
            raise ADBError(
                "scrcpy server started but no video frame arrived within "
                f"{timeout:.0f}s.\n" + self.failure_report()
            )
        if self._start_error is not None:
            raise ADBError(str(self._start_error))
        logger.info(
            "scrcpy stream up (%s): %s %dx%d on %s",
            label,
            self.device_name or "device",
            self.resolution[0],
            self.resolution[1],
            self.serial,
        )

    def stop(self) -> None:
        self._stopping.set()
        for sock in (self._video_sock, self._control_sock):
            try:
                if sock is not None:
                    sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        if self.control is not None:
            self.control.close()
        self._teardown_server()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "ScrcpySession":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- server housekeeping -------------------------------------------------------
    def _teardown_connection(self) -> None:
        """Drop sockets, threads and the server process, keeping the session reusable."""
        self._stopping.set()
        for sock in (self._video_sock, self._control_sock):
            try:
                if sock is not None:
                    sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        self._video_sock = None
        self._control_sock = None
        if self.control is not None:
            self.control.close()
            self.control = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._teardown_server()

    def _teardown_server(self) -> None:
        if self._server_proc is not None:
            try:
                self._server_proc.terminate()
                self._server_proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                try:
                    self._server_proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._server_proc = None
        if self._forward_port is not None:
            try:
                self.adb.run(["forward", "--remove", f"tcp:{self._forward_port}"],
                             include_serial=True, timeout=6.0)
            except Exception:  # noqa: BLE001
                pass
            self._forward_port = None

    def _kill_stale_servers(self) -> int:
        """Kill scrcpy servers left behind by an earlier run (they hold ``localabstract:scrcpy``).

        Without this a second launch dies with "Could not listen on local socket" and the client
        only ever sees "no handshake byte" - an error that looks like a version mismatch but is
        really just a leftover process.
        """
        killed = 0
        text = self._device_shell("ps -A -o PID,NAME")
        for line in (text or "").splitlines():
            if "scrcpy" not in line.lower():
                continue
            # `ps` column order differs between toybox and toolbox builds, so identify the pid by
            # shape instead of position (a header line yields no digits and is skipped)
            pid = next((token for token in line.split() if token.isdigit()), None)
            if pid is None:
                continue
            self._device_shell(f"kill -9 {pid}")
            killed += 1
            logger.info("  killed stale scrcpy server pid %s", pid)
        # also drop forwards pointing at the abstract socket from a previous session
        try:
            listing = self.adb.run(["forward", "--list"], include_serial=True, timeout=6.0)
            forwarded = (listing.stdout or b"").decode("utf-8", "replace")
            for line in forwarded.splitlines():
                if "localabstract:scrcpy" not in line:
                    continue
                token = line.split()[1] if len(line.split()) > 1 else ""
                if token.startswith("tcp:"):
                    self.adb.run(["forward", "--remove", token], include_serial=True, timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        self._stale_servers_killed = killed
        return killed

    def _device_shell(self, command: str, timeout: float = 8.0) -> str:
        """Run a device shell command, returning decoded text and never raising."""
        try:
            result = self.adb.run(["shell", command], include_serial=True, timeout=timeout)
            return ((result.stdout or b"") + (result.stderr or b"")).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return ""

    def server_log(self, tail: int = 40) -> str:
        """Recent scrcpy server output - the first thing to read when capture fails."""
        lines = list(self._server_log)[-tail:]
        return "\n".join(lines)

    def failure_report(self) -> str:
        """A multi-line, actionable description of why the stream did not come up."""
        log = self.server_log()
        report: List[str] = []
        advice = diagnose_server_output(log)
        if advice:
            report.append(advice)
        if self._stale_servers_killed:
            report.append(f"Killed {self._stale_servers_killed} stale scrcpy server(s) from an earlier run.")
        if log:
            report.append("scrcpy server output:\n" + log)
        else:
            report.append(
                "The scrcpy server produced no output at all, which usually means the process never "
                "started. Check `adb shell ps -A | findstr scrcpy` and that the jar was pushed to "
                "/data/local/tmp."
            )
        return "\n".join(report)

    # -- server bootstrap ----------------------------------------------------------
    def _compute_max_size(self) -> int:
        """Cap the *long* side so the streamed width lands near ``target_width``."""
        if self.display is None or self.target_width <= 0:
            return 0
        short_side = min(self.display.width, self.display.height)
        long_side = max(self.display.width, self.display.height)
        if short_side <= 0:
            return 0
        scale = self.target_width / float(short_side)
        if scale >= 1.0:
            return 0  # never upscale; keep native resolution
        return int(round(long_side * scale))

    def _launch_server(self, overrides: Optional[Dict[str, str]] = None) -> subprocess.Popen:
        jar = find_scrcpy_jar(self.jar_path)
        remote_jar = f"/data/local/tmp/{jar.name}"
        self.jar_version = proto.parse_server_version(jar.name)
        self.adb.push(jar, "/data/local/tmp/")
        self._verify_pushed_jar(jar, remote_jar)

        options = proto.build_options(
            self.jar_version,
            bitrate=self.bitrate,
            max_size=self._compute_max_size(),
            max_fps=self.max_fps,
            lock_orientation=self.lock_orientation,
            stay_awake=self.stay_awake,
            video_codec=self.video_codec,
        )
        options["_version"] = ".".join(str(p) for p in self.jar_version[:3]).rstrip(".0") or "1.24"
        if overrides:
            options.update(overrides)
        version_arg = options.pop("_version")
        opt_str = " ".join(f"{k}={v}" for k, v in options.items())
        command = (
            f"CLASSPATH={remote_jar} app_process / com.genymobile.scrcpy.Server "
            f"{version_arg} {opt_str}"
        )
        logger.debug("scrcpy server %s (%s)", version_arg, jar.name)
        logger.debug("Launching scrcpy server: %s", command)
        proc = subprocess.Popen(
            self.adb.base_cmd() + ["shell", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            errors="replace",
        )
        self._drain_server_output(proc)

        # Give the server a moment to fail fast (bad jar, busy socket) before connecting.
        deadline = time.monotonic() + min(3.0, self.connect_timeout)
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                time.sleep(0.05)  # let the drain thread finish
                raise ADBError("scrcpy server exited immediately.\n" + self.failure_report())
            time.sleep(0.05)
        return proc

    def _verify_pushed_jar(self, local: Path, remote: str) -> None:
        """A truncated push is otherwise indistinguishable from a protocol mismatch."""
        try:
            local_size = local.stat().st_size
        except OSError:
            return
        text = self._device_shell(f"stat -c %s {remote}", timeout=6.0)
        try:
            remote_size = int(text.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return  # `stat` is unavailable on some devices; skip the check
        if remote_size <= 0:
            return  # unsupported flag / unreadable file: nothing trustworthy to compare
        if remote_size != local_size:
            raise ADBError(
                f"the scrcpy server jar on the device is {remote_size} bytes but the local file is "
                f"{local_size}. The push was incomplete:\n"
                f"  adb push {local} /data/local/tmp/\n"
                f"  adb shell rm {remote}"
            )

    def _drain_server_output(self, proc: subprocess.Popen) -> None:
        """Keep reading the server's stdout so a crash message is never lost."""

        def _pump() -> None:
            stream = proc.stdout
            if stream is None:
                return
            try:
                for line in stream:
                    line = (line or "").rstrip()
                    if line:
                        self._server_log.append(line)
                        logger.debug("scrcpy> %s", line)
            except Exception:  # noqa: BLE001 - stream closed while stopping
                pass

        self._log_thread = threading.Thread(target=_pump, name="scrcpy-log", daemon=True)
        self._log_thread.start()

    def _open_sockets(self) -> None:
        port = _free_port()
        self._forward_port = port
        self.adb.forward(port, f"localabstract:{proto.SOCKET_NAME}")

        deadline = time.monotonic() + self.connect_timeout
        last_exc: Optional[BaseException] = None
        while time.monotonic() < deadline:
            try:
                video = socket.create_connection(("127.0.0.1", port), timeout=4.0)
                break
            except OSError as exc:
                last_exc = exc
                time.sleep(0.1)
        else:
            raise ADBError(f"Could not connect to the scrcpy video socket: {last_exc}")

        video.settimeout(6.0)
        if not proto.read_dummy_byte(video):
            video.close()
            raise ADBError(
                "scrcpy did not send its handshake byte.\n" + self.failure_report() + "\n"
                "Meanwhile you can play with `--capture screencap` (slower but always available)."
            )

        control_sock = socket.create_connection(("127.0.0.1", port), timeout=4.0)
        self.device_name = proto.read_device_name(video)
        width = height = 0
        if proto.is_modern_version(self.jar_version):
            # v2+ declared no size here; with send_stream_meta=false the raw stream starts now and
            # the real size arrives with the first decoded frame.
            if self._modern_stream_meta:
                width, height = proto.read_session_meta(video)
        else:
            width, height = proto.read_legacy_video_size(video)
        self.resolution = (width, height)
        if width < 0 or height < 0:
            video.close()
            control_sock.close()
            raise ADBError(f"scrcpy reported an invalid video size {self.resolution}")

        video.settimeout(1.0)
        control_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._video_sock = video
        self._control_sock = control_sock
        self.control = ControlChannel(control_sock, *self.resolution, version=self.jar_version)

        if width and height and width > height:
            logger.warning(
                "Streamed frame is landscape (%dx%d). Clash Royale must run in portrait; "
                "rotate the device or lock rotation before starting the bot.",
                width,
                height,
            )

    # -- decode loop ---------------------------------------------------------------
    def _read_loop(self) -> None:
        decoder = None
        try:
            decoder = proto.H264FrameDecoder()
        except Exception as exc:  # noqa: BLE001
            self._start_error = exc
            self._ready.set()
            return

        while not self._stopping.is_set():
            try:
                raw = self._video_sock.recv(1 << 16) if self._video_sock else b""
            except socket.timeout:
                continue
            except OSError as exc:
                if self._stopping.is_set():
                    break
                logger.warning("scrcpy video socket error: %s", exc)
                break

            if not raw:
                if self._stopping.is_set():
                    break
                if self.stats.frames == 0:
                    # The server accepted the sockets and then died - a codec or capture failure,
                    # not a network hiccup. Record it so start() fails (and the caller can retry
                    # with different options or fall back) instead of reporting a healthy stream.
                    self._start_error = ADBError(
                        "the scrcpy server closed the video stream before sending a single frame.\n"
                        + self.failure_report()
                    )
                else:
                    logger.warning("scrcpy video stream ended after %d frames", self.stats.frames)
                break

            for frame in decoder.feed(raw):
                now = time.monotonic()
                self.frames.push(frame, now)
                self.stats.frames += 1
                self.stats.last_frame_at = now
                if not self._ready.is_set():
                    self.resolution = (int(frame.shape[1]), int(frame.shape[0]))
                    if self.control is not None:
                        self.control.set_screen_size(*self.resolution)
                    self._ready.set()

        self._ready.set()

    # -- convenience ---------------------------------------------------------------
    def latest_frame(self, min_seq: int = 0, timeout: float = 0.5) -> Optional[np.ndarray]:
        return self.frames.get(min_seq=min_seq, timeout=timeout)

    @property
    def frame_seq(self) -> int:
        return self.frames.seq

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ======================================================================================
# Legacy screencap session (fallback)
# ======================================================================================


class ScreencapSession:
    """Polling capture through ``adb exec-out screencap``.

    Slower and staler than scrcpy, but it needs nothing but adb, so it is a useful
    fallback and a way to prove the ADB link itself is healthy.
    """

    def __init__(self, adb: Adb, target_width: int = 720, prefer_raw: bool = True) -> None:
        self.adb = adb
        self.target_width = int(target_width)
        self.prefer_raw = bool(prefer_raw)
        self.serial: Optional[str] = None
        self.display: Optional[DisplayInfo] = None
        self.device_name = "screencap"
        self.resolution: Tuple[int, int] = (0, 0)
        self._modern_stream_meta = False
        self.frames = proto.FrameBuffer()
        self.stats = SessionStats()
        self.control: Optional[ControlChannel] = None
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ScreencapSession":
        self.serial = self.adb.connect(self.adb.serial)
        self.display = self.adb.display_info()
        frame = self._grab()
        if frame is None:
            raise ADBError("screencap returned no usable frame")
        self.resolution = (int(frame.shape[1]), int(frame.shape[0]))
        self._stopping.clear()
        self._thread = threading.Thread(target=self._loop, name="screencap", daemon=True)
        self._thread.start()
        logger.info("screencap capture up: %dx%d", *self.resolution)
        return self

    def _loop(self) -> None:
        while not self._stopping.is_set():
            frame = self._grab()
            if frame is not None:
                self.frames.push(frame)
                self.stats.frames += 1
                self.stats.last_frame_at = time.monotonic()
            time.sleep(0.01)

    def _grab(self) -> Optional[np.ndarray]:
        import cv2

        if self.prefer_raw:
            frame = self._grab_raw()
            if frame is not None:
                return frame
        try:
            raw = self.adb.exec_out(["screencap", "-p"], timeout=8.0)
        except ADBError:
            return None
        if not raw or len(raw) < 100:
            return None
        buf = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _grab_raw(self) -> Optional[np.ndarray]:
        import cv2

        try:
            raw = self.adb.exec_out(["screencap"], timeout=8.0)
        except ADBError:
            return None
        if not raw or len(raw) < 16:
            return None
        w = int.from_bytes(raw[0:4], "little")
        h = int.from_bytes(raw[4:8], "little")
        fmt = int.from_bytes(raw[8:12], "little")
        if fmt != 1 or not (0 < w <= 5000) or not (0 < h <= 5000):
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

    def latest_frame(self, min_seq: int = 0, timeout: float = 0.5) -> Optional[np.ndarray]:
        return self.frames.get(min_seq=min_seq, timeout=timeout)

    @property
    def frame_seq(self) -> int:
        return self.frames.seq

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ======================================================================================
# MultiClient
# ======================================================================================


class MultiClient:
    """Capture client for one or several ADB devices at once.

    Usage::

        mc = MultiClient(target_width=720)
        mc.start()
        frame = mc.latest_frame()
        mc.control.drag((210, 1135), (300, 700))

    With several devices::

        mc = MultiClient(serials=["R58M1234", "10.0.0.5:5555"])
        mc.start()
        for serial, session in mc.sessions.items():
            ...
    """

    def __init__(
        self,
        adb_path: str = "adb",
        serial: Optional[str] = None,
        serials: Optional[List[str]] = None,
        target_width: int = 720,
        bitrate: int = 12_000_000,
        max_fps: int = 30,
        backend: str = "auto",
        connect_timeout: float = 12.0,
        stay_awake: bool = True,
        scrcpy_jar: Optional[str] = None,
    ) -> None:
        self.adb_path = adb_path
        self.scrcpy_jar = scrcpy_jar
        self.requested = serials or ([serial] if serial else [None])
        self.target_width = int(target_width)
        self.bitrate = int(bitrate)
        self.max_fps = int(max_fps)
        self.backend = (backend or "auto").lower()
        self.connect_timeout = float(connect_timeout)
        self.stay_awake = bool(stay_awake)

        self.sessions: Dict[str, object] = {}
        self.started = False
        #: backend actually in use after start() (``scrcpy`` / ``scrcpy_py`` / ``screencap``)
        self.backend_used: str = ""
        #: human readable reason when the preferred backend had to be downgraded
        self.fallback_reason: str = ""

    # -- construction helpers ------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "MultiClient":
        return cls(
            adb_path=getattr(cfg, "adb_path", "adb"),
            serial=getattr(cfg, "adb_serial", None),
            target_width=getattr(cfg, "capture_width", 720),
            bitrate=getattr(cfg, "capture_bitrate", 12_000_000),
            max_fps=getattr(cfg, "capture_max_fps", 30),
            backend=getattr(cfg, "capture_backend", "auto"),
            connect_timeout=getattr(cfg, "capture_connect_timeout", 12.0),
        )

    def _new_session(self, adb: Adb):
        backend = self.backend
        if backend == "scrcpy_py":
            lib = _try_scrcpy_py_session(adb, self.target_width, self.bitrate, self.max_fps)
            if lib is not None:
                return lib
            logger.warning("scrcpy_py backend unavailable; falling back to native scrcpy protocol.")
            backend = "scrcpy"

        if backend == "screencap":
            return ScreencapSession(adb, target_width=self.target_width)

        try:
            return ScrcpySession(
                adb,
                target_width=self.target_width,
                bitrate=self.bitrate,
                max_fps=self.max_fps,
                connect_timeout=self.connect_timeout,
                stay_awake=self.stay_awake,
                jar_path=self.scrcpy_jar,
            )
        except Exception as exc:  # noqa: BLE001
            if backend == "scrcpy":
                raise
            logger.warning("Native scrcpy capture failed (%s); falling back to screencap.", exc)
            return ScreencapSession(adb, target_width=self.target_width)

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> "MultiClient":
        started: List[object] = []
        try:
            for serial in self.requested:
                adb = Adb(self.adb_path, serial)
                session = self._start_one(adb)
                self.sessions[str(session.serial)] = session
                started.append(session)
        except BaseException:
            for session in started:
                try:
                    session.stop()
                except Exception:  # noqa: BLE001
                    pass
            raise
        self.started = True
        return self

    def _start_one(self, adb: Adb):
        """Start a session, downgrading to ``screencap`` when scrcpy cannot come up.

        A broken scrcpy server used to abort the whole run. Capture is not the point of the bot -
        playing is - so with ``backend: auto`` a failed scrcpy stream now degrades to screencap
        polling (slower, but verified to work on this device) instead of raising.
        """
        session = self._new_session(adb)
        try:
            session.start()
        except ADBError as exc:
            try:
                session.stop()
            except Exception:  # noqa: BLE001
                pass
            if self.backend not in ("auto", "scrcpy") or isinstance(session, ScreencapSession):
                raise
            self.fallback_reason = str(exc)
            logger.error(
                "Live scrcpy capture is not usable on this device:\n%s\n"
                "Falling back to `screencap` polling (works everywhere, ~5-8 fps). "
                "Fix the scrcpy issue to get 30 fps back; the bot will still play meanwhile.",
                exc,
            )
            fallback = ScreencapSession(adb, target_width=self.target_width)
            fallback.start()
            self.backend_used = "screencap"
            return fallback
        self.backend_used = "scrcpy_py" if type(session).__name__ == "ScrcpyPySession" else "scrcpy"
        return session

    def stop(self) -> None:
        for session in self.sessions.values():
            try:
                session.stop()
            except Exception:  # noqa: BLE001
                pass
        self.sessions.clear()
        self.started = False

    def __enter__(self) -> "MultiClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- accessors -----------------------------------------------------------------
    def session(self, serial: Optional[str] = None):
        if not self.sessions:
            raise RuntimeError("MultiClient.start() has not been called")
        if serial is None:
            return next(iter(self.sessions.values()))
        if serial in self.sessions:
            return self.sessions[serial]
        raise KeyError(f"Unknown device serial {serial!r}")

    def latest_frame(self, serial: Optional[str] = None, min_seq: int = 0, timeout: float = 0.5):
        return self.session(serial).latest_frame(min_seq=min_seq, timeout=timeout)

    def frame_seq(self, serial: Optional[str] = None) -> int:
        return int(self.session(serial).frame_seq)

    @property
    def resolution(self) -> Tuple[int, int]:
        return tuple(self.session().resolution)  # type: ignore[return-value]

    @property
    def control(self) -> Optional[ControlChannel]:
        return self.session().control

    @property
    def fps(self) -> float:
        return self.session().stats.fps

    @property
    def frame_age(self) -> float:
        return self.session().frames.age

    def status(self) -> str:
        return " | ".join(
            f"{serial}: {s.resolution[0]}x{s.resolution[1]} @ {s.stats.fps:.1f}fps"
            for serial, s in self.sessions.items()
        )


# ======================================================================================
# Optional py-scrcpy-client backend
# ======================================================================================

_TOUCH_FIX_APPLIED = False


def _patch_scrcpy_py_touch() -> bool:
    """Repair ``scrcpy.Client.control.touch`` to emit the correct 27-byte payload.

    py-scrcpy-client 0.4.x packs ``>BqiiHHHii`` (31 bytes) against a 1.24 server that
    expects 27, which corrupts the control stream. We swap in the correct serializer so
    the library can be used for capture *and* input safely.
    """
    global _TOUCH_FIX_APPLIED
    if _TOUCH_FIX_APPLIED:
        return True
    try:
        from scrcpy import const as _const  # type: ignore
        from scrcpy import control as _control  # type: ignore
    except Exception:  # noqa: BLE001
        return False

    def touch(self, x, y, action=_const.ACTION_DOWN, touch_id=0x1234567887654321):
        payload = proto.build_touch(
            int(x),
            int(y),
            int(action),
            int(self.parent.resolution[0]),
            int(self.parent.resolution[1]),
            pointer_id=int(touch_id) & ((1 << 63) - 1),
        )
        if self.parent.control_socket is not None:
            with self.parent.control_socket_lock:
                self.parent.control_socket.send(payload)
        return payload

    _control.ControlSender.touch = touch  # type: ignore[method-assign]
    _TOUCH_FIX_APPLIED = True
    logger.info("Patched py-scrcpy-client touch packets to the correct 27-byte layout.")
    return True


class ScrcpyPySession:
    """Adapter over the ``scrcpy`` (py-scrcpy-client) package."""

    def __init__(self, adb: Adb, target_width: int, bitrate: int, max_fps: int) -> None:
        import scrcpy as scrcpy_pkg  # type: ignore

        _patch_scrcpy_py_touch()
        self._pkg = scrcpy_pkg
        self.adb = adb
        self.client = scrcpy_pkg.Client(
            device=adb.serial,
            max_width=int(target_width),
            bitrate=int(bitrate),
            max_fps=int(max_fps),
        )
        self.frames = proto.FrameBuffer()
        self.stats = SessionStats()
        self.resolution: Tuple[int, int] = (0, 0)
        self._modern_stream_meta = False
        self.serial: Optional[str] = adb.serial
        self.device_name = "scrcpy_py"
        self.display: Optional[DisplayInfo] = None
        self.control: Optional[ControlChannel] = None
        #: py-scrcpy-client 0.4.x pins scrcpy-server 1.24
        self.jar_version: Tuple[int, int, int] = (1, 24, 0)
        self._stop_flag = threading.Event()

    def start(self) -> "ScrcpyPySession":
        self.serial = self.adb.connect(self.adb.serial)
        self.client.listeners.setdefault("init", [])
        self.client.listeners.setdefault("frame", [])

        def on_frame(frame):
            if frame is not None:
                self.frames.push(frame)
                self.stats.frames += 1
                self.stats.last_frame_at = time.monotonic()
                self.resolution = (int(frame.shape[1]), int(frame.shape[0]))

        self.client.listeners["frame"].append(on_frame)
        self.client.start(threaded=True)

        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and self.frames.frame is None:
            time.sleep(0.05)
        if self.frames.frame is None:
            raise ADBError("py-scrcpy-client produced no frames")

        # Reuse the library's proven control socket, but through our own serializer.
        self.control = ControlChannel(
            self.client.control_socket, *self.resolution,  # type: ignore[arg-type]
            version=self.jar_version,
        )
        self.control.set_screen_size(*self.resolution)
        return self

    def latest_frame(self, min_seq: int = 0, timeout: float = 0.5):
        return self.frames.get(min_seq=min_seq, timeout=timeout)

    @property
    def frame_seq(self) -> int:
        return self.frames.seq

    @property
    def is_alive(self) -> bool:
        return bool(getattr(self.client, "alive", False))

    def stop(self) -> None:
        self._stop_flag.set()
        try:
            self.client.stop()
        except Exception:  # noqa: BLE001
            pass


def _try_scrcpy_py_session(adb: Adb, target_width: int, bitrate: int, max_fps: int):
    try:
        import scrcpy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        logger.info("scrcpy python package not importable (%s).", exc)
        return None
    try:
        return ScrcpyPySession(adb, target_width, bitrate, max_fps)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build a py-scrcpy-client session: %s", exc)
        return None
