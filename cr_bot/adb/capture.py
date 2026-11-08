"""Screen capture + input channel for one or many Android devices ("MultiClient").

Why a live stream and not ``screencap``
---------------------------------------
``adb exec-out screencap`` costs a 150-400 ms round trip and ships ~10 MB of raw pixels per
frame on a 1080p phone. A bot that reasons about a third of a second old frame is always a
beat behind: it reacts to troops that have already moved and mis-reads elixir. Every real
backend below streams H.264 instead, so frames are 2-30 ms old.

Backends, best first (``capture.backend: auto``)
------------------------------------------------
``scrcpy``        The **native scrcpy protocol** implemented in
                  :mod:`cr_bot.adb.scrcpy_multiclient`, driving the server jar bundled in
                  ``cr_bot/third_party``. This is the best option for this bot by a wide
                  margin, and the only one that also gives low-latency *input*: the same
                  session exposes a control socket, so a card is deployed by writing a few
                  bytes instead of spawning an ``input`` JVM on the device (~100-250 ms per
                  call). 720p60 is routine. Nothing to ``pip install``.
``screenrecord``  Android's own binary. Same H.264 stream without a server jar, so it works
                  where a scrcpy server will not start, but it has **no control channel**:
                  input falls back to ``adb shell input``.
``scrcpy_py``     the ``scrcpy`` PyPI package (py-scrcpy-client), if it happens to be
                  installed. Off by default in the ladder: its own pins (``av<10``,
                  ``adbutils<0.15``) no longer install on current Python, and the native
                  implementation is strictly better here.
``screencap``     ``adb exec-out screencap`` polling. Always works, ~1-8 fps. Proves the ADB
                  link is healthy; never a good way to play.

Input does not have to live in the capture backend: :class:`~cr_bot.adb.touch.TouchInjector`
uses ``session.control`` when it is available and escalates to ``adb shell input`` when it is
not, so every backend above can play.
"""

from __future__ import annotations

import collections
import logging
import os
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from cr_bot.adb.adb_utils import Adb, ADBError, DisplayInfo
from cr_bot.adb.scrcpy_multiclient import (
    ControlChannel,
    ScreencapSession,
    ScrcpyPySession,
    ScrcpySession,
    SessionStats,
    find_scrcpy_jar,
    available_jars,
    diagnose_server_output,
)
from cr_bot.adb.scrcpy_protocol import FrameBuffer, H264FrameDecoder

logger = logging.getLogger(__name__)

__all__ = [
    "MultiClient",
    "ScreenrecordSession",
    "ScreencapSession",
    "ScrcpySession",
    "FrameBuffer",
    "H264FrameDecoder",
    "SessionStats",
    "scrcpy_py_available",
    "available_scrcpy_jars",
]


def available_scrcpy_jars() -> List[str]:
    """Names of the bundled scrcpy server jars, newest last (empty if none)."""
    try:
        return [p.name for p in available_jars()]
    except Exception:  # noqa: BLE001 - diagnostics only
        return []


def scrcpy_py_available() -> Optional[str]:
    """Return the installed ``scrcpy`` package version, or ``None`` if it is absent."""
    try:
        import scrcpy  # noqa: F401, PLC0415

        return str(getattr(scrcpy, "__version__", "installed"))
    except Exception:  # noqa: BLE001 - absence is the normal case
        return None


# ==============================================================================
# screenrecord session (stock Android, no server jar)
# ==============================================================================


class ScreenrecordSession:
    """Live H.264 capture through Android's stock ``screenrecord`` binary.

    ``screenrecord`` is part of AOSP since 4.4, so it is *already on the phone*: no server
    jar to push, no socket handshake, none of the version-dependent failure modes that can
    make a pushed scrcpy server go silent. It streams H.264 at display rate, so it is
    20-30x faster than ``screencap`` (which ships ~10 MB of raw RGBA per frame).

    Limitations, all handled here:

    * it self-terminates after ``--time-limit`` seconds (180 is Android's maximum), so the
      reader loop simply restarts it, dropping a handful of frames every three minutes;
    * **no control socket**, so :attr:`control` stays ``None`` and input uses
      ``adb shell input`` (the bot verifies every deploy, so it does not depend on scrcpy's
      injection to know whether a play landed);
    * ``--size`` must be 8-aligned and some encoders reject non-standard sizes outright, so
      a rejected size is retried once at native resolution before the session gives up.
    """

    name = "screenrecord"

    def __init__(
        self,
        adb: Adb,
        target_width: int = 720,
        bitrate: int = 8_000_000,
        max_fps: int = 30,
        time_limit: int = 180,
        first_frame_timeout: float = 12.0,
    ) -> None:
        self.adb = adb
        self.target_width = int(target_width)
        self.bitrate = int(bitrate)
        self.max_fps = int(max_fps)
        self.time_limit = max(10, min(180, int(time_limit)))
        self.first_frame_timeout = float(first_frame_timeout)

        self.serial: Optional[str] = None
        self.display: Optional[DisplayInfo] = None
        self.device_name = "screenrecord"
        self.resolution: Tuple[int, int] = (0, 0)
        #: always None: ``screenrecord`` exposes no control channel
        self.control: Optional[ControlChannel] = None
        self.jar_version: Tuple[int, int, int] = (0, 0, 0)

        self.frames = FrameBuffer()
        self.stats = SessionStats()

        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._start_error: Optional[BaseException] = None
        self._stderr_tail: "collections.deque[str]" = collections.deque(maxlen=40)
        self._spawned_with_size: Optional[str] = None
        self._native_fallback_done = False
        self.restarts = 0

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> "ScreenrecordSession":
        self.serial = self.adb.connect(self.adb.serial)
        try:
            self.display = self.adb.display_info()
        except Exception:  # noqa: BLE001 - optional, only used to pick a capture size
            self.display = None

        self._stopping.clear()
        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._loop, name="screenrecord", daemon=True)
        self._thread.start()

        if not self._ready.wait(self.first_frame_timeout):
            self.stop()
            raise ADBError(
                f"`screenrecord` produced no video frame within {self.first_frame_timeout:.0f}s"
                + self._error_suffix()
            )
        if self._start_error is not None:
            self.stop()
            raise ADBError(str(self._start_error) + self._error_suffix())
        logger.info("screenrecord capture up: %dx%d (no server jar needed)", *self.resolution)
        return self

    def _error_suffix(self) -> str:
        tail = "\n".join(line for line in self._stderr_tail if line.strip())
        return f"\nscreenrecord said: {tail}" if tail else ""

    def _capture_size(self) -> Optional[str]:
        """Pick a ``--size`` that keeps the short edge near ``target_width``."""
        display = self.display
        if display is None:
            try:
                display = self.adb.display_info()
                self.display = display
            except Exception:  # noqa: BLE001 - size is optional, native res is fine
                display = None
        w = h = 0
        if display is not None:
            w, h = int(display.width), int(display.height)
        if w <= 0 or h <= 0:
            return None
        scale = float(self.target_width) / float(min(w, h))
        tw, th = int(round(w * scale)), int(round(h * scale))
        tw -= tw % 8
        th -= th % 8
        if tw < 64 or th < 64:
            return None
        return f"{tw}x{th}"

    def _command(self, size: Optional[str]) -> List[str]:
        cmd = self.adb.base_cmd() + [
            "exec-out",
            "screenrecord",
            "--output-format=h264",
            f"--bit-rate={self.bitrate}",
            f"--time-limit={self.time_limit}",
        ]
        if size:
            cmd.append(f"--size={size}")
        # "-" makes screenrecord write the elementary stream to stdout instead of /sdcard
        cmd.append("-")
        return cmd

    def _spawn(self) -> subprocess.Popen:
        size = None if self._native_fallback_done else self._capture_size()
        self._spawned_with_size = size
        cmd = self._command(size)
        logger.debug("screenrecord command: %s", " ".join(cmd))
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stderr is not None
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    self._stderr_tail.append(line)
                    logger.debug("[screenrecord] %s", line)
        except Exception:  # noqa: BLE001 - best effort diagnostics only
            return

    def _loop(self) -> None:
        try:
            decoder = H264FrameDecoder()
        except Exception as exc:  # noqa: BLE001
            self._start_error = exc
            self._ready.set()
            return

        while not self._stopping.is_set():
            try:
                proc = self._spawn()
            except Exception as exc:  # noqa: BLE001
                self._start_error = ADBError(f"could not run screenrecord: {exc}")
                self._ready.set()
                return
            self._proc = proc
            threading.Thread(
                target=self._drain_stderr, args=(proc,), name="screenrecord-err", daemon=True
            ).start()

            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            while not self._stopping.is_set():
                try:
                    chunk = os.read(fd, 1 << 16)
                except OSError:
                    break
                if not chunk:
                    break
                for frame in decoder.feed(chunk):
                    now = time.monotonic()
                    self.frames.push(frame, now)
                    self.stats.frames += 1
                    self.stats.last_frame_at = now
                    if not self._ready.is_set():
                        self.resolution = (int(frame.shape[1]), int(frame.shape[0]))
                        self._ready.set()

            self._terminate(proc)
            if self._stopping.is_set():
                break

            # The process ended on its own: either the 3-minute limit elapsed (normal, just
            # restart) or the encoder refused our parameters (retry at native size).
            if self.stats.frames == 0 and not self._ready.is_set():
                if self._spawned_with_size and not self._native_fallback_done:
                    logger.info(
                        "screenrecord rejected --size=%s, retrying at native resolution.",
                        self._spawned_with_size,
                    )
                    self._native_fallback_done = True
                    continue
                self._start_error = ADBError(
                    "screenrecord exited before sending a frame" + self._error_suffix()
                )
                self._ready.set()
                break

            self.restarts += 1
            logger.debug(
                "screenrecord exited after %d frames; restarting (restart #%d).",
                self.stats.frames,
                self.restarts,
            )
            self._stopping.wait(0.15)

        self._ready.set()

    def _terminate(self, proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        self._stopping.set()
        if self._proc is not None:
            self._terminate(self._proc)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None

    # -- convenience ---------------------------------------------------------------
    def latest_frame(self, min_seq: int = 0, timeout: float = 0.5) -> Optional[np.ndarray]:
        return self.frames.get(min_seq=min_seq, timeout=timeout)

    @property
    def frame_seq(self) -> int:
        return self.frames.seq

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ==============================================================================
# Backend selection
# ==============================================================================


class MultiClient:
    """Capture (and, with ``scrcpy``, input) client for one or several ADB devices.

    Usage::

        mc = MultiClient(target_width=720, max_fps=60)
        mc.start()
        frame = mc.latest_frame()          # newest BGR frame, never stale
        mc.drag((210, 1135), (300, 700))   # control socket if we have one, else adb
        mc.stop()

    With several devices::

        mc = MultiClient(serials=["R58M1234", "10.0.0.5:5555"]).start()
        for serial, session in mc.sessions.items():
            ...
    """

    #: accepted by ``backend=``
    BACKENDS = ("auto", "scrcpy", "screenrecord", "scrcpy_py", "screencap")

    def __init__(
        self,
        adb_path: str = "adb",
        serial: Optional[str] = None,
        serials: Optional[List[str]] = None,
        target_width: int = 720,
        bitrate: int = 12_000_000,
        max_fps: int = 60,
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
        #: backend actually in use after start()
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
            max_fps=getattr(cfg, "capture_max_fps", 60),
            backend=getattr(cfg, "capture_backend", "auto"),
            connect_timeout=getattr(cfg, "capture_connect_timeout", 12.0),
            scrcpy_jar=getattr(cfg, "capture_scrcpy_jar", None),
        )

    def _backend_chain(self) -> List[str]:
        """Capture backends to try, best first."""
        backend = self.backend
        if backend in ("", "auto"):
            chain = ["scrcpy", "screenrecord"]
            if scrcpy_py_available():
                chain.append("scrcpy_py")
            chain.append("screencap")
            return chain
        if backend in ("scrcpy", "scrcpy_native"):
            return ["scrcpy"]
        if backend in ("screenrecord", "screenrecord_h264"):
            return ["screenrecord"]
        if backend in ("scrcpy_py", "scrcpy_client"):
            return ["scrcpy_py", "scrcpy"]
        if backend == "screencap":
            return ["screencap"]
        return [backend]

    def _new_session(self, adb: Adb, backend: str):
        if backend == "scrcpy":
            return ScrcpySession(
                adb,
                target_width=self.target_width,
                bitrate=self.bitrate,
                max_fps=self.max_fps,
                connect_timeout=self.connect_timeout,
                stay_awake=self.stay_awake,
                jar_path=self.scrcpy_jar,
            )
        if backend == "screenrecord":
            return ScreenrecordSession(
                adb,
                target_width=self.target_width,
                bitrate=self.bitrate,
                max_fps=self.max_fps,
                first_frame_timeout=self.connect_timeout,
            )
        if backend == "scrcpy_py":
            session = _try_scrcpy_py_session(adb, self.target_width, self.bitrate, self.max_fps)
            if session is None:
                raise ADBError("the `scrcpy` PyPI package is not importable")
            return session
        if backend == "screencap":
            return ScreencapSession(adb, target_width=self.target_width)
        raise ADBError(f"unknown capture backend {backend!r}")

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
        """Start a session, degrading down the capture ladder if a backend cannot come up.

        Capture is not the point of the bot - playing is - so with ``backend: auto`` a failure
        anywhere on the ladder degrades instead of aborting the run.
        """
        chain = self._backend_chain()
        reasons: List[str] = []
        last_exc: Optional[BaseException] = None

        for backend in chain:
            try:
                session = self._new_session(adb, backend)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"{backend}: {exc}")
                last_exc = exc
                continue
            try:
                session.start()
            except Exception as exc:  # noqa: BLE001 - any failure may fall through the ladder
                try:
                    session.stop()
                except Exception:  # noqa: BLE001
                    pass
                reasons.append(f"{backend}: {exc}")
                last_exc = exc
                logger.warning("capture backend %r is not usable here: %s", backend, exc)
                continue

            self.backend_used = backend
            if reasons:
                self.fallback_reason = "\n".join(reasons)
                logger.warning(
                    "capture degraded to %r after %d failed backend(s). Capture still works, "
                    "but frame rate and input latency are worse than the preferred backend "
                    "would give.",
                    backend,
                    len(reasons),
                )
            return session

        report = "\n".join(reasons) or "no capture backend available"
        if self.backend not in ("", "auto"):
            raise ADBError(f"capture backend {self.backend!r} failed:\n{report}") from last_exc
        raise ADBError("no capture backend worked on this device:\n" + report) from last_exc

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
        """The scrcpy control channel, or ``None`` on backends without one."""
        return getattr(self.session(), "control", None)

    @property
    def has_control(self) -> bool:
        """True when input can be injected over the scrcpy socket (sub-10 ms)."""
        return self.control is not None

    @property
    def fps(self) -> float:
        return self.session().stats.fps

    @property
    def frame_age(self) -> float:
        return self.session().frames.age

    def wait_new_frame(self, last_seq: int, timeout: float = 0.4) -> Optional[np.ndarray]:
        """Block until the stream advances past ``last_seq`` (or time out)."""
        session = self.session()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if session.frame_seq > last_seq:
                break
            time.sleep(0.003)
        return session.latest_frame()

    def status(self) -> str:
        return " | ".join(
            f"{serial}: {s.resolution[0]}x{s.resolution[1]} @ {s.stats.fps:.1f}fps"
            for serial, s in self.sessions.items()
        )

    # -- input (only over the scrcpy control socket; adb fallback lives in touch.py) --
    def _adb_of(self, session) -> Optional[Adb]:
        return getattr(session, "adb", None)

    def tap(self, point: Tuple[float, float], hold_ms: float = 70.0, serial=None) -> bool:
        session = self.session(serial)
        control = getattr(session, "control", None)
        if control is not None:
            control.tap(int(point[0]), int(point[1]), hold_ms=hold_ms)
            return True
        adb = self._adb_of(session)
        if adb is None:
            return False
        try:
            adb.shell(
                f"input swipe {int(point[0])} {int(point[1])} "
                f"{int(point[0])} {int(point[1])} {int(hold_ms)}",
                timeout=10.0,
            )
            return True
        except ADBError:
            return False

    def drag(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        duration_ms: float = 220.0,
        serial=None,
    ) -> bool:
        session = self.session(serial)
        control = getattr(session, "control", None)
        if control is not None:
            control.drag((int(start[0]), int(start[1])), (int(end[0]), int(end[1])),
                         duration_ms=duration_ms)
            return True
        adb = self._adb_of(session)
        if adb is None:
            return False
        try:
            adb.shell(
                f"input swipe {int(start[0])} {int(start[1])} "
                f"{int(end[0])} {int(end[1])} {int(duration_ms)}",
                timeout=10.0,
            )
            return True
        except ADBError:
            return False


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
