"""Screen capture for one or many Android devices ("MultiClient").

Why this exists
---------------
The original bot captured with ``adb exec-out screencap``. On an emulator that is tolerable;
on a **real phone** a full-resolution screencap round trip costs roughly 150-400 ms and ships
~10 MB of raw pixels, so the bot reasons about a third of a second old frame and is always a
beat behind.

Backends (tried in this order by ``auto``)
------------------------------------------
``screenrecord``  Android's own binary. H.264 at display rate, no server jar, no socket
                  handshake, nothing version-specific — so it keeps working on Android
                  releases where a pushed scrcpy server goes silent.
``scrcpy_py``     the ``scrcpy`` PyPI package (py-scrcpy-client), if it is importable. It
                  ships and pushes its own server jar, so nothing has to be bundled here.
``screencap``     ``adb exec-out screencap`` polling. Always works, ~1 fps on a 1080p phone.

There is deliberately **no bundled scrcpy server jar and no hand-written scrcpy protocol**
any more. Refresh rates vary between releases, vendor encoders reject configurations that
older Android accepted, and every hand-maintained copy became stale; the two paths above
cover the same need without anything version-sensitive living in this repository.

Touch input does not depend on the capture backend: it uses ``adb shell input`` /
``motionevent`` and verifies that each deploy actually registered on the device.
"""

from __future__ import annotations

import collections
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cr_bot.adb.adb_utils import Adb, ADBError, DisplayInfo

logger = logging.getLogger(__name__)


# ======================================================================================
# Frame plumbing
# ======================================================================================


@dataclass
class FrameBuffer:
    """Holds only the most recent decoded frame.

    A game bot never wants a queue of stale frames: acting on a 300 ms old frame is worse
    than skipping a frame. Readers take whatever is newest and immediately loop.
    """

    frame: Optional[np.ndarray] = None
    seq: int = 0
    captured_at: float = 0.0
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def push(self, frame: np.ndarray, ts: Optional[float] = None) -> None:
        with self._cond:
            self.frame = frame
            self.seq += 1
            self.captured_at = time.monotonic() if ts is None else ts
            self._cond.notify_all()

    def get(self, min_seq: int = 0, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Return the newest frame, waiting briefly for one newer than ``min_seq``."""
        with self._cond:
            if self.seq <= min_seq or self.frame is None:
                self._cond.wait(timeout=timeout)
            return self.frame

    def wait_next(self, last_seq: int, timeout: float = 0.5) -> Optional[np.ndarray]:
        with self._cond:
            self._cond.wait_for(lambda: self.seq > last_seq, timeout=timeout)
            return self.frame

    @property
    def age(self) -> float:
        if self.captured_at <= 0:
            return float("inf")
        return max(0.0, time.monotonic() - self.captured_at)


class H264FrameDecoder:
    """Thin wrapper over PyAV that turns a raw Annex-B byte stream into BGR frames."""

    def __init__(self) -> None:
        from av.codec import CodecContext  # imported lazily so errors are actionable

        self._codec = CodecContext.create("h264", "r")
        self._codec.thread_type = "AUTO"
        self.frames_decoded = 0
        self.width = 0
        self.height = 0

    def feed(self, raw: bytes):
        """Yield BGR numpy frames decoded from this chunk of the elementary stream."""
        if not raw:
            return
        try:
            packets = self._codec.parse(raw)
        except Exception:  # noqa: BLE001 - corrupt NAL, skip chunk
            return

        for packet in packets:
            try:
                frames = self._codec.decode(packet)
            except Exception:  # noqa: BLE001
                continue
            for frame in frames:
                try:
                    bgr = frame.to_ndarray(format="bgr24")
                except Exception:  # noqa: BLE001
                    continue
                self.width, self.height = int(bgr.shape[1]), int(bgr.shape[0])
                self.frames_decoded += 1
                yield bgr

    def flush(self):
        try:
            for frame in self._codec.decode(None):
                yield frame.to_ndarray(format="bgr24")
        except Exception:  # noqa: BLE001
            return


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


# ==============================================================================
# screencap session (last resort)
# ==============================================================================


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
        self.frames = FrameBuffer()
        self.stats = SessionStats()
        #: no backend exposes a control socket; input goes through `adb shell input`
        self.control = None
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


# ==============================================================================
# screenrecord session (stock Android, no server jar)
# ==============================================================================


class ScreenrecordSession:
    """Live H.264 capture through Android's stock ``screenrecord`` binary.

    Why this backend exists
    -----------------------
    ``screenrecord`` is part of AOSP since 4.4, so it is *already on the phone* - there is
    no server jar to push, no socket handshake, no device-meta negotiation, and therefore
    none of the version-dependent failure modes that make the scrcpy server go silent on
    newer Android releases.  It streams H.264 at display rate, so it is 20-30x faster than
    ``screencap`` (which has to ship ~10 MB of raw RGBA per frame over USB).

    Known limitations, all handled here:

    * ``screenrecord`` self-terminates after ``--time-limit`` seconds (180 is the maximum
      Android accepts).  The reader loop simply restarts it, costing a few dropped frames
      every three minutes.
    * No control channel, so :attr:`control` stays ``None``; touch input keeps using
      ``adb shell input`` / ``motionevent`` exactly as it does on the screencap path.  The
      bot already verifies its own taps, so it does not depend on scrcpy's input injection.
    * ``--size`` must be even and some encoders reject non-standard sizes outright, so a
      rejected size is retried once at native resolution before the session gives up.
    """

    #: shown in ``--stats`` output
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
        #: always None: ``screenrecord`` has no control socket
        #: no backend exposes a control socket; input goes through `adb shell input`
        self.control = None

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
        logger.info(
            "screenrecord capture up: %dx%d (no server jar needed)", *self.resolution
        )
        return self

    def _error_suffix(self) -> str:
        tail = "\n".join(line for line in self._stderr_tail if line.strip())
        return f"\nscreenrecord said: {tail}" if tail else ""

    def _capture_size(self) -> Optional[str]:
        """Pick a ``--size`` that keeps the short edge near ``target_width``.

        ``DisplayInfo`` already reports the size as currently displayed (it transposes on a
        quarter turn), which is exactly what ``screenrecord`` records, so no rotation
        handling is needed here.
        """
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
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

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

            # The process ended on its own. Either the 3-minute limit elapsed (normal,
            # just restart) or the encoder refused our parameters (retry at native size).
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


# ======================================================================================
# Backend selection
# ======================================================================================


def scrcpy_py_available() -> Optional[str]:
    """Return the installed ``scrcpy`` package version, or ``None`` if it is absent."""
    try:
        import scrcpy  # noqa: F401, PLC0415

        return str(getattr(scrcpy, "__version__", "installed"))
    except Exception:  # noqa: BLE001 - absence is the normal case
        return None


class MultiClient:
    """Capture client for one or several ADB devices at once.

    Usage::

        mc = MultiClient(target_width=720)
        mc.start()
        frame = mc.latest_frame()
        mc.drag((210, 1135), (300, 700))

    With several devices::

        mc = MultiClient(serials=["R58M1234", "10.0.0.5:5555"])
        mc.start()
        for serial, session in mc.sessions.items():
            ...
    """

    #: backend names accepted by ``backend=``
    BACKENDS = ("auto", "screenrecord", "scrcpy_py", "screencap")

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
    ) -> None:
        self.adb_path = adb_path
        self.requested = serials or ([serial] if serial else [None])
        self.target_width = int(target_width)
        self.bitrate = int(bitrate)
        self.max_fps = int(max_fps)
        self.backend = (backend or "auto").lower()
        self.connect_timeout = float(connect_timeout)
        self.stay_awake = bool(stay_awake)

        self.sessions: Dict[str, object] = {}
        self.started = False
        #: backend actually in use after start() (``screenrecord`` / ``scrcpy_py`` / ``screencap``)
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

    def _new_session(self, adb: Adb, backend: str):
        if backend == "screencap":
            return ScreencapSession(adb, target_width=self.target_width)

        if backend == "scrcpy_py":
            session = _try_scrcpy_py_session(adb, self.target_width, self.bitrate, self.max_fps)
            if session is None:
                raise ADBError(
                    "the `scrcpy` PyPI package is not usable here (not installed, or it could "
                    "not start a stream). Install it with `pip install scrcpy-client` on "
                    "Python 3.10 or older."
                )
            return session

        return ScreenrecordSession(
            adb,
            target_width=self.target_width,
            bitrate=self.bitrate,
            max_fps=self.max_fps,
            first_frame_timeout=self.connect_timeout,
        )

    def _backend_chain(self) -> List[str]:
        """Capture backends to try, best first."""
        backend = self.backend
        if backend in ("", "auto"):
            return ["screenrecord", "scrcpy_py", "screencap"]
        if backend in ("screenrecord", "screenrecord_h264"):
            return ["screenrecord"]
        if backend in ("scrcpy_py", "scrcpy", "scrcpy_client"):
            return ["scrcpy_py"]
        if backend == "screencap":
            return ["screencap"]
        return [backend]

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
        """Start a session, degrading down the capture ladder when a backend cannot come up.

        Capture is not the point of the bot - playing is - so with ``backend: auto`` a
        failure anywhere on the ladder degrades instead of aborting the run.
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
                    "but frame rate is lower than the preferred backend would give.",
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
    def control(self):
        """Always ``None``: no backend exposes a control socket any more."""
        return None

    @property
    def fps(self) -> float:
        return self.session().stats.fps

    @property
    def frame_age(self) -> float:
        return getattr(self.session().frames, "age", float("inf"))

    def drag(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        duration_ms: float = 220.0,
        serial: Optional[str] = None,
    ) -> bool:
        """Best-effort drag through ``adb shell input`` (there is no control socket)."""
        session = self.session(serial)
        adb = getattr(session, "adb", None)
        if adb is None:
            return False
        cmd = (
            f"input swipe {int(start[0])} {int(start[1])} "
            f"{int(end[0])} {int(end[1])} {int(duration_ms)}"
        )
        try:
            adb.shell(cmd, timeout=10.0)
            return True
        except ADBError:
            return False

    def tap(self, point: Tuple[float, float], serial: Optional[str] = None) -> bool:
        session = self.session(serial)
        adb = getattr(session, "adb", None)
        if adb is None:
            return False
        try:
            adb.shell(f"input tap {int(point[0])} {int(point[1])}", timeout=10.0)
            return True
        except ADBError:
            return False


# ==============================================================================
# scrcpy PyPI package adapter (optional)
# ==============================================================================


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
        self.frames = FrameBuffer()
        self.stats = SessionStats()
        self.resolution: Tuple[int, int] = (0, 0)
        self._modern_stream_meta = False
        self.serial: Optional[str] = adb.serial
        self.device_name = "scrcpy_py"
        self.display: Optional[DisplayInfo] = None
        #: capture only - input keeps using `adb shell input`
        self.control = None
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
