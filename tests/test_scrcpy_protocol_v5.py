"""End-to-end tests of the scrcpy wire protocol against a *fake server*, with no phone attached.

The failure this guards against is the one that actually happened on a real Galaxy A52s::

    WARNING | scrcpy video stream ended
    INFO    | scrcpy stream up: SM-A528B (720x1600)
    capture : scrcpy started but produced no frame

which is what you get when a **2021 server (v1.24) meets Android 13/14**: the sockets are accepted,
the handshake succeeds, and then the capture pipeline dies because the display APIs it uses changed.
Two things were wrong on our side and both are pinned down here:

1. the read loop signalled "ready" even when it died, so a dead stream looked healthy;
2. only the v1.x argument set and handshake were implemented, so a modern server could not be used.

The fake server below speaks the v5.0.1 layout (dummy byte, 64-byte name, then a raw Annex-B
stream because ``send_stream_meta=false``) and serves **real H.264** produced by PyAV, so the
decoder path is exercised too.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import List, Tuple

import numpy as np
import pytest

from cr_bot.adb import scrcpy_protocol as proto
from cr_bot.adb.adb_utils import ADBError
from cr_bot.adb.scrcpy_multiclient import MultiClient, ScreencapSession, ScrcpySession


# ======================================================================================
# a real H.264 Annex-B stream to serve
# ======================================================================================

def _annexb_stream(frames: int = 8, width: int = 128, height: int = 96) -> bytes:
    av = pytest.importorskip("av")
    codec = av.CodecContext.create("libx264", "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = "yuv420p"
    codec.time_base = __import__("fractions").Fraction(1, 30)
    codec.options = {"preset": "ultrafast", "tune": "zerolatency"}
    chunks: List[bytes] = []
    for i in range(frames):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :, 1] = 40 + (i * 15) % 200
        img[20:60, 20:100, 2] = 200
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in codec.encode(frame):
            chunks.append(bytes(packet))
    for packet in codec.encode(None):
        chunks.append(bytes(packet))
    return b"".join(chunks)


# ======================================================================================
# fake scrcpy servers
# ======================================================================================

class FakeScrcpyServer:
    """Serves the video + control sockets exactly like ``DesktopConnection`` does."""

    def __init__(self, modern: bool, stream: bytes, send_frames: bool = True,
                 width: int = 720, height: int = 1600, close_after_handshake: bool = False):
        self.modern = modern
        self.stream = stream
        self.send_frames = send_frames
        self.width = width
        self.height = height
        self.close_after_handshake = close_after_handshake
        self.connections: List[str] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> "FakeScrcpyServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def _serve(self) -> None:
        """Serve one connection cycle per attempt, like a real server being relaunched."""
        while not self._stop.is_set():
            if not self._serve_once():
                return

    def _serve_once(self) -> bool:
        """Accept video, handshake, accept control, then stream - the real server's order."""
        try:
            video, _ = self._sock.accept()
        except OSError:
            return False
        self.connections.append("video")
        with video:
            video.sendall(b"\x00")                       # dummy byte
            name = b"SM-A528B" + b"\x00" * (64 - 8)       # 64-byte device name
            video.sendall(name)
            if not self.modern:
                video.sendall(bytes([self.width >> 8, self.width & 0xFF,
                                     self.height >> 8, self.height & 0xFF]))

            try:
                control, _ = self._sock.accept()
            except OSError:
                return False
            self.connections.append("control")
            control.settimeout(0.2)

            if self.close_after_handshake:
                # exactly what v1.24 does on Android 13/14: handshake, then the capture dies
                time.sleep(0.05)
                control.close()
                return True

            for i in range(0, len(self.stream), 512):
                if self._stop.is_set():
                    return False
                try:
                    video.sendall(self.stream[i:i + 512])
                except OSError:
                    return True
                time.sleep(0.002)
            # stay open like a live stream while the client reads
            deadline = time.monotonic() + 0.6
            while time.monotonic() < deadline and not self._stop.is_set():
                try:
                    control.recv(4096)
                except (socket.timeout, OSError):
                    pass
            control.close()
        return True


class FakeAdb:
    """Just enough ``Adb`` for the scrcpy session: no device, no adb binary."""

    def __init__(self) -> None:
        self.serial = "FAKE123"
        self.commands: List[List[str]] = []

    def base_cmd(self, include_serial: bool = True) -> List[str]:
        return ["adb"]

    def connect(self, serial=None) -> str:
        return self.serial

    def display_info(self):
        from cr_bot.adb.adb_utils import DisplayInfo

        return DisplayInfo(width=1080, height=2400, rotation=0, density=450)

    def push(self, local, remote_dir="/data/local/tmp/", timeout: float = 60.0) -> None:
        pass

    def forward(self, local_port: int, remote: str) -> None:
        self.commands.append(["forward", str(local_port), remote])

    def run(self, args, timeout: float = 10.0, include_serial: bool = True, check: bool = False):
        import subprocess

        self.commands.append(list(args))
        return subprocess.CompletedProcess(args, 0, b"", b"")


class _FakeProc:
    def poll(self):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout: float = 0):
        return 0


def _session(adb: FakeAdb, port: int, version: Tuple[int, int, int], **kwargs) -> ScrcpySession:
    kwargs.setdefault("connect_timeout", 1.2)
    session = ScrcpySession(adb, **kwargs)
    session.jar_version = version
    session._launch_server = lambda overrides=None: _FakeProc()      # type: ignore[assignment]
    session._kill_stale_servers = lambda: 0                          # type: ignore[assignment]
    session._verify_pushed_jar = lambda *a, **k: None                # type: ignore[assignment]
    session._compute_max_size = lambda: 1600                         # type: ignore[assignment]
    monkeypatch_free_port(session, port)
    return session


def monkeypatch_free_port(session: ScrcpySession, port: int) -> None:
    import cr_bot.adb.scrcpy_multiclient as module

    session._fixed_port = port                                     # type: ignore[attr-defined]
    original = module._free_port
    module._free_port = lambda: port                               # type: ignore[assignment]
    session._restore_free_port = lambda: setattr(module, "_free_port", original)  # type: ignore[attr-defined]


# ======================================================================================
# argument sets
# ======================================================================================

def test_modern_server_gets_the_v2_argument_set() -> None:
    options = proto.build_options((5, 0, 1), bitrate=8_000_000, max_size=1600, max_fps=30)
    assert options["audio"] == "false", "v2+ needs audio disabled explicitly"
    assert options["video"] == "true"
    assert options["video_codec"] == "h264"
    assert options["video_bit_rate"] == 8_000_000
    assert "bit_rate" not in options, "the key was renamed in v2.0"
    assert options["send_stream_meta"] == "false"
    assert options["send_frame_meta"] == "false"
    assert options["scid"] == -1, "keeps the classic `scrcpy` abstract socket name"


def test_legacy_server_keeps_the_v1_argument_set() -> None:
    options = proto.build_options((1, 24, 0), bitrate=8_000_000, max_size=1600, max_fps=30)
    assert options["bit_rate"] == 8_000_000
    assert "video_bit_rate" not in options
    assert "audio" not in options and "send_stream_meta" not in options
    assert options["send_frame_meta"] == "false"


def test_version_is_parsed_from_the_jar_name() -> None:
    assert proto.parse_server_version("scrcpy-server-v5.0.1.jar") == (5, 0, 1)
    assert proto.parse_server_version("scrcpy-server-v1.24.jar") == (1, 24, 0)
    assert proto.parse_server_version("scrcpy-server-v2.4") == (2, 4, 0)
    assert proto.is_modern_version((5, 0, 1)) is True
    assert proto.is_modern_version((1, 24, 0)) is False


# ======================================================================================
# v5 handshake + decode, end to end
# ======================================================================================

def test_modern_handshake_streams_and_decodes_frames() -> None:
    stream = _annexb_stream()
    server = FakeScrcpyServer(modern=True, stream=stream).start()
    adb = FakeAdb()
    session = _session(adb, server.port, (5, 0, 1))
    try:
        session.start()
        frame = session.latest_frame(timeout=6.0)
        assert frame is not None, "a modern server with send_stream_meta=false must decode"
        assert frame.shape[0] == 96 and frame.shape[1] == 128
        assert session.device_name == "SM-A528B"
        assert session.resolution == (128, 96), "size comes from the first decoded frame"
        assert "video" in server.connections and "control" in server.connections
    finally:
        session.stop()
        server.stop()


def test_legacy_handshake_still_works() -> None:
    stream = _annexb_stream()
    server = FakeScrcpyServer(modern=False, stream=stream).start()
    session = _session(FakeAdb(), server.port, (1, 24, 0))
    try:
        session.start()
        assert session.latest_frame(timeout=6.0) is not None
    finally:
        session.stop()
        server.stop()


def test_a_stream_that_dies_before_the_first_frame_fails_the_start() -> None:
    """This is the exact Galaxy A52s symptom: handshake OK, then silence."""
    server = FakeScrcpyServer(modern=True, stream=b"", close_after_handshake=True).start()
    session = _session(FakeAdb(), server.port, (5, 0, 1))
    try:
        with pytest.raises(ADBError) as excinfo:
            session.start()
    finally:
        session.stop()
        server.stop()
    message = str(excinfo.value)
    assert "before sending a single frame" in message or "no video frame arrived" in message
    assert "attempt" in message.lower() or "configurations" in message.lower()


def test_the_failure_report_is_attached_to_the_error() -> None:
    server = FakeScrcpyServer(modern=True, stream=b"", close_after_handshake=True).start()
    session = _session(FakeAdb(), server.port, (5, 0, 1))
    try:
        with pytest.raises(ADBError) as excinfo:
            session.start()
    finally:
        session.stop()
        server.stop()
    # the operator must be able to see *why*, not just that it failed
    assert "scrcpy server" in str(excinfo.value).lower()


# ======================================================================================
# retry ladder + fallback
# ======================================================================================

def test_attempt_ladder_covers_encoders_and_bitrate_for_modern_servers() -> None:
    session = ScrcpySession(FakeAdb())
    session.jar_version = (5, 0, 1)
    labels = [label for label, _ in session._attempts()]
    overrides = [o for _, o in session._attempts()]
    assert labels[0] == "default"
    assert any("video_encoder" in o for o in overrides)
    assert any(o.get("video_codec") == "h265" for o in overrides)
    assert any("video_bit_rate" in o for o in overrides)


def test_attempt_ladder_uses_legacy_keys_for_old_servers() -> None:
    session = ScrcpySession(FakeAdb())
    session.jar_version = (1, 24, 0)
    overrides = [o for _, o in session._attempts()]
    assert any("bit_rate" in o for o in overrides)
    assert not any("video_bit_rate" in o for o in overrides)
    assert not any("video_encoder" in o for o in overrides)


def test_dead_stream_falls_back_to_screencap_through_multiclient(monkeypatch) -> None:
    """The whole point: a dead scrcpy must not stop the bot."""
    server = FakeScrcpyServer(modern=True, stream=b"", close_after_handshake=True).start()

    class _RecordingScreencap(ScreencapSession):
        def start(self):
            self.serial = "FAKE123"
            return self

    def _make_session(adb, **kwargs):
        return _session(adb, server.port, (5, 0, 1), **kwargs)

    monkeypatch.setattr("cr_bot.adb.scrcpy_multiclient.Adb", lambda *a, **k: FakeAdb())
    monkeypatch.setattr("cr_bot.adb.scrcpy_multiclient.ScrcpySession", _make_session)
    monkeypatch.setattr("cr_bot.adb.scrcpy_multiclient.ScreencapSession", _RecordingScreencap)
    client = MultiClient(adb_path="adb", serial="FAKE123", backend="auto")
    try:
        client.start()
        assert client.backend_used == "screencap"
        assert "frame" in client.fallback_reason.lower()
    finally:
        client.stop()
        server.stop()
