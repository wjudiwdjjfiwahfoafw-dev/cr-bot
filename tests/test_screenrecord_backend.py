"""Tests for the ``screenrecord`` capture backend and the capture ladder.

``screenrecord`` is the middle rung between scrcpy and ``screencap``: it needs no server
jar, so it cannot fail for any of the version/handshake reasons scrcpy does on Android
13+, but it still delivers a compressed H.264 stream at display rate instead of ~10 MB of
raw pixels per frame.

These tests drive the real class against a fake ``adb`` executable that streams a genuine
H.264 elementary stream (encoded with libx264), so the decoder path is exercised for real.
"""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from cr_bot.adb.adb_utils import ADBError, DisplayInfo
from cr_bot.adb.capture import (
    MultiClient,
    ScreenrecordSession,
    ScreencapSession,
)


# ======================================================================================
# helpers
# ======================================================================================


def _annexb_stream(frames: int = 10, width: int = 128, height: int = 224) -> bytes:
    av = pytest.importorskip("av")
    import fractions

    codec = av.CodecContext.create("libx264", "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = "yuv420p"
    codec.time_base = fractions.Fraction(1, 30)
    codec.options = {"preset": "ultrafast", "tune": "zerolatency"}
    chunks: List[bytes] = []
    for i in range(frames):
        img = np.zeros((height, width, 3), np.uint8)
        img[:, :, 2] = 30 + (i * 20) % 180
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in codec.encode(frame):
            chunks.append(bytes(packet))
    for packet in codec.encode(None):
        chunks.append(bytes(packet))
    return b"".join(chunks)


FAKE_ADB_SOURCE = r'''
import os, sys

log = os.environ.get("FAKE_ADB_LOG", "")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(" ".join(sys.argv[1:]) + "\n")

mode = os.environ.get("FAKE_SR_MODE", "stream")

if "screenrecord" not in " ".join(sys.argv):
    # connect()/display_info() etc. are provided by the Python fake, never here.
    sys.exit(0)

reject_named_size = mode == "reject_size" and any(a.startswith("--size=") for a in sys.argv)
if reject_named_size:
    sys.stderr.write("ERROR: unable to configure codec for the requested size\n")
    sys.stderr.flush()
    sys.exit(1)

if mode == "no_frames":
    sys.stderr.write("ERROR: could not create codec 'video/avc'\n")
    sys.stderr.flush()
    sys.exit(1)

with open(os.environ["FAKE_SR_STREAM"], "rb") as fh:
    data = fh.read()
try:
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
except BrokenPipeError:
    pass
sys.exit(0)
'''


class FakeAdb:
    """Minimal ``Adb`` stand-in: only the calls the sessions actually make."""

    def __init__(self, adb_bin: str, display: Optional[DisplayInfo] = None, serial: str = "FAKE123") -> None:
        self.adb_bin = adb_bin
        self.serial = serial
        self._display = display or DisplayInfo(width=1080, height=2400, rotation=0, density=450)

    def base_cmd(self, include_serial: bool = True):
        cmd = [self.adb_bin]
        if include_serial and self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def connect(self, serial=None, autoprobe_emulators: bool = True) -> str:
        return serial or self.serial

    def display_info(self) -> DisplayInfo:
        return self._display

    def run(self, args, timeout: float = 10.0, include_serial: bool = True, check: bool = False):
        class _R:
            returncode = 0
            stdout = b""
            stderr = b""

        return _R()

    def exec_out(self, args, timeout: float = 10.0) -> bytes:
        return b""

    def shell(self, command, timeout: float = 10.0) -> str:
        return ""


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    script = tmp_path / "fake_adb.py"
    script.write_text(FAKE_ADB_SOURCE, encoding="utf-8")
    wrapper = tmp_path / "adb"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    # Windows cannot exec a shebang script, so expose a real argv prefix instead.
    adb_bin = str(wrapper) if os.name != "nt" else str(script)

    stream = tmp_path / "stream.h264"
    stream.write_bytes(_annexb_stream())
    log = tmp_path / "calls.log"
    log.write_text("", encoding="utf-8")

    monkeypatch.setenv("FAKE_SR_STREAM", str(stream))
    monkeypatch.setenv("FAKE_ADB_LOG", str(log))
    monkeypatch.setenv("FAKE_SR_MODE", "stream")
    return {"adb_bin": adb_bin, "log": log, "stream": stream, "tmp": tmp_path}


def _wait(predicate, timeout: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ======================================================================================
# ScreenrecordSession
# ======================================================================================


def test_screenrecord_streams_and_decodes_frames(fake_env) -> None:
    adb = FakeAdb(fake_env["adb_bin"])
    session = ScreenrecordSession(adb, target_width=720, first_frame_timeout=6.0)
    session.start()
    try:
        assert session.resolution == (128, 224)
        assert session.stats.frames > 0
        frame = session.latest_frame(timeout=2.0)
        assert frame is not None
        assert frame.shape == (224, 128, 3)
    finally:
        session.stop()


def test_screenrecord_has_no_control_channel(fake_env) -> None:
    """Input must keep working through adb shell while this backend is in use."""
    session = ScreenrecordSession(FakeAdb(fake_env["adb_bin"]), first_frame_timeout=6.0)
    session.start()
    try:
        assert session.control is None
    finally:
        session.stop()


def test_capture_size_matches_target_width_and_stays_even_aligned() -> None:
    adb = FakeAdb("adb", display=DisplayInfo(width=1080, height=2400, rotation=0, density=450))
    session = ScreenrecordSession(adb, target_width=720)
    assert session._capture_size() == "720x1600"

    # DisplayInfo already reports the *displayed* size, so a landscape frame stays wide.
    landscape = FakeAdb("adb", display=DisplayInfo(width=2400, height=1080, rotation=1, density=450))
    assert ScreenrecordSession(landscape, target_width=720)._capture_size() == "1600x720"


def test_command_includes_h264_output_and_time_limit(fake_env) -> None:
    session = ScreenrecordSession(FakeAdb(fake_env["adb_bin"]), target_width=720, time_limit=120)
    cmd = session._command("720x1600")
    joined = " ".join(cmd)
    assert "exec-out" in cmd
    assert "--output-format=h264" in joined
    assert "--time-limit=120" in joined
    assert "--size=720x1600" in joined
    assert cmd[-1] == "-"


def test_time_limit_is_clamped_to_android_maximum() -> None:
    session = ScreenrecordSession(FakeAdb("adb"), time_limit=9999)
    assert session.time_limit == 180


def test_session_restarts_after_screenrecord_exits(fake_env, monkeypatch) -> None:
    """screenrecord stops itself after 3 minutes; capture must survive that."""
    monkeypatch.setenv("FAKE_SR_MODE", "stream")
    session = ScreenrecordSession(FakeAdb(fake_env["adb_bin"]), first_frame_timeout=6.0)
    session.start()
    try:
        assert _wait(lambda: session.restarts >= 1, timeout=6.0), "no restart happened"
        assert session.is_alive
    finally:
        session.stop()


def test_no_frames_raises_with_the_tool_output(fake_env, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_SR_MODE", "no_frames")
    session = ScreenrecordSession(FakeAdb(fake_env["adb_bin"]), first_frame_timeout=5.0)
    with pytest.raises(ADBError) as excinfo:
        session.start()
    text = str(excinfo.value)
    assert "no video frame" in text or "before sending a frame" in text
    assert "could not create codec" in text


def test_rejected_size_is_retried_at_native_resolution(fake_env, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FAKE_SR_MODE", "reject_size")
    session = ScreenrecordSession(
        FakeAdb(fake_env["adb_bin"]), target_width=720, first_frame_timeout=6.0
    )
    session.start()
    try:
        calls = fake_env["log"].read_text(encoding="utf-8")
        commands = [ln for ln in calls.splitlines() if "screenrecord" in ln]
        assert any("--size=" in c for c in commands), commands
        assert any("--size=" not in c for c in commands), "native retry never happened"
        assert session.stats.frames > 0
    finally:
        session.stop()


def test_stop_is_idempotent(fake_env) -> None:
    session = ScreenrecordSession(FakeAdb(fake_env["adb_bin"]), first_frame_timeout=6.0)
    session.start()
    session.stop()
    session.stop()


# ======================================================================================
# MultiClient: the capture ladder
# ======================================================================================


def test_auto_backend_ladder_order() -> None:
    """The bundled native scrcpy stack is the best backend: 60fps video *and* a control socket.

    It must come first, because that is the only path that injects input without spawning an
    `input` JVM on the device. screenrecord (same video, no control) is the fallback, and
    screencap proves the ADB link when nothing else works.
    """
    client = MultiClient(backend="auto")
    chain = client._backend_chain()
    assert chain[0] == "scrcpy"
    assert chain[1] == "screenrecord"
    assert chain[-1] == "screencap"
    assert set(chain) <= set(MultiClient.BACKENDS)


def test_explicit_backends_do_not_fall_through() -> None:
    assert MultiClient(backend="screenrecord")._backend_chain() == ["screenrecord"]
    assert MultiClient(backend="screencap")._backend_chain() == ["screencap"]
    assert MultiClient(backend="scrcpy")._backend_chain() == ["scrcpy"]
    # py-scrcpy-client is not shipped with the bot, so it may hand over to the native one
    assert MultiClient(backend="scrcpy_py")._backend_chain()[0] == "scrcpy_py"


class _BoomSession:
    serial = "FAKE123"
    control = None
    stats = None

    def __init__(self, *a, **kw) -> None:
        pass

    def start(self):
        raise ADBError("boom")

    def stop(self) -> None:
        pass


class _OkSession:
    serial = "FAKE123"
    control = None

    def __init__(self, *a, **kw) -> None:
        from cr_bot.adb.capture import SessionStats

        self.stats = SessionStats()

    def start(self):
        return self

    def stop(self) -> None:
        pass


def test_failing_scrcpy_degrades_to_screenrecord(monkeypatch) -> None:
    """A dead preferred backend must hand over, not abort the run."""
    client = MultiClient(backend="auto")

    def fake_new_session(adb, backend):
        if backend == "scrcpy":
            return _BoomSession()
        if backend == "screenrecord":
            return _OkSession()
        raise AssertionError("should never reach screencap")

    monkeypatch.setattr(client, "_new_session", fake_new_session)
    session = client._start_one(FakeAdb("adb"))
    assert isinstance(session, _OkSession)
    assert client.backend_used == "screenrecord"
    assert "boom" in client.fallback_reason


def test_ladder_ends_at_screencap_when_everything_fails(monkeypatch, fake_env) -> None:
    client = MultiClient(backend="auto")
    seen: List[str] = []

    def fake_new_session(adb, backend):
        seen.append(backend)
        if backend == "screencap":
            session = ScreencapSession(adb, target_width=720)

            def no_start():
                return session

            session.start = no_start  # skip the real adb round-trip
            return session
        return _BoomSession()

    monkeypatch.setattr(client, "_new_session", fake_new_session)
    session = client._start_one(FakeAdb("adb"))
    assert seen[0] == "scrcpy" and seen[-1] == "screencap"
    assert "screenrecord" in seen
    assert isinstance(session, ScreencapSession)
    assert client.backend_used == "screencap"


def test_auto_raises_only_when_every_backend_fails() -> None:
    client = MultiClient(backend="auto")

    class _Dead(_BoomSession):
        pass

    client._new_session = lambda adb, backend: _Dead()
    with pytest.raises(ADBError) as excinfo:
        client._start_one(FakeAdb("adb"))
    assert "no capture backend worked" in str(excinfo.value)
