"""
Scrcpy wire protocol — version-aware packet builders, stream parsing, H.264 decoding.

This module is dependency-light on purpose: it talks the raw protocol so the bot does not
depend on any particular (and frequently stale) third-party scrcpy wrapper.

Touch payload differs between server generations, and getting it wrong is silent: the
server's control reader throws or mis-parses every later packet, so the screen mirrors
perfectly and *nothing* can ever be tapped.

``1.x`` (**27 bytes**), ``Controller.java`` of v1.24::

    action(B) pointerId(q) x(i) y(i) screenW(H) screenH(H) pressure(H) buttons(i)

``2.0+`` (**31 bytes**), ``ControlMessageReader.parseInjectTouchEvent()`` of v5.0.1::

    action(B) pointerId(q) x(i) y(i) screenW(H) screenH(H) pressure(H)
    actionButton(i) buttons(i)

Note the extra ``actionButton`` int, and that ``buttons`` moved to the end. Widely used
Python wrappers hard-code one of the two layouts, which is why "scrcpy mirrors but never
taps" is such a common bug report. :func:`build_touch` takes the server version and emits
the right one, so the bundled v5.0.1 jar can drive input instead of only video.

``screenW``/``screenH`` must describe the coordinate space of the values you send (i.e.
the *video frame* size you are streaming). The server scales them to the physical display
via ``getEventPointAndDisplayId()``, so a downscaled 720p stream can still be tapped in
frame coordinates exactly as the vision pipeline measured them.

A finger touch must leave ``actionButton`` and ``buttons`` at 0: the server treats any
non-zero button as a *mouse*, and Clash Royale ignores a mouse drag on the card tray.
"""

from __future__ import annotations

import re
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Server version / socket name
# --------------------------------------------------------------------------------------

SERVER_VERSION = "1.24"
SOCKET_NAME = "scrcpy"

#: scrcpy changed its server argument set and handshake in v2.0 (audio forwarding, stream meta,
#: codec metadata). Everything from v2.0 on is handled by the "modern" path below.
MODERN_PROTOCOL_MAJOR = 2


def parse_server_version(name: str) -> Tuple[int, int, int]:
    """Extract a version from a server jar filename such as ``scrcpy-server-v5.0.1.jar``."""
    match = re.search(r"v?(\d+)\.(\d+)(?:\.(\d+))?", name or "")
    if not match:
        return (0, 0, 0)
    parts = [int(g) if g is not None else 0 for g in match.groups()]
    return (parts[0], parts[1], parts[2] if len(parts) > 2 else 0)


def is_modern_version(version: Tuple[int, int, int]) -> bool:
    return version[0] >= MODERN_PROTOCOL_MAJOR


def build_options(
    version: Tuple[int, int, int],
    bitrate: int,
    max_size: int,
    max_fps: int,
    lock_orientation: int = -1,
    stay_awake: bool = True,
    video_codec: str = "h264",
    video_encoder: Optional[str] = None,
    display_id: int = 0,
    scid: int = -1,
) -> Dict[str, str]:
    """Server arguments for this jar version, asking for a *raw* Annex-B stream.

    Both families are configured to send the device name (it is the window title and a useful
    liveness check) followed by a header-less elementary stream, which is exactly what
    :class:`H264FrameDecoder` wants.

    Legacy (1.x)::

        log_level max_size bit_rate max_fps lock_video_orientation tunnel_forward control
        display_id show_touches stay_awake clipboard_autosync send_frame_meta send_device_meta
        send_dummy_byte

    Modern (2.x+) — verified against ``Options.java`` of v5.0.1::

        scid log_level video audio video_codec video_bit_rate max_size max_fps angle
        tunnel_forward control display_id show_touches stay_awake clipboard_autosync
        send_device_meta send_frame_meta send_dummy_byte send_stream_meta

    Two traps on modern servers, both fatal and both silent:

    * ``audio`` defaults to **true**, so it must be explicitly disabled or the server blocks
      waiting for an audio socket that the bot never opens;
    * ``send_stream_meta`` defaults to **true**, which prefixes the stream with a 4-byte codec
      id and a 12-byte session header — feeding that to the H.264 parser yields no frames.
    """
    common = {
        "log_level": "info",
        "max_size": max_size,
        "max_fps": max_fps,
        "tunnel_forward": "true",
        "control": "true",
        "display_id": display_id,
        "show_touches": "false",
        "stay_awake": "true" if stay_awake else "false",
        "clipboard_autosync": "false",
        "send_device_meta": "true",
        "send_frame_meta": "false",
        "send_dummy_byte": "true",
    }
    if not is_modern_version(version):
        options = dict(common)
        options["bit_rate"] = bitrate
        options["lock_video_orientation"] = lock_orientation
        return options

    options = dict(common)
    options.update(
        {
            # -1 keeps the classic `localabstract:scrcpy` name; anything else is hex on the
            # wire because the server parses it with Integer.parseInt(value, 0x10)
            "scid": -1 if scid < 0 else format(int(scid) & 0xFFFFFFFF, "x"),
            "video": "true",
            "audio": "false",          # no audio socket: we only need the picture
            "video_codec": video_codec,
            "video_bit_rate": bitrate,
            "send_stream_meta": "false",  # no codec id / session header before the stream
            "cleanup": "true",
        }
    )
    if video_encoder:
        options["video_encoder"] = video_encoder
    return options


# Control message types (ControlMessage.java)
TYPE_INJECT_KEYCODE = 0
TYPE_INJECT_TEXT = 1
TYPE_INJECT_TOUCH_EVENT = 2
TYPE_INJECT_SCROLL_EVENT = 3
TYPE_BACK_OR_SCREEN_ON = 4
TYPE_EXPAND_NOTIFICATION_PANEL = 5
TYPE_EXPAND_SETTINGS_PANEL = 6
TYPE_COLLAPSE_PANELS = 7
TYPE_GET_CLIPBOARD = 8
TYPE_SET_CLIPBOARD = 9
TYPE_SET_SCREEN_POWER_MODE = 10
TYPE_ROTATE_DEVICE = 11

# MotionEvent actions
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2
ACTION_CANCEL = 3

#: payload layout per server generation
#:   1.x   action(B) pointerId(q) x(i) y(i) screenW(H) screenH(H) pressure(H) buttons(i)
#:   2.0+  action(B) pointerId(q) x(i) y(i) screenW(H) screenH(H) pressure(H)
#:         actionButton(i) buttons(i)
_TOUCH_FMT_LEGACY = ">BqiiHHHi"
_TOUCH_FMT_MODERN = ">BqiiHHHii"

#: virtual finger id used by scrcpy for a synthetic touch
POINTER_ID_FINGER = 0x1234567887654321
POINTER_ID_SECOND = 0x2222222288888888
#: scrcpy's *mouse* pointer id: only this one produces mouse semantics
POINTER_ID_MOUSE = -1

MAX_PRESSURE = 0xFFFF


def touch_payload_length(version: Tuple[int, int, int] = (1, 24, 0)) -> int:
    """Payload size (excluding the 1-byte type) the given server expects."""
    return 31 if is_modern_version(version) else 27


# --------------------------------------------------------------------------------------
# Frame buffer: single-slot, always-latest, lock-free reads
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# Control packet builders
# --------------------------------------------------------------------------------------


def build_touch(
    x: int,
    y: int,
    action: int,
    screen_w: int,
    screen_h: int,
    pointer_id: int = POINTER_ID_FINGER,
    pressure: Optional[float] = None,
    buttons: int = 0,
    action_button: int = 0,
    version: Tuple[int, int, int] = (1, 24, 0),
) -> bytes:
    """Serialize a TYPE_INJECT_TOUCH_EVENT packet for the given server version.

    ``version`` selects the payload layout: 27 bytes for 1.x, 31 bytes for 2.0+.
    """
    if pressure is None:
        pressure = 0.0 if action == ACTION_UP else 1.0
    press_int = int(round(max(0.0, min(1.0, pressure)) * MAX_PRESSURE))
    x_i = int(max(0, min(int(screen_w) - 1, round(x))))
    y_i = int(max(0, min(int(screen_h) - 1, round(y))))
    head = (
        int(action) & 0xFF,
        int(pointer_id),
        x_i,
        y_i,
        int(screen_w) & 0xFFFF,
        int(screen_h) & 0xFFFF,
        press_int,
    )
    if is_modern_version(version):
        payload = struct.pack(_TOUCH_FMT_MODERN, *head, int(action_button), int(buttons))
        expected = 31
    else:
        payload = struct.pack(_TOUCH_FMT_LEGACY, *head, int(buttons))
        expected = 27
    assert len(payload) == expected, (len(payload), expected)
    return struct.pack(">B", TYPE_INJECT_TOUCH_EVENT) + payload


def build_keycode(
    keycode: int,
    action: int = ACTION_DOWN,
    repeat: int = 0,
    meta_state: int = 0,
    version: Tuple[int, int, int] = (1, 24, 0),
) -> bytes:
    """KEYCODE payload.

    1.x  ``type(B) action(B) keycode(i) repeat(i) metaState(i)``
    2.0+ ``type(B) action(B) keycode(i) repeat(i) metaState(i)`` — unchanged, but the
    action byte of 2.0+ servers is a KeyEvent action (0/1) exactly as before.
    """
    return struct.pack(
        ">BBiii",
        TYPE_INJECT_KEYCODE,
        int(action) & 0xFF,
        int(keycode),
        int(repeat),
        int(meta_state),
    )


def build_back_or_screen_on(action: int = ACTION_DOWN) -> bytes:
    return struct.pack(">BB", TYPE_BACK_OR_SCREEN_ON, int(action) & 0xFF)


def build_screen_power_mode(mode: int = 2) -> bytes:
    """Legacy (1.x) ``SET_SCREEN_POWER_MODE``: 0 = off, 2 = normal."""
    return struct.pack(">Bb", TYPE_SET_SCREEN_POWER_MODE, int(mode))


def build_set_display_power(on: bool = True) -> bytes:
    """Modern (2.0+) ``SET_DISPLAY_POWER``: a single boolean."""
    return struct.pack(">B", TYPE_SET_SCREEN_POWER_MODE) + (b"\x01" if on else b"\x00")


def build_rotate_device() -> bytes:
    return struct.pack(">B", TYPE_ROTATE_DEVICE)


# Quick self-check used by tests: the wire length must be 1 + 27.
TOUCH_PACKET_LENGTH = 28


# --------------------------------------------------------------------------------------
# Video stream header parsing
# --------------------------------------------------------------------------------------


@dataclass
class DeviceMeta:
    name: str
    video_width: int
    video_height: int

    @property
    def is_portrait(self) -> bool:
        return self.video_height >= self.video_width


def read_device_name(sock) -> str:
    """64-byte NUL-padded device name (same in every server version)."""
    raw = _recv_exact(sock, 64)
    if not raw:
        raise ConnectionError("scrcpy server closed the socket while sending the device name")
    return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")


def read_session_meta(sock) -> Tuple[int, int]:
    """Modern servers: int32 flags, int32 width, int32 height (only with ``send_stream_meta``)."""
    raw = _recv_exact(sock, 12)
    if not raw or len(raw) < 12:
        raise ConnectionError("scrcpy server closed the socket while sending session metadata")
    _flags, width, height = struct.unpack(">iII", raw)
    return int(width), int(height)


def read_legacy_video_size(sock) -> Tuple[int, int]:
    """1.x servers append the video size as two big-endian uint16 after the device name."""
    raw = _recv_exact(sock, 4)
    if not raw or len(raw) < 4:
        raise ConnectionError("scrcpy server closed the socket while sending the video size")
    width, height = struct.unpack(">HH", raw)
    return int(width), int(height)


def read_dummy_byte(sock) -> bool:
    """Server writes one 0x00 byte as soon as the video socket is accepted."""
    data = _recv_exact(sock, 1)
    return bool(data) and data[0] == 0


def read_device_meta(sock) -> DeviceMeta:
    """64-byte NUL padded device name followed by big-endian uint16 width and height."""
    raw_name = _recv_exact(sock, 64)
    rest = _recv_exact(sock, 4)
    if not raw_name or not rest:
        raise ConnectionError("scrcpy server closed the socket while sending device metadata")
    w, h = struct.unpack(">HH", rest)
    name = raw_name.split(b"\x00", 1)[0].decode("utf-8", "replace")
    return DeviceMeta(name=name, video_width=w, video_height=h)


def _recv_exact(sock, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# --------------------------------------------------------------------------------------
# H.264 decoding
# --------------------------------------------------------------------------------------


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
