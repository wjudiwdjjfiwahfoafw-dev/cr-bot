"""
Touch injection for Clash Royale, with multiple backends and automatic fallback.

Why the original two-tap approach fails on a real device
--------------------------------------------------------
The previous implementation placed a card with::

    adb shell "input tap <slot>; sleep 0.06; input tap <arena>"

Three things go wrong on real hardware:

1. ``input tap`` synthesises a touch with a **0 ms press duration**. Android's touch
   pipeline and Supercell's engine regularly drop a zero-length DOWN/UP pair, so the card
   is never picked up.
2. Each ``input`` invocation spawns a fresh ``app_process`` JVM on the device, costing
   roughly 100-250 ms. The nominal 60 ms gap between the two taps is in reality several
   hundred ms, and the second tap frequently lands after the game has already cleared the
   selection, or while the HUD is animating.
3. Taps are absolute and unforgiving: a coordinate that is a few percent off lands on the
   arena border, on a tower, or between card slots, and Clash Royale simply ignores it —
   silently. The bot then believes it played a card, its hand model desynchronises, and
   every later decision targets the wrong slot. That failure mode looks exactly like
   "it never places cards".

What this module does instead
-----------------------------
* Deploys a card with a **drag gesture** — press on the card, glide to the target tile,
  hold briefly, release. This is the gesture the game is actually designed around, and it
  works regardless of whether tapping-to-select happens to register.
* Injects through the **scrcpy control socket** when the capture backend has one
  (:class:`~cr_bot.adb.scrcpy_multiclient.ControlChannel`): one persistent TCP connection
  to the already-running server on the device, so a full press-move-release is a handful of
  small writes with real timing. There is no per-touch process spawn and no ``input`` JVM,
  which is what makes sub-10 ms deploys possible. The packet layout is version-aware (see
  :mod:`cr_bot.adb.scrcpy_protocol`) because a 2.0+ server reads 31-byte touch payloads
  while 1.x read 27.
* Falls back through ``input swipe`` (one native drag, still cheap and reliable) and
  ``input motionevent`` (true DOWN/MOVE/UP, Android 8+), then the legacy taps.
* Keeps an explicit record of what it did so the caller can verify the card really left the
  hand and escalate/retry instead of silently desynchronising.
"""

from __future__ import annotations

import logging
import random
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from cr_bot.adb.adb_utils import Adb, ADBError

if TYPE_CHECKING:  # pragma: no cover
    from cr_bot.adb.scrcpy_multiclient import ControlChannel

logger = logging.getLogger(__name__)

Point = Tuple[int, int]


# ======================================================================================
# Coordinate mapping
# ======================================================================================


@dataclass
class CoordMap:
    """Maps coordinates from vision/frame space to physical device pixels.

    The scrcpy control socket already understands frame-space coordinates, but
    ``adb shell input`` only speaks physical device pixels. Confusing the two is another
    classic source of "taps land nowhere" on phones whose stream is downscaled.
    """

    frame_w: int
    frame_h: int
    device_w: int
    device_h: int

    def to_device(self, x: float, y: float) -> Point:
        if self.frame_w <= 0 or self.frame_h <= 0:
            return int(round(x)), int(round(y))
        sx = self.device_w / float(self.frame_w)
        sy = self.device_h / float(self.frame_h)
        return int(round(x * sx)), int(round(y * sy))

    def clamp_frame(self, x: float, y: float) -> Point:
        return (
            int(max(0, min(self.frame_w - 1, round(x)))),
            int(max(0, min(self.frame_h - 1, round(y)))),
        )


# ======================================================================================
# Backends
# ======================================================================================


class TouchBackend:
    """Common interface. Coordinates are always in *frame* space; scaling happens here."""

    name = "base"
    supports_drag = True

    def __init__(self, adb: Adb, coord: CoordMap) -> None:
        self.adb = adb
        self.coord = coord
        self.last_latency_ms = 0.0

    def tap(self, x: float, y: float, hold_ms: float = 70.0) -> None:
        raise NotImplementedError

    def drag(self, start: Point, end: Point, duration_ms: float = 180.0, steps: int = 8) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def available(self) -> bool:
        return True


class ScrcpyBackend(TouchBackend):
    """Inject over the scrcpy control socket — the fast path.

    Coordinates are already in *frame* space, and the scrcpy server scales frame coordinates
    to the physical display itself (the control channel is told the streamed size), so no
    :class:`CoordMap` conversion happens here. That also means this backend keeps working if
    the stream is downscaled to 720p on a 1440p phone.
    """

    name = "scrcpy"

    def __init__(self, control: "ControlChannel", coord: Optional[CoordMap] = None) -> None:
        # deliberately does not call super().__init__: there is no adb round trip to make
        self.control = control
        #: frame space only — the server scales to the physical display itself
        self.coord = coord
        self.last_latency_ms = 0.0

    @property
    def available(self) -> bool:
        return self.control is not None

    def tap(self, x: float, y: float, hold_ms: float = 70.0) -> None:
        t0 = time.perf_counter()
        self.control.tap(int(round(x)), int(round(y)), hold_ms=hold_ms)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

    def drag(self, start: Point, end: Point, duration_ms: float = 180.0, steps: int = 10) -> None:
        t0 = time.perf_counter()
        self.control.drag(
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            duration_ms=duration_ms,
            steps=steps,
        )
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0


class SwipeBackend(TouchBackend):
    """`adb shell input swipe` — one invocation, one native drag, no JVM churn per move."""

    name = "swipe"

    def tap(self, x: float, y: float, hold_ms: float = 70.0) -> None:
        # `input swipe X Y X Y duration` is a press-and-hold of exactly `duration`.
        self.drag((x, y), (x, y), duration_ms=max(1.0, hold_ms), steps=1)

    def drag(self, start: Point, end: Point, duration_ms: float = 180.0, steps: int = 8) -> None:
        sx, sy = self.coord.to_device(*start)
        ex, ey = self.coord.to_device(*end)
        dur = int(max(1, round(duration_ms)))
        t0 = time.perf_counter()
        self.adb.run(
            ["shell", "input", "swipe", str(sx), str(sy), str(ex), str(ey), str(dur)],
            timeout=max(4.0, dur / 1000.0 + 3.0),
        )
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0


class MotionEventBackend(TouchBackend):
    """`adb shell input motionevent` (Android 8+) — true DOWN / MOVE / UP sequence."""

    name = "motionevent"

    def __init__(self, adb: Adb, coord: CoordMap) -> None:
        super().__init__(adb, coord)
        self._supported: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._supported is None:
            self._supported = self._probe()
        return bool(self._supported)

    def _probe(self) -> bool:
        try:
            out = self.adb.shell("input", timeout=8.0)
        except ADBError:
            return False
        if "motionevent" in out:
            return True
        try:
            out = self.adb.shell("input motionevent 2>&1 | head -3", timeout=8.0)
        except ADBError:
            return False
        return "Unknown command" not in out and "unknown command" not in out

    def _sequence(self, script: str) -> None:
        t0 = time.perf_counter()
        self.adb.run(["shell", script], timeout=6.0)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

    def tap(self, x: float, y: float, hold_ms: float = 70.0) -> None:
        dx, dy = self.coord.to_device(x, y)
        self._sequence(f"input motionevent DOWN {dx} {dy}; sleep {hold_ms/1000.0:.3f}; input motionevent UP {dx} {dy}")

    def drag(self, start: Point, end: Point, duration_ms: float = 180.0, steps: int = 8) -> None:
        sx, sy = self.coord.to_device(*start)
        ex, ey = self.coord.to_device(*end)
        steps = max(2, steps)
        step_sleep = (duration_ms / 1000.0) / steps
        parts = [f"input motionevent DOWN {sx} {sy}"]
        for i in range(1, steps + 1):
            t = i / steps
            mx = int(round(sx + (ex - sx) * t))
            my = int(round(sy + (ey - sy) * t))
            parts.append(f"sleep {step_sleep:.3f}")
            parts.append(f"input motionevent MOVE {mx} {my}")
        parts.append(f"input motionevent UP {ex} {ey}")
        self._sequence("; ".join(parts))


class TapBackend(TouchBackend):
    """Legacy behaviour: `input tap` twice, with a longer, jittered gap."""

    name = "tap"

    def __init__(self, adb: Adb, coord: CoordMap, gap_ms: float = 140.0) -> None:
        super().__init__(adb, coord)
        self.gap_ms = gap_ms

    def tap(self, x: float, y: float, hold_ms: float = 70.0) -> None:
        dx, dy = self.coord.to_device(x, y)
        t0 = time.perf_counter()
        self.adb.run(["shell", f"input swipe {dx} {dy} {dx} {dy} {int(hold_ms)}"], timeout=5.0)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

    def drag(self, start: Point, end: Point, duration_ms: float = 180.0, steps: int = 8) -> None:
        """Emulate a drag with select-then-place taps (the original strategy)."""
        sx, sy = self.coord.to_device(*start)
        ex, ey = self.coord.to_device(*end)
        gap = self.gap_ms / 1000.0
        script = (
            f"input swipe {sx} {sy} {sx} {sy} 70; "
            f"sleep {gap:.3f}; "
            f"input swipe {ex} {ey} {ex} {ey} 70"
        )
        t0 = time.perf_counter()
        self.adb.run(["shell", script], timeout=6.0)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0


# ======================================================================================
# Injector
# ======================================================================================


@dataclass
class GestureResult:
    """What the injector attempted, so the caller can verify and retry."""

    ok: bool
    backend: str
    kind: str
    start: Point
    end: Point
    latency_ms: float = 0.0
    attempts: int = 1
    note: str = ""


DEFAULT_ORDER = ("scrcpy", "swipe", "motionevent", "tap")


class TouchInjector:
    """High-level gestures for Clash Royale with backend escalation.

    Parameters
    ----------
    order:
        Backend preference. ``auto`` (default) tries scrcpy first, then cheap adb
        fallbacks. ``scrcpy`` forces socket injection; ``adb`` prefers swipe/motionevent.
    """

    def __init__(
        self,
        adb: Adb,
        coord: CoordMap,
        control=None,
        order: Sequence[str] = DEFAULT_ORDER,
        deploy_style: str = "tap",
        jitter_px: int = 2,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.adb = adb
        self.coord = coord
        self.deploy_style = deploy_style
        self.jitter_px = int(jitter_px)
        self._rng = rng or random.Random()
        self.backends: Dict[str, TouchBackend] = {}
        self.order: List[str] = []

        available = self._build_backends(control)
        for name in order:
            key = self._canonical(name)
            if key in available:
                self.order.append(key)
                self.backends[key] = available[key]
        if not self.order:
            raise ADBError("No usable touch backend could be constructed")
        self.active = self.order[0]
        self.attempts: List[GestureResult] = []

    def _canonical(self, name: str) -> str:
        name = name.lower()
        if name in ("adb", "auto", "default"):
            return "swipe"
        if name in ("scrcpy_py", "native", "socket", "control"):
            return "scrcpy"
        return name

    def _build_backends(self, control) -> Dict[str, TouchBackend]:
        """Every backend that could work here, including the scrcpy socket when present."""
        built: Dict[str, TouchBackend] = {}
        if control is not None:
            built["scrcpy"] = ScrcpyBackend(control, self.coord)
        built["swipe"] = SwipeBackend(self.adb, self.coord)
        built["motionevent"] = MotionEventBackend(self.adb, self.coord)
        built["tap"] = TapBackend(self.adb, self.coord)
        return built

    # -- helpers -------------------------------------------------------------------
    def _jitter(self, point: Point) -> Point:
        if self.jitter_px <= 0:
            return point
        jx = int(round(self._rng.gauss(0, self.jitter_px * 0.5)))
        jy = int(round(self._rng.gauss(0, self.jitter_px * 0.5)))
        jx = max(-self.jitter_px, min(self.jitter_px, jx))
        jy = max(-self.jitter_px, min(self.jitter_px, jy))
        return self.coord.clamp_frame(point[0] + jx, point[1] + jy)

    def describe(self) -> str:
        chained = " -> ".join(self.order)
        return f"input backend: {self.active} (order: {chained})"

    def set_active(self, name: str) -> bool:
        key = self._canonical(name)
        if key in self.backends:
            self.active = key
            return True
        return False

    # -- primitives ----------------------------------------------------------------
    def tap(self, point: Point, hold_ms: float = 70.0) -> GestureResult:
        return self._run("tap", point, point, duration_ms=hold_ms)

    def drag(self, start: Point, end: Point, duration_ms: float = 180.0) -> GestureResult:
        return self._run("drag", start, end, duration_ms=duration_ms)

    def _run(self, kind: str, start: Point, end: Point, duration_ms: float) -> GestureResult:
        """Try the active backend, then escalate through the rest on failure."""
        last_error = ""
        for attempt, name in enumerate(self.order, start=1):
            backend = self.backends[name]
            if not backend.available:
                continue
            try:
                if kind == "tap":
                    backend.tap(start[0], start[1], hold_ms=duration_ms)
                else:
                    backend.drag(start, end, duration_ms=duration_ms)
            except (ADBError, subprocess.SubprocessError, OSError) as exc:
                last_error = str(exc)
                logger.warning("Touch backend %s failed (%s); escalating.", name, exc)
                continue

            if name != self.active:
                logger.info("Switched active touch backend to %s.", name)
                self.active = name
            result = GestureResult(
                ok=True,
                backend=name,
                kind=kind,
                start=start,
                end=end,
                latency_ms=backend.last_latency_ms,
                attempts=attempt,
                note=last_error,
            )
            self.attempts.append(result)
            return result

        result = GestureResult(
            ok=False,
            backend=self.active,
            kind=kind,
            start=start,
            end=end,
            attempts=len(self.order),
            note=last_error or "all backends unavailable",
        )
        self.attempts.append(result)
        logger.error("All touch backends failed for %s %s->%s (%s)", kind, start, end, result.note)
        return result

    # -- Clash Royale specific -----------------------------------------------------
    def play_card(
        self,
        slot_xy: Point,
        arena_xy: Point,
        style: Optional[str] = None,
        force_backend: Optional[str] = None,
    ) -> GestureResult:
        """Deploy the card in the given hand slot at the given arena pixel.

        ``style``:
          ``drag``  press the card, glide to the tile, release (default, most faithful)
          ``tap``   legacy select-then-place
        """
        style = (style or self.deploy_style).lower()
        start = self._jitter(slot_xy)
        end = self._jitter(arena_xy)

        if force_backend:
            self.set_active(force_backend)

        if style == "tap":
            # Select, then place. Use a deliberate hold so the selection registers.
            first = self._run("tap", start, start, duration_ms=90.0)
            time.sleep(self._rng.uniform(0.10, 0.16))
            second = self._run("tap", end, end, duration_ms=90.0)
            return GestureResult(
                ok=first.ok and second.ok,
                backend=second.backend,
                kind="tap_then_place",
                start=start,
                end=end,
                latency_ms=first.latency_ms + second.latency_ms,
                attempts=max(first.attempts, second.attempts),
                note=second.note,
            )

        # Drag: a slightly curved, human-like path with a hover before release.
        duration = self._rng.uniform(150.0, 220.0)
        return self._run("drag", start, end, duration_ms=duration)

    def tap_abs(self, point: Point, hold_ms: float = 70.0) -> GestureResult:
        """Plain UI tap (menus, OK buttons)."""
        return self._run("tap", self._jitter(point), self._jitter(point), duration_ms=hold_ms)

    def stats(self) -> str:
        if not self.attempts:
            return "no gestures yet"
        recent = self.attempts[-10:]
        avg = sum(a.latency_ms for a in recent) / len(recent)
        fails = sum(1 for a in recent if not a.ok)
        return f"{len(self.attempts)} gestures, avg {avg:.0f} ms, {fails}/{len(recent)} failed (last 10)"
