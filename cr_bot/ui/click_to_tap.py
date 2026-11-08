"""Click-to-tap: turn a mouse click in the preview window into a real touch on the phone.

Why this exists
---------------
Watching a bot play is much more useful when you can take over for one play. The preview
window already shows every frame the vision pipeline sees, so the natural thing is: click
where you want to touch. This module is the bridge — it maps a click in the *displayed*
(preview) pixels back to *capture frame* pixels, then sends the gesture to the device.

Two gestures, both of which the bot itself uses:

``tap``
    A single click anywhere on the arena is a plain press at that point (menus, the OK
    button after a match, or just poking a tower).

``deploy``
    Click a card in the hand tray, then click the arena. That is exactly the drag gesture the
    game is designed around (press the card, glide to the tile, release), so the card follows
    your cursor and the deployment preview is real. Right-click or Esc cancels.

Threading
---------
``cv2.setMouseCallback`` runs on the GUI thread, inside ``waitKey``. Injecting a gesture
takes 5-300 ms depending on the backend, and stalling the GUI thread would freeze the
preview (and the ``waitKey`` that keeps it alive). So clicks are queued and executed by a
small worker thread; the HUD stays at full frame rate even while a gesture is in flight.

Coordinate mapping
------------------
OpenCV may or may not report mouse coordinates in image or window space depending on the
backend and on whether the user resized the window. :func:`map_click_to_frame` therefore
uses ``getWindowImageRect`` (OpenCV >= 4.5.1) to learn the rectangle the image actually
occupies and scales the click with it, falling back to the value recorded when the frame was
shown. Both paths are unit tested.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Point = Tuple[int, int]


# ======================================================================================
# coordinate mapping
# ======================================================================================


def map_click_to_frame(
    click_x: float,
    click_y: float,
    shown_size: Tuple[int, int],
    frame_size: Tuple[int, int],
    image_rect: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Point]:
    """Map a click in the preview window to capture-frame pixel coordinates.

    ``shown_size`` is the size the last frame was rendered at (``PreviewWindow.scale``),
    ``frame_size`` the size of the captured frame. ``image_rect`` — ``(x, y, w, h)`` from
    ``cv2.getWindowImageRect`` when available — describes where the image actually sits
    inside the window, which differs from ``(0, 0, *shown_size)`` as soon as the user drags
    the window corner.

    Returns ``None`` for clicks outside the image (e.g. on the telemetry sidebar).
    """
    shown_w, shown_h = int(shown_size[0]), int(shown_size[1])
    frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
    if shown_w <= 0 or shown_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return None

    if image_rect is not None:
        rx, ry, rw, rh = [int(v) for v in image_rect]
    else:
        rx, ry, rw, rh = 0, 0, shown_w, shown_h
    if rw <= 0 or rh <= 0:
        rx, ry, rw, rh = 0, 0, shown_w, shown_h

    if not (rx <= click_x < rx + rw and ry <= click_y < ry + rh):
        return None

    fx = (click_x - rx) * (frame_w / float(rw))
    fy = (click_y - ry) * (frame_h / float(rh))
    fx = max(0.0, min(frame_w - 1.0, fx))
    fy = max(0.0, min(frame_h - 1.0, fy))
    return int(round(fx)), int(round(fy))


# ======================================================================================
# requests
# ======================================================================================


@dataclass
class ManualGesture:
    """One gesture the human asked for, in capture-frame pixels."""

    kind: str                      # "tap" | "deploy"
    point: Point                   # target in frame pixels
    start: Optional[Point] = None  # card slot centre, for "deploy"
    hold_ms: float = 70.0
    note: str = ""
    queued_at: float = field(default_factory=time.monotonic)


class ClickToTapController:
    """Queues human clicks and executes them on the device from a worker thread.

    Parameters
    ----------
    frame_size:
        Size of the capture frames, in pixels.
    perform:
        Callable that actually injects a :class:`ManualGesture`. In the bot this is
        ``ClashRoyaleBotRunner._perform_manual_gesture``.
    hand_slot_of:
        Callable mapping a frame point to a hand-slot index (0..3) or ``None``. Used to
        decide whether a click is "pick up this card" or "place/tap here".
    enabled:
        Whether clicks are injected at all (``--dry-run`` and ``--no-input`` set this False
        so a click only reports what it *would* do).
    """

    #: how long a picked-up card stays selected before it is dropped again
    SELECTION_TIMEOUT_S = 6.0

    def __init__(
        self,
        frame_size: Tuple[int, int],
        perform: Callable[[ManualGesture], bool],
        hand_slot_of: Optional[Callable[[Point], Optional[int]]] = None,
        enabled: bool = True,
        tap_hold_ms: float = 70.0,
    ) -> None:
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.perform = perform
        self.hand_slot_of = hand_slot_of
        self.enabled = bool(enabled)
        self.tap_hold_ms = float(tap_hold_ms)

        self._queue: "queue.Queue[ManualGesture]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.selected_slot: Optional[int] = None
        self.selected_slot_xy: Optional[Point] = None
        self.selected_at: float = 0.0
        #: last thing that happened, for the HUD
        self.last_status: str = "click = tap   |   card then arena = deploy   |   right-click cancels"
        self.last_status_ts: float = 0.0
        self.gestures_sent = 0
        self.gestures_failed = 0

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> "ClickToTapController":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._worker, name="manual-input", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_size = (int(width), int(height))

    # -- selection -----------------------------------------------------------------
    @property
    def selected_card(self) -> Optional[int]:
        if self.selected_slot is None:
            return None
        if (time.monotonic() - self.selected_at) > self.SELECTION_TIMEOUT_S:
            self.cancel()
            return None
        return self.selected_slot

    def cancel(self) -> None:
        if self.selected_slot is not None:
            self.selected_slot = None
            self.selected_slot_xy = None
            self._set_status("selection cancelled")

    def _set_status(self, text: str) -> None:
        self.last_status = text
        self.last_status_ts = time.monotonic()
        logger.info("[manual] %s", text)

    # -- the actual mouse callback ------------------------------------------------
    def handle_click(self, x: int, y: int, button: str = "left", in_arena: Optional[bool] = None) -> Optional[ManualGesture]:
        """Process one click already mapped to *frame* pixels.

        ``button`` is ``"left"`` or ``"right"``; ``in_arena`` is an optional hint from the
        caller (the HUD knows where the arena is); when it is ``False`` a click is a plain
        tap even if it happens to land on a card.
        """
        if button == "right":
            self.cancel()
            return None

        point = (int(x), int(y))
        slot = None
        if in_arena is not True and self.hand_slot_of is not None:
            slot = self.hand_slot_of(point)

        if slot is not None and self.selected_slot is None:
            self.selected_slot = int(slot)
            self.selected_slot_xy = point
            self.selected_at = time.monotonic()
            self._set_status(f"picked up card {slot} — click the arena to deploy it")
            return None

        if self.selected_slot is not None and self.selected_slot_xy is not None:
            gesture = ManualGesture(
                kind="deploy",
                point=point,
                start=self.selected_slot_xy,
                note=f"manual deploy of slot {self.selected_slot}",
            )
            self._set_status(
                f"deploying card {self.selected_slot} -> {point}"
            )
            self.selected_slot = None
            self.selected_slot_xy = None
        else:
            gesture = ManualGesture(
                kind="tap", point=point, hold_ms=self.tap_hold_ms, note="manual tap"
            )
            self._set_status(f"tap at {point}")

        self._enqueue(gesture)
        return gesture

    def _enqueue(self, gesture: ManualGesture) -> None:
        if not self.enabled:
            logger.info("[dry-run] would %s at %s", gesture.kind, gesture.point)
            return
        try:
            self._queue.put_nowait(gesture)
        except queue.Full:
            self._set_status("input queue full — click ignored")

    # -- worker --------------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                gesture = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                ok = bool(self.perform(gesture))
            except Exception as exc:  # noqa: BLE001 - never kill the thread on one bad gesture
                logger.warning("manual gesture failed: %s", exc)
                ok = False
            if ok:
                self.gestures_sent += 1
            else:
                self.gestures_failed += 1
                self._set_status(f"{gesture.kind} at {gesture.point} did not register")

    # -- HUD helpers ---------------------------------------------------------------
    def hud_line(self) -> str:
        age = time.monotonic() - self.last_status_ts
        text = self.last_status if age < 4.0 else (
            "click = tap   |   card then arena = deploy   |   right-click cancels"
        )
        if not self.enabled:
            return "MANUAL INPUT OFF (dry-run)"
        return text

    def render_overlay(self, image: np.ndarray, image_rect: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Draw the selection box / hint onto the HUD image (in place)."""
        import cv2

        h, w = image.shape[:2]
        color = (80, 220, 255)
        cv2.putText(
            image,
            self.hud_line()[:78],
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        if self.selected_slot is not None and self.selected_slot_xy is not None:
            sx, sy = self.selected_slot_xy
            cv2.circle(image, (sx, sy), 16, color, 2)
            cv2.putText(
                image,
                "ARMED",
                (max(0, sx - 24), max(14, sy - 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
