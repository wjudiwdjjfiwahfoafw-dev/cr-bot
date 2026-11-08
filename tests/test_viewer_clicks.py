"""A click in the viewer window must become a real tap on the phone.

This is an end-to-end path and it has four joints that can each swallow a click silently:

* the OpenCV mouse callback (a *double* click delivers ``EVENT_LBUTTONDBLCLK`` only, with no
  ``LBUTTONDOWN`` — on some Qt/GTK builds that is the only event a fast click ever produces);
* the mapping from window pixels to capture-frame pixels, once the window has been scaled to
  fit the desktop and possibly resized by the user;
* the sidebar: the HUD canvas is the frame *plus* a telemetry panel, so a click on the panel
  must not be teleported into the right-hand edge of the arena;
* the hand tray: the first click picks a card up, the second places it.

The tests drive the same functions the runner's mouse callback uses, so a break in any joint
fails here instead of on the user's phone.
"""

from __future__ import annotations

import numpy as np
import pytest

from cr_bot.ui.click_to_tap import ClickToTapController, ManualGesture, map_click_to_frame


# ======================================================================================
# window pixels -> frame pixels
# ======================================================================================

def test_a_click_maps_through_the_display_scale() -> None:
    # a 720x1600 frame shown at 324x720 (45%)
    point = map_click_to_frame(162, 360, (324, 720), (720, 1600))
    assert point == (360, 800)


def test_a_click_maps_through_a_resized_window_with_letterboxing() -> None:
    # the image occupies (10, 20, 324x720) inside a larger window
    point = map_click_to_frame(10 + 162, 20 + 360, (324, 720), (720, 1600), (10, 20, 324, 720))
    assert point == (360, 800)


def test_a_click_outside_the_image_is_rejected() -> None:
    assert map_click_to_frame(5, 5, (324, 720), (720, 1600), (10, 20, 324, 720)) is None


def test_the_mapping_never_leaves_the_frame() -> None:
    point = map_click_to_frame(10_000, 10_000, (324, 720), (720, 1600), (0, 0, 324, 720))
    assert point is None, "a click outside the image rect cannot be inside the frame"


def test_a_zero_sized_window_does_not_crash() -> None:
    assert map_click_to_frame(0, 0, (0, 0), (720, 1600)) is None


# ======================================================================================
# click -> gesture
# ======================================================================================

class _Recorder:
    def __init__(self) -> None:
        self.gestures = []

    def __call__(self, gesture: ManualGesture) -> bool:
        self.gestures.append(gesture)
        return True


def _controller(slots=None, enabled=True):
    rec = _Recorder()
    ctrl = ClickToTapController(
        frame_size=(720, 1600),
        perform=rec,
        hand_slot_of=slots,
        enabled=enabled,
    )
    return ctrl, rec


def _drain(ctrl: ClickToTapController) -> None:
    """Run the worker synchronously: the tests must not depend on thread timing."""
    while not ctrl._queue.empty():
        gesture = ctrl._queue.get_nowait()
        ctrl.perform(gesture)


def test_a_single_click_is_a_tap_at_that_point() -> None:
    ctrl, rec = _controller()
    ctrl.handle_click(200, 900, button="left", in_arena=True)
    _drain(ctrl)
    assert len(rec.gestures) == 1
    assert rec.gestures[0].kind == "tap"
    assert rec.gestures[0].point == (200, 900)


def test_card_then_arena_is_a_deploy() -> None:
    slots = {0: (224, 1466), 1: (360, 1466)}      # click -> slot index
    ctrl, rec = _controller(
        slots=lambda p: (0 if abs(p[0] - 224) < 60 else 1 if abs(p[0] - 360) < 60 else None)
    )
    first = ctrl.handle_click(224, 1466)          # pick up card 0
    assert first is None, "picking a card up must not send a gesture on its own"
    assert ctrl.selected_slot == 0
    second = ctrl.handle_click(300, 800)          # place it
    _drain(ctrl)
    assert second is not None and second.kind == "deploy"
    assert second.start == (224, 1466), "the deploy must start on the card the user picked"
    assert second.point == (300, 800)
    assert len(rec.gestures) == 1


def test_right_click_cancels_a_pickup() -> None:
    ctrl, rec = _controller(
        slots=lambda p: 0 if abs(p[0] - 224) < 60 else None
    )
    ctrl.handle_click(224, 1466)
    assert ctrl.selected_slot == 0
    assert ctrl.handle_click(400, 400, button="right") is None
    assert ctrl.selected_slot is None
    _drain(ctrl)
    assert rec.gestures == [], "a cancel must not send anything to the phone"


def test_a_click_on_the_card_area_while_selected_still_deploys() -> None:
    """The second click is a placement even if it lands near the tray."""
    ctrl, rec = _controller(slots=lambda p: 0)
    ctrl.handle_click(224, 1466)
    ctrl.handle_click(250, 1400)                  # still inside the tray region
    _drain(ctrl)
    assert rec.gestures and rec.gestures[0].kind == "deploy"


def test_dry_run_reports_but_sends_nothing() -> None:
    ctrl, rec = _controller(enabled=False)
    ctrl.handle_click(200, 900, button="left", in_arena=True)
    _drain(ctrl)
    assert rec.gestures == []


def test_the_controller_can_be_configured_for_a_new_resolution() -> None:
    ctrl, rec = _controller()
    ctrl.set_frame_size(1080, 2400)
    assert ctrl.frame_size == (1080, 2400)
    ctrl.handle_click(500, 1200, button="left", in_arena=True)
    _drain(ctrl)
    assert rec.gestures[0].point == (500, 1200)


# ======================================================================================
# the runner's mouse callback, as a unit
# ======================================================================================

def _bare_runner():
    """The runner without its constructor, for testing one method in isolation."""
    from cr_bot.bot_runner import ClashRoyaleBotRunner

    runner = object.__new__(ClashRoyaleBotRunner)
    runner._click_binding_frame = None
    return runner


class _FakePreview:
    name = "Clash Royale Bot - MK Zap Bait"

    def __init__(self, shown, rect):
        self.shown_size = shown
        self._rect = rect

    def image_rect(self):
        return self._rect


def test_the_sidebar_is_not_part_of_the_arena() -> None:
    """A click on the telemetry panel must be ignored, not turned into a tap."""
    from cr_bot.bot_runner import ClashRoyaleBotRunner

    canvas = (720 + 440, 1600)          # frame + HUD panel
    preview = _FakePreview(shown=canvas, rect=(0, 0, *canvas))
    ctrl, rec = _controller()

    captured = {}
    import cv2
    original = cv2.setMouseCallback
    cv2.setMouseCallback = lambda name, fn: captured.__setitem__("fn", fn)
    try:
        ClashRoyaleBotRunner._bind_mouse(_bare_runner(), preview, ctrl, canvas)
    finally:
        cv2.setMouseCallback = original
    fn = captured["fn"]

    fn(cv2.EVENT_LBUTTONDOWN, 800, 600, 0, None)      # inside the sidebar
    _drain(ctrl)
    assert rec.gestures == [], "the sidebar is not the game"

    fn(cv2.EVENT_LBUTTONDOWN, 300, 600, 0, None)      # inside the frame
    _drain(ctrl)
    assert len(rec.gestures) == 1
    assert rec.gestures[0].point == (300, 600)


def test_a_double_click_is_accepted_too() -> None:
    """Some Qt/GTK builds only deliver LBUTTONDBLCLK for a fast click."""
    from cr_bot.bot_runner import ClashRoyaleBotRunner
    import cv2

    canvas = (720, 1600)
    preview = _FakePreview(shown=canvas, rect=(0, 0, *canvas))
    ctrl, rec = _controller()
    captured = {}
    original = cv2.setMouseCallback
    cv2.setMouseCallback = lambda name, fn: captured.__setitem__("fn", fn)
    try:
        ClashRoyaleBotRunner._bind_mouse(_bare_runner(), preview, ctrl, canvas)
    finally:
        cv2.setMouseCallback = original
    captured["fn"](cv2.EVENT_LBUTTONDBLCLK, 360, 800, 0, None)
    _drain(ctrl)
    assert len(rec.gestures) == 1, "a double click must not be dropped"
    assert rec.gestures[0].point == (360, 800)


def test_the_callback_is_rebound_when_the_window_geometry_changes() -> None:
    from cr_bot.bot_runner import ClashRoyaleBotRunner
    import cv2

    runner = _bare_runner()
    preview = _FakePreview(shown=(720, 1600), rect=(0, 0, 720, 1600))
    ctrl, _rec = _controller()
    calls = []
    original = cv2.setMouseCallback
    cv2.setMouseCallback = lambda name, fn: calls.append(name)
    try:
        ClashRoyaleBotRunner._bind_mouse(runner, preview, ctrl, (720, 1600))
        ClashRoyaleBotRunner._bind_mouse(runner, preview, ctrl, (720, 1600))
        assert len(calls) == 1, "the callback is installed once per geometry, not per frame"
        ClashRoyaleBotRunner._bind_mouse(runner, preview, ctrl, (1160, 1600))
        assert len(calls) == 2, "a new canvas size must re-bind, or clicks map to the old scale"
    finally:
        cv2.setMouseCallback = original
