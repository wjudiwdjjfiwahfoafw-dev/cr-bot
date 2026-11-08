"""Tests for the screen-fitting preview helpers.

The bot mirrors a 1080x2400 phone; showing that 1:1 in an OpenCV window puts the hand
cards and elixir bar below the bottom of a 1080p screen. ``fit_size``/``fit_image`` scale
previews down to the desktop and never upscale.
"""

from __future__ import annotations

import numpy as np
import pytest

from cr_bot import ui_window


@pytest.fixture(autouse=True)
def _fixed_screen(monkeypatch):
    monkeypatch.setattr(ui_window, "_cached_screen", (1920, 1080))
    yield


def test_tall_phone_frame_is_scaled_to_fit_the_screen() -> None:
    width, height = ui_window.fit_size(1080, 2400)
    assert height <= int(1080 * ui_window.DEFAULT_HEIGHT_FRACTION)
    assert width <= int(1920 * ui_window.DEFAULT_WIDTH_FRACTION)


def test_aspect_ratio_is_preserved() -> None:
    width, height = ui_window.fit_size(1080, 2400)
    assert abs((width / height) - (1080 / 2400)) < 0.02


def test_small_previews_are_never_upscaled() -> None:
    assert ui_window.fit_size(480, 320) == (480, 320)


def test_fit_image_returns_a_resized_copy() -> None:
    image = np.zeros((2400, 1080, 3), np.uint8)
    fitted = ui_window.fit_image(image)
    assert fitted.shape[0] <= 1080
    assert fitted.shape[1] <= 1920
    assert fitted.shape[0] < image.shape[0]


def test_fit_image_keeps_small_images_untouched() -> None:
    image = np.zeros((300, 200, 3), np.uint8)
    assert ui_window.fit_image(image) is image


def test_explicit_limits_win() -> None:
    assert ui_window.fit_size(1000, 2000, max_height=500) == (250, 500)


def test_environment_override_for_screen_height(monkeypatch) -> None:
    monkeypatch.setattr(ui_window, "_cached_screen", None)
    monkeypatch.setenv("CR_BOT_SCREEN_HEIGHT", "900")
    monkeypatch.setenv("CR_BOT_SCREEN_WIDTH", "1600")
    try:
        assert ui_window.screen_size() == (1600, 900)
    finally:
        monkeypatch.setattr(ui_window, "_cached_screen", None)


def test_screen_size_has_a_sane_default(monkeypatch) -> None:
    monkeypatch.setattr(ui_window, "_cached_screen", None)
    monkeypatch.delenv("CR_BOT_SCREEN_HEIGHT", raising=False)
    monkeypatch.setattr(ui_window, "_screen_size_windows", lambda: None)
    monkeypatch.setattr(ui_window, "_screen_size_tk", lambda: None)
    monkeypatch.setattr(ui_window, "_screen_size_x11", lambda: None)
    try:
        width, height = ui_window.screen_size()
        assert width >= 640 and height >= 480
    finally:
        monkeypatch.setattr(ui_window, "_cached_screen", None)
