"""Screen-fitting helpers for the OpenCV preview windows.

The bot mirrors a 1080x2400 phone. OpenCV's default ``WINDOW_AUTOSIZE`` shows images at
1:1 pixel scale, so a full-height preview is *taller than most desktop screens* and the
bottom (hand cards, elixir) - the part you actually want to watch - ends up off-screen or
behind the taskbar.

Every preview in this project therefore goes through :class:`PreviewWindow`, which asks the
desktop for its real resolution and scales the image down to fit, preserving aspect ratio.
Nothing is ever upscaled: a small window stays crisp instead of blurry.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

#: Fraction of the desktop height a preview may occupy.  The rest is left for the window
#: title bar, the taskbar and a little breathing room.
DEFAULT_HEIGHT_FRACTION = 0.86
#: Fraction of the desktop width a preview may occupy.
DEFAULT_WIDTH_FRACTION = 0.92

_cached_screen: Optional[Tuple[int, int]] = None


def _screen_size_tk() -> Optional[Tuple[int, int]]:
    try:
        import tkinter  # noqa: PLC0415 - optional, only used to measure the desktop

        root = tkinter.Tk()
        root.withdraw()
        try:
            return int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001 - no tkinter / no display
        return None


def _screen_size_windows() -> Optional[Tuple[int, int]]:
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes  # noqa: PLC0415

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()  # report physical pixels, not DPI-virtualised ones
        w, h = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:  # noqa: BLE001
        return None
    return None


def _screen_size_x11() -> Optional[Tuple[int, int]]:
    if not sys.platform.startswith("linux"):
        return None
    try:
        import re
        import subprocess  # noqa: PLC0415

        out = subprocess.run(
            ["xrandr", "--current"], capture_output=True, timeout=2.0, check=False
        ).stdout.decode("utf-8", "replace")
        sizes = [
            (int(m.group(1)), int(m.group(2)))
            for m in re.finditer(r"(\d+)x(\d+)\+0\+0", out)
        ]
        if sizes:
            return max(sizes, key=lambda wh: wh[0] * wh[1])
    except Exception:  # noqa: BLE001
        return None
    return None


def screen_size() -> Tuple[int, int]:
    """Best-effort desktop resolution in physical pixels.

    Order: explicit override, then the native widget toolkit, then a sane default.  The
    override exists because window managers sometimes lie (HiDPI scaling in particular):

    ``CR_BOT_SCREEN_HEIGHT=900 python main.py ...`` forces the working height.
    """
    global _cached_screen
    if _cached_screen is not None:
        return _cached_screen

    env_h = os.environ.get("CR_BOT_SCREEN_HEIGHT")
    env_w = os.environ.get("CR_BOT_SCREEN_WIDTH")
    if env_h and env_h.isdigit():
        _cached_screen = (
            int(env_w) if env_w and env_w.isdigit() else 1920,
            int(env_h),
        )
        return _cached_screen

    for probe in (_screen_size_windows, _screen_size_tk, _screen_size_x11):
        found = probe()
        if found:
            _cached_screen = found
            return found

    _cached_screen = (1920, 1080)
    return _cached_screen


def fit_size(
    width: int,
    height: int,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
    height_fraction: float = DEFAULT_HEIGHT_FRACTION,
    width_fraction: float = DEFAULT_WIDTH_FRACTION,
) -> Tuple[int, int]:
    """Largest size <= the given limits that keeps ``width:height``.

    Returns the *input* size unchanged when it already fits, so small previews are not
    upscaled.
    """
    screen_w, screen_h = screen_size()
    if max_height is None:
        max_height = int(screen_h * height_fraction)
    if max_width is None:
        max_width = int(screen_w * width_fraction)
    max_height = max(120, int(max_height))
    max_width = max(120, int(max_width))

    if width <= max_width and height <= max_height:
        return int(width), int(height)

    scale = min(max_width / float(width), max_height / float(height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def fit_image(
    image: np.ndarray,
    max_height: Optional[int] = None,
    max_width: Optional[int] = None,
    height_fraction: float = DEFAULT_HEIGHT_FRACTION,
    width_fraction: float = DEFAULT_WIDTH_FRACTION,
) -> np.ndarray:
    """Downscale ``image`` so it fits on the desktop; never upscales."""
    if image is None:
        return image
    h, w = image.shape[:2]
    tw, th = fit_size(
        w,
        h,
        max_height=max_height,
        max_width=max_width,
        height_fraction=height_fraction,
        width_fraction=width_fraction,
    )
    if (tw, th) == (w, h):
        return image

    import cv2  # noqa: PLC0415 - only needed when a resize happens

    return cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)


class PreviewWindow:
    """An OpenCV window that always fits on screen.

    ``scale`` is the factor applied to the last shown image, so callers that accept mouse
    input (the tower picker) can map click coordinates back to frame coordinates.
    """

    def __init__(
        self,
        name: str,
        height_fraction: float = DEFAULT_HEIGHT_FRACTION,
        width_fraction: float = DEFAULT_WIDTH_FRACTION,
        resizable: bool = True,
    ) -> None:
        import cv2  # noqa: PLC0415

        self.name = name
        self.height_fraction = height_fraction
        self.width_fraction = width_fraction
        self.scale: float = 1.0
        self._size: Optional[Tuple[int, int]] = None
        cv2.namedWindow(name, cv2.WINDOW_NORMAL if resizable else cv2.WINDOW_AUTOSIZE)

    def show(self, image: np.ndarray) -> np.ndarray:
        """Display ``image`` (downscaled to fit) and return what was displayed."""
        import cv2  # noqa: PLC0415

        shown = fit_image(
            image,
            height_fraction=self.height_fraction,
            width_fraction=self.width_fraction,
        )
        h, w = shown.shape[:2]
        src_h = image.shape[0]
        self.scale = (h / float(src_h)) if src_h else 1.0
        if self._size != (w, h):
            # Set the window size once per geometry change; the image itself is already
            # scaled, so the window never has to grow beyond the screen.
            cv2.resizeWindow(self.name, w, h)
            self._size = (w, h)
        cv2.imshow(self.name, shown)
        return shown

    def close(self) -> None:
        import cv2  # noqa: PLC0415

        try:
            cv2.destroyWindow(self.name)
        except Exception:  # noqa: BLE001
            pass
