"""
Tile <-> pixel projection for the Clash Royale arena.

The arena is 18 x 32 tiles, but it is **not** drawn as a flat rectangle on the screen:
Clash Royale renders it with a perspective camera, so tiles near the enemy King Tower are
noticeably smaller on screen than tiles near your own. Measured on a real 720x1280 battle
frame, the river sits ~59 px below the midpoint of the two Princess Tower rows.

A single linear (affine) tile->pixel mapping therefore cannot be correct everywhere. The
repo's original calibration multiplied tile_y by a constant, which is accurate near the
middle of the arena but drifts by a full tile or more near the enemy side — enough for a
"Miner on the tower" placement to land short, or for a bridge push to land in the river.

Because the arena is a flat plane under a perspective camera, the correct model is a
**homography**. Four known landmarks (the four Princess Tower bases sit at fixed tiles)
determine it exactly:

    enemy_left   ( 3.5,  6.5)      enemy_right  (14.5,  6.5)
    ally_left    ( 3.5, 25.5)      ally_right   (14.5, 25.5)

Everything below is plain numpy so it stays testable without a device.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

TILES_X = 18.0
TILES_Y = 32.0
RIVER_TILE_Y = 16.0

#: Canonical tile coordinates of the four Princess Towers (their base centres).
PRINCESS_TILES: Dict[str, Point] = {
    "enemy_left": (3.5, 6.5),
    "enemy_right": (14.5, 6.5),
    "ally_left": (3.5, 25.5),
    "ally_right": (14.5, 25.5),
}

KING_TILES: Dict[str, Point] = {
    "enemy_king": (9.0, 2.8),
    "ally_king": (9.0, 29.2),
}


def solve_homography(src: Sequence[Point], dst: Sequence[Point]) -> np.ndarray:
    """Least-squares homography mapping ``src`` -> ``dst`` (needs >= 4 point pairs)."""
    if len(src) != len(dst):
        raise ValueError("src and dst must have the same length")
    if len(src) < 4:
        raise ValueError("at least 4 point pairs are required")

    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    a = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (collinear landmarks)")
    return h / h[2, 2]


def apply_homography(h: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    vec = h @ np.array([x, y, 1.0], dtype=np.float64)
    w = vec[2]
    if abs(w) < 1e-9:
        return float("nan"), float("nan")
    return float(vec[0] / w), float(vec[1] / w)


class Projection:
    """Base class: converts between arena tiles and screen pixels."""

    kind = "base"

    def tile_to_pixel(self, tile_x: float, tile_y: float) -> Tuple[int, int]:
        raise NotImplementedError

    def pixel_to_tile(self, px: float, py: float) -> Tuple[float, float]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @property
    def river_pixel_y(self) -> float:
        return self.tile_to_pixel(TILES_X / 2.0, RIVER_TILE_Y)[1]


class RectProjection(Projection):
    """Simple affine mapping onto an axis-aligned arena rectangle (legacy behaviour)."""

    kind = "rect"

    def __init__(self, left: float, top: float, right: float, bottom: float) -> None:
        if right - left <= 1 or bottom - top <= 1:
            raise ValueError(f"degenerate arena rect {(left, top, right, bottom)}")
        self.left = float(left)
        self.top = float(top)
        self.right = float(right)
        self.bottom = float(bottom)

    def tile_to_pixel(self, tile_x: float, tile_y: float) -> Tuple[int, int]:
        px = self.left + (tile_x / TILES_X) * (self.right - self.left)
        py = self.top + (tile_y / TILES_Y) * (self.bottom - self.top)
        return int(round(px)), int(round(py))

    def pixel_to_tile(self, px: float, py: float) -> Tuple[float, float]:
        tx = ((px - self.left) / (self.right - self.left)) * TILES_X
        ty = ((py - self.top) / (self.bottom - self.top)) * TILES_Y
        return (
            float(max(0.0, min(TILES_X, tx))),
            float(max(0.0, min(TILES_Y, ty))),
        )

    @property
    def rect(self) -> Tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "rect": [self.left, self.top, self.right, self.bottom]}

    @classmethod
    def from_dict(cls, data: dict) -> "RectProjection":
        return cls(*data["rect"])


class HomographyProjection(Projection):
    """Perspective-correct mapping, calibrated from the four Princess Towers."""

    kind = "homography"

    def __init__(self, h: np.ndarray) -> None:
        self.h = np.asarray(h, dtype=np.float64).reshape(3, 3)
        self.h_inv = np.linalg.inv(self.h)

    def tile_to_pixel(self, tile_x: float, tile_y: float) -> Tuple[int, int]:
        px, py = apply_homography(self.h, tile_x, tile_y)
        return int(round(px)), int(round(py))

    def pixel_to_tile(self, px: float, py: float) -> Tuple[float, float]:
        tx, ty = apply_homography(self.h_inv, px, py)
        return (
            float(max(0.0, min(TILES_X, tx))),
            float(max(0.0, min(TILES_Y, ty))),
        )

    def corners(self) -> List[Tuple[float, float]]:
        return [
            apply_homography(self.h, 0.0, 0.0),
            apply_homography(self.h, TILES_X, 0.0),
            apply_homography(self.h, TILES_X, TILES_Y),
            apply_homography(self.h, 0.0, TILES_Y),
        ]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "homography": self.h.reshape(-1).tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "HomographyProjection":
        return cls(np.asarray(data["homography"], dtype=np.float64).reshape(3, 3))

    def scale(self, factor_x: float, factor_y: float) -> "HomographyProjection":
        """Rescale the mapping when the calibration frame size differs from the live frame."""
        s = np.array([[factor_x, 0.0, 0.0], [0.0, factor_y, 0.0], [0.0, 0.0, 1.0]])
        return HomographyProjection(s @ self.h)


def projection_from_dict(data: dict) -> Projection:
    if data.get("kind") == "homography":
        return HomographyProjection.from_dict(data)
    return RectProjection.from_dict(data)


def projection_from_landmarks(
    tower_pixels: Dict[str, Point],
    extra: Optional[Dict[str, Point]] = None,
) -> Tuple[HomographyProjection, float]:
    """Build a homography from detected tower base centres.

    Returns ``(projection, residual_px)`` where the residual is the RMS reprojection error
    over every landmark supplied. A residual above ~10 px means the landmarks were detected
    badly and the caller should reject the calibration.
    """
    extra = extra or {}
    src: List[Point] = []
    dst: List[Point] = []
    for name, tile in {**PRINCESS_TILES, **KING_TILES}.items():
        if name in tower_pixels:
            src.append(tile)
            dst.append(tuple(tower_pixels[name]))
    if len(src) < 4:
        raise ValueError("need at least four tower landmarks to calibrate")

    h = solve_homography(src, dst)
    proj = HomographyProjection(h)

    errors = []
    for (tx, ty), (px, py) in zip(src, dst):
        ex, ey = apply_homography(h, tx, ty)
        errors.append(math.hypot(ex - px, ey - py))
    residual = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
    return proj, residual


def projection_from_arena_rect(rect: Tuple[float, float, float, float]) -> RectProjection:
    return RectProjection(*rect)


def clamp_rect_to_frame(
    rect: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[float, float, float, float]:
    left, top, right, bottom = rect
    left = max(0.0, min(width - 2.0, left))
    right = max(left + 2.0, min(width, right))
    top = max(0.0, min(height - 2.0, top))
    bottom = max(top + 2.0, min(height, bottom))
    return left, top, right, bottom
