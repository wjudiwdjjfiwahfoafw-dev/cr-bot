"""Detect troops on a *real* device by segmenting them out of the arena floor.

Why not colours and health bars alone
-------------------------------------
The previous detector looked for red/blue horizontal strips and guessed a unit's identity from
the strip's pixel width using 720p emulator constants.  On a real device that produces a lot of
noise (the arena floor itself is intensely coloured: grass, river, bridge planks, tower roofs all
pass a loose red/blue threshold) and it can never say *what* it is looking at.

What this module does instead
-----------------------------
The arena is a **static scene**: the floor, bridges, towers and chests never move, only troops do.
A running background model therefore separates the units exactly, with no colour assumptions and
no trained model:

1. :class:`ArenaBackgroundModel` keeps a selectively-updated median/mean of the arena.
2. :class:`BlobUnitDetector` thresholds the difference, cleans it up, and turns what is left into
   :class:`BlobUnit` records with a real bounding box.
3. Everything geometric is expressed in **tiles** via the calibrated :class:`ArenaGrid`, so the
   same code works at 720p, 1080p or 1440p — no emulator pixel constants anywhere.
4. :class:`HealthBarReader` supplies the side and the HP fraction from the bar floating above a
   unit, using tile-relative size limits and requiring the bar's dark outline; bars are matched to
   blobs, not used as the detector.

Identity (which card it is) is a separate, optional layer: see
:class:`cr_bot.vision.unit_identity.UnitIdentityMemory`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.models import MoveType, Side, WeightClass

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels


# ---------------------------------------------------------------------------------------------
# geometry helper
# ---------------------------------------------------------------------------------------------
def pixels_per_tile(grid: ArenaGrid, samples: int = 5) -> float:
    """Average on-screen size of one arena tile, measured through the calibrated projection."""
    xs: List[float] = []
    for i in range(samples):
        ty = 2.0 + (grid.TILES_Y - 4.0) * i / max(1, samples - 1)
        x1, y1 = grid.tile_to_pixel(4.0, ty)
        x2, y2 = grid.tile_to_pixel(14.0, ty)
        xs.append(abs(x2 - x1) / 10.0)
    val = float(np.median(xs)) if xs else 32.0
    return max(8.0, val)


def classify_size(height_tiles: float) -> Tuple[WeightClass, str]:
    """Map an on-screen sprite height (in tiles) to a weight class and a fallback label."""
    if height_tiles < 1.15:
        return WeightClass.SWARM, "unknown_swarm"
    if height_tiles < 1.9:
        return WeightClass.LIGHT, "unknown_light"
    if height_tiles < 3.1:
        return WeightClass.MEDIUM, "unknown_medium"
    return WeightClass.HEAVY_TANK, "unknown_tank"


# ---------------------------------------------------------------------------------------------
# background model
# ---------------------------------------------------------------------------------------------
class ArenaBackgroundModel:
    """Running model of the static arena, so moving troops pop out as foreground.

    A plain median over the last N frames would eventually absorb a unit that stands still, so
    the model is updated **only where nothing is currently detected** (selective update).  Towers
    that get destroyed do get absorbed, which is exactly what we want.
    """

    def __init__(self, learning_rate: float = 0.06, diff_threshold: float = 26.0,
                 warmup_frames: int = 3) -> None:
        self.learning_rate = float(learning_rate)
        self.diff_threshold = float(diff_threshold)
        self.warmup_frames = int(warmup_frames)
        self._background: Optional[np.ndarray] = None
        self._frames_seen = 0

    @property
    def is_ready(self) -> bool:
        return self._background is not None and self._frames_seen >= self.warmup_frames

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def reset(self) -> None:
        self._background = None
        self._frames_seen = 0

    def _match_shape(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame = frame_bgr.astype(np.float32)
        if self._background is None or self._background.shape != frame.shape:
            self._background = frame.copy()
            self._frames_seen = 1
            return frame
        return frame

    def foreground(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Binary mask (uint8 0/255) of everything that is not the static arena."""
        frame = self._match_shape(frame_bgr)
        if self._background is None:
            return np.zeros(frame_bgr.shape[:2], np.uint8)
        diff = cv2.absdiff(frame, self._background)
        diff_lab = cv2.cvtColor(np.clip(diff, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        diff_lab = cv2.GaussianBlur(diff_lab, (5, 5), 0)
        _thr, mask = cv2.threshold(diff_lab, self.diff_threshold, 255, cv2.THRESH_BINARY)
        return mask

    def update(self, frame_bgr: np.ndarray, foreground_mask: Optional[np.ndarray] = None) -> None:
        """Fold the frame into the background wherever nothing was detected."""
        frame = self._match_shape(frame_bgr)
        if self._background is None:
            return
        if self._frames_seen <= self.warmup_frames:
            # settle quickly during warm-up
            rate = 0.5
        else:
            rate = self.learning_rate
        if foreground_mask is None:
            self._background += rate * (frame - self._background)
        else:
            keep = (foreground_mask == 0).astype(np.float32)[:, :, None]
            blended = self._background + rate * (frame - self._background)
            self._background = self._background * (1.0 - keep) + blended * keep
        self._frames_seen += 1

    def observe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convenience: foreground mask for this frame, then update the model."""
        mask = self.foreground(frame_bgr)
        self.update(frame_bgr, mask)
        return mask


# ---------------------------------------------------------------------------------------------
# blobs
# ---------------------------------------------------------------------------------------------
@dataclass
class BlobUnit:
    """A moving object segmented out of the arena."""

    bbox: Box
    area_px: int
    side: Side
    side_confidence: float
    height_tiles: float
    width_tiles: float
    tile_x: float
    tile_y: float
    weight_class: WeightClass
    fallback_label: str
    hp_ratio: float = 1.0
    has_bar: bool = False
    mean_bgr: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_airborne: bool = False

    @property
    def foot_px(self) -> Tuple[float, float]:
        x1, _y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), float(y2))


@dataclass
class HealthBar:
    """A unit health bar found in the frame."""

    bbox: Box
    side: Side
    hp_ratio: float
    fill_width: int
    total_width: int


class HealthBarReader:
    """Find unit health bars, tile-relative, with a dark-outline sanity check.

    Bars are thin, brightly saturated, horizontally aligned strips that sit inside a dark
    capsule.  Requiring that capsule is what removes the arena floor from the picture.
    """

    MIN_HEIGHT_TILES = 0.035
    MAX_HEIGHT_TILES = 0.34
    MIN_WIDTH_TILES = 0.18
    MAX_WIDTH_TILES = 3.2

    def __init__(self, min_fill_ratio: float = 0.55) -> None:
        self.min_fill_ratio = min_fill_ratio

    def read(self, frame_bgr: np.ndarray, grid: ArenaGrid, ppt: float) -> List[HealthBar]:
        ax1, ay1, ax2, ay2 = grid.arena_pixel_bounds
        ax1, ay1 = max(0, ax1), max(0, ay1)
        ax2, ay2 = min(frame_bgr.shape[1], ax2), min(frame_bgr.shape[0], ay2)
        if ax2 - ax1 < 40 or ay2 - ay1 < 40:
            return []
        roi = frame_bgr[ay1:ay2, ax1:ax2]
        masks = self._colour_masks(roi)

        h_min = max(2, int(round(self.MIN_HEIGHT_TILES * ppt)))
        h_max = max(h_min + 1, int(round(self.MAX_HEIGHT_TILES * ppt)))
        w_min = max(4, int(round(self.MIN_WIDTH_TILES * ppt)))
        w_max = max(w_min + 2, int(round(self.MAX_WIDTH_TILES * ppt)))

        bars: List[HealthBar] = []
        for side, mask in masks:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # A bar can be fused with whatever sits directly above it (a plume, a weapon, a
                # level badge).  Keep only the horizontal run that is actually bar-shaped.
                if bh > h_max:
                    strip = self._bar_rows(mask[by:by + bh, bx:bx + bw])
                    if strip is None:
                        continue
                    top, bottom = strip
                    by, bh = by + top, bottom - top
                if not (w_min <= bw <= w_max and h_min <= bh <= h_max):
                    continue
                if bw / float(max(1, bh)) < 1.9:
                    continue
                filled = float(cv2.countNonZero(mask[by:by + bh, bx:bx + bw]))
                if filled / float(max(1, bw * bh)) < self.min_fill_ratio:
                    continue
                bar = self._with_tail(roi, mask, side, bx, by, bw, bh)
                if bar is not None:
                    bars.append(
                        HealthBar(
                            bbox=(bar[0] + ax1, bar[1] + ay1, bar[2] + ax1, bar[3] + ay1),
                            side=side,
                            hp_ratio=bar[4],
                            fill_width=bw,
                            total_width=bar[2] - bar[0],
                        )
                    )
        return self._merge(bars)

    @staticmethod
    def _bar_rows(component: np.ndarray) -> Optional[Tuple[int, int]]:
        """Vertical extent of the bar-shaped part of a merged component, if there is one."""
        widths = (component > 0).sum(axis=1)
        if widths.size == 0 or widths.max() <= 0:
            return None
        threshold = 0.6 * widths.max()
        rows = np.nonzero(widths >= threshold)[0]
        if rows.size == 0:
            return None
        # the longest run of consecutive qualifying rows is the bar
        best_start = best_len = 0
        start = rows[0]
        prev = rows[0]
        for row in rows[1:]:
            if row != prev + 1:
                if prev - start + 1 > best_len:
                    best_start, best_len = start, prev - start + 1
                start = row
            prev = row
        if prev - start + 1 > best_len:
            best_start, best_len = start, prev - start + 1
        if best_len > 0.5 * component.shape[0]:
            return None
        return (int(best_start), int(best_start + best_len))

    @staticmethod
    def _colour_masks(roi: np.ndarray) -> List[Tuple[Side, np.ndarray]]:
        """Saturated team-coloured pixels, without paying for a full HSV conversion.

        Health-bar reds/pinks and blues are strongly channel-separated, so two comparisons per
        pixel are both faster and less range-fiddly than an HSV threshold band.
        """
        b, g, r = cv2.split(roi.astype(np.int16))
        red = (((r - g) > 55) & ((r - b) > 55) & (r > 110)).astype(np.uint8) * 255
        blue = (((b - r) > 45) & (b > 110)).astype(np.uint8) * 255
        return [(Side.ENEMY, red), (Side.ALLY, blue)]

    def _with_tail(
        self, roi: np.ndarray, mask: np.ndarray, side: Side,
        bx: int, by: int, bw: int, bh: int,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """Extend the coloured fill through the dark (already lost) part of the bar."""
        row = by + bh // 2
        line = roi[row].astype(np.int16)
        ext_right = bx + bw
        limit = min(roi.shape[1] - 1, bx + int(bw * 2.2))
        while ext_right < limit:
            b, g, r = line[ext_right]
            spread = max(int(r), int(g), int(b)) - min(int(r), int(g), int(b))
            value = max(int(r), int(g), int(b))
            if spread < 60 and 30 <= value <= 190:   # the lost part of a bar is desaturated grey
                ext_right += 1
            else:
                break
        if ext_right - bx <= bw:
            return (bx, by, bx + bw, by + bh, 1.0)
        ratio = float(bw) / float(max(1, ext_right - bx))
        return (bx, by, ext_right, by + bh, float(np.clip(ratio, 0.02, 1.0)))

    @staticmethod
    def _merge(bars: Sequence[HealthBar]) -> List[HealthBar]:
        """Drop duplicate detections of the same bar."""
        kept: List[HealthBar] = []
        for bar in sorted(bars, key=lambda b: -(b.bbox[2] - b.bbox[0])):
            x1, y1, x2, y2 = bar.bbox
            dup = False
            for other in kept:
                ox1, oy1, ox2, oy2 = other.bbox
                ix = max(0, min(x2, ox2) - max(x1, ox1))
                iy = max(0, min(y2, oy2) - max(y1, oy1))
                if ix > 0.6 * min(x2 - x1, ox2 - ox1) and iy > 0.5 * min(y2 - y1, oy2 - oy1):
                    dup = True
                    break
            if not dup:
                kept.append(bar)
        return kept


class BlobUnitDetector:
    """Segment moving troops out of the arena and describe them in tile coordinates."""

    MIN_AREA_TILES2 = 0.05
    MAX_AREA_TILES2 = 42.0
    MIN_HEIGHT_TILES = 0.35
    MAX_HEIGHT_TILES = 6.0
    MAX_WIDTH_TILES = 5.0

    #: segmentation runs on a half-resolution copy of the arena: a troop is 30+ px tall, so
    #: nothing is lost, and the per-frame cost drops by ~4x on a phone-sized capture
    WORK_SCALE = 0.5

    def __init__(
        self,
        background: Optional[ArenaBackgroundModel] = None,
        bar_reader: Optional[HealthBarReader] = None,
        diff_threshold: float = 26.0,
    ) -> None:
        self.background = background if background is not None else ArenaBackgroundModel(
            diff_threshold=diff_threshold)
        self.bars = bar_reader if bar_reader is not None else HealthBarReader()
        self.last_bars: List[HealthBar] = []
        self._ppt_cache: Dict[Tuple[int, int, int], float] = {}

    def reset(self) -> None:
        self.background.reset()
        self.last_bars = []

    def tiles_per_pixel(self, grid: ArenaGrid) -> float:
        """Cached tile scale: the projection cannot change without the resolution changing."""
        key = (grid.width, grid.height, id(grid.cal.projection))
        cached = self._ppt_cache.get(key)
        if cached is None:
            cached = pixels_per_tile(grid)
            self._ppt_cache.clear()
            self._ppt_cache[key] = cached
        return cached

    def seed_background(self, frame_bgr: np.ndarray, grid: ArenaGrid) -> bool:
        """Prime the model with a known-empty arena frame (e.g. right after a match starts)."""
        ax1, ay1, ax2, ay2 = grid.arena_pixel_bounds
        ax1, ay1 = max(0, ax1), max(0, ay1)
        ax2, ay2 = min(frame_bgr.shape[1], ax2), min(frame_bgr.shape[0], ay2)
        if ax2 - ax1 < 40 or ay2 - ay1 < 40:
            return False
        roi = frame_bgr[ay1:ay2, ax1:ax2]
        scale = self.WORK_SCALE if min(roi.shape[0], roi.shape[1]) > 120 else 1.0
        small = (
            cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)),
                       interpolation=cv2.INTER_AREA)
            if scale < 1.0 else roi
        )
        self.background.reset()
        self.background._background = small.astype(np.float32).copy()
        self.background._frames_seen = 99
        return True

    def detect(self, frame_bgr: np.ndarray, grid: ArenaGrid, observe: bool = True) -> List[BlobUnit]:
        ppt = self.tiles_per_pixel(grid)
        ax1, ay1, ax2, ay2 = grid.arena_pixel_bounds
        ax1, ay1 = max(0, ax1), max(0, ay1)
        ax2, ay2 = min(frame_bgr.shape[1], ax2), min(frame_bgr.shape[0], ay2)
        if ax2 - ax1 < 40 or ay2 - ay1 < 40:
            return []

        roi = frame_bgr[ay1:ay2, ax1:ax2]
        scale = self.WORK_SCALE if min(roi.shape[0], roi.shape[1]) > 120 else 1.0
        small = (
            cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)),
                       interpolation=cv2.INTER_AREA)
            if scale < 1.0 else roi
        )

        fg = self.background.foreground(small) if self.background.is_ready else \
            np.zeros(small.shape[:2], np.uint8)

        units: List[BlobUnit] = []
        if self.background.is_ready:
            self.last_bars = self.bars.read(frame_bgr, grid, ppt)
            units = self._from_mask(fg, roi, grid, ppt, (ax1, ay1), scale, self.last_bars)
        if observe:
            self.background.update(small, fg)
        return units

    # -- internals -----------------------------------------------------------------
    def _from_mask(
        self, fg: np.ndarray, roi: np.ndarray, grid: ArenaGrid, ppt: float,
        origin: Tuple[int, int], scale: float, bars: Sequence[HealthBar],
    ) -> List[BlobUnit]:
        ax1, ay1 = origin
        mask = fg
        if mask.size == 0:
            return []
        # close small holes (a sprite's own shading), then remove specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        inv = 1.0 / scale
        min_area = self.MIN_AREA_TILES2 * ppt * ppt * scale * scale
        max_area = self.MAX_AREA_TILES2 * ppt * ppt * scale * scale

        out: List[BlobUnit] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = float(cv2.countNonZero(mask[by:by + bh, bx:bx + bw]))
            if area < min_area or area > max_area:
                continue
            h_tiles = bh * inv / ppt
            w_tiles = bw * inv / ppt
            if not (self.MIN_HEIGHT_TILES <= h_tiles <= self.MAX_HEIGHT_TILES):
                continue
            if w_tiles > self.MAX_WIDTH_TILES:
                continue
            # ignore slivers and the long thin "spell trail" shapes
            if bw / float(max(1, bh)) > 3.2:
                continue

            x1 = int(round(bx * inv)) + ax1
            y1 = int(round(by * inv)) + ay1
            x2 = int(round((bx + bw) * inv)) + ax1
            y2 = int(round((by + bh) * inv)) + ay1
            patch = roi[y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1]
            blob_mask = cv2.resize(mask[by:by + bh, bx:bx + bw],
                                   (max(1, x2 - x1), max(1, y2 - y1)),
                                   interpolation=cv2.INTER_NEAREST)
            side, side_conf, mean_bgr = self._side_of(patch, blob_mask)
            foot_x, foot_y = 0.5 * (x1 + x2), float(y2)
            tile_x, tile_y = grid.pixel_to_tile(foot_x, foot_y)
            weight, label = classify_size(h_tiles)

            bar = self._bar_for(x1, y1, x2, y2, bars, ppt)
            if bar is not None and bar.side != side:
                side, side_conf = bar.side, max(side_conf, 0.7)

            out.append(
                BlobUnit(
                    bbox=(x1, y1, x2, y2),
                    area_px=int(area),
                    side=side,
                    side_confidence=float(side_conf),
                    height_tiles=float(h_tiles),
                    width_tiles=float(w_tiles),
                    tile_x=float(tile_x),
                    tile_y=float(tile_y),
                    weight_class=weight,
                    fallback_label=label,
                    hp_ratio=bar.hp_ratio if bar is not None else 1.0,
                    has_bar=bar is not None,
                    mean_bgr=mean_bgr,
                    is_airborne=False,
                )
            )
        return out

    @staticmethod
    def _side_of(patch: np.ndarray, blob_mask: np.ndarray) -> Tuple[Side, float, Tuple[float, float, float]]:
        """Team tint: the game renders your units blue-shifted and the opponent's red-shifted."""
        if patch.size == 0:
            return Side.ENEMY, 0.0, (0.0, 0.0, 0.0)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        sel = blob_mask > 0
        if not np.any(sel):
            return Side.ENEMY, 0.0, (0.0, 0.0, 0.0)
        h = hsv[:, :, 0][sel].astype(np.float32)
        s = hsv[:, :, 1][sel].astype(np.float32)
        v = hsv[:, :, 2][sel].astype(np.float32)
        # only saturated, reasonably bright pixels carry the team tint
        keep = (s > 60) & (v > 40)
        if keep.sum() < 12:
            return Side.ENEMY, 0.0, (
                float(patch[:, :, 0].mean()), float(patch[:, :, 1].mean()), float(patch[:, :, 2].mean())
            )
        hh = h[keep]
        # the game shifts your own units towards blue-violet and the opponent's towards red.
        # Ranges are deliberately wide: a dark knight's "blue" armour sits around hue 120-150.
        red = float(np.count_nonzero((hh < 20) | (hh > 160)))
        blue = float(np.count_nonzero((hh >= 85) & (hh <= 160)))
        total = red + blue + 1e-6
        mean_bgr = (
            float(patch[:, :, 0][sel].mean()),
            float(patch[:, :, 1][sel].mean()),
            float(patch[:, :, 2][sel].mean()),
        )
        if blue > red:
            return Side.ALLY, float(blue / total), mean_bgr
        return Side.ENEMY, float(red / total), mean_bgr

    @staticmethod
    def _bar_for(x1: int, y1: int, x2: int, y2: int, bars: Sequence[HealthBar], ppt: float) -> Optional[HealthBar]:
        """The bar belonging to a blob: horizontally overlapping and sitting just above it."""
        best: Optional[HealthBar] = None
        best_score = -1e9
        cx = 0.5 * (x1 + x2)
        blob_w = max(1.0, float(x2 - x1))
        for bar in bars:
            bx1, by1, bx2, by2 = bar.bbox
            bar_cx = 0.5 * (bx1 + bx2)
            if abs(bar_cx - cx) > 0.55 * ppt:
                continue
            # bar bottom relative to the blob top, positive when the bar floats above the sprite.
            # Negative values are normal for tall units with raised arms: the bar overlaps the box.
            dy = y1 - by2
            if dy < -1.7 * ppt or dy > 1.35 * ppt:
                continue
            # a unit's bar spans roughly its own width; anything much narrower is arena clutter
            width_ratio = bar.total_width / blob_w
            if not (0.22 <= width_ratio <= 1.6):
                continue
            score = (
                -abs(bar_cx - cx)
                - abs(dy - 0.5 * ppt)
                - 20.0 * abs(math.log(width_ratio))
            )
            if score > best_score:
                best_score = score
                best = bar
        return best


def move_type_for(name: str, default: MoveType = MoveType.GROUND) -> MoveType:
    """Air/ground for a card name, from the card database (unknown names stay ground)."""
    try:
        from cr_bot.config.cards_db import get_card_info

        info = get_card_info(name)
    except Exception:  # pragma: no cover - defensive
        return default
    if info is None:
        return default
    return info.move_type


@dataclass
class DetectorStats:
    """Cheap counters to make the detector's behaviour visible in logs/tests."""

    frames: int = 0
    units: int = 0
    bars: int = 0
    identified: int = 0
    per_side: Dict[str, int] = field(default_factory=dict)

    def add(self, units: Sequence[BlobUnit], bars: int = 0, identified: int = 0) -> None:
        self.frames += 1
        self.units += len(units)
        self.bars += bars
        self.identified += identified
        for unit in units:
            key = unit.side.value
            self.per_side[key] = self.per_side.get(key, 0) + 1
