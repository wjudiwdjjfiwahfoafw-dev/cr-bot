"""Detect troops on a *real* device by segmenting them out of the arena floor.

What went wrong before
----------------------
The first generation looked for red/blue horizontal strips and guessed identity from the
strip's pixel width using 720p emulator constants. The second generation segmented moving
pixels, which is the right idea, but it had three failure modes that are very visible in
play:

**"Sometimes troops are not detected."**
    A blob only counted if it differed from a slowly-updated background by a *fixed* margin,
    and a *stationary* troop was only kept in the foreground because the selective update
    happened to skip it. Nothing distinguished "a unit standing still" from "the arena itself
    changed" (a destroyed tower, a spawned elixir golem corpse, a spell splat), so a unit
    parked in one spot could be absorbed, and a newly-arrived one hidden inside an already
    "foreground" region was invisible.

**"Sometimes they are detected as multiple troops."**
    Connected-component labelling merges a swarm into one blob and splits one tall sprite into
    a body plus a detached weapon, a shield, or a shadow. Every fragment passed the size filter
    independently, so a Mega Knight with a raised hammer became two units, and six skeletons
    standing together became one.

**"Sometimes they are detected as enemy although they are ally."**
    Team side was decided from the hue of the blob's *saturated* pixels, with two traps:
    a low-saturation sprite (grey armour, a dark Miner, a brown Skeleton Barrel) fell through
    to a hard-coded ``Side.ENEMY`` default, and any sprite with red in its own art (a red
    Dragon, a red barrel) outvoted the blue team tint on top of it.

How it works now
----------------
1. :class:`ArenaBackgroundModel` labels every pixel **foreground / background / unknown**.
   *Unknown* is the important part: a pixel that has not been observed cleanly yet (or whose
   content is being re-learned) is not used to decide anything, instead of defaulting to
   "background" and hiding a troop. Hysteresis (a low and a high threshold) keeps thin limbs
   and outlines of a sprite attached to its body, and a frame-difference *motion* term keeps a
   unit that has just arrived visible before the model has caught up.
2. :class:`HealthBarReader` reads every unit health bar, using the bar's **outline** to
   survive the arena floor, and returns the team colour of the bar. Bars are the most reliable
   team signal in the game, because the game itself colours them: blue for you, red for the
   opponent. They also give an exact HP fraction.
3. Blobs are cleaned, then **split and merged using the bars as anchors**: one blob with two
   bars becomes two units (a swarm seen as one), two fragments with no bar in a small radius
   become one unit (a sprite seen as two). That is the pair of bugs above, fixed structurally
   rather than by tuning a threshold.
4. :class:`TeamClassifier` votes on the side from four independent signals — the health bar,
   the elliptical *team ring* the game draws under every unit (measured as a colour cast in an
   annulus and de-confounded by subtracting the sprite's own colour cast), a de-confounded
   body chroma term, and motion direction — and returns a real confidence. When it is not
   sure it says so instead of guessing ``ENEMY``.
5. Geometry is expressed in **tiles** through the calibrated :class:`ArenaGrid`, so nothing
   here depends on the resolution: the same code runs on a 720p stream and a 1440p phone.

Identity (which card it is) remains a separate, optional layer: see
:class:`cr_bot.vision.unit_identity.UnitIdentityMemory`. It is never required for detection,
counting, side or position.
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


#: representative sprite height per weight class (tiles), used when a bar - not the sprite's
#: silhouette - is the size evidence
_NOMINAL_HEIGHT_TILES = {
    WeightClass.SWARM: 0.60,
    WeightClass.LIGHT: 1.30,
    WeightClass.MEDIUM: 2.40,
    WeightClass.HEAVY_TANK: 3.40,
}


def _height_for_weight(weight: WeightClass) -> float:
    return _NOMINAL_HEIGHT_TILES.get(weight, 2.0)


def classify_by_bar_width(width_tiles: float) -> Tuple[WeightClass, str]:
    """Secondary size vote from the health bar, whose width tracks the card's footprint.

    A Giant's bar is roughly 1.6 tiles wide on a real device, a Skeleton's roughly 0.35, so
    this is a useful *independent* hint when the sprite is partly hidden behind a tower or
    another unit and its height cannot be measured.
    """
    if width_tiles < 0.48:
        return WeightClass.SWARM, "unknown_swarm"
    if width_tiles < 0.80:
        return WeightClass.LIGHT, "unknown_light"
    if width_tiles < 1.25:
        return WeightClass.MEDIUM, "unknown_medium"
    return WeightClass.HEAVY_TANK, "unknown_tank"


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


# ---------------------------------------------------------------------------------------------
# background model
# ---------------------------------------------------------------------------------------------
#: pixel states produced by :meth:`ArenaBackgroundModel.state`
BG_BACKGROUND = 0
BG_UNKNOWN = 1
BG_FOREGROUND = 2


class ArenaBackgroundModel:
    """Model of the static arena that refuses to answer when it does not know.

    The arena is a static scene — floor, bridges, towers, bushes — and only troops move. So a
    running model of "what the arena looks like" separates units exactly, with no colour
    assumptions and no trained model. The detail that matters in practice is what to do with a
    pixel whose background value is not trustworthy *yet*:

    * a troop that spawns where a troop stood a moment ago,
    * the spot where a Princess Tower used to be (the rubble is not the tower),
    * the frame right after the camera/UI shifts by a pixel.

    A naive model calls those "background" and the troop disappears. This one marks them
    **unknown** and excludes them from the background decision, while an independent
    frame-difference term keeps anything that is *moving right now* alive. Large regions that
    stay perfectly static for a long time (destroyed towers) are slowly adopted into the
    background so they stop shredding the frame into false blobs.
    """

    #: hysteresis: a pixel this far from the model is certainly foreground
    STRONG_MULTIPLIER = 2.0
    #: a region that has been foreground and *identical* this long is scenery, not a troop
    ABSORB_AFTER_FRAMES = 60
    ABSORB_MIN_AREA_TILES2 = 5.0
    ABSORB_RATE = 0.02

    def __init__(
        self,
        learning_rate: float = 0.06,
        diff_threshold: float = 20.0,
        warmup_frames: int = 3,
        motion_threshold: float = 9.0,
        absorb_static_scenery: bool = True,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.diff_threshold = float(diff_threshold)
        self.warmup_frames = int(warmup_frames)
        self.motion_threshold = float(motion_threshold)
        self.absorb_static_scenery = bool(absorb_static_scenery)

        self._background: Optional[np.ndarray] = None
        self._background_u8: Optional[np.ndarray] = None
        self._prev: Optional[np.ndarray] = None
        self._frames_seen = 0
        #: per-pixel number of consecutive frames this pixel has been foreground
        self._foreground_age: Optional[np.ndarray] = None
        #: per-pixel "the content changed" counter, used to decide when a region is static
        self._change_age: Optional[np.ndarray] = None
        #: how many frames a pixel needed to settle the first time (warm-up grace)
        self._settled: Optional[np.ndarray] = None
        self._shift_cooldown = 0
        self._ppt = 1.0
        #: fraction of the arena that must change at once before it is called a scene shift
        self.SCENE_SHIFT_FRACTION = 0.30
        self.SCENE_SHIFT_COOLDOWN = 4

    # -- introspection -------------------------------------------------------------
    @property
    def recently_shifted(self) -> bool:
        """True for a few frames after a scroll/zoom/resolution change.

        Nothing the detector says during a camera move is trustworthy: the whole arena counts
        as "changed", which would otherwise produce a burst of phantom troops at the edges.
        """
        return self._shift_cooldown > 0

    @property
    def is_ready(self) -> bool:
        return self._background is not None and self._frames_seen >= self.warmup_frames

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def threshold(self) -> float:
        return self.diff_threshold

    def reset(self) -> None:
        self._background = None
        self._prev = None
        self._frames_seen = 0
        self._foreground_age = None
        self._change_age = None
        self._settled = None
        self._background_u8 = None
        self._shift_cooldown = 0

    def _background_u8_view(self) -> np.ndarray:
        if self._background_u8 is None and self._background is not None:
            self._background_u8 = np.clip(self._background, 0, 255).astype(np.uint8)
        assert self._background_u8 is not None
        return self._background_u8

    def set_tile_scale(self, ppt: float) -> None:
        """Tell the model how large a tile is, so scenery thresholds are resolution-free."""
        self._ppt = max(1.0, float(ppt))

    # -- internals -----------------------------------------------------------------
    def _ensure(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_u8 = frame_bgr if frame_bgr.dtype == np.uint8 else np.clip(frame_bgr, 0, 255).astype(np.uint8)
        frame = frame_u8.astype(np.float32)
        if self._background is None or self._background.shape != frame.shape:
            self._background = frame.copy()
            self._background_u8 = frame_u8.copy()
            self._prev = frame_u8.copy()
            self._foreground_age = np.zeros(frame.shape[:2], np.int32)
            self._change_age = np.zeros(frame.shape[:2], np.int32)
            self._settled = np.zeros(frame.shape[:2], np.int32)
            self._frames_seen = 1
        return frame

    @staticmethod
    def _difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Per-pixel difference, taken as the largest channel delta, in uint8.

        ``absdiff`` then ``cvtColor(BGR2GRAY)`` averages the channels, which halves the signal
        for a unit that differs from the floor in one channel only (a blue-tinted ally on green
        grass). The per-channel *maximum* is strictly more sensitive, and doing it in uint8
        with SIMD-backed cv2 ops keeps it fast enough for 60 fps.
        """
        diff = cv2.absdiff(a, b)
        if diff.ndim == 3:
            diff = cv2.max(cv2.max(diff[:, :, 0], diff[:, :, 1]), diff[:, :, 2])
        return cv2.GaussianBlur(diff, (3, 3), 0)

    def state(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Classify every pixel as ``BG_BACKGROUND``, ``BG_UNKNOWN`` or ``BG_FOREGROUND``.

        ``FG`` means "something is here that is not the arena". ``UNKNOWN`` means "cannot
        tell", which the detector resolves with other evidence (motion, health bars) instead
        of assuming background.
        """
        self._ensure(frame_bgr)
        if self._frames_seen < self.warmup_frames or self._background is None:
            return np.full(frame_bgr.shape[:2], BG_UNKNOWN, np.uint8)

        frame_u8 = frame_bgr if frame_bgr.dtype == np.uint8 else np.clip(frame_bgr, 0, 255).astype(np.uint8)
        diff = self._difference(frame_u8, self._background_u8_view())
        weak = diff > self.diff_threshold
        strong = diff > self.diff_threshold * self.STRONG_MULTIPLIER

        # A frame where most of the arena changed at once is not "troops everywhere": it is a
        # scroll, a zoom, a resolution change or a UI transition. Say UNKNOWN for a moment
        # instead of reporting dozens of phantom units.
        changed_fraction = float(np.count_nonzero(weak)) / float(weak.size)
        if changed_fraction > self.SCENE_SHIFT_FRACTION:
            if self._shift_cooldown <= 0:
                logger.debug(
                    "arena changed %.0f%% in one frame -> treating it as a scene shift",
                    changed_fraction * 100.0,
                )
            self._shift_cooldown = self.SCENE_SHIFT_COOLDOWN
        if self._shift_cooldown > 0:
            self._shift_cooldown -= 1
            return np.full(frame_bgr.shape[:2], BG_UNKNOWN, np.uint8)

        state = np.full(frame_bgr.shape[:2], BG_BACKGROUND, np.uint8)
        state[weak] = BG_UNKNOWN
        state[strong] = BG_FOREGROUND

        # Hysteresis: a UNKNOWN pixel touching FOREGROUND belongs to that sprite (its outline,
        # a thin weapon, a shadow). Without this, sprites lose limbs and get split in two.
        if weak.any():
            labels_count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
            if labels_count > 1:
                has_strong = np.zeros(labels_count, bool)
                strong_labels = np.unique(labels[strong])
                has_strong[strong_labels[strong_labels > 0]] = True
                has_strong[0] = False
                if has_strong.any():
                    state[has_strong[labels]] = BG_FOREGROUND

        # A pixel the model has not settled yet must not be called background.
        if self._settled is not None:
            unsettled = self._settled < self.warmup_frames
            if unsettled.any():
                state[unsettled & (state == BG_BACKGROUND)] = BG_UNKNOWN
        return state

    def motion(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Binary mask of pixels that changed since the previous frame (a *unit* is moving)."""
        frame = self._ensure(frame_bgr)
        frame_u8 = frame_bgr if frame_bgr.dtype == np.uint8 else np.clip(frame_bgr, 0, 255).astype(np.uint8)
        if self._prev is None or self._prev.shape != frame_u8.shape:
            self._prev = frame_u8.copy()
            return np.zeros(frame_bgr.shape[:2], np.uint8)
        diff = self._difference(frame_u8, self._prev)
        mask = (diff > self.motion_threshold).astype(np.uint8) * 255
        if mask.any():
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            )
        return mask

    def foreground(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Binary mask (uint8 0/255) of what is *certainly* not the static arena.

        Pixels the model cannot vouch for are excluded here (they are ``UNKNOWN``): a binary
        mask should not claim certainty it does not have. :meth:`not_background` gives the
        inclusive version, which is what the background update needs.
        """
        state = self.state(frame_bgr)
        return np.where(state == BG_FOREGROUND, 255, 0).astype(np.uint8)

    def not_background(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Binary mask of everything that is not provably background (includes UNKNOWN)."""
        state = self.state(frame_bgr)
        return np.where(state != BG_BACKGROUND, 255, 0).astype(np.uint8)

    def update(self, frame_bgr: np.ndarray, foreground_mask: Optional[np.ndarray] = None) -> None:
        """Fold the frame into the background where nothing was detected."""
        frame = self._ensure(frame_bgr)
        if self._background is None:
            return
        if self._foreground_age is None:
            self._foreground_age = np.zeros(frame.shape[:2], np.int32)
        if self._change_age is None:
            self._change_age = np.zeros(frame.shape[:2], np.int32)
        if self._settled is None:
            self._settled = np.zeros(frame.shape[:2], np.int32)

        if self._frames_seen <= self.warmup_frames:
            # settle quickly during warm-up
            rate = 0.5
        else:
            rate = self.learning_rate

        if foreground_mask is None:
            occupied = np.zeros(frame.shape[:2], bool)
        else:
            occupied = foreground_mask > 0
            occupied = cv2.dilate(
                occupied.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) > 0

        free = ~occupied
        if free.any():
            self._background[free] += rate * (frame[free] - self._background[free])
            self._settled[free] += 1
            self._foreground_age[free] = 0
            self._change_age[free] = 0

        if occupied.any():
            self._foreground_age[occupied] += 1
            # how long has the *content* of this pixel been constant? (scenery vs a unit)
            frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)
            if self._prev is not None and self._prev.shape == frame_u8.shape:
                changed = self._difference(frame_u8, self._prev)
            else:
                changed = np.zeros(frame.shape[:2], np.float32)
            self._change_age[occupied & (changed < 2.0)] += 1
            self._change_age[occupied & (changed >= 2.0)] = 0

            if self.absorb_static_scenery:
                min_area = self.ABSORB_MIN_AREA_TILES2 * self._ppt * self._ppt
                static = occupied & (self._foreground_age >= self.ABSORB_AFTER_FRAMES) & (
                    self._change_age >= self.ABSORB_AFTER_FRAMES
                )
                if static.any():
                    n, labels = cv2.connectedComponents(static.astype(np.uint8), connectivity=8)
                    if n > 1:
                        areas = np.bincount(labels.ravel(), minlength=n)
                        big = np.zeros(n, bool)
                        big[1:] = areas[1:] >= max(24.0, min_area)
                        big[0] = False
                        adopt = big[labels]
                        if adopt.any():
                            self._background[adopt] += self.ABSORB_RATE * (
                                frame[adopt] - self._background[adopt]
                            )
                            self._foreground_age[adopt] = 0
                            self._change_age[adopt] = 0

        self._prev = np.clip(frame, 0, 255).astype(np.uint8)
        self._background_u8 = None      # invalidate the uint8 mirror after an update
        self._frames_seen += 1

    def observe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convenience: certain-foreground mask for this frame, then update the model.

        The update is told to protect everything that is not *provably* background, so an
        unknown region is never learned as scenery.
        """
        state = self.state(frame_bgr)
        certain = np.where(state == BG_FOREGROUND, 255, 0).astype(np.uint8)
        self.update(frame_bgr, np.where(state != BG_BACKGROUND, 255, 0).astype(np.uint8))
        return certain


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
    #: 0..1 — how much evidence backs this detection (segmentation quality + team vote)
    confidence: float = 0.5
    #: how many sub-parts were merged into this blob (a swarm seen as one, or one sprite
    #: seen as two, both end up here as a number the brain can reason about)
    parts: int = 1
    #: which pipeline step produced it: ``segmentation`` | ``bars`` | ``split`` | ``merged``
    #: | ``motion``
    source: str = "segmentation"
    #: per-signal team evidence, for debugging and for the HUD
    team_evidence: Dict[str, float] = field(default_factory=dict)
    #: health bars this blob is anchored to
    bars: List["HealthBar"] = field(default_factory=list)

    @property
    def foot_px(self) -> Tuple[float, float]:
        x1, _y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), float(y2))

    @property
    def center_px(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area_tiles2(self) -> float:
        return float(self.width_tiles * self.height_tiles)


@dataclass
class HealthBar:
    """A unit health bar found in the frame."""

    bbox: Box
    side: Side
    hp_ratio: float
    fill_width: int
    total_width: int

    @property
    def center_px(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class HealthBarReader:
    """Find unit health bars, tile-relative, with a dark-outline sanity check.

    Bars are thin, brightly saturated, horizontally aligned strips that sit inside a dark
    capsule. Requiring that capsule is what removes the arena floor from the picture — and it
    is also what makes the *team colour* of the strip trustworthy, which matters because the
    detector uses bars as ground truth for which side a blob belongs to.
    """

    MIN_HEIGHT_TILES = 0.035
    MAX_HEIGHT_TILES = 0.34
    MIN_WIDTH_TILES = 0.16
    MAX_WIDTH_TILES = 3.2
    #: minimum aspect ratio that separates a bar from a bright patch of floor
    MIN_ASPECT = 1.9

    def __init__(self, min_fill_ratio: float = 0.55) -> None:
        self.min_fill_ratio = min_fill_ratio

    def read(
        self,
        frame_bgr: np.ndarray,
        grid: ArenaGrid,
        ppt: float,
        min_width_tiles: Optional[float] = None,
        max_height_px: Optional[int] = None,
    ) -> List[HealthBar]:
        ax1, ay1, ax2, ay2 = grid.arena_pixel_bounds
        ax1, ay1 = max(0, ax1), max(0, ay1)
        ax2, ay2 = min(frame_bgr.shape[1], ax2), min(frame_bgr.shape[0], ay2)
        if ax2 - ax1 < 40 or ay2 - ay1 < 40:
            return []
        roi = frame_bgr[ay1:ay2, ax1:ax2]
        masks = self._colour_masks(roi)

        h_min = max(2, int(round(self.MIN_HEIGHT_TILES * ppt)))
        h_max = max(h_min + 1, int(round(self.MAX_HEIGHT_TILES * ppt)))
        if max_height_px is not None:
            h_max = max(h_min, int(max_height_px))
        w_tiles = self.MIN_WIDTH_TILES if min_width_tiles is None else float(min_width_tiles)
        w_min = max(4, int(round(w_tiles * ppt)))
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
                    # Only a *wide* blob can be "a bar with something stuck to it". A blob that
                    # is taller than it is wide is card artwork, and cutting a thin slice out of
                    # it manufactures a bar that is not there.
                    if bw < 1.6 * bh:
                        continue
                    strip = self._bar_rows(mask[by:by + bh, bx:bx + bw])
                    if strip is None:
                        continue
                    top, bottom = strip
                    by, bh = by + top, bottom - top
                if not (w_min <= bw <= w_max and h_min <= bh <= h_max):
                    continue
                if bw / float(max(1, bh)) < self.MIN_ASPECT:
                    continue
                filled = float(cv2.countNonZero(mask[by:by + bh, bx:bx + bw]))
                if filled / float(max(1, bw * bh)) < self.min_fill_ratio:
                    continue
                if not self._is_isolated(mask, bx, by, bw, bh):
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
    def _is_isolated(mask: np.ndarray, bx: int, by: int, bw: int, bh: int) -> bool:
        """Reject "bars" that are really a slice of a bigger coloured blob.

        A health bar floats a few pixels clear of the sprite it belongs to, with the game's dark
        outline between them. A card's own artwork does not: red body parts of a red-skinned
        unit form a thin horizontal run that looks exactly like a bar by bounding box, but the
        colour continues above and below it. Checking two pixels outside the candidate is a
        cheap way to tell those apart, and it matters a lot: bar colour is the single
        strongest side signal, so one phantom bar can flip a troop to the wrong team.
        """
        rows_above = by - 2
        rows_below = by + bh + 1
        if rows_above < 0 or rows_below >= mask.shape[0]:
            return True
        above = mask[rows_above, bx:bx + bw]
        below = mask[rows_below, bx:bx + bw]
        if above.size == 0 or below.size == 0:
            return True
        above_frac = float(cv2.countNonZero(above)) / float(bw)
        below_frac = float(cv2.countNonZero(below)) / float(bw)
        return not (above_frac > 0.5 and below_frac > 0.5)

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

        Health-bar reds and blues are strongly channel-separated, so two saturating
        subtractions are both faster and less range-fiddly than an HSV threshold band — and
        staying in uint8 keeps a 1080p arena at a couple of milliseconds.
        """
        b, g, r = cv2.split(roi)
        r_minus_b = cv2.subtract(r, b)     # saturates at 0, so "redder than blue" falls out
        r_minus_g = cv2.subtract(r, g)
        red = cv2.bitwise_and(
            (r_minus_g > 50).astype(np.uint8), (r_minus_b > 50).astype(np.uint8)
        )
        red = cv2.bitwise_and(red, (r > 105).astype(np.uint8)) * 255
        b_minus_r = cv2.subtract(b, r)
        blue = cv2.bitwise_and(
            (b_minus_r > 40).astype(np.uint8), (b > 105).astype(np.uint8)
        ) * 255
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


# ---------------------------------------------------------------------------------------------
# team classification
# ---------------------------------------------------------------------------------------------
@dataclass
class TeamVote:
    """Result of :meth:`TeamClassifier.classify`."""

    side: Side
    confidence: float
    evidence: Dict[str, float] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.6


class TeamClassifier:
    """Decide which player a sprite belongs to, with independent evidence and a real score.

    The old rule was "count hue-red pixels, count hue-blue pixels, default to enemy". It got
    three things wrong and each of them is a bug you can see in a match:

    * sprites with little saturation (grey armour, dark leather, brown wood) produced fewer
      than twelve usable pixels and fell into the ``ENEMY`` default — so *your* Miner,
      Skeleton Barrel and Goblin Gang were reported as enemy troops and the brain defended
      against its own push;
    * a sprite whose own artwork is red (Inferno Dragon, Skeleton Barrel's barrel, a red
      scarf) outvoted the team tint on top of it;
    * there was no notion of "I don't know", so a coin flip looked exactly like certainty.

    This classifier measures four signals and combines them:

    ``bar``     The unit health bar, which the *game itself* paints blue or red. Weight 3.0.
    ``ring``    The elliptical team ring drawn under every unit. Measured by taking the
                dominant red-vs-blue cast of an annulus around the sprite and **subtracting the
                sprite's own cast**, so a red dragon with a blue ring still reads as yours.
    ``chroma``  The body colour cast alone. Weak (a red unit is not necessarily an enemy), so
                it can never outvote the ring or the bar — it only breaks ties.
    ``motion``  Enemy units move down the screen, allies move up. Only used in the lanes, and
                only as a nudge, because a defending unit can walk backwards.

    ``confidence`` is the normalised weight of the winning side, and it is honest: with only a
    weak chroma signal it stays near 0.5 and the caller can keep the label it had.
    """

    #: how much of the sprite's own colour cast to remove from the ring measurement
    DECONFOUND = 1.0
    RING_RADII = (0.70, 0.82, 0.94, 1.06, 1.18, 1.30)
    MIN_RING_PIXELS = 40

    def classify(
        self,
        frame_bgr: np.ndarray,
        bbox: Box,
        mask: Optional[np.ndarray] = None,
        bar: Optional[HealthBar] = None,
        velocity: Optional[Tuple[float, float]] = None,
        tile_y: Optional[float] = None,
    ) -> TeamVote:
        evidence: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        body_mask = self._normalise_mask(mask, bbox)

        if bar is not None:
            # A bar is drawn by the game in team colour: the strongest single signal there is.
            evidence["bar"] = 1.0 if bar.side == Side.ALLY else -1.0
            weights["bar"] = 3.0

        # both casts are "+ means red", and red is the *opponent*, so they enter the vote
        # with a negative sign (the vote is + for ally).
        ring = self._ring_cast(frame_bgr, bbox, body_mask)
        if ring is not None:
            evidence["ring"] = float(np.clip(-ring * 2.2, -1.0, 1.0))
            weights["ring"] = 2.2

        chroma = self._body_cast(frame_bgr, bbox, body_mask)
        if chroma is not None:
            evidence["chroma"] = float(np.clip(-chroma * 1.6, -1.0, 1.0))
            weights["chroma"] = 0.9

        if velocity is not None and tile_y is not None:
            vy = float(velocity[1])
            # moving towards the enemy side (decreasing tile_y) is an ally, and vice versa;
            # ignore units parked near the river where the direction is meaningless
            if abs(vy) >= 0.25 and 3.0 <= tile_y <= 29.0:
                evidence["motion"] = float(np.clip(vy * 1.2, -1.0, 1.0))
                weights["motion"] = 0.7

        total = sum(weights.values())
        if total <= 0:
            return TeamVote(Side.ENEMY, 0.0, evidence)

        # evidence is +1 for ally, -1 for enemy
        score = sum(evidence[k] * weights[k] for k in evidence) / total
        side = Side.ALLY if score > 0 else Side.ENEMY
        confidence = float(min(1.0, abs(score)))
        evidence["score"] = round(float(score), 3)
        return TeamVote(side, round(confidence, 3), evidence)

    # -- signals -------------------------------------------------------------------
    @staticmethod
    def _normalise_mask(mask: Optional[np.ndarray], bbox: Box) -> Optional[np.ndarray]:
        """Resample the segmentation mask so it lines up with the (pixel-space) bbox.

        Segmentation runs at half resolution, so the mask handed in is smaller than the patch
        it describes. Silently indexing a 140px patch with a 70px mask throws; resizing once
        here keeps both signals aligned with what they measure.
        """
        if mask is None or not getattr(mask, "size", 0):
            return None
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        if mask.shape[:2] == (bh, bw):
            return mask
        return cv2.resize(mask, (bw, bh), interpolation=cv2.INTER_NEAREST)

    def _ring_cast(
        self, frame_bgr: np.ndarray, bbox: Box, mask: Optional[np.ndarray]
    ) -> Optional[float]:
        """Red-vs-blue cast of the team ring, de-confounded by the sprite's own colours.

        The game draws an elliptical decal under every deployed unit, coloured blue for the
        player and red for the opponent. It is the second most reliable side signal after the
        health bar, because it is *added by the renderer* rather than being part of the card's
        art — a red Inferno Dragon on your side still stands on a blue ring.

        Two details decide whether the measurement works on real footage:

        * the area searched must be the band around the sprite's **feet**, not a ring scaled by
          the sprite's height (that lands in the grass on anything tall), and
        * only pixels that are actually *team coloured* may vote. Grass is green, stone is grey,
          and both have a low red/blue "strength", so filtering on strength keeps them out and
          yields the ring instead of the arena floor.

        The sprite's own colour cast is measured separately and subtracted, so the card's own
        artwork (a red barrel, a blue cloak) cannot masquerade as a red or blue ring.
        """
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 10:
            return None
        pad = int(round(0.5 * max(w, h)))
        px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
        px2, py2 = min(frame_bgr.shape[1], x2 + pad), min(frame_bgr.shape[0], y2 + pad)
        roi = frame_bgr[py1:py2, px1:px2]
        if roi.size == 0:
            return None

        b = roi[:, :, 0].astype(np.float32)
        g = roi[:, :, 1].astype(np.float32)
        r = roi[:, :, 2].astype(np.float32)
        cast = (r - b) / (r + b + 30.0)                      # + red, - blue
        # "Team coloured" is *warm-magenta minus green*: it is positive for reds and for blues
        # and negative for grass, which is what keeps the arena floor out of the measurement.
        strength = (r + b - 2.0 * g) / (r + b + g + 30.0)

        # --- the sprite itself (for de-confounding, and to exclude from the ring band) ---
        body = None
        body_pixels = np.zeros(roi.shape[:2], bool)
        if mask is not None and mask.size:
            aligned = np.zeros(roi.shape[:2], np.uint8)
            mx, my = x1 - px1, y1 - py1
            sx, sy = max(0, -mx), max(0, -my)
            dx, dy = max(0, mx), max(0, my)
            w_avail = min(mask.shape[1] - sx, aligned.shape[1] - dx)
            h_avail = min(mask.shape[0] - sy, aligned.shape[0] - dy)
            if w_avail > 0 and h_avail > 0:
                aligned[dy:dy + h_avail, dx:dx + w_avail] = mask[sy:sy + h_avail, sx:sx + w_avail]
            body_pixels = aligned > 0
            if int(body_pixels.sum()) >= 40:
                body = float(np.median(cast[body_pixels]))

        # --- the band around the feet where the ring lives ---
        ys, xs = np.indices(roi.shape[:2])
        cx = 0.5 * (x1 + x2) - px1
        cy = float(y2) - py1
        in_band = (
            (np.abs(xs - cx) <= 0.95 * w)
            & (ys >= cy - 0.55 * h)
            & (ys <= cy + 0.12 * h)
        )
        if body_pixels.any():
            grown = cv2.dilate(
                body_pixels.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            in_band &= grown == 0

        # A ring is only trusted where it *sticks out past the sprite*: the two lateral arcs
        # beside the feet. Everything underneath the silhouette is unreachable, because a
        # card's own red or blue artwork sits there and would be read as a team colour.
        lateral = np.abs(xs - cx) >= 0.55 * w
        in_band &= lateral

        # The background baseline is what the *rest* of the band looks like: grass, stone or
        # floor. Subtracting it means a ring only has to be redder/bluer than its own
        # surroundings, which works on every map and lighting condition without calibration.
        baseline_candidates = in_band & (np.abs(strength) < 0.05)
        baseline = 0.0
        if int(baseline_candidates.sum()) >= 20:
            baseline = float(np.median(cast[baseline_candidates]))
        elif body is not None:
            baseline = body

        team_pixels = in_band & (strength >= 0.07) & (np.abs(cast - baseline) >= 0.10)
        left = team_pixels & (xs < cx)
        right = team_pixels & (xs >= cx)
        if int(left.sum()) < 8 or int(right.sum()) < 8:
            return None
        left_cast = float(np.median(cast[left] - baseline))
        right_cast = float(np.median(cast[right] - baseline))
        if left_cast * right_cast <= 0:
            # The two sides disagree: not a ring, just two unrelated coloured patches.
            return None
        values = np.concatenate([cast[left] - baseline, cast[right] - baseline])
        weights = strength[left | right]
        order = np.argsort(weights)
        strong = values[order[len(order) // 2:]]
        ring = float(np.median(strong))
        # Shape check: the coloured support must be wide and flat, like a decal seen from above.
        ys_t, xs_t = np.nonzero(team_pixels)
        span_x = float(xs_t.max() - xs_t.min() + 1)
        span_y = float(ys_t.max() - ys_t.min() + 1)
        if span_x < 0.9 * w or span_x < 0.9 * span_y:
            return None
        return ring

    @staticmethod
    def _body_cast(
        frame_bgr: np.ndarray, bbox: Box, mask: Optional[np.ndarray]
    ) -> Optional[float]:
        """Red-vs-blue cast of *the sprite itself* — i.e. of the pixels outside the mask.

        ``mask`` marks what segmentation believes is the unit, so the **sprite** is the part of
        the box the mask does *not* cover: a sprite's flat interior is often identical to the
        arena's own colour cast, and measuring inside the mask returns the grass. This value is
        a weak signal by design (a red card is not an enemy), and its main job is to be
        subtracted from the ring measurement so a unit's own artwork cannot masquerade as a
        team colour.
        """
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        patch = frame_bgr[max(0, y1):y2, max(0, x1):x2]
        if patch.size == 0:
            return None
        b = patch[:, :, 0].astype(np.float32)
        g = patch[:, :, 1].astype(np.float32)
        r = patch[:, :, 2].astype(np.float32)
        cast = (r - b) / (r + b + 30.0)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        if mask is not None and mask.size:
            resized = cv2.resize(mask, (patch.shape[1], patch.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0
            sprite_pixels = resized & (sat > 45) & (val > 35)
            if int(sprite_pixels.sum()) < 25:
                return None
            return float(np.median(cast[sprite_pixels]))

        sel = (sat > 60) & (val > 40)
        if int(sel.sum()) < 20:
            return None
        return float(np.median(cast[sel]))


# ---------------------------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------------------------
class BlobUnitDetector:
    """Segment troops out of the arena and describe them in tile coordinates.

    The pipeline is: background state -> detection mask -> connected components -> split and
    merge using health bars as anchors -> per-blob side/geometry/confidence.
    """

    MIN_AREA_TILES2 = 0.05
    MAX_AREA_TILES2 = 42.0
    MIN_HEIGHT_TILES = 0.30
    MAX_HEIGHT_TILES = 6.0
    MAX_WIDTH_TILES = 5.0
    #: a blob this flat is a spell trail / ring / shadow, not a unit body
    MAX_ASPECT = 3.2
    #: two bar-less fragments closer than this (in tiles) are one sprite seen twice
    MERGE_RADIUS_TILES = 0.85
    #: blobs below this solidity are likely hollow effects (a ring, a spell circle)
    MIN_SOLIDITY = 0.28

    #: segmentation runs on a half-resolution copy of the arena: a troop is 30+ px tall, so
    #: nothing is lost, and the per-frame cost drops by ~4x on a phone-sized capture
    WORK_SCALE = 0.5

    def __init__(
        self,
        background: Optional[ArenaBackgroundModel] = None,
        bar_reader: Optional[HealthBarReader] = None,
        diff_threshold: float = 20.0,
        team: Optional[TeamClassifier] = None,
    ) -> None:
        self.background = background if background is not None else ArenaBackgroundModel(
            diff_threshold=diff_threshold)
        self.bars = bar_reader if bar_reader is not None else HealthBarReader()
        self.team = team if team is not None else TeamClassifier()
        self.last_bars: List[HealthBar] = []
        self.last_dropped: int = 0
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
            self.background.set_tile_scale(cached * self.WORK_SCALE)
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
        self.background._prev = small.astype(np.float32).copy()
        self.background._foreground_age = np.zeros(small.shape[:2], np.int32)
        self.background._change_age = np.zeros(small.shape[:2], np.int32)
        self.background._settled = np.full(small.shape[:2], 99, np.int32)
        self.background.set_tile_scale(self.tiles_per_pixel(grid) * scale)
        self.background._frames_seen = 99
        return True

    # -- main entry -----------------------------------------------------------------
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
        self.background.set_tile_scale(ppt * scale)

        ready = self.background.is_ready
        state = self.background.state(small) if ready else np.full(
            small.shape[:2], BG_UNKNOWN, np.uint8)
        motion = self.background.motion(small) if ready else np.zeros(small.shape[:2], np.uint8)

        units: List[BlobUnit] = []
        if ready:
            # Bars are read in *big* coordinates: their colour is the least ambiguous team
            # evidence available, so they are also used as anchors for splitting blobs.
            self.last_bars = self.bars.read(frame_bgr, grid, ppt)
            mask = self._detection_mask(state, motion)
            units = self._from_mask(
                mask, motion, roi, small, grid, ppt, (ax1, ay1), scale, self.last_bars
            )
            # a blob that is only moving cannot be seen by the background model yet (it is
            # indistinguishable from arena until it moves), so it is added on top
            units = self._add_motion_units(units, motion, roi, small, grid, ppt, (ax1, ay1), scale)
        if observe:
            # only learn where we are *certain* it is background; UNKNOWN must not be adopted
            guard = np.where(state == BG_BACKGROUND, 0, 255).astype(np.uint8) if ready else None
            self.background.update(small, guard)
        return units

    # -- motion-only units -------------------------------------------------------
    #: a moving blob smaller than this (tiles) is a stray effect, not a troop
    MIN_MOTION_TILES = 0.55

    def _add_motion_units(
        self,
        units: List[BlobUnit],
        motion: np.ndarray,
        roi: np.ndarray,
        small: np.ndarray,
        grid: ArenaGrid,
        ppt: float,
        origin: Tuple[int, int],
        scale: float,
    ) -> List[BlobUnit]:
        """Pick up troops that the background model has not learned as "not arena" yet.

        A unit that walks into view is, pixel for pixel, indistinguishable from the floor until
        it moves. The static state mask only marks *certain* foreground, so on the first frames
        of a push these units are simply absent - the false negatives that make the bot miss an
        incoming push. Motion is the one signal that sees them immediately, so a moving blob
        that does not coincide with an already-reported unit becomes a low-confidence unit of
        its own.
        """
        if motion.size == 0 or not motion.any():
            return units
        big_motion = cv2.resize(
            motion, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        big_motion = cv2.morphologyEx(
            big_motion, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )
        contours, _ = cv2.findContours(big_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ax1, ay1 = origin
        min_area = (self.MIN_MOTION_TILES * ppt) ** 2
        existing = [u.bbox for u in units]
        added: List[BlobUnit] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < min_area:
                continue
            box = (x + ax1, y + ay1, x + ax1 + w, y + ay1 + h)
            if any(self._near(box, other, 0.5 * ppt) for other in existing):
                continue
            cx, cy = 0.5 * (box[0] + box[2]), box[3]
            tile_x, tile_y = grid.pixel_to_tile(cx, cy)
            if not (0.0 <= tile_x <= grid.TILES_X and 0.0 <= tile_y <= grid.TILES_Y):
                continue
            height_tiles = max(0.4, (box[3] - box[1]) / ppt)
            weight_class, name = classify_size(height_tiles)
            vote = self.team.classify(roi, box, None, None, None, tile_y)
            added.append(
                BlobUnit(
                    bbox=box,
                    area_px=int(w * h),
                    side=vote.side,
                    side_confidence=float(vote.confidence),
                    height_tiles=height_tiles,
                    width_tiles=max(0.3, w / ppt),
                    tile_x=float(tile_x),
                    tile_y=float(tile_y),
                    weight_class=weight_class,
                    fallback_label=name,
                    confidence=0.35,
                    source="motion",
                    team_evidence=dict(vote.evidence),
                )
            )
        if added:
            units = units + added
        return units

    # -- mask ------------------------------------------------------------------
    def _detection_mask(self, state: np.ndarray, motion: np.ndarray) -> np.ndarray:
        """Combine background state and motion into the mask handed to the blob labeller.

        * FRONT pixels (certainly not the arena) are always in.
        * UNKNOWN pixels are in when they are moving, or when they are enclosed by FRONT
          pixels (Hough-fill: a sprite's flat interior is unknown until it moves).
        """
        front = (state == BG_FOREGROUND)
        unknown = (state == BG_UNKNOWN)
        moving = motion > 0

        mask = np.zeros(state.shape, np.uint8)
        mask[front] = 255
        if unknown.any():
            # fill holes: everything the certain front does not reach from the border is inside
            # a sprite, so it belongs to that sprite
            filled = np.zeros(state.shape, np.uint8)
            filled[front] = 255
            ff = filled.copy()
            h, w = ff.shape
            flood = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, flood, (0, 0), 255)
            holes = cv2.bitwise_not(ff) & 255
            mask = cv2.bitwise_or(mask, holes)
            mask[moving & unknown] = 255
        return mask

    # -- blob building -------------------------------------------------------------
    def _from_mask(
        self,
        mask: np.ndarray,
        motion: np.ndarray,
        roi: np.ndarray,
        small: np.ndarray,
        grid: ArenaGrid,
        ppt: float,
        origin: Tuple[int, int],
        scale: float,
        bars: Sequence[HealthBar],
    ) -> List[BlobUnit]:
        ax1, ay1 = origin
        if mask.size == 0:
            return []
        k = 5 if scale < 1.0 else 3
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        inv = 1.0 / scale
        s_ppt = ppt * scale                     # tiles -> small pixels
        min_area = max(6.0, self.MIN_AREA_TILES2 * s_ppt * s_ppt)
        max_area = self.MAX_AREA_TILES2 * s_ppt * s_ppt

        # Bars in *small* coordinates, used both to split blobs and as side ground truth.
        bars_small = [
            (b,
             int(round((0.5 * (b.bbox[0] + b.bbox[2]) - ax1) * scale)),
             int(round((b.bbox[3] - ay1) * scale)))
            for b in bars
        ]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[Box, np.ndarray, Optional[HealthBar]]] = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = float(cv2.countNonZero(mask[by:by + bh, bx:bx + bw]))
            if area < min_area or area > max_area:
                self.last_dropped += 1
                continue
            h_tiles = bh / s_ppt
            w_tiles = bw / s_ppt
            if not (self.MIN_HEIGHT_TILES <= h_tiles <= self.MAX_HEIGHT_TILES):
                self.last_dropped += 1
                continue
            if w_tiles > self.MAX_WIDTH_TILES or bw / float(max(1, bh)) > self.MAX_ASPECT:
                # A wide flat thing is a spell splat, a ring or a shadow — unless a health bar
                # sits on it, in which case something real is inside.
                if not self._bars_inside((bx, by, bx + bw, by + bh), bars_small):
                    self.last_dropped += 1
                    continue
            candidates.append(((bx, by, bx + bw, by + bh), cnt, None))

        # --- split: one blob that carries two or more bars is several units ------------
        split_out: List[Tuple[Box, np.ndarray, Optional[HealthBar], Optional[Tuple[int, int]]]] = []
        for bbox, cnt, _ in candidates:
            inside = [entry for entry in bars_small
                      if bbox[0] - 2 <= entry[1] <= bbox[2] + 2
                      and bbox[1] - 2 * s_ppt <= entry[2] <= bbox[3] + 2]
            if len(inside) < 2:
                split_out.append((bbox, cnt, inside[0][0] if inside else None, None))
                continue
            for sub_box, sub_cnt, sub_bar, anchor in self._split_with_bars(mask, bbox, inside, s_ppt):
                split_out.append((sub_box, sub_cnt, sub_bar, anchor))

        # --- merge: bar-less fragments that obviously belong to one sprite ---------------
        merged = self._merge_fragments(split_out, s_ppt)

        out: List[BlobUnit] = []
        for bbox_s, cnt, bar, anchor, parts in merged:
            bx, by, bx2, by2 = bbox_s
            x1 = int(round(bx * inv)) + ax1
            y1 = int(round(by * inv)) + ay1
            x2 = int(round(bx2 * inv)) + ax1
            y2 = int(round(by2 * inv)) + ay1
            w_tiles = (bx2 - bx) * inv / ppt
            h_tiles = (by2 - by) * inv / ppt

            blob_mask = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
            patch = roi[max(0, y1 - ay1):max(1, y2 - ay1), max(0, x1 - ax1):max(1, x2 - ax1)]
            small_mask = blob_mask[by:by2, bx:bx2]

            area = float(cv2.countNonZero(small_mask))
            solidity = area / float(max(1.0, (bx2 - bx) * (by2 - by)))

            foot_x, foot_y = 0.5 * (x1 + x2), float(y2)
            tile_x, tile_y = grid.pixel_to_tile(foot_x, foot_y)

            bar = bar if bar is not None else self._bar_for(x1, y1, x2, y2, bars, ppt)
            weight, label, h_tiles, w_tiles = self._measure_tiles(
                bbox_s, anchor, cnt, bar, ppt, inv, w_tiles, h_tiles)

            vote = self.team.classify(
                roi, (x1 - ax1, y1 - ay1, x2 - ax1, y2 - ay1),
                mask=small_mask, bar=bar, tile_y=tile_y,
            )

            mean_bgr = (
                float(patch[:, :, 0].mean()) if patch.size else 0.0,
                float(patch[:, :, 1].mean()) if patch.size else 0.0,
                float(patch[:, :, 2].mean()) if patch.size else 0.0,
            )

            # Confidence combines segmentation quality, team certainty and bar support.
            conf = 0.35 + 0.25 * min(1.0, solidity) + 0.25 * vote.confidence + (0.15 if bar else 0.0)
            if area < 2.0 * min_area:
                conf -= 0.1
            out.append(
                BlobUnit(
                    bbox=(x1, y1, x2, y2),
                    area_px=int(area * inv * inv),
                    side=vote.side,
                    side_confidence=vote.confidence,
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
                    confidence=round(float(max(0.0, min(1.0, conf))), 3),
                    parts=parts,
                    source="split" if parts > 1 else ("merged" if bar is None and parts > 1 else "segmentation"),
                    team_evidence=vote.evidence,
                    bars=[bar] if bar is not None else [],
                )
            )
        return out

    @staticmethod
    def _bars_inside(bbox: Box, bars_small) -> List[Tuple[HealthBar, int, int]]:
        bx1, by1, bx2, by2 = bbox
        out = []
        for bar, cx, cy in bars_small:
            if bx1 - 2 <= cx <= bx2 + 2 and by1 - 2 <= cy <= by2 + 2:
                out.append((bar, cx, cy))
        return out

    def _split_with_bars(
        self, mask: np.ndarray, bbox: Box, inside, s_ppt: float
    ) -> List[Tuple[Box, np.ndarray, Optional[HealthBar], Tuple[int, int]]]:
        """Voronoi-split one blob at the health-bar anchors inside it.

        A cluster of six skeletons standing together satisfies every geometric filter as one
        blob, and ends up treated as a single medium unit — so the brain sends a Mega Knight at
        what looks like a threat but is really a swarm, or ignores it. The bars are already
        there, one per unit, so they are the natural split points.
        """
        bx1, by1, bx2, by2 = bbox
        sub = mask[by1:by2, bx1:bx2]
        ys, xs = np.nonzero(sub)
        if xs.size < 20:
            return []
        seeds = np.array([(cx - bx1, cy - by1) for _bar, cx, cy in inside], np.float32)
        if seeds.shape[0] < 2:
            return []
        d = (
            (xs[:, None].astype(np.float32) - seeds[None, :, 0]) ** 2
            + (ys[:, None].astype(np.float32) - seeds[None, :, 1]) ** 2
        )
        labels = np.argmin(d, axis=1)

        out: List[Tuple[Box, np.ndarray, Optional[HealthBar], Tuple[int, int]]] = []
        for label in range(seeds.shape[0]):
            sel = labels == label
            if int(sel.sum()) < 12:
                continue
            lx1, lx2 = int(xs[sel].min()), int(xs[sel].max()) + 1
            ly1, ly2 = int(ys[sel].min()), int(ys[sel].max()) + 1
            piece = np.zeros(sub.shape, np.uint8)
            piece[ys[sel], xs[sel]] = 255
            piece = cv2.morphologyEx(
                piece, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            )
            cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            pbx, pby, pbw, pbh = cv2.boundingRect(cnt)
            bar = inside[label][0]
            # the anchor travels with the piece: a Voronoi cell is larger than the sprite, so
            # the cell cannot be used to measure the unit's size (see _measure_tiles)
            anchor = (int(round(seeds[label][0])), int(round(seeds[label][1])))
            out.append(((bx1 + pbx, by1 + pby, bx1 + pbx + pbw, by1 + pby + pbh), cnt, bar, anchor))
        return out

    def _merge_fragments(
        self,
        items: List[Tuple[Box, np.ndarray, Optional[HealthBar], Optional[Tuple[int, int]]]],
        s_ppt: float,
    ) -> List[Tuple[Box, np.ndarray, Optional[HealthBar], int, Optional[Tuple[int, int]]]]:
        """Join bar-less fragments that are obviously the same sprite, and count parts.

        Splitting one sprite into two detections is the single most damaging vision error for
        the brain: it doubles the perceived threat, so it over-defends one lane and leaves the
        other open. Fragments are joined when they are close, when their boxes overlap, or when
        they are horizontally stacked in the same column (a body plus a detached weapon).
        """
        if not items:
            return []
        # bbox, cnt, bar, anchor, parts   (the split anchor is carried through a merge)
        entries = [list(item) + [1] for item in items]
        radius = self.MERGE_RADIUS_TILES * s_ppt
        changed = True
        while changed:
            changed = False
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    a, b = entries[i], entries[j]
                    # never merge across bars: a bar means a real, separate unit
                    if a[2] is not None and b[2] is not None:
                        continue
                    if a[2] is not None or b[2] is not None:
                        # only merge a bar-less fragment *into* the anchored unit, inside it
                        if not self._near(a[0], b[0], radius * 0.6):
                            continue
                    else:
                        if not self._near(a[0], b[0], radius):
                            continue
                        if not self._same_sprite(a[0], b[0], s_ppt):
                            continue
                    box = (
                        min(a[0][0], b[0][0]), min(a[0][1], b[0][1]),
                        max(a[0][2], b[0][2]), max(a[0][3], b[0][3]),
                    )
                    # sanity: a merged unit is never wider than ~2.5 tiles or taller than 5
                    if (box[2] - box[0]) > 2.6 * s_ppt or (box[3] - box[1]) > 5.2 * s_ppt:
                        continue
                    keep = a if a[2] is not None else b
                    other = b if keep is a else a
                    merged_box = keep[0] if keep[2] is not None else box
                    if keep[2] is not None:
                        # extend the anchored unit's box to include the fragment
                        merged_box = (
                            min(keep[0][0], other[0][0]), min(keep[0][1], other[0][1]),
                            max(keep[0][2], other[0][2]), max(keep[0][3], other[0][3]),
                        )
                    # keep[3] is the anchor now, so the part count is keep[4]
                    entries[i] = [merged_box, keep[1], keep[2], keep[3], keep[4] + other[4]]
                    del entries[j]
                    changed = True
                    break
                if changed:
                    break
        return [tuple(e) for e in entries]  # type: ignore[return-value]

    #: two adjoining bar-less fragments are only one sprite if the small one is a *minor*
    #: part of the result. A Minion Horde is five similar-sized blobs touching each other, and
    #: merging those collapses a swarm into a single "unit" - which is exactly the kind of
    #: phantom the brain cannot recover from.
    MERGE_AREA_RATIO = 0.35

    def _same_sprite(self, a: Box, b: Box, s_ppt: float) -> bool:
        area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        smaller, larger = min(area_a, area_b), max(area_a, area_b)
        if smaller / larger > self.MERGE_AREA_RATIO:
            return False
        # a detached weapon or shield hovers close to the body, near the top of the silhouette.
        # It also has to be small: a fragment a third of the body's area is another troop, not
        # a piece of this one - and dropping its box into this unit's box would both hide that
        # troop and read a stray coloured pixel as this unit's team ring.
        small, big = (a, b) if area_a < area_b else (b, a)
        cx = 0.5 * (small[0] + small[2]) - 0.5 * (big[0] + big[2])
        cy = 0.5 * (small[1] + small[3]) - 0.5 * (big[1] + big[3])
        if abs(cx) > 0.75 * (big[2] - big[0]) or abs(cy) > 0.55 * (big[3] - big[1]):
            return False
        big_h = max(1.0, big[3] - big[1])
        overlap_x = min(small[2], big[2]) - max(small[0], big[0])
        if overlap_x <= 0:
            gap = max(big[0] - small[2], small[0] - big[2])
            if gap > 0.45 * s_ppt:
                return False
        return small[3] > big[1] + 0.35 * big_h

    @staticmethod
    def _near(a: Box, b: Box, radius: float) -> bool:
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        if ix > 0 and iy > 0:
            inter = ix * iy
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            return inter >= 0.25 * smaller
        cx_a, cy_a = 0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3])
        cx_b, cy_b = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        return math.hypot(cx_a - cx_b, cy_a - cy_b) <= radius

    #: fallback window (tiles) used to size a split piece that has no bar to measure
    ANCHOR_BODY_TILES = 0.9
    #: below this bar width (tiles) a bar belongs to a swarm-sized unit, full stop
    SWARM_BAR_TILES = 0.48

    @staticmethod
    def _measure_tiles(
        box_small: Box,
        anchor: Optional[Tuple[int, int]],
        contour: np.ndarray,
        bar: Optional[HealthBar],
        ppt: float,
        inv: float,
        w_tiles: float,
        h_tiles: float,
    ) -> Tuple[WeightClass, str, float, float]:
        """Size a unit, and return ``(weight, label, height_tiles, width_tiles)``.

        Measuring the *bounding cell* is right for an un-split blob and wrong for a split one: a
        Voronoi cell is drawn huge to fill the space around its anchor, so every member of a
        split swarm measured about 1.2-1.3 tiles against a 0.7-tile sprite and was reported as a
        **medium** unit. The brain then treats a Skeleton Army as five medium troops - the wrong
        elixir trade, every time.

        So a split piece is measured from the pixels that are actually near its anchor instead,
        and the bar gets the final say in *both* directions: a bar narrower than a light unit is
        a swarm, and the old code could only ever use the bar to make a unit *bigger*.
        """
        rank = {WeightClass.SWARM: 0, WeightClass.LIGHT: 1, WeightClass.MEDIUM: 2,
                WeightClass.HEAVY_TANK: 3}

        if anchor is not None:
            # contour points are (x, y) in the *small* (half-resolution) coordinate space, the
            # same space the anchor is expressed in. The silhouette's extreme points are on its
            # outline, so the outline bounding box inside the anchor window is exact.
            points = np.asarray(contour, np.float32).reshape(-1, 2)
            xs, ys = points[:, 0], points[:, 1]
            if xs.size >= 8:
                cx = float(anchor[0])
                cy = float(anchor[1])
                half = BlobUnitDetector.ANCHOR_BODY_TILES * ppt
                near = (np.abs(xs - cx) <= half) & (np.abs(ys - cy) <= half)
                # keep the dense core under the anchor; if the piece is a thin sliver the
                # windowed measurement degenerates, so fall back to the contour bounds
                if int(near.sum()) >= 8:
                    w_tiles = float(xs[near].max() - xs[near].min() + 1) * inv / ppt
                    h_tiles = float(ys[near].max() - ys[near].min() + 1) * inv / ppt

        weight, label = classify_size(h_tiles)
        bar_tiles = bar.total_width / ppt if bar is not None else None
        if bar_tiles is not None:
            bar_weight, bar_label = classify_by_bar_width(bar_tiles)
            if bar_tiles < BlobUnitDetector.SWARM_BAR_TILES:
                # a narrow bar is a small unit, whatever the blob says
                weight, label = WeightClass.SWARM, bar_label
                h_tiles = min(h_tiles, BlobUnitDetector.SWARM_BAR_TILES * 2.0)
            elif anchor is not None:
                # A split piece exists because *this unit's bar* was found inside a bigger blob,
                # and the game draws exactly one bar per unit: its width is the cleanest size
                # evidence available, and the cell around it is inflated by construction. Use
                # the bar, and report the height implied by it rather than the cell's.
                weight, label = bar_weight, bar_label
                h_tiles = _height_for_weight(weight)
            elif rank.get(bar_weight, 2) > rank.get(weight, 2) and h_tiles < 3.0:
                # clipped sprite (partly behind a tower or another unit): the bar knows better
                weight, label = bar_weight, bar_label
        return weight, label, h_tiles, w_tiles

    @staticmethod
    def _bar_for(x1: int, y1: int, x2: int, y2: int, bars: Sequence[HealthBar],
                 ppt: float) -> Optional[HealthBar]:
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
            if not (0.20 <= width_ratio <= 1.8):
                continue
            score = (
                -abs(bar_cx - cx)
                - abs(dy - 0.5 * ppt)
                - 20.0 * abs(math.log(max(1e-3, width_ratio)))
            )
            if score > best_score:
                best_score = score
                best = bar
        return best


@dataclass
class DetectorStats:
    """Cheap counters to make the detector's behaviour visible in logs/tests."""

    frames: int = 0
    units: int = 0
    bars: int = 0
    identified: int = 0
    per_side: Dict[str, int] = field(default_factory=dict)
    split: int = 0
    merged: int = 0
    low_confidence: int = 0

    def add(
        self,
        units: Sequence[BlobUnit],
        bars: int = 0,
        identified: int = 0,
    ) -> None:
        self.frames += 1
        self.units += len(units)
        self.bars += bars
        self.identified += identified
        for unit in units:
            key = unit.side.value
            self.per_side[key] = self.per_side.get(key, 0) + 1
            if unit.parts > 1:
                self.split += 1
            if unit.confidence < 0.45:
                self.low_confidence += 1
