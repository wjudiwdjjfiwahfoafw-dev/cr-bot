"""
Automatic screen calibration for Clash Royale on a real device.

The bot needs four things to line up with the actual game:

1. **Where the arena is** — so a tactical tile becomes the right screen pixel.
2. **Where the four hand cards are** — so a card is grabbed from the right slot.
3. **Where the elixir bar is** — so elixir is read correctly.
4. **Where the river is** — so a deploy tap never lands on the illegal half.

The original code hard-coded all four as fractions of a 720x1280 frame. That is fine for a
9:16 emulator and is why the project's own benchmark passes, but on a real phone every one
of those constants is wrong: different aspect ratio, different HUD placement, a capture
stream that may be downscaled, and a perspective camera over the arena.

This module measures the layout from a live battle frame using anchors that are genuinely
stable in Clash Royale:

* the **river** — a narrow, wide coloured band (cyan, magenta or lava-like);
* three or four **magenta elixir-cost badges** on the hand cards — compact discs whose
  spacing helps estimate the card row (obscured or grey badges can prevent a fit);
* the **elixir bar** — the magenta pip strip directly below the card row;
* the **Princess Towers** — thin red/blue health bars, used when they can be found.

From the towers we solve a **homography**, which models the real perspective camera and keeps
every tile accurate (the arena is measurably not affine: on a 720x1280 frame the river sits
~59 px below the midpoint of the two tower rows). One unknown remains — how far a floating
health bar sits above its tower base — and it is solved by requiring the projected river to
coincide with the river actually detected.

Detection quality varies with skin, HUD state and animation frames, so calibration degrades
through an explicit ladder (4 towers -> 2 towers + river -> river + card row) and always
reports a heuristic confidence. ``scripts/calibrate_device.py --manual`` lets you mark four
points yourself; inspect the preview because neither path guarantees an exact mapping.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaCalibration, ArenaGrid
from cr_bot.arena.projection import (
    HomographyProjection,
    PRINCESS_TILES,
    Projection,
    RectProjection,
    TILES_X,
    TILES_Y,
    solve_homography,
)

logger = logging.getLogger(__name__)

# Bump when automatic anchor geometry changes incompatibly. Old manual calibrations are
# never invalidated solely by this number.
AUTO_CALIBRATION_REVISION = 1

Point = Tuple[float, float]

#: Seed for the vertical offset (in tiles) of a floating health bar above its tower base.
DEFAULT_HP_BAR_TILE_OFFSET = 0.9

#: Clash Royale hand-card aspect ratio (width / height), measured from a real frame.
CARD_ASPECT = 128.0 / 115.0


@dataclass
class CalibrationResult:
    """Outcome of a calibration attempt."""

    calibration: ArenaCalibration
    confidence: float = 0.0
    report: List[str] = field(default_factory=list)
    towers_px: Dict[str, Point] = field(default_factory=dict)
    stage: str = "none"

    @property
    def ok(self) -> bool:
        return self.confidence >= 0.55

    def summary(self) -> str:
        return "\n".join(self.report)


class DeviceCalibrator:
    """Derives :class:`ArenaCalibration` for the frame a device is currently producing."""

    def __init__(self, manual_towers: Optional[Dict[str, Point]] = None) -> None:
        self.manual_towers = manual_towers or {}

    # ==================================================================================
    # Public entry point
    # ==================================================================================
    def calibrate(self, frame_bgr: np.ndarray) -> CalibrationResult:
        h, w = frame_bgr.shape[:2]
        report: List[str] = [f"frame {w}x{h}"]

        content = self.detect_content_box(frame_bgr)
        if content != (0, 0, w, h):
            report.append(f"content box {content}")
        ox, oy = content[0], content[1]
        work = frame_bgr[content[1] : content[3], content[0] : content[2]]
        ch, cw = work.shape[:2]

        river_y = self.detect_river(work)
        hand = self.detect_hand_row(work)
        elixir = self.detect_elixir_bar(work, hand)
        bars = self.detect_hp_bars(work, river_y)

        report.append(
            "anchors: river={}, cards={}, elixir={}, tower_bars={}".format(
                "?" if river_y is None else f"{river_y:.0f}",
                "?" if hand is None else f"{len(hand['centers'])}@{hand['centers'][0][1]:.0f}",
                "?" if elixir is None else f"y{elixir[2]:.0f}",
                len(bars),
            )
        )

        # ---- choose the strongest projection the detectors support -------------------
        projection: Optional[Projection] = None
        residual = 0.0
        confidence = 0.0
        stage = "none"
        towers_px: Dict[str, Point] = {}

        if self.manual_towers:
            towers_px = {k: tuple(map(float, v)) for k, v in self.manual_towers.items()}
            projection, drift = self._fit_from_towers(towers_px, river_y)
            if projection is not None:
                confidence, stage = 0.98, "manual-4-tower"
                report.append(f"manual towers -> river drift {drift:.1f}px")
        else:
            best = self._best_tower_fit(bars, river_y, cw, ch)
            if best is not None:
                projection, towers_px, drift, fit_cost = best
                confidence = self._score(len(towers_px), river_y, drift, fit_cost)
                stage = "auto-4-tower"
                report.append(
                    f"auto towers {sorted(towers_px)}: river drift {drift:.0f}px, fit cost {fit_cost:.2f}, conf {confidence:.2f}"
                )
                if confidence < 0.5:
                    projection = None
                    report.append("auto tower fit rejected -> degrading")

            if projection is None:
                assigned = self._assign_towers(bars, river_y)
                pair = self._ally_pair(assigned, river_y)
                if pair is not None:
                    projection = self._fit_symmetric(pair, river_y, cw, ch, hand)
                    if projection is not None:
                        confidence, stage = 0.60, "auto-2-tower+river"
                        report.append("2 ally towers + river -> symmetric model")

        if projection is None:
            rect = self._rect_from_anchors(cw, ch, river_y, hand)
            projection = RectProjection(*rect)
            confidence = 0.48 if river_y is not None else 0.22
            stage = "anchor-rect"
            report.append(f"fallback rect {tuple(int(round(v)) for v in rect)}")

        # ---- local -> full frame ------------------------------------------------------
        projection = self._offset_projection(projection, ox, oy)

        hand_slots = None
        hand_card_size = None
        if hand is not None:
            hand_slots = [(x + ox, y + oy) for x, y in hand["centers"]]
            hand_card_size = (float(hand["card_w"]), float(hand["card_h"]))
            confidence = min(1.0, confidence + 0.06)

        elixir_px = None
        if elixir is not None:
            elixir_px = (elixir[0] + ox, elixir[1] + ox, elixir[2] + oy)
            confidence = min(1.0, confidence + 0.06)

        if river_y is not None:
            river_px = float(river_y + oy)
        else:
            river_px = float(projection.tile_to_pixel(TILES_X / 2.0, 16.0)[1])

        cal = ArenaCalibration(
            frame_size=(w, h),
            projection=projection,
            hand_slots_px=hand_slots,
            hand_card_size_px=hand_card_size,
            elixir_px=elixir_px,
            river_px_y=river_px,
            auto_revision=AUTO_CALIBRATION_REVISION,
            source="manual" if self.manual_towers else "auto",
            residual_px=float(residual),
            notes=" | ".join(report),
        )
        self._sync_ratios(cal, w, h)

        if river_y is not None:
            drift = abs(projection.tile_to_pixel(TILES_X / 2.0, 16.0)[1] - river_px)
            report.append(f"river drift {drift:.0f}px")
            if drift > h * 0.06:
                confidence *= 0.7
                report.append("river drift too large -> confidence reduced")

        return CalibrationResult(
            calibration=cal,
            confidence=float(max(0.0, min(1.0, confidence))),
            report=report,
            towers_px=towers_px,
            stage=stage,
        )

    # ==================================================================================
    # Anchors
    # ==================================================================================
    def detect_content_box(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Strip solid black letterbox / pillarbox bars that some devices add."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row_dark = (gray < 12).mean(axis=1)
        col_dark = (gray < 12).mean(axis=0)

        top = 0
        while top < h // 4 and row_dark[top] > 0.97:
            top += 1
        bottom = h
        while bottom > 3 * h // 4 and row_dark[bottom - 1] > 0.97:
            bottom -= 1
        left = 0
        while left < w // 4 and col_dark[left] > 0.97:
            left += 1
        right = w
        while right > 3 * w // 4 and col_dark[right - 1] > 0.97:
            right -= 1
        if right - left < w * 0.5 or bottom - top < h * 0.5:
            return 0, 0, w, h
        return left, top, right, bottom

    def detect_river(self, frame: np.ndarray) -> Optional[float]:
        """Find a *narrow, wide* water/lava band, or abstain when none is visible.

        Arena skins are not all cyan. On a real magenta-river map the old "largest cyan
        run" was 150 px below the river because the entire cyan floor qualified as water.
        Test cyan, magenta and bright lava, but reject broad coloured regions instead of
        assuming the largest region is a river. This is an anchor, not proof of calibration;
        unusual skins and overlays can still require manual calibration.
        """
        h, w = frame.shape[:2]
        if h < 40 or w < 40:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo, hi = int(h * 0.25), int(h * 0.72)
        masks = (
            ((86, 95, 95), (112, 255, 255)),   # blue water
            ((138, 110, 115), (179, 255, 255)), # magenta river
            ((5, 125, 130), (33, 255, 255)),   # bright lava
        )
        candidates: List[Tuple[float, float]] = []
        for lower, upper in masks:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            counts = (mask > 0).sum(axis=1).astype(np.float32) / float(w)
            # Smooth tiny gaps without merging the river with an expanse of similarly
            # coloured floor. A bridge may interrupt a row but not the whole band.
            band = cv2.blur(counts[lo:hi].reshape(-1, 1), (1, 3)).ravel()
            rows = np.flatnonzero(band >= 0.15)
            if not rows.size:
                continue
            for run in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1):
                length = len(run)
                if length < max(2, int(h * 0.003)) or length > h * 0.08:
                    continue
                strength = float(np.max(band[run]))
                if strength < 0.25:
                    continue
                # Broad, strong horizontal coverage is much more river-like than a
                # coloured unit, but favour a band rather than a single noisy row.
                score = strength * (length / float(h)) ** 0.5
                candidates.append((score, float(lo + run.mean())))
        return max(candidates)[1] if candidates else None

    def detect_hand_row(self, frame: np.ndarray) -> Optional[dict]:
        """Locate the four hand cards via their magenta elixir-cost badges.

        The badges are small, saturated discs at the bottom edge of each card, laid out at a
        perfectly regular pitch. They are far more reliable than trying to segment the card
        artwork itself (which can be arbitrarily dark or pale).
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        magenta = cv2.inRange(hsv, np.array([138, 105, 115]), np.array([176, 255, 255]))
        magenta = cv2.morphologyEx(magenta, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        n, _lab, stats, cents = cv2.connectedComponentsWithStats(magenta, 8)
        discs: List[Tuple[float, float, float]] = []
        # Scale badge bounds with capture width instead of assuming 608 px frames.
        min_badge = max(5.0, 14.0 * w / 608.0)
        max_badge = max(min_badge + 1.0, 64.0 * w / 608.0)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if y < h * 0.80:
                continue
            if not (min_badge <= bw <= max_badge and min_badge <= bh <= max_badge):
                continue
            if not (0.6 <= bw / float(max(1, bh)) <= 1.7):
                continue
            if area < 0.35 * bw * bh:
                continue
            discs.append((float(cents[i][0]), float(cents[i][1]), float(max(bw, bh))))
        if len(discs) < 3:
            return None

        # A transition or an unaffordable card can leave a grey/empty badge. Fit three
        # *or* four badges to the regular hand pitch and infer the missing centre. The
        # old loose four-disc fit sometimes treated the small "next card" badge as slot
        # one and returned wildly uneven centres. Refuse poorly spaced candidates rather
        # than writing bad tap coordinates into the persistent calibration.
        discs.sort(key=lambda d: d[0])
        if len(discs) > 12:
            discs = sorted(discs, key=lambda d: d[2], reverse=True)[:12]
            discs.sort(key=lambda d: d[0])
        best = None
        for count in (4, 3):
            if len(discs) < count:
                continue
            for group in itertools.combinations(discs, count):
                for indices in itertools.combinations(range(4), count):
                    xs = np.asarray([d[0] for d in group], dtype=np.float64)
                    idx = np.asarray(indices, dtype=np.float64)
                    pitch, start = np.polyfit(idx, xs, 1)
                    if not (0.14 * w <= pitch <= 0.23 * w):
                        continue
                    if not (0.22 * w <= start <= 0.40 * w and
                            0.78 * w <= start + 3 * pitch <= 0.99 * w):
                        continue
                    residual = float(np.max(np.abs(xs - (start + pitch * idx))))
                    if residual > max(3.0, pitch * 0.055):
                        continue
                    ys = [d[1] for d in group]
                    # One selected card rises. Reject unrelated rows but allow that lift.
                    y_spread = float(max(ys) - min(ys))
                    if y_spread > h * 0.055:
                        continue
                    sizes = [d[2] for d in group]
                    size_cost = float(np.std(sizes)) / max(1.0, float(np.mean(sizes)))
                    cost = residual / pitch + y_spread / h * 0.5 + size_cost * 0.1
                    if best is None or cost < best[0]:
                        best = (cost, group, float(start), float(pitch), count)
            # Prefer a valid four-badge fit to any inferred three-badge fit.
            if best is not None:
                break
        if best is None:
            return None

        _, group, start, pitch, count = best
        card_w = pitch * 0.94
        card_h = card_w / CARD_ASPECT
        badge_cy = float(np.median([d[1] for d in group]))
        card_cy = badge_cy - card_h * 0.47
        return {
            "centers": [(start + i * pitch, card_cy) for i in range(4)],
            "card_w": card_w,
            "card_h": card_h,
            "badge_cy": badge_cy,
            "pitch": pitch,
            "top": card_cy - card_h / 2.0,
            "bottom": card_cy + card_h / 2.0,
            "observed_badges": count,
        }

    def detect_elixir_bar(
        self, frame: np.ndarray, hand: Optional[dict]
    ) -> Optional[Tuple[float, float, float]]:
        """Return (x_start, x_end, y) of the elixir bar.

        Only the *filled* pips are magenta, so the bar's right-hand extent is taken from the
        card row, which Clash Royale draws immediately above it at the same width.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        magenta = cv2.inRange(hsv, np.array([136, 100, 110]), np.array([176, 255, 255]))

        top = int(h * 0.92) if hand is None else int(min(h - 2, hand["bottom"] + 2))
        band = magenta[top:h, :]
        if band.size == 0:
            return None
        rows = band.sum(axis=1) / 255.0
        if rows.max() < w * 0.05:
            return None
        y = top + int(np.argmax(rows))

        strip = magenta[max(0, y - 3) : min(h, y + 4), :]
        cols = np.where(strip.sum(axis=0) > 0)[0]
        if cols.size < 10:
            return None
        fill_left, fill_right = float(cols.min()), float(cols.max())

        if hand is not None:
            # Clash Royale draws the bar directly under the card row at effectively the same
            # width. Trusting the card row keeps HUD clutter on the far left (the "next card"
            # badge) from being mistaken for the start of the bar.
            card_left = hand["centers"][0][0] - hand["card_w"] / 2.0
            card_right = hand["centers"][3][0] + hand["card_w"] / 2.0
            x1 = max(0.0, card_left - w * 0.02)
            x2 = min(float(w), max(fill_right, card_right))
        else:
            x1, x2 = fill_left, fill_right

        if x2 - x1 < w * 0.25:
            return None
        return x1, x2, float(y)

    def detect_hp_bars(self, frame: np.ndarray, river_y: Optional[float]) -> List[dict]:
        """Find thin, saturated tower health bars (any team colour)."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks = [
            cv2.inRange(hsv, np.array([92, 120, 105]), np.array([128, 255, 255])),  # blue
            cv2.inRange(hsv, np.array([165, 120, 105]), np.array([179, 255, 255])),  # red/magenta
            cv2.inRange(hsv, np.array([0, 130, 110]), np.array([11, 215, 255])),  # red wrap
        ]
        bars: List[dict] = []
        for mask in masks:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
            n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if not (w * 0.07 <= bw <= w * 0.28):
                    continue
                if not (4 <= bh <= max(8, h * 0.022)):
                    continue
                aspect = bw / float(max(1, bh))
                if not (4.0 <= aspect <= 16.0):
                    continue
                if area < bw * 2.0:
                    continue
                cy = float(cents[i][1])
                if river_y is not None and abs(cy - river_y) < h * 0.03:
                    continue  # river water reads as blue
                if y < h * 0.08 or y > h * 0.86:
                    continue
                bars.append({"cx": float(cents[i][0]), "cy": cy, "w": float(bw), "h": float(bh), "area": int(area)})

        # De-duplicate bars found by more than one mask.
        deduped: List[dict] = []
        for bar in sorted(bars, key=lambda b: b["area"], reverse=True):
            if any(abs(bar["cx"] - o["cx"]) < 8 and abs(bar["cy"] - o["cy"]) < 8 for o in deduped):
                continue
            deduped.append(bar)
        return deduped

    # ==================================================================================
    # Fitting
    # ==================================================================================
    def _assign_towers(self, bars: List[dict], river_y: Optional[float]) -> Dict[str, Point]:
        if river_y is None or not bars:
            return {}
        enemy = [b for b in bars if b["cy"] < river_y]
        ally = [b for b in bars if b["cy"] >= river_y]
        out: Dict[str, Point] = {}
        for group, prefix in ((enemy, "enemy"), (ally, "ally")):
            if len(group) < 2:
                continue
            group = sorted(group, key=lambda b: b["cx"])
            # Prefer the outermost pair, which matches the two Princess Towers.
            out[f"{prefix}_left"] = (group[0]["cx"], group[0]["cy"])
            out[f"{prefix}_right"] = (group[-1]["cx"], group[-1]["cy"])
        return out

    def _ally_pair(self, towers: Dict[str, Point], river_y: Optional[float]) -> Optional[Tuple[Point, Point]]:
        if "ally_left" in towers and "ally_right" in towers:
            return towers["ally_left"], towers["ally_right"]
        return None

    def _fit_from_towers(
        self, towers_px: Dict[str, Point], river_y: Optional[float]
    ) -> Tuple[Optional[Projection], float]:
        """Solve the homography, choosing the health-bar offset by matching the river.

        Note that a four-point homography reproduces its own four points *exactly*, so the
        reprojection residual carries no information about whether the fit is right. The
        independent observable is the river: the projected tile-16 line must land on the
        water we actually detected. That, plus the geometric regularity of the tower quad,
        is what makes the fit trustworthy — or exposes it as wrong.
        """
        needed = ["enemy_left", "enemy_right", "ally_left", "ally_right"]
        if not all(k in towers_px for k in needed):
            return None, 0.0

        offsets = [DEFAULT_HP_BAR_TILE_OFFSET] if river_y is None else list(np.linspace(0.1, 2.4, 47))
        best: Optional[Tuple[float, HomographyProjection]] = None
        for dy in offsets:
            src = [(PRINCESS_TILES[k][0], PRINCESS_TILES[k][1] - dy) for k in needed]
            dst = [tuple(towers_px[k]) for k in needed]
            try:
                h_mat = solve_homography(src, dst)
            except Exception:  # noqa: BLE001
                continue
            proj = HomographyProjection(h_mat)
            if river_y is None:
                score = abs(dy - DEFAULT_HP_BAR_TILE_OFFSET)
            else:
                score = abs(proj.tile_to_pixel(TILES_X / 2.0, 16.0)[1] - river_y)
            if best is None or score < best[0]:
                best = (score, proj)
        if best is None:
            return None, float("inf")
        drift = best[0]
        if river_y is None:
            drift = 0.0
        return best[1], drift

    def _best_tower_fit(
        self,
        bars: List[dict],
        river_y: Optional[float],
        width: int,
        height: int,
    ) -> Optional[Tuple[Projection, Dict[str, Point], float, float]]:
        """Search every plausible pairing of tower bars and keep the most self-consistent fit.

        ``bars`` routinely contains extras (a stray tower, a partially depleted bar, the
        crest above a tower). Rather than trusting the two largest, we enumerate pairings and
        score them on independent evidence: does the projected river land on the real river,
        and are the two tower rows geometrically regular?
        """
        if river_y is None or len(bars) < 4:
            return None

        above = sorted([b for b in bars if b["cy"] < river_y], key=lambda b: b["area"], reverse=True)[:6]
        below = sorted([b for b in bars if b["cy"] > river_y], key=lambda b: b["area"], reverse=True)[:6]
        if len(above) < 2 or len(below) < 2:
            return None

        best: Optional[Tuple[float, Projection, Dict[str, Point], float]] = None
        for pa in itertools.combinations(above, 2):
            for pb in itertools.combinations(below, 2):
                pa = tuple(sorted(pa, key=lambda b: b["cx"]))
                pb = tuple(sorted(pb, key=lambda b: b["cx"]))
                towers = {
                    "enemy_left": (pa[0]["cx"], pa[0]["cy"]),
                    "enemy_right": (pa[1]["cx"], pa[1]["cy"]),
                    "ally_left": (pb[0]["cx"], pb[0]["cy"]),
                    "ally_right": (pb[1]["cx"], pb[1]["cy"]),
                }

                pitch_above = towers["enemy_right"][0] - towers["enemy_left"][0]
                pitch_below = towers["ally_right"][0] - towers["ally_left"][0]
                if pitch_above <= width * 0.30 or pitch_below <= width * 0.30:
                    continue
                # Perspective shrinks the far row: the enemy pitch must not exceed ours.
                if pitch_above / pitch_below > 1.25 or pitch_above / pitch_below < 0.45:
                    continue
                # Bar widths are a second scale cue and must shrink with distance too.
                if abs(pa[0]["w"] - pa[1]["w"]) > width * 0.06:
                    continue
                if abs(pb[0]["w"] - pb[1]["w"]) > width * 0.06:
                    continue

                proj, drift = self._fit_from_towers(towers, river_y)
                if proj is None:
                    continue
                drift_norm = drift / max(1.0, height)
                symmetry = abs(
                    (towers["enemy_left"][0] + towers["enemy_right"][0]) / 2.0
                    - (towers["ally_left"][0] + towers["ally_right"][0]) / 2.0
                ) / max(1.0, width)
                row_level = (
                    abs(towers["enemy_left"][1] - towers["enemy_right"][1]) / max(1.0, height)
                    + abs(towers["ally_left"][1] - towers["ally_right"][1]) / max(1.0, height)
                )
                cost = drift_norm * 6.0 + symmetry * 1.5 + row_level * 2.0
                if best is None or cost < best[0]:
                    best = (cost, proj, towers, drift)

        if best is None:
            return None
        cost, proj, towers, drift = best
        if drift > height * 0.06:
            return None
        return proj, towers, drift, cost

    def _fit_symmetric(
        self,
        ally_pair: Tuple[Point, Point],
        river_y: Optional[float],
        width: int,
        height: int,
        hand: Optional[dict],
    ) -> Optional[Projection]:
        """Build a symmetric perspective model from our two Princess Towers plus the river.

        With one depth we cannot recover full perspective, so the vertical mapping is affine
        (anchored on the river at tile 16 and the tower row) while the horizontal mapping
        keeps a per-row scale. That is markedly better than a plain rectangle because the
        arena's horizontal extent and centre are measured rather than assumed, which is the
        part that most often sends taps into the wrong lane.
        """
        (lx, ly), (rx, ry_) = ally_pair
        if rx - lx < width * 0.20:
            return None
        px_per_tile_x = (rx - lx) / 11.0
        centre_x = (lx + rx) / 2.0
        arena_w = px_per_tile_x * TILES_X
        left = centre_x - arena_w / 2.0
        right = centre_x + arena_w / 2.0

        tower_tile_y = 25.5 - DEFAULT_HP_BAR_TILE_OFFSET
        tower_px_y = (ly + ry_) / 2.0
        if river_y is None:
            # Assume a linear camera: 9.5 tiles between the river and the tower row.
            px_per_tile_y = abs(tower_px_y - height * 0.44) / 9.5
            river_y = tower_px_y - 9.5 * px_per_tile_y
        px_per_tile_y = (tower_px_y - river_y) / (tower_tile_y - 16.0)
        if px_per_tile_y <= 1.0:
            return None
        top = river_y - 16.0 * px_per_tile_y
        bottom = river_y + 16.0 * px_per_tile_y
        if hand is not None:
            bottom = min(bottom, hand["top"] - px_per_tile_y * 0.3)
            top = river_y - (bottom - river_y)
        return RectProjection(left, top, right, bottom)

    def _rect_from_anchors(
        self,
        width: int,
        height: int,
        river_y: Optional[float],
        hand: Optional[dict],
    ) -> Tuple[float, float, float, float]:
        """Last-resort axis-aligned arena: full width, river at tile 16, bottom at the tray."""
        left = width * 0.02
        right = width * 0.98
        if river_y is not None:
            if hand is not None:
                bottom = hand["top"] - height * 0.015
            else:
                bottom = river_y + (river_y - height * 0.09)
            bottom = max(river_y + height * 0.10, min(height * 0.95, bottom))
            top = 2.0 * river_y - bottom
        else:
            top = height * 0.09
            bottom = height * 0.78
        if top < height * 0.02:
            top = height * 0.02
        return (left, top, right, bottom)

    def _score(
        self, tower_count: int, river_y: Optional[float], river_drift: float, fit_cost: float
    ) -> float:
        """Confidence from independent evidence, not from the degenerate 4-point residual."""
        score = 0.42
        if tower_count >= 4:
            score += 0.20
        if river_y is not None:
            score += 0.12
        if fit_cost < 0.05:
            score += 0.20
        elif fit_cost < 0.15:
            score += 0.12
        elif fit_cost < 0.35:
            score += 0.05
        if river_drift <= 8.0:
            score += 0.08
        return min(1.0, score)

    @staticmethod
    def _offset_projection(proj: Projection, ox: float, oy: float) -> Projection:
        if ox == 0 and oy == 0:
            return proj
        if isinstance(proj, RectProjection):
            l, t, r, b = proj.rect
            return RectProjection(l + ox, t + oy, r + ox, b + oy)
        if isinstance(proj, HomographyProjection):
            shift = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
            return HomographyProjection(shift @ proj.h)
        return proj

    @staticmethod
    def _sync_ratios(cal: ArenaCalibration, width: int, height: int) -> None:
        """Keep the legacy ratio fields consistent for any code still reading them."""
        try:
            grid = ArenaGrid(width, height, cal)
            x1, y1, x2, y2 = grid.arena_pixel_bounds
            cal.arena_left_ratio = x1 / max(1.0, width)
            cal.arena_top_ratio = y1 / max(1.0, height)
            cal.arena_right_ratio = x2 / max(1.0, width)
            cal.arena_bottom_ratio = y2 / max(1.0, height)
            if cal.hand_slots_px:
                cal.hand_slot_x_ratios = tuple(p[0] / max(1.0, width) for p in cal.hand_slots_px)  # type: ignore[assignment]
                cal.hand_slot_y_ratio = cal.hand_slots_px[0][1] / max(1.0, height)
            if cal.elixir_px:
                cal.elixir_x_start_ratio = cal.elixir_px[0] / max(1.0, width)
                cal.elixir_x_end_ratio = cal.elixir_px[1] / max(1.0, width)
                cal.elixir_y_ratio = cal.elixir_px[2] / max(1.0, height)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not sync legacy ratios: %s", exc)

    # ==================================================================================
    # Helpers
    # ==================================================================================
    @staticmethod
    def _mask_runs(mask: np.ndarray, min_len: int = 4) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        start = None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= min_len:
                    runs.append((start, i - 1))
                start = None
        if start is not None and len(mask) - start >= min_len:
            runs.append((start, len(mask) - 1))
        return runs

    @staticmethod
    def _largest_run(values: Sequence[int]) -> List[int]:
        if not values:
            return []
        best: List[int] = []
        current = [values[0]]
        for v in values[1:]:
            if v - current[-1] <= 3:
                current.append(v)
            else:
                if len(current) > len(best):
                    best = current
                current = [v]
        return current if len(current) > len(best) else best


# ======================================================================================
# Preview rendering
# ======================================================================================


def render_calibration_preview(
    frame_bgr: np.ndarray,
    calibration: ArenaCalibration,
    result: Optional[CalibrationResult] = None,
) -> np.ndarray:
    """Draw the measured grid, anchors, hand slots and elixir bar over a real frame."""
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    grid = ArenaGrid(w, h, calibration)

    for i in range(int(TILES_X) + 1):
        cv2.line(out, grid.tile_to_pixel(i, 0), grid.tile_to_pixel(i, TILES_Y), (0, 230, 255), 1, cv2.LINE_AA)
    for j in range(int(TILES_Y) + 1):
        cv2.line(out, grid.tile_to_pixel(0, j), grid.tile_to_pixel(TILES_X, j), (0, 230, 255), 1, cv2.LINE_AA)

    ry = int(round(grid.river_pixel_y))
    cv2.line(out, (0, ry), (w, ry), (255, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(out, "river / deploy boundary", (8, max(14, ry - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1, cv2.LINE_AA)

    for name, (tx, ty) in ArenaGrid.TOWER_TILES.items():
        px, py = grid.tile_to_pixel(tx, ty)
        cv2.drawMarker(out, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(out, name, (px + 8, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)

    for label, tile in (
        ("L-bridge", grid.LEFT_BRIDGE),
        ("R-bridge", grid.RIGHT_BRIDGE),
        ("kite-L", grid.KITE_LEFT_THREAT_TO_CENTER),
        ("kite-R", grid.KITE_RIGHT_THREAT_TO_CENTER),
        ("back", grid.SPLIT_BACK),
    ):
        px, py = grid.tile_to_pixel(*tile)
        cv2.circle(out, (px, py), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(out, label, (px + 7, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

    for i in range(4):
        x1, y1, x2, y2 = grid.hand_slot_rect(i)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(out, f"slot {i}", (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    ex1, ex2, ey = grid.elixir_geometry()
    cv2.rectangle(out, (ex1, ey - 6), (ex2, ey + 6), (255, 0, 255), 2)

    if result is not None:
        color = (0, 220, 0) if result.ok else (0, 165, 255)
        cv2.rectangle(out, (0, 0), (w - 1, 26), (0, 0, 0), -1)
        cv2.putText(
            out,
            f"calibration {result.confidence:.2f} [{result.stage}]",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


# ======================================================================================
# Persistence
# ======================================================================================


def device_slug(serial: Optional[str], width: int, height: int) -> str:
    raw = f"{serial or 'device'}_{width}x{height}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def calibration_path(directory: str | Path, serial: Optional[str], width: int, height: int) -> Path:
    return Path(directory) / f"{device_slug(serial, width, height)}.json"


def save_calibration(path: str | Path, calibration: ArenaCalibration) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(calibration.to_dict(), fh, indent=2)
    return path


def load_calibration(path: str | Path) -> Optional[ArenaCalibration]:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return ArenaCalibration.from_dict(json.load(fh))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read calibration %s: %s", path, exc)
        return None
