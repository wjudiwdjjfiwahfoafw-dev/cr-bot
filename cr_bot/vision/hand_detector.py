"""
4-slot hand card recognizer.

Real-device notes
-----------------
Two things made the original recognizer unreliable outside an emulator:

1. It cropped each slot with hard-coded fractions of a 720x1280 frame. On a real phone the
   tray sits elsewhere and is a different size, so crops were misaligned and every match was
   noise. Crops now come from the calibrated grid (see ``cr_bot.arena.device_calibrator``).
2. It matched a card template against a *same-size* crop with ``TM_CCOEFF_NORMED``, which for
   equal-size images degenerates into a weak statistic. Comparison now uses a proper
   normalised cross-correlation plus an HSV histogram correlation, and adds a **deck
   elimination pass**: because the four hand cards are always four *distinct* cards drawn
   from the eight-card deck, a single unrecognised slot can be filled in by elimination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.config.cards_db import get_card_info, normalize_card_name
from cr_bot.models import HandSlot

logger = logging.getLogger(__name__)

CardDescriptor = Tuple[np.ndarray, np.ndarray]


class HandCardDetector:
    """Identifies the 4 playable cards in hand, affordability, and active Evolution glow."""

    MATCH_ACCEPT = 0.34
    #: a slot may still be resolved below MATCH_ACCEPT when it is the only one left, because the
    #: hand is guaranteed to be four distinct cards drawn from the eight-card deck
    ELIMINATION_FLOOR = 0.16
    EVO_GLOW_ROI_SCALE = 0.12

    def __init__(
        self,
        grid: ArenaGrid,
        deck_cards: List[str],
        templates_dir: str | Path = "assets/card_templates",
    ) -> None:
        self.grid = grid
        self.deck_cards = [normalize_card_name(c) for c in deck_cards]
        self.templates_dir = Path(templates_dir)
        self._templates: Dict[str, List[CardDescriptor]] = {}
        self.reload_templates()

    # -- templates -----------------------------------------------------------------
    def reload_templates(self) -> int:
        """Load `<card_name>*.png` templates from disk and precompute descriptors."""
        self._templates.clear()
        if not self.templates_dir.exists():
            return 0

        loaded = 0
        for img_path in sorted(self.templates_dir.glob("*.png")):
            stem = img_path.stem.lower()
            matched_card = None
            for deck_card in self.deck_cards:
                if stem == deck_card or stem.startswith(deck_card + "_"):
                    matched_card = deck_card
                    break
            if matched_card is None:
                matched_card = normalize_card_name(stem)

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            self._templates.setdefault(matched_card, []).append(self._prepare_descriptor(img))
            loaded += 1

        if loaded:
            logger.info("Loaded %d card template(s) across %d unique cards.", loaded, len(self._templates))
        return loaded

    @property
    def has_templates(self) -> bool:
        return bool(self._templates)

    # -- cropping ------------------------------------------------------------------
    def crop_slot(self, frame_bgr: np.ndarray, slot_index: int) -> np.ndarray:
        """Extract the portrait crop of hand slot 0..3 using calibrated geometry."""
        h, w = frame_bgr.shape[:2]
        self.grid.resize(w, h)
        x1, y1, x2, y2 = self.grid.hand_slot_rect(slot_index)
        return frame_bgr[y1:y2, x1:x2]

    @staticmethod
    def _prepare_descriptor(crop_bgr: np.ndarray) -> CardDescriptor:
        """Normalised, illumination-tolerant descriptor of a card portrait."""
        if crop_bgr.size == 0:
            raise ValueError("empty crop")
        h, w = crop_bgr.shape[:2]
        # Trim the elixir badge at the bottom and the level/element strip at the top, which
        # vary with level and Evolution state but say nothing about card identity.
        inner = crop_bgr[int(0.06 * h) : int(0.86 * h), int(0.06 * w) : int(0.94 * w)]
        if inner.size == 0:
            inner = crop_bgr
        resized = cv2.resize(inner, (72, 80), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray).astype(np.float32)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return gray, hist

    @staticmethod
    def _similarity(a: CardDescriptor, b: CardDescriptor) -> float:
        ga, ha = a
        gb, hb = b
        if ga.shape != gb.shape:
            gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
        va = ga.ravel() - float(ga.mean())
        vb = gb.ravel() - float(gb.mean())
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        ncc = float(np.dot(va, vb) / denom) if denom > 1e-6 else 0.0
        hist_corr = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
        return 0.68 * max(0.0, ncc) + 0.32 * max(0.0, hist_corr)

    def _best_score(self, desc: CardDescriptor, card: str) -> float:
        variants = self._templates.get(card)
        if not variants:
            return 0.0
        return float(max(self._similarity(desc, v) for v in variants))

    # -- detection -----------------------------------------------------------------
    def detect_hand(self, frame_bgr: np.ndarray, current_elixir: float) -> List[HandSlot]:
        """Classify all 4 hand slots with 1-to-1 uniqueness across the 8-card deck."""
        crops = [self.crop_slot(frame_bgr, i) for i in range(4)]

        if not self._templates:
            return self._fallback_hand(frame_bgr, crops, current_elixir)

        assigned: Dict[int, Tuple[str, float]] = {}
        if all(c.size > 0 for c in crops):
            pairs: List[Tuple[float, int, str]] = []
            descriptors = [self._prepare_descriptor(c) for c in crops]
            # The tray can only ever show cards from the deck: restricting the candidate pool to
            # the deck removes most confusions with look-alike cards that can never appear here.
            deck_pool = [c for c in self.deck_cards if c in self._templates]
            pool = deck_pool if len(deck_pool) >= 4 else list(self._templates)
            for idx, desc in enumerate(descriptors):
                for card_name in pool:
                    variants = self._templates[card_name]
                    best = max(self._similarity(desc, v) for v in variants)
                    pairs.append((best, idx, card_name))

            pairs.sort(key=lambda p: p[0], reverse=True)
            used_cards = set()
            for score, slot_idx, card_name in pairs:
                if slot_idx in assigned or card_name in used_cards:
                    continue
                if score < self.MATCH_ACCEPT:
                    continue
                assigned[slot_idx] = (card_name, round(score, 3))
                used_cards.add(card_name)
                if len(assigned) == 4:
                    break

            # Deck elimination: the hand is always 4 distinct cards from the 8-card deck, so the
            # last unresolved slot can be settled from what is left even with a weak raw score.
            missing_slots = [i for i in range(4) if i not in assigned]
            remaining = [c for c in self.deck_cards if c not in used_cards]
            if len(missing_slots) == 1 and remaining:
                slot_idx = missing_slots[0]
                best_left = max(
                    ((self._best_score(descriptors[slot_idx], c), c) for c in remaining
                     if c in self._templates),
                    default=(0.0, ""),
                )
                if best_left[1] and best_left[0] >= self.ELIMINATION_FLOOR:
                    assigned[slot_idx] = (best_left[1], round(max(0.45, best_left[0]), 3))
                    logger.debug("Filled hand slot %d by deck elimination -> %s (%.3f)",
                                 slot_idx, best_left[1], best_left[0])

        results: List[HandSlot] = []
        for idx in range(4):
            card_name, conf = assigned.get(idx, ("unknown", 0.0))
            info = get_card_info(card_name)
            saturated = self._is_slot_saturated(crops[idx])
            affordable = (current_elixir + 0.15 >= float(info.elixir)) if card_name != "unknown" else saturated
            results.append(
                HandSlot(
                    slot_index=idx,
                    card_name=card_name,
                    elixir_cost=info.elixir,
                    is_ready=bool(saturated and affordable),
                    is_evolved=self._has_evo_glow(frame_bgr, idx),
                    confidence=conf,
                    tap_xy=self.grid.hand_slot_pixel(idx),
                )
            )
        return results

    def _fallback_hand(
        self, frame_bgr: np.ndarray, crops: List[np.ndarray], current_elixir: float
    ) -> List[HandSlot]:
        """Without templates, still expose slot geometry + readiness so calibration works."""
        out: List[HandSlot] = []
        for idx in range(4):
            out.append(
                HandSlot(
                    slot_index=idx,
                    card_name="unknown",
                    elixir_cost=3,
                    is_ready=self._is_slot_saturated(crops[idx]),
                    is_evolved=self._has_evo_glow(frame_bgr, idx),
                    confidence=0.0,
                    tap_xy=self.grid.hand_slot_pixel(idx),
                )
            )
        return out

    # -- pixel helpers ---------------------------------------------------------------
    @staticmethod
    def _is_slot_saturated(crop_bgr: np.ndarray) -> bool:
        """Cards you cannot afford are desaturated/greyed out."""
        if crop_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1])) >= 46.0

    def _has_evo_glow(self, frame_bgr: np.ndarray, slot_index: int) -> bool:
        """Detect the bright purple Evolution diamond at the top edge of a hand slot."""
        h, w = frame_bgr.shape[:2]
        self.grid.resize(w, h)
        x1, y1, x2, y2 = self.grid.hand_slot_rect(slot_index)
        band = int(max(4, (y2 - y1) * self.EVO_GLOW_ROI_SCALE))
        cx = (x1 + x2) // 2
        half = max(6, (x2 - x1) // 3)
        roi = frame_bgr[max(0, y1 - band) : min(h, y1 + band), max(0, cx - half) : min(w, cx + half)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        evo_mask = cv2.inRange(hsv, np.array([135, 135, 165]), np.array([168, 255, 255]))
        return (float(np.count_nonzero(evo_mask)) / float(evo_mask.size)) >= 0.14
