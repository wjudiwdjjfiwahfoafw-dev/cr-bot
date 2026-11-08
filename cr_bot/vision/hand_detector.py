"""4-slot Hand Card recognizer using multi-cue template + HSV matching and unique deck assignment."""

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


class HandCardDetector:
    """Identifies the 4 playable cards in hand, affordability, and active Evolution glow."""

    SLOT_CROP_W_RATIO = 96.0 / 720.0
    SLOT_CROP_H_RATIO = 116.0 / 1280.0

    def __init__(
        self,
        grid: ArenaGrid,
        deck_cards: List[str],
        templates_dir: str | Path = "assets/card_templates",
    ) -> None:
        self.grid = grid
        self.deck_cards = [normalize_card_name(c) for c in deck_cards]
        self.templates_dir = Path(templates_dir)
        self._templates: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.reload_templates()

    def reload_templates(self) -> int:
        """Load `<card_name>*.png` templates from disk and precompute HSV histograms."""
        self._templates.clear()
        if not self.templates_dir.exists():
            return 0

        loaded = 0
        for img_path in sorted(self.templates_dir.glob("*.png")):
            stem = img_path.stem.lower()
            # Strip trailing numeric/evo variant suffixes like `mega_knight_evo` or `bats_1`
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

            norm_img, hist = self._prepare_descriptor(img)
            self._templates.setdefault(matched_card, []).append((norm_img, hist))
            loaded += 1

        if loaded:
            logger.info("Loaded %d card template(s) across %d unique cards.", loaded, len(self._templates))
        return loaded

    def crop_slot(self, frame_bgr: np.ndarray, slot_index: int) -> np.ndarray:
        """Extract the portrait crop of hand slot 0..3."""
        h, w = frame_bgr.shape[:2]
        cx, cy = self.grid.hand_slot_pixel(slot_index)
        half_w = int(round(0.5 * self.SLOT_CROP_W_RATIO * w))
        half_h = int(round(0.5 * self.SLOT_CROP_H_RATIO * h))
        x1, x2 = max(0, cx - half_w), min(w, cx + half_w)
        y1, y2 = max(0, cy - half_h), min(h, cy + half_h)
        return frame_bgr[y1:y2, x1:x2]

    def detect_hand(self, frame_bgr: np.ndarray, current_elixir: float) -> List[HandSlot]:
        """Classify all 4 hand slots with 1-to-1 uniqueness across the 8-card deck."""
        slot_crops = [self.crop_slot(frame_bgr, i) for i in range(4)]

        if not self._templates:
            # Graceful fallback when user hasn't run `scripts/capture_templates.py` yet:
            # still detects Evolution diamonds & saturation readiness so calibration/HUD works!
            fallback_hand: List[HandSlot] = []
            for idx in range(4):
                tap_xy = self.grid.hand_slot_pixel(idx)
                crop = slot_crops[idx]
                fallback_hand.append(
                    HandSlot(
                        slot_index=idx,
                        card_name="unknown",
                        elixir_cost=3,
                        is_ready=self._is_slot_saturated(crop),
                        is_evolved=self._has_evo_glow(frame_bgr, idx),
                        confidence=0.0,
                        tap_xy=tap_xy,
                    )
                )
            return fallback_hand

        # Score every (slot_idx, card_name) pair
        candidate_pairs: List[Tuple[float, int, str]] = []
        for idx, crop in enumerate(slot_crops):
            if crop.size == 0:
                continue
            slot_img, slot_hist = self._prepare_descriptor(crop)
            for card_name, variants in self._templates.items():
                best_score = -1.0
                for tmpl_img, tmpl_hist in variants:
                    ncc = float(cv2.matchTemplate(slot_img, tmpl_img, cv2.TM_CCOEFF_NORMED)[0, 0])
                    hist_corr = float(cv2.compareHist(slot_hist, tmpl_hist, cv2.HISTCMP_CORREL))
                    combined = 0.70 * ncc + 0.30 * max(0.0, hist_corr)
                    if combined > best_score:
                        best_score = combined
                candidate_pairs.append((best_score, idx, card_name))

        # Greedy maximum-weight bipartite assignment ensures no duplicate card in hand
        candidate_pairs.sort(key=lambda item: item[0], reverse=True)
        assigned_slots: Dict[int, Tuple[str, float]] = {}
        used_cards = set()

        for score, slot_idx, card_name in candidate_pairs:
            if slot_idx in assigned_slots or card_name in used_cards:
                continue
            if score >= 0.32:
                assigned_slots[slot_idx] = (card_name, round(score, 3))
                used_cards.add(card_name)
            if len(assigned_slots) == 4:
                break

        results: List[HandSlot] = []
        for idx in range(4):
            card_name, conf = assigned_slots.get(idx, ("unknown", 0.0))
            info = get_card_info(card_name)
            saturated = self._is_slot_saturated(slot_crops[idx])
            affordable = current_elixir + 0.15 >= float(info.elixir) if card_name != "unknown" else saturated
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

    @staticmethod
    def _prepare_descriptor(crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Center-crop slightly to exclude bottom level text & top elixir drop icon variance
        h, w = crop_bgr.shape[:2]
        inner = crop_bgr[int(0.14 * h) : int(0.82 * h), int(0.10 * w) : int(0.90 * w)]
        if inner.size == 0:
            inner = crop_bgr
        resized = cv2.resize(inner, (64, 72), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return resized, hist

    @staticmethod
    def _is_slot_saturated(crop_bgr: np.ndarray) -> bool:
        """Cards that cannot be afforded yet are greyed out (low HSV Saturation)."""
        if crop_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        return mean_sat >= 42.0

    def _has_evo_glow(self, frame_bgr: np.ndarray, slot_index: int) -> bool:
        """Detect the bright purple/magenta Evolution diamond at the top edge of a hand slot."""
        h, w = frame_bgr.shape[:2]
        cx, cy = self.grid.hand_slot_pixel(slot_index)
        half_h = int(round(0.5 * self.SLOT_CROP_H_RATIO * h))
        top_y = cy - half_h
        roi = frame_bgr[max(0, top_y - 14) : min(h, top_y + 12), max(0, cx - 20) : min(w, cx + 20)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        evo_mask = cv2.inRange(hsv, np.array([135, 135, 165]), np.array([168, 255, 255]))
        return (float(np.count_nonzero(evo_mask)) / float(evo_mask.size)) >= 0.16
