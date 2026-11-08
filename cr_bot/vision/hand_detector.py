"""
4-slot hand card recognizer.

Real-device notes
-----------------
Two things made the original recognizer unreliable outside an emulator:

1. It cropped each slot with hard-coded fractions of a 720x1280 frame. On a real phone the
   tray sits elsewhere and is a different size, so crops were misaligned and every match was
   noise. Crops now come from the calibrated grid (see ``cr_bot.arena.device_calibrator``).
2. A selected card moves up and matching just the stationary crop misses it. Search a
   small range of vertical offsets, compare only the configured deck, and abstain on weak
   or ambiguous matches. Four cards in hand are drawn from *eight*: one unknown slot cannot
   be identified merely because the other three have been matched.

The bundled variants are fixed images, not a trained classifier. Device-specific captures
are often needed when the game artwork or screen layout changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from cr_bot.arena.grid import ArenaGrid
from cr_bot.config.cards_db import CARDS_DB, get_card_info, normalize_card_name
from cr_bot.models import HandSlot

logger = logging.getLogger(__name__)

CardDescriptor = Tuple[np.ndarray, np.ndarray]


class HandCardDetector:
    """Identifies the 4 playable cards in hand, affordability, and active Evolution glow."""

    MATCH_ACCEPT = 0.40
    MATCH_MARGIN = 0.08
    GRAY_ACCEPT = 0.74
    GRAY_MARGIN = 0.10
    # The selected card lifts above the stationary tray. These are fractions of card height.
    VERTICAL_OFFSETS = (0.0, -0.06, -0.12, -0.18, -0.24, -0.30, 0.06)
    EVO_GLOW_ROI_SCALE = 0.12

    def __init__(
        self,
        grid: ArenaGrid,
        deck_cards: List[str],
        templates_dir: str | Path = "assets/card_templates",
        evolutions: Optional[List[str]] = None,
    ) -> None:
        self.grid = grid
        self.deck_cards = [normalize_card_name(c) for c in deck_cards]
        # None preserves standalone detector behaviour; a configured list restricts the
        # purple-glow heuristic to the actual evolution slots, not ordinary Bats/Zap.
        self.evolutions = None if evolutions is None else {normalize_card_name(c) for c in evolutions}
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
        deck = set(self.deck_cards)
        # Match the longest known slug *before* testing deck membership. Otherwise a
        # giant_skeleton.png template could be loaded as giant when Giant is in the deck
        # and Giant Skeleton is not, silently teaching the wrong identity.
        names = sorted(set(CARDS_DB) | deck, key=lambda name: (-len(name), name))
        for img_path in sorted(self.templates_dir.glob("*.png")):
            stem = img_path.stem.lower()
            matched_card = next((name for name in names
                                 if stem == name or stem.startswith(name + "_")), None)
            if matched_card not in deck:
                continue  # Never fall back to templates for cards outside the configured deck.

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
        """Identify only well-supported, distinct deck cards; never guess an unknown slot.

        Search a small vertical range to cover raised cards. A per-slot margin prevents a
        look-alike from being selected merely because a stronger slot used the best match.
        When two slots claim the same card, keep the stronger claim and leave the other
        unknown. Unknown/empty slots are never actionable, even if their blue background is
        highly saturated. Disabled (grey) cards are scored by structure, with a higher gate.
        """
        crops = [self.crop_slot(frame_bgr, i) for i in range(4)]
        pool = list(dict.fromkeys(c for c in self.deck_cards if c in self._templates))
        claims: List[Tuple[float, int, str]] = []
        h, w = frame_bgr.shape[:2]
        for idx, crop in enumerate(crops):
            if crop.size == 0 or not pool:
                continue
            x1, y1, x2, y2 = self.grid.hand_slot_rect(idx)
            card_h = y2 - y1
            grey = not self._is_slot_saturated(crop)
            scores = {card: 0.0 for card in pool}
            for fraction in self.VERTICAL_OFFSETS:
                dy = int(round(fraction * card_h))
                if y1 + dy < 0 or y2 + dy > h or x1 < 0 or x2 > w:
                    continue
                shifted = frame_bgr[y1 + dy:y2 + dy, x1:x2]
                if shifted.size == 0:
                    continue
                desc = self._prepare_descriptor(shifted)
                for card in pool:
                    if grey:
                        # An HSV histogram of a grey card carries essentially no identity
                        # information. Comparing it to another grey histogram can produce a
                        # spurious perfect match, so use the portrait structure alone.
                        score = max(
                            float(cv2.matchTemplate(desc[0], ref[0], cv2.TM_CCOEFF_NORMED)[0, 0])
                            for ref in self._templates[card]
                        )
                    else:
                        score = self._best_score(desc, card)
                    scores[card] = max(scores[card], score)
            ranked = sorted(((score, card) for card, score in scores.items()), reverse=True)
            if not ranked:
                continue
            score, card = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            threshold = self.GRAY_ACCEPT if grey else self.MATCH_ACCEPT
            margin = self.GRAY_MARGIN if grey else self.MATCH_MARGIN
            if score >= threshold and score - second >= margin:
                claims.append((score, idx, card))

        assigned: Dict[int, Tuple[str, float]] = {}
        used = set()
        for score, idx, card in sorted(claims, reverse=True):
            if card not in used:
                assigned[idx] = (card, round(score, 3))
                used.add(card)

        results: List[HandSlot] = []
        for idx, crop in enumerate(crops):
            card_name, conf = assigned.get(idx, ("unknown", 0.0))
            info = get_card_info(card_name)
            saturated = self._is_slot_saturated(crop)
            affordable = current_elixir + 0.15 >= float(info.elixir)
            results.append(
                HandSlot(
                    slot_index=idx,
                    card_name=card_name,
                    elixir_cost=info.elixir,
                    is_ready=bool(card_name != "unknown" and saturated and affordable),
                    is_evolved=bool(card_name != "unknown" and
                                    (self.evolutions is None or card_name in self.evolutions) and
                                    self._has_evo_glow(frame_bgr, idx)),
                    confidence=conf,
                    tap_xy=self.grid.hand_slot_pixel(idx),
                )
            )
        return results

    # -- pixel helpers ---------------------------------------------------------------
    @staticmethod
    def _is_slot_saturated(crop_bgr: np.ndarray) -> bool:
        """Test the art, not the blue tray around it, for the grey unaffordable state."""
        if crop_bgr.size == 0:
            return False
        h, w = crop_bgr.shape[:2]
        inner = crop_bgr[int(0.12 * h):int(0.76 * h), int(0.10 * w):int(0.90 * w)]
        if inner.size == 0:
            return False
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
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
