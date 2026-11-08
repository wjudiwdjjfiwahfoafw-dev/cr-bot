"""Give a detected troop its card name by matching it against the real in-game sprite library.

Integration
-----------
``ArenaUnitBackend`` calls this recognizer for eligible measured segmentation boxes. It is lazy,
optional, limited to one attempt per frame and may abstain; the tracker carries accepted names
through subsequent generic frames. The sprite files are not a reliable stand-in for every live
pose and device tint (see ``tools/recognition_benchmark.py``).

Two challenges remain important:

**Cost.** ``SpriteMatcher.identify_window`` fits all 53 cards × 303 frames × 3 scales × 27 offsets
into the box: ~300 ms *per troop*. Useless in a 60 fps loop.

**Robustness.** Fitting every template to the box makes the score depend on how tightly the box
happens to hug the sprite; on a box that is a little too loose, a small high-contrast card wins
(the first run of this module named a Giant "bats"). And a card can only be found at the scale
its box implies, so a template of a differently-sized unit is never considered.

So the search runs in two stages:

1. **coarse** — one cheap masked correlation per card (median frame, template at its *natural*
   size relative to the box height, a handful of offsets). ~1 ms per card, and it is the stage
   that eliminates the 40-odd cards that cannot be what is in the box;
2. **fine** — the surviving handful are re-scored over all their frames and scales with the
   existing ``identify_window`` machinery.

The answer is only returned when it clears an absolute score floor, beats an independent
runner-up, and the winning template explains enough of the detected box. A wrong name is worse
than no name: on real labelled crops a narrow Witch template used to match part of a Tesla and a
Mini P.E.K.K.A with high confidence. The shipped sprite set also contains byte-identical art for
some *different cards*. Such a match is ambiguous, not proof of either card.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cr_bot.vision.sprite_matcher import (
    SpriteLibrary,
    SpriteMatcher,
    masked_zncc_at,
    prepare_gray,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPRITES_DIR = REPO_ROOT / "assets" / "unit_sprites"


@dataclass
class Recognition:
    """What the matcher concluded about one detected troop."""

    card: str
    score: float
    #: best score of another *card* (shared winning art is rejected as ambiguous)
    runner_up: float
    margin: float
    bbox: Tuple[int, int, int, int]

    def describe(self) -> str:
        return f"{self.card} ({self.score:.2f}, +{self.margin:.2f})"


class UnitRecognizer:
    """Names troops by matching real sprite frames against a detection's bbox.

    Parameters
    ----------
    enabled:
        Off means "never name anything", and costs nothing.
    min_score / min_margin:
        The two gates that keep a wrong name out.
    coarse_keep:
        How many cards survive the cheap first stage.
    """

    # Calibrated conservatively against the real labelled crops in tests/fixtures, not against
    # pastes of these very same templates. This is an open-set matcher, not a probability.
    DEFAULT_MIN_SCORE = 0.75
    DEFAULT_MIN_MARGIN = 0.07
    DEFAULT_COARSE_KEEP = 6
    MIN_BOX_COVERAGE = 0.55
    #: fractions of the box height that are treated as a health bar and cropped away
    BAR_CROP_FRACTION = 0.22

    def __init__(
        self,
        sprites_dir: Path | str = DEFAULT_SPRITES_DIR,
        enabled: bool = True,
        min_score: float = DEFAULT_MIN_SCORE,
        min_margin: float = DEFAULT_MIN_MARGIN,
        coarse_keep: int = DEFAULT_COARSE_KEEP,
        max_cards: Optional[int] = None,
        matcher: Optional[SpriteMatcher] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.coarse_keep = max(1, int(coarse_keep))
        self._matcher = matcher
        self._sprites_dir = Path(sprites_dir)
        self._max_cards = max_cards
        self._prepared = False
        #: card -> (gray, mask) of its median frame, prepared once (see _representatives)
        self._reps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._scaled: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        #: cards whose art is byte-identical to another card's (skeletons vs skeleton_army, ...)
        self.duplicate_groups: List[List[str]] = []
        #: (card, frame index) -> other cards with precisely the same frame
        self._frame_aliases: Dict[Tuple[str, int], set[str]] = {}
        self.named = 0
        self.attempted = 0

    # -- lazy loading ---------------------------------------------------------------
    @property
    def matcher(self) -> Optional[SpriteMatcher]:
        """Built on first use, so a bot that never needs names pays nothing at startup."""
        if not self.enabled:
            return None
        if self._matcher is None:
            try:
                library = SpriteLibrary(self._sprites_dir)
            except Exception as exc:      # noqa: BLE001 - a broken library must not stop the bot
                logger.warning("sprite library unavailable (%s); troops stay unnamed", exc)
                self.enabled = False
                return None
            if not library.is_ready:
                logger.warning(
                    "no usable sprite frames in %s; troops stay unnamed "
                    "(run scripts/fetch_game_sprites.py)", self._sprites_dir)
                self.enabled = False
                return None
            self._matcher = SpriteMatcher(library=library)
        if not self._prepared:
            self._prepare()
            self._prepared = True
        return self._matcher

    @property
    def is_ready(self) -> bool:
        return self.matcher is not None

    def cards(self) -> List[str]:
        matcher = self.matcher
        if matcher is None:
            return []
        cards = list(matcher.library.cards)
        if self._max_cards is not None:
            cards = cards[: self._max_cards]
        return cards

    # -- preparation ----------------------------------------------------------------
    def _prepare(self) -> None:
        """Build one representative template per card, and find duplicate artwork."""
        assert self._matcher is not None
        signatures: Dict[bytes, List[Tuple[str, int]]] = {}
        for card in self._matcher.library.cards:
            frames = self._matcher.library.frames_for(card)
            if not frames:
                continue
            # median frame by aspect: the most "typical" pose of that card
            rep = sorted(frames, key=lambda f: abs(f.aspect - float(np.median(
                [x.aspect for x in frames]))))[0]
            self._reps[card] = (rep.gray, rep.mask)
            for frame in frames:
                # Shape and mask area alone are not an identity check. Compare the actual
                # prepared pixels and mask, including shape, for every frame, not only the rep.
                digest = hashlib.sha256()
                digest.update(str(frame.gray.shape).encode("ascii"))
                digest.update(frame.gray.tobytes())
                digest.update(frame.mask.tobytes())
                signatures.setdefault(digest.digest(), []).append((card, frame.index))
        groups: set[Tuple[str, ...]] = set()
        for instances in signatures.values():
            cards = {card for card, _ in instances}
            if len(cards) < 2:
                continue
            groups.add(tuple(sorted(cards)))
            for card, index in instances:
                self._frame_aliases[(card, index)] = cards - {card}
        self.duplicate_groups = [list(group) for group in sorted(groups)]
        if self.duplicate_groups:
            logger.debug("cards sharing identical art (left ambiguous): %s",
                         ", ".join("/".join(g) for g in self.duplicate_groups))

    def _scaled_rep(self, card: str, height: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (card, int(height))
        cached = self._scaled.get(key)
        if cached is not None:
            return cached
        gray, mask = self._reps[card]
        h, w = gray.shape[:2]
        width = max(4, int(round(w * (height / float(h)))))
        g = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        m = (cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)
        if m.sum() < 24:
            m = np.ones_like(m)
        self._scaled[key] = (g, m)
        return g, m

    # -- the entry point ------------------------------------------------------------
    def identify(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        has_bar: bool = True,
        topk: int = 3,
    ) -> Optional[Recognition]:
        """Best card for this bbox, or ``None`` when nothing is convincing enough."""
        matcher = self.matcher
        if matcher is None or frame_bgr is None:
            return None
        box = self._crop_bar(bbox, frame_bgr.shape, has_bar)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        bh, bw = y2 - y1, x2 - x1
        # too small to say anything about: the matcher would be fitting noise
        if bw < 16 or bh < 16:
            return None

        self.attempted += 1
        image = prepare_gray(frame_bgr[y1:y2, x1:x2])

        # --- stage 1: cheap rejection over every card -------------------------------
        coarse: List[Tuple[float, str]] = []
        for card in self.cards():
            score = self._coarse_score(card, image, bh, bw)
            if score > -1.0:
                coarse.append((score, card))
        if not coarse:
            return None
        coarse.sort(reverse=True)
        shortlist = [card for _score, card in coarse[: self.coarse_keep]]

        # --- stage 2: proper ranking of the survivors -------------------------------
        # Display top-k must not truncate the competitor set used for the safety margin.
        # Never promote a coarse score when fine matching could not evaluate a candidate.
        hits = matcher.identify_window(frame_bgr, box, candidates=shortlist, topk=len(shortlist))
        hits = sorted((hit for hit in hits if np.isfinite(hit.score)),
                      key=lambda hit: hit.score, reverse=True)
        if not hits:
            return None
        best = hits[0]
        # An identical frame can represent different *cards* (e.g. Skeletons and Skeleton
        # Army). A single sprite cannot distinguish them; don't make an arbitrary card choice.
        if self._frame_aliases.get((best.card, best.template)):
            return None
        bx1, by1, bx2, by2 = best.bbox
        coverage = max(0, bx2 - bx1) * max(0, by2 - by1) / float(bw * bh)
        if coverage < self.MIN_BOX_COVERAGE:
            return None
        if len(hits) < 2:
            return None  # no independent runner-up: a margin is not measurable
        runner_up = float(hits[1].score)
        margin = float(best.score - runner_up)

        if best.score < self.min_score or margin < self.min_margin:
            logger.debug(
                "not naming %s: score %.2f (floor %.2f) margin %.3f (floor %.3f)",
                best.card, best.score, self.min_score, margin, self.min_margin,
            )
            return None
        self.named += 1
        return Recognition(card=best.card, score=float(best.score),
                           runner_up=float(runner_up), margin=margin, bbox=box)

    def identify_many(
        self,
        frame_bgr: np.ndarray,
        units: Sequence[Tuple[Tuple[int, int, int, int], bool]],
    ) -> List[Optional[Recognition]]:
        """Name several boxes at once (pass ``(bbox, has_bar)`` pairs)."""
        return [self.identify(frame_bgr, bbox, has_bar=bar) for bbox, bar in units]

    # -- stage 1 --------------------------------------------------------------------
    #: offsets tried by the coarse stage, as a fraction of the box size
    COARSE_OFFSETS = (0.0, -0.10, 0.10)

    def _coarse_score(self, card: str, image: np.ndarray, bh: int, bw: int) -> float:
        """One masked correlation of a card's representative frame at one size."""
        gray, mask = self._reps.get(card, (None, None))
        if gray is None:
            return -1.0
        # the sprite fills most of the box after the bar is cropped; a template much larger
        # than the box cannot be in it at all
        target_h = int(round(bh * 0.92))
        if target_h < 8:
            return -1.0
        aspect = gray.shape[1] / max(1.0, float(gray.shape[0]))
        target_w = int(round(target_h * aspect))
        if target_w > bw * 1.15 or target_h > bh * 1.15:
            return -1.0
        tg, tm = self._scaled_rep(card, target_h)
        if tg.shape[0] > image.shape[0] or tg.shape[1] > image.shape[1]:
            return -1.0
        best = -1.0
        for dy in self.COARSE_OFFSETS:
            for dx in self.COARSE_OFFSETS:
                oy = int(round((bh - tg.shape[0]) / 2.0 + dy * bh))
                ox = int(round((bw - tg.shape[1]) / 2.0 + dx * bw))
                oy = max(0, min(image.shape[0] - tg.shape[0], oy))
                ox = max(0, min(image.shape[1] - tg.shape[1], ox))
                window = image[oy:oy + tg.shape[0], ox:ox + tg.shape[1]]
                if window.shape != tg.shape:
                    continue
                score = masked_zncc_at(window, tg, tm)
                if score > best:
                    best = score
        return float(best)

    # -- helpers --------------------------------------------------------------------
    def _aliases_of(self, card: str) -> List[str]:
        for group in self.duplicate_groups:
            if card in group:
                return list(group)
        return [card]

    def _crop_bar(
        self,
        bbox: Tuple[int, int, int, int],
        shape: Tuple[int, int, int],
        has_bar: bool,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Drop the strip where the game draws the HP bar above the sprite's head."""
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        h, w = shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        if has_bar:
            cut = int(round(self.BAR_CROP_FRACTION * (y2 - y1)))
            if (y2 - y1) - cut >= 14:
                y1 += cut
        return (x1, y1, x2, y2)

    def stats(self) -> str:
        if not self.enabled:
            return "naming off"
        if not self.is_ready:
            return "naming unavailable (no sprite library)"
        rate = (self.named / self.attempted * 100.0) if self.attempted else 0.0
        return f"named {self.named}/{self.attempted} ({rate:.0f}%)"
