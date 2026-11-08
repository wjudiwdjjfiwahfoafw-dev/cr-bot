"""Give a detected troop its card name by matching it against the real in-game sprite library.

Why this module exists
----------------------
``SpriteLibrary``/``SpriteMatcher`` were in the repo, and ``assets/unit_sprites`` holds real
extracted animation frames for 53 cards — but nothing in the detection path ever called them. On
a real phone every box therefore came out as ``unknown_tank`` / ``unknown_medium`` /
``unknown_swarm``: the bot knew *where* a troop was and roughly how big, never *what* it was.
That is the difference between "something is at the bridge" and "that is a Mega Knight, keep the
Zap for the Bats".

Two problems had to be solved before it could be switched on, and both are measured in
``tools/recognition_benchmark.py``:

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

The answer is only returned when it clears an absolute score floor *and* beats the runner-up by a
margin: a wrong name is worse than no name, because the brain uses a name to pick a counter and
to decide whether a spell resets it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cr_bot.vision.sprite_matcher import (
    SpriteHit,
    SpriteLibrary,
    SpriteMatcher,
    SpriteTemplate,
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
    #: best score of any other *card* (sibling names for the same art are not counted)
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

    DEFAULT_MIN_SCORE = 0.60
    DEFAULT_MIN_MARGIN = 0.05
    DEFAULT_COARSE_KEEP = 6
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
        self.coarse_keep = int(coarse_keep)
        self._matcher = matcher
        self._sprites_dir = Path(sprites_dir)
        self._max_cards = max_cards
        #: card -> (gray, mask) of its median frame, prepared once (see _representatives)
        self._reps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._scaled: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        #: cards whose art is byte-identical to another card's (skeletons vs skeleton_army, ...)
        self.duplicate_groups: List[List[str]] = []
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
            self._prepare()
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
        groups: Dict[str, List[str]] = {}
        for card in self._matcher.library.cards:
            frames = self._matcher.library.frames_for(card)
            if not frames:
                continue
            # median frame by aspect: the most "typical" pose of that card
            rep = sorted(frames, key=lambda f: abs(f.aspect - float(np.median(
                [x.aspect for x in frames]))))[0]
            self._reps[card] = (rep.gray, rep.mask)
            key = f"{rep.gray.shape}:{int(rep.mask.sum())}"
            groups.setdefault(key, []).append(card)
        self.duplicate_groups = [g for g in groups.values() if len(g) > 1]
        if self.duplicate_groups:
            logger.debug("cards sharing identical art (kept as aliases): %s",
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
        for card in self._reps:
            score = self._coarse_score(card, image, bh, bw)
            if score > -1.0:
                coarse.append((score, card))
        if not coarse:
            return None
        coarse.sort(reverse=True)
        shortlist = [card for _score, card in coarse[: self.coarse_keep]]

        # --- stage 2: proper ranking of the survivors -------------------------------
        hits = matcher.identify_window(frame_bgr, box, candidates=shortlist, topk=max(2, topk))
        if not hits:
            # fall back to the coarse ranking; it is the same evidence, just less thorough
            hits = [SpriteHit(card=card, score=score, bbox=box, height_px=bh, template=0)
                    for score, card in coarse[: max(2, topk)]]

        best = hits[0]
        aliases = self._aliases_of(best.card)
        # The library contains cards whose art is byte-identical (a Skeleton Army frame is a
        # Skeleton frame). Two names for one picture is not ambiguity, so those are excluded
        # from the runner-up: otherwise every skeleton would be rejected for having a "tie".
        runner_up = -1.0
        for hit in hits[1:]:
            if hit.card in aliases:
                continue
            runner_up = max(runner_up, hit.score)
        margin = float(best.score - runner_up) if runner_up > -1.0 else 1.0

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
