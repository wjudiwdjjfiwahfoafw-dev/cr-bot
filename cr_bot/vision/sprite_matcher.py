"""Identify arena troops by comparing against *the game's own* sprite frames (no ML, no training).

Why the card artwork does not work
----------------------------------
The obvious reference — the promo renders at ``cdn.royaleapi.com/static/img/cards/<slug>.png`` —
does not match the arena.  Those are front-facing studio shots; the arena draws small, top-down,
team-tinted animation frames.  Measured on the two units of ``calibration_preview.png`` whose
identity is known (Mega Knight, right lane; Giant, left lane) the correct card ranked **34/78**
for both, i.e. chance, across several scoring variants.  The fix is to change the *reference*,
not the score.

``scripts/fetch_game_sprites.py`` therefore downloads the animation frames the game itself
draws (decrypted ``*_tex.sc`` dumps) and keeps a small, diverse subset per card.  This module
matches against those.

Two matching modes
------------------
``identify`` (**sliding**)
    Masked ZNCC of every template over a whole region.  Useful when you do not know where the
    unit is.  Be aware that on a busy arena background the winner is often a smooth, low-detail
    template that happens to fit some grass or a shadow — free sliding rewards lucky peaks.  Use
    it for coarse localisation only, and treat the score as weak evidence.

``identify_window`` (**fixed**) — preferred
    The unit's bounding box is known (from the foreground/health-bar detector).  Every card is
    forced to compete for *the same box*, only small sub-pixel offsets and a narrow scale range
    are searched.  This is far more discriminative and is what the arena detector uses.

Honest expectations
-------------------
Even with real game frames, a shipped template library recognises a *different* device's sprites
only when pose and scale line up; measured on the sample frame the shipped frames are not
trustworthy on their own. :class:`cr_bot.vision.unit_identity.UnitIdentityMemory` can match
locally labelled crops, but automatic labelling from deployments is heuristic, optional and
disabled by default; it does not establish accuracy on an opponent's different pose or tint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SPRITES_DIR = Path("assets/unit_sprites")

#: how many frames per card are kept in memory by default
DEFAULT_MAX_TEMPLATES = 6

#: scale multipliers searched around the expected sprite size
SCALE_STEPS = (0.82, 1.0, 1.22)


@dataclass(frozen=True)
class SpriteHit:
    """One card's best placement inside the searched region."""

    card: str
    score: float
    bbox: Tuple[int, int, int, int]
    height_px: int
    template: int

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass
class SpriteTemplate:
    """One reference frame of one card."""

    card: str
    gray: np.ndarray  # float32 0..1
    mask: np.ndarray  # float32 0/1, 1 where the sprite is opaque
    index: int
    source: str = ""
    #: file stem inside the card folder ("f03"), used to look up the measured faction
    stem: str = ""

    @property
    def aspect(self) -> float:
        return self.gray.shape[1] / float(self.gray.shape[0])


def decode_rgba(path: Path | str) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def prepare_gray(bgr: np.ndarray) -> np.ndarray:
    """Grayscale, lightly denoised, scaled to 0..1."""
    g = to_gray(bgr)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32) / 255.0


def masked_zncc(image: np.ndarray, template: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked zero-mean normalised cross-correlation of ``template`` over ``image``.

    ``S(x)`` below denotes a correlation with ``kernel`` anchored at the centre, so a handful of
    ``filter2D`` calls score every window position at once::

        score = (N*S(I*T*M) - S(I*M)*st) / (sqrt(N*S(I^2*M) - S(I*M)^2) * sqrt(N*st2 - st^2))

    The result is a score map the size of ``image``; windows that do not fit get ``-1``.
    """
    th, tw = template.shape
    if th >= image.shape[0] or tw >= image.shape[1]:
        return np.full(image.shape, -1.0, np.float32)

    n = float(mask.sum())
    if n < 24.0:
        return np.full(image.shape, -1.0, np.float32)

    t_masked = template * mask
    st = float(t_masked.sum())
    st2 = float((template * template * mask).sum())
    t_var = n * st2 - st * st
    if t_var <= 1e-6:
        return np.full(image.shape, -1.0, np.float32)

    anchor = (tw // 2, th // 2)
    s_i = cv2.filter2D(image, -1, mask, anchor=anchor, borderType=cv2.BORDER_REPLICATE)
    s_ii = cv2.filter2D(image * image, -1, mask, anchor=anchor, borderType=cv2.BORDER_REPLICATE)
    s_it = cv2.filter2D(image, -1, t_masked, anchor=anchor, borderType=cv2.BORDER_REPLICATE)

    numerator = n * s_it - s_i * st
    denom = np.sqrt(np.maximum(n * s_ii - s_i * s_i, 1e-6)) * np.sqrt(t_var)
    score = numerator / denom

    pad_t, pad_l = th // 2, tw // 2
    score[:pad_t, :] = -1.0
    score[max(0, score.shape[0] - (th - 1 - pad_t)):, :] = -1.0
    score[:, :pad_l] = -1.0
    score[:, max(0, score.shape[1] - (tw - 1 - pad_l)):] = -1.0
    return np.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)


def masked_zncc_at(window: np.ndarray, template: np.ndarray, mask: np.ndarray) -> float:
    """Masked ZNCC of a single aligned window (no sliding)."""
    if window.shape[:2] != template.shape[:2]:
        h = min(window.shape[0], template.shape[0])
        w = min(window.shape[1], template.shape[1])
        window = cv2.resize(window, (w, h), interpolation=cv2.INTER_AREA)
        template = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    m = mask > 0.5
    n = float(m.sum())
    if n < 24.0:
        return -1.0
    a = window[m]
    b = template[m]
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den <= 1e-9:
        return -1.0
    return float((a * b).sum() / den)


class SpriteLibrary:
    """Card -> real in-game animation frames, loaded from ``assets/unit_sprites``."""

    def __init__(
        self,
        sprites_dir: Path | str = DEFAULT_SPRITES_DIR,
        cards: Optional[Iterable[str]] = None,
        max_templates: int = DEFAULT_MAX_TEMPLATES,
        factions: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.sprites_dir = Path(sprites_dir)
        self.max_templates = max_templates
        self._wanted = set(cards) if cards else None
        self._templates: Dict[str, List[SpriteTemplate]] = {}
        # ``{card: {frame_stem: "enemy"|"ally"|"neutral"}}`` measured by
        # ``cr_bot.vision.sprite_factions``; empty when the library has not been audited, in
        # which case every frame is used exactly as before.
        self.factions: Dict[str, Dict[str, str]] = (
            factions if factions is not None else self._load_factions()
        )
        self.load()

    def _load_factions(self) -> Dict[str, Dict[str, str]]:
        try:
            from cr_bot.vision.sprite_factions import load_faction_map

            return load_faction_map(self.sprites_dir)
        except Exception:      # a missing/broken map is not an error: frames are used as-is
            return {}

    def load(self) -> int:
        self._templates.clear()
        if not self.sprites_dir.is_dir():
            logger.warning("no in-game sprite library at %s (run scripts/fetch_game_sprites.py)",
                           self.sprites_dir)
            return 0
        for card_dir in sorted(p for p in self.sprites_dir.iterdir() if p.is_dir()):
            card = card_dir.name
            if card.startswith("_"):
                continue
            if self._wanted is not None and card not in self._wanted:
                continue
            frames: List[SpriteTemplate] = []
            for path in self._frame_paths(card, sorted(card_dir.glob("*.png")))[: self.max_templates]:
                img = decode_rgba(path)
                if img is None:
                    continue
                mask = (img[:, :, 3] > 140).astype(np.float32)
                if mask.sum() < 80:
                    continue
                frames.append(
                    SpriteTemplate(card=card, gray=prepare_gray(img[:, :, :3]), mask=mask,
                                   index=len(frames), source=str(path), stem=path.stem)
                )
            if frames:
                self._templates[card] = frames
        total = sum(len(v) for v in self._templates.values())
        if total:
            logger.info("sprite library: %d cards / %d frames from %s",
                        len(self._templates), total, self.sprites_dir)
        return total

    def _frame_paths(self, card: str, paths: List[Path]) -> List[Path]:
        """Order a card's frames so the kept subset spans both team tints.

        Frames are chosen round-robin across the measured factions (neutral first, since a
        neutral render suits both teams) instead of taking the first N files.  Without the
        faction map — the common case, and the behaviour this module shipped with — the files
        are used in name order, so nothing changes for libraries that have not been audited.
        """
        measured = self.factions.get(card) or {}
        if not measured:
            return paths
        buckets: Dict[str, List[Path]] = {"neutral": [], "ally": [], "enemy": []}
        for path in paths:
            buckets.setdefault(measured.get(path.stem, "neutral"), []).append(path)
        ordered: List[Path] = []
        queues = [q for q in (buckets.get("neutral"), buckets.get("ally"), buckets.get("enemy")) if q]
        while any(queues):
            for queue in list(queues):
                if queue:
                    ordered.append(queue.pop(0))
            queues = [q for q in queues if q]
        return ordered

    def frames_by_faction(self, card: str, faction: str) -> List[SpriteTemplate]:
        """The loaded frames of one card that depict ``faction`` (empty if it has none)."""
        measured = self.factions.get(card) or {}
        if not measured:
            return self.frames_for(card)
        return [f for f in self.frames_for(card) if measured.get(f.stem, "neutral") == faction]

    def faction_coverage(self) -> Dict[str, float]:
        """Per card: how many of its loaded frames belong to each faction (for logs/HUD)."""
        out: Dict[str, float] = {}
        for card, frames in self._templates.items():
            measured = self.factions.get(card) or {}
            if not measured or not frames:
                continue
            ally = sum(1 for f in frames if measured.get(f.stem) == "ally")
            out[card] = ally / float(len(frames))
        return out

    @property
    def cards(self) -> List[str]:
        return sorted(self._templates)

    @property
    def is_ready(self) -> bool:
        return bool(self._templates)

    def frames_for(self, card: str) -> List[SpriteTemplate]:
        return self._templates.get(card, [])

    def median_aspect(self, card: str) -> Optional[float]:
        frames = self._templates.get(card)
        if not frames:
            return None
        return float(np.median([f.aspect for f in frames]))


class SpriteMatcher:
    """Scores how well each card's real sprite frames explain an observed patch of arena."""

    def __init__(
        self,
        library: Optional[SpriteLibrary] = None,
        sprites_dir: Path | str = DEFAULT_SPRITES_DIR,
        scale_steps: Sequence[float] = SCALE_STEPS,
        min_score: float = 0.55,
        relative_margin: float = 0.04,
    ) -> None:
        self.library = library if library is not None else SpriteLibrary(sprites_dir)
        self.scale_steps = tuple(scale_steps)
        self.min_score = min_score
        self.relative_margin = relative_margin
        self._scaled_cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray]] = {}

    @property
    def is_ready(self) -> bool:
        return self.library.is_ready

    # -- template cache ------------------------------------------------------------
    def _scaled(self, tpl: SpriteTemplate, height: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (tpl.source, tpl.index, height)
        cached = self._scaled_cache.get(key)
        if cached is not None:
            return cached
        h, w = tpl.gray.shape[:2]
        width = max(4, int(round(w * (height / float(h)))))
        height = max(4, int(height))
        gray = cv2.resize(tpl.gray, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(tpl.mask, (width, height), interpolation=cv2.INTER_AREA)
        mask = (mask > 0.5).astype(np.float32)
        if mask.sum() < 24:
            mask = np.ones_like(mask)
        self._scaled_cache[key] = (gray, mask)
        return gray, mask

    # -- fixed-window matching (preferred) -----------------------------------------
    def identify_window(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        candidates: Optional[Sequence[str]] = None,
        topk: int = 3,
        offset_search: float = 0.08,
    ) -> List[SpriteHit]:
        """Rank cards for a *known* unit bounding box.

        Every card is fitted into the same box (aspect preserved), with a small sub-pixel offset
        and scale search.  Because nothing is allowed to drift to a different part of the frame,
        this reduces, but does not eliminate, lucky matches to low-detail templates.
        """
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 12 or not self.library.is_ready:
            return []

        region = frame_bgr[y1:y2, x1:x2]
        image = prepare_gray(region)

        card_pool = list(candidates) if candidates else self.library.cards
        hits: List[SpriteHit] = []
        offsets = (-int(round(offset_search * bh)), 0, int(round(offset_search * bh)))

        for card in card_pool:
            frames = self.library.frames_for(card)
            if not frames:
                continue
            best_for_card: Optional[SpriteHit] = None
            per_frame: List[float] = []
            for tpl in frames:
                frame_best = -1.0
                frame_hit: Optional[SpriteHit] = None
                for scale in (1.0, 0.88, 1.14):
                    h = int(round(bh * scale))
                    w = int(round(h * tpl.aspect))
                    if w > bw * 1.25 or h > bh * 1.25 or w < 8 or h < 8:
                        continue
                    if h > image.shape[0] or w > image.shape[1]:
                        continue
                    tg, tm = self._scaled(tpl, h)
                    for dy in offsets:
                        for dx in offsets:
                            oy = int(round((bh - h) / 2.0)) + dy
                            ox = int(round((bw - w) / 2.0)) + dx
                            oy = max(0, min(bh - h, oy))
                            ox = max(0, min(bw - w, ox))
                            window = image[oy:oy + h, ox:ox + w]
                            score = masked_zncc_at(window, tg, tm)
                            if score > frame_best:
                                frame_best = score
                                frame_hit = SpriteHit(
                                    card=card, score=score,
                                    bbox=(x1 + ox, y1 + oy, x1 + ox + w, y1 + oy + h),
                                    height_px=h, template=tpl.index,
                                )
                if frame_hit is not None and frame_best > -1.0:
                    per_frame.append(frame_best)
                    if best_for_card is None or frame_best > best_for_card.score:
                        best_for_card = frame_hit
            if best_for_card is None or not per_frame:
                continue
            # a card with many frames gets more chances to hit a lucky peak: shrink the score
            # with the number of frames considered so cards compete on the same footing
            per_frame.sort(reverse=True)
            adjusted = best_for_card.score
            if len(per_frame) > 1:
                adjusted -= 0.25 * float(np.std(per_frame))
            hits.append(
                SpriteHit(card=card, score=float(adjusted), bbox=best_for_card.bbox,
                          height_px=best_for_card.height_px, template=best_for_card.template)
            )

        hits.sort(key=lambda h_: h_.score, reverse=True)
        return hits[:topk]

    # -- sliding matching ----------------------------------------------------------
    def identify(
        self,
        region_bgr: np.ndarray,
        candidates: Optional[Sequence[str]] = None,
        expected_height_px: Optional[float] = None,
        topk: int = 3,
        return_all: bool = False,
    ) -> List[SpriteHit]:
        """Slide every template over ``region_bgr``.

        Weak evidence on a textured background (see the module docstring); intended for coarse
        localisation or for regions that are already known to contain a single sprite.
        """
        if not self.library.is_ready or region_bgr is None or region_bgr.size == 0:
            return []

        image = prepare_gray(region_bgr)
        if expected_height_px is None:
            max_h = 0.8 * min(image.shape[0], image.shape[1])
            heights = [int(round(max_h * s)) for s in (0.45, 0.62, 0.82)]
        else:
            heights = sorted({max(12, int(round(expected_height_px * s))) for s in self.scale_steps})

        card_pool = list(candidates) if candidates else self.library.cards
        hits: List[SpriteHit] = []
        for card in card_pool:
            frames = self.library.frames_for(card)
            if not frames:
                continue
            best: Optional[SpriteHit] = None
            for tpl in frames:
                for height in heights:
                    tg, tm = self._scaled(tpl, height)
                    if tg.shape[0] >= image.shape[0] or tg.shape[1] >= image.shape[1]:
                        continue
                    scores = masked_zncc(image, tg, tm)
                    _mn, mx, _ml, loc = cv2.minMaxLoc(scores)
                    if mx <= -0.99:
                        continue
                    if best is None or mx > best.score:
                        h, w = tg.shape
                        bx = int(loc[0] - w // 2)
                        by = int(loc[1] - h // 2)
                        best = SpriteHit(card=card, score=float(mx), bbox=(bx, by, bx + w, by + h),
                                         height_px=h, template=tpl.index)
            if best is not None:
                hits.append(best)

        hits.sort(key=lambda h_: h_.score, reverse=True)
        return hits if return_all else hits[:topk]

    def confident(self, hits: Sequence[SpriteHit]) -> Optional[SpriteHit]:
        """The winning hit, if it is strong enough *and* clearly ahead of the runner-up."""
        if not hits:
            return None
        best = hits[0]
        if best.score < self.min_score:
            return None
        if len(hits) > 1 and (best.score - hits[1].score) < self.relative_margin:
            return None
        return best
