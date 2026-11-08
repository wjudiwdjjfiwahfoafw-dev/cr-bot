"""Optional per-device appearance memory for locally labelled troop crops.

A high-confidence template match can name a later, visually similar crop, but the correlation
is not a calibrated probability and cannot reliably bridge arbitrary poses, tints or backgrounds.
The library abstains when score/margin gates fail; an empty library leaves units generic.

The bot's experimental, opt-in online learner only uses a verified hand change followed by a
unique, new allied segmentation track near the drop over two frames. This is a *heuristic*, not
human-verified ground truth: missed units, effects and spawned troops can still contaminate it.
It is disabled by default. Samples from older, less guarded versions use a different namespace.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path("assets/unit_identity")
PATCH_SIZE = (40, 48)  # (w, h) of the stored, normalised sample


def normalise_patch(patch_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Crop-agnostic appearance descriptor: fixed size, brightness/flat contrast equalised."""
    if patch_bgr is None or patch_bgr.size == 0:
        return None
    h, w = patch_bgr.shape[:2]
    if h < 6 or w < 6:
        return None
    gray = patch_bgr if patch_bgr.ndim == 2 else cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, PATCH_SIZE, interpolation=cv2.INTER_AREA)
    small = cv2.equalizeHist(small)
    return cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32) / 255.0


def patch_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised correlation in [-1, 1]."""
    if a is None or b is None or a.shape != b.shape:
        return -1.0
    va = a.ravel() - float(a.mean())
    vb = b.ravel() - float(b.mean())
    den = float(np.sqrt((va * va).sum() * (vb * vb).sum()))
    if den <= 1e-9:
        return -1.0
    return float((va * vb).sum() / den)


@dataclass
class IdentityMatch:
    card: str
    score: float
    runner_up: float

    @property
    def margin(self) -> float:
        return self.score - self.runner_up


class UnitIdentityMemory:
    """Persistent card -> crop library; callers are responsible for label quality."""

    def __init__(
        self,
        directory: Path | str = DEFAULT_DIR,
        namespace: str = "default",
        max_samples_per_card: int = 8,
        min_score: float = 0.60,
        min_margin: float = 0.02,
    ) -> None:
        self.directory = Path(directory)
        self.namespace = namespace
        self.max_samples_per_card = int(max_samples_per_card)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self._samples: Dict[str, List[np.ndarray]] = {}
        self._stats: Dict[str, int] = {}
        self.load()

    # -- persistence ---------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.directory / f"{self.namespace}.npz"

    def load(self) -> int:
        self._samples.clear()
        self._stats.clear()
        if not self.path.is_file():
            return 0
        try:
            with np.load(self.path, allow_pickle=False) as data:
                # keys are "<card>__<index>"; everything else is bookkeeping
                for key in data.files:
                    if key in ("cards", "stats"):
                        continue
                    card, _, idx = key.partition("__")
                    if not card or not idx.isdigit():
                        continue
                    sample = np.asarray(data[key], dtype=np.float32)
                    if sample.shape != (PATCH_SIZE[1], PATCH_SIZE[0]) or not np.isfinite(sample).all():
                        logger.warning("ignoring invalid identity sample %s in %s", key, self.path)
                        continue
                    bucket = self._samples.setdefault(card, [])
                    if len(bucket) < self.max_samples_per_card:
                        bucket.append(sample)
        except Exception as exc:  # pragma: no cover - corrupt file should never break the bot
            logger.warning("could not read identity memory %s (%s); starting empty", self.path, exc)
            self._samples.clear()
        return self.sample_count

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        for card, samples in self._samples.items():
            for i, sample in enumerate(samples):
                arrays[f"{card}__{i}"] = sample
        arrays["cards"] = np.array(json.dumps(sorted(self._samples)), dtype="U")
        arrays["stats"] = np.array(json.dumps(self._stats, sort_keys=True), dtype="U")
        # Never replace a usable library with a partially written zip after a crash or I/O error.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{self.namespace}.", suffix=".npz",
                                             dir=self.directory, delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return self.path

    # -- learning ------------------------------------------------------------------
    def add_sample(self, card: str, patch_bgr: np.ndarray, dedupe: float = 0.985) -> bool:
        """File a crop under ``card``; returns False if it was unusable or a duplicate."""
        if not card:
            return False
        patch = normalise_patch(patch_bgr)
        if patch is None:
            return False
        bucket = self._samples.setdefault(card, [])
        for existing in bucket:
            if patch_similarity(existing, patch) >= dedupe:
                return False
        bucket.append(patch)
        if len(bucket) > self.max_samples_per_card:
            del bucket[0]
        self._stats[card] = self._stats.get(card, 0) + 1
        return True

    def add_from_frame(
        self,
        card: str,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        pad: float = 0.06,
    ) -> bool:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        dx = int(round((x2 - x1) * pad))
        dy = int(round((y2 - y1) * pad))
        x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
        x2, y2 = min(frame_bgr.shape[1], x2 + dx), min(frame_bgr.shape[0], y2 + dy)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return False
        return self.add_sample(card, frame_bgr[y1:y2, x1:x2])

    # -- recognition ---------------------------------------------------------------
    def match(
        self,
        patch_bgr: np.ndarray,
        candidates: Optional[Sequence[str]] = None,
    ) -> Optional[IdentityMatch]:
        """Best card for an observed crop, or ``None`` when unsure."""
        patch = normalise_patch(patch_bgr)
        if patch is None or not self._samples:
            return None
        pool = list(candidates) if candidates else list(self._samples)
        scored: List[Tuple[float, str]] = []
        for card in pool:
            samples = self._samples.get(card)
            if not samples:
                continue
            best = max(patch_similarity(sample, patch) for sample in samples)
            if np.isfinite(best):
                scored.append((best, card))
        if not scored:
            return None
        scored.sort(reverse=True)
        best_score, best_card = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        match = IdentityMatch(card=best_card, score=best_score, runner_up=runner_up)
        if match.score < self.min_score or match.margin < self.min_margin:
            return None
        return match

    # -- introspection -------------------------------------------------------------
    @property
    def sample_count(self) -> int:
        return sum(len(v) for v in self._samples.values())

    @property
    def known_cards(self) -> List[str]:
        return sorted(c for c, v in self._samples.items() if v)

    def sample_counts(self) -> Dict[str, int]:
        return {card: len(v) for card, v in sorted(self._samples.items())}

    def clear(self) -> None:
        self._samples.clear()
        self._stats.clear()
