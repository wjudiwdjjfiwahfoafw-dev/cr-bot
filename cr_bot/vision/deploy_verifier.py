"""
Closed-loop verification that a card actually left the hand.

This is the single most important safety net for real-device play. Injected touches fail
silently: the game simply ignores a tap that lands between slots, outside the deploy zone,
or during a UI animation. A bot that assumes every gesture succeeded desynchronises from its
hand model within one or two plays and then spends the rest of the match "playing" cards it
no longer has — which looks exactly like a bot that does nothing.

Rather than depend on card *identity* recognition (which is the fragile part), this compares
the raw pixels of each hand slot before and after the gesture. In Clash Royale a played card
is replaced in its slot by the next card in the cycle, so a successful deploy always changes
that slot's crop, and a failed one never does.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Patch = np.ndarray


@dataclass
class SlotSnapshot:
    """Small, blurred grayscale thumbnails of the four hand slots."""

    patches: List[Patch]
    taken_at: float = field(default_factory=time.monotonic)

    def __len__(self) -> int:
        return len(self.patches)


class DeployVerifier:
    """Detects whether the hand changed after a deploy, per slot."""

    def __init__(self, thumb_size: Tuple[int, int] = (24, 28), change_threshold: float = 9.0) -> None:
        self.thumb_size = thumb_size
        self.change_threshold = float(change_threshold)

    # -- capture -------------------------------------------------------------------
    def snapshot(self, frame_bgr: np.ndarray, grid) -> Optional[SlotSnapshot]:
        if frame_bgr is None:
            return None
        patches: List[Patch] = []
        for i in range(4):
            x1, y1, x2, y2 = grid.hand_slot_rect(i)
            if x2 - x1 < 6 or y2 - y1 < 6:
                return None
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            small = cv2.resize(crop, self.thumb_size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            patches.append(cv2.GaussianBlur(gray, (3, 3), 0))
        return SlotSnapshot(patches=patches)

    # -- comparison ----------------------------------------------------------------
    @staticmethod
    def _diff(a: Patch, b: Patch) -> float:
        if a is None or b is None or a.shape != b.shape:
            return 255.0
        return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

    def slot_changed(self, before: SlotSnapshot, after: SlotSnapshot, slot: int) -> bool:
        if before is None or after is None:
            return False
        if slot < 0 or slot >= len(before) or slot >= len(after):
            return False
        return self._diff(before.patches[slot], after.patches[slot]) >= self.change_threshold

    def changed_slots(self, before: SlotSnapshot, after: SlotSnapshot) -> List[int]:
        return [i for i in range(min(len(before), len(after))) if self.slot_changed(before, after, i)]

    def describe_changes(self, before: SlotSnapshot, after: SlotSnapshot) -> str:
        if before is None or after is None:
            return "no snapshot"
        parts = []
        for i in range(min(len(before), len(after))):
            parts.append(f"s{i}:{self._diff(before.patches[i], after.patches[i]):.1f}")
        return " ".join(parts)


def wait_for_deploy(
    verifier: DeployVerifier,
    frame_source,
    grid,
    before: SlotSnapshot,
    slot: int,
    timeout: float = 0.9,
    poll: float = 0.05,
) -> bool:
    """Poll frames until ``slot`` changes (success) or ``timeout`` elapses (failure)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = frame_source()
        if frame is None:
            time.sleep(poll)
            continue
        after = verifier.snapshot(frame, grid)
        if after is None:
            time.sleep(poll)
            continue
        if verifier.slot_changed(before, after, slot):
            return True
        time.sleep(poll)
    return False
