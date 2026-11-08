"""
Pending Deployment Memory & Real-Game ADB Tap Verification.

Solves two critical real-game problems:
1. During the ~1.15s between sending an ADB tap and the troop finishing its Clash Royale
   deploy animation / health-bar appearance, Vision cannot see the unit yet. Without pending
   memory, bots double- or triple-commit elixir onto the same threat!
2. In real emulators, occasional frame lag or touch drops can cause a tap to be ignored by the
   game client. `verify_hand_transition` detects if the tapped card never left the hand slot
   and clears cooldowns so the bot immediately retries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from cr_bot.config.cards_db import get_card_info
from cr_bot.models import BattleState, DetectedUnit, Side


@dataclass
class PendingDeploy:
    card_name: str
    slot_index: int
    tile_x: float
    tile_y: float
    elixir_before: float
    sent_ts: float
    expire_ts: float
    verified_spent: bool = False


class PendingDeployTracker:
    """Tracks in-flight card plays between ADB tap and visual arena confirmation."""

    PENDING_TTL_SEC = 1.65

    def __init__(self) -> None:
        self._pending: List[PendingDeploy] = []
        self.dropped_tap_retries: int = 0

    def reset(self) -> None:
        self._pending.clear()

    def record_play(
        self,
        card_name: str,
        slot_index: int,
        tile_x: float,
        tile_y: float,
        elixir_before: float,
        now_ts: float,
    ) -> None:
        info = get_card_info(card_name)
        ttl = 0.85 if info.is_spell else self.PENDING_TTL_SEC
        if card_name == "miner":
            ttl = 2.1  # Miner burrows underground before appearing at target tower
        self._pending.append(
            PendingDeploy(
                card_name=card_name,
                slot_index=slot_index,
                tile_x=tile_x,
                tile_y=tile_y,
                elixir_before=elixir_before,
                sent_ts=now_ts,
                expire_ts=now_ts + ttl,
            )
        )

    def reconcile(self, state: BattleState) -> Optional[str]:
        """
        Prune expired or visually confirmed pending deploys, and return `card_name` if a real-game
        ADB tap failed to register (same card still in same slot > 0.85s later with no elixir drop).
        """
        failed_card: Optional[str] = None
        keep: List[PendingDeploy] = []

        for p in self._pending:
            age = state.timestamp - p.sent_ts
            if state.timestamp > p.expire_ts:
                continue

            # Check if hand slot rotated or elixir dropped (confirms server accepted play)
            if not p.verified_spent and age >= 0.25:
                same_slot = next((s for s in state.hand if s.slot_index == p.slot_index), None)
                if same_slot is None or same_slot.card_name != p.card_name:
                    p.verified_spent = True
                elif state.elixir < (p.elixir_before - 0.9):
                    p.verified_spent = True
                elif age >= 0.88:
                    # Tap was dropped by emulator/client! Discard pending entry and signal retry.
                    failed_card = p.card_name
                    self.dropped_tap_retries += 1
                    continue

            # Check if Vision already sees an ally unit of this type near the target tile
            confirmed_on_arena = any(
                a.name == p.card_name
                and (state.timestamp - a.first_seen_ts) <= 1.8
                and math.hypot(a.tile_x - p.tile_x, a.tile_y - p.tile_y) <= 3.2
                for a in state.ally_units
            )
            if confirmed_on_arena:
                continue

            keep.append(p)

        self._pending = keep
        return failed_card

    def virtual_ally_units(self, now_ts: float) -> List[DetectedUnit]:
        """Synthesize virtual ally units for unconfirmed in-flight deploys so ThreatAnalyzer counts them."""
        out: List[DetectedUnit] = []
        for idx, p in enumerate(self._pending):
            info = get_card_info(p.card_name)
            if info.is_spell and p.card_name != "zap":
                continue
            out.append(
                DetectedUnit(
                    track_id=-100 - idx,
                    name=p.card_name,
                    side=Side.ALLY,
                    tile_x=p.tile_x,
                    tile_y=p.tile_y,
                    confidence=1.0,
                    hp_ratio=1.0,
                    move_type=info.move_type,
                    weight_class=info.weight_class,
                    first_seen_ts=p.sent_ts,
                    last_seen_ts=now_ts,
                    source="pending",
                )
            )
        return out

    def has_pending_spell_near(self, tx: float, ty: float, radius: float = 3.2) -> bool:
        return any(
            get_card_info(p.card_name).is_spell and math.hypot(p.tile_x - tx, p.tile_y - ty) <= radius
            for p in self._pending
        )
