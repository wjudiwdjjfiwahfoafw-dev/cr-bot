"""
v2.0 Utility-Scoring Tactical Brain & Timed Combo Sequencer for MK Miner Skeleton Barrel Zap Bait.

Key v2.0 Upgrades over v1.0:
- PendingDeployTracker integration: zero double/triple over-commits during ADB + server spawn delay
- Bridge Sniper immediate interception (answers Princess / Magic Archer / Dart Goblin at y >= 13.6)
- Exact Goblin Barrel timing: Goblin Gang drops immediately (1.0s deploy syncs with landing); Zap waits
  until landing so it never whiffs mid-flight, and never stacks both cards on one barrel
- Splash-Danger & E-Wiz Isolation: avoids feeding Bats/Gang into Baby Dragon/Wizard splash; snipes
  Electro Wizard/support with Miner while flanking Inferno Dragon on the opposite side
- Active Enemy Building & Inferno-Tower Counterplay: avoids throwing Skeleton Barrel into active Cannon/Tesla;
  distracts enemy Inferno Tower with river-center Bats / Miner when our Mega Knight advances
- Opponent 4-Card Counter-Cycle Exploitation: boosts Mega Knight / Swarm aggression when opponent's
  P.E.K.K.A, Inferno Tower, or small spell is tracked out of hand
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from cr_bot.arena.grid import ArenaGrid
from cr_bot.brain.pending_tracker import PendingDeployTracker
from cr_bot.brain.threat_analyzer import BoardAssessment, ThreatAnalyzer
from cr_bot.config.cards_db import get_card_info
from cr_bot.config.loader import BotConfig
from cr_bot.models import BattleState, BotAction, GamePhase, HandSlot, Lane, MoveType, QueuedAction


class MKZapBaitBrain:
    """High-IQ utility & combo planner engineered for Mega Knight Miner Skeleton Barrel Zap Bait."""

    MIN_PLAY_SCORE = 52.0

    def __init__(
        self,
        grid: ArenaGrid,
        config: Optional[BotConfig] = None,
        seed: Optional[int] = None,
        adb_latency_sec: float = 0.12,
    ) -> None:
        self.grid = grid
        self.cfg = config or BotConfig()
        self.analyzer = ThreatAnalyzer(grid, adb_latency_sec=adb_latency_sec)
        self.pending = PendingDeployTracker()
        self._rng = random.Random(seed)
        self._queued_combo: Optional[QueuedAction] = None
        self._last_action_ts: float = -999.0
        self._last_zap_reset_ts: float = -999.0
        self.last_top_candidates: List[BotAction] = []

    def reset(self) -> None:
        self.pending.reset()
        self._queued_combo = None
        self._last_action_ts = -999.0
        self._last_zap_reset_ts = -999.0
        self.last_top_candidates = []

    def decide(self, state: BattleState) -> Optional[BotAction]:
        """Evaluate BattleState and return the highest-utility legal action (or None to hold elixir)."""
        if not state.phase.is_in_battle:
            self.last_top_candidates = []
            return None

        # Reconcile pending in-flight deploys & detect if a previous real-game ADB tap was dropped!
        dropped_card = self.pending.reconcile(state)
        if dropped_card is not None:
            self._last_action_ts = -999.0  # Clear cooldown immediately to retry!

        if (state.timestamp - self._last_action_ts) < self.cfg.min_action_cooldown_sec:
            return None

        virtual_allies = self.pending.virtual_ally_units(state.timestamp)
        board = self.analyzer.analyze(state, pending_allies=virtual_allies)
        max_lane_urgency = max(
            board.lanes[Lane.LEFT].urgency_score,
            board.lanes[Lane.RIGHT].urgency_score,
        )

        # 1. Check queued multi-card combo step (e.g. delayed Miner right before Skeleton Barrel pops)
        if self._queued_combo is not None:
            if state.timestamp > self._queued_combo.max_expire_ts or max_lane_urgency >= 72.0:
                self._queued_combo = None
            elif state.timestamp >= self._queued_combo.execute_at_ts:
                slot = state.find_in_hand(self._queued_combo.card_name)
                if slot and slot.is_ready and state.elixir >= max(slot.elixir_cost, self._queued_combo.required_min_elixir):
                    q = self._queued_combo
                    self._queued_combo = None
                    tx, ty = self._clamp_for_card(q.card_name, q.tile_x, q.tile_y, state)
                    chosen = BotAction(
                        card_name=q.card_name,
                        slot_index=slot.slot_index,
                        tile_x=tx,
                        tile_y=ty,
                        score=89.0,
                        reason=f"[COMBO STEP 2] {q.reason}",
                    )
                    self._commit_action(chosen, state)
                    self.last_top_candidates = [chosen]
                    return chosen
                if max_lane_urgency < 42.0:
                    return None
            elif max_lane_urgency < 42.0:
                # Hold elixir for the queued combo follow-up so we don't spend the Miner's elixir early!
                return None

        # 2. Generate and score candidate plays across ready cards in hand
        candidates: List[BotAction] = []
        ready_slots = [
            s for s in state.hand if s.is_ready and s.card_name != "unknown" and state.elixir + 0.1 >= s.elixir_cost
        ]
        if not ready_slots:
            self.last_top_candidates = []
            return None

        self._score_bridge_sniper_response(state, board, ready_slots, candidates)
        self._score_emergency_defense(state, board, ready_slots, candidates)
        self._score_mega_knight_crush(state, board, ready_slots, candidates)
        self._score_tank_melter_and_kiting(state, board, ready_slots, candidates)
        self._score_zap_tactics(state, board, ready_slots, candidates)
        self._score_counterpush_conversions(state, board, ready_slots, candidates)
        self._score_opposite_lane_punish(state, board, ready_slots, candidates)
        self._score_bait_offense_and_combos(state, board, ready_slots, candidates)
        self._score_cycle_and_overtime_finish(state, board, ready_slots, candidates)

        # Universal Anti-Deadlock Safety Net: NEVER freeze at >= 9.5 elixir due to an awkward 4-card hand!
        if state.elixir >= 9.5 and (not candidates or max(c.score for c in candidates) < self.MIN_PLAY_SCORE):
            self._score_anti_deadlock_fallback(state, board, ready_slots, candidates)

        if not candidates:
            self.last_top_candidates = []
            return None

        finalized: List[BotAction] = []
        for cand in candidates:
            tx, ty = self._clamp_for_card(cand.card_name, cand.tile_x, cand.tile_y, state)
            finalized.append(
                BotAction(
                    card_name=cand.card_name,
                    slot_index=cand.slot_index,
                    tile_x=tx,
                    tile_y=ty,
                    score=round(cand.score, 1),
                    reason=cand.reason,
                    followup=cand.followup,
                )
            )

        finalized.sort(key=lambda a: a.score, reverse=True)
        self.last_top_candidates = finalized[:3]

        best = finalized[0]
        if best.score < self.MIN_PLAY_SCORE:
            return None

        self._commit_action(best, state)
        return best

    def _commit_action(self, action: BotAction, state: BattleState) -> None:
        self._last_action_ts = state.timestamp
        if action.card_name == "zap":
            self._last_zap_reset_ts = state.timestamp
        if action.followup is not None:
            self._queued_combo = action.followup
        self.pending.record_play(
            card_name=action.card_name,
            slot_index=action.slot_index,
            tile_x=action.tile_x,
            tile_y=action.tile_y,
            elixir_before=state.elixir,
            now_ts=state.timestamp,
        )

    # -------------------------------------------------------------------------
    # Tactical Generators
    # -------------------------------------------------------------------------

    def _score_bridge_sniper_response(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        """Immediately eliminate Princess / Dart Goblin / Magic Archer / Siege outranging our tower!"""
        slot_map = {s.card_name: s for s in ready}

        for lane in (Lane.LEFT, Lane.RIGHT):
            sniper = board.lanes[lane].bridge_sniper
            if sniper is None:
                continue

            bx, by = self.grid.LEFT_BRIDGE if lane == Lane.LEFT else self.grid.RIGHT_BRIDGE

            # If Princess/sniper is on enemy side of bridge (y < 16.8), Miner snipes directly OR bridge swarm crosses in 0.5s!
            if "goblin_gang" in slot_map and sniper.name in ("princess", "dart_goblin", "royal_giant", "x_bow", "mortar"):
                out.append(
                    BotAction(
                        card_name="goblin_gang",
                        slot_index=slot_map["goblin_gang"].slot_index,
                        tile_x=sniper.tile_x,
                        tile_y=max(by, sniper.tile_y),
                        score=94.5,
                        reason=f"Instant Goblin Gang eliminates bridge {sniper.name} ({lane.value})",
                    )
                )
            if "miner" in slot_map and sniper.name in ("princess", "magic_archer", "dart_goblin", "x_bow", "mortar"):
                out.append(
                    BotAction(
                        card_name="miner",
                        slot_index=slot_map["miner"].slot_index,
                        tile_x=sniper.tile_x,
                        tile_y=sniper.tile_y,
                        score=93.5,
                        reason=f"Direct Miner assassinate on bridge {sniper.name} ({lane.value})",
                    )
                )
            if "bats" in slot_map and sniper.name in ("princess", "x_bow", "mortar", "royal_giant"):
                # Princess attacks every 3.0s; 5 fast Bats reach & kill her between shots!
                out.append(
                    BotAction(
                        card_name="bats",
                        slot_index=slot_map["bats"].slot_index,
                        tile_x=sniper.tile_x,
                        tile_y=max(by, sniper.tile_y),
                        score=89.5,
                        reason=f"Fast bridge Bats snipe {sniper.name} ({lane.value})",
                    )
                )
            if "spear_goblins" in slot_map and sniper.name in ("princess", "dart_goblin"):
                out.append(
                    BotAction(
                        card_name="spear_goblins",
                        slot_index=slot_map["spear_goblins"].slot_index,
                        tile_x=sniper.tile_x,
                        tile_y=max(by, sniper.tile_y),
                        score=88.0,
                        reason=f"Bridge Spear Goblins trade with {sniper.name} ({lane.value})",
                    )
                )

    def _score_emergency_defense(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        slot_map = {s.card_name: s for s in ready}

        for lane in (Lane.LEFT, Lane.RIGHT):
            leval = board.lanes[lane]
            def_x, def_y = leval.def_anchor_xy

            # A) Goblin Barrel / Miner / Swarm directly on our Tower
            emerg = leval.tower_emergency_unit
            if emerg is not None:
                # Check if we already committed a pending spell or swarm right on this tower!
                already_defended = self.pending.has_pending_spell_near(emerg.tile_x, emerg.tile_y, 3.2) or any(
                    a.tile_y >= 22.0 and math.hypot(a.tile_x - emerg.tile_x, a.tile_y - emerg.tile_y) <= 3.5
                    for a in leval.ally_units
                )
                if not already_defended:
                    # Goblin Gang has 1.0s deploy -> drop IMMEDIATELY when Goblin Barrel flies so Gang wakes as Barrel lands!
                    if emerg.name in ("goblin_barrel", "miner", "knight", "bandit", "hog_rider") and "goblin_gang" in slot_map:
                        if not leval.has_splash_anti_swarm or emerg.name == "goblin_barrel":
                            out.append(
                                BotAction(
                                    card_name="goblin_gang",
                                    slot_index=slot_map["goblin_gang"].slot_index,
                                    tile_x=emerg.tile_x,
                                    tile_y=min(28.0, emerg.tile_y + 0.5),
                                    score=95.5 if emerg.name == "goblin_barrel" else 90.0,
                                    reason=f"Timed Goblin Gang catches {emerg.name} on {lane.value} tower",
                                )
                            )
                    # Zap on Goblin Barrel / Skeleton Army / Bats (ONLY once barrel is ready to land!)
                    if (
                        emerg.name in ("goblin_barrel", "skeleton_army", "bats", "spear_goblins")
                        and "zap" in slot_map
                        and leval.barrel_ready_for_zap
                    ):
                        zx, zy = self.analyzer.predict_intercept_tile(emerg, card_delay_sec=0.45)
                        evo_bonus = 3.5 if slot_map["zap"].is_evolved else 0.0
                        out.append(
                            BotAction(
                                card_name="zap",
                                slot_index=slot_map["zap"].slot_index,
                                tile_x=zx,
                                tile_y=zy,
                                score=96.0 + evo_bonus,
                                reason=f"Timed Zap on {emerg.name} protecting {lane.value} tower",
                            )
                        )
                    # Bats / Spear Goblins ONLY if lane still has unanswered threat AND no splash killer right there!
                    if leval.unanswered_threat >= 1.8 and not leval.has_splash_anti_swarm:
                        for light_def in ("bats", "spear_goblins"):
                            if light_def in slot_map and emerg.name not in ("golem",):
                                out.append(
                                    BotAction(
                                        card_name=light_def,
                                        slot_index=slot_map[light_def].slot_index,
                                        tile_x=emerg.tile_x,
                                        tile_y=min(28.0, emerg.tile_y + 0.6),
                                        score=83.0,
                                        reason=f"Emergency {light_def} DPS on {emerg.name} ({lane.value})",
                                    )
                                )

            # Stop adding extra defense if our deployed/pending troops already cover this lane!
            if leval.unanswered_threat <= 0.8:
                continue

            # B) Fast bridge rushers (Hog Rider, Balloon, Battle Ram, Ram Rider, Bandit, E-Barbs)
            rusher = leval.bridge_rusher
            if rusher is not None:
                r_info = get_card_info(rusher.name)
                if rusher.move_type == MoveType.AIR:
                    if "bats" in slot_map and not leval.has_splash_anti_swarm:
                        out.append(
                            BotAction(
                                card_name="bats",
                                slot_index=slot_map["bats"].slot_index,
                                tile_x=rusher.tile_x,
                                tile_y=max(18.0, rusher.tile_y + 1.2),
                                score=93.5,
                                reason=f"High-DPS Bats shred air rusher ({rusher.name})",
                            )
                        )
                    if "inferno_dragon" in slot_map and not any(a.name == "inferno_dragon" for a in leval.ally_units):
                        ix, iy = (
                            self.grid.IDRAG_OUTER_LEFT if lane == Lane.LEFT else self.grid.IDRAG_OUTER_RIGHT
                        )
                        out.append(
                            BotAction(
                                card_name="inferno_dragon",
                                slot_index=slot_map["inferno_dragon"].slot_index,
                                tile_x=ix,
                                tile_y=iy,
                                score=91.5,
                                reason=f"Inferno Dragon melts air rusher ({rusher.name})",
                            )
                        )
                    if "spear_goblins" in slot_map:
                        out.append(
                            BotAction(
                                card_name="spear_goblins",
                                slot_index=slot_map["spear_goblins"].slot_index,
                                tile_x=7.5 if lane == Lane.LEFT else 10.5,
                                tile_y=23.5,
                                score=79.5,
                                reason=f"Anti-air Spear Goblins pull/chip {rusher.name}",
                            )
                        )
                else:
                    if not r_info.is_splash and not leval.has_splash_anti_swarm:
                        if "goblin_gang" in slot_map:
                            out.append(
                                BotAction(
                                    card_name="goblin_gang",
                                    slot_index=slot_map["goblin_gang"].slot_index,
                                    tile_x=def_x,
                                    tile_y=max(18.5, min(23.8, rusher.tile_y + 2.4)),
                                    score=87.5,
                                    reason=f"Goblin Gang body-blocks & shreds {rusher.name}",
                                )
                            )
                        if "bats" in slot_map:
                            out.append(
                                BotAction(
                                    card_name="bats",
                                    slot_index=slot_map["bats"].slot_index,
                                    tile_x=rusher.tile_x,
                                    tile_y=max(18.0, rusher.tile_y + 1.4),
                                    score=83.5,
                                    reason=f"Uncontested Bats DPS vs {rusher.name}",
                                )
                            )

            # C) Enemy air escorts on our side (even if a ground tank/rusher like Golem is also in front!)
            if leval.air_threat_our_side >= 2.5 and (rusher is None or rusher.move_type == MoveType.GROUND):
                air_enemies = [e for e in leval.enemy_units if e.move_type == MoveType.AIR and e.tile_y >= 16.0]
                if air_enemies:
                    ae = air_enemies[0]
                    ae_info = get_card_info(ae.name)
                    if "inferno_dragon" in slot_map and not any(a.name == "inferno_dragon" for a in leval.ally_units):
                        ix, iy = (
                            self.grid.IDRAG_OUTER_LEFT if lane == Lane.LEFT else self.grid.IDRAG_OUTER_RIGHT
                        )
                        out.append(
                            BotAction(
                                card_name="inferno_dragon",
                                slot_index=slot_map["inferno_dragon"].slot_index,
                                tile_x=ix,
                                tile_y=22.5,
                                score=85.0,
                                reason=f"Inferno Dragon melts enemy air escort ({ae.name})",
                            )
                        )
                    if "bats" in slot_map and not ae_info.is_splash:
                        out.append(
                            BotAction(
                                card_name="bats",
                                slot_index=slot_map["bats"].slot_index,
                                tile_x=ae.tile_x,
                                tile_y=max(18.0, ae.tile_y + 1.2),
                                score=82.0,
                                reason=f"Bats intercept single-target air ({ae.name})",
                            )
                        )
                    if "spear_goblins" in slot_map:
                        # Place toward center away from Baby Dragon splash radius around the main tank!
                        out.append(
                            BotAction(
                                card_name="spear_goblins",
                                slot_index=slot_map["spear_goblins"].slot_index,
                                tile_x=7.6 if lane == Lane.LEFT else 10.4,
                                tile_y=23.2,
                                score=77.5,
                                reason=f"Safe center Spear Goblins shoot air escort ({ae.name})",
                            )
                        )

    def _score_mega_knight_crush(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        mk_slot = next((s for s in ready if s.card_name == "mega_knight"), None)
        if mk_slot is None:
            return

        for lane in (Lane.LEFT, Lane.RIGHT):
            leval = board.lanes[lane]
            stomp = leval.best_mk_stomp
            if stomp is None or stomp.near_pekka_danger:
                continue
            if any(a.name == "mega_knight" and a.tile_y >= 16.0 for a in leval.ally_units):
                continue

            min_req = self.cfg.mk_min_stomp_elixir_value
            is_heavy_beatdown_push = (
                leval.primary_tank is not None
                and leval.primary_tank.name in ("golem", "giant", "goblin_giant", "royal_giant", "unknown_tank")
                and leval.primary_tank.tile_y >= 17.8
                and (leval.air_threat_our_side >= 2.5 or state.elixir >= 8.2)
            )
            if (
                stomp.elixir_value < min_req
                and not (stomp.has_win_condition and stomp.elixir_value >= 3.2)
                and not is_heavy_beatdown_push
            ):
                continue

            evo_bonus = 6.5 if mk_slot.is_evolved else 0.0
            wincon_bonus = 8.0 if stomp.has_win_condition else 0.0
            counter_out_bonus = 5.5 if state.opponent.mk_counter_out_of_cycle else 0.0
            score = min(98.0, 74.5 + (stomp.elixir_value * 3.5) + evo_bonus + wincon_bonus + counter_out_bonus)

            out.append(
                BotAction(
                    card_name="mega_knight",
                    slot_index=mk_slot.slot_index,
                    tile_x=stomp.tile_x,
                    tile_y=stomp.tile_y,
                    score=score,
                    reason=(
                        f"{'Evo ' if mk_slot.is_evolved else ''}Mega Knight intercept crush on {lane.value} "
                        f"({stomp.unit_count} unit(s), ~{stomp.elixir_value:.1f}e stomp)"
                    ),
                )
            )

    def _score_tank_melter_and_kiting(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        slot_map = {s.card_name: s for s in ready}

        for lane in (Lane.LEFT, Lane.RIGHT):
            leval = board.lanes[lane]

            # 1. Flanking Inferno Dragon vs Heavy Tanks (P.E.K.K.A, Golem, Giant, RG, Lava, Enemy MK)
            tank = leval.primary_tank
            if tank is not None and tank.tile_y >= 16.2 and "inferno_dragon" in slot_map:
                already_have_idrag = any(a.name == "inferno_dragon" for a in leval.ally_units)
                if not already_have_idrag:
                    # Place deeper (y=22.8) when enemy air support (Mega Minion / Baby Dragon) trails the tank,
                    # so Inferno Dragon locks onto the front tank without flying into backline air killers!
                    has_air_escort = any(
                        e.move_type == MoveType.AIR and e.name != tank.name for e in leval.enemy_units
                    )
                    supp = leval.enemy_resetter_or_support
                    if supp is not None and abs(supp.tile_x - 9.0) > 4.0:
                        ix, iy = (
                            self.grid.IDRAG_CENTER_LEFT if lane == Lane.LEFT else self.grid.IDRAG_CENTER_RIGHT
                        )
                    else:
                        ix, iy = (
                            self.grid.IDRAG_OUTER_LEFT if lane == Lane.LEFT else self.grid.IDRAG_OUTER_RIGHT
                        )
                    if has_air_escort:
                        iy = 23.0
                    out.append(
                        BotAction(
                            card_name="inferno_dragon",
                            slot_index=slot_map["inferno_dragon"].slot_index,
                            tile_x=ix,
                            tile_y=iy,
                            score=92.5 if tank.name in ("pekka", "golem") else 89.0,
                            reason=f"Safe-flank Inferno Dragon melts enemy {tank.name} ({lane.value})",
                        )
                    )

            # 2. Isolate & Snipe Enemy Electro Wizard / Splash Support behind a Tank with Miner!
            supp = leval.enemy_resetter_or_support
            if supp is not None and "miner" in slot_map and (tank is not None or leval.urgency_score >= 28.0):
                already_on_supp = any(
                    a.name == "miner" and math.hypot(a.tile_x - supp.tile_x, a.tile_y - supp.tile_y) <= 2.8
                    for a in leval.ally_units
                )
                if not already_on_supp:
                    is_ewiz = supp.name == "electro_wizard"
                    out.append(
                        BotAction(
                            card_name="miner",
                            slot_index=slot_map["miner"].slot_index,
                            tile_x=supp.tile_x,
                            tile_y=supp.tile_y,
                            score=90.5 if (is_ewiz and tank is not None) else 81.5,
                            reason=f"Miner isolates & snipes enemy {supp.name} behind push",
                        )
                    )

            # 3. Center-Pull Kiting vs P.E.K.K.A / Prince / Mini P.E.K.K.A / Lumberjack / Knight
            kite_unit = leval.kite_target
            if kite_unit is not None and leval.unanswered_threat >= 1.2:
                kx, ky = (
                    self.grid.KITE_LEFT_THREAT_TO_CENTER
                    if lane == Lane.LEFT
                    else self.grid.KITE_RIGHT_THREAT_TO_CENTER
                )
                k_info = get_card_info(kite_unit.name)
                if "spear_goblins" in slot_map:
                    out.append(
                        BotAction(
                            card_name="spear_goblins",
                            slot_index=slot_map["spear_goblins"].slot_index,
                            tile_x=kx,
                            tile_y=ky,
                            score=87.0 if kite_unit.name == "pekka" else 81.5,
                            reason=f"Center-pull kite {kite_unit.name} into dual-tower fire with Spear Goblins",
                        )
                    )
                if "goblin_gang" in slot_map and not k_info.is_splash and not leval.has_splash_anti_swarm:
                    out.append(
                        BotAction(
                            card_name="goblin_gang",
                            slot_index=slot_map["goblin_gang"].slot_index,
                            tile_x=kx,
                            tile_y=ky + 0.6,
                            score=85.5 if kite_unit.name in ("pekka", "prince", "mini_pekka") else 79.0,
                            reason=f"Center-pull surround {kite_unit.name} with Goblin Gang",
                        )
                    )

    def _score_zap_tactics(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        zap_slot = next((s for s in ready if s.card_name == "zap"), None)
        if zap_slot is None or board.best_zap is None:
            return

        if (state.timestamp - self._last_zap_reset_ts) < 1.6:
            return

        z = board.best_zap
        score = z.value_score

        # Hold Zap against Log Bait / Inferno decks unless this is an actual Barrel defense or Inferno reset!
        if not z.is_reset and (state.opponent.has_goblin_barrel or state.opponent.has_inferno_or_sparky):
            if "goblin_barrel" not in z.reason and score < 80.0:
                score -= 18.0

        if zap_slot.is_evolved:
            score += 5.0

        out.append(
            BotAction(
                card_name="zap",
                slot_index=zap_slot.slot_index,
                tile_x=z.tile_x,
                tile_y=z.tile_y,
                score=score,
                reason=z.reason,
            )
        )

    def _score_counterpush_conversions(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        slot_map = {s.card_name: s for s in ready}

        for lane in (Lane.LEFT, Lane.RIGHT):
            leval = board.lanes[lane]
            other_lane = Lane.RIGHT if lane == Lane.LEFT else Lane.LEFT
            if board.lanes[other_lane].urgency_score >= 48.0:
                continue

            # Special Pro Play: Distract enemy Inferno Tower locking onto our Mega Knight!
            itower = leval.enemy_inferno_vs_mk
            if itower is not None:
                if "bats" in slot_map:
                    out.append(
                        BotAction(
                            card_name="bats",
                            slot_index=slot_map["bats"].slot_index,
                            tile_x=8.8 if lane == Lane.LEFT else 9.2,
                            tile_y=16.9,
                            score=88.5,
                            reason="River-center Bats distract enemy Inferno Tower for Mega Knight!",
                        )
                    )
                if "miner" in slot_map:
                    out.append(
                        BotAction(
                            card_name="miner",
                            slot_index=slot_map["miner"].slot_index,
                            tile_x=itower.tile_x,
                            tile_y=itower.tile_y,
                            score=87.5,
                            reason="Miner snipes & tanks enemy Inferno Tower for Mega Knight!",
                        )
                    )

            lead = leval.counterpush_lead
            if lead is None:
                continue

            # If lead is Skeleton Barrel, wait until it actually nears the tower (y <= 11.8) before sending Miner!
            if lead.name == "skeleton_barrel" and lead.tile_y > 11.8:
                continue

            enemy_king = state.towers.get("enemy_king")
            king_awake = bool(enemy_king and enemy_king.king_activated)

            if "miner" in slot_map and not leval.ally_miner_on_enemy_tower:
                mx, my = self.grid.pick_safe_miner_spot(lane, enemy_king_activated=king_awake, rng=self._rng)
                bait_bonus = 7.5 if state.opponent.small_spell_baited else 0.0
                out.append(
                    BotAction(
                        card_name="miner",
                        slot_index=slot_map["miner"].slot_index,
                        tile_x=mx,
                        tile_y=my,
                        score=83.0 + bait_bonus,
                        reason=f"Counterpush Miner tanks {lane.value} tower for surviving {lead.name}",
                    )
                )

            if lead.name == "mega_knight" and state.elixir >= 4.5:
                bx, by = self.grid.LEFT_BRIDGE if lane == Lane.LEFT else self.grid.RIGHT_BRIDGE
                if "bats" in slot_map:
                    out.append(
                        BotAction(
                            card_name="bats",
                            slot_index=slot_map["bats"].slot_index,
                            tile_x=lead.tile_x,
                            tile_y=max(17.2, lead.tile_y + 1.5),
                            score=80.0 + (6.0 if state.opponent.small_spell_baited else 0.0),
                            reason=f"Bats support counterpushing Mega Knight ({lane.value})",
                        )
                    )
                if "skeleton_barrel" in slot_map and leval.enemy_active_building is None:
                    out.append(
                        BotAction(
                            card_name="skeleton_barrel",
                            slot_index=slot_map["skeleton_barrel"].slot_index,
                            tile_x=bx,
                            tile_y=by,
                            score=78.5,
                            reason=f"Skeleton Barrel pressures with Mega Knight ({lane.value})",
                        )
                    )

    def _score_opposite_lane_punish(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        slot_map = {s.card_name: s for s in ready}
        if state.elixir < 5.0:
            return

        for tank_lane in (Lane.LEFT, Lane.RIGHT):
            heavy_back = board.lanes[tank_lane].enemy_heavy_backline
            if heavy_back is None:
                continue
            punish_lane = Lane.RIGHT if tank_lane == Lane.LEFT else Lane.LEFT
            if board.lanes[punish_lane].urgency_score >= 30.0:
                continue
            # Don't repeat opposite-lane punish if we already have units pushing that lane!
            if any(a.tile_y <= 18.0 for a in board.lanes[punish_lane].ally_units):
                continue

            bx, by = self.grid.LEFT_BRIDGE if punish_lane == Lane.LEFT else self.grid.RIGHT_BRIDGE
            enemy_king = state.towers.get("enemy_king")
            king_awake = bool(enemy_king and enemy_king.king_activated)

            if "skeleton_barrel" in slot_map:
                followup = None
                if state.find_in_hand("miner") is not None and state.elixir >= 5.5:
                    mx, my = self.grid.pick_safe_miner_spot(
                        punish_lane, enemy_king_activated=king_awake, rng=self._rng
                    )
                    followup = QueuedAction(
                        card_name="miner",
                        tile_x=mx,
                        tile_y=my,
                        execute_at_ts=state.timestamp + 1.80,
                        max_expire_ts=state.timestamp + 3.80,
                        reason=f"Tank-swap Miner before Skeleton Barrel pops ({punish_lane.value})",
                        required_min_elixir=3.0,
                    )
                out.append(
                    BotAction(
                        card_name="skeleton_barrel",
                        slot_index=slot_map["skeleton_barrel"].slot_index,
                        tile_x=bx,
                        tile_y=by,
                        score=85.0,
                        reason=f"Opposite-lane punish ({punish_lane.value}) vs backline {heavy_back.name}",
                        followup=followup,
                    )
                )
            elif "miner" in slot_map and "bats" in slot_map and state.elixir >= 5.2:
                mx, my = self.grid.pick_safe_miner_spot(
                    punish_lane, enemy_king_activated=king_awake, rng=self._rng
                )
                followup = QueuedAction(
                    card_name="bats",
                    tile_x=bx,
                    tile_y=by,
                    execute_at_ts=state.timestamp + 0.55,
                    max_expire_ts=state.timestamp + 2.0,
                    reason=f"Fast Bats behind punish Miner ({punish_lane.value})",
                    required_min_elixir=2.0,
                )
                out.append(
                    BotAction(
                        card_name="miner",
                        slot_index=slot_map["miner"].slot_index,
                        tile_x=mx,
                        tile_y=my,
                        score=83.0,
                        reason=f"Opposite-lane Miner+Bats punish vs backline {heavy_back.name}",
                        followup=followup,
                    )
                )

    def _score_bait_offense_and_combos(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        max_urg = max(board.lanes[Lane.LEFT].urgency_score, board.lanes[Lane.RIGHT].urgency_score)
        if max_urg >= 32.0:
            return

        slot_map = {s.card_name: s for s in ready}
        target_lane = board.weaker_enemy_lane
        active_bldg = board.lanes[target_lane].enemy_active_building

        bx, by = self.grid.LEFT_BRIDGE if target_lane == Lane.LEFT else self.grid.RIGHT_BRIDGE
        enemy_king = state.towers.get("enemy_king")
        king_awake = bool(enemy_king and enemy_king.king_activated)

        elixir_lead = state.elixir - state.opponent.estimated_elixir
        baited = state.opponent.small_spell_baited

        # Combo 1: Skeleton Barrel + Timed Miner Tank-Swap (ONLY if no active Cannon/Inferno is waiting!)
        if "skeleton_barrel" in slot_map and state.find_in_hand("miner") is not None and active_bldg is None:
            if state.elixir >= 6.8 or (baited and state.elixir >= 5.6):
                mx, my = self.grid.pick_safe_miner_spot(
                    target_lane, enemy_king_activated=king_awake, rng=self._rng
                )
                followup = QueuedAction(
                    card_name="miner",
                    tile_x=mx,
                    tile_y=my,
                    execute_at_ts=state.timestamp + 1.85,
                    max_expire_ts=state.timestamp + 3.90,
                    reason=f"Miner tanks {target_lane.value} tower right before Skeleton Barrel pops",
                    required_min_elixir=3.0,
                )
                score = 77.0 + (8.5 if baited else 0.0) + max(0.0, elixir_lead * 1.5)
                out.append(
                    BotAction(
                        card_name="skeleton_barrel",
                        slot_index=slot_map["skeleton_barrel"].slot_index,
                        tile_x=bx,
                        tile_y=by,
                        score=score,
                        reason=(
                            f"Launch Skeleton Barrel + Miner combo ({target_lane.value})"
                            + (" [OPPONENT SPELL BAITED!]" if baited else "")
                        ),
                        followup=followup,
                    )
                )

        # Combo 2: Miner + Bats punish when small spell is BAITED (ignores Cannon if Miner goes outer corner!)
        if baited and "miner" in slot_map and "bats" in slot_map and state.elixir >= 5.5:
            mx, my = self.grid.pick_safe_miner_spot(
                target_lane, enemy_king_activated=king_awake, rng=self._rng
            )
            followup = QueuedAction(
                card_name="bats",
                tile_x=bx,
                tile_y=by,
                execute_at_ts=state.timestamp + 0.60,
                max_expire_ts=state.timestamp + 2.20,
                reason=f"Bats rush behind Miner while opponent spell is out of cycle ({target_lane.value})",
                required_min_elixir=2.0,
            )
            out.append(
                BotAction(
                    card_name="miner",
                    slot_index=slot_map["miner"].slot_index,
                    tile_x=mx,
                    tile_y=my,
                    score=82.0,
                    reason=f"Miner + Bats punish ({target_lane.value}) [OPPONENT SPELL BAITED!]",
                    followup=followup,
                )
            )

        # Solo Skeleton Barrel pressure to bait out opponent's Log/Zap (only when no active enemy building!)
        if "skeleton_barrel" in slot_map and state.elixir >= 7.8 and active_bldg is None:
            out.append(
                BotAction(
                    card_name="skeleton_barrel",
                    slot_index=slot_map["skeleton_barrel"].slot_index,
                    tile_x=bx,
                    tile_y=by,
                    score=70.0 + (6.0 if baited else 0.0),
                    reason=f"Skeleton Barrel bridge pressure to bait small spell ({target_lane.value})",
                )
            )

        # Pocket attack when one enemy Princess Tower is already down!
        enemy_left = state.towers.get("enemy_left")
        enemy_right = state.towers.get("enemy_right")
        if enemy_left and not enemy_left.alive and enemy_right and enemy_right.alive:
            if "goblin_gang" in slot_map and baited and state.elixir >= 6.0:
                px, py = self.grid.LEFT_POCKET
                out.append(
                    BotAction(
                        card_name="goblin_gang",
                        slot_index=slot_map["goblin_gang"].slot_index,
                        tile_x=px,
                        tile_y=py,
                        score=78.5,
                        reason="Pocket Goblin Gang pressure on remaining right tower [SPELL BAITED]",
                    )
                )
        elif enemy_right and not enemy_right.alive and enemy_left and enemy_left.alive:
            if "goblin_gang" in slot_map and baited and state.elixir >= 6.0:
                px, py = self.grid.RIGHT_POCKET
                out.append(
                    BotAction(
                        card_name="goblin_gang",
                        slot_index=slot_map["goblin_gang"].slot_index,
                        tile_x=px,
                        tile_y=py,
                        score=78.5,
                        reason="Pocket Goblin Gang pressure on remaining left tower [SPELL BAITED]",
                    )
                )

    def _score_cycle_and_overtime_finish(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        slot_map = {s.card_name: s for s in ready}
        target_lane = board.weaker_enemy_lane
        enemy_king = state.towers.get("enemy_king")
        king_awake = bool(enemy_king and enemy_king.king_activated)
        max_urg = max(board.lanes[Lane.LEFT].urgency_score, board.lanes[Lane.RIGHT].urgency_score)

        # 1. Low-HP Tower Finisher (Miner + Zap relentless closing!)
        if board.lowest_enemy_tower_hp <= self.cfg.overtime_spell_finish_hp and max_urg < 60.0:
            tw_key = "enemy_left" if target_lane == Lane.LEFT else "enemy_right"
            tw_x, tw_y = self.grid.TOWER_TILES[tw_key]

            if "miner" in slot_map and not board.lanes[target_lane].ally_miner_on_enemy_tower:
                mx, my = self.grid.pick_safe_miner_spot(
                    target_lane, enemy_king_activated=king_awake, rng=self._rng
                )
                out.append(
                    BotAction(
                        card_name="miner",
                        slot_index=slot_map["miner"].slot_index,
                        tile_x=mx,
                        tile_y=my,
                        score=89.5,
                        reason=f"Tower finisher Miner ({target_lane.value} tower at ~{board.lowest_enemy_tower_hp} HP)",
                    )
                )
            if "zap" in slot_map and (
                board.lowest_enemy_tower_hp <= 65
                or board.lanes[target_lane].ally_miner_on_enemy_tower
            ):
                out.append(
                    BotAction(
                        card_name="zap",
                        slot_index=slot_map["zap"].slot_index,
                        tile_x=tw_x,
                        tile_y=tw_y,
                        score=91.0 if board.lowest_enemy_tower_hp <= 65 else 79.0,
                        reason=f"Tower finisher Zap chip ({target_lane.value} tower at ~{board.lowest_enemy_tower_hp} HP)",
                    )
                )

        # 2. Anti-Elixir-Leak Passive Cycle (elixir >= 9.3, calm board)
        if state.elixir >= self.cfg.leak_prevention_elixir and max_urg < 25.0:
            # If an enemy heavy Beatdown tank is marching up a lane in 2x/3x elixir, slow-roll Mega Knight
            # behind our King in that lane so MK meets the support push at the river!
            for t_lane in (Lane.LEFT, Lane.RIGHT):
                app_tank = board.lanes[t_lane].enemy_approaching_tank
                if (
                    app_tank is not None
                    and app_tank.name != "pekka"
                    and "mega_knight" in slot_map
                    and state.elixir >= 9.5
                ):
                    if not any(a.name == "mega_knight" for a in board.lanes[t_lane].ally_units):
                        bx = 6.5 if t_lane == Lane.LEFT else 11.5
                        out.append(
                            BotAction(
                                card_name="mega_knight",
                                slot_index=slot_map["mega_knight"].slot_index,
                                tile_x=bx,
                                tile_y=29.5,
                                score=75.0,
                                reason=f"Stage Mega Knight in back vs approaching {app_tank.name} ({t_lane.value})",
                            )
                        )

            sx, sy = self.grid.SPLIT_BACK
            if "spear_goblins" in slot_map:
                out.append(
                    BotAction(
                        card_name="spear_goblins",
                        slot_index=slot_map["spear_goblins"].slot_index,
                        tile_x=sx,
                        tile_y=sy,
                        score=68.5,
                        reason="Anti-leak: Split Spear Goblins behind King Tower",
                    )
                )
            if "miner" in slot_map:
                mx, my = self.grid.pick_safe_miner_spot(
                    target_lane, enemy_king_activated=king_awake, rng=self._rng
                )
                out.append(
                    BotAction(
                        card_name="miner",
                        slot_index=slot_map["miner"].slot_index,
                        tile_x=mx,
                        tile_y=my,
                        score=68.0,
                        reason=f"Anti-leak: Safe Anti-Tornado Miner chip ({target_lane.value})",
                    )
                )
            if "bats" in slot_map:
                out.append(
                    BotAction(
                        card_name="bats",
                        slot_index=slot_map["bats"].slot_index,
                        tile_x=sx,
                        tile_y=sy - 1.0,
                        score=66.5,
                        reason="Anti-leak: Split Bats in back",
                    )
                )
            if "goblin_gang" in slot_map:
                out.append(
                    BotAction(
                        card_name="goblin_gang",
                        slot_index=slot_map["goblin_gang"].slot_index,
                        tile_x=sx,
                        tile_y=sy,
                        score=65.0,
                        reason="Anti-leak: Split Goblin Gang behind King Tower",
                    )
                )
            if (
                "mega_knight" in slot_map
                and state.phase in (GamePhase.BATTLE_2X, GamePhase.BATTLE_3X)
                and state.elixir >= 9.7
            ):
                back_x = 8.2 if target_lane == Lane.LEFT else 9.8
                out.append(
                    BotAction(
                        card_name="mega_knight",
                        slot_index=slot_map["mega_knight"].slot_index,
                        tile_x=back_x,
                        tile_y=30.4,
                        score=61.5,
                        reason=f"2x/3x Elixir slow-roll Mega Knight behind King ({target_lane.value})",
                    )
                )

    def _score_anti_deadlock_fallback(
        self,
        state: BattleState,
        board: BoardAssessment,
        ready: List[HandSlot],
        out: List[BotAction],
    ) -> None:
        """Guarantees the bot never sits frozen at 10 elixir during unusual 4-card hand states."""
        slot_map = {s.card_name: s for s in ready}
        hot_lane = (
            Lane.LEFT
            if board.lanes[Lane.LEFT].urgency_score >= board.lanes[Lane.RIGHT].urgency_score
            else Lane.RIGHT
        )
        enemies = board.lanes[hot_lane].enemy_units

        for card_pref in ("spear_goblins", "bats", "goblin_gang", "inferno_dragon", "miner", "mega_knight", "skeleton_barrel", "zap"):
            if card_pref not in slot_map:
                continue
            slot = slot_map[card_pref]
            if card_pref == "miner":
                if enemies:
                    e0 = enemies[0]
                    tx, ty = e0.tile_x, max(17.2, e0.tile_y - 1.2)
                else:
                    tx, ty = self.grid.pick_safe_miner_spot(board.weaker_enemy_lane, rng=self._rng)
            elif card_pref == "zap":
                if not enemies:
                    continue
                tx, ty = enemies[0].tile_x, enemies[0].tile_y
            elif card_pref == "skeleton_barrel":
                tx, ty = (
                    self.grid.LEFT_BRIDGE if board.weaker_enemy_lane == Lane.LEFT else self.grid.RIGHT_BRIDGE
                )
            else:
                tx = 7.8 if hot_lane == Lane.LEFT else 10.2
                ty = 24.5 if enemies else 29.5
            out.append(
                BotAction(
                    card_name=card_pref,
                    slot_index=slot.slot_index,
                    tile_x=tx,
                    tile_y=ty,
                    score=56.0,
                    reason=f"Anti-deadlock cycle/defense ({card_pref}) at 9.5+ elixir",
                )
            )
            break

    def _clamp_for_card(
        self,
        card_name: str,
        tile_x: float,
        tile_y: float,
        state: BattleState,
    ) -> Tuple[float, float]:
        info = get_card_info(card_name)
        return self.grid.clamp_deploy_tile(
            tile_x,
            tile_y,
            deploy_anywhere=info.deploy_anywhere,
            towers=state.towers,
        )
