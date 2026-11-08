"""
Competent Meta Archetype Opponent AIs (Hog 2.6, Log Bait, P.E.K.K.A Bridge Spam, Golem Beatdown)
plus Baseline Comparison Bots (RandomBot & NaiveBridgeSpamBot).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from cr_bot.config.cards_db import get_card_info
from cr_bot.models import Lane, MoveType, Side
from cr_bot.sim.engine import ClashRoyaleSimEngine, SimEntity


META_DECKS: Dict[str, List[str]] = {
    "hog_26": [
        "hog_rider",
        "musketeer",
        "cannon",
        "ice_golem",
        "skeletons",
        "ice_spirit",
        "fireball",
        "the_log",
    ],
    "log_bait": [
        "goblin_barrel",
        "princess",
        "knight",
        "goblin_gang",
        "inferno_tower",
        "ice_spirit",
        "rocket",
        "the_log",
    ],
    "pekka_bridge_spam": [
        "pekka",
        "battle_ram",
        "bandit",
        "royal_ghost",
        "electro_wizard",
        "magic_archer",
        "zap",
        "fireball",
    ],
    "golem_beatdown": [
        "golem",
        "baby_dragon",
        "mega_minion",
        "mini_pekka",
        "lumberjack",
        "zap",
        "fireball",
        "barbarian_barrel",
    ],
}


class MetaOpponentAI:
    """
    Archetype-specific opponent AI playing from the TOP side (Side.ENEMY, y in [0..15.1]).
    Obeys strict Fog of War (only sees deployed arena entities & tower HP, never peeks at ALLY hand/elixir).
    """

    def __init__(self, archetype: str, seed: int = 100) -> None:
        if archetype not in META_DECKS:
            raise ValueError(f"Unknown opponent archetype: {archetype}")
        self.archetype = archetype
        self.deck = list(META_DECKS[archetype])
        self.rng = random.Random(seed)
        self._last_play_ts: float = -99.0

    def step_ai(self, sim: ClashRoyaleSimEngine) -> None:
        if (sim.time - self._last_play_ts) < 0.75:
            return

        hand = sim.enemy.hand
        elixir = sim.enemy.elixir

        # Visible player (ALLY) units on board
        player_units = [e for e in sim.entities if e.side == Side.ALLY and e.hp > 0]
        my_units = [e for e in sim.entities if e.side == Side.ENEMY and e.hp > 0]

        # Target weaker ALLY tower
        al_tw = sim.towers["ally_left"]
        ar_tw = sim.towers["ally_right"]
        atk_lane = Lane.LEFT if (al_tw.alive and (not ar_tw.alive or al_tw.hp <= ar_tw.hp)) else Lane.RIGHT
        atk_bx = 3.5 if atk_lane == Lane.LEFT else 14.5
        atk_tw = al_tw if atk_lane == Lane.LEFT else ar_tw

        action = None
        if self.archetype == "hog_26":
            action = self._play_hog_26(sim, hand, elixir, player_units, my_units, atk_bx, atk_tw)
        elif self.archetype == "log_bait":
            action = self._play_log_bait(sim, hand, elixir, player_units, my_units, atk_bx, atk_tw)
        elif self.archetype == "pekka_bridge_spam":
            action = self._play_pekka_bs(sim, hand, elixir, player_units, my_units, atk_bx, atk_tw)
        elif self.archetype == "golem_beatdown":
            action = self._play_golem(sim, hand, elixir, player_units, my_units, atk_bx, atk_tw)

        if action is not None:
            card, tx, ty, why = action
            if sim.play_card(Side.ENEMY, card, tx, ty, reason=why):
                self._last_play_ts = sim.time

    # --- 1. Hog 2.6 Cycle AI ---
    def _play_hog_26(
        self, sim: ClashRoyaleSimEngine, hand: List[str], elixir: float,
        p_units: List[SimEntity], my_units: List[SimEntity], atk_bx: float, atk_tw
    ) -> Optional[Tuple[str, float, float, str]]:
        # Spell-finish low tower
        if atk_tw.alive and atk_tw.hp <= 255 and "fireball" in hand and elixir >= 4:
            return ("fireball", atk_tw.x, atk_tw.y, "2.6 Fireball tower finish")

        # Catch Miner on our tower with Skeletons / Ice Golem / Musketeer
        miner_on_tw = next((u for u in p_units if u.card_slug == "miner" and u.y <= 9.5), None)
        if miner_on_tw is not None:
            already_def = any(math.hypot(m.x - miner_on_tw.x, m.y - miner_on_tw.y) <= 2.5 for m in my_units)
            if not already_def:
                if "skeletons" in hand and elixir >= 1:
                    return ("skeletons", miner_on_tw.x, miner_on_tw.y, "2.6 Skeletons catch Miner")
                if "ice_spirit" in hand and elixir >= 1:
                    return ("ice_spirit", miner_on_tw.x, miner_on_tw.y + 1.0, "2.6 Ice Spirit freeze Miner")

        # Log ground swarms (Goblin Gang, Spear Gobs, popped Skelly Barrel skeletons) on our half
        ground_swarms = [
            u for u in p_units if u.arch.move_type == MoveType.GROUND and u.max_hp <= 250 and u.y <= 16.5
        ]
        if len(ground_swarms) >= 3 and "the_log" in hand and elixir >= 2:
            s0 = ground_swarms[0]
            return ("the_log", s0.x, max(2.0, s0.y - 3.5), "2.6 Log swarm clear")

        # Defend Mega Knight / Skeleton Barrel crossing river (y <= 16.8)
        threats = [u for u in p_units if u.y <= 16.8]
        if threats:
            lead = min(threats, key=lambda u: u.y)
            has_cannon = any(m.card_slug == "cannon" for m in my_units)
            if not has_cannon and "cannon" in hand and elixir >= 3 and lead.card_slug in ("mega_knight", "skeleton_barrel", "miner"):
                cx = 8.4 if lead.x < 9.0 else 9.6
                return ("cannon", cx, 11.2, "2.6 center-pull Cannon")

            if lead.card_slug == "mega_knight" and lead.y <= 15.5:
                # Kite MK with Ice Golem to center-opposite lane!
                if "ice_golem" in hand and elixir >= 2 and not any(m.card_slug == "ice_golem" for m in my_units):
                    kx = 9.7 if lead.x < 9.0 else 8.3
                    return ("ice_golem", kx, 11.5, "2.6 Ice Golem center kite vs MK")
                if "musketeer" in hand and elixir >= 4 and not any(m.card_slug == "musketeer" for m in my_units):
                    mx = 1.8 if lead.x < 9.0 else 16.2
                    return ("musketeer", mx, 9.5, "2.6 outer Musketeer DPS")

            if lead.arch.move_type == MoveType.AIR and "musketeer" in hand and elixir >= 4:
                if not any(m.card_slug == "musketeer" for m in my_units):
                    mx = 2.5 if lead.x < 9.0 else 15.5
                    return ("musketeer", mx, 9.8, "2.6 Musketeer anti-air")
            if lead.arch.move_type == MoveType.AIR and "ice_spirit" in hand and elixir >= 1:
                return ("ice_spirit", lead.x, max(5.0, lead.y - 2.0), "2.6 Ice Spirit splash air")

        # Offense: Hog Rider at bridge!
        if "hog_rider" in hand and elixir >= 5.5 and len(threats) == 0:
            return ("hog_rider", atk_bx, 15.0, "2.6 Hog Rider bridge rush")
        if any(m.card_slug == "hog_rider" and m.y >= 14.5 for m in my_units) and "ice_spirit" in hand and elixir >= 1.5:
            return ("ice_spirit", atk_bx, 15.0, "2.6 Ice Spirit behind Hog")

        # Cycle at high elixir
        if elixir >= 9.2:
            for cyc in ("skeletons", "ice_spirit", "ice_golem", "the_log"):
                if cyc in hand:
                    if cyc == "the_log":
                        return ("the_log", atk_bx, 14.5, "2.6 Log chip cycle")
                    return (cyc, 9.0, 2.5, f"2.6 cycle {cyc} in back")
            if "musketeer" in hand:
                return ("musketeer", atk_bx, 3.5, "2.6 Musketeer slow-roll")
        return None

    # --- 2. Classic Log Bait AI ---
    def _play_log_bait(
        self, sim: ClashRoyaleSimEngine, hand: List[str], elixir: float,
        p_units: List[SimEntity], my_units: List[SimEntity], atk_bx: float, atk_tw
    ) -> Optional[Tuple[str, float, float, str]]:
        if atk_tw.alive and atk_tw.hp <= 370 and "rocket" in hand and elixir >= 6:
            return ("rocket", atk_tw.x, atk_tw.y, "LogBait Rocket tower finish")

        # Catch Miner with Knight or Goblin Gang
        miner_on_tw = next((u for u in p_units if u.card_slug == "miner" and u.y <= 9.5), None)
        if miner_on_tw is not None:
            if not any(math.hypot(m.x - miner_on_tw.x, m.y - miner_on_tw.y) <= 2.5 for m in my_units):
                if "knight" in hand and elixir >= 3:
                    return ("knight", miner_on_tw.x, miner_on_tw.y, "LogBait Knight catches Miner")
                if "goblin_gang" in hand and elixir >= 3:
                    return ("goblin_gang", miner_on_tw.x, miner_on_tw.y, "LogBait Gang catches Miner")

        ground_swarms = [
            u for u in p_units if u.arch.move_type == MoveType.GROUND and u.max_hp <= 250 and u.y <= 16.5
        ]
        if len(ground_swarms) >= 3 and "the_log" in hand and elixir >= 2:
            s0 = ground_swarms[0]
            return ("the_log", s0.x, max(2.0, s0.y - 3.5), "LogBait Log clear")

        threats = [u for u in p_units if u.y <= 16.8]
        if threats:
            lead = min(threats, key=lambda u: u.y)
            if lead.card_slug == "mega_knight" and "inferno_tower" in hand and elixir >= 5:
                if not any(m.card_slug == "inferno_tower" for m in my_units):
                    return ("inferno_tower", 8.8 if lead.x < 9.0 else 9.2, 10.8, "LogBait Inferno Tower vs MK")
            if lead.arch.move_type == MoveType.AIR and "princess" in hand and elixir >= 3:
                if not any(m.card_slug == "princess" for m in my_units):
                    px = 13.5 if lead.x < 9.0 else 4.5
                    return ("princess", px, 9.5, "LogBait opposite-lane Princess splash")
            if "knight" in hand and elixir >= 3 and lead.arch.move_type == MoveType.GROUND and not any(m.card_slug == "knight" for m in my_units):
                return ("knight", lead.x, max(6.5, lead.y - 2.2), "LogBait Knight block")
            if "goblin_gang" in hand and elixir >= 3 and lead.card_slug != "mega_knight":
                return ("goblin_gang", lead.x, max(6.5, lead.y - 2.0), "LogBait Gang DPS")

        # Offense: Goblin Barrel on weaker tower + Princess at bridge!
        if "goblin_barrel" in hand and elixir >= 5.5:
            off_y = self.rng.choice([0.0, -0.9])  # Mix center & deep tricky barrel
            return ("goblin_barrel", atk_tw.x, atk_tw.y + off_y, "LogBait Goblin Barrel attack")
        if "princess" in hand and elixir >= 7.5 and len(threats) == 0:
            return ("princess", atk_bx, 14.8, "LogBait bridge Princess chip")

        if elixir >= 9.2:
            for cyc in ("ice_spirit", "knight", "goblin_gang", "princess"):
                if cyc in hand:
                    return (cyc, 9.0, 2.5, f"LogBait cycle {cyc}")
        return None

    # --- 3. P.E.K.K.A Bridge Spam AI ---
    def _play_pekka_bs(
        self, sim: ClashRoyaleSimEngine, hand: List[str], elixir: float,
        p_units: List[SimEntity], my_units: List[SimEntity], atk_bx: float, atk_tw
    ) -> Optional[Tuple[str, float, float, str]]:
        if atk_tw.alive and atk_tw.hp <= 220 and "fireball" in hand and elixir >= 4:
            return ("fireball", atk_tw.x, atk_tw.y, "PBS Fireball finish")

        # Zap player Bats / Skeleton Barrel / Inferno Dragon!
        zap_targets = [
            u for u in p_units
            if (u.card_slug in ("bats", "inferno_dragon", "skeleton_barrel") or u.max_hp <= 150)
            and (u.y <= 15.5 or any(math.hypot(m.x - u.x, m.y - u.y) <= 4.0 for m in my_units))
        ]
        if zap_targets and "zap" in hand and elixir >= 2:
            zt = zap_targets[0]
            return ("zap", zt.x, zt.y, f"PBS Zap on {zt.card_slug}")

        # Drop P.E.K.K.A directly in front of player Mega Knight!
        player_mk = next((u for u in p_units if u.card_slug == "mega_knight" and u.y <= 18.0), None)
        if player_mk is not None and "pekka" in hand and elixir >= 7:
            if not any(m.card_slug == "pekka" for m in my_units):
                return ("pekka", player_mk.x, min(14.5, max(7.0, player_mk.y - 2.8)), "PBS P.E.K.K.A hard-counters MK!")

        # Support active P.E.K.K.A with Electro Wizard (stuns Inferno Dragon!) or Magic Archer!
        my_pekka = next((m for m in my_units if m.card_slug == "pekka"), None)
        if my_pekka is not None:
            if "electro_wizard" in hand and elixir >= 4 and not any(m.card_slug == "electro_wizard" for m in my_units):
                return ("electro_wizard", my_pekka.x, max(3.0, my_pekka.y - 2.5), "PBS E-Wiz behind P.E.K.K.A")
            if "magic_archer" in hand and elixir >= 4 and not any(m.card_slug == "magic_archer" for m in my_units):
                return ("magic_archer", my_pekka.x, max(3.0, my_pekka.y - 3.2), "PBS Magic Archer behind P.E.K.K.A")

        # Defend Miner / swarms / air on our half
        threats = [u for u in p_units if u.y <= 16.2]
        if threats:
            lead = min(threats, key=lambda u: u.y)
            if lead.arch.move_type == MoveType.AIR and "electro_wizard" in hand and elixir >= 4:
                return ("electro_wizard", lead.x, max(6.0, lead.y - 1.5), "PBS E-Wiz spawn-zap air")
            if lead.arch.move_type == MoveType.GROUND:
                if "royal_ghost" in hand and elixir >= 3:
                    return ("royal_ghost", lead.x, max(6.0, lead.y - 1.8), "PBS Ghost splash defense")
                if "bandit" in hand and elixir >= 3:
                    return ("bandit", lead.x, max(6.0, lead.y - 4.0), "PBS Bandit dash defense")

        # Bridge Spam pressure!
        if len(threats) == 0 and elixir >= 6.8:
            for rush in ("battle_ram", "bandit", "royal_ghost"):
                if rush in hand:
                    return (rush, atk_bx, 14.8, f"PBS bridge spam {rush}")

        if elixir >= 9.3:
            if "pekka" in hand and sim.time >= 120.0:
                return ("pekka", atk_bx, 3.0, "PBS 2x P.E.K.K.A in back")
            for cyc in ("bandit", "royal_ghost", "battle_ram", "magic_archer"):
                if cyc in hand:
                    return (cyc, atk_bx, 4.5, f"PBS cycle {cyc}")
        return None

    # --- 4. Golem Beatdown AI ---
    def _play_golem(
        self, sim: ClashRoyaleSimEngine, hand: List[str], elixir: float,
        p_units: List[SimEntity], my_units: List[SimEntity], atk_bx: float, atk_tw
    ) -> Optional[Tuple[str, float, float, str]]:
        if atk_tw.alive and atk_tw.hp <= 220 and "fireball" in hand and elixir >= 4:
            return ("fireball", atk_tw.x, atk_tw.y, "Golem Fireball finish")

        # Zap our Inferno Dragon when Golem is nearby!
        my_golem = next((m for m in my_units if m.card_slug == "golem"), None)
        idrag = next((u for u in p_units if u.card_slug == "inferno_dragon"), None)
        if my_golem is not None and idrag is not None and "zap" in hand and elixir >= 2:
            if math.hypot(my_golem.x - idrag.x, my_golem.y - idrag.y) <= 5.0:
                return ("zap", idrag.x, idrag.y, "Golem Zap resets Inferno Dragon!")

        # Stack Lumberjack + Baby Dragon + Mega Minion behind active Golem!
        if my_golem is not None and my_golem.y >= 8.5:
            for supp in ("baby_dragon", "lumberjack", "mega_minion"):
                if supp in hand and elixir >= get_card_info(supp).elixir:
                    if not any(m.card_slug == supp for m in my_units):
                        return (supp, my_golem.x, max(3.0, my_golem.y - 2.2), f"Golem stack {supp}")

        # Defend threats on our side
        threats = [u for u in p_units if u.y <= 16.0]
        if threats:
            lead = min(threats, key=lambda u: u.y)
            if lead.card_slug == "mega_knight" and "mini_pekka" in hand and elixir >= 4:
                return ("mini_pekka", lead.x, max(6.5, lead.y - 1.8), "Golem Mini P.E.K.K.A chops MK")
            if lead.arch.move_type == MoveType.GROUND and lead.max_hp <= 250 and "barbarian_barrel" in hand and elixir >= 2:
                return ("barbarian_barrel", lead.x, max(5.0, lead.y - 2.5), "Golem BarbBarrel swarm clear")
            if lead.arch.move_type == MoveType.AIR:
                if "baby_dragon" in hand and elixir >= 4:
                    return ("baby_dragon", lead.x, max(6.0, lead.y - 2.0), "Golem Baby Dragon anti-air")
                if "mega_minion" in hand and elixir >= 3:
                    return ("mega_minion", lead.x, max(6.0, lead.y - 1.5), "Golem Mega Minion anti-air")
            if "lumberjack" in hand and elixir >= 4 and lead.y <= 11.0:
                return ("lumberjack", lead.x, max(6.0, lead.y - 1.5), "Golem Lumberjack emergency DPS")

        # Start Golem Beatdown push in back when safe!
        if "golem" in hand and elixir >= (8.5 if sim.time >= 110.0 else 9.6) and len(threats) == 0:
            if my_golem is None:
                return ("golem", atk_bx, 2.8, "Golem slow-roll behind King")

        if elixir >= 9.4:
            for cyc in ("baby_dragon", "mega_minion", "barbarian_barrel", "lumberjack"):
                if cyc in hand:
                    return (cyc, atk_bx, 3.8, f"Golem cycle {cyc}")
        return None


# -----------------------------------------------------------------------------
# Baseline Bots Playing the Exact Same MK Zap Bait Deck (Side.ALLY)
# -----------------------------------------------------------------------------


class RandomBaselineBot:
    """Plays random affordable cards from hand at random legal tiles."""

    def __init__(self, seed: int = 1) -> None:
        self.rng = random.Random(seed)
        self._last_ts = -99.0

    def step_bot(self, sim: ClashRoyaleSimEngine) -> None:
        if (sim.time - self._last_ts) < 1.2:
            return
        affordable = [c for c in sim.ally.hand if sim.ally.elixir >= get_card_info(c).elixir]
        if not affordable:
            return
        card = self.rng.choice(affordable)
        info = get_card_info(card)
        tx = self.rng.uniform(1.5, 16.5)
        ty = self.rng.uniform(4.0, 28.0) if info.deploy_anywhere else self.rng.uniform(17.0, 30.0)
        if sim.play_card(Side.ALLY, card, tx, ty, reason="random"):
            self._last_ts = sim.time


class NaiveBridgeSpamBot:
    """Typical 'dumb bot': drops cards immediately at the bridge or on the nearest enemy without tactical math."""

    def __init__(self, seed: int = 1) -> None:
        self.rng = random.Random(seed)
        self._last_ts = -99.0

    def step_bot(self, sim: ClashRoyaleSimEngine) -> None:
        if (sim.time - self._last_ts) < 0.9:
            return
        affordable = [c for c in sim.ally.hand if sim.ally.elixir >= get_card_info(c).elixir]
        if not affordable:
            return
        card = affordable[0]
        enemies = [e for e in sim.entities if e.side == Side.ENEMY and e.hp > 0]
        if enemies:
            nearest = max(enemies, key=lambda e: e.y)
            tx = nearest.x
            ty = nearest.y if get_card_info(card).deploy_anywhere else max(17.0, nearest.y + 1.0)
        else:
            bx = self.rng.choice([3.5, 14.5])
            tx = bx
            ty = 6.5 if card == "miner" else 17.0
            if card == "zap":
                return  # Even a naive script usually doesn't zap empty grass
        if sim.play_card(Side.ALLY, card, tx, ty, reason="naive_bridge_spam"):
            self._last_ts = sim.time
