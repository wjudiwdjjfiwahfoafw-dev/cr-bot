"""Comprehensive Clash Royale card database with tactical attributes & alias resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from cr_bot.models import MoveType, WeightClass


@dataclass(frozen=True)
class CardInfo:
    name: str
    display_name: str
    elixir: int
    move_type: MoveType = MoveType.GROUND
    weight_class: WeightClass = WeightClass.MEDIUM
    is_spell: bool = False
    deploy_anywhere: bool = False
    targets_air: bool = False
    targets_buildings_only: bool = False
    is_splash: bool = False
    is_resettable: bool = False
    is_small_spell: bool = False
    stomp_vulnerable: bool = True


def _g(
    name: str,
    display: str,
    elixir: int,
    weight: WeightClass = WeightClass.MEDIUM,
    *,
    air: bool = False,
    spell: bool = False,
    anywhere: bool = False,
    hits_air: bool = False,
    wincon_bldg: bool = False,
    splash: bool = False,
    resettable: bool = False,
    small_spell: bool = False,
    stomp_vuln: bool = True,
) -> CardInfo:
    return CardInfo(
        name=name,
        display_name=display,
        elixir=elixir,
        move_type=MoveType.AIR if air else MoveType.GROUND,
        weight_class=WeightClass.SPELL_PROJECTILE if spell else weight,
        is_spell=spell,
        deploy_anywhere=anywhere or spell,
        targets_air=hits_air,
        targets_buildings_only=wincon_bldg,
        is_splash=splash,
        is_resettable=resettable,
        is_small_spell=small_spell,
        stomp_vulnerable=False if (air or spell) else stomp_vuln,
    )


CARDS_DB: Dict[str, CardInfo] = {
    # === User's 8-Card MK Zap Bait Deck ===
    "mega_knight": _g(
        "mega_knight", "Mega Knight", 7, WeightClass.HEAVY_TANK,
        splash=True, stomp_vuln=False
    ),
    "miner": _g(
        "miner", "Miner", 3, WeightClass.MEDIUM,
        anywhere=True, stomp_vuln=True
    ),
    "inferno_dragon": _g(
        "inferno_dragon", "Inferno Dragon", 4, WeightClass.MEDIUM,
        air=True, hits_air=True, resettable=True, stomp_vuln=False
    ),
    "skeleton_barrel": _g(
        "skeleton_barrel", "Skeleton Barrel", 3, WeightClass.LIGHT,
        air=True, wincon_bldg=True, stomp_vuln=False
    ),
    "zap": _g(
        "zap", "Zap", 2, spell=True, anywhere=True,
        hits_air=True, splash=True, small_spell=True
    ),
    "bats": _g(
        "bats", "Bats", 2, WeightClass.SWARM,
        air=True, hits_air=True, stomp_vuln=False
    ),
    "spear_goblins": _g(
        "spear_goblins", "Spear Goblins", 2, WeightClass.SWARM,
        hits_air=True, stomp_vuln=True
    ),
    "goblin_gang": _g(
        "goblin_gang", "Goblin Gang", 3, WeightClass.SWARM,
        hits_air=True, stomp_vuln=True
    ),

    # === Common Swaps & Swarms ===
    "skeleton_army": _g("skeleton_army", "Skeleton Army", 3, WeightClass.SWARM, stomp_vuln=True),
    "goblin_barrel": _g("goblin_barrel", "Goblin Barrel", 3, WeightClass.SWARM, anywhere=True, stomp_vuln=True),
    "skeletons": _g("skeletons", "Skeletons", 1, WeightClass.SWARM, stomp_vuln=False),
    "goblins": _g("goblins", "Goblins", 2, WeightClass.SWARM, stomp_vuln=True),
    "guards": _g("guards", "Guards", 3, WeightClass.SWARM, stomp_vuln=True),
    "minions": _g("minions", "Minions", 3, WeightClass.SWARM, air=True, hits_air=True),
    "minion_horde": _g("minion_horde", "Minion Horde", 5, WeightClass.SWARM, air=True, hits_air=True),
    "ice_spirit": _g("ice_spirit", "Ice Spirit", 1, WeightClass.LIGHT, hits_air=True, splash=True, stomp_vuln=False),
    "electro_spirit": _g("electro_spirit", "Electro Spirit", 1, WeightClass.LIGHT, hits_air=True, splash=True, stomp_vuln=False),
    "fire_spirit": _g("fire_spirit", "Fire Spirit", 1, WeightClass.LIGHT, hits_air=True, splash=True, stomp_vuln=False),

    # === Win Conditions ===
    "hog_rider": _g("hog_rider", "Hog Rider", 4, WeightClass.MEDIUM, wincon_bldg=True, stomp_vuln=True),
    "royal_hogs": _g("royal_hogs", "Royal Hogs", 5, WeightClass.MEDIUM, wincon_bldg=True, stomp_vuln=True),
    "battle_ram": _g("battle_ram", "Battle Ram", 4, WeightClass.MEDIUM, wincon_bldg=True, resettable=True, stomp_vuln=True),
    "ram_rider": _g("ram_rider", "Ram Rider", 5, WeightClass.MEDIUM, hits_air=True, wincon_bldg=True, resettable=True, stomp_vuln=True),
    "wall_breakers": _g("wall_breakers", "Wall Breakers", 2, WeightClass.LIGHT, wincon_bldg=True, stomp_vuln=True),
    "giant": _g("giant", "Giant", 5, WeightClass.HEAVY_TANK, wincon_bldg=True, stomp_vuln=False),
    "golem": _g("golem", "Golem", 8, WeightClass.HEAVY_TANK, wincon_bldg=True, stomp_vuln=False),
    "electro_giant": _g("electro_giant", "Electro Giant", 7, WeightClass.HEAVY_TANK, hits_air=True, wincon_bldg=True, stomp_vuln=False),
    "goblin_giant": _g("goblin_giant", "Goblin Giant", 6, WeightClass.HEAVY_TANK, hits_air=True, wincon_bldg=True, stomp_vuln=False),
    "royal_giant": _g("royal_giant", "Royal Giant", 6, WeightClass.HEAVY_TANK, wincon_bldg=True, stomp_vuln=False),
    "balloon": _g("balloon", "Balloon", 5, WeightClass.MEDIUM, air=True, wincon_bldg=True),
    "lava_hound": _g("lava_hound", "Lava Hound", 7, WeightClass.HEAVY_TANK, air=True, wincon_bldg=True),
    "graveyard": _g("graveyard", "Graveyard", 5, spell=True, anywhere=True),
    "goblin_drill": _g("goblin_drill", "Goblin Drill", 4, WeightClass.BUILDING, anywhere=True, wincon_bldg=True, stomp_vuln=True),
    "elixir_golem": _g("elixir_golem", "Elixir Golem", 3, WeightClass.MEDIUM, wincon_bldg=True, stomp_vuln=True),

    # === Heavy Killers / Mini-Tanks ===
    "pekka": _g("pekka", "P.E.K.K.A", 7, WeightClass.HEAVY_TANK, stomp_vuln=False),
    "mini_pekka": _g("mini_pekka", "Mini P.E.K.K.A", 4, WeightClass.MEDIUM, stomp_vuln=True),
    "prince": _g("prince", "Prince", 5, WeightClass.MEDIUM, resettable=True, stomp_vuln=False),
    "dark_prince": _g("dark_prince", "Dark Prince", 4, WeightClass.MEDIUM, splash=True, resettable=True, stomp_vuln=True),
    "elite_barbarians": _g("elite_barbarians", "Elite Barbarians", 6, WeightClass.MEDIUM, stomp_vuln=True),
    "barbarians": _g("barbarians", "Barbarians", 5, WeightClass.SWARM, stomp_vuln=True),
    "knight": _g("knight", "Knight", 3, WeightClass.MEDIUM, stomp_vuln=True),
    "valkyrie": _g("valkyrie", "Valkyrie", 4, WeightClass.MEDIUM, splash=True, stomp_vuln=True),
    "ice_golem": _g("ice_golem", "Ice Golem", 2, WeightClass.MEDIUM, wincon_bldg=True, stomp_vuln=False),
    "bandit": _g("bandit", "Bandit", 3, WeightClass.MEDIUM, stomp_vuln=True),
    "royal_ghost": _g("royal_ghost", "Royal Ghost", 3, WeightClass.MEDIUM, splash=True, stomp_vuln=True),
    "lumberjack": _g("lumberjack", "Lumberjack", 4, WeightClass.MEDIUM, stomp_vuln=True),
    "giant_skeleton": _g("giant_skeleton", "Giant Skeleton", 6, WeightClass.HEAVY_TANK, stomp_vuln=False),
    "bowler": _g("bowler", "Bowler", 5, WeightClass.HEAVY_TANK, splash=True, stomp_vuln=False),
    "sparky": _g("sparky", "Sparky", 6, WeightClass.HEAVY_TANK, splash=True, resettable=True, stomp_vuln=False),
    "mighty_miner": _g("mighty_miner", "Mighty Miner", 4, WeightClass.MEDIUM, resettable=True, stomp_vuln=False),
    "golden_knight": _g("golden_knight", "Golden Knight", 4, WeightClass.MEDIUM, stomp_vuln=True),
    "skeleton_king": _g("skeleton_king", "Skeleton King", 4, WeightClass.MEDIUM, splash=True, stomp_vuln=False),

    # === Ranged / Support ===
    "musketeer": _g("musketeer", "Musketeer", 4, WeightClass.MEDIUM, hits_air=True, stomp_vuln=True),
    "wizard": _g("wizard", "Wizard", 5, WeightClass.MEDIUM, hits_air=True, splash=True, stomp_vuln=True),
    "witch": _g("witch", "Witch", 5, WeightClass.MEDIUM, hits_air=True, splash=True, stomp_vuln=True),
    "electro_wizard": _g("electro_wizard", "Electro Wizard", 4, WeightClass.MEDIUM, hits_air=True, stomp_vuln=True),
    "ice_wizard": _g("ice_wizard", "Ice Wizard", 3, WeightClass.LIGHT, hits_air=True, splash=True, stomp_vuln=True),
    "magic_archer": _g("magic_archer", "Magic Archer", 4, WeightClass.LIGHT, hits_air=True, splash=True, stomp_vuln=True),
    "princess": _g("princess", "Princess", 3, WeightClass.LIGHT, hits_air=True, splash=True, stomp_vuln=True),
    "firecracker": _g("firecracker", "Firecracker", 3, WeightClass.LIGHT, hits_air=True, splash=True, stomp_vuln=True),
    "dart_goblin": _g("dart_goblin", "Dart Goblin", 3, WeightClass.LIGHT, hits_air=True, stomp_vuln=True),
    "archers": _g("archers", "Archers", 3, WeightClass.LIGHT, hits_air=True, stomp_vuln=True),
    "executioner": _g("executioner", "Executioner", 5, WeightClass.MEDIUM, hits_air=True, splash=True, stomp_vuln=True),
    "bomber": _g("bomber", "Bomber", 2, WeightClass.LIGHT, splash=True, stomp_vuln=True),
    "baby_dragon": _g("baby_dragon", "Baby Dragon", 4, WeightClass.MEDIUM, air=True, hits_air=True, splash=True),
    "mega_minion": _g("mega_minion", "Mega Minion", 3, WeightClass.MEDIUM, air=True, hits_air=True),
    "electro_dragon": _g("electro_dragon", "Electro Dragon", 5, WeightClass.MEDIUM, air=True, hits_air=True),
    "phoenix": _g("phoenix", "Phoenix", 4, WeightClass.MEDIUM, air=True, hits_air=True),
    "little_prince": _g("little_prince", "Little Prince", 3, WeightClass.LIGHT, hits_air=True, stomp_vuln=True),

    # === Buildings ===
    "cannon": _g("cannon", "Cannon", 3, WeightClass.BUILDING, stomp_vuln=False),
    "tesla": _g("tesla", "Tesla", 4, WeightClass.BUILDING, hits_air=True, stomp_vuln=False),
    "inferno_tower": _g("inferno_tower", "Inferno Tower", 5, WeightClass.BUILDING, hits_air=True, resettable=True, stomp_vuln=False),
    "bomb_tower": _g("bomb_tower", "Bomb Tower", 4, WeightClass.BUILDING, splash=True, stomp_vuln=False),
    "tombstone": _g("tombstone", "Tombstone", 3, WeightClass.BUILDING, stomp_vuln=False),
    "goblin_cage": _g("goblin_cage", "Goblin Cage", 4, WeightClass.BUILDING, stomp_vuln=False),
    "x_bow": _g("x_bow", "X-Bow", 6, WeightClass.BUILDING, stomp_vuln=True),
    "mortar": _g("mortar", "Mortar", 4, WeightClass.BUILDING, splash=True, stomp_vuln=True),

    # === Spells ===
    "the_log": _g("the_log", "The Log", 2, spell=True, splash=True, small_spell=True),
    "arrows": _g("arrows", "Arrows", 3, spell=True, hits_air=True, splash=True, small_spell=True),
    "giant_snowball": _g("giant_snowball", "Giant Snowball", 2, spell=True, hits_air=True, splash=True, small_spell=True),
    "barbarian_barrel": _g("barbarian_barrel", "Barbarian Barrel", 2, spell=True, splash=True, small_spell=True),
    "rage": _g("rage", "Rage", 2, spell=True, hits_air=True, splash=True, small_spell=True),
    "royal_delivery": _g("royal_delivery", "Royal Delivery", 3, spell=True, hits_air=True, splash=True, small_spell=True),
    "fireball": _g("fireball", "Fireball", 4, spell=True, hits_air=True, splash=True),
    "poison": _g("poison", "Poison", 4, spell=True, hits_air=True, splash=True),
    "tornado": _g("tornado", "Tornado", 3, spell=True, hits_air=True, splash=True, small_spell=True),
    "rocket": _g("rocket", "Rocket", 6, spell=True, hits_air=True, splash=True),
    "lightning": _g("lightning", "Lightning", 6, spell=True, hits_air=True),
    "freeze": _g("freeze", "Freeze", 4, spell=True, hits_air=True),
    "earthquake": _g("earthquake", "Earthquake", 3, spell=True, splash=True),

    # === Generic Fallback Classes (produced by OpenCV Healthbar Detector when YOLO weights aren't loaded) ===
    "unknown_swarm": _g("unknown_swarm", "Enemy Swarm", 3, WeightClass.SWARM, stomp_vuln=True),
    "unknown_light": _g("unknown_light", "Enemy Light Unit", 3, WeightClass.LIGHT, stomp_vuln=True),
    "unknown_medium": _g("unknown_medium", "Enemy Medium Push", 4, WeightClass.MEDIUM, stomp_vuln=True),
    "unknown_tank": _g("unknown_tank", "Enemy Heavy Tank", 6, WeightClass.HEAVY_TANK, stomp_vuln=False),
    "unknown_air": _g("unknown_air", "Enemy Air Unit", 4, WeightClass.MEDIUM, air=True, hits_air=True),
}


CARD_ALIASES: Dict[str, str] = {
    "mk": "mega_knight",
    "megaknight": "mega_knight",
    "evo_mega_knight": "mega_knight",
    "idrag": "inferno_dragon",
    "idragon": "inferno_dragon",
    "skelly_barrel": "skeleton_barrel",
    "skarrel": "skeleton_barrel",
    "evo_skeleton_barrel": "skeleton_barrel",
    "evo_zap": "zap",
    "evo_bats": "bats",
    "speer_goblins": "spear_goblins",
    "spear_gobs": "spear_goblins",
    "goblin_army": "goblin_gang",
    "koboldarmee": "goblin_gang",
    "koboldgang": "goblin_gang",
    "gob_gang": "goblin_gang",
    "skarmy": "skeleton_army",
    "log": "the_log",
    "ewiz": "electro_wizard",
    "ebarbs": "elite_barbarians",
    "rg": "royal_giant",
    "egolem": "elixir_golem",
    "egiant": "electro_giant",
    "p_e_k_k_a": "pekka",
}


def normalize_card_name(raw_name: str) -> str:
    """Convert any user-typed or detector-emitted card label to its canonical slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw_name.strip().lower()).strip("_")
    if cleaned.startswith("evo_") and cleaned not in CARD_ALIASES and cleaned[4:] in CARDS_DB:
        cleaned = cleaned[4:]
    return CARD_ALIASES.get(cleaned, cleaned)


def get_card_info(card_name: str) -> CardInfo:
    """Look up CardInfo with safe fallback for unknown labels."""
    slug = normalize_card_name(card_name)
    if slug in CARDS_DB:
        return CARDS_DB[slug]
    return CardInfo(
        name=slug,
        display_name=slug.replace("_", " ").title(),
        elixir=4,
        move_type=MoveType.GROUND,
        weight_class=WeightClass.MEDIUM,
        stomp_vulnerable=True,
    )
