"""Which faction does a reference sprite frame depict? (classical colour measurement, no ML)

Why this module exists
----------------------
``assets/unit_sprites`` holds a handful of real in-game animation frames per card, harvested from
the decrypted asset dump ``smlbiobot/cr-assets-png`` by ``scripts/fetch_game_sprites.py``.  That
dump keeps **both team tints** for every character in one directory, and the game paints a unit's
team colour onto its clothing:

    Lumberjack   blue jacket / red jacket, alternating pose by pose
    Knight       blue scarf / red scarf
    Golem, Cannon, Tesla, Sparky, ...  crystal or rope details in blue / red

The fetch script used to keep the eight most *visually diverse* frames, and faction is one of the
things that makes two frames look different — so it deliberately collected a mix.  The result is a
folder that alternates factions, and no way to ask for "the frames of the side I am looking at".

What mixing does and does not cost (measured, not assumed)
----------------------------------------------------------
Pose variation dominates: masked ZNCC between two frames of the *same* faction averages 0.10, and
between the two factions 0.10-0.18.  So a mixed reference set does **not** measurably degrade card
identification, and the matching code is deliberately left alone.

What it does cost is coverage: most folders are, in effect, single-faction — the Giant, Hog Rider,
Bats and Minions are *entirely* one tint — so half the cards in the game cannot be matched against
the side they are actually seen on.  That is a data problem with a data fix: record the faction of
every frame (:func:`frame_faction`) and harvest both factions on purpose
(``scripts/fetch_game_sprites.py --balance-factions``).

Method
------
Count strongly team-coloured pixels and compare them:

* **red / magenta** — hue outside the orange band (skin, wood, gold), which is *not* team colour
* **blue / cyan** — hue in the blue band

and score ``log((blue + 1) / (red + 1))``: positive means the blue render, negative the red one.
Using counts of *hue-restricted* pixels rather than an average hue is what makes this robust: a
Lumberjack's axe and a Knight's skin are warm in both renders, but they sit in the orange band and
are excluded, so they cannot outvote the clothing the game actually tints.

Everything here is arithmetic on pixel counts — no training, no weights, no GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ENEMY = "enemy"      # red tint: the opponent's render
ALLY = "ally"        # blue tint: the player's render
NEUTRAL = "neutral"  # no measurable team colour (grey stone, bone, a building's timber)

#: |score| below this is "no decisive team colour" — ~2:1 pixel majority is required, so a
#: handful of tinted highlights on an otherwise grey sprite cannot decide a faction
DECISIVE = 0.70

#: HSV bands, calibrated against frames whose faction is visible to the eye.
#: The orange band (roughly H 8-25 in OpenCV's 0..179 scale) is skin, wood, gold and fire, and is
#: excluded from both teams on purpose — it carries no team information and dominates some cards.
RED_HUE_LOW = 8        # H < 8 or H > 167  -> red / crimson / magenta
BLUE_HUE_LOW = 95      # 95 <= H <= 135    -> blue / cyan
BLUE_HUE_HIGH = 135
MIN_SATURATION = 70
MIN_VALUE = 55

#: a frame whose opaque area is below this share of the card's *largest* frame is a body part,
#: not a pose (relative to the largest, because a folder can be mostly fragments - Sparky ships
#: seven frames of a detached wheel and one of the whole machine)
FRAGMENT_AREA_RATIO = 0.45


@dataclass
class FrameFaction:
    name: str
    faction: str
    score: float          # log((blue+1)/(red+1)); + is the ally render
    red_px: int = 0
    blue_px: int = 0
    coverage: float = 0.0  # share of opaque pixels that are team-coloured
    #: opaque area as a share of the card's typical frame.  Frame selection upstream keeps some
    #: stray *body parts* — a shoulder, a wheel, a detached arm — because they differ from every
    #: other pose and so win the diversity contest.  They are worse than useless as templates:
    #: they never match a whole unit, and they pull ``median_aspect`` (which the matcher uses to
    #: size its search window) towards the aspect of a fragment.
    area_ratio: float = 1.0

    @property
    def is_fragment(self) -> bool:
        return self.area_ratio < FRAGMENT_AREA_RATIO


def tint_counts(img: np.ndarray) -> Tuple[int, int, int]:
    """(red_px, blue_px, opaque_px) for an RGBA/BGRA sprite frame."""
    if img is None or img.ndim != 3 or img.shape[2] < 4:
        return 0, 0, 0
    alpha = img[:, :, 3] > 128
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)
    strong = alpha & (s >= MIN_SATURATION) & (v >= MIN_VALUE)
    red = strong & ((h < RED_HUE_LOW) | (h > 167))
    blue = strong & (h >= BLUE_HUE_LOW) & (h <= BLUE_HUE_HIGH)
    return int(red.sum()), int(blue.sum()), int(alpha.sum())


def frame_score(img: np.ndarray) -> Tuple[float, int, int, float]:
    """(score, red_px, blue_px, coverage) — see the module docstring for the sign convention."""
    red, blue, opaque = tint_counts(img)
    if opaque < 50:
        return 0.0, red, blue, 0.0
    score = float(np.log((blue + 1.0) / (red + 1.0)))
    return score, red, blue, (red + blue) / float(opaque)


def frame_faction(img: np.ndarray) -> FrameFaction:
    """Classify a single frame. Per-frame, because a folder can hold both factions."""
    score, red, blue, coverage = frame_score(img)
    if coverage < 0.02 or abs(score) < DECISIVE:
        faction = NEUTRAL
    else:
        faction = ALLY if score > 0 else ENEMY
    area = int((img[:, :, 3] > 128).sum()) if img is not None and img.ndim == 3 else 0
    info = FrameFaction(name="", faction=faction, score=round(score, 3),
                        red_px=red, blue_px=blue, coverage=round(coverage, 3))
    object.__setattr__(info, "_area", area)
    return info


@dataclass
class CardFactions:
    """Per-frame factions of one card, plus a summary the fetch script can act on."""

    card: str
    frames: Dict[str, FrameFaction] = field(default_factory=dict)

    def count(self, faction: str) -> int:
        return sum(1 for f in self.frames.values() if f.faction == faction)

    @property
    def enemy(self) -> List[str]:
        return [n for n, f in self.frames.items() if f.faction == ENEMY]

    @property
    def ally(self) -> List[str]:
        return [n for n, f in self.frames.items() if f.faction == ALLY]

    @property
    def neutral(self) -> List[str]:
        return [n for n, f in self.frames.items() if f.faction == NEUTRAL]

    @property
    def fragments(self) -> List[str]:
        """Frames that are body parts rather than poses (see ``FRAGMENT_AREA_RATIO``)."""
        return [n for n, f in self.frames.items() if f.is_fragment]

    @property
    def is_mixed(self) -> bool:
        return bool(self.enemy) and bool(self.ally)

    @property
    def missing(self) -> Optional[str]:
        """Which faction this card has no frames for (``None`` when both or neither exist)."""
        if self.enemy and not self.ally:
            return ALLY
        if self.ally and not self.enemy:
            return ENEMY
        return None


def card_factions(card: str, images: Sequence[Tuple[str, np.ndarray]]) -> CardFactions:
    """Classify every ``(frame_stem, image)`` pair of one card, and flag body-part frames."""
    out = CardFactions(card=card)
    areas: List[int] = []
    for name, img in images:
        info = frame_faction(img)
        info.name = name
        areas.append(getattr(info, "_area", 0))
        out.frames[name] = info
    reference = float(max(areas)) if areas else 0.0
    for info in out.frames.values():
        area = getattr(info, "_area", 0)
        info.area_ratio = round(area / reference, 3) if reference > 0 else 1.0
    return out


def load_images(card_dir: Path) -> List[Tuple[str, np.ndarray]]:
    images: List[Tuple[str, np.ndarray]] = []
    for path in sorted(Path(card_dir).glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append((path.stem, img))
    return images


def audit_library(sprites_dir: Path) -> List[CardFactions]:
    """Measure every card folder in a sprite library."""
    root = Path(sprites_dir)
    if not root.is_dir():
        return []
    return [
        card_factions(p.name, load_images(p))
        for p in sorted(root.iterdir())
        if p.is_dir() and not p.name.startswith("_")
    ]


def write_faction_map(sprites_dir: Path, audits: Iterable[CardFactions], generated: str) -> Path:
    """Persist the measurements next to the sprites, for the matcher and for review."""
    payload = {
        "generated": generated,
        "description": (
            "Faction of every reference sprite frame, measured from its own colours. "
            "score = log((blue_px+1)/(red_px+1)); positive is the ally (blue) render, "
            "negative the enemy (red) render, |score| < 0.70 is 'neutral'. "
            "Written by tools/sanitize_sprite_library.py; consumed by SpriteLibrary."
        ),
        "cards": {
            audit.card: {
                "mixed": audit.is_mixed,
                "missing_faction": audit.missing,
                "frames": {
                    name: {"faction": f.faction, "score": f.score,
                           "red_px": f.red_px, "blue_px": f.blue_px,
                           "fragment": f.is_fragment}
                    for name, f in audit.frames.items()
                },
            }
            for audit in audits
        },
    }
    path = Path(sprites_dir) / "_factions.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_faction_map(sprites_dir: Path) -> Dict[str, Dict[str, str]]:
    """``{card: {frame_stem: faction}}`` from ``_factions.json``; empty dict when absent."""
    path = Path(sprites_dir) / "_factions.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        card: {name: info.get("faction", NEUTRAL) for name, info in (entry.get("frames") or {}).items()}
        for card, entry in (payload.get("cards") or {}).items()
    }
