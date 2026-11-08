"""Tests for the sprite-library data-quality work (factions, body-part frames, thin folders).

These are the guards for the bug report "some sprite folders mix the enemy and ally renders, and
the library is incomplete": every assertion here is about *the data the matcher is given*, not
about scoring quality, because that is where the defect lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cr_bot.vision.sprite_factions import (
    ALLY,
    ENEMY,
    NEUTRAL,
    CardFactions,
    audit_library,
    card_factions,
    frame_faction,
    load_faction_map,
    write_faction_map,
)
from cr_bot.vision.sprite_matcher import SpriteLibrary

SPRITES = Path("assets/unit_sprites")


# -- helpers -----------------------------------------------------------------------------------
def _sprite(alpha_shape=(60, 50), colour=(200, 200, 200)) -> np.ndarray:
    """A BGRA frame: an opaque body on a transparent background."""
    h, w = alpha_shape
    img = np.zeros((h, w, 4), np.uint8)
    img[10:h - 10, 8:w - 8] = (*colour, 255)
    return img


def _tinted(img: np.ndarray, fraction: float, colour_bgr=(40, 40, 220)) -> np.ndarray:
    """Repaint ``fraction`` of the opaque area in a team colour (the game's clothing tint)."""
    out = img.copy()
    ys, xs = np.nonzero(out[:, :, 3] > 128)
    n = int(len(ys) * fraction)
    out[ys[:n], xs[:n], :3] = colour_bgr
    return out


# -- faction measurement -----------------------------------------------------------------------
def test_a_blue_frame_reads_as_ally_and_a_red_frame_as_enemy():
    blue = _tinted(_sprite(), 0.6, (200, 60, 40))     # BGR: high blue
    red = _tinted(_sprite(), 0.6, (40, 60, 200))      # BGR: high red
    assert frame_faction(blue).faction == ALLY
    assert frame_faction(red).faction == ENEMY


def test_orange_skin_cannot_vote_a_faction():
    """Skin, wood and gold live in the orange band and carry no team information.

    This is the reason the measurement counts hue-restricted pixels instead of averaging the
    red/blue balance: a Lumberjack's axe and a Knight's skin outweigh their clothing otherwise,
    and every render then reads "red".
    """
    orange = _tinted(_sprite(), 1.0, (30, 110, 230))   # BGR orange, H≈20
    assert frame_faction(orange).faction == NEUTRAL


def test_a_grey_sprite_has_no_faction():
    assert frame_faction(_sprite()).faction == NEUTRAL


def test_one_folder_can_hold_both_factions():
    ally = (200, 60, 40)      # BGR: blue-dominant, i.e. the player's render
    enemy = (40, 60, 200)     # BGR: red-dominant, i.e. the opponent's render
    images = [("f00", _tinted(_sprite(), 0.7, ally)),
              ("f01", _tinted(_sprite(), 0.7, enemy)),
              ("f02", _tinted(_sprite(), 0.7, ally))]
    audit = card_factions("mixed_card", images)
    assert audit.is_mixed
    assert audit.ally == ["f00", "f02"]
    assert audit.enemy == ["f01"]
    assert audit.missing is None


def test_a_single_faction_folder_reports_which_half_is_missing():
    red = (40, 60, 200)       # BGR red-dominant -> enemy
    images = [("f00", _tinted(_sprite(), 0.7, red)), ("f01", _tinted(_sprite(), 0.7, red))]
    audit = card_factions("enemy_only", images)
    assert not audit.is_mixed
    assert audit.missing == ALLY


def test_a_body_part_frame_is_flagged():
    """A shoulder or a detached wheel is a bad template and must be recognisable as such."""
    full = _sprite((80, 60))
    part = _sprite((24, 20))
    images = [("f00", full), ("f01", full), ("f02", full), ("f03", part)]
    audit = card_factions("card", images)
    assert audit.fragments == ["f03"]


# -- the shipped library -----------------------------------------------------------------------
@pytest.mark.skipif(not SPRITES.is_dir(), reason="sprite library not present")
def test_the_shipped_library_is_auditable_end_to_end(tmp_path):
    audits = audit_library(SPRITES)
    assert len(audits) > 40, "the shipped library should have dozens of cards"
    # every frame must end up with a definite classification
    for audit in audits:
        assert audit.frames
        for info in audit.frames.values():
            assert info.faction in {ALLY, ENEMY, NEUTRAL}

    stamp = "test"
    path = write_faction_map(tmp_path, audits, stamp)
    payload = json.loads(path.read_text())
    assert payload["cards"]
    loaded = load_faction_map(tmp_path)
    assert set(loaded) == {a.card for a in audits}


@pytest.mark.skipif(not SPRITES.is_dir(), reason="sprite library not present")
def test_lumberjack_frames_really_do_alternate_factions():
    """The user's report, pinned as a regression test.

    Upstream ships the Lumberjack with a blue jacket and with a red jacket; the old selection
    kept a mix.  If this ever becomes single-faction again, ``--balance-factions`` has stopped
    working (or the library was re-fetched without it).
    """
    audit = card_factions("lumberjack", [
        (p.stem, cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
        for p in sorted((SPRITES / "lumberjack").glob("*.png"))
    ])
    assert audit.is_mixed, "lumberjack should hold both the ally and the enemy render"


@pytest.mark.skipif(not SPRITES.is_dir(), reason="sprite library not present")
def test_library_loads_the_faction_map_and_spans_both_teams():
    library = SpriteLibrary(SPRITES)
    assert library.factions, "the faction map next to the sprites should be picked up"
    mixed = [c for c in library.cards if library.frames_by_faction(c, ALLY)
             and library.frames_by_faction(c, ENEMY)]
    assert mixed, "at least some cards must offer templates for both teams"
    # frames_by_faction must never invent frames for a card that has none of that tint
    giant_frames = library.frames_for("giant")
    if giant_frames:
        ally = library.frames_by_faction("giant", ALLY)
        assert len(ally) <= len(giant_frames)


def test_library_without_a_faction_map_keeps_the_old_behaviour(tmp_path):
    """No map == use every frame in name order, exactly as before the audit existed."""
    for i in range(3):
        img = _sprite()
        cv2.imwrite(str(tmp_path / f"f{i:02d}.png"), img)
    (tmp_path / "card_a").mkdir(exist_ok=True)
    for i in range(3):
        cv2.imwrite(str(tmp_path / "card_a" / f"f{i:02d}.png"), _sprite())
    library = SpriteLibrary(tmp_path, max_templates=8)
    frames = library.frames_for("card_a")
    assert [f.stem for f in frames] == ["f00", "f01", "f02"]
    assert library.frames_by_faction("card_a", ALLY) == frames   # unknown map: no filtering


def test_frames_are_interleaved_across_factions_so_both_tints_survive_the_template_cap(tmp_path):
    """With a template cap of 4, a folder whose first frames are all ally must not lose the enemy.

    This is the failure mode that made half the game unmatchable: ``max_templates`` truncates the
    list, so name order decides which single tint survives.
    """
    card_dir = tmp_path / "card_b"
    card_dir.mkdir()
    team = [ALLY, ALLY, ALLY, ALLY, ENEMY, ENEMY]
    colours = [(200, 60, 40), (40, 60, 200)]     # index by faction
    for i, faction in enumerate(team):
        colour = colours[0] if faction == ALLY else colours[1]
        cv2.imwrite(str(card_dir / f"f{i:02d}.png"), _tinted(_sprite(), 0.7, colour))
    factions = {f"f{i:02d}": faction for i, faction in enumerate(team)}
    library = SpriteLibrary(tmp_path, max_templates=4, factions={"card_b": factions})
    stems = [f.stem for f in library.frames_for("card_b")]
    assert len(stems) == 4
    assert any(factions[s] == ALLY for s in stems)
    assert any(factions[s] == ENEMY for s in stems), (
        f"enemy frames were lost to the template cap: {stems}")
