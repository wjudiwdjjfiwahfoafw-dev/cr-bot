"""Phase detection must survive a single odd frame, and auto-queue must walk the whole lobby.

Both of these come straight from a real session log on an SM-A528B (Android 16):

    === Match started: resetting trackers ===
    === Match ended (unknown) ===            <- 8 s later, no match had ended
    === Match started: resetting trackers ===
    === Match ended (match_over) ===          <- 1 s later

A single frame in which the phase detector was unsure was enough to end the match, reset every
tracker and increment ``matches_played``. With ``--auto-queue`` that also starts tapping menus
mid-battle. The detector works on narrow colour bands in fixed screen regions, so an odd frame is
normal (a white spell flash over the elixir tray, the end-of-match dimming, a frame caught
mid-transition). The fix belongs in the *consumer*: a match is a long-lived state that needs
confirmation to leave, not a per-frame guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from cr_bot.models import GamePhase
from cr_bot.vision.gamescreen import PhaseStabilizer

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_FRAME = REPO_ROOT / "calibration_preview.png"


# ======================================================================================
# the stabilizer contract
# ======================================================================================

def _feed(stab: PhaseStabilizer, phases: List[GamePhase], ts0: float = 0.0,
          dt: float = 0.05) -> GamePhase:
    out = GamePhase.UNKNOWN
    for i, p in enumerate(phases):
        out = stab.update(p, ts0 + i * dt)
    return out


def test_a_single_unknown_frame_does_not_end_a_match() -> None:
    stab = PhaseStabilizer(battle_exit_seconds=1.0)
    assert _feed(stab, [GamePhase.BATTLE_1X] * 4) is GamePhase.BATTLE_1X
    assert stab.in_battle is True
    # one odd frame, then the match continues as before
    out = _feed(stab, [GamePhase.UNKNOWN], ts0=0.20)
    assert out is GamePhase.BATTLE_1X
    assert stab.in_battle is True
    out = _feed(stab, [GamePhase.BATTLE_1X] * 20, ts0=0.25)
    assert out is GamePhase.BATTLE_1X
    assert stab.in_battle is True


def test_one_odd_menu_frame_does_not_start_tapping_the_lobby() -> None:
    """The mirror case: a menu-looking frame in the middle of a match must not queue a match."""
    stab = PhaseStabilizer(battle_exit_seconds=1.0)
    _feed(stab, [GamePhase.BATTLE_2X] * 4)
    assert _feed(stab, [GamePhase.MAIN_MENU], ts0=0.20) is GamePhase.BATTLE_2X
    assert stab.in_battle is True


def test_a_real_match_end_is_accepted_after_the_grace_period() -> None:
    stab = PhaseStabilizer(battle_exit_seconds=1.0)
    _feed(stab, [GamePhase.BATTLE_1X] * 4)
    # the match-over screen stays up: the phase must be believed, promptly but not instantly
    assert _feed(stab, [GamePhase.MATCH_OVER], ts0=0.20) is GamePhase.BATTLE_1X
    out = _feed(stab, [GamePhase.MATCH_OVER] * 40, ts0=0.25, dt=0.05)   # 2 s of match-over
    assert out is GamePhase.MATCH_OVER
    assert stab.in_battle is False


def test_a_new_match_starts_immediately() -> None:
    """Entering a battle is never delayed: the bot must not waste the first seconds of a match."""
    stab = PhaseStabilizer(battle_exit_seconds=5.0)
    _feed(stab, [GamePhase.MAIN_MENU] * 4)
    assert stab.in_battle is False
    assert _feed(stab, [GamePhase.BATTLE_1X], ts0=1.0) is GamePhase.BATTLE_1X
    assert stab.in_battle is True


def test_battle_sub_phases_are_not_mistaken_for_an_exit() -> None:
    stab = PhaseStabilizer(battle_exit_seconds=1.0)
    _feed(stab, [GamePhase.BATTLE_1X] * 4)
    out = _feed(stab, [GamePhase.BATTLE_2X, GamePhase.BATTLE_3X, GamePhase.BATTLE_2X], ts0=0.2)
    assert out.is_in_battle
    assert stab.in_battle is True


def test_reset_clears_the_match_state() -> None:
    stab = PhaseStabilizer(battle_exit_seconds=1.0)
    _feed(stab, [GamePhase.BATTLE_1X] * 4)
    stab.reset_match()
    assert stab.in_battle is False


def test_stabilizer_output_is_one_of_the_real_phases() -> None:
    stab = PhaseStabilizer()
    seen = set()
    for p in [GamePhase.UNKNOWN, GamePhase.MAIN_MENU, GamePhase.MATCH_LOADING,
              GamePhase.BATTLE_1X, GamePhase.BATTLE_1X, GamePhase.MATCH_OVER]:
        seen.add(stab.update(p, 0.0))
    assert seen <= set(GamePhase)


# ======================================================================================
# the real frame must not read as "match over"
# ======================================================================================

def test_the_real_battle_frame_is_a_battle() -> None:
    """Guard against a colour band that quietly stops matching on a real screenshot."""
    from cr_bot.arena.grid import ArenaCalibration, ArenaGrid
    from cr_bot.arena.projection import RectProjection
    from cr_bot.vision.game_state import GameStateDetector

    assert REAL_FRAME.is_file(), "the sample device frame is the fixture for this test"
    frame = cv2.imread(str(REAL_FRAME))
    assert frame is not None
    h, w = frame.shape[:2]
    cal = ArenaCalibration(
        frame_size=(w, h),
        projection=RectProjection(30, 100, w - 20, h - 250),
        hand_slots_px=[(0.31 * w, 0.908 * h), (0.50 * w, 0.908 * h),
                       (0.69 * w, 0.908 * h), (0.875 * w, 0.908 * h)],
        hand_card_size_px=(0.16 * w, 0.09 * h),
        elixir_px=(0.20 * w, 0.965 * w, 0.983 * h),
        river_px_y=0.44 * h,
    )
    grid = ArenaGrid(w, h, cal)
    detector = GameStateDetector(grid)
    phase = detector.detect_phase(frame, 0.0)
    assert phase.is_in_battle, f"a live battle frame must not read as {phase}"
    assert detector._has_battle_hud(frame) is True


# ======================================================================================
# the lobby ladder
# ======================================================================================

def test_auto_queue_walks_battle_then_mode_then_play() -> None:
    """One tap per call, in order, with a rate limit between them."""
    from cr_bot.vision.gamescreen import LobbyNavigator

    nav = LobbyNavigator(gap_s=1.0)
    taps = []
    now = 0.0
    for _ in range(30):
        step = nav.next_tap(now)
        if step is not None:
            taps.append(step[0])
        now += 1.1
    # in order, one step at a time; a step may be retried before the next one is tried
    assert list(dict.fromkeys(taps)) == ["battle", "mode", "play"], taps
    assert nav.finished is True


def test_the_ladder_is_rate_limited() -> None:
    from cr_bot.vision.gamescreen import LobbyNavigator

    nav = LobbyNavigator(gap_s=2.0)
    assert nav.next_tap(0.0) is not None
    assert nav.next_tap(0.5) is None, "two taps 0.5 s apart would be swallowed by the game"
    assert nav.next_tap(2.1) is not None


def test_a_step_is_retried_a_bounded_number_of_times() -> None:
    from cr_bot.vision.gamescreen import LobbyNavigator

    nav = LobbyNavigator(gap_s=0.0, max_attempts_per_step=2)
    assert nav.next_tap(0.0)[0] == "battle"
    assert nav.next_tap(0.1)[0] == "battle"     # retried once
    assert nav.next_tap(0.2)[0] == "mode"       # then moves on
    assert nav.trail == ["battle", "battle", "mode"]


def test_the_ladder_restarts_for_the_next_match() -> None:
    from cr_bot.vision.gamescreen import LobbyNavigator

    nav = LobbyNavigator(gap_s=0.0)
    while nav.next_tap(0.0) is not None:
        pass
    assert nav.finished
    nav.reset()
    assert nav.next_tap(0.0)[0] == "battle"


def test_ladder_coordinates_are_resolution_independent() -> None:
    from cr_bot.vision.gamescreen import DEFAULT_LOBBY_LADDER, absolute

    assert all(0.0 < s.rel_xy[0] < 1.0 and 0.0 < s.rel_xy[1] < 1.0
               for s in DEFAULT_LOBBY_LADDER)
    assert absolute((0.5, 0.5), 720, 1600) == (360, 800)
    assert absolute((0.5, 0.5), 1080, 2400) == (540, 1200)


# ======================================================================================
# the frame profiler
# ======================================================================================

def test_the_profiler_attributes_time_to_stages() -> None:
    import time as _time

    from cr_bot.perf import CAPTURE, VISION, FrameProfiler

    prof = FrameProfiler()
    for _ in range(5):
        prof.begin_frame()
        with prof.stage(CAPTURE):
            _time.sleep(0.001)
        with prof.stage(VISION):
            _time.sleep(0.004)
        prof.end_frame()
    summary = dict((name, mean) for name, mean, _p95, _share in prof.summary())
    assert summary[VISION] > summary[CAPTURE]
    assert prof.bottleneck() == VISION
    assert "loop" in prof.report() and "vision" in prof.report()


def test_the_profiler_can_be_disabled_without_changing_the_code_path() -> None:
    from cr_bot.perf import VISION, FrameProfiler

    prof = FrameProfiler(enabled=False)
    with prof.stage(VISION):
        pass
    prof.begin_frame()
    prof.end_frame()
    assert prof.frames == 0
    assert prof.summary() == []
