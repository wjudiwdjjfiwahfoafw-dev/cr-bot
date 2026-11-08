"""Phase hysteresis and lobby navigation.

Two problems that only show up on a real phone, and that are both about *timing* rather than
about colour thresholds:

**Phase flapping.**
    :class:`~cr_bot.vision.game_state.GameStateDetector` answers "what is on screen right now"
    from narrow colour bands in fixed screen regions. On a real device a single frame regularly
    contradicts its neighbours: a spell flash crosses the elixir tray, the end-of-match dimming
    starts, the encoder returns a frame caught mid-transition, the game shows a "connection"
    overlay. Consuming that answer directly made a match end eight seconds after it started::

        === Match started: resetting trackers ===
        === Match ended (unknown) ===          <- nothing had ended
        === Match started: resetting trackers ===

    Every tracker was wiped, ``matches_played`` was incremented, and with ``--auto-queue`` the
    bot started tapping lobby buttons *during* the match. A battle is a long-lived state, so it
    needs more than one frame to leave, while entering one needs no delay at all (delay there is
    just lost match time).

**Lobby navigation.**
    ``--auto-queue`` tapped the Battle button and then waited for a battle to appear. Between the
    two the game usually shows a mode/arena selection screen (and, sometimes, a "your previous
    team" screen), so the queue stalled until something else moved it along. Queueing therefore
    walks a small ladder of taps, each verified against the frames the caller saw, instead of
    assuming one tap is enough.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from cr_bot.models import GamePhase

logger = logging.getLogger(__name__)


# =============================================================================================
# phase hysteresis
# =============================================================================================
class PhaseStabilizer:
    """Turn a per-frame phase guess into a stable phase for the bot to act on.

    Rules, all of them deliberate:

    * a battle sub-phase (1x/2x/3x) is accepted **immediately** — no lost match time;
    * leaving a battle requires the incoming phase to be a non-battle phase, *stable* (the same
      family as the previous frames) and held for ``battle_exit_seconds``. Then it is reported;
    * inside a battle, a stray non-battle phase is reported as the battle sub-phase it replaced,
      so the loop keeps behaving as if the match is running;
    * outside a battle, ``MATCH_LOADING`` is treated as "a match is starting" and reported as a
      battle, because the loader is the last screen before the arena appears and the bot can
      already be doing useful work (calibration) during it.
    """

    #: consecutive frames of the same non-battle family needed to leave a battle
    EXIT_CONFIRM_FRAMES = 2

    def __init__(self, battle_exit_seconds: float = 1.0) -> None:
        self.battle_exit_seconds = float(battle_exit_seconds)
        self._phase = GamePhase.UNKNOWN
        self._in_battle = False
        self._last_battle_phase = GamePhase.BATTLE_1X
        self._candidate: Optional[GamePhase] = None
        self._candidate_since = 0.0
        self._candidate_frames = 0
        #: counted so the log can explain *why* a match was considered over
        self.exit_latched = 0

    # -- state -----------------------------------------------------------------------
    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def in_battle(self) -> bool:
        return self._in_battle

    def reset_match(self) -> None:
        self._in_battle = False
        self._phase = GamePhase.UNKNOWN
        self._candidate = None
        self._candidate_frames = 0

    # -- main entry point ------------------------------------------------------------
    def update(self, raw: GamePhase, now_ts: float) -> GamePhase:
        """Feed one frame's raw guess; return the phase the bot should act on."""
        if raw.is_in_battle or raw == GamePhase.MATCH_LOADING:
            if raw.is_in_battle:
                self._last_battle_phase = raw
            if not self._in_battle:
                logger.debug("phase: entering battle on %s", raw.value)
            self._in_battle = True
            self._candidate = None
            self._candidate_frames = 0
            self._phase = raw if raw.is_in_battle else self._last_battle_phase
            return self._phase

        # a non-battle phase: either we are already out, or we are being asked to leave
        if not self._in_battle:
            self._phase = raw
            return self._phase

        if raw is not self._candidate:
            self._candidate = raw
            self._candidate_since = now_ts
            self._candidate_frames = 1
        else:
            self._candidate_frames += 1

        held = now_ts - self._candidate_since
        if self._candidate_frames >= self.EXIT_CONFIRM_FRAMES and held >= self.battle_exit_seconds:
            logger.debug("phase: match confirmed over on %s (held %.2fs)", raw.value, held)
            self._in_battle = False
            self._phase = raw
            self.exit_latched += 1
            self._candidate = None
            self._candidate_frames = 0
            return self._phase

        # not confirmed: keep the match alive
        self._phase = self._last_battle_phase
        return self._phase


# =============================================================================================
# lobby navigation
# =============================================================================================
@dataclass
class LobbyStep:
    """One tap on the way from "match over" to "in a battle"."""

    name: str
    #: relative position on the screen (0..1), independent of resolution
    rel_xy: Tuple[float, float]
    #: how long to wait after this tap before the next one
    wait_s: float = 1.2


#: The lobby ladder. Positions are *relative* so they work at 720x1600 and 1080x2400 alike, and
#: each one is a standard Clash Royale button: the Battle label, the primary mode tile
#: ("Rage / Ladder / Party" row's default entry) and the mode screen's own Play button.
DEFAULT_LOBBY_LADDER: Tuple[LobbyStep, ...] = (
    LobbyStep("battle", (0.50, 0.73), wait_s=1.4),
    LobbyStep("mode", (0.50, 0.46), wait_s=1.2),      # mode/arena selection tile
    LobbyStep("play", (0.50, 0.86), wait_s=1.0),      # the big coloured Play button
)


@dataclass
class LobbyNavigator:
    """Walks the request → battle ladder, tapping at most one step per call.

    The caller drives it from the main loop (so a slow frame or a tap in flight never blocks the
    loop), and it only advances when it has actually been asked to. A step is retried a bounded
    number of times: if the game did not react (a popup, a slow network), tapping the same spot
    forever would be worse than giving up and letting the next match-over restart the ladder.
    """

    ladder: Sequence[LobbyStep] = DEFAULT_LOBBY_LADDER
    #: minimum seconds between two taps
    gap_s: float = 1.0
    #: how many times a single step may be tapped before moving on
    max_attempts_per_step: int = 3

    _index: int = 0
    _attempts: int = 0
    _last_tap_ts: float = -1e9
    _started_ts: float = -1e9
    _log: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self._index = 0
        self._attempts = 0
        self._last_tap_ts = -1e9

    @property
    def finished(self) -> bool:
        return self._index >= len(self.ladder)

    @property
    def current(self) -> Optional[LobbyStep]:
        return None if self.finished else self.ladder[self._index]

    @property
    def trail(self) -> List[str]:
        return list(self._log)

    def next_tap(self, now_ts: float) -> Optional[Tuple[str, Tuple[float, float]]]:
        """Return ``(step_name, rel_xy)`` for the tap to send now, or ``None`` to wait.

        Fractional coordinates are returned on purpose: the caller owns the screen size.
        """
        if self.finished:
            return None
        if now_ts - self._last_tap_ts < self.gap_s:
            return None
        step = self.ladder[self._index]
        self._last_tap_ts = now_ts
        self._attempts += 1
        self._log.append(step.name)
        if self._attempts >= self.max_attempts_per_step:
            self._index += 1
            self._attempts = 0
        return step.name, step.rel_xy

    def note_progress(self) -> None:
        """Called when the screen changed (any non-battle screen other than the last one)."""
        self._attempts = 0


def absolute(rel_xy: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    return int(round(rel_xy[0] * width)), int(round(rel_xy[1] * height))
