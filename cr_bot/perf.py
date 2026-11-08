"""Frame-loop timing: find out where the milliseconds actually go before changing anything.

Why this exists
---------------
"the fps is low" is not an actionable statement — on a real session the loop can be limited by
capture (a slow backend), by vision (segmentation on a big frame), by the HUD (hstack + imshow +
waitKey), or simply by the configured loop rate. Guessing which one it is leads to optimising the
wrong thing, which is why this module measures each stage and reports shares.

It is deliberately tiny and allocation-free in the steady state: a fixed list of stage timers, a
ring buffer of samples, and a summary that can be formatted into a single log line.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterator, List, Optional, Tuple

#: stage names used by the runner; kept as constants so the log stays greppable
CAPTURE = "capture"
VISION = "vision"
BRAIN = "brain"
DEPLOY = "deploy"
HUD = "hud"
SLEEP = "sleep"
TOTAL = "total"

STAGE_ORDER = (CAPTURE, VISION, BRAIN, HUD, SLEEP, TOTAL)


@dataclass
class StageStats:
    name: str
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=240))

    def add(self, seconds: float) -> None:
        self.samples.append(max(0.0, seconds))

    @property
    def mean_ms(self) -> float:
        return 1000.0 * sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        return 1000.0 * ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    @property
    def max_ms(self) -> float:
        return 1000.0 * max(self.samples) if self.samples else 0.0

    def share(self, total: float) -> float:
        return (self.mean_ms / total) if total > 0 else 0.0


class FrameProfiler:
    """Accumulates per-stage timings for the main loop.

    Typical use::

        prof = FrameProfiler()
        while running:
            with prof.stage(CAPTURE):
                frame = capture()
            with prof.stage(VISION):
                state = process(frame)
            prof.end_frame()

    ``stage()`` is a no-op context manager when profiling is off, so it can be sprinkled through
    the hot path without a runtime cost worth measuring.
    """

    def __init__(self, enabled: bool = True, window: int = 240) -> None:
        self.enabled = enabled
        self._stages: Dict[str, StageStats] = {
            name: StageStats(name=name, samples=deque(maxlen=window))
            for name in STAGE_ORDER
        }
        self._frame_start: Optional[float] = None
        self._frames = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stats(name).add(time.perf_counter() - start)

    def add(self, name: str, seconds: float) -> None:
        """Record a stage that is timed by hand (e.g. a block that also pumps the GUI queue)."""
        if self.enabled:
            self._stats(name).add(seconds)

    def begin_frame(self) -> None:
        if self.enabled:
            self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        if not self.enabled:
            return
        self._frames += 1
        if self._frame_start is not None:
            self._stats(TOTAL).add(time.perf_counter() - self._frame_start)
            self._frame_start = None

    def _stats(self, name: str) -> StageStats:
        stats = self._stages.get(name)
        if stats is None:
            stats = StageStats(name=name)
            self._stages[name] = stats
        return stats

    # -- reporting -----------------------------------------------------------------
    @property
    def frames(self) -> int:
        return self._frames

    def summary(self) -> List[Tuple[str, float, float, float]]:
        """``[(stage, mean_ms, p95_ms, share_of_total)]`` ordered by cost, most expensive first."""
        total = self._stages[TOTAL].mean_ms
        rows = [
            (name, stats.mean_ms, stats.p95_ms, stats.share(total))
            for name, stats in self._stages.items()
            if name != TOTAL and stats.samples
        ]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows

    def report(self) -> str:
        """One-line summary, e.g. ``loop 18.2ms (55fps) | vision 11.4ms 63% | capture 3.1ms 17%``."""
        total = self._stages[TOTAL]
        if not total.samples:
            return "no frames measured"
        fps = 1000.0 / total.mean_ms if total.mean_ms > 0 else 0.0
        parts = [
            f"{name} {mean:.1f}ms ({share * 100:.0f}%)"
            for name, mean, _p95, share in self.summary()
        ]
        return f"loop {total.mean_ms:.1f}ms ({fps:.0f} fps) | " + " | ".join(parts)

    def bottleneck(self) -> Optional[str]:
        rows = self.summary()
        return rows[0][0] if rows else None

    def reset(self) -> None:
        for stats in self._stages.values():
            stats.samples.clear()
        self._frames = 0
        self._frame_start = None
