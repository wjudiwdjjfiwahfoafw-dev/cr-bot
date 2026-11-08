"""Measure synthetic/composited scenarios on a real device-frame background.

Why
---
"I see boxes where there are no troops" and "some cards are not detected" are statements about
rates, so they can only be fixed if the rates are measured. This harness builds a repeatable
sequence from the real 720x1280 device frame shipped in the repo:

1. the three troops that are actually in the frame are inpainted away, giving a clean arena floor;
2. the background model is warmed on that artificial clean floor;
3. known sprites are pasted one at a time, with the real arena, real HUD and real calibration,
   and the detector's output is scored against what was pasted.

Reported per configuration: recall, false positives per frame, duplicates, side accuracy and
milliseconds per frame. Run it before and after a change to see whether the change helped.

    python tools/vision_benchmark.py
    python tools/vision_benchmark.py --frames 30 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cr_bot.arena.grid import ArenaCalibration, ArenaGrid  # noqa: E402
from cr_bot.models import Side  # noqa: E402
from cr_bot.vision.unit_detector import BlobUnitDetector, pixels_per_tile  # noqa: E402

FRAME = REPO_ROOT / "calibration_preview.png"
CALIBRATION = REPO_ROOT / "configs" / "calibration" / "RZCT20A0MBJ_720x1600.json"
SPRITES = REPO_ROOT / "assets" / "unit_sprites"

#: troops that really are in the sample frame (x1, y1, x2, y2) in frame pixels
REAL_UNITS = (
    ((68, 486, 236, 626), "giant"),
    ((52, 203, 143, 277), "miner"),
    ((486, 268, 626, 404), "mega_knight"),
)

#: (card, x, y, height_px, tint) — placed on grass that is empty in the real frame
PASTES = (
    ("musketeer", 60, 700, 92, "ally"),
    ("giant", 300, 690, 132, "enemy"),
    ("mega_knight", 470, 640, 150, "enemy"),
    ("bats", 250, 560, 54, "ally"),
    ("minions", 590, 560, 74, "enemy"),
)


def load_frame() -> np.ndarray:
    frame = cv2.imread(str(FRAME))
    if frame is None:
        raise SystemExit(f"cannot read {FRAME}")
    return frame


def clean_floor(frame: np.ndarray) -> np.ndarray:
    """Inpaint the troops that are really in the frame, leaving a plausible empty arena."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    for (x1, y1, x2, y2), _name in REAL_UNITS:
        mask[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 255
    return cv2.inpaint(frame, mask, 7, cv2.INPAINT_TELEA)


def sprite(card: str, index: int = 0) -> np.ndarray:
    path = sorted((SPRITES / card).glob("*.png"))[index]
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def paste(canvas: np.ndarray, spr: np.ndarray, x: int, y: int, height: int,
          tint: str = "neutral") -> np.ndarray:
    h, w = spr.shape[:2]
    scale = height / float(h)
    resized = cv2.resize(spr, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)
    rgb = resized[:, :, :3].astype(np.float32)
    if tint == "enemy":
        rgb = rgb * np.array([0.55, 0.45, 1.0], np.float32) + np.array([30, 10, 70], np.float32)
    elif tint == "ally":
        rgb = rgb * np.array([1.0, 0.7, 0.55], np.float32) + np.array([70, 15, 30], np.float32)
    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
    out = canvas.copy()
    x = max(0, min(out.shape[1] - resized.shape[1], x))
    y = max(0, min(out.shape[0] - resized.shape[0], y))
    roi = out[y:y + resized.shape[0], x:x + resized.shape[1]]
    blended = np.clip(rgb, 0, 255) * alpha + roi.astype(np.float32) * (1 - alpha)
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = blended.astype(np.uint8)
    return out


@dataclass
class GroundTruth:
    card: str
    tile_x: float
    tile_y: float
    side: Side


@dataclass
class Tally:
    frames: int = 0
    truth: int = 0
    hits: int = 0
    false_positives: int = 0
    duplicates: int = 0
    side_correct: int = 0
    side_checked: int = 0
    ms: List[float] = field(default_factory=list)

    def add_frame(self, units, truth: Sequence[GroundTruth], ms: float) -> None:
        self.frames += 1
        self.truth += len(truth)
        self.ms.append(ms)
        used = set()
        matched_pairs: Dict[int, List[int]] = {}
        for gi, gt in enumerate(truth):
            best, best_d = None, 2.0
            for ui, u in enumerate(units):
                if ui in used:
                    continue
                d = float(np.hypot(u.tile_x - gt.tile_x, u.tile_y - gt.tile_y))
                if d < best_d:
                    best, best_d = ui, d
            if best is not None:
                used.add(best)
                self.hits += 1
                matched_pairs.setdefault(gi, []).append(best)
                if units[best].side is gt.side:
                    self.side_correct += 1
                self.side_checked += 1
                # any *extra* box on the same troop is a duplicate / split
                for ui, u in enumerate(units):
                    if ui in used:
                        continue
                    if np.hypot(u.tile_x - gt.tile_x, u.tile_y - gt.tile_y) < 1.2:
                        self.duplicates += 1
                        used.add(ui)
        self.false_positives += len(units) - len(used)

    # -- reporting
    @property
    def recall(self) -> float:
        return self.hits / self.truth if self.truth else 0.0

    @property
    def fp_per_frame(self) -> float:
        return self.false_positives / self.frames if self.frames else 0.0

    @property
    def dup_per_frame(self) -> float:
        return self.duplicates / self.frames if self.frames else 0.0

    @property
    def side_accuracy(self) -> float:
        return self.side_correct / self.side_checked if self.side_checked else 0.0

    @property
    def mean_ms(self) -> float:
        return sum(self.ms) / len(self.ms) if self.ms else 0.0

    def row(self, label: str) -> str:
        return (
            f"{label:22s} recall {self.recall * 100:5.1f}%  "
            f"FP/frame {self.fp_per_frame:4.2f}  dup/frame {self.dup_per_frame:4.2f}  "
            f"side {self.side_accuracy * 100:5.1f}%  {self.mean_ms:5.1f} ms"
        )

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "recall": round(self.recall, 4),
            "fp_per_frame": round(self.fp_per_frame, 4),
            "dup_per_frame": round(self.dup_per_frame, 4),
            "side_accuracy": round(self.side_accuracy, 4),
            "mean_ms": round(self.mean_ms, 2),
        }


def run(detector_cls=BlobUnitDetector, frames: int = 12, quiet: bool = False, **det_kwargs) -> Tally:
    frame = load_frame()
    clean = clean_floor(frame)
    height, width = clean.shape[:2]

    grid = _grid(width, height)
    det = detector_cls(**det_kwargs)
    for _ in range(4):                      # warm the background on the clean floor
        det.detect(clean, grid)

    tally = Tally()
    # each "frame" advances every troop a little, which is what really happens in a match
    for f in range(frames):
        scene = clean
        truth: List[GroundTruth] = []
        for card, x, y, h, tint in PASTES:
            xf = x + (f % 4)          # a couple of pixels of drift per frame
            yf = y + (f % 3)
            scene = paste(scene, sprite(card), xf, yf, h, tint=tint)
            tile_x, tile_y = grid.pixel_to_tile(xf + h * 0.35, yf + h)
            truth.append(GroundTruth(card, tile_x, tile_y, Side.ALLY if tint == "ally" else Side.ENEMY))
        t0 = time.perf_counter()
        units = det.detect(scene, grid)
        ms = 1000.0 * (time.perf_counter() - t0)
        tally.add_frame(units, truth, ms)
        if not quiet and f == frames - 1:
            for u in units:
                print(f"    {u.fallback_label:16s} {u.side.value:5s} conf={u.confidence:.2f} "
                      f"t=({u.tile_x:4.1f},{u.tile_y:4.1f}) src={u.source}")
    return tally


def run_real_frame(detector_cls=BlobUnitDetector, frames: int = 6, quiet: bool = False,
                   **det_kwargs) -> Tally:
    """Warm up on the empty floor, then show the *original* frame (troops + structures appear).

    This is an inpaint-and-reveal stress test, not a real temporal sequence or live recall test.
    """
    frame = load_frame()
    clean = clean_floor(frame)
    height, width = clean.shape[:2]
    grid = _grid(width, height)
    det = detector_cls(**det_kwargs)
    for _ in range(4):
        det.detect(clean, grid)
    truth = []
    for (x1, y1, x2, y2), card in REAL_UNITS:
        tx, ty = grid.pixel_to_tile(0.5 * (x1 + x2), y2)
        truth.append(GroundTruth(card, tx, ty, Side.ENEMY if card == "giant" else Side.ALLY))

    tally = Tally()
    for f in range(frames):
        t0 = time.perf_counter()
        units = det.detect(frame, grid)
        tally.add_frame(units, truth, 1000.0 * (time.perf_counter() - t0))
        if not quiet and f == frames - 1:
            for u in units:
                print(f"    {u.fallback_label:16s} {u.side.value:5s} conf={u.confidence:.2f} "
                      f"t=({u.tile_x:4.1f},{u.tile_y:4.1f}) src={u.source}")
    return tally


def _grid(width: int, height: int):
    raw = json.loads(CALIBRATION.read_text())
    from cr_bot.arena.projection import projection_from_dict

    calibration = ArenaCalibration(
        frame_size=(width, height),
        projection=projection_from_dict(raw["projection"]),
        hand_slots_px=[tuple(p) for p in raw["hand_slots_px"]],
        hand_card_size_px=tuple(raw["hand_card_size_px"]),
        elixir_px=tuple(raw["elixir_px"]),
        river_px_y=float(raw["river_px_y"]),
    )
    grid = ArenaGrid(width, height, calibration)
    grid.resize(width, height)
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--scenario", choices=("pastes", "real", "both"), default="both")
    args = ap.parse_args()

    frame = load_frame()
    print(f"benchmark: {FRAME.name} {frame.shape[1]}x{frame.shape[0]}, {args.frames} frames")
    out = {}
    if args.scenario in ("pastes", "both"):
        out["pastes"] = run(frames=args.frames, quiet=args.quiet)
        print(out["pastes"].row("pasted troops"))
    if args.scenario in ("real", "both"):
        out["real"] = run_real_frame(frames=args.frames, quiet=args.quiet)
        print(out["real"].row("real frame"))
        print("  (single still revealed after artificial inpainting; not live-stream recall)")
    if args.json:
        args.json.write_text(json.dumps({k: v.as_dict() for k, v in out.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
