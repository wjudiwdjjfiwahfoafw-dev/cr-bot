"""Run the whole vision pipeline on a real screenshot and draw what it found.

The repo's unit tests synthesise frames (real sprites pasted onto a real arena), which is the
right way to test geometry and thresholds, but it cannot answer "does this work on a screenshot
someone took on a phone". This tool does: it calibrates the frame the same way the bot does,
runs detection, prints every unit, and writes an annotated PNG so the result can be *seen* as
well as measured.

    python tools/real_frame_check.py tests/fixtures/real_frames/*.png
    python tools/real_frame_check.py shot.png --expect "giant@3.2,13.6:enemy" --expect "..."

``--expect`` entries use frame-tile coordinates, so a test can assert location and side.
The name is only a human-readable label; this tool does not check the predicted card name:

    name@tile_x,tile_y:side[:tolerance]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cr_bot.arena.device_calibrator import DeviceCalibrator  # noqa: E402
from cr_bot.arena.grid import ArenaGrid  # noqa: E402
from cr_bot.vision.unit_detector import pixels_per_tile  # noqa: E402


@dataclass
class Expectation:
    name: str
    tile_x: float
    tile_y: float
    side: Optional[str] = None
    tolerance: float = 2.0

    @classmethod
    def parse(cls, text: str) -> "Expectation":
        head, *rest = text.split(":")
        name, coords = head.split("@")
        x, y = coords.split(",")
        side = rest[0] if rest else None
        tol = float(rest[1]) if len(rest) > 1 else 2.0
        return cls(name=name, tile_x=float(x), tile_y=float(y), side=side, tolerance=tol)


@dataclass
class FrameReport:
    path: Path
    frame_shape: Tuple[int, int]
    calibrated: bool
    evaluated: bool = False  # true only when explicit expectations were provided
    note: str = ""
    units: List[dict] = field(default_factory=list)
    matched: int = 0
    missed: List[str] = field(default_factory=list)
    false_positives: List[str] = field(default_factory=list)
    side_wrong: List[str] = field(default_factory=list)
    duplicates: int = 0
    streak: int = 0          # identical detections over 3 consecutive frames
    ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.calibrated and self.evaluated and not (
            self.missed or self.false_positives or self.side_wrong or self.duplicates)


def run_frame(
    path: Path,
    expectations: Sequence[Expectation] = (),
    warm_frames: int = 6,
    annotated_path: Optional[Path] = None,
) -> FrameReport:
    """Calibrate, warm up, detect — exactly like the runner does at the start of a match."""
    from cr_bot.vision.arena_detector import ArenaUnitBackend

    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"cannot read {path}")
    h, w = frame.shape[:2]
    report = FrameReport(path=path, frame_shape=(w, h), calibrated=False,
                         evaluated=bool(expectations))

    try:
        result = DeviceCalibrator().calibrate(frame)
    except Exception as exc:                              # noqa: BLE001
        report.note = f"calibration failed: {type(exc).__name__}: {exc}"
        return report
    calibration = getattr(result, "calibration", None)
    if calibration is None:
        report.note = f"calibration failed: {getattr(result, 'notes', result)}"
        return report
    report.calibrated = True
    report.note = ("Single still with artificial jitter: no real prior background; "
                   "not a recall or real-sequence measurement.")
    grid = ArenaGrid(w, h, calibration)
    grid.resize(w, h)

    backend = ArenaUnitBackend()
    # A single still cannot supply a real prior background or a real sequence. Use small,
    # fixed-size translations only as a noise/stability stress test, not as evidence of recall.
    # Cropping to different dimensions resets the model on every frame and creates phantoms.
    warm_crops = [_jitter(frame, i) for i in range(warm_frames)]
    for crop in warm_crops:
        grid_c = ArenaGrid(crop.shape[1], crop.shape[0], calibration)
        grid_c.resize(crop.shape[1], crop.shape[0])
        backend.detect(crop, grid_c)
    import time
    trials = []
    for i in range(3):
        crop = _jitter(frame, 100 + i)
        grid_c = ArenaGrid(crop.shape[1], crop.shape[0], calibration)
        grid_c.resize(crop.shape[1], crop.shape[0])
        t0 = time.perf_counter()
        trials.append(backend.detect(crop, grid_c))
        report.ms += 1000.0 * (time.perf_counter() - t0)
    report.ms /= 3.0
    keys = [sorted((round(d.tile_x, 1), round(d.tile_y, 1), d.side.value) for d in got)
            for got in trials]
    report.streak = 3 if all(k == keys[0] for k in keys) else 0
    detections = trials[0]

    report.units = [
        {"name": d.name, "side": d.side.value, "tile": (round(d.tile_x, 2), round(d.tile_y, 2)),
         "conf": d.confidence, "source": d.source}
        for d in detections
    ]
    # pairwise duplicates
    for i, a in enumerate(detections):
        for b in detections[i + 1:]:
            if np.hypot(a.tile_x - b.tile_x, a.tile_y - b.tile_y) < 0.7:
                report.duplicates += 1

    used = set()
    for exp in expectations:
        best, best_d = None, 1e9
        for index, det in enumerate(detections):
            if index in used:
                continue
            d = float(np.hypot(det.tile_x - exp.tile_x, det.tile_y - exp.tile_y))
            if d < best_d:
                best, best_d = index, d
        if best is not None and best_d <= exp.tolerance:
            used.add(best)
            report.matched += 1
            if exp.side and detections[best].side.value != exp.side:
                report.side_wrong.append(f"{exp.name} ({detections[best].side.value})")
        else:
            report.missed.append(exp.name)
    for index, det in enumerate(detections):
        if expectations and index not in used:
            report.false_positives.append(
                f"({det.tile_x:.1f},{det.tile_y:.1f}) {det.side.value} conf={det.confidence:.2f}")

    if annotated_path is not None:
        draw(frame, grid, detections, expectations, annotated_path)
    return report


def _jitter(frame: np.ndarray, seed: int, amount: int = 2) -> np.ndarray:
    """Translate at constant dimensions; a size-changing crop resets the background model."""
    dx = int((seed * 7) % (2 * amount + 1)) - amount
    dy = int((seed * 13) % (2 * amount + 1)) - amount
    h, w = frame.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def draw(frame: np.ndarray, grid: ArenaGrid, detections, expectations,
         out_path: Path) -> None:
    canvas = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.pixel_bbox
        colour = (0, 0, 255) if det.side.value == "enemy" else (255, 128, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(canvas, f"{det.name}", (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{det.tile_x:.1f},{det.tile_y:.1f}", (x1, min(canvas.shape[0] - 3, y2 + 11)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1, cv2.LINE_AA)
    for exp in expectations:
        px, py = grid.tile_to_pixel(exp.tile_x, exp.tile_y)
        cv2.drawMarker(canvas, (int(px), int(py)), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
        cv2.putText(canvas, exp.name, (int(px) + 8, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def format_report(report: FrameReport) -> str:
    lines = [f"{report.path.name}  {report.frame_shape[0]}x{report.frame_shape[1]}  "
             f"calibrated={report.calibrated}  {report.ms:.0f} ms/frame"]
    if report.note:
        lines.append(f"    {report.note}")
    for unit in report.units:
        lines.append(f"    {unit['name']:14s} {unit['side']:5s} "
                     f"({unit['tile'][0]:5.1f},{unit['tile'][1]:5.1f}) "
                     f"conf={unit['conf']:.2f} {unit['source']}")
    if report.evaluated:
        lines.append(
            f"    -> matched {report.matched}, missed {report.missed or '-'}, "
            f"extra {report.false_positives or '-'}, wrong side {report.side_wrong or '-'}, "
            f"duplicates {report.duplicates}, stable {report.streak}/3"
        )
    else:
        lines.append(f"    -> no labels: false positives and misses not evaluated; "
                     f"duplicates {report.duplicates}, stable {report.streak}/3")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--expect", action="append", default=[],
                    help="name@x,y:side[:tolerance]")
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None, help="directory for annotated copies")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    expectations = [Expectation.parse(e) for e in args.expect]
    reports = []
    for path in args.images:
        out = None
        if args.out is not None:
            out = args.out / f"{path.stem}_detected.png"
        report = run_frame(path, expectations, warm_frames=args.warm, annotated_path=out)
        print(format_report(report))
        print()
        reports.append(report)
    if args.json:
        args.json.write_text(json.dumps([
            {"file": r.path.name, "calibrated": r.calibrated, "units": r.units,
             "evaluated": r.evaluated,
             "matched": r.matched if r.evaluated else None,
             "missed": r.missed if r.evaluated else None,
             "false_positives": r.false_positives if r.evaluated else None,
             "side_wrong": r.side_wrong if r.evaluated else None,
             "duplicates": r.duplicates, "stable": r.streak,
             "ms": round(r.ms, 1), "note": r.note}
            for r in reports], indent=2))
    failed = [r for r in reports if not r.calibrated or (expectations and not r.ok)]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
