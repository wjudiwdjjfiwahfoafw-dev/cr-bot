"""Score the vision pipeline against labelled, real, in-game screenshots.

``tools/real_frame_check.py`` answers "what does the bot see here"; this answers "is it *right*",
using ground truth a human wrote down (see ``samples/README.md``). The distinction matters for the
things that make a detector unusable rather than merely weak:

* **misses** - a troop that is on the board and was not reported;
* **phantoms** - a box on something that is not a troop (the "detected two dozen troops" failure);
* **duplicates** - the same troop reported twice;
* **wrong side** - an ally called an enemy, which makes the brain cast at its own tower;
* **flicker** - the same troop appearing/disappearing/teleporting between consecutive frames.

Naming is reported separately: exact card agreement, wrong names and abstentions are distinguished.
A missing name is not counted as a correct prediction.

    python tools/verify_screenshots.py samples/*.png
    python tools/verify_screenshots.py samples/*.png --annotate samples/annotated --json out.json
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

from cr_bot.arena.device_calibrator import DeviceCalibrator      # noqa: E402
from cr_bot.arena.grid import ArenaGrid                          # noqa: E402
from cr_bot.vision.arena_detector import ArenaUnitBackend        # noqa: E402
from cr_bot.vision.unit_recognizer import UnitRecognizer         # noqa: E402

#: a detection this close (tiles) to a labelled troop is that troop
MATCH_RADIUS_TILES = 1.5
#: two detections this close to each other are the same troop reported twice
DUPLICATE_RADIUS_TILES = 0.9
#: a troop that moves more than this between consecutive frames is a flicker, not a walk
MAX_STEP_TILES = 3.5


@dataclass
class Label:
    box: Tuple[int, int, int, int]
    card: Optional[str] = None
    side: Optional[str] = None
    partial: bool = False
    note: str = ""


@dataclass
class FrameScore:
    image: Path
    labels: List[Label]
    detections: List[dict]
    matched: List[Tuple[Label, dict]] = field(default_factory=list)
    missed: List[Label] = field(default_factory=list)
    phantoms: List[dict] = field(default_factory=list)
    duplicates: int = 0
    wrong_side: List[Tuple[Label, dict]] = field(default_factory=list)
    names_right: int = 0
    names_wrong: List[str] = field(default_factory=list)
    names_unnamed: List[str] = field(default_factory=list)
    ms: float = 0.0
    calibrated: bool = False
    note: str = ""
    scenario: str = "arrival"

    def summary_line(self) -> str:
        flags = []
        if self.missed:
            flags.append(f"{len(self.missed)} missed")
        if self.phantoms:
            flags.append(f"{len(self.phantoms)} phantom")
        if self.duplicates:
            flags.append(f"{self.duplicates} dup")
        if self.wrong_side:
            flags.append(f"{len(self.wrong_side)} wrong-side")
        return "clean" if not flags else ", ".join(flags)


def load_labels(image_path: Path) -> List[Label]:
    sidecar = image_path.with_suffix(".json")
    if not sidecar.exists():
        raise SystemExit(
            f"no labels for {image_path.name}: write {sidecar.name} (see samples/README.md)")
    data = json.loads(sidecar.read_text())
    labels = []
    for index, entry in enumerate(data.get("expect", [])):
        box = entry.get("box")
        if not box or len(box) != 4:
            raise SystemExit(f"{sidecar.name}: entry {index} has no 4-value 'box'")
        labels.append(Label(
            box=tuple(int(v) for v in box),
            card=entry.get("card"),
            side=entry.get("side"),
            partial=bool(entry.get("partial", False)),
            note=str(entry.get("note", "")),
        ))
    return labels


def _inside(det: dict, label: Label) -> float:
    """IoU-ish agreement between a detection's pixel box and a labelled box."""
    ax1, ay1, ax2, ay2 = det["box"]
    bx1, by1, bx2, by2 = label.box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(1.0, float(union))


def score_frame(image_path: Path, warm_frames: int = 5, recognizer: Optional[UnitRecognizer] = None,
                annotate: Optional[Path] = None, scenario: str = "arrival",
                enable_recognition: bool = True) -> FrameScore:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise SystemExit(f"cannot read {image_path}")
    labels = load_labels(image_path)
    h, w = frame.shape[:2]

    result = DeviceCalibrator().calibrate(frame)
    calibration = getattr(result, "calibration", None)
    score = FrameScore(image=image_path, labels=labels, detections=[], scenario=scenario,
                       calibrated=calibration is not None)
    if calibration is None:
        score.note = f"calibration failed (confidence {getattr(result, 'confidence', 0):.2f})"
        return score
    if getattr(result, "confidence", 1.0) < 0.55:
        score.note = f"low calibration confidence {result.confidence:.2f}"
    grid = ArenaGrid(w, h, calibration)
    grid.resize(w, h)

    # A background-subtraction detector needs to have seen the arena *before* the troops were
    # placed, and a single screenshot has no such history. Feeding the frame itself makes the
    # troops part of the floor and the arena reads as empty, which is not a fair test of the
    # detector (and warm-up on identical frames is not a valid scenario either). So the arena is
    # reconstructed by inpainting the labelled sprites out: "this is the empty board, now the
    # troops arrive". This is an artificial scenario, not the true prior match history; it uses
    # only the labels, never the detector's own answer.
    scenario = scenario or "arrival"
    empty = _empty_board(frame, labels) if scenario == "arrival" else frame
    backend = ArenaUnitBackend(recognizer=recognizer, enable_recognition=enable_recognition)
    # Exclude one-time asset loading from the frame timing, as it can happen at startup.
    if enable_recognition and recognizer is not None:
        _ = recognizer.is_ready
    for i in range(warm_frames):
        crop, _dx, _dy = _crop(empty, i)
        backend.detect(crop, _grid_for(crop, calibration))
    crop, dx, dy = _crop(empty, warm_frames)
    backend.detect(crop, _grid_for(crop, calibration))
    crop, dx, dy = _crop(frame, warm_frames)
    t0 = time.perf_counter()
    detections = backend.detect(crop, _grid_for(crop, calibration))
    score.ms = 1000.0 * (time.perf_counter() - t0)

    for det in detections:
        x1, y1, x2, y2 = det.pixel_bbox
        score.detections.append({
            "name": det.name, "side": det.side.value, "source": det.source,
            "confidence": round(float(det.confidence), 3),
            "tile": [round(det.tile_x, 2), round(det.tile_y, 2)],
            "box": [int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)],
        })

    # --- match detections to labels (greedy, nearest first) -------------------------
    pairs = []
    for li, label in enumerate(labels):
        cx = 0.5 * (label.box[0] + label.box[2])
        cy = 0.5 * (label.box[1] + label.box[3])
        for di, det in enumerate(score.detections):
            distance_tiles = float(np.hypot(det["tile"][0] - _tile_of(grid, cx, cy)[0],
                                            det["tile"][1] - _tile_of(grid, cx, cy)[1]))
            pairs.append((distance_tiles, li, di, _inside(det, label)))
    pairs.sort(key=lambda p: (p[0] > MATCH_RADIUS_TILES, p[0], -p[3]))
    used_labels, used_dets = set(), set()
    for distance, li, di, iou in pairs:
        if li in used_labels or di in used_dets:
            continue
        # accept on proximity, or on a decent box overlap (a big unit's centre and its sprite
        # centre can differ by a tile and a half)
        if distance > MATCH_RADIUS_TILES and iou < 0.15:
            continue
        used_labels.add(li)
        used_dets.add(di)
        score.matched.append((labels[li], score.detections[di]))
    score.missed = [lab for i, lab in enumerate(labels) if i not in used_labels]
    score.phantoms = [d for i, d in enumerate(score.detections) if i not in used_dets]

    for label, det in score.matched:
        if label.side and det["side"] != label.side:
            score.wrong_side.append((label, det))
        if not label.card:
            continue
        named = det["name"]
        if named.startswith("unknown_"):
            score.names_unnamed.append(label.card)
        elif named == label.card:
            score.names_right += 1
        else:
            score.names_wrong.append(f"{label.card} -> {named}")

    for i, a in enumerate(score.detections):
        for b in score.detections[i + 1:]:
            if float(np.hypot(a["tile"][0] - b["tile"][0], a["tile"][1] - b["tile"][1])) \
                    < DUPLICATE_RADIUS_TILES:
                score.duplicates += 1

    if annotate is not None:
        _draw(frame, grid, score, annotate)
    return score


def _empty_board(frame: np.ndarray, labels: Sequence[Label]) -> np.ndarray:
    """The same arena with the labelled troops painted out (see the comment in score_frame)."""
    if not labels:
        return frame
    mask = np.zeros(frame.shape[:2], np.uint8)
    for label in labels:
        x1, y1, x2, y2 = label.box
        pad = 3
        cv2.rectangle(mask, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 255, -1)
    return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)


def _tile_of(grid: ArenaGrid, px: float, py: float) -> Tuple[float, float]:
    return grid.pixel_to_tile(px, py)


def _crop(frame: np.ndarray, seed: int, amount: int = 2) -> Tuple[np.ndarray, int, int]:
    """Use the original fixed-size image; a still screenshot does not contain real motion."""
    return frame, 0, 0


def _grid_for(frame: np.ndarray, calibration) -> ArenaGrid:
    h, w = frame.shape[:2]
    grid = ArenaGrid(w, h, calibration)
    grid.resize(w, h)
    return grid


def _draw(frame: np.ndarray, grid: ArenaGrid, score: FrameScore, out_path: Path) -> None:
    canvas = frame.copy()
    for label in score.labels:
        x1, y1, x2, y2 = label.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 200, 60), 1)
        cv2.putText(canvas, f"gt:{label.card or '?'}", (x1, max(9, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (60, 200, 60), 1, cv2.LINE_AA)
    for det in score.detections:
        x1, y1, x2, y2 = det["box"]
        colour = (0, 0, 255) if det["side"] == "enemy" else (255, 120, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(canvas, det["name"], (x1, min(canvas.shape[0] - 2, y2 + 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def stability(scores: Sequence[FrameScore]) -> List[str]:
    """Frame-to-frame consistency for a sequence of captures of the same battle."""
    problems: List[str] = []
    previous: Optional[List[dict]] = None
    for score in scores:
        current = score.detections
        if previous is None:
            previous = current
            continue
        for det in current:
            best_d = 1e9
            for old in previous:
                d = float(np.hypot(det["tile"][0] - old["tile"][0], det["tile"][1] - old["tile"][1]))
                if d < best_d:
                    best_d = d
            if best_d > MAX_STEP_TILES:
                problems.append(f"{score.image.name}: {det['name']} at {det['tile']} has no "
                                f"match in the previous frame (appeared out of nowhere)")
        previous = current
    return problems


def format_score(score: FrameScore) -> str:
    lines = [f"{score.image.name}: {len(score.labels)} labelled, {len(score.detections)} detected, "
             f"{score.ms:.0f} ms -> {score.summary_line()}"]
    if not score.calibrated:
        lines.append("    NOT CALIBRATED - the arena could not be located in this screenshot")
    if score.note:
        lines.append(f"    note: {score.note}")
    for label in score.missed:
        lines.append(f"    MISSED    {label.card or '?'} {label.side or ''} box={list(label.box)}"
                     + (" (partial)" if label.partial else ""))
    for det in score.phantoms:
        lines.append(f"    PHANTOM   {det['name']} {det['side']} at {det['tile']} "
                     f"conf={det['confidence']} src={det['source']}")
    for label, det in score.wrong_side:
        lines.append(f"    WRONG SIDE {label.card or '?'} labelled {label.side}, "
                     f"reported {det['side']} at {det['tile']}")
    for wrong in score.names_wrong:
        lines.append(f"    MISNAMED  {wrong}")
    for unnamed in score.names_unnamed:
        lines.append(f"    UNNAMED   {unnamed} (detected but not identified)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--annotate", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-naming", action="store_true", help="skip sprite recognition (faster)")
    args = ap.parse_args()

    recognizer = None
    if not args.no_naming:
        recognizer = UnitRecognizer()
    scores = []
    for path in args.images:
        annotate = (args.annotate / f"{path.stem}_scored.png") if args.annotate else None
        score = score_frame(path, warm_frames=args.warm, recognizer=recognizer,
                            annotate=annotate, enable_recognition=not args.no_naming)
        print(format_score(score))
        print()
        scores.append(score)

    total_labels = sum(len(s.labels) for s in scores)
    total_matched = sum(len(s.matched) for s in scores)
    total_phantoms = sum(len(s.phantoms) for s in scores)
    total_dups = sum(s.duplicates for s in scores)
    total_wrong_side = sum(len(s.wrong_side) for s in scores)
    named_attempts = sum(s.names_right + len(s.names_wrong) + len(s.names_unnamed) for s in scores)
    print("=" * 72)
    print(f"screenshots        {len(scores)}")
    print(f"troops labelled    {total_labels}")
    print(f"troops found       {total_matched}  ({100.0 * total_matched / max(1, total_labels):.0f}%)")
    print(f"phantoms           {total_phantoms}")
    print(f"duplicates         {total_dups}")
    print(f"wrong side         {total_wrong_side}")
    if named_attempts:
        right = sum(s.names_right for s in scores)
        print(f"names correct      {right}/{named_attempts} "
              f"({100.0 * right / named_attempts:.0f}%)")
    if len(scores) > 1:
        # Each still is scored with its own inpainted background. This is only a positional
        # comparison, not a continuous replay of the live tracker.
        drift = stability(scores)
        print(f"position comparison {'clean' if not drift else str(len(drift)) + ' change(s)'}")
        for line in drift[:10]:
            print(f"    {line}")

    if args.json:
        args.json.write_text(json.dumps([{
            "image": s.image.name, "calibrated": s.calibrated, "note": s.note,
            "labelled": len(s.labels), "detected": len(s.detections),
            "matched": len(s.matched), "missed": len(s.missed),
            "phantoms": s.phantoms, "duplicates": s.duplicates,
            "wrong_side": len(s.wrong_side), "names_right": s.names_right,
            "names_wrong": s.names_wrong, "names_unnamed": s.names_unnamed,
            "ms": round(s.ms, 1),
        } for s in scores], indent=2))
    print("Counts are relative to supplied labels; inpainted stills are not live-stream recall.")
    clean = all(s.calibrated for s in scores) and total_phantoms == 0 and total_dups == 0 \
        and total_wrong_side == 0 and total_matched == total_labels
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
