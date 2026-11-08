#!/usr/bin/env python3
"""Show what the troop detector sees — on your device, on your screenshots.

Draws every detected unit with its side, size class, estimated card name (if the identity memory
knows it), HP fraction and tile position, plus the detected health bars and the arena bounds.

Typical use while tuning a new phone:

    python3 scripts/inspect_detection.py --serial ABC123 --frames 40
    python3 scripts/inspect_detection.py --image shot.png
    python3 scripts/inspect_detection.py --image-dir frames/ --identities assets/unit_identity

Segmentation needs a few consecutive frames to learn the static arena.  With ``--serial`` the
script captures them; with ``--image-dir`` it reads a folder of *consecutive* frames; with a single
``--image`` only health bars can be used, and the script says so.  Supply ``--background`` (a frame
of the same arena with no troops on it) to get full segmentation from a single image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cr_bot.arena.device_calibrator import (  # noqa: E402
    ArenaCalibration,
    DeviceCalibrator,
    calibration_path,
    load_calibration,
)
from cr_bot.arena.grid import ArenaGrid  # noqa: E402
from cr_bot.vision.arena_detector import ArenaUnitBackend  # noqa: E402
from cr_bot.vision.unit_detector import pixels_per_tile  # noqa: E402
from cr_bot.vision.unit_identity import UnitIdentityMemory  # noqa: E402

SIDE_COLOUR = {"ally": (255, 190, 40), "enemy": (60, 60, 240)}


def calibrate(frame: np.ndarray, serial: Optional[str], directory: Path,
              allow_auto: bool = True) -> Optional[ArenaCalibration]:
    h, w = frame.shape[:2]
    if serial:
        saved = load_calibration(calibration_path(directory, serial, w, h))
        if saved is not None:
            print(f"using stored calibration for {serial} at {w}x{h}")
            return saved
    if not allow_auto:
        return None
    try:
        result = DeviceCalibrator().calibrate(frame)
    except Exception as exc:  # noqa: BLE001 - a bad frame must not crash the inspector
        print(f"auto-calibration failed: {exc}")
        return None
    cal = result.calibration if result is not None else None
    if cal is not None:
        print(f"auto-calibrated from the frame: confidence {result.confidence:.2f}, stage {result.stage}")
    return cal


def draw(frame: np.ndarray, detections, bars, grid: ArenaGrid, ppt: float) -> np.ndarray:
    out = frame.copy()
    for bar in bars:
        x1, y1, x2, y2 = bar.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 140, 255), 1)
    for det in detections:
        x1, y1, x2, y2 = det.pixel_bbox
        colour = SIDE_COLOUR.get(det.side.value, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        height_tiles = (y2 - y1) / max(1.0, ppt)
        label = f"{det.name} {det.side.value[0].upper()} {height_tiles:.1f}t"
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1,
                    cv2.LINE_AA)
        cv2.putText(out, f"hp {det.hp_ratio:.0%} ({det.tile_x:.1f},{det.tile_y:.1f})",
                    (x1, min(out.shape[0] - 4, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1,
                    cv2.LINE_AA)
    ax1, ay1, ax2, ay2 = grid.arena_pixel_bounds
    cv2.rectangle(out, (ax1, ay1), (ax2, ay2), (60, 220, 60), 1)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Visualise troop detection on a device or image.")
    ap.add_argument("--serial", help="ADB serial: capture live frames from the device")
    ap.add_argument("--image", help="a single battle screenshot")
    ap.add_argument("--image-dir", help="a folder of consecutive battle frames")
    ap.add_argument("--background", help="a frame of the same arena with no troops (enables "
                                        "segmentation for --image)")
    ap.add_argument("--frames", type=int, default=40, help="live/folder frames to feed the model")
    ap.add_argument("--identities", default="assets/unit_identity")
    ap.add_argument("--calibration-dir", default="configs/calibration")
    ap.add_argument("--out", default="detection_overlay.png")
    ap.add_argument("--max-width", type=int, default=720)
    args = ap.parse_args(argv)

    frames: List[np.ndarray] = []
    if args.serial:
        from cr_bot.adb.controller import ADBController

        adb = ADBController(serial=args.serial, max_width=args.max_width)
        adb.connect()
        print(f"capturing {args.frames} frames from {args.serial} ...")
        for _ in range(max(4, args.frames)):
            frame = adb.capture_frame(timeout=0.5)
            if frame is not None:
                frames.append(frame)
        adb.close()
    elif args.image_dir:
        for path in sorted(Path(args.image_dir).glob("*.png"))[: max(4, args.frames)]:
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(img)
    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"could not read {args.image}")
            return 1
        frames = [img]
    else:
        ap.error("give one of --serial, --image or --image-dir")

    if not frames:
        print("no frames captured")
        return 1

    frame = frames[-1]
    h, w = frame.shape[:2]
    cal = calibrate(frames[0], args.serial, Path(args.calibration_dir))
    if cal is None:
        print("could not calibrate this frame; pass a stored calibration dir or check the capture")
        return 1
    grid = ArenaGrid(w, h, cal)
    ppt = pixels_per_tile(grid)

    identity = UnitIdentityMemory(directory=args.identities, namespace=f"{w}x{h}")
    if identity.sample_count:
        print(f"identity memory: {identity.sample_count} sample(s) for {len(identity.known_cards)} card(s)")
    else:
        print("identity memory is empty (nothing learned yet) - labels will stay generic")

    backend = ArenaUnitBackend(identity_memory=identity, enable_segmentation=True)
    if args.background:
        bg = cv2.imread(args.background)
        if bg is not None:
            if (bg.shape[1], bg.shape[0]) != (w, h):
                bg = cv2.resize(bg, (w, h))
            if backend.segmenter.seed_background(bg, grid):
                print("seeded the background model from --background")

    detections = []
    for f in frames:
        if (f.shape[1], f.shape[0]) != (w, h):
            f = cv2.resize(f, (w, h))
        detections = backend.detect(f, grid)

    bars = backend.segmenter.last_bars
    if len(frames) < 4 and detections and all(d.source == "bars" for d in detections):
        print("note: segmentation needs a few consecutive frames; showing health-bar detections only")

    print(f"\n{len(detections)} unit(s), {len(bars)} bar(s), tile scale {ppt:.1f} px/tile")
    for det in detections:
        print(f"  {det.side.value:<5} {det.name:<18} {det.source:<12} "
              f"conf {det.confidence:.2f} hp {det.hp_ratio:.0%} "
              f"tile ({det.tile_x:5.1f},{det.tile_y:5.1f}) bbox {det.pixel_bbox}")

    out = draw(frame, detections, bars, grid, ppt)
    cv2.imwrite(args.out, out)
    print(f"\noverlay -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
