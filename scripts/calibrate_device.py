#!/usr/bin/env python3
"""
Calibrate the bot's view of Clash Royale on your device.

Run this **while a battle is on screen** (Training Camp is perfect), because it needs to see
the river, your hand cards, the elixir bar and the towers.

    # Automatic: detect everything, save the result, write a preview image
    python scripts/calibrate_device.py

    # Manual: click the four Princess Towers yourself (exact, ~10 seconds)
    python scripts/calibrate_device.py --manual

    # Manual without a display (e.g. over SSH): paste the four tower points
    python scripts/calibrate_device.py --towers "140,300;580,300;140,745;580,745"

    # Offline: calibrate from a saved screenshot
    python scripts/calibrate_device.py --image shot.png

Always check the generated ``*.preview.png``: the yellow grid must follow the arena, the
cyan boxes must sit exactly on your four cards, and the red line must lie on the river.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr_bot.arena.device_calibrator import (  # noqa: E402
    CalibrationResult,
    DeviceCalibrator,
    calibration_path,
    render_calibration_preview,
    save_calibration,
)
from cr_bot.arena.grid import ArenaGrid  # noqa: E402
from cr_bot.config.loader import load_config  # noqa: E402


def grab_frame(cfg, serial: str | None, image: str | None) -> tuple[np.ndarray, str]:
    if image:
        frame = cv2.imread(image, cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image {image!r}")
        return frame, Path(image).stem

    from cr_bot.adb.controller import ADBController

    adb = ADBController(
        adb_path=cfg.adb_path,
        serial=serial or cfg.adb_serial,
        target_width=cfg.capture_width,
        target_height=cfg.target_height,
        capture_backend=cfg.capture_backend,
        capture_bitrate=cfg.capture_bitrate,
        capture_max_fps=cfg.capture_max_fps,
    )
    adb.connect()
    print(f"Capture: {adb.status_line()}")
    frame = None
    for _ in range(60):
        frame = adb.capture_frame(timeout=0.5)
        if frame is not None:
            break
    adb.close()
    if frame is None:
        raise SystemExit("Could not capture a frame. Is the game visible on screen?")
    return frame, adb.serial or "device"


def parse_towers(text: str) -> dict[str, tuple[float, float]]:
    """Parse 'x,y;x,y;x,y;x,y' into named tower points, ordered by row then x."""
    pts: list[tuple[float, float]] = []
    for chunk in text.replace(" ", "").split(";"):
        if not chunk:
            continue
        x_str, y_str = chunk.split(",")
        pts.append((float(x_str), float(y_str)))
    if len(pts) != 4:
        raise SystemExit("--towers needs exactly four 'x,y' points: enemy-left, enemy-right, ally-left, ally-right")

    top = sorted(pts, key=lambda p: p[1])[:2]
    bottom = sorted(pts, key=lambda p: p[1])[2:]
    top = sorted(top, key=lambda p: p[0])
    bottom = sorted(bottom, key=lambda p: p[0])
    return {
        "enemy_left": top[0],
        "enemy_right": top[1],
        "ally_left": bottom[0],
        "ally_right": bottom[1],
    }


def has_display() -> bool:
    """True when an interactive OpenCV window can actually be created."""
    import os

    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if sys.platform == "darwin":
        return True
    return True  # Windows


def manual_pick(frame: np.ndarray) -> dict[str, tuple[float, float]]:
    """Let the user click the four Princess Tower bases in any order."""
    if not has_display():
        raise SystemExit(
            "No display available for interactive picking.\n"
            "Use --towers \"x,y;x,y;x,y;x,y\" with the four Princess Tower bases instead\n"
            "(read them off the preview image), e.g.\n"
            "  --towers \"140,300;580,300;140,745;580,745\""
        )
    points: list[tuple[float, float]] = []
    win = "Click the 4 Princess Towers (their base). ENTER=done, R=reset, ESC=cancel"

    def redraw() -> np.ndarray:
        img = frame.copy()
        for idx, (x, y) in enumerate(points):
            cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(img, str(idx + 1), (int(x) + 12, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{len(points)}/4 towers clicked  (ENTER=done, R=reset, ESC=cancel)",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return img

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            cv2.imshow(win, redraw())
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(points) == 4:
                break
            if key in (ord("r"), ord("R")):
                points.clear()
            if key == 27:
                raise SystemExit("Cancelled.")
    finally:
        cv2.destroyWindow(win)

    top = sorted(points, key=lambda p: p[1])[:2]
    bottom = sorted(points, key=lambda p: p[1])[2:]
    top = sorted(top, key=lambda p: p[0])
    bottom = sorted(bottom, key=lambda p: p[0])
    return {
        "enemy_left": top[0],
        "enemy_right": top[1],
        "ally_left": bottom[0],
        "ally_right": bottom[1],
    }


def report(frame: np.ndarray, result: CalibrationResult, grid: ArenaGrid) -> None:
    h, w = frame.shape[:2]
    print()
    print(f"  frame          : {w}x{h}")
    print(f"  stage          : {result.stage}")
    print(f"  confidence     : {result.confidence:.2f}")
    for line in result.report:
        print(f"    - {line}")
    print(f"  river y        : {grid.river_pixel_y:.0f}")
    ex1, ex2, ey = grid.elixir_geometry()
    print(f"  elixir bar     : x {ex1}..{ex2} at y {ey}")
    for i in range(4):
        print(f"  hand slot {i}    : centre {grid.hand_slot_pixel(i)}")
    print("  sample deployments:")
    for label, tile in (
        ("left bridge", grid.LEFT_BRIDGE),
        ("right bridge", grid.RIGHT_BRIDGE),
        ("kite centre", grid.KITE_LEFT_THREAT_TO_CENTER),
        ("back corner", grid.BACK_LEFT_CORNER),
    ):
        px = grid.tile_to_pixel(*tile)
        legal = grid.is_legal_deploy_pixel(*px)
        print(f"    {label:<14} tile {tile} -> pixel {px}  {'OK' if legal else 'ILLEGAL (will be clamped)'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate the Clash Royale bot for your device.")
    parser.add_argument("--config", default="configs/mk_zap_bait.yaml")
    parser.add_argument("--serial", default=None, help="ADB serial (default: auto-detect)")
    parser.add_argument("--image", default=None, help="Calibrate from a saved screenshot instead")
    parser.add_argument("--manual", action="store_true", help="Click the four Princess Towers")
    parser.add_argument("--towers", default=None, help="'x,y;x,y;x,y;x,y' tower points (headless manual)")
    parser.add_argument("--out", default=None, help="Where to save the calibration JSON")
    parser.add_argument("--no-save", action="store_true", help="Only print, do not write files")
    parser.add_argument("--no-window", action="store_true", help="Never open a GUI window")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)

    print("Grabbing a frame...")
    frame, name = grab_frame(cfg, args.serial, args.image)

    towers = None
    if args.towers:
        towers = parse_towers(args.towers)
    elif args.manual:
        towers = manual_pick(frame)

    calibrator = DeviceCalibrator(manual_towers=towers)
    result = calibrator.calibrate(frame)
    calibration = result.calibration

    w, h = frame.shape[1], frame.shape[0]
    grid = ArenaGrid(w, h, calibration)
    report(frame, result, grid)

    preview = render_calibration_preview(frame, calibration, result)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = calibration_path(ROOT / cfg.calibration_dir, name, w, h)

    if not args.no_save:
        save_calibration(out_path, calibration)
        preview_path = out_path.with_suffix(".preview.png")
        cv2.imwrite(str(preview_path), preview)
        print()
        print(f"Saved calibration : {out_path}")
        print(f"Saved preview     : {preview_path}")
        print("Open the preview and confirm the yellow grid follows the arena and the cyan")
        print("boxes sit on your four cards. If not, re-run with --manual.")
    else:
        preview_out = ROOT / "calibration_preview_out.png"
        cv2.imwrite(str(preview_out), preview)
        print(f"\nPreview written to {preview_out} (nothing saved)")

    if result.confidence < 0.55:
        print()
        print("!! Low confidence. Automatic detection was not sure. This usually means the")
        print("!! frame was mid-animation, or a non-standard arena skin is in use.")
        print("!! Re-run during the battle countdown, or use --manual for an exact fit.")

    if not args.no_window and has_display():
        cv2.imshow("calibration", preview)
        print("Press any key in the preview window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("(Headless: open the preview PNG to check the alignment.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
