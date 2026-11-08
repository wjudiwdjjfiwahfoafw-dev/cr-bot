#!/usr/bin/env python3
"""
Diagnose why cards are (or are not) being placed on your device.

This is the tool to run first when the bot "captures fine but never plays anything".
It tests every touch backend independently and reports the only thing that matters:
**did the card actually leave the hand?**

Run it while a battle is on screen (Training Camp is ideal — it costs nothing):

    python scripts/probe_placement.py

It will, for each backend (scrcpy control socket, `input swipe`, `input motionevent`,
`input tap`), drag the first hand card onto a safe tile in your own half, wait, and check
whether that slot changed. The first backend that reliably works is the one the bot should
use; you can pin it with `input.backend` in configs/mk_zap_bait.yaml.

Options:
    --slot N            which hand slot to test (default 0)
    --tile X,Y          arena tile to deploy on (default: back-left corner, always legal)
    --dry-run           print coordinates and write an annotated PNG, inject nothing
    --backend NAME      probe only one backend
    --repeat N          attempts per backend (default 2)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr_bot.adb.controller import ADBController  # noqa: E402
from cr_bot.arena.device_calibrator import (  # noqa: E402
    DeviceCalibrator,
    calibration_path,
    load_calibration,
    render_calibration_preview,
    save_calibration,
)
from cr_bot.arena.grid import ArenaGrid  # noqa: E402
from cr_bot.config.loader import load_config  # noqa: E402
from cr_bot.vision.deploy_verifier import DeployVerifier  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe which touch backend actually places cards.")
    parser.add_argument("--config", default="configs/mk_zap_bait.yaml")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--tile", default=None, help="Arena tile 'x,y' to deploy on")
    parser.add_argument("--dry-run", action="store_true", help="Do not inject any touches")
    parser.add_argument("--backend", default=None, help="Probe a single backend")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)

    adb = ADBController(
        adb_path=cfg.adb_path,
        serial=args.serial or cfg.adb_serial,
        target_width=cfg.capture_width,
        target_height=cfg.target_height,
        jitter_px=0,
        dry_run=args.dry_run,
        capture_backend=cfg.capture_backend,
        input_backend=cfg.input_backend,
        deploy_style="drag",
        capture_bitrate=cfg.capture_bitrate,
        capture_max_fps=cfg.capture_max_fps,
        keep_awake=cfg.keep_screen_on,
        lock_portrait=cfg.lock_portrait,
        input_order=cfg.input_backend_order(),
    )
    adb.connect()
    print(adb.status_line())
    print()

    frame = adb.capture_frame(timeout=3.0)
    if frame is None:
        print("Could not capture a frame.")
        return 1
    h, w = frame.shape[:2]

    # -- calibration --------------------------------------------------------------
    cal_file = calibration_path(ROOT / cfg.calibration_dir, adb.serial, w, h)
    calibration = load_calibration(cal_file)
    if calibration is None:
        print(f"No saved calibration for this device; measuring one now ({cal_file.name})...")
        result = DeviceCalibrator().calibrate(frame)
        calibration = result.calibration
        print(f"  confidence {result.confidence:.2f} [{result.stage}]")
        for line in result.report:
            print(f"    - {line}")
        save_calibration(cal_file, calibration)
        cv2.imwrite(str(cal_file.with_suffix(".preview.png")),
                    render_calibration_preview(frame, calibration, result))
        print(f"  saved to {cal_file}")
    else:
        print(f"Using saved calibration {cal_file}")

    grid = ArenaGrid(w, h, calibration)

    # -- pick a deploy target -----------------------------------------------------
    if args.tile:
        tx, ty = (float(v) for v in args.tile.split(","))
        tile = grid.clamp_deploy_tile(tx, ty)
    else:
        tile = grid.BACK_LEFT_CORNER
    arena_xy = grid.tile_to_pixel(*tile)
    slot_xy = grid.hand_slot_pixel(max(0, min(3, args.slot)))

    print()
    print(f"hand slot {args.slot} centre : {slot_xy}   (frame {w}x{h})")
    print(f"deploy tile {tile}        : {arena_xy}   legal={grid.is_legal_deploy_pixel(*arena_xy)}")
    print(f"river boundary            : y={grid.river_pixel_y:.0f}")
    print()

    annotated = render_calibration_preview(frame, calibration, None)
    cv2.circle(annotated, slot_xy, 14, (0, 128, 255), 3)
    cv2.circle(annotated, arena_xy, 14, (0, 0, 255), 3)
    cv2.arrowedLine(annotated, slot_xy, arena_xy, (0, 255, 255), 3, tipLength=0.04)
    out_png = ROOT / "probe_placement.png"
    cv2.imwrite(str(out_png), annotated)
    print(f"Annotated probe image -> {out_png}")

    if args.dry_run:
        print("\n--dry-run: not injecting anything. The arrow above is the gesture that would be sent.")
        adb.close()
        return 0

    # -- probe each backend -------------------------------------------------------
    verifier = DeployVerifier(change_threshold=cfg.deploy_verify_threshold)
    backends = [args.backend] if args.backend else cfg.input_backend_order()
    results: dict[str, list[bool]] = {}

    try:
        for backend in backends:
            if not adb.set_input_backend(backend):
                print(f"{backend:<12} : unavailable on this setup")
                continue
            outcomes: list[bool] = []
            for attempt in range(1, max(1, args.repeat) + 1):
                frame = adb.capture_frame(timeout=2.0)
                if frame is None:
                    break
                before = verifier.snapshot(frame, grid)
                t0 = time.monotonic()
                gesture = adb.deploy_card(slot_xy, arena_xy, force_backend=backend)
                # Give the game time to swap the card out of the hand.
                changed = False
                deadline = time.monotonic() + 1.4
                while time.monotonic() < deadline:
                    nf = adb.capture_frame(timeout=0.3)
                    if nf is None:
                        continue
                    grid.resize(nf.shape[1], nf.shape[0])
                    after = verifier.snapshot(nf, grid)
                    if after is not None and before is not None and verifier.slot_changed(before, after, args.slot):
                        changed = True
                        break
                    time.sleep(0.03)
                dt = (time.monotonic() - t0) * 1000.0
                outcomes.append(changed)
                status = "PLACED" if changed else "NO-OP"
                print(f"{backend:<12} : attempt {attempt} {status}  ({dt:.0f} ms, "
                      f"{gesture.kind}, gesture {gesture.latency_ms:.0f} ms)")
                if changed:
                    # Let elixir regenerate a little before the next attempt.
                    time.sleep(1.8)
            results[backend] = outcomes
    finally:
        adb.close()

    print()
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    working = []
    for backend, outcomes in results.items():
        ok = sum(1 for o in outcomes if o)
        total = len(outcomes)
        verdict = "WORKS" if ok == total and total else ("PARTIAL" if ok else "FAILS")
        if ok and ok == total:
            working.append(backend)
        print(f"  {backend:<12} {ok}/{total} placed   {verdict}")

    print()
    if working:
        print(f"Recommended: set `input.backend: \"{working[0]}\"` in configs/mk_zap_bait.yaml")
        print("(the bot also tries the rest automatically, so this is only a preference).")
    else:
        print("No backend managed to place a card. Things to check, in order:")
        print("  1. Is a battle actually running? The hand does not accept cards on menus.")
        print("  2. Do the hand slots in probe_placement.png sit on your actual cards?")
        print("     If not: python scripts/calibrate_device.py --manual")
        print("  3. Is the deploy target inside your own half? The red line is the river.")
        print("  4. Does the device accept synthetic input at all? Try:")
        print("       adb shell input tap 360 640      # should register somewhere on screen")
        print("     Some devices require 'USB debugging (Security settings)' to inject input.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
