#!/usr/bin/env python3
"""
One-command Hand Card Template Slicer & Screen Calibration Verifier.

Usage (while inside a Training Camp or Friendly Battle on your emulator):
    python scripts/capture_templates.py
Or from a saved 720x1280 screenshot:
    python scripts/capture_templates.py --image my_battle_screenshot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr_bot.adb.controller import ADBController
from cr_bot.arena.device_calibrator import (
    AUTO_CALIBRATION_REVISION,
    DeviceCalibrator,
    calibration_path,
    load_calibration,
    render_calibration_preview,
    save_calibration,
)
from cr_bot.arena.grid import ArenaGrid
from cr_bot.config.loader import load_config
from cr_bot.config.cards_db import normalize_card_name
from cr_bot.vision.elixir_tracker import ElixirReader
from cr_bot.vision.hand_detector import HandCardDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture 8-card hand templates & verify calibration.")
    parser.add_argument("--config", default="configs/mk_zap_bait.yaml", help="Bot YAML config")
    parser.add_argument("--image", default=None, help="Optional path to an existing screenshot PNG")
    parser.add_argument("--serial", default=None, help="Optional ADB serial override")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)

    serial = args.serial or cfg.adb_serial
    if args.image:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Error: could not read image '{args.image}'")
            return 1
    else:
        adb = ADBController(
            adb_path=cfg.adb_path,
            serial=serial,
            target_width=cfg.capture_width,
            target_height=cfg.target_height,
            capture_backend=cfg.capture_backend,
            capture_bitrate=cfg.capture_bitrate,
            capture_max_fps=cfg.capture_max_fps,
        )
        try:
            adb.connect()
            serial = adb.serial or serial
            frame = adb.capture_frame()
        finally:
            adb.close()
        if frame is None:
            print("Error: failed to grab frame from ADB.")
            return 1

    # Card crops must use the same current, validated tray geometry as the live bot.
    frame_h, frame_w = frame.shape[:2]
    path = calibration_path(ROOT / cfg.calibration_dir, serial, frame_w, frame_h)
    calibration = load_calibration(path)
    if calibration is not None and calibration.source == "auto" and (
        calibration.auto_revision < AUTO_CALIBRATION_REVISION or not calibration.hand_slots_px
        or len(calibration.hand_slots_px) != 4 or calibration.frame_size != (frame_w, frame_h)
    ):
        print("Ignoring stale/incomplete automatic calibration; measuring again...")
        calibration = None
    if calibration is None:
        print("No current calibration for this device/frame size; measuring one now...")
        result = DeviceCalibrator().calibrate(frame)
        calibration = result.calibration
        print(f"  confidence {result.confidence:.2f} [{result.stage}]")
        if not calibration.hand_slots_px or result.confidence < 0.50:
            print("Could not locate enough hand/arena anchors. Capture a clear battle frame "
                  "with at least three visible magenta badges, or calibrate manually first.")
            return 1
        save_calibration(path, calibration)
    if not calibration.hand_slots_px or len(calibration.hand_slots_px) != 4:
        print("The saved calibration has no measured four-card hand. Recalibrate on a clear "
              "battle frame before capturing templates.")
        return 1

    grid = ArenaGrid(frame_w, frame_h, calibration)
    detector = HandCardDetector(grid, cfg.normalized_deck(), templates_dir=ROOT / cfg.templates_dir,
                                evolutions=cfg.evolutions)
    elixir_reader = ElixirReader(grid.cal, grid=grid)

    out_dir = ROOT / cfg.templates_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save annotated calibration overlay so user can visually verify alignment
    preview = render_calibration_preview(frame, calibration)

    preview_path = out_dir / "_capture_calibration_preview.png"
    cv2.imwrite(str(preview_path), preview)
    detected_elixir = elixir_reader.read_elixir(frame)
    print(f"Saved calibration overlay -> {preview_path} (Detected Elixir: {detected_elixir:.1f})")
    print("Check that all cyan boxes align before labelling. Use unraised, affordable cards;")
    print("a wrong label or grey/occluded crop will make the template matcher less reliable.")

    deck = cfg.normalized_deck()
    print("\nYour configured 8-card deck:")
    for i, c in enumerate(deck, start=1):
        print(f"  [{i}] {c}")
    print("  [s] Skip slot    [q] Quit\n")

    for slot_idx in range(4):
        crop = detector.crop_slot(frame, slot_idx)
        raw_slot_path = out_dir / f"_raw_slot_{slot_idx}.png"
        cv2.imwrite(str(raw_slot_path), crop)
        choice = input(
            f"Which card is in Slot {slot_idx} (saved preview at {raw_slot_path.name})? Enter 1-{len(deck)}, card slug, or 's': "
        ).strip()
        if choice.lower() == "q":
            raw_slot_path.unlink(missing_ok=True)
            break
        if not choice or choice.lower() == "s":
            raw_slot_path.unlink(missing_ok=True)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(deck):
            card_slug = deck[int(choice) - 1]
        else:
            card_slug = normalize_card_name(choice)
        if card_slug not in deck:
            print("  Card is not in the configured deck; skipping this slot.")
            raw_slot_path.unlink(missing_ok=True)
            continue

        suffix = ""
        evo_ans = input(f"  Is '{card_slug}' currently glowing with Evolution ready? [y/N]: ").strip().lower()
        if evo_ans == "y":
            suffix = "_evo"

        # Preserve shipped templates and earlier device captures. The hand matcher
        # recognizes the card prefix and treats each file as another variant.
        dest = out_dir / f"{card_slug}_device{suffix}.png"
        number = 2
        while dest.exists():
            dest = out_dir / f"{card_slug}_device{number}{suffix}.png"
            number += 1
        if not cv2.imwrite(str(dest), crop):
            print(f"  Could not write {dest}")
            raw_slot_path.unlink(missing_ok=True)
            continue
        raw_slot_path.unlink(missing_ok=True)
        print(f"  -> Saved template: {dest}")

    print("\nDone! Play 4 cards in Training Camp and run again to capture your remaining 4 cards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
