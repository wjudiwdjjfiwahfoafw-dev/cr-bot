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
from cr_bot.arena.grid import ArenaGrid
from cr_bot.config.loader import load_config
from cr_bot.vision.elixir_tracker import ElixirReader
from cr_bot.vision.hand_detector import HandCardDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture 8-card hand templates & verify calibration.")
    parser.add_argument("--config", default="configs/mk_zap_bait.yaml", help="Bot YAML config")
    parser.add_argument("--image", default=None, help="Optional path to an existing screenshot PNG")
    parser.add_argument("--serial", default=None, help="Optional ADB serial override")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    grid = ArenaGrid(cfg.target_width, cfg.target_height)
    detector = HandCardDetector(grid, cfg.normalized_deck(), templates_dir=ROOT / cfg.templates_dir)
    elixir_reader = ElixirReader(grid.cal)

    if args.image:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Error: could not read image '{args.image}'")
            return 1
        frame = cv2.resize(frame, (cfg.target_width, cfg.target_height))
    else:
        adb = ADBController(
            adb_path=cfg.adb_path,
            serial=args.serial or cfg.adb_serial,
            target_width=cfg.target_width,
            target_height=cfg.target_height,
        )
        adb.connect()
        frame = adb.capture_frame()
        if frame is None:
            print("Error: failed to grab frame from ADB.")
            return 1

    out_dir = ROOT / cfg.templates_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save annotated calibration overlay so user can visually verify alignment
    preview = frame.copy()
    ax1, ay1, ax2, ay2 = grid.arena_pixel_bounds
    cv2.rectangle(preview, (ax1, ay1), (ax2, ay2), (0, 255, 120), 2)
    for idx in range(4):
        sx, sy = grid.hand_slot_pixel(idx)
        cv2.rectangle(preview, (sx - 48, sy - 58), (sx + 48, sy + 58), (255, 200, 0), 2)
        cv2.putText(
            preview, f"Slot {idx}", (sx - 30, sy - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2
        )

    preview_path = ROOT / "calibration_preview.png"
    cv2.imwrite(str(preview_path), preview)
    detected_elixir = elixir_reader.read_elixir(frame)
    print(f"Saved calibration overlay -> {preview_path} (Detected Elixir: {detected_elixir:.1f})")

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
            break
        if not choice or choice.lower() == "s":
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(deck):
            card_slug = deck[int(choice) - 1]
        else:
            card_slug = choice.strip().lower().replace(" ", "_")

        suffix = ""
        evo_ans = input(f"  Is '{card_slug}' currently glowing with Evolution ready? [y/N]: ").strip().lower()
        if evo_ans == "y":
            suffix = "_evo"

        dest = out_dir / f"{card_slug}{suffix}.png"
        cv2.imwrite(str(dest), crop)
        raw_slot_path.unlink(missing_ok=True)
        print(f"  -> Saved template: {dest}")

    print("\nDone! Play 4 cards in Training Camp and run again to capture your remaining 4 cards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
