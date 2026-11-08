#!/usr/bin/env python3
"""Main CLI entrypoint for the Clash Royale MK Zap Bait Tactical AI Bot."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cr_bot.bot_runner import ClashRoyaleBotRunner
from cr_bot.config.loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clash Royale Hybrid YOLOv8+OpenCV Tactical Bot (MK Zap Bait)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mk_zap_bait.yaml",
        help="Path to deck & bot YAML configuration (default: configs/mk_zap_bait.yaml)",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Explicit ADB device serial (e.g. 127.0.0.1:5555). Auto-discovers if omitted.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Override path to YOLOv8 .pt or .onnx weights file.",
    )
    parser.add_argument(
        "--auto-queue",
        action="store_true",
        help="Automatically click Battle on Main Menu and OK on Match Over screen.",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable the live OpenCV tactical HUD window (headless mode).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Observe & render live HUD decisions without sending ADB touch inputs.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Stop automatically after playing N matches.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    if args.serial:
        cfg.adb_serial = args.serial
    if args.yolo_weights:
        cfg.yolo_weights_path = args.yolo_weights
    if args.auto_queue:
        cfg.auto_queue_battle = True

    runner = ClashRoyaleBotRunner(
        config=cfg,
        show_hud=not args.no_hud,
        dry_run=args.dry_run,
    )
    runner.run_forever(max_matches=args.max_matches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
