#!/usr/bin/env python3
"""Main CLI entrypoint for the Clash Royale MK Zap Bait Tactical AI Bot."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from cr_bot.bot_runner import ClashRoyaleBotRunner
from cr_bot.config.loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clash Royale Tactical Bot (MK Zap Bait) — runs on a real Android device over ADB"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mk_zap_bait.yaml",
        help="Path to deck & bot YAML configuration (default: configs/mk_zap_bait.yaml)",
    )
    parser.add_argument("--serial", type=str, default=None,
                        help="Explicit ADB device serial (e.g. R58M1234 or 192.168.1.5:5555).")
    parser.add_argument("--yolo-weights", type=str, default=None,
                        help="Override path to YOLOv8 .pt or .onnx weights file.")

    # -- behaviour --
    parser.add_argument("--auto-queue", action="store_true",
                        help="Automatically click Battle on the main menu and OK after a match.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Watch and plan without injecting any touches (live coach mode).")
    parser.add_argument("--no-hud", action="store_true",
                        help="Disable the OpenCV HUD window (headless).")
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Stop automatically after playing N matches.")

    # -- real-device plumbing --
    parser.add_argument("--capture", choices=["auto", "screenrecord", "scrcpy_py", "screencap"],
                        default=None, help="Capture backend (default from config).")
    parser.add_argument("--input", dest="input_backend",
                        choices=["auto", "swipe", "motionevent", "tap"],
                        default=None, help="Touch injection backend (default from config).")
    parser.add_argument("--deploy-style", choices=["drag", "tap"], default=None,
                        help="Card placement gesture. `drag` is the game's native gesture.")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Ignore the saved calibration and measure the screen again.")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Do not auto-calibrate; use the bundled default geometry.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip post-deploy hand verification (faster, but loses self-healing).")
    parser.add_argument("--stats", action="store_true",
                        help="Print device + capture diagnostics, probe the capture path, then exit.")
    parser.add_argument("--package", type=str, default=None,
                        help="Application id of the build to play, e.g. nullsroyale.rel.free. "
                             "Any build of the game is accepted; the default list already "
                             "includes the stock app and known custom APKs.")
    parser.add_argument("--launch", action="store_true",
                        help="Bring the game to the foreground before starting.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level.")
    return parser.parse_args()


def print_stats(cfg) -> int:
    """Report what the device looks like and *prove the capture path works*, then exit.

    This is the first command anyone runs when the bot misbehaves, so it must never end in a
    traceback: every failure is turned into a diagnosis plus what to do about it.
    """
    from cr_bot.adb.adb_utils import ADBError, Adb

    try:
        adb = Adb(cfg.adb_path, cfg.adb_serial)
        serial = adb.connect(cfg.adb_serial)
        info = adb.display_info()
    except ADBError as exc:
        print("\nADB DEVICE PROBLEM\n" + str(exc))
        return 2

    print(f"device      : {serial}")
    print(f"resolution  : {info.width}x{info.height}  rotation={info.rotation}  dpi={info.density}")
    foreground = adb.foreground_package()
    print(f"foreground  : {foreground or '?'}")
    packages, hints = cfg.game_packages, cfg.game_package_hints
    in_game = adb.is_clash_royale_foreground(packages, hints)
    if in_game:
        print(f"clash royale: yes ({foreground})")
    else:
        installed = adb.find_game_package(packages, hints)
        if installed:
            print(f"clash royale: NO  <- installed as '{installed}' but not in front; "
                  f"start it or use --launch")
        else:
            print("clash royale: NO  <- no build of the game found; configure game.packages "
                  "if you play a custom APK")
    ladder = {
        "auto": "screenrecord -> scrcpy_py -> screencap",
        "screenrecord": "screenrecord only",
        "scrcpy_py": "scrcpy library only",
        "screencap": "screencap only",
    }.get(cfg.capture_backend, cfg.capture_backend)
    print(f"capture     : {cfg.capture_backend} ({ladder}) @ {cfg.capture_width}px, {cfg.capture_max_fps}fps")
    print(f"input       : {' -> '.join(cfg.input_backend_order())} ({cfg.deploy_style})")

    from cr_bot.adb.capture import MultiClient, scrcpy_py_available

    lib = scrcpy_py_available()
    if lib:
        print(f"scrcpy lib  : scrcpy {lib} installed (brings its own server jar)")
    else:
        print("scrcpy lib  : not installed - `screenrecord` needs no jar and is the default")

    print("\nprobing capture ...")
    client = MultiClient(
        adb_path=cfg.adb_path,
        serial=serial,
        target_width=cfg.capture_width,
        bitrate=cfg.capture_bitrate,
        max_fps=cfg.capture_max_fps,
        backend=cfg.capture_backend,
    )
    try:
        client.start()
    except ADBError as exc:
        print("\nCAPTURE FAILED\n" + str(exc))
        print("\nNext: retry with `--capture screencap` to confirm the rest of the link works.")
        return 2

    try:
        backend = client.backend_used or type(client.session()).__name__
        frame = client.latest_frame(timeout=6.0)
        if frame is None:
            print(f"capture     : {backend} started but produced no frame")
            return 2
        time.sleep(0.6)
        fps = client.fps
        print(f"capture     : {backend} OK - {frame.shape[1]}x{frame.shape[0]} @ {fps:.1f} fps")
        if backend == "screenrecord":
            print(
                "              Running on `screenrecord` (Android's own tool): full frame rate, "
                "\n              but touch goes through `adb shell input`, so deploys land a few "
                "ms later."
            )
        else:
            print(
                "              NOTE: running on the screencap fallback (~1-8 fps). Playable, but "
                "reactions are late.\n              See the reason above; `--capture screencap` "
                "makes this choice explicit."
            )
        if not in_game:
            print("              The game is not in the foreground: start a battle before playing.")
        print("\nAll good. Run `python main.py` to play.")
    finally:
        client.stop()
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    if args.serial:
        cfg.adb_serial = args.serial
    if args.yolo_weights:
        cfg.yolo_weights_path = args.yolo_weights
    if args.auto_queue:
        cfg.auto_queue_battle = True
    if args.capture:
        cfg.capture_backend = args.capture
    if args.input_backend:
        cfg.input_backend = args.input_backend
    if args.deploy_style:
        cfg.deploy_style = args.deploy_style
    if args.package:
        cfg.game_packages = [args.package]

    if args.stats:
        return print_stats(cfg)

    if args.launch:
        from cr_bot.adb.adb_utils import Adb

        adb = Adb(cfg.adb_path, cfg.adb_serial)
        try:
            adb.connect(cfg.adb_serial)
            target = adb.find_game_package(cfg.game_packages, cfg.game_package_hints)
            if target and adb.launch_package(target):
                print(f"launched {target}")
                time.sleep(4.0)
        except Exception as exc:  # noqa: BLE001 - launching is a convenience, never fatal
            print(f"could not launch the game automatically: {exc}")

    runner = ClashRoyaleBotRunner(
        config=cfg,
        show_hud=not args.no_hud,
        dry_run=args.dry_run,
        auto_calibrate=not args.no_calibrate,
        force_recalibrate=args.recalibrate,
        verify_deploys=not args.no_verify,
    )
    try:
        runner.run_forever(max_matches=args.max_matches)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0
    except Exception as exc:  # noqa: BLE001 - never show a raw traceback to a player
        from cr_bot.adb.adb_utils import ADBError

        if isinstance(exc, ADBError):
            print("\nDEVICE ERROR\n" + str(exc))
            print("\nRun `python main.py --stats` for a full diagnosis.")
            return 2
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
