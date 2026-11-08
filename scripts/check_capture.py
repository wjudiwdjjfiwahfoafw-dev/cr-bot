#!/usr/bin/env python3
"""Measure the real frame rate of every capture backend on the connected device.

The bot silently degrades down its capture ladder (scrcpy -> screenrecord -> screencap), so
"it works" says nothing about how *fast* it works. This script starts each backend in turn,
counts the frames that actually arrive, and prints a table - the quickest way to see whether
a device is on a 30 fps stream or limping along on 1 fps screencap polling.

Usage::

    python scripts/check_capture.py                 # every backend
    python scripts/check_capture.py --seconds 8     # measure longer
    python scripts/check_capture.py --only screenrecord
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cr_bot.adb.capture import MultiClient  # noqa: E402
from cr_bot.config.loader import load_config  # noqa: E402

ORDER = ("screenrecord", "scrcpy_py", "screencap")


def measure(backend: str, serial, seconds: float, width: int, bitrate: int, max_fps: int) -> dict:
    client = MultiClient(
        serial=serial,
        target_width=width,
        bitrate=bitrate,
        max_fps=max_fps,
        backend=backend,
        connect_timeout=12.0,
    )
    started = time.monotonic()
    try:
        client.start()
        startup = time.monotonic() - started
    except Exception as exc:  # noqa: BLE001 - report, never crash the sweep
        client.stop()
        return {"backend": backend, "ok": False, "error": str(exc).splitlines()[0]}

    session = client.session()
    first = None
    deadline = time.monotonic() + seconds
    baseline = session.stats.frames
    frame = None
    while time.monotonic() < deadline:
        candidate = client.latest_frame(timeout=0.2)
        if candidate is not None:
            frame = candidate
            if first is None:
                first = time.monotonic()
        time.sleep(0.005)
    elapsed = max(1e-6, time.monotonic() - (first or started))
    produced = session.stats.frames - baseline
    result = {
        "backend": backend,
        "ok": True,
        "used": getattr(client, "backend_used", backend),
        "startup": startup,
        "fps": produced / elapsed,
        "frames": session.stats.frames,
        "frame": None if frame is None else (frame.shape[1], frame.shape[0]),
        "fallback": getattr(client, "fallback_reason", ""),
    }
    client.stop()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=None, help="Device serial (default: first device)")
    ap.add_argument("--seconds", type=float, default=5.0, help="Measure duration per backend")
    ap.add_argument("--config", default=None, help="Config file (default: configs/mk_zap_bait.yaml)")
    ap.add_argument("--only", choices=ORDER, default=None, help="Measure just one backend")
    args = ap.parse_args()

    default_cfg = ROOT / "configs" / "mk_zap_bait.yaml"
    cfg = load_config(args.config or default_cfg)
    backends = [args.only] if args.only else list(ORDER)

    print(f"Measuring {len(backends)} backend(s) for {args.seconds:.0f}s each ...\n")
    results = []
    for backend in backends:
        print(f"  {backend:<13} ", end="", flush=True)
        res = measure(
            backend,
            args.serial or cfg.adb_serial,
            args.seconds,
            cfg.capture_width,
            cfg.capture_bitrate,
            cfg.capture_max_fps,
        )
        results.append(res)
        if res["ok"]:
            size = f"{res['frame'][0]}x{res['frame'][1]}" if res["frame"] else "n/a"
            print(f"{res['fps']:6.1f} fps   {size:<11} start {res['startup']:.1f}s")
        else:
            print(f"FAILED       {res['error']}")

    print()
    best = max((r for r in results if r["ok"]), key=lambda r: r["fps"], default=None)
    if best is None:
        print("No backend delivered frames. Is the device unlocked and does `adb devices` list it?")
        return 1
    print(f"Fastest: {best['backend']} at {best['fps']:.1f} fps")
    playable = 12.0
    if best["fps"] < playable:
        print(
            "That is below the ~12 fps the brain needs to react to a push. Try\n"
            "  python scripts/check_capture.py --only screenrecord   (no server jar involved)\n"
            "and check `adb shell screenrecord --help` works on the device."
        )
    for res in results:
        if res["ok"] and res["fallback"]:
            print(f"\n{res['backend']} fell back:\n  " + res["fallback"].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
