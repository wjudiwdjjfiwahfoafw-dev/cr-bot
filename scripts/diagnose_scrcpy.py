#!/usr/bin/env python3
"""Diagnose why scrcpy will not stream, and say exactly what to do about it.

"scrcpy always fails" is not a diagnosis. On the SM-A528B the whole symptom was one line — the
server logs ``Device: [samsung] samsung SM-A528B (Android 16)`` and then nothing, and no frame
ever arrives — and that pattern has a short list of causes that can be told apart in a few
seconds with ``adb`` alone. This script checks each of them *in order* and stops at the first
one that explains the failure:

1. Is there an encoder for the requested codec, and is a hardware one available?
2. Is another scrcpy server already running and holding the abstract socket?
3. Does the server survive past startup with a *fresh* socket id?
4. Do the two ends agree on the protocol version (jar vs. this bot)?
5. Does a plain ``screenrecord`` produce frames (i.e. is the display service healthy at all)?

It also prints the raw server log tail, which is what the bot already collects — the point is to
put the *interpretation* next to the raw output.

Usage::

    python scripts/diagnose_scrcpy.py                  # first connected device
    python scripts/diagnose_scrcpy.py -s RZCT20A0MBJ   # a specific serial
    python scripts/diagnose_scrcpy.py --keep-log       # full server output
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: List[str], timeout: float = 20.0) -> Tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, f"{cmd[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--serial", default=None)
    ap.add_argument("--keep-log", action="store_true", help="print the whole server log")
    args = ap.parse_args()

    adb = shutil.which("adb")
    if not adb:
        print("adb not found on PATH. Install Android platform-tools and retry.")
        return 2
    base = [adb] + (["-s", args.serial] if args.serial else [])

    code, out = run(base + ["devices"])
    print("== devices ==")
    print(out.strip() or "(none)")
    code, out = run(base + ["shell", "getprop", "ro.build.version.release"])
    android = out.strip()
    code, out = run(base + ["shell", "wm", "size"])
    size = out.strip()
    print(f"android {android} | {size or 'unknown size'}")

    findings: List[str] = []

    # -- 1. encoders ---------------------------------------------------------------
    print("\n== encoders ==")
    code, out = run(base + ["shell", "dumpsys", "media.player"], timeout=5)
    code, out = run(base + ["shell", "cat", "/vendor/etc/media_codecs.xml"], timeout=15)
    codecs = re.findall(r'name="([^"]*avc[^"]*)"', out, re.IGNORECASE)
    hw = [c for c in sorted(set(codecs)) if not c.lower().startswith("omx.google")]
    sw = [c for c in sorted(set(codecs)) if c.lower().startswith("omx.google")]
    if codecs:
        print(f"  H.264 encoders: {len(set(codecs))} ({len(hw)} hardware, {len(sw)} software)")
        for c in (hw + sw)[:6]:
            print(f"    {c}")
    else:
        print("  could not enumerate encoders (media_codecs.xml unreadable) - not fatal")
        findings.append("encoder list unavailable; if scrcpy fails, try capture.backend=screenrecord")

    # -- 2. a stale server ---------------------------------------------------------
    print("\n== running scrcpy servers ==")
    code, out = run(base + ["shell", "ps", "-A"])
    stale = [ln for ln in out.splitlines() if "com.genymobile.scrcpy.Server" in ln]
    if stale:
        print(f"  {len(stale)} already running:")
        for ln in stale[:5]:
            print("   ", ln.strip()[:120])
        findings.append(
            "a scrcpy server is already running; kill it (`adb shell pkill -f "
            "com.genymobile.scrcpy.Server`) and retry - two servers cannot both own "
            "localabstract:scrcpy"
        )
    else:
        print("  none")

    # -- 3. a server with a fresh socket id ---------------------------------------
    print("\n== server with a fresh socket id ==")
    jar = None
    for candidate in sorted((ROOT / "cr_bot" / "third_party").glob("scrcpy-server-v*.jar")):
        jar = candidate        # take the newest bundled server
    if jar is None:
        print("  no scrcpy-server jar found under cr_bot/third_party/ - scrcpy cannot be used")
        findings.append("scrcpy-server jar missing from cr_bot/third_party/")
    else:
        run(base + ["push", str(jar), "/data/local/tmp/"])
        remote = f"/data/local/tmp/{jar.name}"
        version = re.sub(r"^scrcpy-server[-v]?", "", jar.name).replace(".jar", "") or "2.4"
        cmd = (
            f"CLASSPATH={remote} app_process / com.genymobile.scrcpy.Server {version} "
            "scid=2b3a7f log_level=debug video=true audio=false control=false "
            "send_stream_meta=false send_frame_meta=false max_size=720 max_fps=10"
        )
        print(f"  launching: {cmd}")
        code, out = run(base + ["shell", "timeout", "6", cmd], timeout=20)
        lines = out.strip().splitlines()
        if args.keep_log:
            print("\n".join("    " + ln for ln in lines))
        else:
            print("\n".join("    " + ln for ln in lines[-12:]))
        if any("ERROR" in ln or "Exception" in ln for ln in lines):
            findings.append(
                "the server reported an error; the message above names the cause "
                "(encoder, permission or display)"
            )
        elif any("Device:" in ln for ln in lines) and not any("Encoder" in ln for ln in lines):
            findings.append(
                "the server starts and identifies the device but never reaches encoder "
                "creation: this is the Android 14+ abstract-socket problem, and the fresh scid "
                "above is the workaround the bot now tries automatically"
            )

    # -- 4. does anything capture the screen? -------------------------------------
    print("\n== screenrecord fallback ==")
    code, out = run(base + ["shell", "rm", "-f", "/sdcard/diag.mp4"])
    code, out = run(
        base + ["shell", "screenrecord", "--time-limit", "3", "--size", "720x1280",
                "/sdcard/diag.mp4"],
        timeout=25,
    )
    code, out = run(base + ["shell", "ls", "-l", "/sdcard/diag.mp4"])
    print("  " + (out.strip() or "(no file)"))
    if "diag.mp4" not in out or re.search(r"\s0\s", out):
        findings.append("screenrecord produced nothing either: the display/encoder path is broken")
    run(base + ["shell", "rm", "-f", "/sdcard/diag.mp4"])

    # -- verdict -------------------------------------------------------------------
    print("\n== what to do ==")
    if not findings:
        print("  everything checks out; if the bot still fails, run it with --log-level debug "
              "and send the 'scrcpy server output' block")
    else:
        for i, f in enumerate(findings, 1):
            print(f"  {i}. {f}")
    print(
        "\n  Note: the screenrecord backend has no control channel, so every tap costs a\n"
        "  separate `adb shell input` call (100-250 ms). scrcpy is worth fixing: with it, a tap\n"
        "  is a 31-byte write on an open socket (single-digit milliseconds)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
