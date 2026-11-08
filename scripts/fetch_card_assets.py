#!/usr/bin/env python3
"""Download the official Clash Royale art used by the vision pipeline (no training involved).

The bot needs two kinds of reference image:

``assets/card_art/<card>.png``
    Clean 3:3.6 portrait art, straight from the official asset dump.  Replaces the old
    ``assets/card_templates`` screenshots that were cropped out of a low-res emulator frame
    (blurry, letterboxed, with the elixir bubble baked in) and therefore matched badly.

The arena *troop* reference images are **not** these portraits: measured on a real device frame
the front-facing renders only identify the correct card 34 times out of 78 (chance).  Real
in-game animation frames are fetched by ``scripts/fetch_game_sprites.py`` instead; this script
only handles portraits and the hand-slot templates derived from them.

Both come from public mirrors.  Everything is downloaded once and committed to the repo, so the
bot itself never needs network access.

Usage::

    python3 scripts/fetch_card_assets.py            # fetch whatever is missing
    python3 scripts/fetch_card_assets.py --force    # re-download everything
    python3 scripts/fetch_card_assets.py --only mega_knight,musketeer
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cr_bot.config.cards_db import CARDS_DB  # noqa: E402



CARD_URLS = (
    "https://cdn.royaleapi.com/static/img/cards/{slug}.png",
    "https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards/{slug}.png",
)

#: card-db slug -> file base name in the asset mirror (only where they differ)



ART_SIZE = (104, 124)      # (w, h) full-size portrait used by the docs / simulator
TEMPLATE_SIZE = (72, 86)   # (w, h) hand-slot templates: same aspect, cheap to load


def _fetch(url: str, timeout: float = 25.0) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "cr-bot/assets"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def _imdecode(data: Optional[bytes], flags: int) -> Optional[np.ndarray]:
    if not data:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def _trim_alpha(img: np.ndarray, pad: int = 2) -> Optional[np.ndarray]:
    """Crop an RGBA image to its opaque content (plus a small pad)."""
    if img.ndim != 3 or img.shape[2] != 4:
        return None
    alpha = img[:, :, 3]
    ys, xs = np.nonzero(alpha > 12)
    if ys.size == 0:
        return None
    y1, y2 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    x1, x2 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    return img[y1:y2, x1:x2]


def _save_png(path: Path, img: np.ndarray) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    path.write_bytes(buf.tobytes())
    return path.stat().st_size


def fetch_art(out_dir: Path, cards: Iterable[str], force: bool = False, verbose: bool = True) -> Dict[str, str]:
    made: Dict[str, str] = {}
    for card in sorted(cards):
        dest = out_dir / f"{card}.png"
        if dest.is_file() and not force:
            made[card] = dest.name
            continue
        for template in CARD_URLS:
            url = template.format(slug=card.replace("_", "-"))
            img = _imdecode(_fetch(url), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                # composite on the card's own dark frame is not needed: keep alpha as-is
                pass
            resized = cv2.resize(img, ART_SIZE, interpolation=cv2.INTER_AREA)
            size = _save_png(dest, resized)
            made[card] = dest.name
            if verbose:
                print(f"  art     {card:<16} -> {dest.name} ({size//1024} KB)")
            break
    return made


def build_hand_templates(art_dir: Path, out_dir: Path, force: bool = False, verbose: bool = True) -> int:
    """Re-cut ``assets/card_templates`` from the clean portraits.

    The shipped folder used to hold screenshots cropped out of a low-res emulator frame: blurred,
    with the elixir bubble and card border baked in.  Measured on the sample device frame those
    crops identified 1 of 4 hand cards correctly.  Deriving the templates from the official
    portrait art instead is both sharper and free of UI chrome.
    """
    made = 0
    for src in sorted(art_dir.glob("*.png")):
        dest = out_dir / src.name
        if dest.is_file() and not force:
            made += 1
            continue
        img = _imdecode(src.read_bytes(), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            flat = (img[:, :, :3].astype(np.float32) * alpha
                    + np.array([28, 28, 34], np.float32) * (1.0 - alpha))
            img = flat.astype(np.uint8)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_png(dest, cv2.resize(img, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA))
        made += 1
    if verbose:
        print(f"hand templates: {made} -> {out_dir}")
    return made


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Download official Clash Royale art for the vision stack.")
    ap.add_argument("--root", default=str(REPO_ROOT), help="repo root (assets/ lives here)")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    ap.add_argument("--only", default="", help="comma separated card slugs")
    ap.add_argument("--no-art", action="store_true", help="skip card portraits")
    args = ap.parse_args(argv)

    root = Path(args.root)
    art_dir = root / "assets" / "card_art"
    art_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        cards = [c.strip() for c in args.only.split(",") if c.strip()]
    else:
        cards = [c for c in CARDS_DB if not c.startswith("unknown_")]

    print(f"Fetching assets for {len(cards)} cards into {root / 'assets'}")
    manifest: dict = {"art": {}}
    if not args.no_art:
        manifest["art"] = fetch_art(art_dir, cards, force=args.force)
        print(f"card art: {len(manifest['art'])} cards -> {art_dir}")

    if not args.no_art:
        build_hand_templates(art_dir, root / "assets" / "card_templates", force=args.force)

    manifest_path = root / "assets" / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path}")

    total = sum(p.stat().st_size for p in root.joinpath("assets").rglob("*.png"))
    print(f"assets/card_* total {total/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
