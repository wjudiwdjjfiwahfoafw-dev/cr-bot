#!/usr/bin/env python3
"""Download *real in-game* troop sprites so detection works on what the device actually shows.

Why this exists
---------------
The bot used to identify troops by matching the large promo renders from
``cdn.royaleapi.com/cards/<slug>.png`` against the arena.  Those renders are front-facing
studio shots; the arena shows small, top-down, team-tinted animation frames.  Measured on
two units whose identity is known (Mega Knight on the right lane, Giant on the left lane of
``calibration_preview.png``) the correct card ranked **34/78** — i.e. chance.  No amount of
score tuning fixes a pose mismatch.

This script instead pulls the animation frames that the game itself draws, from
``smlbiobot/cr-assets-png`` (decrypted ``*_tex.sc`` dumps of the APK).  Each character ships
hundreds of frames covering every facing direction, animation phase and *both team tints*;
we keep a small, diverse, non-empty subset per card:

    assets/unit_sprites/<card>/f00.png ... f07.png
    assets/unit_sprites/manifest.json

Everything is plain PNG + numpy.  No training, no labelling, no GPU: the "model" is the
game's own art.

Usage::

    python3 scripts/fetch_game_sprites.py                  # fetch missing cards
    python3 scripts/fetch_game_sprites.py --force          # rebuild everything
    python3 scripts/fetch_game_sprites.py --only mega_knight,giant --frames 12
    python3 scripts/fetch_game_sprites.py --contact-sheet  # also write assets/unit_sprites/_contact_sheet.png
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW = "https://raw.githubusercontent.com/smlbiobot/cr-assets-png/master/{path}"
TREE_API = "https://api.github.com/repos/smlbiobot/cr-assets-png/git/trees/master?recursive=1"
SPRITE_PREFIX = "assets/sc/"

#: card-db slug -> directory under assets/sc/ holding that character's animation frames.
#: Names are Supercell's internal names ("Axe Man" is the Executioner, "Zap Machine" Sparky,
#: "Black Knight" the Dark Prince, "Rage Barbarian" the Lumberjack, ...).
GAME_DIR: Dict[str, str] = {
    "archers": "chr_archer_out",
    "baby_dragon": "chr_baby_dragon_out",
    "balloon": "chr_balloon_out",
    "bandit": "chr_bandit_out",
    "barbarian_barrel": "chr_barbarian_out",
    "barbarians": "chr_barbarian_out",
    "bats": "chr_bats_out",
    "battle_ram": "chr_battle_ram_out",
    "bomber": "chr_bomber_out",
    "bowler": "chr_bowler_out",
    "cannon": "building_basic_cannon_out",
    "dark_prince": "chr_black_knight_out",
    "dart_goblin": "chr_goblin_blowdart_out",
    "electro_wizard": "chr_electro_wizard_out",
    "executioner": "chr_axe_man_out",
    "fire_spirit": "chr_fire_firespirit_out",
    "firecracker": "chr_goblin_archer_out",
    "giant": "chr_giant_out",
    "giant_skeleton": "chr_giant_skeleton_out",
    "goblin_gang": "chr_goblin_out",
    "goblin_hut": "building_goblin_hut_out",
    "goblins": "chr_goblin_out",
    "golem": "chr_golem_out",
    "guards": "chr_skeleton_warrior_out",
    "hog_rider": "chr_hog_rider_out",
    "ice_spirit": "chr_ice_spirits_out",
    "ice_wizard": "chr_ice_wizard_out",
    "inferno_tower": "building_inferno_tower_out",
    "knight": "chr_knight_out",
    "lava_hound": "chr_lava_hound_out",
    "lumberjack": "chr_rage_barbarian_out",
    "mega_knight": "chr_mega_knight_out",
    "mega_minion": "chr_mega_minion_out",
    "miner": "chr_miner_out",
    "mini_pekka": "chr_mini_pekka_out",
    "minion_horde": "chr_minion_out",
    "minions": "chr_minion_out",
    "mortar": "building_mortar_out",
    "musketeer": "chr_musketeer_out",
    "pekka": "chr_pekka_out",
    "prince": "chr_prince_out",
    "princess": "chr_princess_out",
    "royal_giant": "chr_royal_giant_out",
    "royal_hogs": "chr_hog_rider_out",
    "skeleton_army": "chr_skeleton_out",
    "skeletons": "chr_skeleton_out",
    "sparky": "chr_zap_machine_out",
    "tesla": "building_tesla_out",
    "tombstone": "building_tombstone_out",
    "valkyrie": "chr_valkyrie_out",
    "witch": "chr_witch_out",
    "wizard": "chr_wizard_out",
    "x_bow": "building_xbow_out",
    "bomb_tower": "building_bomb_tower_out",
}

Frame = Tuple[int, np.ndarray]  # (index, RGBA)


def _fetch_bytes(path: str, timeout: float = 25.0) -> Optional[bytes]:
    try:
        req = urllib.request.Request(RAW.format(path=path), headers={"User-Agent": "cr-bot/assets"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read() if resp.status == 200 else None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def list_frames(verbose: bool = True) -> Dict[str, List[str]]:
    """Map ``assets/sc/<dir>`` -> sorted list of frame paths, from one GitHub tree call."""
    try:
        req = urllib.request.Request(TREE_API, headers={"User-Agent": "cr-bot/assets"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            tree = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network
        print(f"! could not list the asset repo ({exc}); using directories already on disk")
        return {}
    out: Dict[str, List[str]] = {}
    for node in tree.get("tree", []):
        path = node.get("path", "")
        if not path.startswith(SPRITE_PREFIX) or not path.endswith(".png") or path.count("/") != 3:
            continue
        _, _, directory, _name = path.split("/")
        out.setdefault(directory, []).append(path)
    for frames_ in out.values():
        frames_.sort()
    if verbose:
        print(f"asset repo lists {len(out)} sprite directories / {sum(len(v) for v in out.values())} frames")
    return out


def _decode(data: Optional[bytes]) -> Optional[np.ndarray]:
    if not data:
        return None
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] < 4:
        return None
    return img[:, :, :4]


def _frame_stats(img: np.ndarray) -> Optional[dict]:
    """Silhouette statistics used to reject effects, detached limbs and cropped frames."""
    alpha = img[:, :, 3]
    mask = alpha > 24
    n = int(mask.sum())
    if n < 120:
        return None
    ys, xs = np.nonzero(mask)
    h0, w0 = alpha.shape[:2]
    y1, y2, x1, x2 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    bh, bw = y2 - y1 + 1, x2 - x1 + 1
    # a sprite that runs into the canvas border was clipped by the animation cell; skip it
    touches = y1 <= 1 or x1 <= 1 or y2 >= h0 - 2 or x2 >= w0 - 2
    return {
        "bbox": (x1, y1, x2 + 1, y2 + 1),
        "area": n,
        "box_area": bh * bw,
        "height": bh,
        "aspect": bw / float(bh),
        "coverage": n / float(max(1, bh * bw)),
        "touches_border": touches,
    }


def _trim(img: np.ndarray, bbox: Sequence[int], pad: int = 1) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    y1, x1 = max(0, y1 - pad), max(0, x1 - pad)
    y2, x2 = min(img.shape[0], y2 + pad), min(img.shape[1], x2 + pad)
    return img[y1:y2, x1:x2]


def _resize(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def _descriptor(img: np.ndarray) -> np.ndarray:
    """Low-res appearance descriptor used only to pick a *diverse* set of frames."""
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    h, w = img.shape[:2]
    canvas = np.zeros((64, 64), np.float32)
    s = min(64 / h, 64 / w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    small = cv2.resize(alpha * cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                       (nw, nh), interpolation=cv2.INTER_AREA)
    small_a = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
    oy, ox = (64 - nh) // 2, (64 - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = small
    amask = np.zeros((64, 64), np.float32)
    amask[oy:oy + nh, ox:ox + nw] = small_a
    vec = np.concatenate([canvas.ravel(), amask.ravel() * 0.5])
    vec -= vec.mean()
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-6 else vec


def select_diverse(frames: Sequence[Frame], keep: int, min_dist: float = 0.06) -> List[Frame]:
    """Greedy farthest-point selection: keeps a spread of poses / directions / tints."""
    if not frames:
        return []
    descs = [_descriptor(f[1]) for f in frames]
    areas = [float((f[1][:, :, 3] > 24).sum()) for f in frames]
    order = sorted(range(len(frames)), key=lambda i: -areas[i])
    chosen = [order[0]]
    while len(chosen) < min(keep, len(frames)):
        best_i, best_d = -1, -1.0
        for i in range(len(frames)):
            if i in chosen:
                continue
            d = min(1.0 - float(np.dot(descs[i], descs[j])) for j in chosen)
            if d > best_d:
                best_i, best_d = i, d
        if best_i < 0 or best_d < min_dist:
            break
        chosen.append(best_i)
    return [frames[i] for i in chosen]


def build_card(
    card: str,
    directory: str,
    frame_paths: Sequence[str],
    out_root: Path,
    frames_wanted: int,
    candidates: int,
    max_side: int,
    force: bool,
    verbose: bool,
) -> Optional[dict]:
    card_dir = out_root / card
    if card_dir.is_dir() and not force and any(card_dir.glob("*.png")):
        names = sorted(p.name for p in card_dir.glob("*.png"))
        return {"dir": directory, "frames": names, "cached": True}

    step = max(1, len(frame_paths) // candidates)
    picks = frame_paths[::step][:candidates]

    def grab(path: str) -> Frame:
        img = _decode(_fetch_bytes(path))
        if img is None:
            return (-1, np.zeros((1, 1, 4), np.uint8))
        try:
            idx = int(Path(path).stem.rsplit("_", 1)[-1])
        except ValueError:
            idx = 0
        return (idx, img)

    decoded: List[Frame] = []
    stats: List[dict] = []
    with futures.ThreadPoolExecutor(max_workers=12) as pool:
        for idx, img in pool.map(grab, picks):
            if idx < 0 or img.shape[0] < 8:
                continue
            st = _frame_stats(img)
            if st is None or st["touches_border"]:
                continue
            if not (0.45 <= st["aspect"] <= 2.2) or st["coverage"] < 0.34:
                continue
            decoded.append((idx, _trim(img, st["bbox"])))
            stats.append(st)
    if not decoded:
        if verbose:
            print(f"  ! {card:<18} no usable frames in {directory}")
        return None

    # keep frames whose silhouette is close to the character's typical size: this is what
    # removes the detached fists / weapon blur frames that would otherwise poison matching
    heights = sorted(st["height"] for st in stats)
    median_h = heights[len(heights) // 2]
    keep_mask = [0.80 * median_h <= st["height"] <= 1.25 * median_h for st in stats]
    decoded = [f for f, k in zip(decoded, keep_mask) if k]
    if not decoded:
        if verbose:
            print(f"  ! {card:<18} every frame rejected by the size filter")
        return None

    chosen = select_diverse(decoded, frames_wanted)
    card_dir.mkdir(parents=True, exist_ok=True)
    for old in card_dir.glob("*.png"):
        old.unlink()
    names: List[str] = []
    for i, (_idx, img) in enumerate(chosen):
        name = f"f{i:02d}.png"
        ok, buf = cv2.imencode(".png", _resize(img, max_side), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if ok:
            (card_dir / name).write_bytes(buf.tobytes())
            names.append(name)
    if verbose:
        sizes = ", ".join(f"{img.shape[1]}x{img.shape[0]}" for _i, img in chosen[:3])
        print(f"  {card:<18} {len(names)} frame(s) from {len(decoded)}/{len(picks)} candidates  ({sizes} ...)")
    return {"dir": directory, "frames": names}


def contact_sheet(out_root: Path, manifest: dict, path: Path, cell: int = 56) -> None:
    cards = sorted(manifest)
    cols = 12
    rows = (len(cards) + cols - 1) // cols
    sheet = np.full((rows * (cell + 14), cols * cell, 3), 40, np.uint8)
    for i, card in enumerate(cards):
        r, c = divmod(i, cols)
        tile = np.full((cell, cell, 3), 110, np.uint8)
        frame_name = manifest[card]["frames"][0]
        img = cv2.imread(str(out_root / card / frame_name), cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            s = min((cell - 4) / h, (cell - 4) / w)
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            oy, ox = (cell - img.shape[0]) // 2, (cell - img.shape[1]) // 2
            roi = tile[oy:oy + img.shape[0], ox:ox + img.shape[1]]
            tile[oy:oy + img.shape[0], ox:ox + img.shape[1]] = (
                img[:, :, :3] * a + roi * (1 - a)
            ).astype(np.uint8)
        sheet[r * (cell + 14):r * (cell + 14) + cell, c * cell:(c + 1) * cell] = tile
        cv2.putText(sheet, card[:11], (c * cell + 1, r * (cell + 14) + cell + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (230, 230, 230), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    print(f"contact sheet -> {path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Download real in-game troop sprites (no training).")
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help="comma separated card slugs")
    ap.add_argument("--frames", type=int, default=8, help="frames kept per card")
    ap.add_argument("--candidates", type=int, default=36, help="frames sampled per card before selection")
    ap.add_argument("--max-side", type=int, default=96)
    ap.add_argument("--contact-sheet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root)
    out_root = root / "assets" / "unit_sprites"
    out_root.mkdir(parents=True, exist_ok=True)

    cards = [c.strip() for c in args.only.split(",") if c.strip()] if args.only else list(GAME_DIR)
    listing = list_frames()
    if not listing and not args.only:
        print("! no frame listing available and no --only given")
        return 1

    manifest_path = out_root / "manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    print(f"Building sprite sets for {len(cards)} cards -> {out_root}")
    done = 0
    for card in cards:
        directory = GAME_DIR.get(card)
        if not directory:
            print(f"  ! {card:<18} has no mapped game directory")
            continue
        frame_paths = listing.get(directory) or []
        if not frame_paths:
            if (out_root / card).is_dir() and any((out_root / card).glob("*.png")):
                manifest.setdefault(card, {"dir": directory,
                                           "frames": sorted(p.name for p in (out_root / card).glob("*.png"))})
                continue
            print(f"  ! {card:<18} directory {directory} not found in the listing")
            continue
        entry = build_card(card, directory, frame_paths, out_root, args.frames, args.candidates,
                           args.max_side, args.force, verbose=True)
        if entry:
            manifest[card] = entry
            done += 1

    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in out_root.rglob("*.png"))
    print(f"{done} card(s) built, {len(manifest)} in manifest, {total/1e6:.2f} MB, manifest -> {manifest_path}")
    if args.contact_sheet:
        contact_sheet(out_root, {c: v for c, v in manifest.items() if v.get("frames")},
                      out_root / "_contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
