"""Synthetic positive-path smoke test for the sprite recognizer.

This pastes the library's OWN templates onto one arena background. It is deliberately easy and
must not be interpreted as accuracy on real gameplay, unseen poses, or real detector boxes.

    python tools/sprite_smoke_benchmark.py --verbose --json /tmp/sprite-smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cr_bot.vision.unit_recognizer import UnitRecognizer  # noqa: E402

CARDS = ("giant", "mega_knight", "musketeer", "baby_dragon", "witch", "bandit", "hog_rider",
         "knight", "ice_wizard", "valkyrie", "bats", "dart_goblin", "mortar",
         "giant_skeleton", "balloon", "mini_pekka", "royal_giant", "tesla", "fire_spirit",
         "skeletons")


def evaluate():
    background = cv2.imread(str(ROOT / "tests/fixtures/real_frames/real_solo_battle_empty.png"))
    if background is None:
        raise FileNotFoundError("empty-arena smoke-test fixture is missing")
    recognizer = UnitRecognizer()
    if not recognizer.is_ready:
        raise RuntimeError("sprite library is unavailable")
    results = []
    for card in CARDS:
        paths = sorted((ROOT / "assets/unit_sprites" / card).glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"no sprite for {card}")
        sprite = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
        if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
            raise ValueError(f"invalid RGBA sprite {paths[0]}")
        sprite = cv2.resize(sprite, (round(sprite.shape[1] * 100 / sprite.shape[0]), 100))
        h, w = sprite.shape[:2]
        crop = background[350:350+h, 350:350+w].copy()
        alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
        crop = (sprite[:, :, :3] * alpha + crop * (1.0 - alpha)).astype(np.uint8)
        start = time.perf_counter()
        match = recognizer.identify(crop, (0, 0, w, h), has_bar=False)
        elapsed = 1000.0 * (time.perf_counter() - start)
        prediction = match.card if match else None
        status = "abstained" if prediction is None else ("correct" if prediction == card else "wrong")
        results.append({"truth": card, "predicted": prediction, "status": status,
                        "ms": round(elapsed, 2)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = evaluate()
    if args.verbose:
        for result in results:
            print(f"{result['truth']:20s} -> {str(result['predicted']):20s} {result['status']}")
    counts = Counter(result["status"] for result in results)
    print(f"synthetic pastes: {len(results)}; correct={counts['correct']} "
          f"wrong={counts['wrong']} abstained={counts['abstained']}")
    print("Smoke test uses the same templates as the recognizer; not real-world accuracy.")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return int(args.fail_on_errors and counts["wrong"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
