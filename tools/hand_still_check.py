#!/usr/bin/env python3
"""Small attributed, labelled hand-still check; not representative live accuracy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cr_bot.arena.device_calibrator import DeviceCalibrator  # noqa: E402
from cr_bot.arena.grid import ArenaGrid  # noqa: E402
from cr_bot.vision.hand_detector import HandCardDetector  # noqa: E402

FIX = ROOT / "tests/fixtures/paired_battle"
HOG = ["hog_rider", "skeletons", "cannon", "ice_spirit", "musketeer", "ice_golem", "fireball", "the_log"]
BAIT = ["mega_knight", "miner", "inferno_dragon", "skeleton_barrel", "zap", "bats", "spear_goblins", "goblin_gang"]
CASES = [
    (ROOT / "samples/calibration_preview.png", BAIT, ["skeleton_barrel", "bats", "spear_goblins", "zap"]),
    (ROOT / "tests/fixtures/real_frames/real_solo_battle_empty.png", HOG, ["hog_rider", "skeletons", "cannon", "ice_spirit"]),
    (FIX / "0RqoBtTgewOXP37lNwMI.jpg", HOG, ["fireball", "skeletons", "ice_golem", "hog_rider"]),
    (FIX / "04FED5slLByee0jMFkCR.jpg", HOG, ["ice_spirit", "skeletons", "cannon", "musketeer"]),
    (FIX / "0J09bneS0qzybsMz6ypX.jpg", HOG, ["ice_spirit", "skeletons", "cannon", "musketeer"]),
    (FIX / "0JPlKsDPop5jfLGdqEya.jpg", HOG, ["fireball", "skeletons", "cannon", "musketeer"]),
]


def run():
    rows = []
    counts = {"correct": 0, "wrong": 0, "unknown": 0}
    for path, deck, expected in CASES:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        cal = DeviceCalibrator().calibrate(image)
        grid = ArenaGrid(image.shape[1], image.shape[0], cal.calibration)
        detector = HandCardDetector(grid, deck, ROOT / "assets/card_templates")
        actual = detector.detect_hand(image, 10.0)
        slots = []
        for label, got in zip(expected, actual):
            outcome = "unknown" if got.card_name == "unknown" else ("correct" if got.card_name == label else "wrong")
            counts[outcome] += 1
            slots.append({"expected": label, "predicted": got.card_name, "confidence": got.confidence,
                          "is_ready_at_elixir_10": got.is_ready, "outcome": outcome})
        rows.append({"image": str(path.relative_to(ROOT)), "calibration_stage": cal.stage, "slots": slots})
    return {
        "warning": "Six hand stills only; four are correlated frames from one match. Fixed templates were taken from independent screenshots. Not representative or a live test.",
        "summary": {**counts, "total": sum(counts.values())},
        "frames": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Save the complete report as JSON")
    args = parser.parse_args()
    report = run()
    print(report["warning"])
    for row in report["frames"]:
        print(row["image"], [(slot["expected"], slot["predicted"]) for slot in row["slots"]])
    print("summary:", report["summary"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("saved:", args.json)


if __name__ == "__main__":
    main()
