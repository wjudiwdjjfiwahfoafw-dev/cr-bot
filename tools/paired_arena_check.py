#!/usr/bin/env python3
"""Reproducible, deliberately small *still-pair* arena diagnostic (no model training).

This is not video and the source labels may be incomplete. It exposes misses, wrong sides and
extra boxes rather than calling fewer detections an improvement. See the fixture README.
"""
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
from cr_bot.vision.arena_detector import HybridArenaDetector  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "paired_battle"


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def match_detections(detections, labels, threshold):
    """Greedy one-to-one box IoU matching, independent of name and side."""
    possibilities = sorted(
        ((iou(d.pixel_bbox, label["bbox"]), di, li)
         for di, d in enumerate(detections) for li, label in enumerate(labels)),
        reverse=True,
    )
    used_d, used_l, matches = set(), set(), []
    for overlap, di, li in possibilities:
        if overlap < threshold:
            break
        if di in used_d or li in used_l:
            continue
        used_d.add(di)
        used_l.add(li)
        d, label = detections[di], labels[li]
        matches.append({
            "label": label["name"], "expected_side": label["side"],
            "predicted_side": d.side.value, "predicted_name": d.name,
            "label_bbox": label["bbox"], "predicted_bbox": list(d.pixel_bbox),
            "iou": round(overlap, 3), "side_correct": d.side.value == label["side"],
        })
    misses = [label for i, label in enumerate(labels) if i not in used_l]
    extras = [{"bbox": list(d.pixel_bbox), "side": d.side.value, "name": d.name,
               "source": d.source} for i, d in enumerate(detections) if i not in used_d]
    return matches, misses, extras


def run(experimental=False):
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    base = cv2.imread(str(FIXTURES / manifest["background"]))
    if base is None:
        raise FileNotFoundError(manifest["background"])
    h, w = base.shape[:2]
    calibration = DeviceCalibrator().calibrate(base)
    grid = ArenaGrid(w, h, calibration.calibration)
    cases = []
    for case in manifest["cases"]:
        target = cv2.imread(str(FIXTURES / case["image"]))
        if target is None or target.shape != base.shape:
            raise ValueError(f"Missing or incompatible image: {case['image']}")
        detector = HybridArenaDetector(grid, yolo_weights_path=None, enable_recognition=False,
                                      experimental_overlay_normalization=experimental)
        for i in range(manifest["warm_frames"]):
            detector.detect_and_track(base, i * 0.08)
        for i in range(manifest["target_frames"]):
            detections = detector.detect_and_track(target, 1.0 + i * 0.08)
        matches, misses, extras = match_detections(detections, case["labels"],
                                                  manifest["minimum_iou"])
        cases.append({
            "name": case["name"], "image": case["image"],
            "labelled_units": len(case["labels"]), "matched_locations": len(matches),
            "correct_sides": sum(m["side_correct"] for m in matches),
            "wrong_sides": sum(not m["side_correct"] for m in matches),
            "misses": len(misses), "extras_vs_labels": len(extras),
            "matches": matches, "missed_labels": misses, "extra_detections": extras,
        })
    fields = ("labelled_units", "matched_locations", "correct_sides", "wrong_sides",
              "misses", "extras_vs_labels")
    return {
        "method": "six repeated baseline stills followed by six repeated target stills; IoU >= 0.15, one-to-one; names not scored",
        "warning": "Sparse correlated stills, not live video; labels may be incomplete. Extra boxes are relative to labels, not confirmed false positives.",
        "experimental_overlay_normalization": bool(experimental),
        "calibration_stage": calibration.stage,
        "summary": {key: sum(case[key] for case in cases) for key in fields},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental", action="store_true",
                        help="Compare opt-in, unvalidated overlay compensation")
    parser.add_argument("--json", type=Path, help="Also save the full report as JSON")
    args = parser.parse_args()
    report = run(experimental=args.experimental)
    print(report["warning"])
    print("experimental overlay compensation:", report["experimental_overlay_normalization"])
    print("case | labelled | located | correct side | wrong side | missed | extras vs labels")
    for case in report["cases"]:
        print("{name} | {labelled_units} | {matched_locations} | {correct_sides} | "
              "{wrong_sides} | {misses} | {extras_vs_labels}".format(**case))
    print("total:", report["summary"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("saved:", args.json)


if __name__ == "__main__":
    main()
