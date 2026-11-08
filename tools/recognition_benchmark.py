"""Evaluate open-set sprite naming on labelled *real gameplay* crops.

This measures naming of supplied boxes only, not detector recall, side, or performance on an
unseen match. See tests/fixtures/real_unit_crops/README.md for attribution and limitations.

    python tools/recognition_benchmark.py --verbose
    python tools/recognition_benchmark.py --split validation --json /tmp/naming.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cr_bot.vision.unit_recognizer import UnitRecognizer  # noqa: E402

DEFAULT_DATA = ROOT / "tests" / "fixtures" / "real_unit_crops"


def evaluate(directory: Path = DEFAULT_DATA, split: str = "all", recognizer=None,
             group: str = "all"):
    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    selected = [row for row in data
                if (split == "all" or row["split"] == split)
                and (group == "all" or row.get("evaluation_group") == group)]
    if not selected:
        raise ValueError(f"no examples for split {split!r}, group {group!r}")
    recognizer = recognizer if recognizer is not None else UnitRecognizer()
    if not recognizer.is_ready:
        raise RuntimeError("sprite library is unavailable")
    results = []
    for row in selected:
        image = cv2.imread(str(directory / row["path"]))
        if image is None:
            raise FileNotFoundError(directory / row["path"])
        h, w = image.shape[:2]
        started = time.perf_counter()
        # Source boxes are supplied as-is; they may include overlays or occlusions. Do not
        # pretend a bar is inside every box or trim off a fraction of the real sprite.
        found = recognizer.identify(image, (0, 0, w, h), has_bar=False)
        elapsed = 1000.0 * (time.perf_counter() - started)
        predicted = found.card if found else None
        status = "abstained" if predicted is None else ("correct" if predicted == row["card"] else "wrong")
        results.append({"image": row["path"], "truth": row["card"], "predicted": predicted,
                        "split": row["split"], "evaluation_group": row.get("evaluation_group"),
                        "status": status, "ms": round(elapsed, 2)})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all",
                        help="filter by the original dataset split")
    parser.add_argument("--group", choices=("all", "development", "additional"), default="all",
                        help="filter by crops used for tuning vs those collected afterwards")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--min-correct", type=int, default=0,
                        help="also fail unless at least this many crops are correctly named")
    args = parser.parse_args()
    if args.min_correct < 0:
        parser.error("--min-correct must be non-negative")
    results = evaluate(args.data, args.split, group=args.group)
    counts = Counter(row["status"] for row in results)
    for row in results:
        if args.verbose or row["status"] == "wrong":
            print(f"{row['image']:30s} truth={row['truth']:20s} "
                  f"predicted={str(row['predicted']):20s} {row['status']}")
    named = counts["correct"] + counts["wrong"]
    precision = f"{counts['correct'] / named:.1%}" if named else "n/a (no predictions)"
    print(f"real labelled crops: {len(results)} (group={args.group}, source split={args.split}); "
          f"correct={counts['correct']} wrong={counts['wrong']} abstained={counts['abstained']}")
    print(f"precision among named={precision}; mean={sum(r['ms'] for r in results)/len(results):.1f} ms/crop")
    print("This is a small open-set naming check, not full-frame detection recall.")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if counts["correct"] < args.min_correct:
        print(f"required at least {args.min_correct} correct names; got {counts['correct']}")
        return 1
    return int(args.fail_on_errors and counts["wrong"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
