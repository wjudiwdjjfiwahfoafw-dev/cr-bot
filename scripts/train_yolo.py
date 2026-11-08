#!/usr/bin/env python3
"""
Train a custom YOLOv8 Clash Royale Arena Detector and export to ONNX + .names sidecar
for instant zero-overhead inference inside `HybridArenaDetector`.

Compatible with any YOLO-formatted Clash Royale dataset (e.g. Roboflow Clash Royale / KataCR):
  dataset/
    data.yaml
    train/images, train/labels
    valid/images, valid/labels

Usage:
    python scripts/train_yolo.py --data path/to/data.yaml --epochs 80 --imgsz 640
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8n Clash Royale detector & export ONNX.")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base Ultralytics model (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--out",
        default="weights/clash_yolov8n.onnx",
        help="Destination ONNX weights path loaded by configs/mk_zap_bait.yaml",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("Install ultralytics first: pip install ultralytics onnx")
        return 1

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    onnx_exported = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_exported, out_path)

    names_path = out_path.with_suffix(".names")
    ordered_names = [model.names[i] for i in sorted(model.names.keys())]
    names_path.write_text("\n".join(ordered_names) + "\n", encoding="utf-8")

    print(f"Exported ONNX model  -> {out_path}")
    print(f"Exported class names -> {names_path} ({len(ordered_names)} classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
