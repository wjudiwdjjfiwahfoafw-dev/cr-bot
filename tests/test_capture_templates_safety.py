"""Capturing hand references should not poison calibration or overwrite shipped templates."""
from pathlib import Path
import sys

import cv2
import numpy as np

from cr_bot.arena.device_calibrator import calibration_path
import scripts.capture_templates as capture

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures/paired_battle"


def config(tmp_path):
    path = tmp_path / "capture.yaml"
    path.write_text(
        'deck: [hog_rider, skeletons, cannon, ice_spirit, musketeer, ice_golem, fireball, the_log]\n'
        f'vision:\n  templates_dir: "{tmp_path / "templates"}"\n'
        f'calibration:\n  dir: "{tmp_path / "calibrations"}"\n'
    )
    return path


def test_capture_preserves_existing_art_and_creates_device_variants(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    out = tmp_path / "templates"
    out.mkdir()
    original = out / "fireball.png"
    assert cv2.imwrite(str(original), np.full((40, 40, 3), 55, np.uint8))
    original_bytes = original.read_bytes()
    image = FIX / "0RqoBtTgewOXP37lNwMI.jpg"
    root_preview = ROOT / "calibration_preview.png"
    previous_preview = root_preview.read_bytes()
    for _ in range(2):
        monkeypatch.setattr(sys, "argv", ["capture_templates.py", "--config", str(cfg), "--image", str(image)])
        answers = iter(["fireball", "n", "s", "s", "s"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
        assert capture.main() == 0
    assert original.read_bytes() == original_bytes
    assert (out / "fireball_device.png").is_file()
    assert (out / "fireball_device2.png").is_file()
    assert (out / "_capture_calibration_preview.png").is_file()
    assert root_preview.read_bytes() == previous_preview
    assert not list(out.glob("_raw_slot*"))


def test_incomplete_hand_never_saves_a_bad_calibration(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    image = FIX / "17fmmbFkR9sklKs725OJ.jpg"  # only two pink badges
    monkeypatch.setattr(sys, "argv", ["capture_templates.py", "--config", str(cfg), "--image", str(image)])
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt")))
    assert capture.main() == 1
    assert not list((tmp_path / "calibrations").glob("*.json"))


def test_connected_device_uses_resolved_serial_for_its_calibration(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    image = cv2.imread(str(FIX / "0RqoBtTgewOXP37lNwMI.jpg"))
    state = {}
    class FakeADB:
        def __init__(self, **kwargs):
            self.serial = None
        def connect(self):
            self.serial = "actual-device"
        def capture_frame(self):
            return image
        def close(self):
            state["closed"] = True
    monkeypatch.setattr(capture, "ADBController", FakeADB)
    monkeypatch.setattr(sys, "argv", ["capture_templates.py", "--config", str(cfg)])
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    assert capture.main() == 0
    assert state["closed"]
    assert calibration_path(tmp_path / "calibrations", "actual-device", 608, 1080).is_file()
