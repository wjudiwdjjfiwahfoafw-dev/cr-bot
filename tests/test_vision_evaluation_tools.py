"""Evaluation tools should exercise the requested pipeline and report abstentions honestly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from cr_bot.vision.unit_recognizer import Recognition
from tools import real_frame_check, recognition_benchmark, verify_screenshots

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples/calibration_preview.png"


class FakeRecognizer:
    is_ready = True

    def __init__(self, card=None):
        self.card = card
        self.calls = []

    def identify(self, frame, bbox, has_bar=True):
        self.calls.append((bbox, has_bar))
        return Recognition(self.card, .95, .5, .45, bbox) if self.card else None


def test_screenshot_evaluator_uses_injected_recognizer_and_scores_wrong_names():
    recognizer = FakeRecognizer("knight")
    score = verify_screenshots.score_frame(SAMPLE, recognizer=recognizer)
    assert len(recognizer.calls) == 1  # one naming attempt per frame
    assert score.names_right == 0
    assert len(score.names_wrong) == 1
    assert score.names_wrong[0].endswith(" -> knight")
    assert len(score.names_unnamed) == 2


def test_screenshot_evaluator_can_really_disable_naming():
    recognizer = FakeRecognizer("knight")
    score = verify_screenshots.score_frame(SAMPLE, recognizer=recognizer,
                                           enable_recognition=False)
    assert not recognizer.calls
    assert len(score.names_unnamed) == 3
    assert not score.names_wrong


def test_no_naming_cli_does_not_construct_recognizer(monkeypatch, capsys):
    def forbidden():
        raise AssertionError("--no-naming must not instantiate the recognizer")
    monkeypatch.setattr(verify_screenshots, "UnitRecognizer", forbidden)
    monkeypatch.setattr(sys, "argv", ["verify_screenshots", str(SAMPLE), "--no-naming"])
    assert verify_screenshots.main() == 0
    assert "troops found       3" in capsys.readouterr().out


def test_real_crop_benchmark_uses_supplied_boxes_and_filters_splits(tmp_path):
    rows = [
        {"path": "a.png", "card": "knight", "split": "train"},
        {"path": "b.png", "card": "giant", "split": "validation"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(rows))
    for row in rows:
        cv2.imwrite(str(tmp_path / row["path"]), np.zeros((30, 40, 3), np.uint8))
    recognizer = FakeRecognizer("knight")
    results = recognition_benchmark.evaluate(tmp_path, "validation", recognizer)
    assert len(results) == 1 and results[0]["status"] == "wrong"
    assert recognizer.calls == [((0, 0, 40, 30), False)]


def test_benchmark_can_require_positive_correct_coverage(monkeypatch, capsys):
    row = {"image": "x.png", "truth": "giant", "predicted": None,
           "split": "validation", "status": "abstained", "ms": 1.0}
    monkeypatch.setattr(recognition_benchmark, "evaluate", lambda *args, **kwargs: [row])
    monkeypatch.setattr(sys, "argv", ["recognition_benchmark", "--fail-on-errors", "--min-correct", "1"])
    assert recognition_benchmark.main() == 1
    text = capsys.readouterr().out
    assert "precision among named=n/a (no predictions)" in text
    assert "required at least 1 correct names" in text


def test_real_crop_regression_includes_open_set_negatives_and_reports_abstentions():
    results = recognition_benchmark.evaluate()
    assert len(results) == 55
    assert {r["split"] for r in results} == {"train", "validation", "test"}
    assert sum(r["evaluation_group"] == "additional" for r in results) == 29
    # Some labels are imperfect and some classes are absent from the template set. This guards
    # against the specific confident cross-class guesses, not a claim of usable recall.
    assert not [r for r in results if r["status"] == "wrong"]


def test_jitter_keeps_dimensions_and_unlabelled_report_does_not_claim_zero_false_positives():
    image = np.zeros((40, 60, 3), np.uint8)
    image[10:15, 10:15] = 255
    assert real_frame_check._jitter(image, 0).shape == image.shape
    report = real_frame_check.FrameReport(Path("x.png"), (60, 40), calibrated=True)
    text = real_frame_check.format_report(report)
    assert "no labels: false positives and misses not evaluated" in text
    assert "extra -" not in text


def test_screenshot_evaluator_fails_when_calibration_fails_even_with_no_labels(monkeypatch):
    score = verify_screenshots.FrameScore(image=Path("bad.png"), labels=[], detections=[],
                                          calibrated=False)
    monkeypatch.setattr(verify_screenshots, "score_frame", lambda *args, **kwargs: score)
    monkeypatch.setattr(sys, "argv", ["verify_screenshots", "bad.png", "--no-naming"])
    assert verify_screenshots.main() == 1


def test_unlabelled_frame_check_still_fails_if_calibration_failed(monkeypatch):
    report = real_frame_check.FrameReport(Path("bad.png"), (60, 40), calibrated=False)
    monkeypatch.setattr(real_frame_check, "run_frame", lambda *args, **kwargs: report)
    monkeypatch.setattr(sys, "argv", ["real_frame_check", "bad.png"])
    assert real_frame_check.main() == 1
