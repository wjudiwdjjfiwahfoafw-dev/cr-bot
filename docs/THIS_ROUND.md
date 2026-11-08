# Current vision changes

The detector already had an invocation of `UnitRecognizer`; this revision repairs and tests the
end-to-end path rather than adding a second call. The recognizer now only examines measured
segmentation boxes, schedules one attempt per frame with round-robin fairness, and rejects low
scores, small partial matches, missing independent competitors and shared artwork. YOLO/OpenCV
fusion and tracking preserve supported names without turning a single conflicting OpenCV guess
into an immediate identity switch. `vision.unit_recognition: false` turns this pass off.

Real screenshot inspection showed that the old tower-hue calibration had learned from troops and
scenery and inverted the sample's team labels. Calibration now requires corroborated outlined tower
bars and resets between matches. Bar-only tracks require two hits and expire sooner. Detection
confidence is no longer increased merely because a classifier returned a name.

Automatic appearance learning is now **disabled by default**. When explicitly enabled, a verified
hand change queues a candidate for the main vision thread. It excludes spells, buildings and
multi-unit cards, requires an empty prior drop neighbourhood and a unique, new, high-confidence
allied segmentation track in two frames, and persists only eligible samples. This association is
still heuristic, not a confirmed label; older memory files are isolated by a new device-and-size
namespace. Memory writes are atomic and corrupt/non-finite samples are rejected on load.

## Measured here

* `python -m pytest -q` — **281 passed** in the current environment.
* `python tools/recognition_benchmark.py` — 55 attributed real gameplay crops: **0 correct, 0 wrong,
  55 abstained**, versus **0 correct, 3 wrong, 52 abstained** before these changes. Naming is not
  useful on this test despite avoiding the observed wrong guesses.
* `python tools/sprite_smoke_benchmark.py` — 8/20 correct on synthetic pastes of the same templates;
  this is a positive integration smoke test, not a real-world accuracy claim.
* `python tools/verify_screenshots.py samples/calibration_preview.png` — 3/3 labelled locations,
  correct visible sides, no extras or duplicates relative to the labels, 0/3 names, **with an
  artificial inpainted background**. The labels include a partly hidden unit and may be incomplete.

The 55-crop data is small, source-labelled and imperfect; 29 crops were collected after selecting
the gates but may come from related matches. A screenshot is not a continuous gameplay sequence.
There was no phone or live-match validation. See [SCREENSHOT_VERIFICATION.md](SCREENSHOT_VERIFICATION.md)
for methods, commands, attribution, timings and next steps.
