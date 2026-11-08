# Vision verification and limitations

## Reproduce

Run from the repository root after installing `requirements.txt`:

```bash
python -m pytest -q
python tools/verify_screenshots.py samples/calibration_preview.png --annotate samples/annotated --json samples/annotated/report.json
python tools/verify_screenshots.py samples/calibration_preview.png --no-naming
python tools/recognition_benchmark.py --group development --verbose
python tools/recognition_benchmark.py --group additional --verbose
python tools/sprite_smoke_benchmark.py --verbose
python tools/vision_benchmark.py --frames 10 --quiet
python tools/real_frame_check.py tests/fixtures/real_frames/real_solo_battle_empty.png
```

`recognition_benchmark.py --fail-on-errors` fails on any wrong card but **does not** require a
correct one. Use `--min-correct 1` as well to demand positive coverage; it currently fails.

## Real gameplay crops: naming remains an open problem

We used 55 source-annotated crops from 15 screenshots by Nejc Zavodnik, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The creator warns that labels are inaccurate; some crops contain other units, effects, or
structures. They come from spectator-layout screenshots and may share matches. Attribution,
source image IDs, original boxes and splits are in [`tests/fixtures/real_unit_crops`](../tests/fixtures/real_unit_crops/README.md).
None of these crops are supplied as templates or training data to the shipped recognizer. The
per-crop results for the original recognizer at commit `b168a42` are in
[`baseline_report.json`](../tests/fixtures/real_unit_crops/baseline_report.json).

| Supplied real crops | Earlier matcher | Current matcher |
|---|---:|---:|
| 26 development crops used to choose the gates | 0 correct, 2 wrong, 24 abstained | 0 correct, 0 wrong, 26 abstained |
| 29 additional crops collected after the gates were set | 0 correct, 1 wrong, 28 abstained | 0 correct, 0 wrong, 29 abstained |
| **Total 55** | **0 correct, 3 wrong, 52 abstained** | **0 correct, 0 wrong, 55 abstained** |

This is evidence of fewer wrong labels in this sample, **not** proof of precision: there were no
current predictions to calculate precision from, and real-crop naming recall is zero here. The
additional set was not used for those gates, but is small, imperfectly labelled and not demonstrably
independent of the development set. A larger held-out set of labelled gameplay sequences is needed.
The crop benchmark measures naming given a box, not full-frame detection or side accuracy.

As a positive-path **synthetic** smoke test, 20 known library templates were pasted onto an arena
background: 8 correct, 0 wrong, 12 abstained. The recognizer is matching images it already contains,
so that result is not a real-world accuracy estimate. An integration test also shows that one
successful Mega Knight match reaches the tracker.

## One training-camp still: localization and side

The included `samples/calibration_preview.png` is a real 720×1280 still. The sidecar labels three
visible locations: a red-bar **enemy Giant**, a blue-bar **ally Mega Knight**, and a partially
occluded blue-bar **ally Miner**. The original sidecar incorrectly marked all three as enemies and
called the Miner a Musketeer; the revised label reflects inspection of the visible pixels. The
Miner is partly hidden and the labels may not be exhaustive.

A background-subtraction detector needs a previous frame, which the still cannot provide. The
verification script therefore inpaints those three labelled regions to make an *artificial* prior
background, warms the detector, and reveals the original still. In one run it matched all **3/3
labelled locations**, with **0 side disagreements, 0 extra boxes and 0 duplicates relative to
those labels**, and **0/3 names**. This is a single inpaint-and-reveal scenario, not a measurement
of real-stream recall or false-positive rate. Timing on this machine was approximately **123 ms**
with one sprite attempt and **41 ms** without naming; timing varies and does not guarantee live FPS.
The 40 ms recognition budget is soft: one synchronous match can exceed it, and initial asset loading
adds a startup cost.

`real_frame_check.py` similarly cannot recover a real prior background from a still. It uses
fixed-size artificial translations to stress jitter; without explicit expectations it now reports
false positives and misses as **not evaluated**, not as zero. Comparing multiple stills in
`verify_screenshots.py` is only a position comparison, not continuous tracker evaluation.

## Changes motivated by the data

* Prevent tower-colour calibration from treating nearby troops or floor decoration as tower bars;
  require multiple outlined, suitably positioned bars on each side across consecutive frames.
  If evidence is insufficient, retain the default colour mapping.
* Match only measured segmentation boxes, rotate one naming attempt per frame, reject shared
  sprite frames and weak/partial matches, and preserve detection confidence separately from naming.
* Keep tentative bar-only tracks out of the brain until a second observation and expire them
  sooner. Preserve specific tracker names across generic observations and avoid an immediate
  switch on a single conflicting OpenCV name.
* Do not self-train from guessed labels. Experimental online identity sampling is **off by default**;
  when enabled it queues a verified hand change for the vision thread and applies scene checks.
  Even then, a spatial association is not a human-confirmed label.

For meaningful deployment validation, record real consecutive gameplay frames (including empty
arena warm-up, both sides, varied cards, occlusion, effects and different devices), annotate all
visible objects and sides, and measure recall, precision, identity changes and end-to-end frame
latency on matches kept separate from tuning. No device or live match was available for this run.
