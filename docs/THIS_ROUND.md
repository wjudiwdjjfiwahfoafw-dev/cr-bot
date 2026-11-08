# Vision update: what changed and what remains open

No machine-learning or reinforcement-learning training was added. Hand recognition uses fixed
image templates; the OpenCV arena detector uses heuristics and a background model. Optional
persistent identity sampling is still **off by default**. The configured evolution cards are
**Mega Knight and Skeleton Barrel**; the in-game deck must also be equipped accordingly.
The simulator tracks these cycles but does not yet model Evo Skeleton Barrel's double-drop effect.

## Hand and calibration

* Search several vertical offsets for a lifted card. Compare only the configured deck, including
  independently sourced gameplay-portrait variants, and require a score and runner-up margin.
  Grey cards use a stricter structural comparison. An unknown/empty slot is never marked ready.
* Do **not** infer the fourth card from the other three in an eight-card deck. Template loading
  resolves the longest known card name first, so an off-deck `giant_skeleton` reference cannot be
  mistaken for `giant`. Independent portrait sources are credited in
  [`assets/card_templates/REFERENCES.md`](../assets/card_templates/REFERENCES.md).
* River detection recognises narrow cyan, magenta and lava-like bands rather than the largest
  cyan region; three/four regularly spaced badges can locate the hand, with resolution-scaled
  badge dimensions. Unusual arenas and heavily obscured badges can still defeat calibration.
* Old unversioned **automatic** calibrations are remeasured. Manual calibrations are preserved.
  If automatic calibration has no reliable hand row or insufficient arena evidence, it waits and
  retries instead of saving incomplete geometry or making a deployment. It also skips decisions
  on the frame processed before a successful calibration rebuild. `--no-calibrate` explicitly
  permits default geometry, which may not match the device. Inspect the preview.

## Arena changes and remaining failure

* The fallback segmentation ROI avoids some HUD area without changing the tile projection.
  Synthetic edge-sprite checks cover a few crop boundaries, not all edge units or layouts.
* Fixed a foreground-at-corner flood-fill bug and an ROI-coordinate bug in motion-only team
  classification. A persistent global-shift guard reduces one-pixel jitter artifacts in a
  fixed-size synthetic translation test, but can suppress units during widespread changes.
* Require outlined, team-coloured bar candidates for segmentation anchors; do not give a tower's
  own wide bar automatic troop authority. Bars-only filtering rejects very wide UI strips and a
  visible white emote bubble. These heuristics can miss troops overlapping towers or in unusual
  arenas; fewer boxes alone is not evidence of good recall.
* Photometric compensation for card-selection overlays is included solely behind
  `vision.experimental_overlay_normalization: true`. It is **off by default** because it increased
  extra detections and has not been validated on video. Its line removal/opening can erase small
  troops. The default path can miss many units on overlay transitions too.

## Measurements in this workspace

* `python -m pytest -q`: **323 passed** (no device involved).
* `python tools/hand_still_check.py`: **23 correct, 0 wrong, 1 unknown / 24**; before these changes,
  the same six frames produced 14/24 including an unsafe elimination-based Barrel guess. Four
  frames come from one match. Separate checks cover an empty slot, a grey Cannon, an unknown grey
  Musketeer, a raised card and off-deck portraits. This is a small, non-representative still set.
* `python tools/paired_arena_check.py`: default **1/8 labelled locations**, correct side for that
  one, **7 missed**, and **8 extra boxes relative to the labels**. With `--experimental`: **6/8
  locations**, **5 correct sides, 1 wrong side, 2 missed, 19 extras**. Each pair warms on six
  repeats of one selected-card still, then repeats a different still six times; it is not video.
  All cases are from one match and labels may be incomplete. Reports, boxes and attribution are
  included in `samples/paired_arena_*.json` and `tests/fixtures/paired_battle/`.
* On 55 source-labelled gameplay crops, the troop recognizer produced **0 names, 0 wrong names,
  55 abstentions**; real naming recall is still zero on this small test. The synthetic 8/20 sprite
  test matches the recognizer's own artwork and is only an integration smoke test.
* One training-camp still with a deliberately inpainted prior background yielded 3/3 labelled
  locations with correct visible sides, no extras relative to possibly incomplete labels, and
  0/3 names. This cannot establish live recall or performance.

There was **no connected phone and no continuous live-match test**. Arena reliability remains an
open issue, not a completed accuracy improvement. See [SCREENSHOT_VERIFICATION.md](SCREENSHOT_VERIFICATION.md)
for methods and suggested validation.
