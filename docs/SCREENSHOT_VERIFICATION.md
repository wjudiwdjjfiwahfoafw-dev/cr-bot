# Verifying the vision on real screenshots

## What counts as evidence

Only **real in-match captures**. Not menu screens, not store art, not generated images: the
pipeline is being judged on the frames it will actually see, and a menu screen silently measures
a different problem (it calibrates, finds no arena, and "passes").

`calibration_preview.png` (720x1280, trainer match) is the one hand-labelled real frame in the
repo, with its ground truth in `samples/calibration_preview.json`. Everything else that was
collected from image search during this round was discarded — it was either generated artwork or
a collection/menu screen.

## Running it

```
python tools/verify_screenshots.py samples/*.png --annotate samples/annotated
python tools/real_frame_check.py  shot.png --out /tmp/seen     # what did it find, no labels
python -m pytest -q                                            # 220 tests
python tools/vision_benchmark.py --frames 40                   # synthetic, per-frame timing
```

`samples/README.md` describes the `.json` sidecar format for a new screenshot.

## The one thing a screenshot cannot show, and what the tool does instead

The detector is background subtraction: it learns the arena, then reports what changed. A single
screenshot has no "before", so two traps have to be avoided:

* feeding it the frame itself makes the troops part of the learned floor — the board then reads as
  empty, which is correct behaviour and a useless test;
* feeding it the same frame repeatedly is not a scenario either.

So `verify_screenshots.py` reconstructs the empty board by **inpainting the labelled sprites out**
and warms the model on that, then presents the real frame: "this was the arena, now the troops
arrive". It uses only the human labels, never the detector's own answer.

## Result on the real frame (2026-11-08)

| | |
|---|---|
| troops labelled | 3 (Giant, Mega Knight, Musketeer — all enemy) |
| found | 3 (100%) |
| phantoms | 0 |
| duplicates | 0 |
| wrong side | 0 |
| names correct | 0/3 |
| cost | ~25 ms/frame detection, +~40 ms naming |

Size classes and positions are right (`unknown_tank` 3.4 tiles tall for the Giant, `unknown_light`
for the Musketeer). Naming is the open gap — see below.

## Defects this evidence found, and the fixes

| symptom on a real capture | cause | fix |
|---|---|---|
| 20–40 phantom troops on one frame | 1–2 px viewpoint wobble (stream, hand-held phone) makes every edge differ from the learned background | estimate the shift by phase correlation at half scale, require the same estimate twice in a row before acting, freeze the model while moved, and treat "motion over >10% of the arena" as a camera move |
| empty arena reported a "unit" | a tower's level badge ("16" in a blue pill) reads as a health bar | bar must be ≥4 px tall, ≥60 HSV saturation, <8% near-white pixels (badges 11–22%, real bars 0–2%) |
| every enemy called an ally | ally/enemy colour is *not* constant: on this frame our towers draw red and the enemy's blue | learn the team colours each frame from the six known tower positions |
| detected troop vanished after a second | a unit that stands still is absorbed into the background | a health bar is never absorbed |
| sprite boxes were much larger than the sprite | box came from the merged blob union, not the sprite contour | tighten to the contour's own bounds |
| a Giant called "bats" | the matcher fitted every template to the box height and only searched one scale | (partially fixed) |

## Open problem: naming

Naming a troop from an in-game sprite is not working on real frames, and it is worth being
straight about why:

* the sprite library holds 53 cards x ~6 frames with **rotated, tumbling and mirrored poses**
  (the Giant's frames include him upside-down), and the real Giant is upright;
* the live sprite is **tinted** (red glow for enemy, blue for ally) so greyscale correlation
  degenerates; gradient-magnitude was tried and is worse;
* the sprite is partly **occluded** by its own health bar, its level pill and the ground shadow;
* global colour-histogram matching puts the real Giant at rank 18 of 53, the Mega Knight at 45 —
  no combination of cheap features tested (masked ZNCC at 3 scales x 9 offsets, canonical-height
  normalisation, colour histograms, gradient magnitude) ranks the truth first;
* on synthetic pastes of the *same* library sprites the identical matcher gets 93% top-1, which
  is why this looked solved. Real captures are a different distribution.

What does still work is everything the brain needs to defend: position (tile-accurate), side,
size class, HP. Naming should be treated as an optional extra that only speaks when it is sure,
which is what `UnitRecognizer` does (score ≥0.60 and a margin over the runner-up; identical-art
cards such as Skeleton Army vs Skeletons are treated as aliases, not as ambiguity).

Next candidates, in order of expected value: match against the *deck* only (a 1v1 deck is 8 cards,
not 53) once the opponent's cards have been seen; use the in-game sprite at the scale the box
implies with a rotation-search restricted to the handful of upright frames; and keep the
`UnitIdentityMemory` path, which learns from the bot's own verified deploys rather than the library.
