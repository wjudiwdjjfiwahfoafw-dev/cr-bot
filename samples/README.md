# Labelled real screenshots

`hand_still_report.json` and `paired_arena_default.json` / `paired_arena_experimental.json`
are reproducible diagnostic reports, not live-match accuracy measurements. Methods, caveats
and commands are in [SCREENSHOT_VERIFICATION.md](../docs/SCREENSHOT_VERIFICATION.md).

This directory holds real match captures and human-readable labels. Labels can be incomplete or
wrong, especially for occluded units; review the pixels before treating a score as ground truth.

Drop a capture here (PNG or JPG) plus a `.json` sidecar with the same stem:

```
samples/
  giant_push.png
  giant_push.json
```

The sidecar format (omit `card` when you do not know the identity):

```json
{
  "note": "1v1, 2x elixir, my Giant at the bridge",
  "expect": [
    {"name": "miner",
     "box": [52, 203, 143, 277],
     "card": "miner",
     "side": "ally", "partial": true},
    {"name": "mega knight",
     "box": [493, 262, 629, 400],
     "card": "mega_knight",
     "side": "ally"},
    {"name": "giant",
     "box": [63, 484, 241, 630],
     "card": "giant",
     "side": "enemy"}
  ]
}
```

Rules that make the numbers mean something:

* **`box` is pixel `[x1, y1, x2, y2]`** in the image as saved, around the *sprite* — do not
  include the health bar, the detection code crops that itself.
* **`card`** is the asset directory name from `assets/unit_sprites/` (`mega_knight`,
  `hog_rider`, `skeleton_army`, ...). Unknown names are reported as a problem, not silently
  skipped — a typo must never look like a pass.
* **`side`** is `ally` or `enemy`, according to team colour. A friendly troop can cross the
  river, so its screen position alone does not determine allegiance.
* **List everything you can see**, including troops you expect the bot to be wrong about. It is
  the misses and the phantoms that matter, and they can only be counted if the full board is
  written down.
* If a troop is only partly visible or hidden behind a tower, say so with
  `"partial": true` — it is still counted, but reported separately so a hard case does not get
  averaged into an easy one.

Then run:

```
python tools/verify_screenshots.py samples/*.png
```

and it prints per-screenshot misses / extra boxes / duplicates / wrong side, plus exact card
results. Add `--annotate samples/annotated --json samples/annotated/report.json` to save an
annotated copy and report. Extra boxes are only valid false positives if the labels are complete.

Each still is scored independently after inpainting its labelled boxes. Passing several stills
produces only a rough positional comparison; it does **not** run a continuous tracker or establish
real-sequence flicker, FPS or recall. Those require an annotated gameplay video and true prior frames.
