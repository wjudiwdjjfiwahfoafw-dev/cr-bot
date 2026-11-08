# Labelled real screenshots

This directory is the ground truth for the vision pipeline: **real captures of an actual match**,
with the operator saying which troops are on the board. Nothing synthetic, no menu screens, no
images from search engines that might be renders.

Drop a capture here (PNG or JPG) plus a `.json` sidecar with the same stem:

```
samples/
  giant_push.png
  giant_push.json
```

The sidecar format (drop the `label` on a box when you do not care what it gets called):

```json
{
  "note": "1v1, 2x elixir, my Giant at the bridge",
  "expect": [
    {"name": "bar",
     "box": [99, 194, 189, 260],
     "card": "musketeer",
     "side": "ally"},
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
* **`side`** is `ally` or `enemy`. Ally = the bottom half of the screen.
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

and it prints per-screenshot misses / phantoms / duplicates / wrong side, plus the naming
accuracy, and writes an annotated copy to `samples/annotated/` so you can see exactly what the
bot thought it saw.

`validation` and `fps` runs (does the detection flicker frame to frame?) need a *sequence*: name
the files with an index (`giant_push_0001.png`, `giant_push_0002.png`, ...) and pass them in
order; the tool scores stability across consecutive frames.
