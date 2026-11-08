# This round: what changed, and how it was verified

All four complaints from the real-device runs (SM-A528B, Android 16, 1080x2400 → 720x1600) were
reproduced against the sample device frame shipped in the repo, then fixed, then measured.

`python -m pytest -q` → **209 passed**. `python tools/vision_benchmark.py` → recall **100 %**,
false positives **0.00/frame**, duplicates **0.00/frame**, side **100 %**, **~21 ms/frame**.

---

## 1. "Bounding boxes where there are no troops"

The boxes were not noise: they were **towers**, **a tower's ornamental ring**, and **the card
tray**. Five separate mechanisms, all measured on the real frame:

| Cause | Evidence | Fix |
|---|---|---|
| The bar reader accepted **white HP digits printed over a tower's bar** and a **tower's crown ring** as unit health bars | 33 "bars" on one real frame, most of them UI text | `HealthBarReader`: every candidate must be a *filled capsule* (`_is_solid`), not text, and must sit in the game's **dark UI outline** (`_has_dark_outline`). Real troop bars: 6; UI text rejected: all |
| Each phantom bar is a **Voronoi split point** inside a real blob | one Mega Knight → **3 boxes** | only capsule-backed bars may split a blob; a bare strip now folds into the sprite it is painted on |
| The **bars-only fallback** ran over the whole calibrated arena rectangle, which on a phone extends over the **card tray and elixir bar** | 4 of 7 "units" sat at *y ≈ 31 tiles* — inside the tray | fallback is bounded to the playfield (between the kings' rows), requires a capsule, and requires a bar ≥ 0.9 tiles wide |
| Tower exclusion used a **2.2-tile circle** | wider than the gap between a tower and the troop attacking it, so a Mega Knight at the princess tower was discarded | radius is the tower's own footprint (1.2 tiles) and one-sided in y — the tower's bar is drawn *on* its crown |
| One sprite → body + detached shoulder pad/shield | one Mega Knight → 2 boxes | fragment merge now allows a **bar-anchored** merge to grow to 5 tiles (a bar-less guess stays at 2.6), and a bar may only attach to a blob if it floats less than 0.85 tiles above it |

## 2. "Some cards are not detected"

* A troop **standing still for ~5 s was absorbed into the background** and disappeared. The
  caller now tells the model where a unit's bar is (`protect_from_absorb`), and protected pixels
  can never become scenery.
* **Off-by-one in the warm-up gate**: `_ensure()` bumps the frame counter, so the "settled" test
  was unsatisfiable and the model could never leave `UNKNOWN`. Fixed (`_frames_seen - 1`).
* Detections **on top of each other** are now collapsed by a rule that scales with the sprite's
  own width (`_dedupe`), keeping the best-evidenced one.
* A frame with **bars but no silhouettes** means the background model has gone stale; after 4
  such frames the segmenter re-learns the arena instead of reporting nothing.

## 3. "The fps is low"

Not the capture. The decode was fine; the *vision* was doing unnecessary work:

* `_has_dark_outline` did a **full-frame HSV conversion per bar candidate** → 150/frame.
* The background blend used **boolean fancy indexing on a float image**:
  `bg[free] += rate * (frame[free] - bg[free])` — three full-size temporaries, **14 ms** of a
  36 ms frame. Replaced with one `cv2.addWeighted` + `np.copyto(where=)`.
* The absorb pass re-labelled the whole arena every frame; it is now gated on "some region has
  actually been still for 60 frames".

Net: **36 ms → ~21 ms** per frame with *more* checks than before.

To answer "where does the time go" in future — including the HUD, which this build cannot
measure — the runner now profiles every stage (`cr_bot/perf.py`) and logs:

```
Frame cost: loop 24.1ms (42 fps) | vision 13.6ms (56%) | hud 6.2ms (26%) | capture 3.1ms (13%)
  bottleneck: vision (13.6 ms mean, 18.0 ms p95, 56% of the frame) — loop target is 12 Hz
```

## 4. "scrcpy always fails and falls back to screenrecord"

The symptom (server prints its `Device:` line, then no frame for 12 s) is what an Android 14+
device looks like when the abstract socket `localabstract:scrcpy` is unavailable. Two changes,
both in the retry ladder:

* a retry with a **fresh `scid`** (its own socket name, immune to a stale/half-dead predecessor);
* the retries get a **25 s handshake** instead of 12 s, so a cold hardware encoder is not thrown
  away.

Because this cannot be validated without the phone, `scripts/diagnose_scrcpy.py` was added: it
checks encoders, running servers, a fresh-scid launch, the server log, and `screenrecord` — and
prints the one thing to do about the first failure it explains.

## 5. Deploying, as requested

Cards are no longer dragged. **Tap the card, tap the tile** is now the default everywhere
(`deploy_style: "tap"`, config + CLI + code defaults). The existing `deploy_card()` path already
supported it; it just was not the default.

## 6. `--auto-queue` (found in the logs)

The log showed `Match started → Match ended (unknown) → Match started → Match ended (match_over)`
within nine seconds: **one odd frame ended the match**, which reset every tracker, counted a
match, and (with `--auto-queue`) started tapping lobby buttons mid-battle. Now:

* `PhaseStabilizer` — entering a battle is instant, leaving one needs the phase to hold for
  1 s (`battle_exit_confirm_s`); a stray frame inside a battle is ignored;
* `LobbyNavigator` — the queue walks **Battle → mode tile → Play**, one rate-limited tap at a
  time, instead of tapping Battle once and waiting.

## 7. Clicks in the viewer

Untestable without a GUI here, so the whole path is now covered by unit tests
(`tests/test_viewer_clicks.py`, 14 tests): double-clicks (some Qt/GTK builds deliver *only*
`EVENT_LBUTTONDBLCLK`), scaled/resized windows, sidebar clicks (ignored instead of being
teleported into the arena edge), card-then-arena deploys, right-click cancel, dry-run.

---

## How to re-measure

```bash
python -m pytest -q                        # 209 passed
python tools/vision_benchmark.py           # detection rates on the real frame
python scripts/diagnose_scrcpy.py -s RZCT20A0MBJ   # why scrcpy will not stream
python main.py --auto-queue                # look for the "Frame cost:" line every 30 s
```
