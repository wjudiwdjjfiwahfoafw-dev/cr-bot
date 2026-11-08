# Attributed diagnostic screenshots

These images are unaltered JPGs from **Clash Royale**, Roboflow Universe project
[clash-royale-training/clash-royale-of3d3-qch1f](https://universe.roboflow.com/clash-royale-training/clash-royale-of3d3-qch1f),
made available by the dataset publisher under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
No endorsement is implied. The filenames are the source image IDs; their source pages are
`https://universe.roboflow.com/clash-royale-training/clash-royale-of3d3-qch1f/images/<ID>`.

The `0JP5Zs9NiQk4rwMjXHGh` image is from a different match and exercises off-deck cards.
The other images are correlated screenshots from one match. The six hand *reference templates*
in `assets/card_templates/*_nejc.png` come from a different dataset/match; these screenshots
are evaluation inputs, not templates or learned identity crops.

`manifest.json` records the original dataset's troop boxes, rounded to pixels, and manually
checked sides. The underlying labels are not independently guaranteed complete or correct, so
unmatched outputs are called **extras relative to these labels**, not proven false positives.
The repeat-still pair diagnostic at `tools/paired_arena_check.py` warms on the same still six
times then repeats a target still six times: it is **not** real-time video, cannot measure
motion or real FPS, and is not a representative accuracy test.
