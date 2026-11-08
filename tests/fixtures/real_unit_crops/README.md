# Labelled real-gameplay crops

These are 55 bounding-box crops from 15 Clash Royale screenshots in Nejc Zavodnik's
[Clash Royale troop detection dataset](https://universe.roboflow.com/nejc-zavodnik/clash-royale-troop-detection),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: Nejc Zavodnik,
“Clash Royale troop detection”, Roboflow Universe. The images were **cropped and converted to PNG**;
some source labels were normalized to this project's card names. This attribution does not imply
endorsement by the creator. Individual source images, annotations, and creator notes are at:

```
https://universe.roboflow.com/nejc-zavodnik/clash-royale-troop-detection/images/<image_id>
```

`manifest.json` records each image ID, original source label, normalized label, original pixel
box, original dataset split, and this project's evaluation group. The source uses centre/width/
height; boxes were rounded to pixel `[x1,y1,x2,y2]` before cropping.

* `development`: 26 crops from five screenshots, examined when setting the conservative gates.
* `additional`: 29 crops from ten further screenshots, collected **after** those gates were set.

These groups may contain related matches or frames, and are not a representative, independent
benchmark. The creator warns the annotations are not very accurate. Some boxes contain adjacent
units, occlusions, effects, buildings or classes missing from the sprite library. The source's
`train`, `validation` and `test` designations are preserved, but they are **not** a held-out split
for this project: all of the first 26 crops, including a source-validation image, informed tuning.
The additional screenshots were not used as templates or training data. `baseline_report.json`
records the original recognizer's per-crop predictions and source commit.

```bash
python tools/recognition_benchmark.py --group development
python tools/recognition_benchmark.py --group additional --verbose
python tools/recognition_benchmark.py --fail-on-errors --min-correct 1
```

The last command fails if there are wrong predictions **or** no correct predictions. An all-abstain
classifier can pass `--fail-on-errors` alone; it has no measurable naming precision or useful recall.
This is an open-set check on *supplied boxes*, including unsupported classes; it does not measure
full-frame detection, side accuracy, tracking, or real-stream recall.
