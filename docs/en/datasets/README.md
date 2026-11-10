# Datasets

> There is exactly one data entry point: the **dataset descriptor yaml**. train / eval / bench_io
> / infer all locate it through `--data <name|.yaml>`; the model's class count (nc) is also
> decided by it — the model yaml no longer needs editing.

Format differences: [coco.md](coco.md) and [yolo.md](yolo.md); training-side usage:
[../pipeline/train.md](../pipeline/train.md).

## 1. Descriptor schema

Template: `config/datasets/coco.yaml`.

| Key | Required | Description |
|---|---|---|
| `format` | ✔ | `coco` (json annotations) or `yolo` (`labels/*.txt`) |
| `path` | – | dataset root; relative paths are resolved against the descriptor's directory; empty = absolute paths only |
| `names` | ✔ | class name list (or a `{0: name}` mapping, equivalent); **index = class id = nc** |
| `kpt_shape` | pose ✔ | `[K, D]` (COCO `[17, 3]`); D=2 is x/y only, D=3 adds visibility |
| `flip_idx` / `kpt_names` / `skeleton` | pose optional | horizontal-flip permutation / per-class keypoint names / skeleton pairs (0-based, visualization only) |
| `train` / `val` / `test` | train, val ✔ | role keys (`test` accepted but not consumed yet) |
| ├ `images` | ✔ | coco: image directory; yolo: image directory or a `.txt` image list |
| ├ `ann` | coco ✔ | annotation json path; **forbidden for yolo** (errors out) |
| └ `labels` | yolo optional | label directory; default = the `images` segment in the `images` path replaced with `labels` |

```yaml
# COCO
format: coco
path: /path/to/coco
names: [person, bicycle, ...]
train: {images: images/train2017, ann: annotations/instances_train2017.json}
val:   {images: images/val2017,   ann: annotations/instances_val2017.json}
```

```yaml
# YOLO txt (labels auto-inferred from images when omitted)
format: yolo
path: /path/to/mydata
names: [cat, dog]
train: {images: images/train}
val:   {images: images/val}
```

```yaml
# COCO-pose (keypoints; template: config/datasets/coco-pose.yaml)
format: coco
path: /path/to/coco
names: [person]
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
train: {images: train2017, ann: annotations/person_keypoints_train2017.json}
val:   {images: val2017,   ann: annotations/person_keypoints_val2017.json}
```

## 2. Name resolution and local/ overrides

- `--data coco`: first look up `config/datasets/local/coco.yaml` (gitignored, **machine paths go
  only here**), then fall back to the packaged template `config/datasets/coco.yaml` (`path`
  intentionally left empty).
- `--data /path/to/x.yaml`: the path is used directly.
- **A bare directory errors out as a migration error** (the old interface is deprecated):
  `--data /data/coco` ✗ → write a descriptor or copy the template.

## 3. Shared conventions

- The order of `names` is the class id used in training/inference; **reordering = changing
  semantics** — old weights would be misaligned.
- The descriptor's `names` are hard cross-checked against the COCO json `categories` (the first
  mismatching index errors out).
- Training drops `iscrowd=1` GTs; evaluation keeps them (official convention) — see
  [coco.md](coco.md).
- Corrupt images are handled lazily: no per-image decode during scanning; `load_image` failures
  are counted and warned about, training continues (train returns a blank canvas + empty labels,
  val keeps the GT and honestly records a miss).
- `--limit N` samples differently per format: COCO follows json order, YOLO follows filename
  order; cross-format comparisons need the full set or aligned subsets.
- Pose currently supports **COCO keypoints JSON only** (YOLO-pose txt is not wired);
  `kpt_shape` is structural and must match the weights metadata. `require_keypoints` drops
  images without valid keypoints (COCO-pose: 56,599 train / 2,346 val).

## 4. What `--data` means in each command

| Command | Purpose |
|---|---|
| `train.py` | train/val data, nc, class names (weight metadata carries names/nc) |
| `eval.py` | evaluation data (`--split` picks the role, default val); nc builds the head |
| `bench_io.py` | sampling data for the I/O benchmark (train role only) |
| `infer.py` | only `names` + `nc` (labels for models trained on non-COCO data); no images read |
| `export.py` | no `--data`; non-COCO models pass `--nc N` for the ONNX head class count |

## 5. Custom dataset checklist

1. Directory layout: `images/` + `labels/` (YOLO) or `images/` + `annotations/*.json` (COCO);
2. Descriptor: `names` order = the class ids used in training/inference;
3. Self-check: `python scripts/eval.py --weights <same-nc weights> --data <desc> --limit 20`, or
   `python scripts/train.py --data <desc> --epochs 1 --val-epochs 0`;
4. Models with `anchors` (YOLOv3-tiny): run `python scripts/compute_anchors.py --data <desc>`
   first and read the verdict — see [../models/yolov3-tiny.md](../models/yolov3-tiny.md);
5. Segmentation: COCO polygons / RLE or YOLO-seg polygon rows; `mask_ratio` must divide `imgsz`
   (default 4; 640 → 160, the same grid as Proto26).

## 6. Small-dataset loop

```bash
# ① subset from COCO (seeded sampling + hard links; also writes coco/yolo descriptors into local/)
python scripts/make_coco_subset.py --src /path/to/coco --dst /path/to/coco-tiny \
    --n-train 2000 --n-val 1000 --seed 0 --link-images --emit-descriptors

# ② YOLO detection conversion (dataflow-cv; writes only labels/ and classes.txt, images/ linked)
dataflow-cv convert coco2yolo <json> <out-yolo>

# ②b YOLO segmentation conversion (this repo's script; polygons follow the official merge convention)
python scripts/convert_coco_seg.py --src <json> --dst <out-yolo>/labels

# ③ afterwards --data coco-tiny / coco-tiny-yolo work directly
python scripts/train.py --data coco-tiny --epochs 5 --batch 16 --workers 8
```

The only val difference between the two formats is crowd: full coco-tiny measures mAP50-95 0.4080
(COCO) vs 0.4024 (YOLO); training-side statistics are equal item by item.
