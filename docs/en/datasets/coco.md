# COCO format

Shared descriptor fields: [README.md](README.md). `--data coco` resolves to
`config/datasets/local/coco.yaml` (machine paths) or the packaged template
`config/datasets/coco.yaml`.

## 1. Layout and descriptor

```text
<root>/
├── images/train2017/          # images
├── images/val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

```yaml
format: coco
path: /path/to/coco
names: [person, bicycle, ...]   # 80 classes, index = class id
train: {images: images/train2017, ann: annotations/instances_train2017.json}
val:   {images: images/val2017,   ann: annotations/instances_val2017.json}
```

- `names` are **hard cross-checked** against the json `categories`: the first mismatching index
  errors out.
- Scan statistics (printed by train/eval): `n_backgrounds` (images with no boxes after crowd
  removal), `n_missing` (absent on disk, filtered out with a warning), `n_crowd_excluded`
  (nonzero on the training side only).

## 2. Detection annotations

- `iscrowd=1`: **dropped on the training side, kept on the val side**. The COCO evaluator
  handles crowd per the official convention.
- Official train2017 is missing 1 image (`000000391895.jpg`): the loader filters by file
  existence at init and warns — expected behavior.
- **Corrupt images are lazy**: no per-image decode during scanning; `load_image` failures are
  counted and warned about, training continues (train returns a blank canvas + empty labels; val
  returns a blank image + kept GT, honestly recorded as a miss).
- Background images (no GT at all): counted in `n_backgrounds` on the training side, no error.

## 3. Segmentation annotations (`--model yolo26-seg` / `flash-yolo-seg`)

Each instance additionally reads `segmentation`:

- **Polygon list**: multiple segments are merged with the official two-pass
  `merge_multi_segment` convention, transformed point-by-point through augmentation, filled with
  `cv2.fillPoly` at letterbox resolution (int32 truncation, official convention) and downsampled
  to `imgsz / mask_ratio` (default 4, i.e. Proto26's 160×160 grid).
- **RLE**: compressed strings or counts lists are decoded into contour polygons; the scan line
  counts `seg RLE N decoded`.
- **Fallback**: corrupt / missing / invalid segmentation falls back to the bbox rectangle,
  counted in `n_seg_fallback` with a warning; **a full fallback errors out** (usually a wrong
  descriptor or annotation choice).
- Boxes are derived from the polygon bounding box (official convention).
- `overlap_mask=True` (default) produces an area-descending instance index map; labels are
  reordered accordingly — **the index-map IDs must match the targets row order**.

## 4. Small datasets

```bash
python scripts/make_coco_subset.py --src /path/to/coco --dst /path/to/coco-tiny \
    --n-train 2000 --n-val 1000 --seed 0 --link-images --emit-descriptors
```

The script samples by seed, hard-links images, generates coco/yolo descriptors and copies them
into `config/datasets/local/`. `--data coco-tiny` works directly afterwards.

## 5. Gotchas

- `--limit N` takes the first N in json order — not the same image set as YOLO's filename order;
  cross-format comparisons need the full set.
- The only val difference between COCO and YOLO comes from crowd (full coco-tiny mAP50-95 0.4080
  vs 0.4024); the training side drops crowd in both formats, statistics equal item by item.
- A full segmentation fallback errors out — it will not silently degrade into box training.
