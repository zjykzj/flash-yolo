# LVIS Dataset (v1)

> LVIS is a **federated annotation** protocol on top of the COCO format: each image annotates
> only a subset of classes, and evaluation uses `faster_coco_eval`'s `lvis_style`. This project
> uses it for YOLOE open-vocabulary evaluation and fine-tuning; descriptor template:
> `config/datasets/lvis.yaml`.

## 1. Differences from COCO

| Item | COCO | LVIS |
|---|---|---|
| Classes | 80 | 1,203 (r 337 / c 461 / f 405) |
| Per-image labels | exhaustive | partial |
| Image-level fields | — | `neg_category_ids` / `not_exhaustive_category_ids` |
| Evaluation | pycocotools | faster_coco_eval `lvis_style` |
| Metrics | AP / AR | AP / APr / APc / APf |
| maxDets | 100 | 300 (official) |
| Raw JSON | has `file_name` / `iscrowd` | **missing**, needs a compat conversion |

- `neg_category_ids`: classes **verified absent** from this image; a prediction is a FP.
- `not_exhaustive_category_ids`: this image's annotations for these classes are **incomplete**;
  predictions for them are ignored.
- A `(image, class)` pair that is neither in GT nor in neg / not_exhaustive is dropped.
- **Do not evaluate LVIS with pycocotools**: it does not understand these two image-level
  fields, and the same predictions can differ by several times.

## 2. Layout and annotations

```text
<root>/
├── images/train2017/          # LVIS train images (from COCO train2017)
├── images/val2017/            # LVIS val / minival images (see below)
└── annotations/
    ├── lvis_v1_train.json     # 100,170 images / 1,270,141 annotations
    ├── lvis_v1_val.json       # 19,809 images / 244,707 annotations
    ├── lvis_v1_minival.json   # optional: 5,000 COCO val2017 images / 50,672 annotations
    └── lvis_v1_*_compat.json  # conversion outputs (what this project reads)
```

- All 100,170 LVIS train images live in COCO train2017; hard links avoid a second copy.
- LVIS val's 19,809 images come from 15,000 COCO train2017 + 4,809 COCO val2017 images;
  this repo's example places them all in `images/val2017` (20,000 images, including the 191
  minival background images).
- minival is 5,000 COCO val2017 images, used to match official YOLOE numbers; it is not a
  different annotation format.
- minival contains 191 **background images** (no annotations and empty `neg_category_ids`);
  filtering must not drop them, otherwise the evaluation set shrinks from 5,000 to 4,809.

## 3. Compat conversion

Raw LVIS lacks `file_name` (only `coco_url`) and `iscrowd`, and its class names use
underscores. This project reads compat JSON:

```bash
python scripts/convert_lvis.py \
    --src annotations/lvis_v1_train.json \
    --dst annotations/lvis_v1_train_compat.json
python scripts/convert_lvis.py \
    --src annotations/lvis_v1_val.json \
    --dst annotations/lvis_v1_val_compat.json
# minival: convert the official minival annotation directly
python scripts/convert_lvis.py \
    --src annotations/lvis_v1_minival.json \
    --dst annotations/lvis_v1_minival_compat.json
# optional: derive minival from val (ref: LVIS/COCO json, or an image-id / file-name list)
python scripts/convert_lvis.py \
    --src annotations/lvis_v1_val.json \
    --dst annotations/lvis_v1_minival_compat.json \
    --minival-ref annotations/lvis_v1_minival.json
```

The conversion does four things:

1. Add `file_name` to images: `coco_url` basename first, else `%012d.jpg`.
2. Add `iscrowd: 0` to annotations (LVIS has no crowd).
3. Renumber categories to `0..1202` in sorted-id order and rename them to `--names-from`
   (default `config/datasets/lvis.yaml`).
4. Renumber `neg_category_ids` / `not_exhaustive_category_ids` with the same map.

- With a custom `--names-from`, its class count must match LVIS's 1,203.
- With a JSON `--minival-ref`, images missing from the source are materialized from the ref
  (including background images); a text list only filters images already present in the source.
- compat is a derived file: do not hand-edit it, rerun the script.

## 4. Descriptor

```bash
cp config/datasets/lvis.yaml config/datasets/local/lvis.yaml   # then edit `path:`
```

- `format: coco`, `eval_style: lvis`, `names` with 1,203 classes.
- `train` / `val` point at `lvis_v1_{train,val}_compat.json`.
- minival: copy it to `config/datasets/local/lvis-minival.yaml`, change `val.ann` to
  `lvis_v1_minival_compat.json`, keep `images/val2017`.
- Then `--data lvis` / `--data lvis-minival`.

## 5. Training (YOLOE)

LVIS training is only used for YOLOE text-mode PE / full fine-tuning; there is no from-scratch path.

```bash
# PE linear probing (10 epochs, lr0 0.01)
python scripts/train.py --data lvis --model yoloe-26 --scale n --pe \
    --weights weights/yoloe-26n-seg.safetensors --epochs 10 --batch 16 --lr0 0.01 \
    --val-limit 5000 --name yoloe-pe-lvis

# Full fine-tuning (10 epochs, lr0 0.001)
python scripts/train.py --data lvis --model yoloe-26 --scale n --full-finetune \
    --weights weights/yoloe-26n-seg.safetensors --epochs 10 --batch 16 --lr0 0.001 \
    --warmup-epochs 3 --val-limit 5000 --name yoloe-ft-lvis
```

- `--val-limit 5000` only limits the training-time validation to the first 5,000 val images
  (to control cost); it is not minival. minival is for the final evaluation only.
- Federated annotations build a per-image valid-class mask: positives (including crowd) ∪
  `neg_category_ids` participate in BCE; `not_exhaustive_category_ids` and unlabeled classes
  are ignored.
- Federated data automatically disables mosaic / mixup / cutmix / copy_paste (the mask is
  built from the central image).
- `--no-federated-mask` is only for A/B comparisons or reproducing old behavior; do not use it
  for a real run.
- The recommended budget is 10 epochs; extend only if validation has not plateaued.

## 6. Evaluation

```bash
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival \
    --eval-max-dets 100      # match ultralytics' current val convention
```

- Automatically uses `faster_coco_eval`'s `lvis_style`; segmentation weights produce both
  Box and Mask tables.
- Output adds a `LVIS freq APr ... · APc ... · APf ...` row.
- Official LVIS maxDets is 300; `--eval-max-dets` is LVIS-only.
- Before a full run, do a `--limit 20` pass to confirm the class names match the descriptor.

## 7. Gotchas

- **`frequency` is required**: `lvis_style=True` needs `r/c/f` on every category; missing it
  errors out in the evaluator constructor.
- **`eval_style: lvis` requires `format: coco`**; YOLO txt does not support the LVIS protocol.
- **Do not use pycocotools for LVIS**: it does not understand neg / not_exhaustive, so the
  result is not a LVIS metric.
- **Zero-detection images must count**: the evaluator sets `params.imgIds` to every traversed
  image_id; a `--limit` subset must keep the same GT set.
- **Class names use the first synonym**: `config/datasets/lvis.yaml` uses the official first
  synonym (space-separated); a custom descriptor must match `--names-from`.
- **minival is not another annotation format**: it is only a 5,000-image subset of val2017;
  images still live in `images/val2017`.

Related pages: [models/yoloe.md](../models/yoloe.md) ·
[pipeline/eval.md](../pipeline/eval.md) · [pipeline/train.md](../pipeline/train.md).
