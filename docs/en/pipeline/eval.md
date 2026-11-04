# Eval

> Entry point: `python scripts/eval.py --weights <w> --data <name|.yaml>`.
> **This script is the only source of external numbers** (the pycocotools pipeline); the
> FastMetrics printed during training are for trends and best-picking only.

## 1. Usage

```bash
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco                # E2E
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --nms          # o2m + NMS
python scripts/eval.py --weights weights/yolo26n.onnx --engine onnx --data coco         # switch runtime
python scripts/eval.py --weights w.safetensors --data coco --imgsz 416                  # input size
python scripts/eval.py --weights w.safetensors --data coco --split val --limit 100      # subset
python scripts/eval.py --weights weights/yolo26n-seg.safetensors --data coco            # Box + Mask
python scripts/eval.py --weights weights/yoloe-26n-seg-pf.safetensors --data coco       # pf vocab mapping
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis          # text prompts (1203 classes)
```

| Argument | Default | Description |
|---|---|---|
| `--weights` | — | `.safetensors` / `.onnx` / `.engine` |
| `--data` | — | descriptor (see [../datasets/README.md](../datasets/README.md)); yoloe text mode uses its names as the prompts |
| `--split` | `val` | role key (train/val/test) |
| `--engine` | `pt` | `pt` / `onnx` / `trt` |
| `--model` / `--scale` | inferred from weights | pass explicitly when nothing can be inferred |
| `--nms` | off | o2m + NMS (yolo26 only; errors for v3) |
| `--conf` / `--iou` / `--max-det` | 0.001 / 0.7 / 300 | matches the official val convention |
| `--imgsz` | metadata > 640 | onnx follows the graph shape (conflicts error out) |
| `--limit` | 0 (full) | evaluate the first N images only (GT restricted to the same subset) |
| `--summary-only` | off | print only the all row |
| `--device` | auto | pt engine device |

## 2. The two metric conventions

| Convention | Implementation | Use | Approximations |
|---|---|---|---|
| **Formal** (this script) | pycocotools | external numbers, acceptance | — |
| In-training | pure-numpy FastMetrics | per-epoch trends + picking `best.safetensors` | ignores crowd, maxDets=100, no area grouping; greedy matching + 101-point interpolation |

FastMetrics tracks the formal convention well, but **cross-run comparisons and external
references always use the formal convention**.

## 3. Reading the output

```text
      Class           Images Instances         P         R     mAP50  mAP50-95    AR@100
      all               5000     36781    0.6220    0.4594    0.4863    0.3377    0.5561
      person            2693     11004    0.7892    0.6300    0.7221    0.4776    0.6174
      Size buckets   mAP_small ... · mAP_medium ... · mAP_large ...
      Thresholds     mAP50 ... · mAP75 ... · mAP50-95 ...
Speed: 27.0 ms/image = load 2.5 + preprocess 1.8 + inference 7.5 + postprocess 14.5 + other 0.7
       model-only inference 7.5 ms/image (133 img/s) at shape (1, 3, 640, 640)
```

- The per-class table lists only classes with GT inside the evaluation scope; pycocotools' `-1`
  invalid combinations are dropped during aggregation.
- `Speed` is the per-stage cost of the evaluation pipeline (including loading and
  postprocessing); `model-only` is pure inference.
- Results land in `runs/val/valN/` (with `run.log`).

## 4. E2E vs NMS

- **E2E (default)**: o2o head, no NMS, output `(B,300,6)`; the official 40.1 goes through this.
- **NMS**: o2m head + class-aware NMS, usually ~0.9 points higher, slightly slower;
  `--agnostic-nms` switches to cross-class suppression.
- **Single-class models**: `--single-cls` maps GT classes to 0 (model nc must be 1).
- YOLOv3-tiny has a single "decode + obj×cls + NMS" path; `--nms` / `--raw` error out.

## 5. Segmentation (yolo26-seg / flash-yolo-seg)

Segmentation weights automatically produce two tables: the same results evaluated by **box IoU
(Box)** and by **mask IoU (Mask)**.

- Masks are decoded to the original resolution by `utils/postprocess.process_mask`; the
  evaluator encodes RLE on the fly for pycocotools (`iou_type="segm"`).
- `runs/val/valN/results.json` writes the Mask group (with RLE) — submittable as COCO results.
- Mask decoding is pixel-identical with the official val pipeline (cross-checked by
  `tests/test_seg_inference.py`).
- Datasets accept COCO json and YOLO-seg txt; YOLO has no crowd, so its GT lacks crowd instances.
- In-training validation computes Box + Mask simultaneously (low-resolution approximation), best
  fitness = box + mask mAP50-95; formal numbers still come from this script.

**Official yolo26n-seg @640 acceptance**: Box 0.3967 / 0.5548, Mask 0.3367 / 0.5276 (mAP50-95 /
mAP50). The official training-time validator records box 0.3975 / mask 0.3386 — the ~0.002 gap
is metric-implementation noise.

## 6. YOLOE (open-vocabulary)

- **prompt-free (`-pf` weights)**: the head's baked 4,585-name vocabulary is mapped onto the
  dataset classes — exact name match first, `vocab_synonyms` fallback (11 COCO entries built
  into the descriptor), missing classes error out; predictions outside the dataset classes are
  dropped (the `CocoEvaluator` contract is unchanged).
- **text prompts (`-seg` weights)**: the dataset's own names become the prompts
  (`set_classes`); `--data lvis` evaluates against the 1,203 LVIS categories
  (`config/datasets/lvis.yaml`, first-synonym names — the official LVIS prompt convention).
  Requires `--engine pt` and `mobileclip2_b.safetensors` next to the model weights.
- Both modes report Box + Mask tables like the other segmentation architectures.
- Model details: [../models/yoloe.md](../models/yoloe.md).

## 7. Reference baselines (COCO val2017)

| Weights | Input | E2E mAP50-95 / mAP50 | NMS mAP50-95 / mAP50 |
|---|---|---|---|
| YOLO26n official | 640 | 40.27 / 55.80 | 40.89 / 56.87 |
| YOLO26n from-scratch 300 ep | 640 | 37.82 / 53.24 | 38.65 / 54.46 |
| Flash-YOLO from-scratch 300 ep | 640 | 35.14 / 50.15 | 36.09 / 51.56 |
| YOLOv3-tiny official | 640 | — (single path) | 17.16 / 35.90 |

The full weight list, download links and latency live on the model pages
([../models/README.md](../models/README.md)).

## 8. Gotchas

- **`--limit` must restrict GT too**: the evaluator aligns the subset via `params.imgIds` — keep
  this in mind when editing the metrics code yourself.
- **Empty results**: `loadRes([])` raises IndexError; the evaluator short-circuits to all zeros.
- **crowd**: COCO keeps it (official convention); YOLO has no crowd — the val difference comes
  only from this.
- **segm without masks errors out**: `iou_type="segm"` receiving Detections without `masks`
  raises ValueError — no silent degradation to box evaluation.
- pt's `--imgsz` overriding metadata re-letterboxes; onnx/trt input sizes are decided by the
  exported graph.
- `eval` defaults to conf 0.001, `infer` to 0.25 — don't mix the two.
