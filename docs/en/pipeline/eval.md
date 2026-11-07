# Eval

> Entry point: `python scripts/eval.py --weights <w> --data <name|.yaml>`.
> **This script is the only source of external numbers**: COCO protocol runs on pycocotools,
> LVIS protocol on faster_coco_eval (`lvis_style`); the FastMetrics printed during training are
> for trends and best-picking only.

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
python scripts/eval.py --weights weights/yoloe-26n-seg.safetensors --data lvis-minival  # LVIS minival (5k)
python scripts/eval.py --weights w.safetensors --data lvis-minival --eval-max-dets 100  # ultralytics-matching protocol
```

| Argument | Default | Description |
|---|---|---|
| `--weights` | — | `.safetensors` / `.onnx` / `.engine` |
| `--data` | — | descriptor (see [../datasets/README.md](../datasets/README.md)); yoloe text mode uses its names as the prompts |
| `--split` | `val` | role key (train/val/test) |
| `--engine` | `pt` | `pt` / `onnx` / `trt` |
| `--model` / `--scale` | inferred from weights | pass explicitly when nothing can be inferred |
| `--nms` | off | o2m + NMS (yolo26 family + yoloe-26 text; errors for v3, prompt-free is E2E only) |
| `--conf` / `--iou` / `--max-det` | 0.001 / 0.7 / 300 | matches the official val convention |
| `--imgsz` | metadata > 640 | onnx follows the graph shape (conflicts error out) |
| `--limit` | 0 (full) | evaluate the first N images only (GT restricted to the same subset) |
| `--eval-style` | `auto` | `auto` (descriptor `eval_style`) / `coco` / `lvis` |
| `--eval-max-dets` | LVIS 300 | evaluator maxDets; `100` mirrors ultralytics' current val behavior |
| `--summary-only` | off | print only the all row |
| `--device` | auto | pt engine device |

## 2. Metric conventions and evaluation protocols

| Convention | Implementation | Use | Approximations |
|---|---|---|---|
| **Formal · COCO protocol** (default) | pycocotools | COCO external numbers, acceptance | — |
| **Formal · LVIS protocol** | faster_coco_eval (`lvis_style=True`) | LVIS / YOLOE open-vocabulary comparisons | — |
| In-training | pure-numpy FastMetrics | per-epoch trends + picking `best.safetensors` | ignores crowd, maxDets=100, no area grouping; greedy matching + 101-point interpolation |

FastMetrics tracks the formal convention well, but **cross-run comparisons and external
references always use the formal convention**.

### 2.1 LVIS protocol (`eval_style: lvis`)

LVIS annotations are not exhaustive, so evaluation must honor two image-level fields that
pycocotools does not understand:

- `neg_category_ids`: categories **verified absent** in that image — detections count as FP;
- `not_exhaustive_category_ids`: annotations for that category are **incomplete** — its
  detections are ignored;
- detections on `(image, category)` pairs that are neither GT nor in either list are dropped.

LVIS dataset descriptors therefore set `eval_style: lvis`; the evaluator switches to
faster_coco_eval with `lvis_style=True`, the official maxDets=300, and additionally reports
`APr / APc / APf`. **Running pycocotools on LVIS does not produce the LVIS metric** (the same
predictions can differ by several times) and must not be compared with published numbers.

- `--eval-max-dets 100`: ultralytics' current val sets only `lvis_style` and leaves
  faster_coco_eval's default `maxDets=[1,10,100]`; use this for a literal code-level match.
  The product default remains the official 300.
- LVIS minival is the 5,000 COCO val2017 images; the `lvis-minival` descriptor is ready.
- `lvis_style` requires a `frequency` (r/c/f) on every category; the evaluator errors out
  clearly when it is missing.

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

- The per-class table lists only classes with GT inside the evaluation scope; the evaluator's
  `-1` invalid combinations are dropped during aggregation.
- Under the LVIS protocol the last column is `AR@300` (or the `--eval-max-dets` value), plus an
  `LVIS freq  APr ... · APc ... · APf ...` line.
- `Speed` is the per-stage cost of the evaluation pipeline (including loading and
  postprocessing); `model-only` is pure inference.
- Results land in `runs/val/valN/` (with `run.log`); `results.json` is a COCO result list that
  can be rescored directly with faster_coco_eval.

## 4. E2E vs NMS

- **E2E (default)**: o2o head, no NMS, output `(B,300,6)`; the official 40.1 goes through this.
- **NMS**: o2m head + class-aware NMS, usually ~0.9 points higher, slightly slower;
  `--agnostic-nms` switches to cross-class suppression. Supported by the yolo26 family and
  **yoloe-26 text mode**; prompt-free (lrpc) remains E2E only.
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
  Compare with the official LVIS minival numbers via `--data lvis-minival` (faster_coco_eval
  `lvis_style`, see §2.1). Requires `--engine pt` and `mobileclip2_b.safetensors` next to the
  model weights.
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

- **`--limit` must restrict GT too**: the evaluator sets `params.imgIds` to every image traversed
  (including zero-detection images) — do not regress to "only images with predictions", which
  would exclude misses.
- **Empty results**: `loadRes([])` raises IndexError; the evaluator short-circuits to all zeros.
- **LVIS `frequency`**: `lvis_style=True` requires `frequency` (r/c/f) on every category; a plain
  COCO json with `eval_style: lvis` errors out in the evaluator constructor.
- **LVIS maxDets default**: faster_coco_eval's `Params.maxDets` defaults to `[1,10,100]`; the
  official LVIS value is 300 and this project sets it explicitly. Use `--eval-max-dets 100` for
  an ultralytics-matching comparison.
- **`--eval-max-dets` is LVIS-only**: COCO protocol keeps maxDets=100 (pycocotools convention);
  passing the flag otherwise errors out.
- **crowd**: COCO keeps it (official convention); YOLO has no crowd — the val difference comes
  only from this.
- **segm without masks errors out**: `iou_type="segm"` receiving Detections without `masks`
  raises ValueError — no silent degradation to box evaluation.
- pt's `--imgsz` overriding metadata re-letterboxes; onnx/trt input sizes are decided by the
  exported graph.
- `eval` defaults to conf 0.001, `infer` to 0.25 — don't mix the two.
