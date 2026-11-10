# Models

This page gives the model matrix and the conventions shared by all architectures; per-architecture
details live on their own pages.

## 1. Model matrix

| Architecture | Task | Scale | Params | GFLOPs | Doc |
|---|---|---|---:|---:|---|
| YOLO26 | detection | n / s / m / l / x | 2.57M (n) | 5.48 (n) | [yolo26.md](yolo26.md) |
| YOLO26-seg | instance segmentation | n / s / m / l / x | 3.13M (n) | 9.3 (n) | [yolo26-seg.md](yolo26-seg.md) |
| YOLO26-pose | pose estimation (COCO 17 keypoints) | n / s / m / l / x | 3.68M (n) | 7.6 (n) | [yolo26-pose.md](yolo26-pose.md) |
| Flash-YOLO | detection | flash (single) | 1.97M | 3.80 | [flash-yolo.md](flash-yolo.md) |
| Flash-YOLO-seg | instance segmentation | flash (single) | 2.53M | 7.6 | [flash-yolo-seg.md](flash-yolo-seg.md) |
| YOLOv3-tiny / -SiLU | detection | tiny (single) | 8.85M | 5.56 @416 | [yolov3-tiny.md](yolov3-tiny.md) |
| YOLOE-26 | open-vocabulary segmentation (pf / text / visual prompts) | n / s / m / l / x | 7.84M (n) | 16.0 (n) | [yoloe.md](yoloe.md) |

How to pick:

```bash
python scripts/train.py --data coco --model yolo26 --scale n
python scripts/train.py --data coco --model yolo26-seg --scale n
python scripts/train.py --data coco-pose --model yolo26-pose --scale n
python scripts/train.py --data coco --model flash-yolo
python scripts/train.py --data coco --model flash-yolo-seg
python scripts/train.py --data coco --model yolov3-tiny --imgsz 416
```

- `yolo26` / `yolo26-seg` / `yolo26-pose` / `yoloe-26` have the `n/s/m/l/x` scales;
  `flash-yolo` / `flash-yolo-seg` only `flash`, `yolov3-tiny` only `tiny` — passing `--scale`
  for those warns and does nothing.
- `yolo26-pose` uses `--data coco-pose` (COCO keypoints JSON, 17 points, `nc=1`); the official
  400-epoch recipe is O365 detection pretraining + COCO-pose finetuning, so from-scratch runs
  are not like-for-like. Training/eval/export/infer are all wired; see
  [yolo26-pose.md](yolo26-pose.md).
- `yoloe-26` supports prompt-free (-pf) and text-prompt weights; `--raw` does not apply and
  `--nms` is text-mode only (o2m + NMS; pf stays E2E, see yoloe.md). Text mode additionally
  needs `mobileclip2_b.safetensors` (the converted Apple MobileCLIP2-B text encoder) next to
  the model weights.
- `yoloe-26` training covers PE linear probing (`--pe --weights <text checkpoint>`: only the
  cv3 tails train, backbone/BN/mask branch stay frozen; the class-locked output evaluates and
  exports directly) and full fine-tuning (`--full-finetune`: all weights train, reprta in-graph;
  the output stays unfused, so evaluation needs the same dataset descriptor). From-scratch
  training is not wired.
- Structures live in `config/models/<model>.yaml`; `--cfg` can point at a custom yaml (custom
  variants).
- From-scratch weight download links point to the **latest release** assets (the v1.0.0
  checkpoints); the official YOLOE weights are linked from yoloe.md (Ultralytics assets), and
  the text-encoder conversion `mobileclip2_b.safetensors` is provided through Releases (next
  to the model weights, see yoloe.md).

## 2. Shared structure conventions

### state_dict and strict load

- Keys are **attribute paths** (e.g. `model.23.one2one_cv2.0.0.conv.weight`), not class names.
- The attribute tree must stay aligned with the official layout; renaming module attributes
  breaks the zero-mapping strict load and the alignment tests catch it.
- The official YOLO26 ckpt's `ckpt["model"]` is the full module object (`ema` is None);
  `scripts/convert_weights.py` unwraps it once into plain safetensors; the runtime does not
  depend on ultralytics.

### BN and numeric conventions

- BN **eps = 0.001** (official non-default; PyTorch's default 1e-5 diverges activations from
  layer 0).
- BN momentum = 0.03 (only affects running stats, not the train forward).
- The inference/export pipeline's letterbox, BGR→RGB, /255 match the official val.

### Scale rules (YOLO26 family)

| scale | depth | width | max_channels |
|---|---|---|---|
| n | 0.50 | 0.25 | 1024 |
| s | 0.50 | 0.50 | 1024 |
| m | 0.50 | 1.00 | 512 |
| l | 1.00 | 1.00 | 512 |
| x | 1.00 | 1.50 | 512 |

- `repeats = max(round(repeats * depth), 1)`;
- `c2 = make_divisible(min(c2, max_ch) * width, 8)`;
- m/l/x force `c3k=True` on every C3k2 (yaml's False applies to n/s only).

### Weight metadata

Training saves and converters both write optional JSON into the safetensors header:

- model parameters: `arch` / `scale` / `nc` / `imgsz` / `names`, anchor models also `anchors`,
  pose also `kpt_shape`;
- provenance: `version` / `repo` / `commit` / `describe` (best-effort; a missing key is not
  written).

Read order: **CLI explicit > metadata > filename/default**. So artifacts like `best.safetensors`
whose filename cannot reveal the scale skip `--model/--scale`; custom-dataset artifacts skip
`--data` (nc/names travel with the file). Exception: the onnx input size follows the graph shape
(an explicit conflicting `--imgsz` errors out).

### E2E vs NMS

- **E2E (default)**: o2o head, no NMS, output `(B, 300, 6)`; the official 40.1 goes through this.
  Pose appends `nk` keypoint channels to the last dimension: `(B, 300, 6+nk)`.
- **NMS**: o2m head + class-aware NMS, usually ~0.9 points higher, slightly slower; enabled with
  `--nms`. Pose's raw/NMS output is the **decoded** `(B, 4+nc+nk, N)` (xywh + sigmoid +
  keypoints), unlike the detection head's raw ltrb+logits; see
  [yolo26-pose.md](yolo26-pose.md).
- YOLOv3-tiny has a single "decode + obj×cls + NMS" path; `--nms` / `--raw` error out for it.

### anchors (YOLOv3-tiny only)

- Anchor values are **yaml data** (darknet fixed pixel values), never scaled with `imgsz`; under
  multi-scale training the official weights self-compensate the input scale.
- `anchors` / `stride` are `persistent=False` buffers: not in the state_dict, moved with
  `.to(device)`.
- Re-clustering is needed only when from-scratch training changes the input size:
  `python scripts/compute_anchors.py --data <desc>`.

## 3. Official vs from scratch

- The official YOLO26 / YOLO26-seg / YOLO26-pose weights are **Objects365v1 pretraining +
  COCO(-pose) finetuning**, not from-scratch training; comparing from-scratch numbers against
  the official ones is not like-for-like. The pose COCO stage runs 400 epochs and initializes
  from O365 **detection** weights.
- Each model page gives the available final results (mostly 300-epoch from scratch; pose
  currently has official-weight reproduction only, with no full from-scratch run yet), weight
  downloads and latency; the same-budget cross-model comparison lives in the root README's
  Results.
- Formal metrics always come from `scripts/eval.py` (pycocotools); in-training FastMetrics are
  for trends and best-picking only.
