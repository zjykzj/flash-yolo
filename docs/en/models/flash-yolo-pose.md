# Flash-YOLO-pose (pose estimation)

> The `flash-yolo` lightweight backbone/neck plus a `Pose26` head; single `flash` scale,
> **3.08M params / 5.9 GFLOPs** at 640. The pose head is identical to
> [yolo26-pose.md](yolo26-pose.md) (keypoints + visibility + training-time RLE); detection and
> segmentation structures are in [flash-yolo.md](flash-yolo.md) /
> [flash-yolo-seg.md](flash-yolo-seg.md).
> Status: structure / training / evaluation / export / inference are wired; **no public
> checkpoint or formal metrics yet**.

## 1. Structure

- The backbone/neck is line-for-line identical to `flash-yolo` (the first 24 rows of the
  25-row yaml); only the final `Detect` becomes `Pose26`: `config/models/flash-yolo-pose.yaml`.
- `Pose26` = `Detect` + `cv4` (two 3×3 convs per level, `c4 = max(ch[0]//4, K·(D+2)) = 85`)
  + `cv4_kpts` (1×1 → 51, interleaved x/y/v) + `cv4_sigma` (training-time RLE, 2K=34)
  + `flow_model` (RealNVP, training-time RLE); `end2end=True` adds the `one2one_*` deep copies.
- Head input channels = `[64, 128, 256]` (P3/P4/P5); the module tree has **365 layers**, of
  which the head is 1,416,840 params.
- `cv4_sigma` / `flow_model` are training-only and are not in the ONNX/TRT inference graph.
- `kpt_shape = (17, 3)` is structural and travels in the weights metadata; custom keypoint
  datasets must update the descriptor/weights metadata, not just `nc`.

## 2. Contracts

- **E2E (default)**: `(B, 300, 6 + nk)` = `[x1,y1,x2,y2,score,cls, kpts...]`; keypoints share
  the same two-stage top-k with the boxes. For COCO 17 points `nk=51`, i.e. `(B, 300, 57)`.
- **raw / NMS**: `(B, 4 + nc + nk, N)` with **decoded** pixel xywh + sigmoid scores + decoded
  keypoints; `(B, 56, 8400)` for COCO, NMS outside. This differs from the detection head's raw
  ltrb+logits contract.
- Keypoint decode is `(t + anchor) * stride`, visibility through sigmoid; no v8-Pose
  `2t - 0.5` convention.
- Validation computes Box + Pose; best fitness = box + pose mAP50-95; formal numbers still come
  from `scripts/eval.py`.

## 3. Training

Use the built-in `default` recipe (100 epochs) from scratch — `yolo26-pose-coco` is the official
pose checkpoint's 400-epoch finetune recipe and relies on an O365-pretrained start; do not use it
directly for a from-scratch flash model:

```bash
# architecture sanity check (no data)
python model/summary.py --model flash-yolo-pose

# from scratch (COCO-pose; fp32 by default)
python scripts/train.py --data coco-pose --model flash-yolo-pose \
    --name flash-pose --batch 32 --workers 8 --no-amp

# optional: initialize backbone/neck from flash-yolo detection weights; pose head is random
python scripts/train.py --data coco-pose --model flash-yolo-pose \
    --weights flash-yolo-e300-640.safetensors \
    --name flash-pose-ft --batch 32 --workers 8 --amp
```

- When initializing from detection weights, `cv4/cv4_kpts/cv4_sigma/flow_model` are absent;
  `load_weights_finetune(..., allow_missing_head=True)` initializes them randomly and loads
  every other key bit-for-bit (see `tests/test_finetune_load.py`).
- `copy_paste > 0` errors out for pose; `fliplr` uses the descriptor's `flip_idx`.
- FastPoseMetrics during training can differ from pycocotools by ~0.1 AP; training logs are for
  trends/best-picking only — formal numbers come from `scripts/eval.py` (same protocol as
  [yolo26-pose.md](yolo26-pose.md)).

## 4. Evaluate / export / infer

```bash
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose --nms

python scripts/export.py --weights runs/train/<run>/weights/best.safetensors
python scripts/export.py --weights runs/train/<run>/weights/best.safetensors --raw
python scripts/infer.py --weights runs/train/<run>/weights/best.safetensors \
    --image assets/traffic.jpg --data coco-pose
```

- COCO-pose val evaluates **2,346 images with valid keypoints** (not the 5,000-image detection
  val); pycocotools OKS, maxDets=20.
- Export metadata carries `arch=flash-yolo-pose / scale / nc / imgsz / names / kpt_shape /
  end2end`; infer/eval no longer need the original `.safetensors`.
- `--nms` is supported (pose o2m branch); `--single-cls` is not applicable to pose.

## 5. Notes

- Only the `flash` scale exists; do not pass `--scale`.
- The official `yolo26n-pose` reference numbers (E2E 0.5687 / 0.8250) do **not** apply to this
  structure; this page does not publish unfinished flash-yolo-pose training numbers. Add formal
  metrics after a complete run.
- Structure: 3,081,432 params / 5.9 GFLOPs @640; the first 24 rows are shared with `flash-yolo`.
- `val_rle` is always 0 in validation logs (the val forward does not return sigma); it does not
  mean RLE was not trained.
