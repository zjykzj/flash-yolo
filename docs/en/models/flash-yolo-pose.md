# Flash-YOLO-pose (pose estimation)

> The `flash-yolo` lightweight backbone/neck plus a `Pose26` head; single `flash` scale,
> **3.08M params / 5.9 GFLOPs** at 640. The pose head is identical to
> [yolo26-pose.md](yolo26-pose.md) (keypoints + visibility + training-time RLE); detection and
> segmentation structures are in [flash-yolo.md](flash-yolo.md) /
> [flash-yolo-seg.md](flash-yolo-seg.md).
> Status: structure / training / evaluation / export / inference are wired; **300-epoch
> from-scratch result: E2E 0.5004 / NMS 0.5055** on COCO-pose val2017 (2,346 images).

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

## 3. Formal metrics (300 epochs from scratch)

Protocol: COCO-pose val2017, 2,346 images, pycocotools OKS, maxDets=20; `default` recipe,
batch 32, nbs 64, fp32, single GPU; best = ep300.

| Training / path | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---|---:|---:|---:|---:|---:|---:|
| 300 ep from scratch · E2E | **0.5004** | 0.7832 | 0.5285 | 0.4024 | 0.6390 | 0.5681 |
| 300 ep from scratch · NMS | **0.5055** | 0.7884 | 0.5380 | 0.4051 | 0.6450 | 0.5669 |
| Reference: yolo26n-pose 300 ep E2E | 0.5529 | 0.8200 | 0.5947 | 0.4686 | 0.6748 | 0.6193 |
| Reference: yolo26n-pose 300 ep NMS | 0.5541 | 0.8249 | 0.5984 | 0.4704 | 0.6747 | 0.6166 |

- At the same budget flash is **5.25 AP (E2E) / 4.86 AP (NMS)** below yolo26n-pose, about −9%,
  while using 16.3% fewer parameters (3.08M vs 3.68M) and 22.4% fewer GFLOPs (5.9 vs 7.6).
- The 10-epoch gap was 3.54 AP (E2E: 0.2719 vs 0.3073); at 300 epochs it widened to 5.25 AP.
  The gap is driven by architecture capacity and does not close with a longer schedule.
- FastPoseMetrics now follows the COCO keypoint convention (ignores `num_keypoints==0`, uses the
  annotation `area`); on the same best checkpoint it reads 0.4791, within ~0.02 of the formal
  0.5004, with AR@20 matching.

## 4. Training

Use the built-in `default` recipe from scratch (100 epochs by default; the formal run below uses
300) — `yolo26-pose-coco` is the official pose checkpoint's 400-epoch finetune recipe and relies
on an O365-pretrained start; do not use it directly for a from-scratch flash model:

```bash
# architecture sanity check (no data)
python model/summary.py --model flash-yolo-pose

# from scratch (COCO-pose; fp32 by default)
python scripts/train.py --data coco-pose --model flash-yolo-pose \
    --epochs 300 --batch 32 --workers 12 --no-amp \
    --name flash-pose-scratch-300ep

# optional: initialize backbone/neck from flash-yolo detection weights; pose head is random
python scripts/train.py --data coco-pose --model flash-yolo-pose \
    --weights flash-yolo-e300-640.safetensors \
    --name flash-pose-ft --batch 32 --workers 8 --amp
```

- When initializing from detection weights, `cv4/cv4_kpts/cv4_sigma/flow_model` are absent;
  `load_weights_finetune(..., allow_missing_head=True)` initializes them randomly and loads
  every other key bit-for-bit (see `tests/test_finetune_load.py`).
- `copy_paste > 0` errors out for pose; `fliplr` uses the descriptor's `flip_idx`.
- The FastPoseMetrics correction and its re-evaluation are in §3; training logs remain
  trend/best-picking only — formal numbers come from `scripts/eval.py` (same protocol as
  [yolo26-pose.md](yolo26-pose.md)).

## 5. Inference latency (model-only, 640)

> Protocol: the inference segment of `engine.predict_timed`, median of 30 consecutive calls
> per round x 5 rounds, best-case min; engines are warmed up at init. b1 = single image,
> b8 = per-image average of batch 8 (batch-8 engines); b1 is a per-run overhead floor and does
> not extrapolate to b8. Weights = the 300-epoch from-scratch `best.safetensors` (latency is the
> same across different weights of one architecture).

### Cloud server / workstation

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | — | — |
| RTX 5090 | TensorRT 11.4 · fp16 | — | — |
| Xeon 8470Q (25-core quota) | ONNX Runtime 1.30 · fp32 | — | — |

- The 5090 / 8470Q rows are placeholders to be filled after server measurements (ORT conditions
  as in [yolo26.md](yolo26.md) §5).

### Local laptop

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 4060 Laptop | TensorRT 11.4 · fp32 | 3.61 ms | 2.08 ms |
| RTX 4060 Laptop | TensorRT 11.4 · fp16 | 4.37 ms | 1.22 ms |
| RTX 4060 Laptop | PyTorch · CUDA fp32 | 13.8 ms | — |
| Ultra 9 185H (22 threads) | PyTorch · CPU fp32 | 54.9 ms | — |

- b1 is a multi-round best-case min (laptop GPU dynamic clocks make single shots drift);
  b8 is stable.
- fp16 is slower than fp32 at b1 (small graphs are scheduling-overhead-bound, the same
  phenomenon as in [yolov3-tiny.md](yolov3-tiny.md)); b8 reflects the compute difference.
- TensorRT engines are bound to the build machine's GPU and TRT version; rebuild with
  `scripts/export.py --trt` on another machine.
- Build commands: `scripts/export.py --weights <best.safetensors> --trt [--fp16]` (E2E single
  output (B,300,57)); batch-8 engines are built separately per the b8 protocol.

## 6. Evaluate / export / infer

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

## 7. Notes

- Only the `flash` scale exists; do not pass `--scale`.
- The official `yolo26n-pose` reference numbers (E2E 0.5687 / 0.8250) do **not** apply to this
  structure; this page's 300-epoch from-scratch numbers are in §3, together with the same-budget
  yolo26n-pose comparison.
- Structure: 3,081,432 params / 5.9 GFLOPs @640; the first 24 rows are shared with `flash-yolo`.
- The 300-epoch from-scratch weights are local training artifacts and are not distributed with
  the Release.
- `val_rle` is always 0 in validation logs (the val forward does not return sigma); it does not
  mean RLE was not trained.
