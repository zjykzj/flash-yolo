# YOLO26-pose (pose estimation)

> Reproduction target: official `yolo26n-pose` weights through this repo's pipeline reach
> **0.5687** pose AP / **0.8250** AP50 on COCO-pose val2017 (2,346 images, pycocotools OKS);
> Ultralytics' default val (NMS) reaches 0.570 / 0.831 and E2E reaches 0.572 / 0.833 on the
> same data.
> 300 epochs from scratch: E2E **0.5529** / NMS **0.5541** (same protocol; not like-for-like
> with O365 + 400 COCO-pose epochs).
> Structure: `config/models/yolo26-pose.yaml`; shared detection/segmentation structure:
> [yolo26.md](yolo26.md) / [yolo26-seg.md](yolo26-seg.md).

## 1. Conventions

- The official pose checkpoint is **Objects365 detection pretraining + 400 epochs of COCO-pose
  finetuning** (official `train_args.pretrained = yolo26n-objv1-150.pt`, `epochs = 400`,
  `batch = 128`). Training from scratch with this repo's recipe works, but it is **not
  like-for-like** with the official pose numbers.
- The COCO-pose protocol evaluates **2,346 images with valid keypoints**, not the 5,000-image
  detection `val2017`; `scripts/eval.py` filters by `num_keypoints > 0` automatically.
- Metrics = pycocotools `iou_type="keypoints"` (OKS, maxDets=20); the table columns are
  AP / AP50 / AP75 / AP(M) / AP(L) / AR — no bbox-style P/R and no small bucket.
- "Official" means the pose weights/recipes/numbers published by Ultralytics; this repo only
  provides download and conversion scripts and does not redistribute them.

## 2. Structure (n)

backbone/neck are **identical line-by-line** with the detection variant (the official yaml only
replaces the last line); the head becomes `Pose26` = `Detect` + a keypoint branch:

| Piece | Content |
|---|---|
| `cv4` / `one2one_cv4` | two 3×3 convs per level; `c4 = max(ch[0]//4, K·(D+2))`, 85 for n |
| `cv4_kpts` | 1×1 per level → `nk = K·D` (COCO 17×3 = 51), interleaved x/y/v |
| `cv4_sigma` | 1×1 per level → `2K = 34`; RLE uncertainty, training-only (not used by infer/export) |
| `flow_model` (RealNVP) | 6 coupling layers; training-only for RLE, weights travel in the ckpt |
| `one2one_cv4*` | deep copies of the three branches when `end2end=True` (o2o branch) |

- n totals **3,679,464 params / 363 layers / 7.6 GFLOPs @640** (module-tree convention, equal to
  the official `sum(p.numel())`; the 3,747,554 in the official yaml comment is stale).
- `kpt_shape = (17, 3)` is a structural parameter: 17 keypoints × (x, y, visibility) for n.
  It travels in the weights metadata and drives head construction in `PtEngine`, export and
  training.
- bias_init only covers `cv2/cv3`; `cv4` has no prior (official convention).

## 3. Inference and export contracts

- **E2E (default)**: `(B, 300, 6 + nk)` = `[x1,y1,x2,y2,score,cls, kpts...]`; keypoints share
  the **same** two-stage top-k with the boxes (`e2e_topk`'s extra mechanism).
- **raw / NMS**: `(B, 4 + nc + nk, N)` with **decoded** pixel xywh + sigmoid scores + decoded
  keypoints (official pose `_inference` convention). This differs from the detection head's
  raw "ltrb+logits" contract: keypoint decoding needs anchors/strides, so it stays in-graph and
  the `--raw` output is directly comparable with the official pose output.
- Keypoint decode: `(t + anchor) * stride`, v through sigmoid; there is no v8-Pose
  `2t - 0.5` convention here.
- The exported graph keeps only the inference branch: `cv4_sigma` / `flow_model` are
  training-only and are not in ONNX/TRT.
- ONNX metadata / TRT sidecar JSON carry `arch/scale/nc/imgsz/names/kpt_shape/end2end`;
  `infer.py` / `eval.py` no longer need the original `.safetensors`.

## 4. Weight download and initialization

Official pose weights are not distributed with the Release; download and convert them:

```bash
python scripts/download_weights.py --model yolo26n-pose
python scripts/convert_weights.py --src weights/yolo26n-pose.pt \
    --dst weights/yolo26n-pose.safetensors
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose
```

- The converter writes `arch=yolo26-pose` / `scale` / `nc=1` / `names=["person"]` /
  `kpt_shape=[17,3]`; strict load matches every key (including `cv4*` and `flow_model`).
- To finetune from the official pose weights, pass `--weights` directly. To initialize from
  O365 **detection** weights, the pose head's `cv4/cv4_kpts/cv4_sigma/flow_model` are absent;
  `load_weights_finetune(..., allow_missing_head=True)` initializes them randomly and loads
  every other key bit-for-bit.

## 5. Formal metrics (COCO-pose val2017, 2,346 images)

| Training / implementation | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---|---:|---:|---:|---:|---:|---:|
| Official weights · this repo E2E | **0.5687** | 0.8250 | 0.6176 | 0.4814 | 0.6924 | 0.6350 |
| Official weights · this repo NMS | **0.5670** | 0.8280 | 0.6116 | 0.4774 | 0.6923 | 0.6303 |
| 300 epochs from scratch · this repo E2E | **0.5529** | 0.8200 | 0.5947 | 0.4686 | 0.6748 | 0.6193 |
| 300 epochs from scratch · this repo NMS | **0.5541** | 0.8249 | 0.5984 | 0.4704 | 0.6747 | 0.6166 |
| Official weights · Ultralytics E2E (same 2,346 images) | 0.572 | 0.833 | 0.618 | 0.499 | 0.689 | 0.635 |
| Official weights · Ultralytics NMS (default val) | 0.570 | 0.831 | 0.614 | 0.496 | 0.686 | 0.631 |

- The difference from Ultralytics comes from square letterbox (the official val uses auto/rect)
  and metric-implementation noise; the forward itself is bit-identical on the same input (see
  [../development/alignment.md](../development/alignment.md)).
- The official training-time validator records pose 0.5717 / AP50 0.8332, the same order.
- The 300-epoch from-scratch run (default recipe, batch 32, nbs 64, fp32, single GPU) reaches
  E2E **0.5529** / NMS **0.5541**; the official model is O365 + 400 epochs, so this is still
  not like-for-like.
- FastPoseMetrics now follows the COCO keypoint convention: `num_keypoints==0` annotations are
  ignored and OKS uses the annotation `area`. On the same checkpoint it is within ~0.02 AP of
  pycocotools and AR@20 matches; training logs remain trend/best-picking only — formal numbers
  come from `scripts/eval.py`.
- The full 400-epoch official recipe has not been reproduced here; the 300-epoch baseline and
  the 10-epoch pipeline parity check are in
  [../development/alignment.md](../development/alignment.md) §3.1 / §3.2.
- The 300-epoch from-scratch weights are local training artifacts and are not distributed with
  the Release; the Release ships download/conversion scripts for the official weights.

## 6. Inference latency (model-only, 640)

> Protocol: the inference segment of `engine.predict_timed`, median of 30 consecutive calls
> per round x 5 rounds, best-case min; engines are warmed up at init. b1 = single image,
> b8 = per-image average of batch 8 (batch-8 engines); b1 is a per-run overhead floor and does
> not extrapolate to b8. Latency is the same across different weights of one architecture
> (the 300-epoch from-scratch artifact is structurally identical to the official weights).

### Cloud server / workstation

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | — | — |
| RTX 5090 | TensorRT 11.4 · fp16 | — | — |
| Xeon 8470Q (25-core quota) | ONNX Runtime 1.30 · fp32 | 12.4 ms | — |
| Xeon 8470Q (25-core quota) | PyTorch · CPU fp32 | 47.3 ms | — |

- The 5090 TRT rows are placeholders to be filled after server measurements; the ORT / PT CPU
  rows carry over from the previous table on this page (same 8470Q machine, median-of-30
  protocol), ORT conditions as in [yolo26.md](yolo26.md) §5.

### Local laptop

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 4060 Laptop | TensorRT 11.4 · fp32 | 3.61 ms | 2.25 ms |
| RTX 4060 Laptop | TensorRT 11.4 · fp16 | 4.59 ms | 1.26 ms |
| RTX 4060 Laptop | PyTorch · CUDA fp32 | 14.5 ms | — |
| Ultra 9 185H (22 threads) | PyTorch · CPU fp32 | 55.4 ms | — |

- b1 is a multi-round best-case min (laptop GPU dynamic clocks make single shots drift);
  b8 is stable.
- fp16 is slower than fp32 at b1 (small graphs are scheduling-overhead-bound, the same
  phenomenon as in [yolov3-tiny.md](yolov3-tiny.md)); b8 reflects the compute difference.
- TensorRT engines are bound to the build machine's GPU and TRT version; rebuild with
  `scripts/export.py --trt` on another machine.

## 7. Training

- Data: `config/datasets/coco-pose.yaml` (COCO keypoints JSON; 17 points, `nc=1`);
  descriptor fields `kpt_shape` / `flip_idx` / `kpt_names` / `skeleton`, see
  [../datasets/README.md](../datasets/README.md).
- Loss: the three detection terms (box/cls/l1 + ProgLoss) reuse `ComputeLoss`, plus
  `pose` (KeypointLoss with COCO OKS sigmas), `kobj` (visibility BCE) and `rle` (RealNVP +
  RLELoss); default gains `pose_gain=24` / `kobj_gain=4` / `rle_gain=1`.
- In-training validation computes Box + Pose; best fitness = box mAP50-95 + pose mAP50-95
  (official `PoseMetrics.fitness` convention); formal numbers still come from `scripts/eval.py`.
- Official recipe: `config/recipes/yolo26-pose-coco.yaml` — 400 epochs, batch 128, nbs 64,
  lr0 0.005, lrf 0.1, cls_w 2, o2m 1.0, muon_w/sgd_w 0.5, mosaic 1.0, fliplr 0.5.
- To align item-by-item with the released ultralytics code (rather than reproducing the
  official checkpoint's `train_args`), override `--cls-w 1.0 --prog-alpha-init 0.8
  --muon-w 0.2 --sgd-w 1.0`: that version does not consume `cls_w`, and the other three are
  hardcoded. The 10-epoch result is in
  [../development/alignment.md](../development/alignment.md) §3.1.
- For from-scratch training use `--no-amp` (the official `amp=true` relies on O365 pretrained
  initialization; the detection from-scratch AMP NaN mechanism applies here too).
- `copy_paste > 0` errors out for pose (the official pose recipe uses 0 and there are no
  segmentation polygons); `fliplr` needs the descriptor's `flip_idx`, otherwise the flipped
  keypoint semantics are wrong.

```bash
# finetune the official pose weights (start with 10 epochs to validate the pipeline)
python scripts/train.py --data coco-pose --model yolo26-pose \
    --weights weights/yolo26n-pose.safetensors \
    --recipe yolo26-pose-coco --epochs 10 --batch 32 --device 0

# from scratch, 300 epochs (default recipe; not like-for-like with the official numbers)
python scripts/train.py --data coco-pose --model yolo26-pose \
    --epochs 300 --batch 32 --workers 12 --device 0 --no-amp
```

## 8. Usage

```bash
# evaluate (E2E / NMS)
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose
python scripts/eval.py --weights weights/yolo26n-pose.safetensors --data coco-pose --nms

# infer: draw the skeleton and write YOLO-pose labels
python scripts/infer.py --weights weights/yolo26n-pose.safetensors \
    --image /path/to/person.jpg --data coco-pose

# export: default E2E single output (B,300,57); --raw emits the decoded (B,56,8400) graph
python scripts/export.py --weights weights/yolo26n-pose.safetensors
python scripts/export.py --weights weights/yolo26n-pose.safetensors --raw
python scripts/export.py --weights weights/yolo26n-pose.safetensors --trt --fp16
```

Training arguments, DDP and resume: [../pipeline/train.md](../pipeline/train.md); evaluation
conventions: [../pipeline/eval.md](../pipeline/eval.md); export contracts:
[../pipeline/export.md](../pipeline/export.md).

## 9. Notes

- `kpt_shape` is structural: for a custom keypoint dataset update the descriptor/weights
  metadata, not just `nc`.
- `cv4_sigma`/`flow_model` are training-only; validation's `val_rle` is always 0 (the val
  forward does not return sigma), which does not mean RLE was not trained.
- `--single-cls` is not applicable to pose (COCO-pose is already `nc=1`); the CLI errors out.
- COCO-pose val has 2,346 images; do not mix it with the 5,000-image detection split.
- Official pose weights come from O365 detection pretraining + 400 COCO-pose epochs;
  from-scratch numbers are this repo's baseline only.
- `flash-yolo-pose` reuses the same `Pose26` head with the flash-yolo backbone/neck; see
  [flash-yolo-pose.md](flash-yolo-pose.md).
