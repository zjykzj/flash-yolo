# YOLOv3-tiny / YOLOv3-tiny-SiLU (detection)

> The repo's second architecture, and the source of the "cheap spatial downsampling" idea. The
> structural reference is the **official darknet cfg** (`cfg/yolov3-tiny.cfg`, the arXiv:1804.02767
> lineage, native 416 input).
> Structure: `config/models/yolov3-tiny.yaml` / `config/models/yolov3-tiny-silu.yaml`.

## 1. Conventions

- **The structural reference is darknet, not ultralytics**. Ultralytics 8.4.x's "v3" (asset
  `yolov3-tinyu.pt`) is an anchor-free DFL rewrite (SiLU + reg_max=16, `Detect` has no `anchors`
  parameter) — not a valid structure source or numeric alignment target.
- **SiLU variant**: `yolov3-tiny-silu` is **line-by-line isomorphic** with the original; the only
  difference = LeakyReLU(0.1) → SiLU (the activation has no parameters: state_dict key sets,
  param count 8,852,366 and @416 GFLOPs are all identical).
  The official `.weights` only match the original variant; the two are distinguished by the
  weights metadata `arch` field and the filename.
- Shared with YOLO26: only the training pipeline (trainer/EMA/optimizer/augmentation/data
  loading/engines); the model, head, decode, loss and assigner are independent implementations.

## 2. Structure (416 input, 21 modules)

| Part | idx | Content |
|---|---|---|
| Backbone | 0–13 | LeakyConv k3 s1 + MaxPool2d downsampling; 13 layers, 1024 channels |
| P5 branch | 14–15 | 1×1 → 3×3, 512 channels (P5/32 head input) |
| Neck | 16–19 | route(-4) → 1×1 → Upsample ×2 → Concat(P4) → 3×3 |
| Head | 20 | V3Detect: per-level 1×1 → 3·(5+nc), two scales P4/16 + P5/32 |

- Size: **21 layers / 8,852,366 params (8.85M) / 5.56 GFLOPs @416**.
- **Size-preserving pooling**: darknet's `[maxpool] size=2 stride=1` implies bottom-right
  padding, written in PyTorch as `ZeroPad2d([0,1,0,1]) + MaxPool2d(2,1)` (idx 11+12). Don't use
  centered padding.
- **The two head convs fold into V3Detect** (the standalone conv layers at the end of the
  darknet cfg): no cv2/cv3 attributes, no `end2end` switch, no bias_init.
- **route relative indices normalize at parse time**: layer 16 writes `-2` in yaml (darknet
  `route -4`); the assembler folds it to absolute index 14.

## 3. anchors (the biggest trap)

3 slots × [w, h] per level, level order = V3Detect's `from` order ([P4/16, P5/32]):

| Level | stride | Slots (pixels) |
|---|---|---|
| P4/16 | 16 | [23,27] · [37,58] · [81,82] |
| P5/32 | 32 | [81,82] · [135,169] · [344,319] |

- The official mask is kept as-is (P5: 3,4,5 / P4: 1,2,3; 81,82 shared across both scales, 10,14
  unused).
- **Fixed pixel units, never scaled with imgsz**. Official weights self-compensate the input
  scale under multi-scale training. Measured (500 images, FastMetrics, official weights): 416
  native mAP50 0.384 / 640 original values 0.412 / 640 linear scaling **0.154**.
- `anchors` / `stride` are `persistent=False` buffers: not in the state_dict, moved with
  `.to(device)`.
- Re-clustering is needed only when from-scratch training changes the input size:
  `python scripts/compute_anchors.py --data <desc>`. Re-clustered anchors are written into the
  weight metadata; `PtEngine` applies them via `apply_meta` after construction.

## 4. Decode contract

Single implementation: `model/head/v3_detect.py::decode_level` (shared by head and loss, the
darknet `get_yolo_box`):

```text
raw (B, 3·(5+nc), H, W)  channel order = per anchor [x, y, w, h, obj, cls...]
xy = (σ(t) + grid) · stride        # grid offset 0 (darknet convention)
wh = anchor · exp(t)               # tw/th use the raw logits, clamp(±8) against exp overflow
```

- Eval forward output: `(B, NA, 5+nc) = [x1,y1,x2,y2, σ(obj), σ(cls)...]`,
  NA = 3·ΣH·W (416 → 2535; 640 → 6000).
- **Never switch to the v5-style `(2σ−0.5)/(2σ)²`**: it breaks alignment with the official
  weights, cv2.dnn output and the 33.1 AP acceptance.
- Final confidence = **obj × cls**, computed in `utils/postprocess.v3_detections`; class-aware
  NMS (default conf 0.001 / iou 0.7 / max_det 300).
- No E2E path: `--nms` / `--raw` are YOLO26-only switches and error out for v3.

## 5. Training (loss and matching)

- **Matching (darknet semantics)**: per GT, per level, the slot with the max shape IoU among the
  3 slots is the only positive; the responsible cell = the cell containing the GT center. At most
  one positive per GT per level — no TAL/STAL, no o2m/o2o dual heads, no ProgLoss.
- **Ignore band**: non-positive slots whose predicted decoded box has IoU ≥ 0.7
  (`ignore_thresh`) with any GT are not penalized on obj.
- **Three loss terms**: `BCE(obj)` (per-image normalized over non-ignored slots) + one-hot
  `BCE(cls)` (positives only) + `CIoU` (positives only; box_gain 7.5 / cls_gain 0.5 / obj_gain
  1.0).
- Recommended training config: `--model yolov3-tiny-silu --obj-iou-target` (+ the default
  recipe); the official `.weights` only match the original variant.
- `--obj-iou-target`: positive obj target = IoU(predicted box, responsible GT), weight ×5;
  default is the darknet hard 0/1 target.

## 6. Weight download and conversion

Download links point to the **v1.0.0** release assets.

| Weights | Training | Input | Download |
|---|---|---|---|
| `yolov3-tiny-e300-416` | 300 ep from scratch · original + IoU-aware objectness | 416 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/yolov3-tiny-e300-416.safetensors) |
| `yolov3-tiny-silu-e300-416` | 300 ep from scratch · SiLU + IoU-aware objectness | 416 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/yolov3-tiny-silu-e300-416.safetensors) |

The official darknet `.weights` are not distributed with the Release; download and convert them
yourself:

```bash
python scripts/download_weights.py --model yolov3-tiny
python scripts/convert_weights.py --src weights/yolov3-tiny.weights --dst weights/yolov3-tiny.safetensors
```

### darknet `.weights` conversion details

- File layout: 4- or 5-int32 header + a float32 little-endian tensor stream; the header width is
  auto-detected from the file length, and the file must be consumed exactly.
- Tensor order = darknet cfg layer order: **the P5 head conv (512→255) comes before the P4
  branch convs**.
- In-block order: BN = `[bias(β), scale(γ), mean, var, conv weights]`; output layer =
  `[bias, weights]`; conv weight layout `(out, in, kh, kw)`, no transpose needed.
- Mapping is keyed by input channel count: 512ch → P5 head / 256ch → P4 head (`_darknet_specs`;
  keep in sync when the yaml changes).
- Measured: the official file is 35,434,956 B = 8,858,734 floats + 20 B header; strict load
  passes.

## 7. Formal metrics (COCO val2017)

| Weights | Input | mAP50-95 | mAP50 |
|---|---|---:|---:|
| Official darknet weights | 640 | **17.16** | **35.90** |
| Official darknet weights | 416 | 16.94 | 34.37 |
| From scratch 300 ep (SiLU + obj-iou) | 416 | 16.15 | 29.93 |
| From scratch 300 ep (original + obj-iou) | 416 | 16.03 | 29.59 |

- The officially published 33.1 AP50 is the **COCO test-dev @416** convention; the same weights
  score higher at 640 than at 416 in this pipeline — the two are not directly comparable.
- Official weights score higher at 640 because anchors don't scale + higher effective resolution.

## 8. Inference latency

> Protocol: **model-only** (no pre/postprocessing), **416** input; b1 = single image, b8 =
> per-image average of batch 8. Engines warm up at init; numbers are steady-state repeats.
> Latency is the same across different weights of one architecture.

### Cloud server / workstation

| Weights | Platform / runtime | Precision | b1 | b8 (per image) |
|---|---|---|---:|---:|
| v3-SiLU + objectness | RTX 5090 · TensorRT 11.4 | fp32 | 0.68 ms | 0.133 ms |
| v3-SiLU + objectness | RTX 5090 · TensorRT 11.4 | fp16 | 1.19 ms | 0.048 ms |
| v3 original + objectness | RTX 5090 · TensorRT 11.4 | fp32 | 0.68 ms | 0.133 ms |
| v3 original + objectness | RTX 5090 · TensorRT 11.4 | fp16 | 1.15 ms | 0.049 ms |
| v3-SiLU + objectness | Xeon 8470Q (25-core quota) · ONNX Runtime 1.30 | fp32 | 4.82 ms | 3.74 ms |
| v3 original + objectness | Xeon 8470Q (25-core quota) · ONNX Runtime 1.30 | fp32 | 4.52 ms | 3.61 ms |

- ORT conditions: intra_op=16 / inter_op=1; b1 fixed graph, b8 dynamic graph; best-case min over
  300 samples.
- TRT conditions: b1 fixed engine, b8 batch-8 engine; 30 single shots + 100 pipelined runs.
- **v3's fp16 is ~70% slower at b1** (small graphs are scheduling-overhead-bound) → fp32 at b1,
  fp16 at b8.
- b1 hugs the launch-overhead floor, b8 reflects the compute difference; ORT absolute values are
  affected by neighbor load on the host.

### Local laptop

| Weights | Platform / runtime | Precision | b1 | b8 (per image) |
|---|---|---|---:|---:|
| v3-SiLU + objectness | RTX 4060 Laptop · TensorRT 11.4 | fp32 | 1.53 ms | 0.620 ms |
| v3-SiLU + objectness | RTX 4060 Laptop · TensorRT 11.4 | fp16 | 1.97 ms | 0.197 ms |
| v3 original + objectness | RTX 4060 Laptop · TensorRT 11.4 | fp32 | 1.47 ms | 0.615 ms |
| v3 original + objectness | RTX 4060 Laptop · TensorRT 11.4 | fp16 | 1.99 ms | 0.193 ms |

- b1 is the best-case min of three rounds (single shots drift noticeably under laptop GPU
  dynamic clocks); b8 is stable.
- fp16 is slower at b1 (1.99 vs 1.47 ms, the same phenomenon as on the 5090) → fp32 at b1, fp16
  at b8.

## 9. Usage

```bash
# official weights: download + convert + evaluate
python scripts/download_weights.py --model yolov3-tiny
python scripts/convert_weights.py --src weights/yolov3-tiny.weights --dst weights/yolov3-tiny.safetensors
python scripts/eval.py --weights weights/yolov3-tiny.safetensors --data coco

# from-scratch training (recommended: SiLU + obj-iou-target; native 416 input)
python scripts/train.py --data coco --model yolov3-tiny-silu --imgsz 416 \
    --obj-iou-target --name v3tiny-e300 --batch 32 --workers 8

# released weights: evaluate / infer
python scripts/eval.py --weights yolov3-tiny-silu-e300-416.safetensors --data coco
python scripts/infer.py --weights yolov3-tiny-silu-e300-416.safetensors --image assets/traffic.jpg

# anchor coverage verdict / re-clustering
python scripts/compute_anchors.py --data coco
```

## 10. Notes

- **416 is the canonical input**; `config/train.yaml` defaults to 640, so reproducing the v3
  protocol needs an explicit `--imgsz 416`.
- The structure is fixed — don't pass `--scale` (warns and does nothing).
- The official `.weights` only match the original `yolov3-tiny`; the SiLU variant needs its own
  training or the corresponding artifact.
- Anchors are never scaled linearly with `imgsz`; re-cluster only when from-scratch training
  changes the size.
