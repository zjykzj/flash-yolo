# YOLO26 (detection)

> Reproduction target: official YOLO26n weights through this repo's pipeline reach **40.27**
> mAP50-95 E2E / **40.89** NMS.
> Structure: `config/models/yolo26.yaml`; segmentation variant:
> [yolo26-seg.md](yolo26-seg.md), pose variant: [yolo26-pose.md](yolo26-pose.md).

## 1. Conventions

- The official 40.1 (E2E) / 40.9 (NMS) are **not from-scratch training**: every official
  checkpoint is Objects365v1 pretraining (150 epochs) + COCO finetuning (n: 245 epochs,
  batch 128). The official docs state: "No YOLO26 checkpoint was trained on COCO from random
  weights … a from-scratch comparison against those numbers is not like for like".
- This repo's reproduction = rerunning official weights through the same preprocessing,
  postprocessing, pycocotools metrics and hardware; from-scratch results are in §4.
- "Official" below refers to the YOLO26 weights/recipes/numbers published by Ultralytics, which
  this repo does not redistribute.

## 2. Structure (n)

24 yaml modules (`model.model` 0–23), a 260-layer module tree, **2,572,280 params / 5.48 GFLOPs
@640**:

| Part | idx | Content |
|---|---|---|
| Backbone | 0–10 | Conv k3 s2 downsampling → C3k2 → SPPF (k=5, n=3) → C2PSA (1× PSABlock) |
| Neck | 11–22 | Upsample ×2 → Concat → C3k2, P3/P4/P5 scales |
| Head | 23 | dual Detect (`cv2/cv3` + `one2one_cv2/one2one_cv3`), reg_max=1, E2E top-k in-graph |

- Detection layers P3/P4/P5 (stride 8/16/32), 640×640 input, up to 300 detections per image.
- Channels: n = 3 → 16 → 32 → 64 → 128 → 256; m/l/x force `c3k=True` on C3k2.
- BN eps=0.001, momentum=0.03; scale rules: [README.md](README.md).

### Detect head

- Box: `cv2[i]` = Conv 3×3 → Conv 3×3 → Conv2d(c2→4), `c2 = max(16, ch[0]//4, 4·reg_max)`;
  with reg_max=1 the 4 channels are ltrb distances (no DFL).
- Cls: `cv3[i]` = DWConv+Conv two-stage → Conv2d(c3→nc), `c3 = max(ch[0], min(nc,100))`.
- `one2one_cv2/one2one_cv3 = deepcopy(cv2/cv3)`.
- bias_init: box final-layer bias=2.0; cls final-layer bias = `log(5/nc/(S/stride)²)`,
  S = training input side (the official code hardcodes 640; from-scratch training at a non-640
  size gets the prior skewed by (640/S)², which only affects the start).

### Decode and postprocess

- **E2E**: ltrb → `xyxy = anchor ± dist × stride`; scores=sigmoid; two-stage top-k(300):
  ① per-anchor max class score → top-300 anchors; ② global top-300 over the 300×80 candidates →
  output `(B, 300, 6) = [x1,y1,x2,y2,conf,cls]`. The ordering must match exactly.
- **NMS**: o2m decode `(B, 4+nc, 8400)`, **xywh** (official `xywh=not end2end`) → to xyxy →
  class-aware greedy NMS (conf 0.001 / iou 0.7 / max_det 300); candidates sorted globally by
  score, cross-class coordinate-offset isolation, early stop at max_det.
- Preprocess: letterbox to 640 (gray 114), BGR→RGB, /255, no mean/std.

## 3. Weight download and initialization

Download links point to the **v1.0.0** release assets.

| Weights | Training | Input | Download |
|---|---|---|---|
| `yolo26n-from-scratch-e300-640` | 300 ep from scratch | 640 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/yolo26n-from-scratch-e300-640.safetensors) |

Official weights (O365 pretraining + COCO finetuning) are not distributed with the Release;
download and convert them yourself:

```bash
python scripts/download_weights.py --model yolo26n
python scripts/convert_weights.py --src weights/yolo26n.pt --dst weights/yolo26n.safetensors
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco
```

- The official ckpt's `ckpt["model"]` is the full module object (`ema` is None); the converter
  unwraps it once into plain safetensors — zero ultralytics dependency at runtime.
- `strict=True` full-key matching is the first gate of structural alignment; success proves the
  topology matches.
- Local artifacts carry `arch=yolo26` / `scale` / `nc` / `imgsz` / `names` metadata, so
  eval/infer/export resolve automatically.

## 4. Formal metrics (COCO val2017 @640)

| Training | E2E mAP50-95 | E2E mAP50 | NMS mAP50-95 | NMS mAP50 |
|---|---:|---:|---:|---:|
| Official weights (O365 + COCO) | **40.27** | 55.80 | **40.89** | 56.87 |
| From scratch 300 ep | **37.82** | **53.24** | **38.65** | **54.46** |

- The ~0.1 gap between the reproduced 40.27 / 40.89 and the official 40.1 / 40.9 is
  metric-implementation noise.

## 5. Inference latency

> Protocol: **model-only** (no pre/postprocessing), 640 input; b1 = single image, b8 = per-image
> average of batch 8. Engines warm up at init; numbers are steady-state repeats. Latency is the
> same across different weights of one architecture.

### Cloud server / workstation

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | 1.37 ms | 0.245 ms |
| RTX 5090 | TensorRT 11.4 · fp16 | 1.28 ms | 0.135 ms |
| Xeon 8470Q (25-core quota) | ONNX Runtime 1.30 · fp32 | 14.6 ms | 6.86 ms |

- ORT conditions: intra_op=16 / inter_op=1; b1 fixed graph, b8 dynamic graph; best-case min over
  300 samples.
- TRT conditions: b1 fixed engine, b8 batch-8 engine; 30 single shots + 100 pipelined runs.
- b1 hugs the launch-overhead floor; b8 reflects the compute difference. ORT absolute values are
  affected by neighbor load on the host.

### Local laptop

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 4060 Laptop | TensorRT 11.4 · fp32 | 3.63 ms | 1.40 ms |
| RTX 4060 Laptop | TensorRT 11.4 · fp16 | 3.41 ms | 0.606 ms |

- b1 is the best-case min of three rounds (single shots drift noticeably under laptop GPU
  dynamic clocks); b8 is stable.

## 6. Usage

```bash
python scripts/train.py --data coco --model yolo26 --name yolo26n-coco --batch 32 --workers 8
python scripts/eval.py --weights yolo26n-from-scratch-e300-640.safetensors --data coco
python scripts/export.py --weights yolo26n-from-scratch-e300-640.safetensors --trt
python scripts/infer.py --weights yolo26n-from-scratch-e300-640.safetensors \
    --image assets/traffic.jpg
```

Training arguments, recipes, DDP: [../pipeline/train.md](../pipeline/train.md); evaluation
protocol: [../pipeline/eval.md](../pipeline/eval.md).

## 7. Notes

- The attribute tree (`cv1/cv2/cv3/m`, `one2one_*`) must keep the official names; renaming breaks
  strict load.
- Assembler metadata uses `m_.repeats` — **never attach `m_.n`** (SPPF's `self.n` is its pool
  count).
- The E2E two-stage top-k order must match the official implementation, or the bit-level
  cross-check fails.
- From-scratch training defaults to fp32; `--amp` is only suggested for finetuning.
