# YOLO26-seg (instance segmentation)

> Detection structure, loss and training pipeline: [yolo26.md](yolo26.md); this page documents
> only the segmentation differences.
> Structure: `config/models/yolo26-seg.yaml`.

## 1. Structure

backbone/neck are **identical line-by-line** with the detection variant (the official yaml diff
is exactly the last line); the head becomes `Segment26` = `Detect` + two segmentation-specific
pieces:

| Piece | Content |
|---|---|
| `cv4` / `one2one_cv4` | per-level 3×3 conv stack → `nm=32` mask coefficients (`c4 = max(ch[0]//4, 32)`) |
| `proto` (Proto26) | P4/P5 → 1×1 onto P3 channels → nearest upsample ×2/×4 summed → 3×3 → ConvTranspose2d ×2 → 3×3 → 1×1, output `(B, 32, H/4, W/4)` |
| `proto.semseg` | training-time auxiliary semantic head: BCE + multi-channel Dice, `loss[4] *= box_gain`; not computed in val/export (zero cost) |

- `npr` (yaml 256) scales with width: n → 64; `nm=32` is a task constant, not scaled.
- n totals **3,126,280 params / 309 layers / 9.3 GFLOPs @640**.
- The o2o branch's proto is `detach()`ed (gradients flow back through o2m only).

## 2. Inference contract

- E2E: `(B, 300, 6+nm)` detections+coefficients + `(B, nm, H/4, W/4)` prototypes.
- raw: `(B, 4+nc+nm, N)` + prototypes.
- The E2E two-stage top-k is shared with detection (coefficients fetched with the same anchor
  indices).
- Mask decode `utils/postprocess.process_mask`: coefficients @ prototypes → bilinear upsample to
  the network input size → binarize `>0` → crop by box → de-letterbox resample to the original
  image → binarize `>0.5` (the official `process_mask(upsample=True)` + `scale_masks` two-stage
  convention).

## 3. Training

- `ComputeLossSeg` reuses the detection dual-branch losses (box/cls/l1 + ProgLoss), with an extra
  mask term per branch: coefficients @ prototypes → BCE (cropped by the responsible GT box,
  scaled by normalized box area, summed over the batch / positive samples), mask gain =
  `box_gain`.
- Data accepts COCO polygons/RLE and YOLO-seg polygon rows (see
  [../datasets/coco.md](../datasets/coco.md) / [../datasets/yolo.md](../datasets/yolo.md)).
- `mask_ratio` (default 4) downsamples GT masks to `imgsz/mask_ratio` (640 → 160, the Proto26
  grid); `overlap_mask=True` produces an area-descending instance index map.
- Official seg recipe: `yolo26-seg-coco-ft` (`mask_ratio=1` / `overlap_mask=true` /
  `copy_paste_mode=flip`), needs the 365-class `yolo26*-objv1-seg` initialization.
- Validation computes Box + Mask simultaneously; best fitness = box mAP50-95 + mask mAP50-95;
  formal numbers still come from `scripts/eval.py`.

## 4. Weight download and initialization

Download links point to the **v1.0.0** release assets.

| Weights | Training | Input | Download |
|---|---|---|---|
| `yolo26n-seg-from-scratch-e300-640` | 300 ep from scratch | 640 | [safetensors](https://github.com/zjykzj/flash-yolo/releases/download/v1.0.0/yolo26n-seg-from-scratch-e300-640.safetensors) |

Official weights are not distributed with the Release; download them yourself:

```bash
# official COCO 80-class weights (evaluate / infer directly)
python scripts/download_weights.py --model yolo26n-seg
python scripts/convert_weights.py --src weights/yolo26n-seg.pt --dst weights/yolo26n-seg.safetensors

# O365 365-class segmentation init (for the official COCO-stage finetune)
python scripts/download_weights.py --model yolo26n-objv1-seg
python scripts/convert_weights.py --src weights/yolo26n-objv1-seg.pt \
    --dst weights/yolo26n-objv1-seg.safetensors
```

The official O365 seg init is **365-class**, while COCO training builds an 80-class head.
`--weights` loads through `load_weights_finetune` by key+shape intersection:

- backbone/neck/proto/cv2 load bit-for-bit;
- class-dependent heads (`cv3/one2one_cv3/semseg`, …) are rebuilt for the target nc;
- n measures 740/844 tensors loaded, 104 class-head tensors reinitialized;
- key mismatches / wrong architecture / non-head shape mismatches error out.

## 5. Formal metrics (COCO val2017 @640, Box + Mask)

| Training | Box mAP50-95 | Box mAP50 | Mask mAP50-95 | Mask mAP50 |
|---|---:|---:|---:|---:|
| Official weights (O365 + COCO) | **39.67** | 55.48 | **33.67** | 52.76 |
| From scratch 300 ep | **37.90** | **53.69** | **32.37** | **51.16** |

- The official training-time validator records box 39.75 / mask 33.86; this pipeline reports
  39.67 / 33.67 — a ~0.002 gap from metric-implementation noise.

## 6. Inference latency

> Protocol: **model-only** (no mask decode / pre/postprocessing), 640 input; b1 = single image,
> b8 = per-image average of batch 8. Engines warm up at init. b1 is single-shot (with H2D and
> output readback), b8 is pipelined (per image).

### Cloud server / workstation

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 5090 | TensorRT 11.4 · fp32 | 1.52 ms | 0.303 ms |
| RTX 5090 | TensorRT 11.4 · fp16 | 2.34 ms | 0.148 ms |
| Xeon 8470Q (25-core quota) | ONNX Runtime 1.30 · fp32 | 16.8 ms | 12.0 ms |

- ORT conditions: intra_op=16 / inter_op=1; b1 fixed graph, b8 dynamic graph; best-case min over
  300 samples.
- TRT conditions: b1 fixed engine, b8 batch-8 engine; 30 single shots + 100 pipelined runs.
- **fp16 is slower than fp32 at b1**: dominated by the proto output (32×160×160) readback and
  the half→float conversion; pure GPU execution is close (0.71 vs 0.73 ms). b8 is where fp16
  wins (0.148 vs 0.303 ms).
- ORT absolute values are affected by neighbor load on the host; b1 hugs the launch-overhead
  floor, b8 reflects the compute difference.

### Local laptop

| Platform | Runtime / precision | b1 | b8 (per image) |
|---|---|---:|---:|
| RTX 4060 Laptop | TensorRT 11.4 · fp32 | 4.16 ms | 2.01 ms |
| RTX 4060 Laptop | TensorRT 11.4 · fp16 | 3.73 ms | 0.791 ms |

- b1 is the best-case min of three rounds (single shots drift noticeably under laptop GPU
  dynamic clocks); b8 is stable.

> Mask decode and IoU are currently mostly on the CPU; deployed end-to-end latency is
> significantly higher than model-only and needs a separate measurement.

## 7. Usage

```bash
python scripts/train.py --data coco --model yolo26-seg --name seg-coco --batch 32 --workers 8
python scripts/train.py --data coco --recipe yolo26-seg-coco-ft \
    --weights weights/yolo26n-objv1-seg.safetensors --batch 32 --workers 8

python scripts/eval.py --weights yolo26n-seg-from-scratch-e300-640.safetensors --data coco
python scripts/export.py --weights yolo26n-seg-from-scratch-e300-640.safetensors --trt
python scripts/infer.py --weights yolo26n-seg-from-scratch-e300-640.safetensors \
    --image assets/traffic.jpg
```

## 8. Notes

- Mask decode + IoU currently run on the CPU (the in-training `process_mask_lowres` chunks
  automatically); `process_mask` on GPU uses the engine device, bit-identical.
- `mask_ratio=1 @640` mask loss uses the per-GT-box sliced implementation, with far lower GPU
  memory than a whole-image gather.
- `val_sem` is always 0 in val mode (official convention) — it does not mean semseg is untrained.
