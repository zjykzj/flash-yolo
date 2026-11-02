# Alignment and verification against the official implementation

> This page collects official alignment, numeric verification and test documentation. The project
> runtime **does not depend on ultralytics**; all official cross-checks are dev-time only
> (`requirements-dev.txt`) and never enter the delivery chain.

## 1. Official weight alignment (YOLO26n detection)

Protocol: COCO val2017, pycocotools (`scripts/eval.py`), 640 input. E2E = the NMS-free default
path; NMS = one-to-many head + NMS. The `ultralytics` row uses **exactly the same**
preprocessing, postprocessing and evaluation pipeline — only the forward implementation differs.

| Model | Implementation | Training | E2E mAP50-95 | E2E mAP50 | NMS mAP50-95 | NMS mAP50 |
|---|---|---|---:|---:|---:|---:|
| YOLO26n official weights | this repo | O365 150 ep + COCO 245 ep | **40.27** | 55.80 | **40.89** | 56.87 |
| YOLO26n official weights | ultralytics | same checkpoint | 40.27 | 55.80 | 40.89 | 56.87 |
| YOLO26n from scratch | this repo | 300 ep | **37.82** | **53.24** | **38.65** | **54.46** |
| Flash-YOLO from scratch | this repo | 300 ep | **35.14** | **50.15** | **36.09** | **51.56** |

- **Bit-identical reproduction**: the official checkpoint (260 layers · 2,572,280 params) loads
  with `strict=True` and zero key mapping; the forward output is bit-identical with the official
  implementation on the same input.
- The ~0.1 gap between the officially published 40.1 / 40.9 and this pipeline's 40.27 / 40.89
  comes from metric-implementation noise.
- The from-scratch rows are **not the same budget** as the official rows: every official
  checkpoint starts from Objects365 pretraining.

```bash
python scripts/download_weights.py --model yolo26n
python scripts/convert_weights.py --src weights/yolo26n.pt --dst weights/yolo26n.safetensors
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco
python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --nms

# optional cross-validation: the same weights run through this repo's pipeline
# using the ultralytics implementation (dev-time only)
python scripts/compare_official.py --data coco
python scripts/compare_official.py --data coco --nms
```

## 2. Official weight alignment (YOLO26n-seg)

Protocol: full COCO val2017 (5000 images), `scripts/eval.py` (pycocotools, Box + Mask).

| Convention | mAP50 | mAP50-95 |
|---|---:|---:|
| Official weights Box | 0.5548 | **0.3967** |
| Official weights Mask | 0.5276 | **0.3367** |
| This repo from-scratch 300 ep Box | 0.5369 | 0.3790 |
| This repo from-scratch 300 ep Mask | 0.5116 | 0.3237 |

The gap between from-scratch 300 ep (default recipe) and official (O365 + COCO) is **1.77 / 1.30**
(Box / Mask mAP50-95), the same magnitude as the detection side.

The official training-time validator records box **0.3975** / mask **0.3386**; the ~0.002 gap vs
this pipeline is metric-implementation noise, the same class as 40.1 vs 40.27 on the detection
side. Mask decoding is pixel-identical with the official val; on the export side pt(CPU) vs
onnx(CPU) raw outputs differ by ~1e-6 relative, cross-device (CUDA vs CPU oneDNN) by 1e-3~4e-3
with mask-boundary pixel flips (worst instance IoU 0.993) — cross-device alignment should
compare within one runtime or go through TensorRT.

## 3. Training-internal alignment

Item-by-item cross-checks against the official implementation at the same weights and the same
real batch:

- **Loss / gradients**: total loss difference <1%, gradient-norm difference <2% (this repo 2091
  vs official 2130).
- **o2o branch**: detaches the feature input, not the output — gradients flow back to
  backbone/neck through o2m only, the o2o head itself still trains.
- **Optimizer**: MuSGD grouping, NS iterations and coefficients match the official; gradient
  clipping `max_norm=10` runs unconditionally every step; the EMA formula
  `decay*(1-exp(-steps/tau))` and its update timing match; ProgLoss updates every epoch.
- **Augmentation**: mosaic canvas 1,638,400 px 0-diff, affine output 409,600 px 0-diff, labels
  0-diff.
- **Data formats**: COCO-json and YOLO-txt loaders produce identical boxes on the same image set
  (0 diff); training-side statistics equal item by item (14205 / 14205).

Related rules: [../pipeline/train.md](../pipeline/train.md),
[../pipeline/eval.md](../pipeline/eval.md) and the model pages.

## 4. Runtime consistency

- **pt ↔ onnx**: raw output relative diff <5e-4, E2E detections <1e-3.
- **TensorRT**: the fp32 engine differs from PyTorch by <2e-3; the fp16 engine's raw output is
  **bit-identical** with the PyTorch half model (both are fp16 multiply + fp32 accumulate +
  final rounding).
- **Segmentation dual output**: `output0` (detections + coefficients) and `output1` (prototypes)
  keep one contract across the pt / onnx / TRT backends; mask postprocessing is pixel-identical
  with the official val.

## 5. Reproducibility

- `--deterministic` by default: cuDNN autotune off, deterministic kernels, augmentation RNG
  replays every epoch; two runs with the same seed produce **bit-identical** weights, and resume
  equals uninterrupted training **bit-identically**.
- `FLASH_YOLO_REPRO=1` gates the CPU/GPU dual-run and resume-equivalence tests (measured
  0-bit-diff).
- Training artifacts record arch/scale/nc/imgsz/names/anchors and the provenance commit in
  metadata — weights are self-contained and auditable.

## 6. Tests

```bash
pytest tests/ -q          # core suite; env-gated suites skip by default
pytest tests/test_weight_alignment.py -v   # needs dev deps + converted official weights
```

- **Environment gates**: `FLASH_YOLO_SMOKE` (CPU smoke / mini training loop), `FLASH_YOLO_REPRO`
  (CPU/GPU dual-run + resume equivalence), `FLASH_YOLO_DDP` (2-process CPU gloo).
- **Coverage**: bit-identical weight alignment, export parity, evaluator and metric correctness,
  training components (assigner / loss parity / MuSGD / EMA / checkpoint / augment / dataset /
  config / smoke / head-train), v3 (model / loss / darknet converter / engine / export
  cross-checks), segmentation (model / loss / dataset / metrics / validator / smoke / repro).
- CI: `.github/workflows/tests.yml` runs the ungated CPU core suite on push/PR (no ultralytics
  installed).
