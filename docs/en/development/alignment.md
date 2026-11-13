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

## 3. Official weight alignment (YOLO26n-pose)

Convention: the **2,346 COCO-pose val2017 images with valid keypoints**, `scripts/eval.py`
(pycocotools keypoints, OKS, maxDets=20); the `ultralytics` row uses the same images and the
same pre/postprocessing, only the forward differs.

| Model | Implementation | E2E AP | E2E AP50 | NMS AP | NMS AP50 |
|---|---|---|---:|---:|---:|---:|
| YOLO26n-pose official weights | this repo | **0.5687** | 0.8250 | **0.5670** | 0.8280 |
| YOLO26n-pose official weights | Ultralytics (same 2,346 images) | 0.572 | 0.833 | 0.570 | 0.831 |

- **Bit-exact reproduction**: the official `yolo26n-pose` (363 layers, **3,679,464 params**)
  loads with `strict=True` on every key (including `cv4_kpts` / `cv4_sigma` / `flow_model`);
  on the same input the E2E `(1,300,57)` and raw `(1,56,8400)` outputs match the official
  forward with **max diff 0.0**.
- The ~0.003 AP gap to Ultralytics comes from square letterbox (the official val uses auto/rect)
  and metric-implementation noise; the official training-time validator records pose 0.5717 /
  AP50 0.8332, the same order.
- Export/runtime: ONNX E2E/raw metrics differ from pt by ≤0.001 on the 50-image subset; raw
  output differs element-wise by ≤1e-2 px (mean ~3e-5); TRT fp32/fp16 E2E AP on the same subset
  is 0.6442 / 0.6448 vs 0.6451 for pt (no detection flips above conf=0.25).
- **Training internals**: `ComputeLossPose` matches the official `PoseLoss26` on a real
  COCO-pose batch with total loss **40.0884819 vs 40.0884819 (rel 0)**; the o2o pose/kobj/rle
  terms match item by item (`tests/test_loss_pose.py`).
- Tests: `tests/test_pose.py` (strict load / forward parity / export-branch regression),
  `tests/test_dataset_pose.py` (keypoint alignment / flip_idx / mosaic),
  `tests/test_metrics_pose.py` (OKS AP), `tests/test_train_smoke.py -k pose` (gated).

### 3.1 10-epoch finetune pipeline parity (2026-11)

Protocol: official `yolo26n-pose` weights; this repo and ultralytics 8.4.173 use the same
**56,599 training images / 2,346 val images**, 10 epochs, batch 64, nbs 64, 640, MuSGD, AMP,
seed 0. Both disable `close_mosaic` and override the effective values of the released
ultralytics code (`cls_w=1.0` / `prog_alpha_init=0.8` / `muon_w=0.2` / `sgd_w=1.0`;
that version does not consume `cls_w`, and the other three are hardcoded — the
`yolo26-pose-coco` recipe records the official checkpoint's `train_args`). Both checkpoints
are then re-scored by the same `scripts/eval.py --data coco-pose`.

| Implementation | E2E AP | E2E AP50 | NMS AP | NMS AP50 |
|---|---:|---:|---:|---:|
| this repo, 10 epochs | **0.5476** | 0.8089 | **0.5528** | 0.8170 |
| ultralytics, 10 epochs | **0.5477** | 0.8164 | **0.5508** | 0.8203 |

- E2E AP difference **0.0001**, NMS AP difference **0.0020**; the largest AP75 / AP(M) /
  AP(L) / AR difference is about **0.004**, inside single-seed, 10-epoch noise — one
  end-to-end parity check of data / augmentation / loss / optimizer / EMA / validation.
- This is a **pipeline parity check, not a convergence result**: finetuning the converged
  official weights for 10 epochs at lr0 0.005 drops both to ~0.548 (official start 0.5687);
  it is not a model-capability ceiling.
- Training-time validator pose AP previously differed strongly from pycocotools (this repo
  FastPose 0.4367, ultralytics PoseMetrics 0.3676 vs formal 0.5476 / 0.5477). The repo side was
  corrected in 2026-11 for `num_keypoints==0` and OKS `area`; on the 300-epoch best checkpoint
  it is now within ~0.02 AP of the formal number, with AR@20 matching (see §3.2). Training logs
  remain trend/best-picking only and are not comparable across implementations.

```bash
# this repo; the ultralytics side uses the same data and effective hyperparameters (dev-time)
python scripts/train.py --data coco-pose --model yolo26-pose \
    --weights weights/yolo26n-pose.safetensors \
    --recipe yolo26-pose-coco --epochs 10 --batch 64 \
    --close-mosaic 0 --cls-w 1.0 --prog-alpha-init 0.8 \
    --muon-w 0.2 --sgd-w 1.0 --amp
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose
python scripts/eval.py --weights runs/train/<run>/weights/best.safetensors --data coco-pose --nms
```

- The full 400-epoch official recipe has not been reproduced here; the 300-epoch from-scratch
  baseline is in §3.2.

### 3.2 300-epoch from-scratch baseline (2026-11)

Protocol: `default` recipe, batch 32, nbs 64, fp32, 640, one GPU per run; the two runs below
were executed concurrently on separate GPUs. Formal evaluation is still `scripts/eval.py` on
the 2,346-image COCO-pose val2017 split with maxDets=20.

| Model | Path | AP | AP50 | AP75 | AP(M) | AP(L) | AR |
|---|---|---:|---:|---:|---:|---:|---:|
| yolo26n-pose | E2E | **0.5529** | 0.8200 | 0.5947 | 0.4686 | 0.6748 | 0.6193 |
| yolo26n-pose | NMS | **0.5541** | 0.8249 | 0.5984 | 0.4704 | 0.6747 | 0.6166 |
| flash-yolo-pose | E2E | **0.5004** | 0.7832 | 0.5285 | 0.4024 | 0.6390 | 0.5681 |
| flash-yolo-pose | NMS | **0.5055** | 0.7884 | 0.5380 | 0.4051 | 0.6450 | 0.5669 |

- Same-budget gap: 5.25 AP (E2E) / 4.86 AP (NMS); flash uses 16.3% fewer parameters and 22.4%
  fewer GFLOPs.
- The gap widened from 3.54 AP at 10 epochs to 5.25 AP at 300 epochs, so it is driven by
  architecture capacity rather than an undertrained schedule.
- Official `yolo26n-pose` is O365 detection pretraining + 400 COCO-pose epochs, so this is still
  not like-for-like; the from-scratch 0.5529 is 1.58 AP below the official-weight 0.5687.

## 4. Training-internal alignment

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

## 5. Runtime consistency

- **pt ↔ onnx**: raw output relative diff <5e-4, E2E detections <1e-3.
- **TensorRT**: the fp32 engine differs from PyTorch by <2e-3; the fp16 engine's raw output is
  **bit-identical** with the PyTorch half model (both are fp16 multiply + fp32 accumulate +
  final rounding).
- **Segmentation dual output**: `output0` (detections + coefficients) and `output1` (prototypes)
  keep one contract across the pt / onnx / TRT backends; mask postprocessing is pixel-identical
  with the official val.

## 6. Reproducibility

- `--deterministic` by default: cuDNN autotune off, deterministic kernels, augmentation RNG
  replays every epoch; two runs with the same seed produce **bit-identical** weights, and resume
  equals uninterrupted training **bit-identically**.
- `FLASH_YOLO_REPRO=1` gates the CPU/GPU dual-run and resume-equivalence tests (measured
  0-bit-diff).
- Training artifacts record arch/scale/nc/imgsz/names/anchors and the provenance commit in
  metadata — weights are self-contained and auditable.

## 7. Tests

```bash
pytest tests/ -q          # core suite; env-gated suites skip by default
pytest tests/test_weight_alignment.py -v   # needs dev deps + converted official weights
```

- **Environment gates**: `FLASH_YOLO_SMOKE` (CPU smoke / mini training loop), `FLASH_YOLO_REPRO`
  (CPU/GPU dual-run + resume equivalence), `FLASH_YOLO_DDP` (2-process CPU gloo).
- **Coverage**: bit-identical weight alignment, export parity, evaluator and metric correctness,
  training components (assigner / loss parity / MuSGD / EMA / checkpoint / augment / dataset /
  config / smoke / head-train), v3 (model / loss / darknet converter / engine / export
  cross-checks), segmentation (model / loss / dataset / metrics / validator / smoke / repro),
  pose (model / loss / dataset / metrics / smoke / export-branch regression).
- CI: `.github/workflows/tests.yml` runs the ungated CPU core suite on push/PR (no ultralytics
  installed).
