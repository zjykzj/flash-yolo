# Flash-YOLO

A lightweight framework for real-time detection: **train / evaluate / export / infer**.

The opposite design philosophy of ultralytics: not everything-in-one, but **every module
independent, complete, and readable on its own**. Core runtime deps: PyTorch + NumPy only —
inference and evaluation have zero dependency on the ultralytics ecosystem.

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **M1** | YOLO26 architecture reproduction + weight alignment (260 layers, 2,572,280 params; zero-mapping strict load; bit-identical to official output) | ✅ Done |
| **M2** | Inference (.pt / .onnx, E2E NMS-free & NMS paths), pt→onnx export, COCO evaluation (40.27 / 40.89 vs official 40.1 / 40.9) | ✅ Done |
| **M3** | Training pipeline (ProgLoss, STAL label assignment, MuSGD optimizer) | ⏳ Planned |
| **M4** | Extend to segmentation / classification | ⏳ Planned |

## Verified Results (COCO val2017, YOLO26n)

| Path | Flash-YOLO | Official | Note |
|---|---|---|---|
| E2E (one-to-one, NMS-free) | **40.27** | 40.1 | bit-identical to official inference output |
| NMS (one-to-many) | **40.89** | 40.9 | raw output matches element-wise (< 1e-4) |

- 2,572,280 parameters, identical to the official summary (260 layers); weights load with
  `strict=True` zero-key-mapping — a successful load proves structural identity
- Metrics use pycocotools (the de-facto COCO standard); the official numbers come from
  ultralytics' own metric implementation, so a ~0.1 gap is expected metric-implementation noise

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Prepare weights (official models are AGPL-3.0, personal research / verification only).
#    Convert once to pure safetensors; runtime never needs ultralytics.
python scripts/download_weights.py --model yolo26n
python scripts/convert_weights.py --src weights/yolo26n.pt --dst weights/yolo26n.safetensors

# 3. Inference (default E2E NMS-free; --nms for o2m+NMS; --engine onnx for onnx;
#    --image accepts a single image or a directory)
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/bus.jpg
python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/

# 4. Export pt -> onnx (E2E fused graph, single output (B,300,6); --raw for the raw head output)
python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n.onnx

# 5. COCO evaluation (both paths verified against official numbers)
python scripts/eval.py --weights weights/yolo26n.safetensors --data /path/to/coco          # E2E -> 40.1
python scripts/eval.py --weights weights/yolo26n.safetensors --data /path/to/coco --nms    # NMS -> 40.9
python scripts/eval.py --weights weights/yolo26n.onnx --engine onnx --data /path/to/coco   # onnx engine
```

### Console output (ultralytics-style)

```
Flash-YOLO 0.1.0 🚀 Python 3.12.13 · torch 2.13.0+cu130 · CUDA NVIDIA GeForce RTX 4060 Laptop GPU
YOLO26n · 260 layers · 2,572,280 params · E2E (NMS-free) · engine pt
image 1/1 assets/bus.jpg: 810x1080 4 persons, 1 bus, 42.0ms
Speed: 9.0ms preprocess, 32.8ms inference, 0.2ms postprocess per image at shape (1, 3, 640, 640)
Total: 77.6ms end-to-end (incl. image read + save)
```

## Results & Logs

Results follow the ultralytics `runs/` convention — task-scoped directories with an
incrementing suffix, so repeated runs never overwrite each other:

```
runs/predict/predictN/   annotated images + labels/<same-name>.txt (YOLO format: cls xc yc w h, normalized)
runs/val/valN/           metrics.txt (table identical to console) + results.json (COCO format, re-evaluatable)
runs/export/             exported <weights>.onnx (default when --out is omitted)
logs/                    one timestamped log file per run (10MB × 5 rotation)
```

Every script logs to console (level-colored) and `logs/<script>_<timestamp>.log` by default.
Concurrent runs are naturally isolated by timestamped filenames (no cross-process locking).

## Project Structure

```
config/    model config (yolo26.yaml, n/s/m/l/x scales) + default hyperparams
data/      COCO dataset reader + preprocessing (letterbox)
eval/      COCO evaluation (pycocotools wrapper)
model/     assembler (yolo26.py) / dual Detect head (head.py) / weight loading (weights.py) /
           summary.py (print architecture: `python model/summary.py`)
  basic/   basic operator layer, split by type — readable file by file:
           conv.py (Conv/DWConv) · pool.py (SPPF) · block.py (residual/CSP + attention blocks)
scripts/   download_weights / convert_weights / infer / export / eval
utils/     anchors & decode / postprocessing (NMS) / pt·onnx engines / visualization / IoU /
           logger (dual console+file) / paths (runs/ increment)
assets/    demo images (bus.jpg / zidane.jpg, provenance in assets/README.md)
runs/      runtime results (gitignored)
logs/      runtime logs (gitignored)
tests/     three acceptance tests (weight alignment / export parity / metric correctness)
```

## Tests

```bash
pytest tests/    # 8 tests: param count, strict weight load, numeric alignment vs official,
                 # onnx parity, synthetic metric correctness
```

`tests/test_weight_alignment.py` needs the official .pt as a dev-time reference
(see requirements-dev.txt).

## License

- **Code**: Apache-2.0 — independently reimplemented, contains no ultralytics source
- **Official weights**: AGPL-3.0 — not distributed here; fetched by `scripts/download_weights.py`
  at the user's own discretion
