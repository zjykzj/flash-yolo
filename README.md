# Flash-YOLO

> ⚡ **Real-time speed, end-to-end training.** A complete, independent framework for lightweight detection — train · evaluate · export · infer, every module readable on its own.
>
> No monolithic abstractions, no heavyweight dependencies — core runtime deps: PyTorch + NumPy only.

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version 0.1.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg" alt="PyTorch"></a>
</p>

Detection performance — YOLO26n on COCO val2017:

| Path | mAP@[.5:.95] | mAP@50 | Params | Latency¹ |
|---|---|---|---|---|
| This repo · E2E (NMS-free) | **40.27** | 55.80 | 2.57M | 20.4ms GPU · 57.2ms ONNX CPU |
| Official · E2E² | 40.27 | 55.80 | 2.57M | 18.6ms GPU |
| This repo · NMS (o2m) | **40.89** | 56.88 | 2.57M | 16.5ms GPU |
| Official · NMS² | 40.89 | 56.88 | 2.57M | 14.6ms GPU |

- 260 layers · 2,572,280 params — identical to the official summary; weights load with `strict=True` zero-key-mapping, and the inference output is **bit-identical** to the official model
- ¹ Latency = inference stage only, averaged over 20 runs on this machine: RTX 4060 Laptop GPU / WSL2 CPU (onnxruntime)
- ² The official weights re-run through this repo's pipeline (same preprocessing, postprocessing, and pycocotools metrics, same hardware) — the numbers match this repo's exactly, which is the expected consequence of bit-identical reproduction
- The official published 40.1 / 40.9 come from the official metric implementation; the ~0.1 delta to this table is metric-implementation noise, not a model difference

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

# 4. Export pt -> onnx (E2E fused graph, single output (B,300,6); --raw for the raw head output;
#    --dynamic for dynamic batch, default fixed batch=1)
python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n.onnx

# 5. COCO evaluation (both paths verified against official numbers)
python scripts/eval.py --weights weights/yolo26n.safetensors --data /path/to/coco          # E2E -> 40.1
python scripts/eval.py --weights weights/yolo26n.safetensors --data /path/to/coco --nms    # NMS -> 40.9
python scripts/eval.py --weights weights/yolo26n.onnx --engine onnx --data /path/to/coco   # onnx engine

# 6. Train from scratch (COCO-format dataset: images/<split>/ + annotations/instances_<split>.json;
#    fp32 by default, per-epoch validation, best/last weights -> runs/train/<name>/weights/;
#    default (no --recipe) = built-in baseline, 100 epochs; --recipe <name> loads
#    config/recipes/<name>.yaml (e.g. yolo26-coco-ft = published COCO-stage recipe, needs
#    Objects365-pretrained init; official 40.1 = Objects365 pretrain +
#    COCO finetune, from-scratch COCO is not a like-for-like comparison;
#    YOLO labels -> COCO json: dataflow-cv convert yolo2coco <images> <labels> <classes.txt> <out.json>)
python scripts/train.py --data /path/to/coco                     # default recipe (100 epochs)
python scripts/train.py --data /path/to/coco --recipe yolo26-coco-ft  # published COCO stage (n: 245)

# 7. Evaluate trained weights (official pycocotools numbers)
python scripts/eval.py --weights runs/train/<name>/weights/best.safetensors --data /path/to/coco
```

## Project Structure

```
assets/    demo images (bus.jpg / zidane.jpg, provenance in assets/README.md)
config/    model config (yolo26.yaml, n/s/m/l/x scales) + train config (train.yaml + TrainConfig)
data/      COCO readers + training dataset & mosaic augmentation pipeline
eval/      COCO evaluation (pycocotools wrapper)
logs/      runtime logs (gitignored)
model/     model implementation (assembler / dual Detect head / basic operator layer / weight loading)
runs/      runtime results (gitignored)
scripts/   download_weights / convert_weights / infer / export / eval / train / compare_official
tests/     50+ tests: weight alignment / export parity / metric correctness / training components
train/     training: TAL+STAL assigner / dual-head ProgLoss / MuSGD / EMA / trainer / FastMetrics
utils/     anchors & decode / postprocessing (NMS) / pt·onnx engines / visualization / IoU /
           logger (dual console+file) / progress bar / paths (runs/ increment)
```

## Tests

```bash
pytest tests/    # 50+ tests: weight alignment / export parity / metric correctness / training components (assigner, loss, MuSGD, EMA, checkpoint, FastMetrics)
```

`tests/test_weight_alignment.py` compares against the official .pt as a dev-time reference — install requirements-dev.txt to run it.

## 🚀 Changelog

- **Unreleased**: M3 training pipeline — from-scratch YOLO26 dual-head training (ProgLoss · STAL · MuSGD), FastMetrics validation, best/last checkpoints; training recipes as files — built-in `default` (100 epochs) plus `config/recipes/yolo26-coco-ft.yaml` /
`yolo26-o365-pt.yaml` (published per-scale recipes, `--recipe <name>` or a .yaml path); val losses written to results.csv; EMA decay / BN momentum / scale-aug parity fixes vs the official training loop
- **v0.1.0** (2026-10-04): Initial release — a faithful YOLO26 reproduction, verified against the official model on COCO val2017 (inference · export · evaluation), bit-identical to the official model.

See [CHANGELOG.md](CHANGELOG.md) for the full history.

## 📄 License

The code is licensed under **Apache-2.0** — see [LICENSE](LICENSE) for details.

- **Pretrained weights**: the official YOLO26 weights (AGPL-3.0) are not distributed here — download them via `scripts/download_weights.py` for personal use
