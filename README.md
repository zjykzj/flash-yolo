# Flash-YOLO

> ⚡ **Real-time speed, end-to-end training.** A complete, independent framework for lightweight detection — train · evaluate · export · infer, every module readable on its own.
>
> No monolithic abstractions, no heavyweight dependencies — core runtime deps: PyTorch + NumPy only.

<p align="center">
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/version-0.2.0-blue.svg" alt="Version 0.2.0"></a>
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
| This repo · from scratch, 100 ep · E2E (NMS-free)³ | **36.22** | 51.32 | 2.57M | — |
| This repo · from scratch, 100 ep · NMS (o2m)³ | **37.15** | 52.67 | 2.57M | — |

- 260 layers · 2,572,280 params — identical to the official summary; weights load with `strict=True` zero-key-mapping, and the inference output is **bit-identical** to the official model
- ¹ Latency = inference stage only, averaged over 20 runs on this machine: RTX 4060 Laptop GPU / WSL2 CPU (onnxruntime)
- ² The official weights re-run through this repo's pipeline (same preprocessing, postprocessing, and pycocotools metrics, same hardware) — the numbers match this repo's exactly, which is the expected consequence of bit-identical reproduction
- ³ From scratch: trained on COCO train2017 from random weights with the built-in baseline recipe (100 epochs, fp32, one RTX 5090, ~14 h) — official pycocotools numbers. Architecture is identical to the rows above, so latency is unchanged. Not like for like with them: every official number starts from Objects365 pretraining (see Training)
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

# 6. Train YOLO26 — from scratch, any model size, finetune or resume (see "Training" below
#    for recipes, artifacts and the from-scratch vs finetuned distinction)
python scripts/train.py --data /path/to/coco --name yolo26n                     # from scratch, 100 epochs
python scripts/train.py --data /path/to/coco --name yolo26s --scale s           # same recipe, other sizes
python scripts/train.py --data /path/to/coco --name ft --weights w.safetensors  # finetune from weights
python scripts/train.py --data /path/to/coco --resume runs/train/<name>         # resume (full state)
python scripts/bench_io.py --data /path/to/coco               # optional: pick batch/workers for this box

# 7. Evaluate trained weights (official pycocotools numbers)
python scripts/eval.py --weights runs/train/<name>/weights/best.safetensors --data /path/to/coco
```

## Training

```bash
# COCO-format dataset: <split>/ or images/<split>/ + annotations/instances_<split>.json.
# YOLO labels -> COCO json: dataflow-cv convert yolo2coco <images> <labels> <classes.txt> <out.json>
#
# --data is required: the shipped config deliberately carries no machine paths, and neither do
# batch/workers (16/8 are the minimal portable values). scripts/bench_io.py sweeps loader-only
# then end-to-end and prints what to pass for *this* box — it only advises, it never edits the config.
python scripts/bench_io.py --data /path/to/coco
python scripts/train.py --data /path/to/coco --name yolo26n --batch 64 --workers 16
```

**Recipes.** With no `--recipe` you get the built-in baseline — the from-scratch recipe (fp32,
EMA, ProgLoss, close_mosaic over the last epochs, 100 epochs). Named recipes live in
`config/recipes/*.yaml` and are selected by name or by `.yaml` path:

| Recipe | Stage | Requires | Scale epochs (n/s/m/l/x) |
|---|---|---|---|
| *(none)* | from scratch / your own data | — | 100 |
| `yolo26-coco-ft` | published COCO finetuning | Objects365-pretrained init | 245/70/80/60/40 |
| `yolo26-o365-pt` | published Objects365 pretraining | the Objects365 dataset | 150 |

```bash
python scripts/train.py --data /path/to/coco --recipe yolo26-coco-ft \
    --weights yolo26n-objv1-150.safetensors        # published COCO stage (n: 245 epochs)
```

`yolo26-coco-ft` is a **finetuning** recipe — its `lr0` is 54% of the baseline's, tuned by
evolutionary search from an already-converged start. Run from random init it learns measurably
slower (same-augment screen at epoch 7: 0.215 vs 0.236 mAP@[.5:.95]). Use the baseline recipe.

**What a run writes** to `runs/train/<name>/`: `weights/best.safetensors` (EMA) · `last` ·
`best_raw` · `epochNNN.safetensors` every 20 epochs (fork mid-run) · `results.csv` (per-epoch
metrics + val losses) · `diag/train_diag.csv` (pre-clip gradient norm, per-group lr, loss
breakdown) · `samples/*.png` (augmented-sample grids with GT boxes) · `meta.json` (git/env/config
snapshot) · `run.log` (that run's console/file log) · `resume.pt`.

**Not like for like.** The published 40.1 is Objects365 pretrain (150 epochs) + COCO finetune
(245 epochs) — no official checkpoint was trained on COCO from random weights, so a from-scratch
COCO number is not comparable to it. For reference, this repo's own baseline run (100 epochs from
random init on COCO train2017, built-in recipe) reaches **36.22** mAP@[.5:.95] / 51.32 mAP@50 on the
E2E path (**37.15** / 52.67 with NMS) — the gap to the official numbers is the Objects365
pretraining plus the longer finetune schedule:

```bash
python scripts/train.py --data /path/to/coco --name yolo26n-from-scratch --batch 64 --workers 16
# -> runs/train/train-yolo26n-from-scratch/weights/best.safetensors (evaluate with scripts/eval.py)
```

## Project Structure

```
assets/    demo images (bus.jpg / zidane.jpg, provenance in assets/README.md)
config/    model config (yolo26.yaml, n/s/m/l/x scales) + train config (train.yaml + TrainConfig)
           + recipes/ (named training recipes: yolo26-coco-ft, yolo26-o365-pt)
data/      COCO readers + training dataset & augmentation pipeline (official-parity augment)
eval/      COCO evaluation (pycocotools wrapper)
logs/      logs of scripts without a run dir (gitignored; run-dir scripts write <run>/run.log)
model/     model implementation (assembler / dual Detect head / basic operator layer / weight loading)
runs/      runtime results (gitignored)
scripts/   download_weights / convert_weights / infer / export / eval / train / bench_io /
           compare_official
tests/     67 tests: weight alignment / export parity / metric correctness / training components
train/     training: TAL+STAL assigner / dual-head ProgLoss / MuSGD / EMA / trainer / FastMetrics
           (+ per-run artifacts: periodic checkpoints, gradient diag CSV, augment samples, meta.json)
utils/     anchors & decode / postprocessing (NMS) / pt·onnx engines / visualization / IoU /
           logger (dual console+file) / progress bar / paths (runs/ increment)
```

## Tests

```bash
pytest tests/    # 67 tests: weight alignment / export parity / metric correctness / training components (assigner, loss, MuSGD, EMA, checkpoint, augment geometry, config, FastMetrics)
```

`tests/test_weight_alignment.py` compares against the official .pt as a dev-time reference — install requirements-dev.txt to run it.

## 🔥 Updates

- **v0.2.0** (2026-10-08) — M3 training pipeline:
  - from-scratch YOLO26 dual-head training (ProgLoss · STAL · MuSGD) with FastMetrics validation
  - **augment pipeline aligned with the official implementation** — mosaic tile geometry, HSV space, `bgr` semantics, copy_paste, box filtering (verified pixel-identical; boxes kept per image 9.5 -> 15.4, +17.7% mAP50-95 at epoch 20 over the old path)
  - training artifacts — periodic checkpoints, gradient diagnostics, augment samples, `meta.json`, `best_raw`
  - recipes as files — built-in `default` plus `config/recipes/yolo26-coco-ft.yaml` / `yolo26-o365-pt.yaml`
  - `scripts/bench_io.py` throughput benchmark · val losses in `results.csv` · EMA decay / BN momentum parity fixes
- **v0.1.0** (2026-10-04): Initial release — a faithful YOLO26 reproduction, verified against the official model on COCO val2017 (inference · export · evaluation), bit-identical to the official model.

See [CHANGELOG.md](CHANGELOG.md) for the full history.

## 📄 License

The code is licensed under **Apache-2.0** — see [LICENSE](LICENSE) for details.

- **Pretrained weights**: the official YOLO26 weights (AGPL-3.0) are not distributed here — download them via `scripts/download_weights.py` for personal use
