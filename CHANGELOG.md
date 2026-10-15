# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-10-15

### Added

- **Bit-reproducible training by default (`--deterministic`, on by default)**: deterministic cuDNN/CUBLAS settings make same-seed runs byte-identical on the same machine at noise-level cost (3-5% on isolated compute, indistinguishable end-to-end); `--no-deterministic` restores the throughput-first behavior.
- **Weight provenance in safetensors metadata (`version` / `repo` / `commit` / `describe`)**: every checkpoint and converter output records which flash-yolo code produced it; the git-derived keys are omitted when unavailable (e.g. inside Docker), and existing files load unchanged.
- **Switchable optimizer (`--optimizer sgd`)**: a plain SGD path (with `--[no-]nesterov`) for darknet-style reproductions; the default `musgd` leaves existing runs bit-identical.
- **TensorRT fp16 export (`scripts/export.py --trt --fp16`)**: builds half-precision engines the TRT 11 way (half-precision ONNX into a strongly-typed network, since the `FP16` builder flag is gone); fp32/fp16 artifacts coexist, a mismatched precision input is rejected, and engine outputs are normalized back to fp32.
- **Docker development environment**: `Dockerfile` + `docker/run.sh` + pinned constraints mirror the verified environment (Python 3.12, torch 2.13.0+cu130, CUDA 13.0, TensorRT 11.4.0.106, ultralytics 8.4.173), with the full train/evaluate/export/infer workflow running in the container.
- **Operator documentation set under `docs/`** (Chinese): six manuals (dataset, train, val, export, infer, docker), a FAQ of cross-cutting traps, and per-architecture design & training records.
- **YOLOv3-tiny SiLU operator-swap variant (`--model yolov3-tiny-silu`)**: line-for-line darknet topology with one change (LeakyReLU(0.1) → SiLU) — same state_dict keys, same 8,852,366 parameters, same 5.6 GFLOPs @416; the darknet-pure variant stays untouched for official-weight parity.
- **IoU-aware objectness for YOLOv3-tiny (`--obj-iou-target`)**: optional v3 loss mode where the positive objectness target becomes the box IoU (5× weight), so inference scores (obj × cls) rank by localization quality; default off, darknet hard 0/1 targets unchanged.

### Changed

- **README restructured around the four workflows** (train · evaluate · export · infer): comparison table vs YOLO26n, four one-command Quick Start subsections, dedicated Evaluation/Verification sections, a consolidated Results section, and the TensorRT compatibility story.
- **Demo assets refreshed**: hero images are COCO val2017 samples with rendered flash-yolo detections (image ids in `assets/README.md`); the ultralytics demo images are removed, and the export-parity test uses `traffic.jpg`.
- **Scorecard split into accuracy & latency tables** (ORT-CPU and TensorRT, b1/b8, fp32/fp16), re-measured on an idle machine with the final engines; flash-yolo's batch-8 latency lands at its 1.44× GFLOP ratio vs YOLO26n.
- **Scorecard carries the final 100-epoch results plus a YOLO26n alignment table**: 33.77/48.63 mAP@[.5:.95]/mAP@50 E2E and 34.72/50.16 with NMS — 2.4 points behind the budget-matched YOLO26n control for 31% fewer GFLOPs; the official YOLO26n checkpoint evaluates identically through both implementations.
- **README trimmed to the essentials**: the long-form Datasets/Training/Evaluation/Deployment sections and the Docker subsection move to `docs/`; the YOLOv3-tiny chapter is removed (code, tests and CLI unchanged).
- **Benchmark refresh + a released-checkpoint profile page** (`docs/weights.md`): every published checkpoint profiled — accuracy, ORT-CPU/TensorRT latency (b1/b8 × fp32/fp16), deployment artifacts and metadata/provenance.

### Removed

- **`yolov3-tiny-darknet` recipe retired**: the darknet-alignment investigation closed (official SGD numbers not reproducible under the unconditional grad-norm clip; remaining knobs probed flat). Recommended yolov3-tiny config: `--model yolov3-tiny-silu --obj-iou-target` — 16.52/30.44 on COCO val2017 @416 vs 14.23/27.50 for the plain default, 0.42 mAP50-95 short of the official darknet weights through the same pipeline.

### Fixed

- **Single-process training was never reproducible (`workers=0`)**: the main-process augmentation RNG fell back to an unseeded generator per call, so the same seed produced different weights; the stream is now seeded per epoch from `seed + epoch` (stdlib `random` too), making fixed-seed runs bit-identical and resumed runs match uninterrupted ones.
- **YOLOv3-tiny fp16 export**: the v3 decode grid used a half-dtype `torch.arange`, which the TorchScript ONNX exporter rejects — the whole `--trt --fp16` path failed; the grid is now built in fp32 and cast (same treatment as `make_anchors`).

## [0.5.0] - 2026-10-10

### Added

- **Multi-scale training + step LR schedule (with a `yolov3-tiny-darknet` recipe)**: `multi_scale: r` jitters each batch's input size within imgsz×(1±r) (quantized to stride multiples), and `lr_schedule: step` adds darknet step decay beside the linear/cosine curves (also via CLI); the recipe bundles both for the yolov3-tiny residual-gap experiment with single-variable toggles.
- **Flash-YOLO — the framework's own lightweight architecture**: a pooled-downsampling redesign (`PoolConv` = max-pool + 1×1, replacing every stride-2 3×3 conv) plus a channel-projection stem, budgeting **3.8 GFLOPs / 1.97M params @640** (−31%/−23% vs yolo26n) and beating yolo26n at batch-1 CPU/CUDA-graph inference (+45% ORT-CPU, +11% torch-CPU, +11% CUDA-graph). `scripts/search_arch.py` searches the design space, `LiteBlock` joins `model/basic`, and `--cfg` trains unregistered variant yamls for screening.
- **TensorRT inference — the three-engine matrix is complete**: `TRTEngine` joins the PyTorch and ONNX-Runtime-CPU engines behind the shared `predict` contract; `--trt` chains safetensors → onnx → `.engine` in one command and `eval.py`/`infer.py` take `--engine trt`. Batch-1 fp32: flash-yolo **1.35 ms** vs yolo26n 1.52 ms (+12.6% throughput). `tensorrt` is an optional dependency; a built engine is bound to the build machine's GPU + TRT version.

### Changed

- **onnxruntime sessions use a bounded thread pool**: `intra_op_num_threads` ≤16 / `inter_op_num_threads` 1 instead of ORT's core-count default — on container hosts that over-report CPU counts the default pool ran the engine 15-20× slow (120-196 ms → 10-14 ms per 640² image, batch 1); inference numbers unchanged.
- **flash-yolo adopts the DW-stem design**: the round-1 screening promoted S2 (depthwise-3×3 stem, +0.67 mAP at equal GFLOPs, batch-1 latency unchanged) into `config/models/flash-yolo.yaml`; the pre-promotion structure stays as `flash-yolo-s1.yaml`.
- **README training documentation**: a Training Results section (from-scratch commands + metric tables for YOLO26n and YOLOv3-tiny, official weights through the same pipeline) and a Flash-YOLO section with the design premise and screening table.

### Fixed

- **TensorRT batch-dim strip**: `TRTEngine._forward` now strips the batch dimension like the other engines — `predict_timed` (and with it `eval.py`/`infer.py --engine trt`) crashed on the un-stripped `(1, …)` output.

## [0.4.0] - 2026-10-09

### Added

- **Weights metadata — model parameters that travel with the checkpoint**: every training save and both official-weight converters write an optional JSON block into the safetensors header (`arch`, `scale`, `nc`, `imgsz`, class names, v3 anchors) — header only, no tensor touched. Loading resolves through CLI > metadata > filename/default, so a custom-run `best.safetensors` evaluates without hand-passing parameters; weights without metadata behave exactly as before.
- **`--imgsz` for eval / inference / export**: the input size resolves end to end (CLI > weights metadata > 640); the ONNX engine reads its fixed size from the graph itself and rejects a conflicting `--imgsz` (re-export instead of silently mis-sizing).
- **YOLOv3-tiny model support**: a darknet-faithful architecture (LeakyReLU blocks, max-pool downsampling, route/upsample neck, anchor-based two-scale head with the official COCO anchors kept as-is, including the original mask quirk) assembled by the generalized `model/build.py` factory; anchors live as non-persistent buffers so checkpoints keep a converter-friendly key set.
- **YOLOv3-tiny training**: `train/loss_v3.py` (BCE objectness + one-hot BCE classes + CIoU box, per-level best-anchor matching with a 0.7 ignore band) runs through the existing Trainer; `TrainConfig.model` selects the architecture, and trainer/validator output derives from the loss' item keys, so yolo26 console/CSV output is byte-identical.
- **`scripts/compute_anchors.py`**: checks the current yolov3-tiny anchor set against a dataset's train split (YOLOv5-style coverage, `anchor_t` 4.0) and re-clusters deterministically, printing a paste-ready `anchors:` fragment with a swap verdict; custom-dataset from-scratch training should run it first.
- **YOLOv3-tiny weights and inference pipeline**: `download_weights.py --model yolov3-tiny` fetches the official darknet weights and `convert_weights.py` gains a darknet parser; the engines accept `model=` and decode + NMS the v3 head output, with filename inference (`resolve_arch_scale`) across eval/infer/export.

### Changed

- **Entry-point scripts now open with a resolved-parameter preview line**: every script logs `scripts/<name>.py: key=value, …` as its first line, showing effective values (inferred architecture, filled-in defaults).

### Fixed

- **In-training validation no longer grinds on early checkpoints (yolov3-tiny)**: with an untrained model every candidate cleared the confidence floor (~200k NMS candidates per image, 7.2 s/image — hours for a full val pass); `nms_per_image` is now a class-aware greedy over globally sorted candidates with an early exit — outputs unchanged, pathological case down to ~12 ms/image.
- **Progress lines no longer wrap (and smear) on narrow terminals**: the scan line shows just the file name, and every progress line clamps to terminal width with a word-boundary ellipsis (bar/count/speed suffix always kept), so a `\r` redraw can never wrap.

## [0.3.0] - 2026-10-08

### Added

- **Inference and export for models with a non-COCO class count**: `infer.py --data <descriptor>` builds the head with the descriptor's `nc` and renders its label names; `export.py --nc N` exports the right class count. A custom checkpoint's strict-load size mismatch stays the explicit error, now with a way to proceed.
- **`scripts/make_coco_subset.py`**: extracts a seeded small COCO subset (hard-linked, ready-to-use descriptors) so a full train + eval cycle runs in minutes instead of hours.
- **YOLO txt datasets (`data/yolo.py`) and the descriptor-driven factory (`data/build.py`)**: sibling `<stem>.txt` labels (normalized, numeric-stem fallback), missing labels count as background, a wrong labels dir fails loudly, and standalone evaluation works; shared scan/loader machinery keeps the two formats from drifting.
- **Dataset descriptors (`config/datasets/spec.py`)**: ultralytics-style yaml (`format`, `path`, `names`, role blocks) resolved into validated specs with actionable errors; machine paths live in gitignored `config/datasets/local/`.
- **`scripts/eval.py` reports the validation split the way training does**: same scan line, throttled progress bar and `└ instances · categories · parse + scan` summary; engine summaries are rendered per backend.
- **Dataset scan statistics + progress bar (`data/coco.py::scan_split`)**: the three passes over the training split are one loop that reports counts (images / backgrounds / missing) on a throttled bar plus a parse/scan summary line.
- **Lazy "corrupt" accounting**: unreadable images no longer abort training — they become blank-canvas backgrounds (train) or keep their GT (val), with a once-per-epoch warning and a `meta.json` count; no per-image decode at scan time, so startup is unaffected.

### Changed

- **`--data` takes a dataset descriptor and the model's `nc` follows it** (breaking): descriptors replace `(data_dir, train_split)`; passing a directory is a migration error; `eval --split` selects a role (default `val`); the trainer builds the model with `nc = len(descriptor names)`.
- **Config package reorganized: one home per kind of thing**: models under `config/models/`, dataset names under `config/datasets/`, and the training-config mechanism moved to `config/train_config.py` (removing the name collision with `scripts/train.py`). `resume.pt` files from before this release pin the old module path and can no longer be resumed — continue from `weights/*.safetensors`.
- **`scripts/eval.py` header follows the training layout**: env → model → dataset → eval parameters → details, with the parameters on their own line, no more silent ~1.5 s startup.
- **Training startup block now prints in build order**: environment + hyperparameters immediately, then model table → dataset scan → components → `Starting training` — the ~20 s silent startup is gone.

### Fixed

- **Classification bias prior follows the training image size**: `Detect.bias_init` hard-coded 640, so non-640 runs started with a per-cell classification prior off by `(640/imgsz)^2`; the size is now threaded through (default 640 runs and official-weight alignment unchanged).
- **`scripts/infer.py` can run non-`n` scales**: scale now derives from the weights filename (`yolo26s.safetensors` → `s`), `--scale` overrides it, and a scale-less name raises a clear CLI error; the weights load happens after `run.log` attaches, so load failures land in the log.
- **Missing-image warning no longer disappears**: it is now an aggregated `logger.warning` naming the first missing file, with the count carried into the scan line and `meta.json`.

## [0.2.0] - 2026-10-08

### Added

- **`scripts/bench_io.py` — training I/O throughput benchmark**: two-stage auto-narrowing (loader-only sweep, then end-to-end timing) because a faster loader does not mean faster training; the tool only advises — it never edits the defaults.
- **Worker thread capping (`data/dataset.py::worker_init_fn`)**: each DataLoader worker runs a single OpenCV/torch thread, avoiding N workers × cores oversubscription (measured +34% img/s at batch 64, up to +64% with more workers).
- **Training artifacts (default-on)**: periodic checkpoints, gradient diagnostics (pre-clip norm, per-group lr, loss breakdown), augmentation sample grids, `meta.json` (env/config/data stats/argv), `best_raw.safetensors` pairs, `stop_after` for epoch-aligned A/B screens, and the warmup momentum ramp.
- **Validation losses (results.csv only)**: every val round computes the same five losses on the same EMA model (val letterbox now follows `imgsz`); deliberately not printed, matching the official behavior.
- **Training pipeline (M3)**: complete from-scratch COCO training — TAL+STAL assigner, dual-head loss (CIoU + BCE + box-L1) with ProgLoss schedule, MuSGD optimizer, EMA, warmup + linear LR, resume checkpoints, per-epoch COCO validation and `scripts/train.py` (finetune via `--weights`, resume via `--resume`); official hyperparameters in `config/train.yaml`.

### Changed

- **Run logs now live in the run directory**: `runs/<kind>/<name>/run.log` instead of `logs/`, so a run directory is self-contained; a resumed run appends to the same log and `--resume <dir>` reuses the directory.
- **The eval console table now matches the training layout and shows P/R and size buckets**: 11-wide columns plus `P`/`R` (max-F1 convention) and `AR@100`; per-class rows by default with `--summary-only` to collapse; `metrics.txt` stores the full table. Two reporting bugs fixed along the way (progress-line collision, unreconcilable speed line).
- **No machine-specific paths in the config**: `--data` is required (omitting it is a hard error) and the shipped config keeps only portable values.
- **Training artifacts are on by default**: `save_period` 20, `aug_samples` 8 (~45 MB per 100-epoch run); opt out with `--save-period 0 --aug-samples 0`.
- **Five augmentation geometry/colour mismatches aligned with the official pipeline**: mosaic tile aspect ratio (stretch removed), additive HSV hue, correct BGR-swap semantics, official box filtering, affine translation/shear in official units, copy-paste inert by default; boxes kept per image 9.5 → 15.4 and a 20-epoch screen improves mAP 0.2435 → 0.2865 (+17.7%).
- **Three training-loop parity fixes**: official EMA decay ramp, BN momentum 0.03, affine scale gain `U(1-s, 1+s)`.
- **`dfl_loss` renamed to `l1_loss`** in the epoch table, results.csv and val row (reg_max=1 has no DFL; matches official naming); the config field stays `dfl_gain`.
- **Training throughput ~5×**: batched `(B,N,M)` assigner + loss and `_foreach_` MuSGD (step 1250 → 250 ms at batch 64/640² on RTX 5090), outputs bit-identical.
- **Training throughput, round 2 (~1.4×)**: channels_last + batched Newton-Schulz + `_foreach_` EMA (408.8 → 287.8 ms/step at batch 64); inference/export stay NCHW with untouched parity.
- **Faster in-training validation**: batched val forwards (89.5 → 146 img/s) and single-sync assigner trimming.
- **Training recipes split into files**: `config/train.yaml` is the general 100-epoch recipe; named recipes live in `config/recipes/<name>.yaml` (`--recipe <name>`, or a `.yaml` path) with provenance headers — shipped: `yolo26-coco-ft.yaml` (requires Objects365-pretrained init) and `yolo26-o365-pt.yaml` (unverified here).
- **Training display aligned with ultralytics semantics**: running-mean losses, cumulative-average speed, and peak (`GPU_peak`) memory in the epoch row.
- **Faster in-training metrics (~4×)**: vectorized `FastMetrics` (one IoU matrix reused across thresholds, short-circuit, vectorized AP interpolation), outputs bit-identical.

### Fixed

- **Training deadlocked at the first epoch boundary**: `cv2.setNumThreads(1)` inside forked DataLoader workers deadlocked against OpenCV's non-fork-safe thread pool once the parent had used cv2 — a COCO run froze after epoch 1 with all 16 workers parked in `futex`. The cap now happens at import time in the parent.
- **Training-time mAP was under-reported ~10×**: `FastMetrics` accumulated the PR curve in per-image order instead of global score order (mAP50 0.0183 → 0.1973 on a fixed set, matching pycocotools); P/R/AR were unaffected.
- **Added the official gradient clipping (global norm, max_norm=10)**: without it, from-scratch runs at full lr explode within ~15 steps; with it they are stable.
- **Training loss normalization now matches the official implementation**: cls divides by the soft-target sum (was diluted ~10⁶), L1 is normalized like the reference (was 10¹–10²× too strong), and TAL soft targets follow the official form — from-scratch runs never converged before this.
- **Removed the extra `cls_w` factor on the o2o classification loss**: it had no counterpart in the official implementation and made the o2o P3 head blow up once ProgLoss reached 0.9; the field is removed from config and CLI.

## [0.1.0] - 2026-10-05

### Added

- **Initial release (M1+M2)**: faithful YOLO26 detection reproduction — architecture with verified weight alignment, inference (.pt/.onnx, E2E NMS-free & NMS paths), pt→onnx export, COCO evaluation (40.27/40.89 vs official 40.1/40.9), ultralytics-style logging, progress bar and runs/ layout.
- **Export `--dynamic` flag**: dynamic batch opt-in (default fixed batch=1); `dynamo=False` pins the TorchScript exporter (torch 2.13's dynamo default splits weights into `.onnx.data`).
