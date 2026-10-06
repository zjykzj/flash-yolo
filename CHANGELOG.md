# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Training pipeline (M3)**: complete from-scratch COCO training — `train/` package with
  TAL+STAL assigner, dual-head loss (CIoU + BCE + box-L1) with ProgLoss schedule (o2m 0.8→0.1),
  MuSGD optimizer (Muon+Newton-Schulz orthogonalization blended with Nesterov SGD), EMA,
  warmup + linear LR, resume checkpoints (safetensors weights + resume.pt state), per-epoch
  COCO validation and `scripts/train.py` (finetune via `--weights`, resume via `--resume`);
  Detect head train mode (o2o branch trained on detached features); official yolo26n
  COCO-stage hyperparameters in `config/train.yaml` (nbs=128 keeps lr semantics independent of
  physical batch size); training defaults to fp32 (half-precision from-scratch training NaNs).

### Changed

- **Training throughput ~5x**: TAL assigner + loss rewritten as batched `(B,N,M)` masked ops
  (the per-image Python loop was ~105k kernel launches and thousands of device syncs per step)
  and MuSGD elementwise updates switched to `_foreach_*` with a single device sync per step;
  measured batch 64 / 640² on RTX 5090: step 1250→250 ms, 51→~250 img/s (245-epoch COCO ETA
  ~6.5 d → ~1.4 d). Loss forward values and optimizer updates are bit-identical to the previous
  implementation; gradients no longer flow through the label assignment graph (official
  `@torch.no_grad()` + detached-input semantics).
- **Training throughput, round 2 (~1.4x)**: channels_last (NHWC, `channels_last: true` in
  train.yaml, `--channels-last/--no-channels-last`) + MuSGD Newton-Schulz batched across
  same-shape parameter groups + `_foreach_*` EMA updates; measured end-to-end 408.8→287.8 ms/step
  at batch 64 (156→222 img/s on RTX 5090). `save_weights` normalizes tensors to contiguous
  (NHWC parameter strides are rejected by safetensors) and C2PSA/Detect use `reshape` instead
  of `view` for channels_last compatibility; inference/export stay NCHW, bit-identical parity
  untouched.
- **Faster in-training validation**: validation forwards are batched (16 images) instead of one
  image at a time — steady 89.5→146 img/s (~56→~34 s per val2017 epoch); assigner chunk trimming
  now needs a single host transfer (≤8 device syncs/step → 1) and preds NaN diagnostics are
  sampled every 10 steps (loss finiteness still checked every step).
- **Training recipes — default is now the general 100-epoch recipe**: `config/train.yaml` ships
  two recipes: `default` (100 epochs, ultralytics-aligned lr/loss/aug defaults) and `official`
  (published YOLO26 recipe per scale — 245/70/80/60/40 epochs for n/s/m/l/x, lr0 0.0054/0.00038,
  selectable via `--recipe official`). Previously the file only contained the official yolo26n
  COCO-stage values (245 epochs) as the defaults.

## [0.1.0] - 2026-10-05

### Added

- **Initial release (M1+M2)**: faithful YOLO26 detection reproduction — architecture with
  verified weight alignment (2,572,280 params, strict load, bit-identical to official output),
  inference (.pt/.onnx, E2E NMS-free & NMS paths), pt→onnx export, COCO evaluation
  (40.27 / 40.89 vs official 40.1 / 40.9), ultralytics-style logging, progress bar and
  runs/ result layout.
- **Export `--dynamic` flag**: dynamic batch opt-in, default fixed batch=1 (edge toolchains
  prefer fixed shapes); ONNX weights embedded in a single file — `dynamo=False` pins the
  TorchScript exporter (torch 2.13's dynamo default split weights into `.onnx.data`).
