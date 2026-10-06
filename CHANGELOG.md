# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Validation losses (results.csv only)**: every validation round also computes the val loss
  with the same `ComputeLoss` on the same EMA model, from the *same* backbone/neck forward
  (`YOLO26.forward_feats`; eval-mode BN, no running-stat pollution — metrics stay bit-identical
  to the previous validator, 200-image baseline diff 0.0). The five items (box/cls/l1/o2m/o2o,
  same definitions as the train row) are written to `results.csv` as
  `val_box, val_cls, val_l1, val_o2m, val_o2o` (empty on non-validation epochs) and are
  deliberately **not printed** — matching the official behavior (ultralytics' console shows no
  val loss either; its values live in `val/*_loss` CSV columns). Cost ≈ +10 s per full val2017
  round (~2% of a 100-epoch run). Val letterbox size now follows the configured `imgsz` (was
  hardcoded 640, which mis-sized the loss anchors for non-640 runs); the train row's loss
  columns use a width-aware formatter (fixed-point as before, scientific fallback for
  degenerate values, so a diverging run cannot break column alignment); the gated smoke asserts
  the new CSV columns.
- **Training pipeline (M3)**: complete from-scratch COCO training — `train/` package with
  TAL+STAL assigner, dual-head loss (CIoU + BCE + box-L1) with ProgLoss schedule (o2m 0.8→0.1),
  MuSGD optimizer (Muon+Newton-Schulz orthogonalization blended with Nesterov SGD), EMA,
  warmup + linear LR, resume checkpoints (safetensors weights + resume.pt state), per-epoch
  COCO validation and `scripts/train.py` (finetune via `--weights`, resume via `--resume`);
  Detect head train mode (o2o branch trained on detached features); official yolo26n
  COCO-stage hyperparameters in `config/train.yaml` (nbs=128 keeps lr semantics independent of
  physical batch size); training defaults to fp32 (half-precision from-scratch training NaNs).

### Changed

- **`dfl_loss` renamed to `l1_loss`** in the epoch table header, `results.csv` column and val
  row (reg_max=1 has no DFL — the term is a plain L1; this matches the official ultralytics
  loss names for YOLO26). The config field stays `dfl_gain` (same name as `hyp.dfl` in the
  official implementation).
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
- **Training display aligned with ultralytics semantics**: the epoch progress row now shows
  losses as the running mean within the epoch (ultralytics `tloss`; the live value converges to
  the frozen epoch mean at epoch end) and speed as the cumulative average (`n/elapsed`, tqdm
  semantics) instead of 10-batch sliding windows; the memory column now reports the process peak
  (`max_memory_reserved`, header renamed `GPU_peak`) instead of the jumpy current reserved value.
- **Faster in-training metrics (~4x)**: `FastMetrics.compute()` vectorized — one IoU matrix per
  (image, class) reused across all 10 IoU thresholds, all thresholds matched in a single
  detection pass, short-circuit when a block's max IoU is below the lowest threshold, IoU skipped
  for (image, class) pairs without GT (counted as FP only), vectorized 101-point AP interpolation
  and a single `bincount` for per-class GT counts. Outputs are bit-identical to the previous
  implementation (49-case baseline, max diff 0.0), so the metric semantics are unchanged.

### Fixed

- **Training loss normalization now matches the official implementation (ultralytics E2ELoss)** —
  from-scratch runs never converged because two of the three loss terms were mis-normalized
  versus the reference the recipe is calibrated against (diagnosed from a 12-epoch run: every cls
  head bias still at its init value, val mAP 0.0000, BN `running_var` up to 1e9, all losses
  rising):
  - **cls** divided the BCE by the raw element count (B·N·nc ≈ 4.3e7) instead of the official
    soft-target sum `Σtarget_scores` (~10² per batch), diluting the classification gradient by
    ~10⁶ — the head received effectively zero gradient (predictions stayed at the bias-init
    sigmoid ≈ 1e-5, so every detection was filtered and mAP was 0 by construction). Now
    `ΣBCE / Σt`, the same denominator the box and L1 terms use;
  - **L1** summed `|Δltrb|` in grid units; the official reg_max=1 branch normalizes ltrb by
    `stride/imgsz`, takes the mean over the 4 sides and soft-weights by t (grid units made the
    term 10¹–10²× too strong relative to CIoU — L1 was 94% of the total loss, official balance
    is ≈1:1);
  - label-assignment soft targets now follow the official TAL normalization
    (`align/align_max × CIoU_max`, value ⊂ (0, 1] — previously "sums to 1 per GT"), and the o2o
    second-stage top-k selects by alignment (was CIoU).
  `tests/test_loss_parity.py` pins all three items and the ProgLoss-weighted total of both
  branches against the installed ultralytics E2ELoss on fixed inputs (rel < 1e-4).
- **Removed the extra `cls_w` factor on the o2o classification loss** — the published-recipe
  doc's "algorithm constant" 2.74 had no counterpart in the official implementation (ultralytics
  E2ELoss applies no branch-specific cls multiplier; effective o2o:o2m cls weight there is 9×,
  ours was 24.7×). With it, once ProgLoss pushes the o2o weight to 0.9 the o2o cls P3 head
  enters a BN-masked compounding growth (loss stays healthy while the head's weights grow
  geometrically 1.8 → 1e10, then explode; reproduced and A/B-tested by resuming from the
  epoch-1 checkpoint: with `cls_w=2.74` it blew up within ~110 batches, with `cls_w=1.0` the
  same schedule ran stable). The field is removed from `TrainConfig` / `train.yaml` / CLI.

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
